"""Isolated, parallel official ARC games: native Codex versus Buffalo."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..models import new_id
from .arc_protocol import FixedGameControl, load_validation
from .bridge import OfficialWorker
from .codex_harness import run_codex
from .harness import run_buffalo
from .inference_gate import InferenceGate
from .runner import contract, load, resolve, source_identity
from .schema import NotRun, accounting, digest, file_digest, save, timestamp


def infrastructure_failure(exc):
    if isinstance(exc, (OSError, TimeoutError)):
        return True
    return bool(
        re.search(
            r"transport|connection|http_429|http_5\d\d|os_permission_retry|worker exited|toolkit.*(?:failed|no |could not)|app-server exited|timed out|stream.*(?:interrupt|disconnect)|servererror|toomanyrequests",
            str(exc),
            re.IGNORECASE,
        )
    )


def report(row):
    lines = ["ARC-AGI-3", ""]
    for profile, title in (("codex", "Codex"), ("buffalo", "Buffalo")):
        result = row.get("profiles", {}).get(profile, {})
        lines.append(title + ":")
        if result.get("status") == "COMPLETED":
            u = result["usage"]
            lines += [
                f"RHAE = {result['primary_score']:.6f}%",
                f"games = {result['task_count']}",
                f"tokens = {u['total_tokens']} (input {u['input_tokens']}, output {u['output_tokens']})",
                f"model calls = {u['model_calls']}",
                f"wall time = {u['wall_seconds']:.2f}s",
            ]
        else:
            lines += [
                result.get("status", "NOT RUN"),
                f"reason = {result.get('reason', 'Profile not selected')}",
            ]
        lines.append("")
    profiles = row.get("profiles", {})
    if all(profiles.get(p, {}).get("status") == "COMPLETED" for p in ("codex", "buffalo")):
        lines += [
            f"Buffalo lift over Codex: {profiles['buffalo']['primary_score'] - profiles['codex']['primary_score']:+.6f} percentage points"
        ]
    lines += [
        f"concurrency used: {row.get('concurrency', {}).get('peak_inflight', 0)}",
        f"failed/retried games: {row.get('retries', [])}",
        "BASELINE = actual Codex harness",
        "NOT custom run_base()",
    ]
    return "\n".join(lines) + "\n"


async def execute_arc(args):
    if args.benchmark != "arc-agi-3":
        raise ValueError("This phase runs ARC-AGI-3 only")
    if max(args.games_concurrency, args.inference_concurrency) > 16:
        raise ValueError("Concurrency above 16 requires a separate measured capacity experiment")
    config, error = load(args.config)
    if error:
        raise NotRun(error)
    validation_path = getattr(args, "protocol_validation", None)
    policy = load_validation(validation_path, args.limit, args.seed) if validation_path else None
    if policy:
        if config.benchmarks["arc-agi-3"].options.get("operation_mode", "OFFLINE") != "OFFLINE":
            raise NotRun("Fixed-game validation requires the pinned local OFFLINE environments")
        config.run.limits.max_subagents = 0
        config.run.limits.max_depth = 0
        config.run.permissions = [p for p in config.run.permissions if p != "agents"]
    config = await resolve(config)
    if (
        config.run.provider.name != "codex_subscription"
        or config.run.provider.model != "gpt-6-astra"
        or config.run.provider.parameters.get("reasoning_effort") != "xhigh"
    ):
        raise NotRun(
            "ARC comparison requires the authenticated Codex subscription, gpt-6-astra and xhigh"
        )
    executable = shutil.which("codex")
    if not executable:
        raise NotRun("Actual Codex CLI is not installed")
    version = (
        await asyncio.to_thread(subprocess.check_output, [executable, "--version"], text=True)
    ).strip()
    auth = await asyncio.to_thread(
        subprocess.run, [executable, "login", "status"], text=True, capture_output=True
    )
    if auth.returncode or "ChatGPT" not in auth.stdout + auth.stderr:
        raise NotRun("Installed Codex is not authenticated through the ChatGPT subscription")
    root = (args.output or Path("results/evaluation") / ("arc-codex-" + new_id())).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty; preserve earlier artifacts")
    root.mkdir(parents=True, exist_ok=True)
    save(root / "manifest.json", config.model_dump(mode="json"))
    save(root / "buffalo-source.json", source_identity())
    save(
        root / "codex-runtime.json",
        {
            "executable": executable,
            "sha256": file_digest(executable),
            "version": version,
            "authentication": "ChatGPT subscription",
            "baseline": "actual Codex app-server",
        },
    )
    setup = config.benchmarks["arc-agi-3"].model_copy(deep=True)
    setup.task_ids = []
    probe = OfficialWorker(setup, "arc-agi-3", root / "official-aggregator")
    gate = InferenceGate(root / "global-inference", args.inference_concurrency)
    begin, started = time.monotonic(), timestamp()
    row = {
        "benchmark": "ARC-AGI-3",
        "status": "RUNNING",
        "start_time": started,
        "profiles": {},
        "retries": [],
        "trajectory_reference": str(root),
    }
    results = {}
    jobs = set()
    try:
        provenance = await probe.start()
        task_ids = provenance["task_ids"][: args.limit] if args.limit else provenance["task_ids"]
        if not args.limit and len(task_ids) != 25:
            raise NotRun(f"Expected the pinned 25 official environments, found {len(task_ids)}")
        shared = contract(config, provenance, task_ids, args.seed)
        if policy:
            shared["protocol_validation"] = {
                "policy_sha256": digest(policy),
                "action_limit": 500,
                "batch_limit": 20,
                "whole_game_retries": 0,
                "snapshots": "diagnostic action/continuation boundaries; no inferred scaling thresholds",
            }
        row.update(
            provenance,
            task_ids=task_ids,
            task_count=len(task_ids),
            full_benchmark=not args.limit,
            comparison_contract=shared,
            comparison_contract_sha256=digest(shared),
        )
        save(root / "comparison-contract.json", shared)
        if args.check:
            row.update(status="PREFLIGHT", reason="Read-only ARC/Codex setup validation passed")
            return 0
        await gate.start()
        profiles = ["codex", "buffalo"] if args.profile == "paired" else [args.profile]
        semaphore = asyncio.Semaphore(args.games_concurrency)
        initial_hashes = {}

        async def game(index, task_id, profile, attempt=1):
            async with semaphore:
                directory = root / profile / f"game-{index:05d}" / f"attempt-{attempt}"
                directory.mkdir(parents=True)
                local_setup = setup.model_copy(deep=True)
                local_setup.task_ids = [task_id]
                worker = OfficialWorker(local_setup, "arc-agi-3", directory / "official")
                owner = f"{profile}-{index}-{attempt}"
                result = {
                    "task_id": task_id,
                    "profile": profile,
                    "attempt": attempt,
                    "status": "RUNNING",
                    "start_time": timestamp(),
                    "trajectory_reference": str(directory),
                }
                game_begin = time.monotonic()
                controller = None
                try:
                    await worker.start()
                    card = await worker.call("start_profile", profile=profile, seed=args.seed)
                    task = await worker.call(
                        "start_task", task_id=task_id, **({"fixed_game": True} if policy else {})
                    )
                    raw_hash = digest(task)
                    if task_id in initial_hashes and initial_hashes[task_id] != raw_hash:
                        raise NotRun("Initial observation differs between isolated game workers")
                    initial_hashes[task_id] = raw_hash
                    identity = {
                        "task_id": task_id,
                        "input_sha256": raw_hash,
                        "comparison_contract_sha256": digest(shared),
                        "worker_pid": worker.process.pid,
                        "scorecard_id": card["scorecard_id"],
                        "workspace": str(directory / "workspace"),
                        "recording_directory": str(directory / "official" / profile / "recordings"),
                    }
                    save(directory / "identity.json", identity)
                    # Same task and generic interaction instruction for both actual harnesses.
                    if policy:
                        controller = FixedGameControl(worker, directory, policy, gate, owner)
                        controller.identity = identity
                        task = await controller.start()
                    else:
                        task["messages"][-1]["content"] += (
                            "\nKeep interacting autonomously until the game is won or your run budget expires. Use only observations to infer rules; do not inspect environment implementation files or known solutions. Any delegated agents must use gpt-6-astra with xhigh reasoning."
                        )
                    save(directory / "task-input.json", task)
                    action_count = 0
                    action_lock = asyncio.Lock()
                    last_observation = None if policy else await worker.call("arc_observation")
                    integrity_errors = []

                    async def action(**kwargs):
                        nonlocal action_count, last_observation
                        async with action_lock:
                            # Another worker must never mutate this worker's observation.
                            if digest(await worker.call("arc_observation")) != digest(
                                last_observation
                            ):
                                integrity_errors.append(
                                    "Worker observation changed without an action"
                                )
                                raise NotRun(
                                    "Isolated worker changed without an action; comparison invalid"
                                )
                            if action_count >= config.run.limits.max_tool_calls:
                                return {"error": "Shared game action budget exhausted"}
                            action_count += 1
                            result = await worker.call("action", **kwargs)
                            last_observation = await worker.call("arc_observation")
                            return result

                    runner = run_codex if profile == "codex" else run_buffalo
                    agent = await runner(
                        config.run,
                        task,
                        directory,
                        action=None if controller else action,
                        gate=gate,
                        owner=owner,
                        **({"controller": controller} if controller else {}),
                    )
                    result.update(agent)
                    output_budget = config.run.limits.output_token_budget
                    if (
                        output_budget is not None
                        and agent["usage"]["output_tokens"] > output_budget
                    ):
                        raise NotRun(
                            "Output budget exceeded by provider response: "
                            f"{agent['usage']['output_tokens']} > {output_budget}; "
                            "the subscription endpoint has no supported hard output limit. "
                            "Usage retained; this comparison is invalid."
                        )
                    if integrity_errors:
                        raise NotRun("; ".join(integrity_errors))
                    final = await worker.call("finish_profile")
                    save(directory / "official-scorecard.json", final.pop("raw"))
                    save(directory / "official-card.json", final.pop("official_card"))
                    result.update(
                        final,
                        status="COMPLETED",
                        actions=agent["game_state"]["actions_taken"]
                        if controller
                        else action_count,
                        isolation_verified=True,
                        input_sha256=raw_hash,
                    )
                except asyncio.CancelledError:
                    result.update(status="CANCELLED", reason="Evaluation stopped")
                    raise
                except Exception as exc:
                    result.update(
                        status="FAILED",
                        reason=f"{type(exc).__name__}: {exc}",
                        infrastructure_invalidated=infrastructure_failure(exc),
                    )
                    with contextlib.suppress(Exception):
                        save(
                            directory / "unscored-official-output.json",
                            await worker.call("finish_profile"),
                        )
                finally:
                    if controller:
                        await controller.close()
                    await worker.close()
                    usage_file = directory / "usage.json"
                    result["usage"] = (
                        json.loads(usage_file.read_text())
                        if usage_file.exists()
                        else accounting([], time.monotonic() - game_begin)
                    )
                    result["end_time"] = timestamp()
                    save(directory / "result.json", result)
                    results[(task_id, profile)] = result
                    save(
                        root / "progress.json",
                        {
                            "time": timestamp(),
                            "expected_games_per_profile": len(task_ids),
                            "completed": {
                                p: sum(
                                    v["status"] == "COMPLETED"
                                    for (tid, prof), v in results.items()
                                    if prof == p
                                )
                                for p in profiles
                            },
                            "failures": [
                                {"task_id": tid, "profile": p, "reason": v.get("reason")}
                                for (tid, p), v in results.items()
                                if v["status"] == "FAILED"
                            ],
                            "concurrency": gate.summary(),
                        },
                    )
                    print(
                        f"ARC {profile} {index + 1}/{len(task_ids)}: {result['status']} {result.get('reason', '')}",
                        flush=True,
                    )

        jobs = {
            asyncio.create_task(game(i, tid, p)) for i, tid in enumerate(task_ids) for p in profiles
        }
        await asyncio.gather(*jobs)
        # Retry only invalid attempts, never a completed low score or a healthy counterpart.
        for attempt in () if policy else (2, 3):
            retry_jobs = []
            for i, task_id in enumerate(task_ids):
                for profile in profiles:
                    previous = results[(task_id, profile)]
                    if previous["status"] != "FAILED" or not previous.get(
                        "infrastructure_invalidated"
                    ):
                        continue
                    row["retries"].append(
                        {
                            "task_id": task_id,
                            "profile": profile,
                            "attempt": attempt,
                            "reason": previous.get("reason"),
                        }
                    )

                    async def retry(
                        i=i, task_id=task_id, profile=profile, previous=previous, attempt=attempt
                    ):
                        await game(i, task_id, profile, attempt=attempt)
                        current = results[(task_id, profile)]
                        current["failed_attempts"] = [
                            *previous.get("failed_attempts", []),
                            {
                                "attempt": previous["attempt"],
                                "reason": previous.get("reason"),
                                "usage": previous["usage"],
                                "trajectory_reference": previous["trajectory_reference"],
                            },
                        ]
                        save(Path(current["trajectory_reference"]) / "result.json", current)

                    retry_jobs.append(asyncio.create_task(retry()))
            if not retry_jobs:
                break
            jobs = set(retry_jobs)
            await asyncio.gather(*jobs)
        for profile in profiles:
            selected = [results[(tid, profile)] for tid in task_ids]
            failures = [s for s in selected if s["status"] != "COMPLETED"]
            from datetime import datetime

            elapsed = max(
                datetime.fromisoformat(s["end_time"]).timestamp() for s in selected
            ) - min(datetime.fromisoformat(s["start_time"]).timestamp() for s in selected)
            usage = accounting([s["usage"] for s in selected], elapsed)
            value = {
                "status": "FAILED" if failures else "COMPLETED",
                "task_count": len(selected) - len(failures),
                "usage": usage,
                "tasks": selected,
            }
            if failures:
                value["reason"] = "; ".join(s.get("reason", "Incomplete") for s in failures)
            else:
                cards = [
                    json.loads((Path(s["trajectory_reference"]) / "official-card.json").read_text())
                    for s in selected
                ]
                summary = await probe.call("aggregate_arc", cards=cards, task_ids=task_ids)
                raw = root / profile / "official-scorecard.json"
                save(raw, summary.pop("raw"))
                value.update(summary, raw_evaluator_output=str(raw))
            overhead = [
                attempt["usage"] for s in selected for attempt in s.get("failed_attempts", [])
            ]
            if overhead:
                value["failed_attempt_usage"] = accounting(
                    overhead, sum(u["wall_seconds"] for u in overhead)
                )
            value["all_attempt_usage"] = accounting([usage, *overhead], elapsed)
            value["summed_game_wall_seconds"] = sum(s["usage"]["wall_seconds"] for s in selected)
            row["profiles"][profile] = value
        row["status"] = (
            "COMPLETED"
            if all(p["status"] == "COMPLETED" for p in row["profiles"].values())
            else "FAILED"
        )
        return 0 if row["status"] == "COMPLETED" else 2
    except asyncio.CancelledError:
        row.update(status="CANCELLED", reason="Evaluation stopped")
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    except Exception as exc:
        row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await gate.close()
        await probe.close()
        row.update(end_time=timestamp(), concurrency=gate.summary())
        row["usage"] = accounting(
            [
                u
                for v in results.values()
                for u in [v["usage"], *[a["usage"] for a in v.get("failed_attempts", [])]]
            ],
            time.monotonic() - begin,
        )
        save(root / "run.json", row)
        (root / "report.txt").write_text(report(row))
        print(report(row), flush=True)
