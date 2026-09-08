"""Isolated, parallel official ARC games: native Codex versus Buffalo."""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from ..models import new_id
from .bridge import OfficialWorker
from .codex_harness import run_codex
from .harness import run_buffalo
from .inference_gate import InferenceGate
from .runner import contract, load, resolve, source_identity
from .schema import NotRun, accounting, digest, file_digest, save, timestamp


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
                try:
                    await worker.start()
                    await worker.call("start_profile", profile=profile, seed=args.seed)
                    task = await worker.call("start_task", task_id=task_id)
                    raw_hash = digest(task)
                    if task_id in initial_hashes and initial_hashes[task_id] != raw_hash:
                        raise NotRun("Initial observation differs between isolated game workers")
                    initial_hashes[task_id] = raw_hash
                    save(
                        directory / "identity.json",
                        {
                            "task_id": task_id,
                            "input_sha256": raw_hash,
                            "comparison_contract_sha256": digest(shared),
                            "worker_pid": worker.process.pid,
                            "recording_directory": str(
                                directory / "official" / profile / "recordings"
                            ),
                        },
                    )
                    # Same task and generic interaction instruction for both actual harnesses.
                    task["messages"][-1]["content"] += (
                        "\nKeep interacting autonomously until the game is won or your run budget expires. Use only observations to infer rules; do not inspect environment implementation files or known solutions. Any delegated agents must use gpt-6-astra with xhigh reasoning."
                    )
                    save(directory / "task-input.json", task)
                    action_count = 0

                    async def action(**kwargs):
                        nonlocal action_count
                        action_count += 1
                        if action_count > config.run.limits.max_tool_calls:
                            return {"error": "Shared game action budget exhausted"}
                        return await worker.call("action", **kwargs)

                    runner = run_codex if profile == "codex" else run_buffalo
                    agent = await runner(
                        config.run, task, directory, action=action, gate=gate, owner=owner
                    )
                    result.update(agent)
                    final = await worker.call("finish_profile")
                    save(directory / "official-scorecard.json", final.pop("raw"))
                    save(directory / "official-card.json", final.pop("official_card"))
                    result.update(final, status="COMPLETED", actions=action_count)
                except asyncio.CancelledError:
                    result.update(status="CANCELLED", reason="Evaluation stopped")
                    raise
                except Exception as exc:
                    result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                    with contextlib.suppress(Exception):
                        save(
                            directory / "unscored-official-output.json",
                            await worker.call("finish_profile"),
                        )
                finally:
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
        # Retry infrastructure-invalidated games as matched pairs, independent of scores.
        for i, task_id in enumerate(task_ids):
            failed = [
                results[(task_id, p)]
                for p in profiles
                if results[(task_id, p)]["status"] != "COMPLETED"
            ]
            if not failed:
                continue
            row["retries"].append(
                {"task_id": task_id, "reasons": [f.get("reason") for f in failed]}
            )
            previous = {p: results[(task_id, p)] for p in profiles}
            await asyncio.gather(*(game(i, task_id, p, attempt=2) for p in profiles))
            for p in profiles:
                results[(task_id, p)]["prior_attempt_usage"] = previous[p]["usage"]
        for profile in profiles:
            selected = [results[(tid, profile)] for tid in task_ids]
            failures = [s for s in selected if s["status"] != "COMPLETED"]
            usage = accounting([s["usage"] for s in selected], time.monotonic() - begin)
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
            overhead = [s["prior_attempt_usage"] for s in selected if "prior_attempt_usage" in s]
            if overhead:
                value["prior_attempt_usage"] = accounting(
                    overhead, sum(u["wall_seconds"] for u in overhead)
                )
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
    finally:
        await gate.close()
        await probe.close()
        row.update(end_time=timestamp(), concurrency=gate.summary())
        row["usage"] = accounting([v["usage"] for v in results.values()], time.monotonic() - begin)
        save(root / "run.json", row)
        (root / "report.txt").write_text(report(row))
        print(report(row), flush=True)
