from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..models import RunConfig, new_id
from . import BENCHMARKS
from .bridge import OfficialWorker
from .harness import invoke_judge, run_base, run_buffalo
from .schema import (
    BenchmarkSetup,
    EvaluationConfig,
    NotRun,
    accounting,
    digest,
    file_digest,
    save,
    timestamp,
)


def arguments(parser):
    parser.add_argument("benchmark", choices=[*BENCHMARKS, "all"])
    parser.add_argument("--config", type=Path, help="Evaluation manifest (run, judge, benchmarks)")
    parser.add_argument("--output", type=Path, help="New output directory; never overwritten")
    parser.add_argument("--profile", choices=["base", "buffalo", "paired"], default="paired")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=positive, help="Explicit task subset; never labeled a full result"
    )
    parser.add_argument(
        "--token-budgets",
        type=budget_points,
        help="Comma-separated ARC budget points; runs fresh matched worlds at each point",
    )
    parser.add_argument(
        "--check", action="store_true", help="Preflight only; no inference or gameplay"
    )


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def budget_points(value):
    values = sorted({positive(part) for part in value.split(",")})
    if len(values) < 2:
        raise argparse.ArgumentTypeError("Supply at least two distinct budget points")
    return values


def load(path):
    if path is None:
        return EvaluationConfig(
            run=RunConfig()
        ), "Missing evaluation manifest: pass --config (see configs/evaluation.example.json)"
    raw = json.loads(path.read_text())
    for provider in [raw.get("run", {}).get("provider", {}), raw.get("judge") or {}]:
        model = provider.get("model", "")
        if model.startswith("${") and model.endswith("}"):
            name = model[2:-1]
            if not os.environ.get(name):
                raise NotRun(f"Set {name} to the model ID for this evaluation")
            provider["model"] = os.environ[name]
    config = EvaluationConfig.model_validate(raw)
    unknown = config.benchmarks.keys() - BENCHMARKS.keys()
    if unknown:
        raise NotRun(f"Unsupported benchmark configuration: {sorted(unknown)}")
    for setup in config.benchmarks.values():
        for field in ("source", "dataset"):
            value = getattr(setup, field)
            if value and not value.is_absolute():
                setattr(setup, field, (path.resolve().parent / value).resolve())
        if "/" in setup.python and not Path(setup.python).is_absolute():
            setup.python = str((path.resolve().parent / setup.python).absolute())
        for field in ("world_save", "environments_dir"):
            value = setup.options.get(field)
            if value and not Path(value).is_absolute():
                setup.options[field] = str((path.resolve().parent / value).resolve())
    return config, None


async def resolve(config):
    from ..cli import resolved_config
    from ..providers import default_providers

    if config.run.models or config.run.routing.default or config.run.routing.roles:
        raise NotRun(
            "Matched evaluations require a single provider/model without routing overrides"
        )
    if config.run.extensions or config.run.mcp_servers:
        raise NotRun(
            "Matched evaluations require explicit built-in tools; external extensions/MCP are not part of the comparison harness"
        )
    if not config.run.features.persistent_repl:
        raise NotRun("Buffalo evaluation requires features.persistent_repl=true")
    config.run = await resolved_config(config.run)
    providers = [config.run.provider]
    if config.judge:
        temporary = config.run.model_copy(deep=True)
        temporary.provider = config.judge
        config.judge = (await resolved_config(temporary)).provider
        providers.append(config.judge)
    for provider in providers:
        if provider.name not in default_providers():
            raise NotRun(f"Unsupported model provider: {provider.name}")
        if not provider.model:
            raise NotRun("Missing provider.model")
        if provider.api_key_env and not os.environ.get(provider.api_key_env):
            raise NotRun(f"Missing API credentials: {provider.api_key_env} is unset")
    return config


def source_identity():
    source = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(source)): file_digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".rs"}
    }
    root = source.parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "commit": result.stdout.strip() or None,
        "source_sha256": digest(files),
        "source_files": files,
    }


def contract(config, provenance, task_ids, seed):
    return {
        "provider": config.run.provider.model_dump(mode="json"),
        "reasoning_level": config.run.provider.parameters.get(
            "reasoning_effort", config.run.provider.parameters.get("reasoning", "provider default")
        ),
        "budget": config.run.limits.model_dump(mode="json"),
        "context_max_tokens": config.run.context.max_tokens,
        "benchmark_version": provenance["benchmark_version"],
        "dataset_environment_version": provenance["dataset_environment_version"],
        "starting_state": provenance["starting_state"],
        "task_ids": task_ids,
        "seed": seed,
    }


async def evaluate_family(benchmark, setup, config, args, directory, config_error=None):
    directory.mkdir(parents=True)
    begin = time.monotonic()
    record = {
        "benchmark": BENCHMARKS[benchmark],
        "benchmark_id": benchmark,
        "status": "NOT RUN",
        "reason": None,
        "benchmark_version": setup.commit,
        "dataset_environment_version": setup.dataset_revision,
        "task_ids": setup.task_ids,
        "model": config.run.provider.model,
        "provider": config.run.provider.name,
        "reasoning_level": config.run.provider.parameters.get(
            "reasoning_effort", "provider default"
        ),
        "buffalo_configuration": config.run.model_dump(mode="json"),
        "budget": config.run.limits.model_dump(mode="json"),
        "start_time": timestamp(),
        "end_time": None,
        "primary_score": None,
        "profiles": {},
        "raw_evaluator_output": None,
        "trajectory_reference": str(directory.resolve()),
        "usage": accounting([], 0),
    }
    worker = OfficialWorker(setup, benchmark, directory / "official")
    provenance = None
    try:
        provenance = await worker.start()
        record.update(provenance)
        if config_error:
            raise NotRun(config_error)
        config = await resolve(config.model_copy(deep=True))
        record.update(
            model=config.run.provider.model,
            provider=config.run.provider.name,
            buffalo_configuration=config.run.model_dump(mode="json"),
        )
        if args.check:
            raise NotRun("Preflight passed; --check requested (inference and gameplay not started)")
        tasks = provenance["task_ids"][: args.limit] if args.limit else provenance["task_ids"]
        record.update(
            task_ids=tasks,
            full_benchmark=len(tasks) == len(provenance["task_ids"]) and not setup.task_ids,
            task_count=len(tasks),
        )
        shared = contract(config, provenance, tasks, args.seed)
        record.update(
            comparison_contract=shared,
            comparison_contract_sha256=digest(shared),
            reasoning_level=shared["reasoning_level"],
        )
        save(directory / "comparison-contract.json", shared)
        profiles = ["base", "buffalo"] if args.profile == "paired" else [args.profile]
        interactive = benchmark in {"arc-agi-3", "factorio"}
        inputs = (
            {} if interactive else {tid: await worker.call("task", task_id=tid) for tid in tasks}
        )
        starting_inputs = {}
        for profile in profiles:
            profile_dir = directory / profile
            profile_dir.mkdir()
            usages, judge_usages, grades, results = [], [], [], []
            profile_begin = time.monotonic()
            profile_record = {
                "status": "RUNNING",
                "primary_score": None,
                "tasks": results,
                "comparison_contract_sha256": digest(shared),
                "start_time": timestamp(),
            }
            record["profiles"][profile] = profile_record

            async def judge(request, profile_dir=profile_dir, judge_usages=judge_usages):
                return await invoke_judge(
                    config.judge or config.run.provider, request, profile_dir, judge_usages
                )

            worker.judge = judge
            try:
                profile_record.update(
                    await worker.call("start_profile", profile=profile, seed=args.seed)
                )
                for index, task_id in enumerate(tasks):
                    print(
                        f"{BENCHMARKS[benchmark]} / {profile}: task {index + 1}/{len(tasks)} ({task_id})",
                        file=sys.stderr,
                        flush=True,
                    )
                    task = (
                        await worker.call("start_task", task_id=task_id)
                        if interactive
                        else inputs[task_id]
                    )
                    # Compare actual observations/instructions after fresh world initialization.
                    task_hash = digest({k: v for k, v in task.items() if k not in {"run_id"}})
                    if profile == profiles[0]:
                        starting_inputs[task_id] = task_hash
                    elif starting_inputs[task_id] != task_hash:
                        raise NotRun(
                            f"Starting state differs between base and Buffalo for task {task_id}"
                        )
                    task_dir = profile_dir / f"task-{index:05d}"
                    task_dir.mkdir()
                    save(task_dir / "task-input.json", task)
                    save(
                        task_dir / "identity.json",
                        {
                            "task_id": task_id,
                            "input_sha256": task_hash,
                            "comparison_contract_sha256": digest(shared),
                        },
                    )

                    async def action(**kwargs):
                        return await worker.call("action", **kwargs)

                    runner = run_base if profile == "base" else run_buffalo
                    result = await runner(
                        config.run,
                        task,
                        task_dir,
                        action=action if interactive else None,
                        persistent=benchmark == "factorio",
                        long_context=benchmark == "longbench-v2",
                    )
                    usages.append(result["usage"])
                    result["task_id"] = task_id
                    results.append(result)
                    save(task_dir / "result.json", result)
                    if not interactive:
                        grade = await worker.call(
                            "grade", task_id=task_id, response=result["response"], timeout=3600
                        )
                        grades.append(grade)
                        save(task_dir / "official-grade.json", grade)
                    save(directory / "run.json", record)
                summary = (
                    await worker.call("finish_profile")
                    if interactive
                    else await worker.call("summarize", grades=grades, profile=profile)
                )
                raw = profile_dir / "raw-evaluator-output.json"
                save(raw, summary.pop("raw"))
                profile_record.update(
                    summary,
                    status="COMPLETED",
                    task_count=len(results),
                    raw_evaluator_output=str(raw.resolve()),
                )
            except asyncio.CancelledError:
                profile_record.update(
                    status="NOT RUN", reason="Evaluation interrupted", primary_score=None
                )
                raise
            except Exception as exc:
                profile_record.update(status="NOT RUN", reason=f"{type(exc).__name__}: {exc}")
            finally:
                # Include attempted/incomplete task usage, which each harness flushes in finally.
                usages = [
                    json.loads(p.read_text()) for p in sorted(profile_dir.glob("task-*/usage.json"))
                ]
                profile_record.update(
                    end_time=timestamp(),
                    agent_usage=accounting(usages, time.monotonic() - profile_begin),
                    evaluator_usage=accounting(judge_usages, 0),
                    usage=accounting([*usages, *judge_usages], time.monotonic() - profile_begin),
                )
                save(profile_dir / "result.json", profile_record)
                save(directory / "run.json", record)
        selected = record["profiles"].get("buffalo", record["profiles"].get("base"))
        record.update(
            status=selected["status"],
            primary_score=selected.get("primary_score"),
            reason=selected.get("reason"),
            raw_evaluator_output=selected.get("raw_evaluator_output"),
            usage=accounting(
                [p["usage"] for p in record["profiles"].values()], time.monotonic() - begin
            ),
        )
        if profiles == ["base", "buffalo"]:
            completed = all(p["status"] == "COMPLETED" for p in record["profiles"].values())
            record["comparison"] = {
                "status": "COMPLETED" if completed else "NOT RUN",
                "contract_sha256": digest(shared),
                "base": record["profiles"]["base"].get("primary_score"),
                "buffalo": record["profiles"]["buffalo"].get("primary_score"),
            }
    except asyncio.CancelledError:
        record.update(status="NOT RUN", reason="Evaluation interrupted", primary_score=None)
        raise
    except Exception as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        await worker.close()
        record["end_time"] = timestamp()
        record["usage"]["wall_seconds"] = time.monotonic() - begin
        save(directory / "run.json", record)
    return record


def report(records):
    lines = ["BUFFALO EVALUATION", ""]
    for key, title in BENCHMARKS.items():
        row = records.get(key)
        lines.append(title)
        if not row or row["status"] != "COMPLETED":
            lines += [
                "NOT RUN",
                "reason: "
                + (row.get("reason") or "Run incomplete" if row else "Not selected for this run"),
            ]
        else:
            score = row["primary_score"]
            if key == "factorio":
                selected = row.get("profiles", {}).get(
                    "buffalo", row.get("profiles", {}).get("base", {})
                )
                lines += [
                    f"Technologies completed: {score['technologies_completed']}",
                    f"Current research progress: {score['current_research_progress_pct']}%",
                    f"Current technology: {selected.get('current_technology') or 'none'}",
                ]
            elif key == "arc-agi-3":
                lines.append(f"RHAE: {score}%")
            else:
                lines.append(f"Score: {score}")
                if key == "manyih-if":
                    selected = row.get("profiles", {}).get(
                        "buffalo", row.get("profiles", {}).get("base", {})
                    )
                    if "csr" in selected:
                        lines.append(f"CSR: {selected['csr']}")
                        lines.append(f"Official null constraints: {selected['null_constraints']}")
            if not row.get("full_benchmark", True):
                lines.append(
                    f"Explicit subset: {row['task_count']} tasks; not a full benchmark result"
                )
        lines.append("")
    lines += ["Secondary accounting (all descendants included)", ""]
    for key, row in records.items():
        for profile, result in row.get("profiles", {}).items():
            if result["status"] == "NOT RUN":
                lines.append(
                    f"{BENCHMARKS[key]} / {profile}: NOT RUN; reason: {result.get('reason', 'Run incomplete')}"
                )
            u = result["usage"]
            lines.append(
                f"{BENCHMARKS[key]} / {profile}: input={u['input_tokens']}, output={u['output_tokens']}, "
                f"total={u['total_tokens']}, API cost={u['api_cost'] if u['api_cost'] is not None else 'unavailable'}, "
                f"time={u['wall_seconds']:.2f}s"
            )
            if result.get("prior_attempt_usage"):
                prior = result["prior_attempt_usage"]
                lines.append(
                    f"{BENCHMARKS[key]} / {profile} unscored attempts: "
                    f"input={prior['input_tokens']}, output={prior['output_tokens']}, "
                    f"total={prior['total_tokens']}, API cost="
                    f"{prior['api_cost'] if prior['api_cost'] is not None else 'unavailable'}, "
                    f"time={prior['wall_seconds']:.2f}s"
                )
    if any("comparison" in r for r in records.values()):
        lines += [
            "",
            "BASE vs BUFFALO (same model, provider, reasoning, tasks, starting state and budget)",
        ]
        for key, row in records.items():
            if "comparison" in row:
                c = row["comparison"]
                lines.append(f"{BENCHMARKS[key]}: {c['base']} vs {c['buffalo']} ({c['status']})")
                if key == "factorio":
                    profiles = row.get("profiles", {})
                    lines.append(
                        "Current technologies: "
                        f"BASE={profiles.get('base', {}).get('current_technology') or 'none'}, "
                        f"BUFFALO={profiles.get('buffalo', {}).get('current_technology') or 'none'}"
                    )
    return "\n".join(lines) + "\n"


async def execute(args):
    output = args.output or Path("results/evaluation") / new_id()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use a new directory to preserve prior trajectories")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        config, config_error = load(args.config)
    except Exception as exc:
        config, config_error = EvaluationConfig(run=RunConfig()), f"{type(exc).__name__}: {exc}"
    save(output / "buffalo-source.json", source_identity())
    save(output / "manifest.json", config.model_dump(mode="json"))
    names = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    if args.token_budgets and args.benchmark not in {"arc-agi-3", "all"}:
        raise ValueError("--token-budgets is an ARC-AGI-3 sweep option")
    records = {}
    for name in names:
        setup = config.benchmarks.get(name, BenchmarkSetup())
        records[name] = await evaluate_family(
            name, setup, config, args, output / name, config_error
        )
        save(output / "report.json", records)
        (output / "report.txt").write_text(report(records))
        if name == "arc-agi-3" and args.token_budgets:
            points = []
            for budget in args.token_budgets:
                variant = config.model_copy(deep=True)
                variant.run.limits.token_budget = budget
                row = await evaluate_family(
                    name, setup, variant, args, output / f"arc-budget-{budget}", config_error
                )
                for profile, result in row.get("profiles", {}).items():
                    if result["status"] == "COMPLETED":
                        points.append(
                            {
                                "budget": budget,
                                "profile": profile,
                                "rhae": result["primary_score"],
                                **result["usage"],
                            }
                        )
            save(output / "arc-budget-points.json", points)
            curves(output, points)
    text = report(records)
    (output / "report.txt").write_text(text)
    print(text, end="")
    print(f"Run artifacts: {output.resolve()}")
    return (
        0
        if all(
            r["status"] == "COMPLETED"
            and r.get("comparison", {}).get("status", "COMPLETED") == "COMPLETED"
            for r in records.values()
        )
        else 2
    )


def curves(output, points):
    # Standalone SVGs need no plotting dependency and contain only actual official scorecards.
    for metric, label in (("output_tokens", "Output tokens"), ("api_cost", "API cost (USD)")):
        valid = [p for p in points if p[metric] is not None]
        if not any(
            len({p["budget"] for p in valid if p["profile"] == profile}) >= 2
            for profile in ("base", "buffalo")
        ):
            continue
        maximum = max(p[metric] for p in valid) or 1
        ymax = max(100, max(p["rhae"] for p in valid))
        elements = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
            '<rect width="720" height="420" fill="white"/>',
            '<path d="M70 35V355H675" fill="none" stroke="black"/>',
            f'<text x="250" y="405">{label} (0 to {maximum:g})</text>',
            f'<text x="20" y="22">RHAE % (0 to {ymax:g})</text>',
        ]
        for profile, color in (("base", "#555"), ("buffalo", "#0969da")):
            rows = sorted((p for p in valid if p["profile"] == profile), key=lambda p: p[metric])
            coords = [(70 + 590 * p[metric] / maximum, 355 - 310 * p["rhae"] / ymax) for p in rows]
            elements.append(
                f'<polyline points="{" ".join(f"{x},{y}" for x, y in coords)}" stroke="{color}" fill="none"/>'
            )
            for (x, y), point in zip(coords, rows, strict=True):
                elements.append(
                    f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>{profile}: {point[metric]}, RHAE {point["rhae"]}%</title></circle>'
                )
            elements.append(
                f'<text x="{460 if profile == "base" else 550}" y="22" fill="{color}">{profile}</text>'
            )
        elements.append("</svg>")
        (output / f"rhae-vs-{metric}.svg").write_text("\n".join(elements))
