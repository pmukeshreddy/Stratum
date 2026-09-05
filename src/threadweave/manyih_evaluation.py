"""Evaluate externally installed official ManyIH Coding data through Runtime.

This is task formatting and final grading, not a benchmark-specific agent. The
official prompts and grader are imported from the supplied checkout. Hidden tests
and expected-style metadata never enter the agent workspace or model context.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .cli import resolved_config
from .configuration import load_config
from .evaluation import metrics as trajectory_metrics
from .models import TaskConfig, Usage
from .runtime import Runtime

HARNESS_ROOT = Path(__file__).resolve().parents[2]

OFFICIAL_CALL = """
import json,sys
sys.path.insert(0, sys.argv[1])
payload=json.load(sys.stdin)
if sys.argv[2] == 'format':
    from manyih.coding.evaluator import format_datapoint_for_llm
    result=format_datapoint_for_llm(**payload)
else:
    from manyih.coding.evaluate import judge_response
    result=judge_response(**payload)
print(json.dumps(result))
"""


async def official(source, python, operation, payload):
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-c",
        OFFICIAL_CALL,
        str(source),
        operation,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Do not give generated code/grading subprocesses inherited credentials.
        env={
            key: os.environ[key]
            for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
            if key in os.environ
        },
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps(payload).encode()), 90
        )
    except BaseException:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(
            f"Official ManyIH {operation} failed: {stderr.decode(errors='replace')[:2000]}"
        )
    return json.loads(stdout)


async def evaluate_manyih(source, python, config, output, *, start=0, limit=2):
    source, python, output = Path(source).resolve(), Path(python).absolute(), Path(output).resolve()  # noqa: ASYNC240 - preserve virtualenv interpreter path rather than its symlink target
    if start < 0 or limit < 1:
        raise ValueError("start must be nonnegative and limit positive")
    data_path = source / "manyih" / "data" / "coding.json"
    if not data_path.is_file() or not python.is_file():
        raise ValueError("Supply the official ManyIH checkout and its installed Python environment")
    raw_bytes = data_path.read_bytes()
    dataset = json.loads(raw_bytes)
    rows, data_config = dataset["data"], dataset.get("config", {})
    selected = rows[start : start + limit]
    if not selected:
        raise ValueError("Selected ManyIH range contains no instances")
    config = await resolved_config(config.model_copy(deep=True))
    if config.provider.name != "codex_subscription" or config.models:
        raise ValueError(
            "This benchmark run requires the production subscription provider without routing overrides"
        )
    config.task = TaskConfig(adapter="workspace", verify_each_turn=False, require_verifier=False)
    config.tool_allowlist = [
        "python",
        "finish",
        "artifact_read",
        "history_search",
        "history_get",
        "session_inspect",
    ]
    config.limits.max_subagents = 0
    # Automatic refinement is disabled in this fixed smoke configuration, not reported as an ablation.
    config.refinement.enabled = False
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; use a new directory to preserve prior trajectories")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    provenance = {
        "benchmark": "ManyIH Coding",
        "harness_commit": (
            await asyncio.to_thread(
                subprocess.check_output,
                ["git", "rev-parse", "HEAD"],
                cwd=HARNESS_ROOT,
                text=True,
            )
        ).strip(),
        "source_commit": await asyncio.to_thread(
            subprocess.check_output, ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ),
        "data_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "data_config": data_config,
        "total_public_instances": len(rows),
        "start": start,
        "limit": limit,
        "config": config.model_dump(mode="json"),
        "hidden_verifier_feedback": False,
        "comparison": "NOT DIRECTLY COMPARABLE: model, budget and subset differ from the paper",
    }
    print(json.dumps(provenance, indent=2), flush=True)
    (output / "run_config.json").write_text(json.dumps(provenance, indent=2))
    results_path = output / "results.jsonl"
    if results_path.exists():
        raise ValueError(
            "Output already contains results; choose a new directory (no silent overwrite)"
        )
    results = []
    for offset, row in enumerate(selected, start=start):
        directory = output / f"instance-{offset:04d}"
        directory.mkdir(mode=0o700)
        workspace = directory / "workspace"
        workspace.mkdir()
        formatted = await official(
            source,
            python,
            "format",
            {
                "datapoint": {"prompt": row["prompt"]},
                "include_system_prompt": True,
                "hierarchy_format": data_config.get("hierarchy_format", "scalar"),
                "annotation_style": data_config.get("annotation_style", "inline"),
            },
        )
        if formatted["user_prompt"] != row["prompt"]:
            raise RuntimeError("Official formatter changed the dataset prompt")
        (directory / "task_input.json").write_text(
            json.dumps(formatted, ensure_ascii=False, indent=2)
        )
        runtime = Runtime(directory / "state")
        started = time.monotonic()
        session = runtime.create(
            "Complete the supplied official coding instance. Return your final code with finish.",
            workspace,
            config=config,
        )
        event = runtime.store.event(
            session.id,
            "benchmark_input",
            {
                "benchmark": "ManyIH Coding",
                "task_id": row["task_id"],
                "instance_id": row["id"],
                "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            },
        )
        runtime.store.add_context(
            session.id,
            event,
            [
                {"role": "system", "content": formatted["system_prompt"]},
                {"role": "user", "content": formatted["user_prompt"]},
            ],
        )
        try:
            await runtime.start()
            finished = await runtime.wait(session.id, timeout=config.limits.wall_seconds + 15)
            # Exactly one independent official judgment after the agent has stopped.
            runtime.store.charge(session.id, Usage(verifier_calls=1))
            grade = await official(
                source,
                python,
                "judge",
                {
                    "response": finished.result if finished.outcome == "completed" else "",
                    "test_code": row["test_code"],
                    "expected_styles": row.get("metadata", {}).get("expected_styles", {}),
                    "timeout": 5.0,
                },
            )
            runtime.store.event(session.id, "official_benchmark_verifier", grade)
            result = {
                "benchmark": "ManyIH Coding",
                "instance_id": row["id"],
                "task_id": row["task_id"],
                "offset": offset,
                "session_id": session.id,
                "outcome": finished.outcome,
                "solved": grade["overall_passed"],
                "score": float(grade["overall_passed"]),
                "grade": grade,
                "wall_seconds": time.monotonic() - started,
                "metrics": trajectory_metrics(runtime, session.id),
                "usage": runtime.store.usage(session.id, tree=True).model_dump(),
                "official_verifier_calls": 1,
            }
            (directory / "result.json").write_text(json.dumps(result, indent=2))
            with results_path.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            results.append(result)
            print(
                json.dumps({k: result[k] for k in ("task_id", "solved", "wall_seconds", "usage")}),
                flush=True,
            )
        finally:
            await runtime.shutdown()
    summary = {
        "tasks_run": len(results),
        "solved": sum(r["solved"] for r in results),
        "solve_rate": sum(r["solved"] for r in results) / len(results),
        "full_benchmark": start == 0 and len(results) == len(rows),
        "cost": None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    result = asyncio.run(
        evaluate_manyih(
            args.source,
            args.python,
            load_config(args.config),
            args.output,
            start=args.start,
            limit=args.limit,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
