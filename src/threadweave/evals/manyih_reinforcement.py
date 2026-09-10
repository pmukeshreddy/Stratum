"""Paired real-model lifecycle evaluation; NEVER a canonical ManyIH score run.

Each stage is a new substantive work request in one persistent root session.
There are no scheduler calls, refinement commands, forced decisions, or empty turns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from ..models import ProviderConfig, TaskConfig
from ..providers import default_providers
from ..runtime import Runtime
from .harness import MatchedProvider
from .manyih_coding import prepare_workspace, run_config
from .manyih_full_harness import source_hashes
from .reinforcement_metrics import lifecycle
from .reinforcement_report import report
from .schema import file_digest, save

ROOT = Path(__file__).parents[3]
ALGORITHMS = [3, 14, 28, 34, 35, 72, 78, 79, 88, 93, 25, 83, 19, 23]
INTERVAL_ALGORITHMS = [25, 83, 51, 8, 16, 19, 23, 7, 11, 76, 54, 37, 46, 89]


def scenario(tasks, name):
    ids = [28] if name == "exposure" else ALGORITHMS
    if name in {"interval", "interval_stream"}:
        ids = INTERVAL_ALGORITHMS
    if name == "compaction":
        ids = ALGORITHMS + [i for i in range(100) if i not in ALGORITHMS]
    if name == "natural_explicit":
        ids = [14, 28, 34, 35, 72, 78]
    stages = []
    for tid in ids:
        task = json.loads((tasks / str(tid) / "task.json").read_text())
        if name == "interval_stream":
            stages.append(
                {
                    "task_id": tid,
                    "work": "implementation_and_focused_checks",
                    "prompt": task["messages"][-1]["content"]
                    + f"\n\nAdd this function to task_{tid}/solution.py "
                    "in our ongoing algorithm collection. These style instructions apply only to this module. "
                    "Run focused correctness checks appropriate to its specification and report the results concisely. "
                    "Do not retrieve hidden tests or reference solutions.",
                }
            )
            continue
        stages.append(
            {
                "task_id": tid,
                "work": "implementation",
                "prompt": task["messages"][-1]["content"]
                + f"\n\nThis is a module in our ongoing algorithm collection. "
                f"Write the implementation to task_{tid}/solution.py and independently authored tests to task_{tid}/test_solution.py. "
                "The supplied style instructions apply to this solution file, not unrelated modules. "
                "Run the tests in the workspace's Python environment. Explain any ambiguous input-domain assumptions. "
                "Keep execution output to commands, test counts and failures; retain full test cases and logs on disk. "
                "Do not retrieve reference solutions or hidden benchmark tests.",
            }
        )
        if name == "exposure":
            stages.append(
                {
                    "task_id": tid,
                    "work": "capability_diagnostic",
                    "prompt": "Without changing any files or persistent harness entries, explain which live Python APIs "
                    "you could use to request a focused continual-harness refinement and inspect whether one is "
                    "pending. Inspect their signatures or help if needed. Explain the return values, local/global "
                    "scope, timing, and how later requests receive learned state. Also explain when you would "
                    "delegate this coding work, and the available delegation API. This is an API diagnostic, "
                    "not a request to perform refinement or delegation.",
                }
            )
        else:
            stages.append(
                {
                    "task_id": tid,
                    "work": "differential_verification",
                    "prompt": f"Now independently audit task_{tid}/solution.py. Add a simple reference oracle and "
                    "exhaustive small-domain or deterministic randomized differential tests appropriate to this "
                    "algorithm. Check mutation/aliasing, boundary inputs, and worst-case complexity where relevant. "
                    "Run these tests, investigate discrepancies against the specification, correct actual defects, "
                    "and document the evidence and complexity in this module's README.md. Keep tests reproducible "
                    "and avoid relying on a previous implementation as the only correctness oracle.",
                }
            )
    return {
        "mode": "manyih_reinforcement",
        "scenario": name,
        "canonical_score": False,
        "system": json.loads((tasks / "0/task.json").read_text())["messages"][0]["content"],
        "stages": stages,
        "provider": run_config().provider.model_dump(mode="json"),
        "buffalo_source": str(ROOT),
        "source_hashes": source_hashes(ROOT / "src"),
        "input_hashes": {str(tid): file_digest(tasks / str(tid) / "task.json") for tid in ids},
        "policy": run_config().refinement.model_dump(mode="json"),
        "limits": {"wall_seconds": 14400, "max_turns": 1000},
    }


async def buffalo(spec, directory):
    config = run_config()
    config.provider = ProviderConfig.model_validate(spec["provider"])
    config.context.max_tokens = spec["context_tokens"]
    config.limits.wall_seconds = spec["limits"]["wall_seconds"]
    config.limits.max_turns = spec["limits"]["max_turns"]
    config.limits.token_budget = 30_000_000
    config.task = TaskConfig(instruction_messages=[{"role": "system", "content": spec["system"]}])
    assert config.refinement.model_dump(mode="json") == spec["policy"]
    provider = MatchedProvider(
        default_providers()[config.provider.name],
        config.provider,
        directory / "provider-calls.jsonl",
    )
    runtime = Runtime(directory / "state", providers={config.provider.name: provider})
    save(directory / "buffalo-config.json", config.model_dump(mode="json"))
    session = runtime.create(spec["stages"][0]["prompt"], directory / "workspace", config=config)
    if spec.get("initial_harness_state"):
        save(
            runtime.store.harness.path(session.id) / "harness_state.json",
            spec["initial_harness_state"],
        )
    save(directory / "initial-harness-state.json", runtime.store.harness.merged(session.id))
    await runtime.start()
    try:
        for index, stage in enumerate(spec["stages"]):
            if index:
                runtime.interact(session.id, stage["prompt"])
            session = await runtime.wait(session.id, timeout=1800)
            save(directory / "lifecycle.json", lifecycle(directory))
            with (directory / "stages.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "stage": index,
                            "root_turns": session.turns,
                            "outcome": session.outcome,
                            "time": time.time(),
                            "response": session.result,
                        }
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "engine": "buffalo",
                        "stage": index,
                        "root_turns": session.turns,
                        "outcome": session.outcome,
                    }
                ),
                flush=True,
            )
            if session.outcome != "completed":
                raise RuntimeError(f"Stage {index}: {session.outcome}: {session.last_error}")
    finally:
        await runtime.shutdown()
        save(directory / "lifecycle.json", lifecycle(directory))


async def prime(spec, manifest, directory, prime_source):
    executable = prime_source / "node_modules/.bin/tsx"
    if not executable.is_file():
        raise RuntimeError("Prime source requires npm ci; no installed tsx was found")
    with (directory / "stderr.log").open("w") as stderr:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            "--tsconfig",
            str(prime_source / "tsconfig.json"),
            str(Path(__file__).with_name("prime_reinforcement.mjs")),
            str(prime_source),
            str(manifest),
            str(directory),
            sys.executable,
            stderr=stderr,
            env={**os.environ, "PRIME_AGENT_TELEMETRY": "0", "DO_NOT_TRACK": "1"},
        )
        try:
            async with asyncio.timeout(spec["limits"]["wall_seconds"]):
                code = await process.wait()
            if code:
                raise RuntimeError(f"Prime exited {code}; see {directory / 'stderr.log'}")
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()


async def execute(args):
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    spec = scenario(args.tasks, args.scenario)
    spec["context_tokens"] = args.context_tokens
    if args.output_tokens:
        spec["provider"]["max_output_tokens"] = args.output_tokens
    if args.seed_state:
        spec["initial_harness_state"] = json.loads(args.seed_state.read_text())
        spec["seed_state_sha256"] = file_digest(args.seed_state)
        spec["seed_state_source"] = str(args.seed_state)
    if args.stage_limit:
        spec["stages"] = spec["stages"][: args.stage_limit]
    spec["prime_source_hashes"] = {
        str(path.relative_to(args.prime_source)): file_digest(path)
        for name in (
            "packages/coding-agent/src",
            "packages/ai/src",
            "packages/coding-agent/skills/refine",
        )
        for path in (args.prime_source / name).rglob("*")
        if path.is_file() and path.suffix in {".ts", ".py", ".md"}
    }
    spec["prime_driver_sha256"] = file_digest(Path(__file__).with_name("prime_reinforcement.mjs"))
    if args.seed_workspace:
        frozen = directory / "seed-workspace/task_28"
        shutil.copytree(
            args.seed_workspace / "task_28",
            frozen,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
        spec["seed_workspace_hashes"] = {
            str(p.relative_to(frozen)): file_digest(p) for p in frozen.rglob("*") if p.is_file()
        }
    manifest = directory / "scenario.json"
    save(manifest, spec)
    for engine in args.engines.split(","):
        target = directory / engine
        target.mkdir()
        prepare_workspace(
            target / "workspace", {"messages": [{"content": spec["stages"][0]["prompt"]}]}
        )
        if args.seed_workspace:
            seeded = target / "workspace/task_28"
            shutil.copytree(
                directory / "seed-workspace/task_28",
                seeded,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
            )
    jobs = [
        buffalo(spec, directory / "buffalo")
        if engine == "buffalo"
        else prime(spec, manifest, directory / "prime", args.prime_source)
        for engine in args.engines.split(",")
    ]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    print(report(directory), flush=True)
    save(
        directory / "run-status.json",
        {
            engine: "completed" if not isinstance(result, BaseException) else {"error": str(result)}
            for engine, result in zip(args.engines.split(","), results, strict=True)
        },
    )
    if any(isinstance(result, BaseException) for result in results):
        raise RuntimeError(f"Incomplete evaluation; see {directory / 'run-status.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", required=True, help="Directory containing official ID/task.json inputs"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=96000,
        help="Shared evaluation context allocation; does not change refinement settings",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        help="Shared model output reservation for a context-limited profile",
    )
    parser.add_argument(
        "--seed-state",
        type=Path,
        help="Observed harness JSON fixture; no refinement function is called",
    )
    parser.add_argument(
        "--seed-workspace", type=Path, help="Observed task_28 model-authored workspace fixture"
    )
    parser.add_argument(
        "--stage-limit",
        type=int,
        help="Bound the actual work requests, never assistant turns or triggers",
    )
    parser.add_argument("--prime-source", default=str(ROOT.parent / "prime-agent-main"))
    parser.add_argument(
        "--scenario",
        choices=["exposure", "interval", "interval_stream", "compaction", "natural_explicit"],
        required=True,
    )
    parser.add_argument(
        "--engines", choices=["buffalo", "prime", "buffalo,prime"], default="buffalo,prime"
    )
    args = parser.parse_args()
    for key in ("tasks", "output", "prime_source"):
        setattr(args, key, Path(getattr(args, key)).resolve())
    for key in ("seed_state", "seed_workspace"):
        if getattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
