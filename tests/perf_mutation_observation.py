"""Local diagnostic measurement, not an agent/model/task benchmark.

Run: uv run python -m tests.perf_mutation_observation --output results/hardening-step1/mutation-overhead.json
"""

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from threadweave.gitops import git
from threadweave.models import Action, RunConfig, new_id
from threadweave.runtime import Runtime

from .fakes import ScriptedProvider


async def measure(directory, count, repetitions):
    root = directory / "repo"
    root.mkdir(parents=True)
    for i in range(count):
        (root / f"file{i:05d}.txt").write_text((f"file {i} content\n" + "x" * 4096)[:4096])
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Measurement",
        "-c",
        "user.email=measurement@localhost",
        "commit",
        "-qm",
        "Input",
    )
    config = RunConfig(
        provider={"name": "test", "model": "not-invoked"},
        task={"adapter": "coding", "require_tests": False, "capture_baseline": False},
        limits={"wall_seconds": 3600},
    )
    runtime = Runtime(directory / "state", providers={"test": ScriptedProvider({})})
    session = runtime.create("Measure observation only", root, config=config, mode="interactive")
    try:
        await runtime.environment.prepare(session.id, force=True)
        observer = runtime.environment.mutations
        metrics = {"seconds": 0, "hash_bytes": 0, "hash_files": 0}
        begin, end, scan = observer.begin, observer.end, observer.scan

        def timed(fn):
            def call(*a, **kw):
                start = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    metrics["seconds"] += time.perf_counter() - start

            return call

        def scanned(*args):
            result = scan(*args)
            metrics["hash_bytes"] += observer.stats["hashed_bytes"]
            metrics["hash_files"] += observer.stats["hashed_files"]
            return result

        observer.begin, observer.end, observer.scan = timed(begin), timed(end), scanned

        async def cell(code):
            event = runtime.store.event(session.id, "measurement", {})
            result = await runtime._execute_action(
                session.id, new_id(), Action(name="ipython", arguments={"code": code}), event
            )
            if result.get("error"):
                raise RuntimeError(result)

        await cell("measurement = 0")  # warm kernel; cold baseline is excluded
        rows = []
        for edits in (0, 1, 10) if count < 1000 else (0, 1):
            elapsed, observed, reads = [], [], []
            for iteration in range(repetitions):
                code = (
                    f"measurement = {iteration}"
                    if not edits
                    else f"for i in range({edits}):\n    Path(f'file{{i:05d}}.txt').write_text({str(iteration)!r} + 'z' * 4095)"
                )
                metrics.update(seconds=0, hash_bytes=0, hash_files=0)
                start = time.perf_counter()
                await cell(code)
                elapsed.append((time.perf_counter() - start) * 1000)
                observed.append(metrics["seconds"] * 1000)
                reads.append({"bytes": metrics["hash_bytes"], "files": metrics["hash_files"]})
            rows.append(
                {
                    "repository_files": count,
                    "repository_bytes": count * 4096,
                    "edited_files": edits,
                    "repetitions": repetitions,
                    "median_cell_ms": statistics.median(elapsed),
                    "median_observation_ms": statistics.median(observed),
                    "max_observation_ms": max(observed),
                    "cell_ms": elapsed,
                    "observation_ms": observed,
                    "content_hashed_per_cell": reads,
                }
            )
        return rows
    finally:
        await runtime.shutdown()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-files", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="threadweave-observation-") as temporary:
        result = {
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
            "model_calls": 0,
            "scope": "warm local Python cells; observation boundary time excludes preparation and cold hashing",
            "measurements": [
                *await measure(Path(temporary) / "small", 100, args.repetitions),
                *await measure(Path(temporary) / "large", args.large_files, args.repetitions),
            ],
        }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
