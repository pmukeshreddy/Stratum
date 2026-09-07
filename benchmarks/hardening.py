"""Local component performance, not coding solve-rate or external benchmark results.

uv run python benchmarks/hardening.py --output results/hardening/performance.json
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from threadweave.artifacts import atomic_write
from threadweave.coding import CodingTask
from threadweave.gitops import GitWorkspace, git
from threadweave.machine import metadata
from threadweave.models import Action, RunConfig, new_id
from threadweave.retrieval import search
from threadweave.runtime import Runtime
from threadweave.test_selection import related
from threadweave.tools import ToolContext


async def measure(directory, count, repetitions):
    root = directory / "repo"
    root.mkdir(parents=True)
    for i in range(count):
        group = root / f"package{i // 100:04d}"
        group.mkdir(exist_ok=True)
        (group / f"file{i:05d}.py").write_text(
            f"def function_{i}(value):\n    return helper(value)\n"
        )
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Measurement",
        "-c",
        "user.email=local@localhost",
        "commit",
        "-qm",
        "Input",
    )
    config = RunConfig(
        task={
            "adapter": "coding",
            "capture_baseline": False,
            "require_tests": False,
            "require_change": False,
        },
        permissions=["python", "process", "workspace.read", "workspace.write", "agents", "state"],
        limits={"wall_seconds": 3600, "max_subagents": 30},
    )
    runtime = Runtime(directory / "state")  # production provider configured but never invoked
    session = runtime.create("Component measurement only", root, config=config, mode="interactive")
    context = ToolContext(
        runtime, session.id, new_id(), runtime.store.event(session.id, "measurement", {})
    )
    results = []

    async def timed(name, operation, n=None):
        samples = []
        last = None
        for _ in range(n or repetitions):
            start = time.perf_counter()
            last = operation()
            if hasattr(last, "__await__"):
                last = await last
            samples.append((time.perf_counter() - start) * 1000)
        row = {
            "operation": name,
            "files": count,
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        return last

    async def cell(code):
        return await runtime._execute_action(
            session.id,
            new_id(),
            Action(name="ipython", arguments={"code": code}),
            runtime.store.event(session.id, "measurement", {}),
        )

    try:
        index = runtime.index(session.id)
        await timed("index_initial", index.refresh, 1)
        await runtime.environment.prepare(session.id, force=True)
        index.definition("function_0")  # watch stream warmup
        await timed("symbol_lookup", lambda: index.definition("function_0"))
        await timed("references_lookup", lambda: index.references("helper", limit=10))
        await timed("dependency_lookup", lambda: index.dependencies("package0000/file00000.py"))
        for size in (1, 10):
            serial = 0

            def update(size=size):
                nonlocal serial
                serial += 1
                paths = [f"package0000/file{i:05d}.py" for i in range(size)]
                for i, path in enumerate(paths):
                    (root / path).write_text(
                        f"def function_{i}(value):\n    return helper(value) + {serial}\n"
                    )
                return index.refresh(paths)

            await timed(f"index_update_{size}", update)
        observer = runtime.environment.mutations
        observer.reconcile(context, reason="checkpoint")
        await cell("warm = 0")
        for edits in (0, 1, 10):
            observed = []
            for iteration in range(repetitions):
                code = (
                    "warm += 1"
                    if not edits
                    else (
                        f"for i in range({edits}):\n    Path(f'package0000/file{{i:05d}}.py').write_text(f'def function_{{i}}(value):\\n    return value + {iteration}\\n')"
                    )
                )
                begin, end = observer.begin, observer.end
                times = []

                def wrapped(fn, times=times):
                    def call(*args, **kwargs):
                        start = time.perf_counter()
                        try:
                            return fn(*args, **kwargs)
                        finally:
                            times.append(time.perf_counter() - start)

                    return call

                observer.begin, observer.end = wrapped(begin), wrapped(end)
                try:
                    await cell(code)
                finally:
                    observer.begin, observer.end = begin, end
                observed.append(sum(times) * 1000)
            row = {
                "operation": f"mutation_{edits}_edits",
                "files": count,
                "samples_ms": observed,
                "median_ms": statistics.median(observed),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        await cell(
            "import subprocess, sys\nbackground = subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(.05); Path('background.txt').write_text('done')\"]) "
        )
        await asyncio.sleep(0.15)
        await timed(
            "background_observation",
            lambda: observer.reconcile(context, reason="background_measurement"),
            1,
        )
        await timed(
            "full_reconciliation", lambda: observer.reconcile(context, reason="before_verifier")
        )
        for i in range(1000):
            runtime.store.event(
                session.id, "observation", {"file": f"file{i}.py", "text": "helper function failed"}
            )
        await timed(
            "history_retrieval", lambda: search(runtime.store, session.id, "helper failed", limit=5)
        )
        for i in range(20):
            e = runtime.store.event(
                session.id, "observation", {"detail": "Failed attempt " + str(i)}
            )
            runtime.store.add_context(
                session.id, e, [{"role": "user", "content": "failure evidence " * 1000}]
            )
        await timed("context_compaction", lambda: runtime.context.compact(session.id), 1)
        await timed(
            "related_test_lookup", lambda: related(context, files=["package0000/file00000.py"])
        )
        await timed("repl_noop", lambda: cell("warm += 1"))
        for name, code in [
            ("small", "small = 1"),
            ("large_bytes", "data = b'x' * 8000000"),
            ("unchanged_large_bytes", "small += 1"),
            ("many_variables", "globals().update({f'value_{i}': i for i in range(1000)})"),
            ("large_container", "items = list(range(100000))"),
            ("unchanged_container", "small += 1"),
        ]:
            context.action_id = new_id()
            result = await runtime.execute_python(context, code)
            results.append(
                {
                    "operation": "snapshot_" + name,
                    "files": count,
                    **result.get("snapshot_metrics", {}),
                }
            )
        from threadweave.background import BackgroundProcesses

        runtime.background = BackgroundProcesses(runtime)
        await timed("process_spawn", lambda: runtime.background.start(context, "true"))
        await runtime.background.close_session(session.id)
        await timed(
            "rlm_shared_admission",
            lambda: runtime.spawn(session.id, "No scheduled model work", purpose="shared"),
        )
        workspace = GitWorkspace(context)
        for number in (1, 4):
            start = time.perf_counter()
            checkpoint = workspace.snapshot_tree()
            paths = [workspace.isolate(checkpoint) for _ in range(number)]
            elapsed = (time.perf_counter() - start) * 1000
            results.append(
                {
                    "operation": f"candidate_spawn_{number}",
                    "files": count,
                    "median_ms": elapsed,
                    "checkout_bytes": sum(
                        p.stat().st_size for root in paths for p in root.rglob("*") if p.is_file()
                    ),
                    "object_database": "shared",
                    "admission": "serial synchronous worktree admission; child execution is concurrent",
                }
            )
            await timed("candidate_diff", lambda paths=paths: git(paths[0], "diff", "HEAD"), 1)
            await timed(
                "candidate_cleanup",
                lambda paths=paths: [workspace.cleanup_isolation(p) for p in paths],
                1,
            )
        await timed("verifier_no_commands", lambda: CodingTask().verify(context, config.task), 1)
        assert runtime.store.usage(session.id, tree=True).model_calls == 0
        return results
    finally:
        await runtime.shutdown()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 10000, 30000])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="threadweave-performance-") as directory:
        for size in args.sizes:
            rows += await measure(Path(directory) / str(size), size, args.repetitions)
            atomic_write(
                args.output,
                json.dumps(
                    {
                        "environment": metadata(),
                        "model_calls": 0,
                        "scope": "synthetic file corpus for component latency only, not coding effectiveness",
                        "rows": rows,
                    },
                    indent=2,
                ).encode(),
            )


if __name__ == "__main__":
    asyncio.run(main())
