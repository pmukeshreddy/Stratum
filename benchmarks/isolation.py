"""Measure actual GitWorkspace isolation with concurrent backend requests.

The runtime admission method is still synchronous; this is backend capacity, not
a claim that its scheduler parallelizes admissions. No model is invoked.
"""

import argparse
import asyncio
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from threadweave.artifacts import Artifacts, atomic_write
from threadweave.gitops import GitWorkspace, git
from threadweave.machine import metadata
from threadweave.models import RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.storage import Store
from threadweave.tools import ToolContext


async def measure(directory, count):
    root = directory / "repository"
    root.mkdir(parents=True)
    for i in range(count):
        path = root / f"package{i // 100}" / f"file{i}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"VALUE = {i}\n" + "# data\n" * 500)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Local", "-c", "user.email=a@b", "commit", "-qm", "Input")
    runtime = Runtime(directory / "state")
    session = runtime.create("Component measurement only", root, config=RunConfig())
    event = runtime.store.event(session.id, "measurement", {})
    workspace = GitWorkspace(ToolContext(runtime, session.id, new_id(), event))
    rows = []
    try:
        # Capture a dirty parent's deterministic input, unchanged by parallel admissions.
        (root / "dirty.py").write_text("VALUE = 1\n")
        for number in (1, 4):
            start = time.perf_counter()
            checkpoint = workspace.snapshot_tree()
            checkpoint_ms = (time.perf_counter() - start) * 1000
            barrier = threading.Barrier(number)

            def spawn(barrier=barrier, checkpoint=checkpoint):
                store = Store(directory / "state")  # connection belongs to this thread
                try:
                    service = SimpleNamespace(store=store, artifacts=Artifacts(store))
                    context = ToolContext(service, session.id, new_id(), event)
                    barrier.wait(timeout=10)
                    return GitWorkspace(context).isolate(checkpoint)
                finally:
                    store.close()

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=number) as pool:
                futures = [pool.submit(spawn) for _ in range(number)]
                paths = [future.result() for future in futures]
            spawn_ms = (time.perf_counter() - start) * 1000
            assert all((p / "dirty.py").read_text() == "VALUE = 1\n" for p in paths)
            (paths[0] / "dirty.py").write_text("VALUE = 2\n")
            assert (root / "dirty.py").read_text() == "VALUE = 1\n"
            assert all((p / "dirty.py").read_text() == "VALUE = 1\n" for p in paths[1:])
            start = time.perf_counter()
            diff = git(paths[0], "diff", "HEAD")
            diff_ms = (time.perf_counter() - start) * 1000
            checkout_bytes = sum(
                p.stat().st_size for child in paths for p in child.rglob("*") if p.is_file()
            )
            start = time.perf_counter()
            for path in paths:
                workspace.cleanup_isolation(path)
            row = {
                "files": count + 1,
                "concurrent_backend_requests": number,
                "checkpoint_ms": checkpoint_ms,
                "spawn_ms": spawn_ms,
                "diff_ms": diff_ms,
                "diff_bytes": len(diff.encode()),
                "cleanup_ms": (time.perf_counter() - start) * 1000,
                "checkout_bytes": checkout_bytes,
                "source_bytes": sum(p.stat().st_size for p in root.rglob("*.py")),
                "git_objects": "shared",
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        return rows
    finally:
        await runtime.shutdown()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 10000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="threadweave-isolation-") as directory:
        for count in args.sizes:
            rows += await measure(Path(directory) / str(count), count)
    atomic_write(
        args.output, json.dumps({"environment": metadata(), "rows": rows}, indent=2).encode()
    )


if __name__ == "__main__":
    asyncio.run(main())
