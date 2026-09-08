"""Current production component latency and local semantic retrieval measurements.

Generated files are latency fixtures only, never coding-evaluation tasks.
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
from threadweave.evaluation import source_identity
from threadweave.gitops import git
from threadweave.kernel import Kernel
from threadweave.machine import metadata
from threadweave.models import RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext


async def main(output, embedding_cache):
    rows = []

    async def timed(name, operation, *, count=None, n=3):
        samples = []
        result = None
        for _ in range(n):
            start = time.perf_counter()
            result = operation()
            if hasattr(result, "__await__"):
                result = await result
            samples.append((time.perf_counter() - start) * 1000)
        row = {
            "operation": name,
            "files": count,
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        return result

    with tempfile.TemporaryDirectory(prefix="threadweave-quality-") as temporary:
        base = Path(temporary)
        for count in (100, 10000, 30000):
            root = base / str(count) / "repo"
            root.mkdir(parents=True)
            for i in range(count):
                file = root / f"pkg{i // 100:04d}" / f"f{i:05d}.py"
                file.parent.mkdir(exist_ok=True)
                file.write_text(f"def function_{i}(x):\n    return x + 1\n")
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
                "Latency input",
            )
            config = RunConfig(
                permissions=[
                    "python",
                    "process",
                    "workspace.read",
                    "workspace.write",
                    "agents",
                    "state",
                ],
                task={
                    "adapter": "coding",
                    "capture_baseline": False,
                    "require_tests": False,
                    "require_change": False,
                },
                limits={"max_subagents": 10, "wall_seconds": 3600},
            )
            runtime = Runtime(base / str(count) / "state")
            session = runtime.create(
                "Latency measurement; no model invocation", root, config=config, mode="interactive"
            )
            context = ToolContext(
                runtime, session.id, new_id(), runtime.store.event(session.id, "measurement", {})
            )
            try:
                await runtime.environment.prepare(session.id, force=True)
                (root / "pkg0000/f00000.py").write_text("def function_0(x):\n    return x + 2\n")
                runtime.environment.mutations.reconcile(context, reason="measurement")
                await timed(
                    "verifier_full_trust_boundary",
                    lambda context=context, config=config: CodingTask().verify(
                        context, config.task
                    ),
                    count=count,
                )
                index = runtime.index(session.id)
                await timed(
                    "resolved_symbol_lookup",
                    lambda index=index: index.definition("function_0"),
                    count=count,
                )
                if count < 30000:

                    async def candidates(runtime=runtime, session=session):
                        children = await asyncio.gather(
                            *(
                                runtime.spawn_async(
                                    session.id,
                                    "Return evidence",
                                    name="candidate-" + new_id()[:8],
                                    purpose="candidate",
                                )
                                for _ in range(4)
                            )
                        )
                        await asyncio.gather(*(runtime._kernel(c.id).start() for c in children))
                        assert len({runtime._kernel(c.id).process.pid for c in children}) == 4
                        return [c.id for c in children]

                    ids = await timed(
                        "four_candidate_admissions_and_kernels", candidates, count=count, n=1
                    )
                    rows[-1]["child_ids"] = ids
                assert runtime.store.usage(session.id, tree=True).model_calls == 0
            finally:
                await runtime.shutdown()

        async def bridge(name, arguments):
            raise ValueError("No host actions used by snapshot measurement")

        kernel = Kernel(base / "kernel", base, bridge)
        try:
            cases = [
                ("large_list", "value = list(range(150000))", "value[400] += 1"),
                ("large_dict", "value = dict(enumerate(range(100000)))", "value[400] += 1"),
                (
                    "large_array",
                    "import numpy as np\nvalue = np.arange(500000,dtype=np.float64)",
                    "value[400] += 1",
                ),
            ]
            await kernel.start()
            for name, setup, edit in cases:
                result = await timed(
                    name + "_initial", lambda setup=setup: kernel.execute(new_id(), setup, 15), n=1
                )
                rows[-1]["snapshot"] = result.get("snapshot_metrics")
                assert not result["error"], result
                result = await timed(
                    name + "_unchanged", lambda: kernel.execute(new_id(), "pass", 15)
                )
                rows[-1]["snapshot"] = result.get("snapshot_metrics")
                result = await timed(
                    name + "_changed", lambda edit=edit: kernel.execute(new_id(), edit, 15)
                )
                rows[-1]["snapshot"] = result.get("snapshot_metrics")
        finally:
            await kernel.close()

        if embedding_cache:
            from threadweave.retrieval import search

            runtime = Runtime(base / "retrieval")
            try:
                config = RunConfig(
                    context={
                        "embedding_model": "BAAI/bge-small-en-v1.5",
                        "embedding_cache": embedding_cache,
                    }
                )
                session = runtime.create(
                    "Investigate a traversal that never returns",
                    base,
                    config=config,
                    mode="interactive",
                )
                wanted = runtime.store.event(
                    session.id,
                    "failure",
                    {
                        "detail": "Cyclic edges make graph traversal loop forever unless visited vertices are remembered"
                    },
                )
                runtime.store.event(
                    session.id,
                    "failure",
                    {"detail": "CSS border colors are incorrect on the login screen"},
                )
                runtime.store.event(
                    session.id,
                    "failure",
                    {"detail": "Database inserts fail because the disk is full"},
                )
                result = await timed(
                    "hybrid_semantic_cold",
                    lambda: search(
                        runtime.store,
                        session.id,
                        "avoid nontermination during breadth-first exploration",
                        limit=2,
                    ),
                    n=1,
                )
                assert result[0]["id"] == wanted, result
                assert "semantic" in result[0].get("sources", []), result
                result = await timed(
                    "hybrid_semantic_warm",
                    lambda: search(
                        runtime.store,
                        session.id,
                        "avoid nontermination during breadth-first exploration",
                        limit=2,
                    ),
                )
                rows[-1]["evidence"] = result
            finally:
                await runtime.shutdown()
    atomic_write(
        output,
        json.dumps(
            {
                "environment": metadata(),
                "source": source_identity(),
                "model_invocations": 0,
                "scope": "component measurements; synthetic latency/retrieval fixtures, not coding task scores",
                "rows": rows,
            },
            indent=2,
        ).encode(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache")
    args = parser.parse_args()
    asyncio.run(main(args.output, args.embedding_cache))
