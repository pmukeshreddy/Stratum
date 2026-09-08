"""Live API usability probe on this source tree. Not a coding benchmark score."""

import argparse
import asyncio
import collections
import json
import os
import shutil
from pathlib import Path

from threadweave.artifacts import atomic_write
from threadweave.evaluation import source_identity
from threadweave.models import RunConfig
from threadweave.runtime import Runtime


async def run(output, turns=12, tokens=50000):
    os.environ.pop("OPENAI_API_KEY", None)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    repository = Path(__file__).resolve().parent.parent  # noqa: ASYNC240 - pre-runtime fixture setup
    workspace = output / "workspace"
    for name in ("src", "tests"):
        shutil.copytree(
            repository / name, workspace / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    shutil.copyfile(repository / "pyproject.toml", workspace / "pyproject.toml")
    config = RunConfig(
        provider={
            "model": "gpt-6-astra",
            "parameters": {"reasoning_effort": "medium"},
            "max_output_tokens": 4096,
            "timeout_seconds": 150,
        },
        permissions=["python", "workspace.read", "agents", "state"],
        execution={"read_only": True},
        limits={
            "max_turns": turns,
            "token_budget": tokens,
            "wall_seconds": 300,
            "max_subagents": 2,
            "concurrency": 3,
        },
        refinement={"automatic": False},
    )
    task = """Read-only API usability check on the supplied real Threadweave source tree. This is not a bug-fixing benchmark. Do not modify any repository file.
Inspect repo.help(), tests.help(), agents.help(), context.help(). Find the Runtime.message definition and its callers using repo navigation, then request related tests for that implementation. Store the investigation using context.focus. Retrieve relevant durable evidence with context.search.
Create one research child with await rlm, asking it to independently inspect message persistence in storage.py and send you an explicit message with a source citation. Keep the handle and continue your own inspection while it works. Read its message and integrate the evidence. Do not wait forever: respect resource limits.
Conclude with short source-cited findings and the actual related-test choices. Use these public APIs and print bounded results, rather than loading whole files. Report any API error honestly."""
    runtime = Runtime(output / "state")
    session = runtime.create(task, workspace, config=config)
    atomic_write(output / "config.json", config.model_dump_json(indent=2).encode())
    atomic_write(output / "source.json", json.dumps(source_identity(), indent=2).encode())
    try:
        await runtime.start()
        result = await runtime.wait(session.id, timeout=330)
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in runtime.store.db.execute("SELECT * FROM events ORDER BY seq")
        ]
        atomic_write(
            output / "events.jsonl", "".join(json.dumps(e) + "\n" for e in events).encode()
        )
        summary = {
            "scope": "live API usability, not a solved coding task",
            "session_id": session.id,
            "outcome": result.outcome,
            "usage": runtime.store.usage(session.id, tree=True).model_dump(),
            "event_counts": dict(collections.Counter(e["type"] for e in events)),
            "tools": dict(
                collections.Counter(
                    e["payload"]["name"] for e in events if e["type"] == "tool_call"
                )
            ),
        }
        atomic_write(output / "summary.json", json.dumps(summary, indent=2).encode())
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=50000)
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve(), args.turns, args.tokens))
