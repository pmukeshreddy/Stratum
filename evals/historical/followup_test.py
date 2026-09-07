"""Evaluator-only regression for a real historical completed-child defect."""

import asyncio

from tests.conftest import response
from tests.fakes import ScriptedProvider
from threadweave.models import ModelResponse, RunConfig
from threadweave.runtime import Runtime


async def test_same_completed_child_continues(tmp_path):
    config = RunConfig(
        provider={"name": "mock", "model": "test-only-verifier"},
        refinement={"enabled": False},
        features={"model_compaction": False},
        limits={"max_turns": 20, "wall_seconds": 60},
    )
    provider = ScriptedProvider(
        {
            "worker": [
                response("ipython", code="x = 123"),
                ModelResponse(text="done"),
                response(
                    "ipython",
                    code="assert x == 123\nx += 1\nawait agent_message.send(str(x), receiver_role='parent')",
                ),
                ModelResponse(text="done again"),
            ]
        }
    )
    directory = tmp_path / "state"
    runtime = Runtime(directory, providers={"mock": provider})
    try:
        root = runtime.create("parent", tmp_path, config=config, mode="interactive")
        child = runtime.spawn(root.id, "first work", name="worker")
        await runtime.start()
        await runtime.wait(child.id, timeout=10)
        identity = child.kernel_id
        await runtime.shutdown()
        runtime = Runtime(directory, providers={"mock": provider})
        runtime.message(root.id, child.id, "increment x and reply")
        await runtime.start()
        async with asyncio.timeout(10):
            while not any(m["body"] == "124" for m in runtime.store.messages(root.id)):  # noqa: ASYNC110 - bounded independent state observation
                await asyncio.sleep(0.02)
        assert runtime.store.session(child.id).kernel_id == identity
        assert runtime.store.usage(root.id, tree=True).subagent_count == 1
    finally:
        await runtime.shutdown()
