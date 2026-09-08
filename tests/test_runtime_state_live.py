"""Opt-in production Python/subscription smoke; never part of automatic CI inference."""

import asyncio
import json
import os
import platform
from pathlib import Path

import pytest

from threadweave.models import RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext

pytestmark = pytest.mark.skipif(
    os.environ.get("BUFFALO_LIVE_STATE_SMOKE") != "1", reason="opt-in real subscription smoke"
)


async def test_live_persistent_python_restore_and_recursive_child(tmp_path):
    directory = Path(os.environ.get("BUFFALO_LIVE_STATE_OUTPUT", tmp_path))
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    workspace = directory / "workspace"
    workspace.mkdir()
    config = RunConfig(
        provider={
            "model": "gpt-6-astra",
            "parameters": {"reasoning_effort": "xhigh"},
            "max_output_tokens": 8192,
            "timeout_seconds": 180,
        },
        limits={"wall_seconds": 900, "token_budget": 500000, "max_subagents": 2},
    )
    config.refinement.automatic = False
    runtime = Runtime(directory / "state")
    report = {
        "platform": platform.platform(),
        "pid": os.getpid(),
        "proc_available": await asyncio.to_thread(Path("/proc").exists),
    }

    async def say(sid, body):
        before = runtime.store.session(sid).turns
        runtime.interact(sid, body)
        async with asyncio.timeout(240):
            while True:
                session = runtime.store.session(sid)
                assert session.outcome == "active", session.last_error
                if session.turns > before and not session.runnable and sid not in runtime.tasks:
                    return
                await asyncio.sleep(0.05)

    async def inspect(sid, code):
        event = runtime.store.event(sid, "live_smoke_inspection", {"code": code})
        result = await runtime.execute_python(ToolContext(runtime, sid, new_id(), event), code)
        assert not result.get("error"), result
        return result

    try:
        root = runtime.create(
            "Verify persistent runtime state with actual Python execution.",
            workspace,
            config=config,
            mode="interactive",
        )
        await runtime.start()
        await say(
            root.id,
            "Use ipython now to set persistent_probe = {'value': 937, 'label': 'cedar'}. Reply briefly after execution.",
        )
        first_pid = runtime.kernels[root.id].process.pid
        await say(
            root.id,
            "Without assigning persistent_probe again, read it in ipython, assert its value is 937 and label is cedar, then reply briefly.",
        )
        await inspect(root.id, "assert persistent_probe == {'value': 937, 'label': 'cedar'}")
        assert runtime.kernels[root.id].process.pid == first_pid
        report["persistent_python"] = "PASS"
        report["first_worker_pid"] = first_pid
        await runtime.shutdown()
        runtime = Runtime(directory / "state")
        await runtime.start()
        await say(
            root.id,
            "The runtime has been restored. Without assigning persistent_probe, use ipython to assert its value is 937 and label is cedar. Reply briefly.",
        )
        await inspect(root.id, "assert persistent_probe == {'value': 937, 'label': 'cedar'}")
        report["restored_worker_pid"] = runtime.kernels[root.id].process.pid
        assert report["restored_worker_pid"] != first_pid
        report["restore"] = "PASS"
        await say(
            root.id,
            "Use await rlm(..., name='state-child', purpose='shared') to create exactly one child. Its task: use its own ipython to assert persistent_probe is not in globals(), set child_probe = 271, and write child-proof.json in workspace containing the number 271. Wait for the child's completion and report. Keep its handle in Python. Use gpt-6-astra xhigh.",
        )
        children = [s for s in runtime.store.sessions(root_id=root.id) if s.parent_id]
        assert len(children) == 1
        child = await runtime.wait(children[0].id, timeout=240)
        assert child.outcome == "completed", child.last_error
        assert json.loads((workspace / "child-proof.json").read_text()) == 271
        await inspect(
            child.id, "assert child_probe == 271; assert 'persistent_probe' not in globals()"
        )
        report["child"] = "PASS"
        report["root_id"], report["child_id"] = root.id, child.id
        report["root_usage"] = runtime.store.usage(root.id).model_dump()
        report["child_usage"] = runtime.store.usage(child.id).model_dump()
        report["tree_usage"] = runtime.store.usage(root.id, tree=True).model_dump()
        assert report["child_usage"]["model_calls"] > 0
    finally:
        (directory / "smoke.json").write_text(json.dumps(report, indent=2))
        await runtime.shutdown()
