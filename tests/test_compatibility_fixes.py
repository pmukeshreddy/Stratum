"""Regression cases use test model doubles with production sessions/kernels/daemon APIs."""

import asyncio

import pytest

from threadweave.chat import Chat
from threadweave.daemon import Daemon
from threadweave.models import ModelResponse, Outcome, Usage, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider
from .test_chat import InputTerminal


async def test_completed_child_followup_reuses_kernel_after_restart(tmp_path, python_config):
    from .test_continual_harness import edit, proposal

    provider = ScriptedProvider(
        {
            "worker": [
                response(
                    "ipython",
                    code="x = 123\nawait agent_message.send('first result', receiver_role='parent')",
                ),
                ModelResponse(text="Initial work done"),
                response(
                    "ipython",
                    code="assert x == 123\nx += 1\nawait agent_message.send('second result: ' + str(x), receiver_role='parent')",
                ),
                ModelResponse(text="Follow-up done"),
            ],
            "root": [ModelResponse(text="Waiting")] * 10,
        }
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        child = runtime.spawn(root.id, "initial work", name="worker")
        runtime.store.harness.apply(root.id, proposal(edit()), id="refine_fixture", global_=True)
        await runtime.start()
        await eventually(
            lambda: (
                runtime.store.session(child.id).outcome == "completed"
                and child.id not in runtime.tasks
            )
        )
        assert child.id not in runtime.kernels
        before = runtime.store.usage(child.id)
        states = runtime.store.harness.entries(child.id)
        history_ids = {e["id"] for e in runtime.store.events(child.id, limit=1000)}
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        mid = runtime.message(root.id, child.id, "increment the existing x and reply")
        assert runtime.store.session(child.id).outcome == "active"
        await runtime.start()
        await eventually(
            lambda: runtime.store.session(child.id).turns == 4 and child.id not in runtime.tasks
        )
        restored = runtime.store.session(child.id)
        assert restored.outcome == "completed" and restored.kernel_id == child.kernel_id
        assert restored.workspace == child.workspace
        assert runtime.store.harness.entries(child.id) == states and len(states) == 1
        assert history_ids <= {e["id"] for e in runtime.store.events(child.id, limit=1000)}
        assert runtime.store.usage(root.id, tree=True).subagent_count == 1
        assert next(m for m in runtime.store.messages(child.id) if m["id"] == mid)["received_at"]
        assert any(m["body"] == "second result: 124" for m in runtime.store.messages(root.id))
        assert runtime.store.usage(child.id).model_calls == before.model_calls + 2
    finally:
        await runtime.shutdown()


async def test_chat_refine_is_requested_instead_of_unknown(tmp_path, python_config):
    daemon = Daemon(tmp_path / "state")
    terminal = InputTerminal(tmp_path / "ui")

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    daemon.runtime.providers["mock"] = ScriptedProvider({})
    chat = Chat(tmp_path / "state", tmp_path, terminal, rpc=rpc)
    try:
        await chat.open(config=python_config)
        assert await chat.submit("/refine")
        output = terminal.output.getvalue()
        assert "Unknown command" not in output
        assert "Refinement requested" in output
        assert daemon.runtime.store.events(chat.session["id"], kind="refine_scheduled")
    finally:
        await daemon.runtime.shutdown()
        daemon.lock.close()


async def python(runtime, sid, code):
    event = runtime.store.event(sid, "python_execution", {"code": code})
    result = await runtime.execute_python(ToolContext(runtime, sid, new_id(), event), code)
    assert not result.get("error"), result
    return result


async def test_python_handle_followup_returns_two_explicit_results(tmp_path, python_config):
    parent_code = """
import asyncio
worker = await rlm('retain value and reply', name='worker')
async def wait_done():
    async with asyncio.timeout(10):
        while (await rlm.list_subagents())[0]['outcome'] != 'completed':
            await asyncio.sleep(0.1)
await wait_done()
received = await agent_message.receive()
assert any(m['body'] == 'first: 7' for m in received)
receipt = await agent_message.send('increment and reply', receiver_role='child', receiver_name=worker.name)
assert receipt['receipts'][0]['recipient_id'] == worker.session_id
await wait_done()
received_again = await agent_message.receive()
assert any(m['body'] == 'second: 8' for m in received_again)
assert worker.session_id == (await rlm.list_subagents())[0]['id']
"""
    provider = ScriptedProvider(
        {
            "root": [response("ipython", code=parent_code), ModelResponse(text="Both received")],
            "worker": [
                response(
                    "ipython",
                    code="value = 7\nawait agent_message.send('first: 7', receiver_role='parent')",
                ),
                ModelResponse(text="done"),
                response(
                    "ipython",
                    code="value += 1\nawait agent_message.send('second: ' + str(value), receiver_role='parent')",
                ),
                ModelResponse(text="done again"),
            ],
        }
    )
    runtime = Runtime(tmp_path / "data", providers={"mock": provider})
    try:
        root = runtime.create("parent", tmp_path, config=python_config, mode="interactive")
        runtime.interact(root.id, "delegate then follow up")
        await runtime.start()
        await eventually(
            lambda: runtime.store.session(root.id).turns >= 2 and not runtime.tasks,
            seconds=15,
        )
        child = runtime.store.sessions(root_id=root.id)[1]
        assert not runtime.store.events(root.id, kind="python_error")
        assert len(runtime.store.sessions(root_id=root.id)) == 2
        assert runtime.store.usage(child.id).model_calls == 4
        assert all(
            [t["function"]["name"] for t in r.tools] == ["ipython"] for r in provider.requests
        )
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("boundary", ["model", "cleanup", "recovery"])
async def test_followup_completion_boundary_no_stranding_or_duplicate_turns(
    tmp_path, python_config, boundary
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def completing(request):
        if boundary == "model":
            entered.set()
            await release.wait()
        return ModelResponse(text="initial done")

    provider = ScriptedProvider({"worker": [completing, ModelResponse(text="follow-up done")]})
    data = tmp_path / "data"
    runtime = Runtime(data, providers={"mock": provider})
    root = runtime.create("parent", tmp_path, config=python_config, mode="interactive")
    await runtime.pause(root.id)
    child = runtime.spawn(root.id, "initial", name="worker")
    close = runtime._close_kernel

    async def blocked_close(sid):
        if sid == child.id and runtime.store.session(sid).turns == 1:
            entered.set()
            await release.wait()
        await close(sid)

    if boundary == "cleanup":
        runtime._close_kernel = blocked_close
    try:
        await runtime.start()
        if boundary == "recovery":
            await eventually(
                lambda: (
                    runtime.store.session(child.id).outcome == "completed"
                    and child.id not in runtime.tasks
                )
            )
            # Simulate a durable message admitted before a process died, without
            # the normal Runtime.message continuation step.
            mids = [runtime.store.send(root.id, child.id, f"follow-up {i}") for i in range(2)]
            await runtime.shutdown()
            runtime = Runtime(data, providers={"mock": provider})
            await runtime.start()
        else:
            await asyncio.wait_for(entered.wait(), 5)
            old_task = runtime.tasks[child.id]
            mids = [runtime.message(root.id, child.id, f"follow-up {i}") for i in range(2)]
            assert runtime.tasks[child.id] is old_task
            assert len(provider.requests) == 1
            release.set()
        await eventually(
            lambda: runtime.store.session(child.id).turns == 2 and child.id not in runtime.tasks
        )
        assert runtime.store.session(child.id).outcome == "completed"
        assert provider.peak_active == 1  # serial within this one session
        assert len(provider.requests) == 2
        events = runtime.store.events(child.id, kind="agent_message_received", limit=100)
        assert sorted(e["payload"]["id"] for e in events) == sorted(mids)
        assert len(runtime.store.events(child.id, kind="subagent_continued")) == 1
    finally:
        release.set()
        await runtime.shutdown()


@pytest.mark.parametrize(
    "blocked",
    [
        "cancelled",
        "failed",
        "limited",
        "paused",
        "root_cancelled",
        "root_failed",
        "root_limited",
        "turns",
        "tokens",
        "calls",
        "wall",
        "goal",
        "ancestor_goal",
    ],
)
async def test_followup_cannot_revive_ineligible_or_reset_budget(tmp_path, python_config, blocked):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        child = runtime.spawn(root.id, "child")
        runtime.store.charge(child.id, Usage(model_calls=2, turns=2, input_tokens=25))
        runtime.store.finish(child.id, Outcome.COMPLETED)
        if blocked in {"cancelled", "failed", "limited"}:
            runtime.store.finish(child.id, Outcome(blocked))
        elif blocked == "paused":
            runtime.store.update(child.id, paused=True)
        elif blocked.startswith("root_"):
            runtime.store.finish(root.id, Outcome(blocked.removeprefix("root_")))
        elif blocked in {"goal", "ancestor_goal"}:
            sid = child.id if blocked == "goal" else root.id
            runtime.store.db.execute(
                "INSERT INTO goals VALUES(?,?,?,?,?)", (sid, "goal", "completed", 0, 0)
            )
            runtime.store.db.execute("INSERT INTO goal_budgets VALUES(?,?,?)", (sid, 25, 0))
        elif blocked == "turns":
            runtime.store.charge(root.id, Usage(turns=python_config.limits.max_turns))
        elif blocked == "tokens":
            runtime.store.charge(root.id, Usage(input_tokens=python_config.limits.token_budget))
        elif blocked == "calls":
            runtime.store.charge(root.id, Usage(model_calls=python_config.limits.max_model_calls))
        elif blocked == "wall":
            runtime.store.charge(root.id, Usage(wall_seconds=python_config.limits.wall_seconds))
        before, usage = runtime.store.session(child.id), runtime.store.usage(root.id, tree=True)
        runtime.message(root.id, child.id, "more work")
        after = runtime.store.session(child.id)
        assert after.outcome == before.outcome and not after.runnable
        assert after.kernel_id == before.kernel_id and after.paused == before.paused
        assert runtime.store.usage(root.id, tree=True) == usage
        assert runtime.store.messages(child.id, pending=True)
        assert not runtime.store.events(child.id, kind="subagent_continued")
    finally:
        await runtime.shutdown()


async def test_child_result_does_not_reactivate_completed_parent(tmp_path, python_config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        child = runtime.spawn(root.id, "child")
        grandchild = runtime.spawn(child.id, "grandchild")
        runtime.store.finish(child.id, Outcome.COMPLETED)
        runtime.message(grandchild.id, child.id, "Child session completed; result ready")
        assert runtime.store.session(child.id).outcome == "completed"
        assert not runtime.store.session(child.id).runnable
        assert runtime.store.messages(child.id, pending=True)
    finally:
        await runtime.shutdown()
