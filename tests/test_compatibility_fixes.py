"""Regression cases use test model doubles with production sessions/kernels/daemon APIs."""

import asyncio
import json

import pytest

from threadweave.chat import Chat
from threadweave.daemon import Daemon
from threadweave.models import ModelResponse, Outcome, Usage, new_id
from threadweave.runtime import Runtime
from threadweave.terminal import EventRenderer
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider
from .test_chat import InputTerminal


async def test_completed_child_followup_reuses_kernel_after_restart(tmp_path, python_config):
    provider = ScriptedProvider(
        {
            "worker": [
                response(
                    "ipython",
                    code="x = 123\nnote = harness.create_memory('Retained value', 'x is 123')\nawait agent_message.send('first result', receiver_role='parent')",
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
        await runtime.start()
        await eventually(
            lambda: (
                runtime.store.session(child.id).outcome == "completed"
                and child.id not in runtime.tasks
            )
        )
        assert child.id not in runtime.kernels
        before = runtime.store.usage(child.id)
        states = runtime.store.states(child.id)
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
        assert runtime.store.states(child.id) == states and len(states) == 1
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
        assert daemon.runtime.store.events(chat.session["id"], kind="refinement_trigger")
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


class EvidenceProvider:
    """Unit-test double for auxiliary requests only; never a live model result."""

    def __init__(self, *, invalid=False, gate=None):
        self.requests, self.invalid, self.gate = [], invalid, gate
        self.empty = False

    async def invoke(self, request, emit):
        assert request.metadata["purpose"] in {"refinement", "refinement_review"}
        self.requests.append(request)
        if self.gate:
            self.gate[0].set()
            await self.gate[1].wait()
        if self.invalid:
            return ModelResponse(text="not json")
        if request.metadata["purpose"] == "refinement_review":
            return ModelResponse(
                text=json.dumps({"shouldRefine": True, "rationale": "Observed computation"}),
                usage=Usage(input_tokens=40, output_tokens=20),
            )
        evidence = json.loads(request.messages[-1]["content"])["evidence"]
        if self.empty or not evidence:
            return ModelResponse(text='{"proposals": []}')
        return ModelResponse(
            text=json.dumps(
                {
                    "proposals": [
                        {
                            "kind": "memory",
                            "title": "Observed procedure",
                            "content": {"text": "Retain computed values"},
                            "source_events": [evidence[-1]["id"]],
                            "intended_effect": "Reuse demonstrated computation",
                        }
                    ]
                }
            ),
            usage=Usage(input_tokens=40, output_tokens=20),
        )


def seed_evidence(runtime, sid):
    return runtime.store.event(sid, "python_result", {"stdout": "Computed 123 successfully"})


async def settled_request(runtime, sid, rid):
    await eventually(
        lambda: (
            runtime.store.refinement_request(sid, rid)["status"] in {"applied", "skipped", "failed"}
        )
    )
    await eventually(lambda: sid not in runtime.tasks)
    return runtime.store.refinement_request(sid, rid)


async def test_idle_chat_refine_applies_once_and_recovers_versioned_state(tmp_path, python_config):
    python_config.refinement.automatic = False
    python_config.features.automatic_refinement = False
    daemon = Daemon(tmp_path / "data")
    runtime, provider = daemon.runtime, EvidenceProvider()
    runtime.providers["mock"] = provider
    terminal = InputTerminal(tmp_path / "ui")

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    chat = Chat(tmp_path / "data", tmp_path, terminal, rpc=rpc)
    try:
        await chat.open(config=python_config)
        sid = chat.session["id"]
        source = seed_evidence(runtime, sid)
        await chat.submit("/help")
        await chat.submit("/refine")
        assert "Refinement requested; no changes applied yet" in terminal.output.getvalue()
        rid = runtime.store.pending_refinement_requests(sid)[0]["id"]
        assert (await daemon.dispatch("refine", {"session_id": sid, "request_id": rid}))[
            "request_id"
        ] == rid
        await runtime.start()
        result = await settled_request(runtime, sid, rid)
        assert result["status"] == "applied" and result["applied_count"] == 1
        assert len(provider.requests) == 1
        assert runtime.store.session(sid).turns == 0
        assert not runtime.store.events(sid, kind="environment_prepared")
        entry = runtime.store.state(sid, result["entry_ids"][0])
        assert entry["version"] == 1 and source in entry["provenance"]["source_events"]
        renderer = EventRenderer(terminal, sid)
        for event in runtime.store.events(sid, kind="refinement_status"):
            await renderer.render(event)
        assert "Refinement applied: 1 versioned changes committed" in terminal.output.getvalue()
        assert (await daemon.dispatch("refinement_status", {"session_id": sid, "request_id": rid}))[
            "status"
        ] == "applied"
        await runtime.shutdown()
        runtime = Runtime(tmp_path / "data", providers={"mock": provider})
        await runtime.start()
        assert runtime.store.state(sid, entry["id"]) == entry
        assert (
            runtime.request_refinement(sid, source="human", request_id=rid)["status"] == "applied"
        )
        provider.empty = True  # The manual planner decides there is nothing more to apply.
        rid2 = runtime.request_refinement(sid, source="human")["request_id"]
        assert (await settled_request(runtime, sid, rid2))["status"] == "skipped"
        assert len(provider.requests) == 2
    finally:
        await runtime.shutdown()
        daemon.lock.close()


@pytest.mark.parametrize(
    "case", ["disabled", "empty", "invalid", "turn_limit", "model_limit", "stopped"]
)
async def test_refine_honest_terminal_outcomes_without_unwanted_turn(tmp_path, python_config, case):
    python_config.refinement.enabled = case != "disabled"
    python_config.features.automatic_refinement = False
    provider = EvidenceProvider(invalid=case == "invalid")
    runtime = Runtime(tmp_path / "data", providers={"mock": provider})
    try:
        session = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        sid = session.id
        if case != "empty":
            seed_evidence(runtime, sid)
        if case == "turn_limit":
            runtime.store.charge(sid, Usage(turns=python_config.limits.max_turns))
        if case == "model_limit":
            runtime.store.charge(sid, Usage(model_calls=python_config.limits.max_model_calls))
        rid = runtime.interact(sid, "/refine")
        if case == "stopped":
            await runtime.stop(sid)
        await runtime.start()
        result = await settled_request(runtime, sid, rid)
        assert result["status"] == ("skipped" if case in {"empty", "disabled"} else "failed")
        assert result["reason"]
        assert len(provider.requests) == (1 if case in {"invalid", "empty"} else 0)
        assert not runtime.store.states(sid)
        assert not runtime.store.messages(sid, pending=True)
        assert runtime.store.session(sid).turns == 0
    finally:
        await runtime.shutdown()


async def test_refine_during_model_turn_waits_for_boundary(tmp_path, python_config):
    entered, release = asyncio.Event(), asyncio.Event()
    refiner = EvidenceProvider()

    class Provider:
        async def invoke(self, request, emit):
            if request.metadata.get("purpose") in {"refinement", "refinement_review"}:
                assert release.is_set()
                return await refiner.invoke(request, emit)
            entered.set()
            await release.wait()
            return ModelResponse(text="Ready")

    runtime = Runtime(tmp_path / "data", providers={"mock": Provider()})
    try:
        session = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        seed_evidence(runtime, session.id)
        runtime.interact(session.id, "work")
        await runtime.start()
        await asyncio.wait_for(entered.wait(), 5)
        result = runtime.request_refinement(session.id, source="human")
        assert result["status"] == "requested" and not refiner.requests
        release.set()
        assert (await settled_request(runtime, session.id, result["request_id"]))[
            "status"
        ] == "applied"
        assert runtime.store.session(session.id).turns == 1
    finally:
        release.set()
        await runtime.shutdown()


async def test_python_refine_paused_request_survives_restart(tmp_path, python_config):
    provider = EvidenceProvider()

    class ResumableProvider:
        async def invoke(self, request, emit):
            if request.metadata.get("purpose") in {"refinement", "refinement_review"}:
                return await provider.invoke(request, emit)
            return ModelResponse(text="Resumed")

    data = tmp_path / "data"
    runtime = Runtime(data, providers={"mock": ResumableProvider()})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        seed_evidence(runtime, root.id)
        await python(
            runtime, root.id, "request = await refine()\nassert request['status'] == 'requested'"
        )
        await runtime.pause(root.id)
        rid = runtime.store.pending_refinement_requests(root.id)[0]["id"]
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": ResumableProvider()})
        await runtime.start()
        assert runtime.store.session(root.id).paused
        assert runtime.store.refinement_request(root.id, rid)["status"] == "requested"
        runtime.resume(root.id)
        assert (await settled_request(runtime, root.id, rid))["status"] == "applied"
        assert len(provider.requests) == 1
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("interruption", ["cancel", "crash"])
async def test_interrupted_refinement_is_not_replayed(tmp_path, python_config, interruption):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = EvidenceProvider(gate=(entered, release))
    data = tmp_path / "data"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        seed_evidence(runtime, root.id)
        rid = runtime.request_refinement(root.id, source="human")["request_id"]
        if interruption == "crash":
            runtime.store.db.execute(
                "UPDATE refinement_requests SET status='running' WHERE id=?", (rid,)
            )
        else:
            await runtime.start()
            await asyncio.wait_for(entered.wait(), 5)
            await runtime.pause(root.id)
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        await runtime.start()
        result = runtime.store.refinement_request(root.id, rid)
        assert result["status"] == "failed" and result["uncertain"]
        assert not runtime.store.states(root.id)
        assert len(provider.requests) == (1 if interruption == "cancel" else 0)
    finally:
        release.set()
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


@pytest.mark.parametrize("json_mode", [False, True])
async def test_chat_disabled_refinement_and_direct_state_edit_are_distinct(
    tmp_path, python_config, json_mode
):
    python_config.refinement.enabled = False
    daemon = Daemon(tmp_path / "data")
    daemon.runtime.providers["mock"] = EvidenceProvider()
    terminal = InputTerminal(tmp_path / "ui", json_mode=json_mode)

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    chat = Chat(tmp_path / "data", tmp_path, terminal, rpc=rpc)
    try:
        await chat.open(config=python_config)
        await chat.submit("/refine")
        output = terminal.output.getvalue()
        assert "disabled" in output and "skipped" in output
        assert "Unknown command" not in output
        if json_mode:
            assert '"command": "refine"' in output
        assert not daemon.runtime.store.pending_refinement_requests(chat.session["id"])
        assert not daemon.runtime.providers["mock"].requests
        enabled = python_config.model_copy(deep=True)
        enabled.refinement.enabled = True
        root = daemon.runtime.create("editable", tmp_path, config=enabled, mode="interactive")
        source = seed_evidence(daemon.runtime, root.id)
        result = await daemon.dispatch(
            "refine",
            {
                "session_id": root.id,
                "edit": {
                    "kind": "memory",
                    "title": "Operator authored",
                    "content": {"text": "Observed value"},
                    "source_events": [source],
                    "intended_effect": "Retain the observation",
                },
            },
        )
        assert "refinement_id" in result and "request_id" not in result
        assert not daemon.runtime.store.pending_refinement_requests(root.id)
        assert not daemon.runtime.store.states(root.id)  # StateEdit is queued, not applied yet.
        daemon.runtime.store.apply_refinements(root.id)
        assert daemon.runtime.store.states(root.id)[0]["version"] == 1
        assert not daemon.runtime.providers["mock"].requests
    finally:
        await daemon.runtime.shutdown()
        daemon.lock.close()


async def test_new_refinement_request_during_pass_is_not_lost(tmp_path, python_config):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = EvidenceProvider(gate=(entered, release))
    runtime = Runtime(tmp_path / "data", providers={"mock": provider})
    try:
        root = runtime.create("root", tmp_path, config=python_config, mode="interactive")
        seed_evidence(runtime, root.id)
        rid1 = runtime.request_refinement(root.id, source="human")["request_id"]
        await runtime.start()
        await asyncio.wait_for(entered.wait(), 5)
        seed_evidence(runtime, root.id)
        rid2 = runtime.request_refinement(root.id, source="human")["request_id"]
        assert (
            runtime.request_refinement(root.id, source="human", request_id=rid2)["status"]
            == "requested"
        )
        release.set()
        assert (await settled_request(runtime, root.id, rid1))["status"] == "applied"
        assert (await settled_request(runtime, root.id, rid2))["status"] == "applied"
        assert len(provider.requests) == 2
        assert len(runtime.store.states(root.id)) == 2
    finally:
        release.set()
        await runtime.shutdown()
