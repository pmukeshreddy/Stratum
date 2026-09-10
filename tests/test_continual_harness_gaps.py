"""Focused ports of Prime runtime/test_harness.py and serialized-refine tests."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from threadweave.host_api import harness as dispatch_harness
from threadweave.kernel_api import Harness, Recursive
from threadweave.models import ModelResponse
from threadweave.runtime import Runtime
from threadweave.storage import Store

from .conftest import eventually, response
from .test_continual_harness import REF, edit, proposal
from .test_continual_harness import harness_runtime as harness_runtime


def kernel_harness(runtime, sid):
    class Host:
        def call(self, operation, **payload):
            return dispatch_harness(
                SimpleNamespace(runtime=runtime, session_id=sid), operation.split(".")[1], payload
            )

    return Harness(Host())


async def test_local_history_is_session_history_global_is_jsonl_and_restart_merges(harness_runtime):
    rt, sid, _ = harness_runtime
    local = rt.store.harness.apply(sid, proposal(edit()), id="local_pass")
    global_result = rt.store.harness.apply(sid, proposal(edit()), id="global_pass", global_=True)
    assert rt.store.refinement_history(sid) == [local, global_result]
    assert not (rt.store.harness.path(sid) / "refinements.jsonl").exists()
    global_path = rt.store.harness.path() / "refinements.jsonl"
    assert [json.loads(line) for line in global_path.read_text().splitlines()] == [global_result]
    assert {r["id"] for r in rt.store.harness.history(sid)} == {"local_pass", "global_pass"}
    other = Store(rt.store.directory)
    try:
        assert other.refinement_history(sid) == [local, global_result]
        assert other.harness.history(sid) == rt.store.harness.history(sid)
        # Session records take precedence when an ID also occurs in the global log.
        other.record_harness_refinement(sid, {**global_result, "summary": "session wins"})
        assert (
            next(r for r in other.harness.history(sid) if r["id"] == "global_pass")["summary"]
            == "session wins"
        )
    finally:
        other.close()
    for review in (False, True):
        history = rt.refinement_input(sid, review=review)["refinement_history"]
        assert "[local_pass]" in history and "[global_pass]" in history


@pytest.mark.parametrize("kind", ["memory", "prompt", "skill", "subagent"])
@pytest.mark.parametrize("global_", [False, True])
async def test_writable_kernel_crud_all_kinds_scopes_and_versions(harness_runtime, kind, global_):
    rt, sid, _ = harness_runtime
    h = kernel_harness(rt, sid)
    helper = "prompt_note" if kind == "prompt" else kind
    options = (
        {"reference": REF, "arguments": {"text": {"type": "string"}}} if kind == "skill" else {}
    )
    entry = getattr(h, "create_" + helper)(
        "Saved lesson", "v1", id="entry", global_=global_, metadata={"kept": True}, **options
    )
    assert entry.kind == kind and entry.version == 1 and entry.source == "agent"
    assert entry.scope == ("global" if global_ else "local")
    assert entry.path == ("policy" if kind == "prompt" else "general")
    updated = getattr(h, "update_" + helper)("entry", "Changed", "v2", global_=global_)
    assert (
        updated.version == 2 and updated.id == entry.id and updated.created_at == entry.created_at
    )
    assert updated.metadata == {"kept": True} and updated.path == entry.path
    if kind == "skill":
        assert updated.reference == entry.reference and updated.arguments == entry.arguments
    assert h.get(kind, "entry", global_=global_) == updated
    assert h.list(kind, global_=global_) == [updated]
    assert "Changed" in h.overview(global_=global_)
    assert h.snapshot(global_=global_)["entries"][kind]["entry"]["version"] == 2
    with pytest.raises(ValueError, match="already exists"):
        getattr(h, "create_" + helper)("Again", "v3", id="entry", global_=global_, **options)
    assert getattr(h, "delete_" + helper)("entry", global_=global_) is True
    assert getattr(h, "delete_" + helper)("entry", global_=global_) is False
    assert h.get(kind, "entry", global_=global_) is None
    with pytest.raises(ValueError, match="does not exist"):
        getattr(h, "update_" + helper)("entry", "Missing", "v3", global_=global_)
    # Prime's CRUD does not invent a planner pass/audit. record_refinement is explicit.
    assert rt.store.harness.history(sid) == []
    assert not rt.store.harness.load(None if global_ else sid)["refinements"]


async def test_direct_scope_prefixes_overrides_and_exact_prompt_aliases(harness_runtime):
    rt, sid, _ = harness_runtime
    h = kernel_harness(rt, sid)
    global_entry = h.create_memory("Shared", "global", id="global:shared")
    assert global_entry.id == "shared" and global_entry.scope == "global"
    assert h.get("memory", "shared") is None
    with pytest.raises(ValueError, match="does not exist"):
        h.update_memory("shared", "Local", "wrong target")
    assert h.delete_memory("shared") is False
    h.create_memory("Shared", "local override", id="local:shared")
    assert h.get("memory", "shared").content == "local override"
    assert h.get("memory", "global:shared").content == "global"
    h.update_memory("global:shared", "Shared", "explicit global update")
    assert h.get("memory", "shared", global_=True).content == "explicit global update"
    h.create_prompt_note("Narrow instruction", "Always inspect inputs", id="prompt")
    h.update_prompt_note("prompt", "Narrow instruction", "Inspect inputs first")
    assert h.get("prompt", "prompt").version == 2
    assert h.delete_prompt_note("prompt")
    assert h.create_memory("Keyword global", "global", **{"global": True}).scope == "global"
    with pytest.raises(TypeError, match="global must be a bool"):
        h.create_memory("Invalid", "flag", **{"global": "false"})
    with pytest.raises(TypeError, match="unexpected keyword"):
        h.create_memory("Invalid", "field", invented=True)
    recursive = Recursive(h.host)
    assert recursive.get_harness_state() is recursive.harness
    assert recursive.get_harness_state(global_=True).get("memory", "shared").scope == "global"


async def test_direct_writes_reload_external_changes_and_record_explicit_event(harness_runtime):
    rt, sid, _ = harness_runtime
    h, other = kernel_harness(rt, sid), kernel_harness(rt, sid)
    entry = h.create_skill("Parse", "Parse text", id="parse", reference=REF)
    assert entry.arguments == {}
    other.update_skill("parse", "Parser", "New contract", arguments={"text": {"required": True}})
    h.update_skill("parse", "Parser", "New description")
    entry = h.get("skill", "parse")
    assert entry.version == 3 and entry.arguments == {"text": {"required": True}}
    h.update_skill("parse", "Parser", "No external inputs", arguments={})
    assert h.get("skill", "parse").arguments == {}
    with pytest.raises(ValueError, match="reference.type"):
        h.create_skill("No inline bodies", "procedure", reference={"type": "code"})
    other.create_memory("Externally created", "content", id="external")
    with pytest.raises(ValueError, match="already exists"):
        h.create_memory("Clobber", "wrong", id="external")
    event = h.record_refinement(
        "corrected procedure", "updated parse", evidence="observed", outcome="verified"
    )
    assert event.id == "refine_0001" and event.changes == ["updated parse"]
    assert "refinements: 1" in h.overview()
    assert h.snapshot()["refinements"][0]["id"] == event.id
    assert not (rt.store.harness.path(sid) / "refinements.jsonl").exists()


def active_turn(rt, sid):
    rt.store.update(
        sid,
        pending_turn={
            "event_id": "response-1",
            "context_committed": False,
            "response": {"text": "Reusable observed lesson", "actions": []},
        },
    )


async def test_ready_plan_waits_for_tools_then_single_consumer_applies(harness_runtime):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    receipt = rt.request_refinement(sid, instructions="save the observed lesson")
    assert receipt["scheduled"] and rt.refinement_status(sid) == {
        "pending": False,
        "in_flight": True,
    }
    await rt.refinement_state(sid).background
    assert (
        len(provider.requests) == 1
        and "Reusable observed lesson" in provider.requests[0].messages[-1]["content"]
    )
    assert not rt.store.harness.entries(sid)
    assert not await rt.refinement_checkpoint(sid)
    pending = rt.store.session(sid).pending_turn
    pending["context_committed"] = True
    rt.store.update(sid, pending_turn=pending)
    results = await asyncio.gather(rt.refinement_checkpoint(sid), rt.refinement_checkpoint(sid))
    assert sum(results) == 1 and len(provider.requests) == 1
    assert len(rt.store.refinement_history(sid)) == 1
    assert len(rt.store.events(sid, kind="refinement_notice")) == 1


async def test_boundary_waits_for_pending_plan_and_blocks_duplicate_consumers(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await provider.entered.wait()
    rt.store.update(sid, pending_turn={"context_committed": True})
    first = asyncio.create_task(rt.refinement_checkpoint(sid))
    second = asyncio.create_task(rt.refinement_checkpoint(sid))
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    assert not rt.store.harness.entries(sid)
    provider.release.set()
    assert sum(await asyncio.gather(first, second)) == 1
    assert len(provider.requests) == 1


async def test_background_baseline_rejects_same_entry_kernel_write(harness_runtime):
    rt, sid, provider = harness_runtime
    h = kernel_harness(rt, sid)
    h.create_memory("Shared", "baseline", id="lesson")
    provider.proposal = proposal(edit(action="update", content="stale planner content"))
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await provider.entered.wait()
    h.update_memory("lesson", "Shared", "new kernel content")
    h.create_memory("Unrelated", "kept too", id="unrelated")
    provider.release.set()
    rt.store.update(sid, pending_turn={"context_committed": True})
    assert not await rt.refinement_checkpoint(sid)
    assert h.get("memory", "lesson").content == "new kernel content"
    assert h.get("memory", "lesson").version == 2
    assert h.get("memory", "unrelated")
    assert (
        "changed during refinement planning"
        in rt.store.refinement_history(sid)[-1]["appliedEdits"][0]["error"]
    )
    assert not rt.store.events(sid, kind="refinement_notice")


@pytest.mark.parametrize("decision", [False, True])
async def test_interval_background_review_gate_and_one_counter_increment(harness_runtime, decision):
    rt, sid, provider = harness_runtime
    provider.decision = decision
    active_turn(rt, sid)
    rt.refinement_state(sid).turns_since_review = 24
    rt.refinement_message_end(sid)
    assert rt.refinement_state(sid).turns_since_review == 25
    await rt.refinement_state(sid).background
    pending = rt.store.session(sid).pending_turn
    pending["context_committed"] = True
    rt.store.update(sid, pending_turn=pending)
    assert await rt.refinement_checkpoint(sid, completed_turn=True) == decision
    assert rt.refinement_state(sid).turns_since_review == 0
    assert len(provider.requests) == (2 if decision else 1)
    assert not rt.has_pending_refinement(sid)


@pytest.mark.parametrize("explicit", [False, True])
async def test_background_failure_retries_only_explicit_at_boundary(harness_runtime, explicit):
    rt, sid, provider = harness_runtime
    original = rt.plan_refinement
    calls = []

    async def plan(session, options):
        calls.append(options)
        if len(calls) == 1:
            raise ValueError("planning failed")
        return await original(session, options)

    rt.plan_refinement = plan
    active_turn(rt, sid)
    if explicit:
        rt.request_refinement(sid, instructions="specific lesson")
    else:
        rt.refinement_state(sid).turns_since_review = 25
        rt.maybe_start_refinement_plan(sid)
    await rt.refinement_state(sid).background
    rt.store.update(sid, pending_turn={"context_committed": True})
    assert await rt.refinement_checkpoint(sid) == explicit
    assert len(calls) == (2 if explicit else 1)
    assert rt.refinement_state(sid).last_review_at > 0
    assert not rt.refinement_status(sid)["in_flight"]
    assert not await rt.refinement_checkpoint(sid)


async def test_branch_cancellation_does_not_apply_or_requeue_old_explicit_work(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await provider.entered.wait()
    rt.invalidate_refinement(sid)
    provider.release.set()
    await rt.refinement_state(sid).background
    rt.store.update(sid, pending_turn={"context_committed": True})
    assert not await rt.refinement_checkpoint(sid)
    assert rt.store.refinement_history(sid) == []
    assert rt.refinement_status(sid) == {"pending": False, "in_flight": False}
    rt.request_refinement(sid, instructions="new branch")
    assert await rt.refinement_checkpoint(sid)
    assert len(rt.store.refinement_history(sid)) == 1


async def test_new_explicit_request_supersedes_plan_that_ignores_abort(harness_runtime):
    rt, sid, provider = harness_runtime
    entered, release = asyncio.Event(), asyncio.Event()
    original = rt.plan_refinement
    calls = []

    async def plan(session, options):
        calls.append(options)
        if len(calls) == 1:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
        return await original(session, options)

    rt.plan_refinement = plan
    active_turn(rt, sid)
    rt.refinement_state(sid).turns_since_review = 25
    rt.maybe_start_refinement_plan(sid)
    await entered.wait()
    rt.request_refinement(sid, instructions="new explicit request")
    rt.store.update(sid, pending_turn={"context_committed": True})
    checkpoint = asyncio.create_task(rt.refinement_checkpoint(sid))
    await asyncio.sleep(0)
    assert len(calls) == 1 and not checkpoint.done()
    release.set()
    assert await checkpoint
    assert len(calls) == 2 and calls[-1]["instructions"] == "new explicit request"
    assert len(rt.store.refinement_history(sid)) == 1


@pytest.mark.parametrize("explicit", [True, False])
async def test_real_turn_plans_while_tool_waits_and_next_model_observes_notice(
    tmp_path, python_config, explicit
):
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    active = peak = 0

    class Provider:
        async def invoke(self, request, emit):
            nonlocal active, peak
            purpose = request.metadata.get("purpose", "agent")
            calls.append(purpose)
            active += 1
            peak = max(peak, active)
            try:
                if purpose == "refinement_review":
                    return ModelResponse(
                        text=json.dumps({"shouldRefine": True, "rationale": "useful"})
                    )
                if purpose == "refinement":
                    started.set()
                    await release.wait()
                    return ModelResponse(text=json.dumps(proposal(edit())))
                if request.turn == 0:
                    assert "# Continual Harness State" in str(request.messages)
                    code = (
                        "receipt = await refine.run('retain lesson')\nassert receipt['scheduled']\n"
                        if explicit
                        else ""
                    )
                    return response(
                        "ipython",
                        code=code
                        + "workspace.joinpath('tool-waiting').touch()\nwhile not workspace.joinpath('release-tool').exists():\n    await asyncio.sleep(0.01)",
                    )
                assert "Run checks in the project environment" not in request.messages[0]["content"]
                assert ("[self-refinement]" if explicit else "[auto-refinement]") in str(
                    request.messages
                )
                return ModelResponse(text="The learned project environment is now available.")
            finally:
                active -= 1

    if not explicit:
        python_config.refinement.turn_interval = 1
    rt = Runtime(tmp_path / "state", providers={"mock": Provider()})
    root = rt.create("Learn a reusable project lesson", tmp_path, config=python_config)
    turn = asyncio.create_task(rt._run_turn(root.id))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await eventually(lambda: (tmp_path / "tool-waiting").exists())
        assert not rt.store.harness.entries(root.id)
        release.set()
        await rt.refinement_state(root.id).background
        assert not turn.done() and not rt.store.harness.entries(root.id)
        assert calls.count("agent") == 1
        (tmp_path / "release-tool").touch()
        await asyncio.wait_for(turn, 5)
        assert rt.store.session(root.id).runnable
        assert len(rt.store.refinement_history(root.id)) == 1
        await rt._run_turn(root.id)
        assert calls.count("agent") == 2 and calls.count("refinement") == 1 and peak == 1
        assert not rt.store.events(root.id, kind="python_error")
    finally:
        release.set()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
        await rt.shutdown()


async def test_pause_aborts_a_settled_background_plan(harness_runtime):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await rt.refinement_state(sid).background
    rt.store.update(sid, pending_turn={"context_committed": True})
    await rt.pause(sid)
    assert not await rt.refinement_checkpoint(sid)
    assert not rt.store.refinement_history(sid)
    rt.store.update(sid, paused=False)
    assert not await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == 1


async def test_abort_cancels_unapplied_background_work_without_restart_replay(tmp_path, config):
    from .test_continual_harness import Refiner

    provider = Refiner()
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    rt = Runtime(tmp_path / "state", providers={"mock": provider})
    root = rt.create("Interrupted refinement", tmp_path, config=config)
    active_turn(rt, root.id)
    rt.request_refinement(root.id)
    await asyncio.wait_for(provider.entered.wait(), 5)
    rt.invalidate_refinement(root.id)
    provider.release.set()
    await rt.shutdown()
    provider.entered = provider.release = None
    resumed = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        assert resumed.refinement_status(root.id) == {"pending": False, "in_flight": False}
        assert not resumed.store.harness.entries(root.id)
        assert not resumed.store.refinement_history(root.id)
        assert not (resumed.store.harness.path(root.id) / "refinements.jsonl").exists()
        resumed.store.update(root.id, pending_turn=None)
        resumed.resume(root.id)
        assert not await resumed.refinement_checkpoint(root.id)
        assert (await resumed.refine(root.id))["appliedEdits"][0]["applied"]
        assert len(resumed.store.refinement_history(root.id)) == 1
    finally:
        await resumed.shutdown()
