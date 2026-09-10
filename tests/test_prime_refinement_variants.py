"""Prime lifecycle variants: assertions follow source, not suggestive test titles."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from threadweave.context import python_instructions
from threadweave.daemon import Daemon
from threadweave.harness import format_harness_state, load_harness_state, save_harness_state
from threadweave.kernel_api import bootstrap
from threadweave.models import RefinementPolicy
from threadweave.refinement_context import convert_to_llm

from .test_continual_harness import edit, input_sections, proposal
from .test_continual_harness import harness_runtime as harness_runtime
from .test_continual_harness_gaps import active_turn


@pytest.mark.parametrize("scope", [None, False, True])
async def test_daemon_preserves_scope_omission(harness_runtime, scope):
    rt, sid, _ = harness_runtime
    daemon = object.__new__(Daemon)
    daemon.runtime = rt
    rt.refine = AsyncMock(return_value={})
    supplied = {} if scope is None else {"global_": scope}
    await daemon.dispatch("refine", {"session_id": sid, **supplied})
    assert rt.refine.call_args.kwargs == supplied


@pytest.mark.parametrize("serialized", [True, False])
@pytest.mark.parametrize("requested", [True, False])
async def test_pending_explicit_consumption_suppresses_same_boundary_auto(
    harness_runtime, serialized, requested
):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    )
    state = rt.refinement_state(sid)
    if requested:
        state.pending_request = {"source": "self", "instructions": "explicit"}
        state.turns_since_review = 25
    await rt.refinement_checkpoint(sid)
    if state.task:
        await state.task
    if state.auto_task:
        await state.auto_task
    assert [r.metadata["purpose"] for r in provider.requests] == (
        ["refinement"] if requested else []
    )
    assert state.pending_request is None


@pytest.mark.parametrize("pending", [True, False])
@pytest.mark.parametrize("failure", ["review", "plan"])
async def test_interactive_failure_cooldown_and_retained_approval(
    harness_runtime, pending, failure
):
    rt, sid, _ = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    decision = {"shouldRefine": True, "rationale": "durable lesson", "instructions": "retain"}
    if pending:
        state.pending_review = ("turn_interval", decision)
    rt.review_refinement = AsyncMock(
        return_value=decision,
        side_effect=ValueError("review failed") if failure == "review" else None,
    )

    async def fail(*args, **kwargs):
        assert state.reviewing
        raise ValueError("plan failed")

    rt.refine = AsyncMock(side_effect=fail)
    await rt.maybe_auto_refine(sid)
    assert state.last_review_at > 0 and state.turns_since_review == 25
    assert not state.reviewing
    calls = (rt.refine.await_count, rt.review_refinement.await_count)
    await rt.maybe_auto_refine(sid)
    assert calls == (rt.refine.await_count, rt.review_refinement.await_count)
    assert bool(state.pending_review) == pending
    state.turns_since_review = 0
    state.pending_review = None


async def test_compact_decline_keeps_concurrent_interval_review(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(
        sid,
        rt.store.config(sid).model_copy(
            update={"serialized_refine": False, "refinement": RefinementPolicy(cooldown_seconds=0)}
        ),
    )
    provider.decision = False
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    first = asyncio.create_task(rt.maybe_auto_refine(sid, "compact"))
    await provider.entered.wait()
    await rt.maybe_auto_refine(sid, "turn_interval")
    assert state.pending_interval
    provider.release.set()
    await first
    if state.auto_task:
        await state.auto_task
    assert len(provider.requests) == 2
    assert "compact" in input_sections(provider.requests[0])["trigger"]
    assert "turn_interval" in input_sections(provider.requests[1])["trigger"]
    assert state.turns_since_review == 0


@pytest.mark.parametrize("reason", ["compact", "turn_interval"])
async def test_cooldown_keeps_each_automatic_trigger(harness_runtime, reason):
    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)
    state.last_review_at, state.turns_since_review = time.time(), 25
    await rt.maybe_auto_refine(sid, reason)
    assert not provider.requests
    assert state.pending_compact if reason == "compact" else state.pending_interval
    state.pending_compact = state.pending_interval = False
    state.turns_since_review = 0


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"proposal": {"edits": [None, {"kind": "bad"}]}},
        {"proposal": proposal(edit("skill", reference={"type": "shell"}))},
    ],
)
async def test_extension_absent_malformed_and_invalid_proposals(
    harness_runtime, monkeypatch, value
):
    rt, sid, provider = harness_runtime
    monkeypatch.setattr(
        rt.environment.adapter(sid), "session_before_refine", lambda *_: value, raising=False
    )
    result = await rt.refine(sid)
    assert len(provider.requests) == (1 if value is None else 0)
    if value is not None:
        assert not any(item["applied"] for item in result["appliedEdits"])
    assert len(rt.store.harness.history(sid)) == 1


async def test_disposal_explicit_extension_skip_reports_failure(harness_runtime, monkeypatch):
    rt, sid, provider = harness_runtime
    monkeypatch.setattr(
        rt.environment.adapter(sid),
        "session_before_refine",
        lambda *_: {"skip": True},
        raising=False,
    )
    active_turn(rt, sid)
    rt.request_refinement(sid)
    rt.store.update(sid, pending_turn=None)
    await rt.drain_refinement(sid)
    assert not provider.requests
    assert (
        rt.store.events(sid, kind="refine_failed")[0]["payload"]["error"]
        == "Refinement skipped by extension"
    )


@pytest.mark.parametrize("failure", [False, True])
async def test_command_result_persistence_failure_is_context_only(
    harness_runtime, monkeypatch, failure
):
    rt, sid, provider = harness_runtime
    provider.fail = failure
    original = rt.store.add_context

    def append(sid, event, messages):
        if messages[0].get("customType") == "session_slash_command_result":
            raise OSError("result row failed")
        return original(sid, event, messages)

    monkeypatch.setattr(rt.store, "add_context", append)
    rt.message(None, sid, "/refine requested")
    await rt.refinement_state(sid).command_task
    assert len(rt.store.events(sid, kind="refine_failed")) == int(failure)
    messages = rt.context.unpersisted_refinement_messages[sid]
    assert messages[-1]["customType"] == "session_slash_command_result"
    assert messages[-1]["details"]["success"] == (not failure)
    assert "Command failed:" not in rt.refinement_input(sid)["conversation"]


async def test_branch_navigation_waits_for_active_command(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    rt.message(None, sid, "/refine on this branch")
    await provider.entered.wait()
    change = asyncio.create_task(rt.refinement_branch_changed(sid))
    await asyncio.sleep(0)
    assert not change.done() and rt.refinement_state(sid).branch_version == 0
    provider.release.set()
    await change
    assert len(rt.store.harness.history(sid)) == 1
    assert rt.refinement_state(sid).branch_version == 1


@pytest.mark.parametrize("status", ["invalidated", "plan"])
async def test_stale_background_consumption_and_newer_request(harness_runtime, status):
    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)

    async def result():
        return {"status": status, "branch": -1}

    state.background = asyncio.create_task(result())
    if status == "invalidated":
        state.pending_request = {"source": "self", "instructions": "newest"}
    await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == (1 if status == "invalidated" else 0)
    assert state.background is None and state.last_review_at > 0
    assert state.turns_since_review == 0 and state.pending_request is None


@pytest.mark.parametrize("serialized", [True, False])
async def test_model_and_serialized_setting_survive_refinement_and_disposal(
    harness_runtime, serialized
):
    rt, sid, _ = harness_runtime
    config = rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    rt.store.reconfigure(sid, config)
    assert rt.serialized_refinement(sid) == serialized
    before = rt.store.config(sid).provider.model_dump()
    await rt.refine(sid)
    await rt.drain_refinement(sid)
    assert rt.store.config(sid).provider.model_dump() == before
    assert rt.store.config(sid).serialized_refine == serialized


async def test_refine_resource_controls_actual_root_prompt(harness_runtime):
    rt, sid, _ = harness_runtime
    config = rt.store.config(sid)
    hidden = {
        "path": "/user/refine/SKILL.md",
        "kind": "python",
        "import_name": "refine",
        "disable_model_invocation": True,
    }
    assert "await refine.run" not in python_instructions(config, refine_skill=hidden)
    markdown = {"path": "/user/refine/SKILL.md", "kind": "markdown", "description": "User guide"}
    prompt = python_instructions(config, refine_skill=markdown)
    assert "Treat continual harness refinement" not in prompt
    assert "<python_import>refine" not in prompt
    assert "/user/refine/SKILL.md" in prompt and "User guide" in prompt


@pytest.mark.parametrize(
    "depth,enabled,exposed", [(0, True, True), (1, True, False), (0, False, False)]
)
def test_kernel_refine_exposure_matches_resource_and_root_policy(depth, enabled, exposed):
    values = {"workspace": Path(".")}
    bootstrap(SimpleNamespace(), values, {"depth": depth, "enable_builtin_skills": enabled})
    assert ("refine" in values) == exposed
    if exposed:
        assert callable(values["refine"].run) and callable(values["refine"].status)


def test_user_refine_module_replaces_owned_builtin_on_kernel_reload(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "refine", None)
    values = {"workspace": tmp_path}
    bootstrap(SimpleNamespace(), values, {"depth": 0})
    builtin = values["refine"]
    module = tmp_path / "__init__.py"
    module.write_text("async def run(): return 'USER_SKILL'\n")
    entry = {"name": "refine", "import_name": "refine", "package": str(module), "sha256": "user"}
    fresh = {"workspace": tmp_path}
    bootstrap(SimpleNamespace(), fresh, {"depth": 0, "skills": [entry]})
    assert fresh["refine"] is not builtin
    assert fresh["refine"].__file__ == str(module)
    assert not getattr(fresh["refine"], "__buffalo_builtin__", False)


async def test_scope_directories_history_and_saved_scope_are_authoritative(harness_runtime):
    rt, sid, _ = harness_runtime
    store = rt.store.harness
    assert store.path() == rt.store.directory / "harness"
    assert store.path(sid) == rt.store.directory / "sessions" / sid / "harness"
    result = await rt.refine(sid, global_=True)
    assert store.history(sid)[0]["id"] == result["id"]
    state = store.load()
    state["entries"]["memory"]["lesson"]["scope"] = "local"
    save_harness_state(store.path(), state)
    assert (
        load_harness_state(store.path(), "global")["entries"]["memory"]["lesson"]["scope"]
        == "local"
    )


async def test_compaction_digest_is_fresh_on_initial_and_merged_heads(harness_runtime):
    rt, sid, _ = harness_runtime
    for index in range(2):
        rt.store.harness.mutate(
            sid, "upsert", "memory", id="lesson", title="Lesson", content=f"revision {index}"
        )
        event = rt.store.event(sid, "seed", {})
        rt.store.add_context(sid, event, [{"role": "user", "content": "ordinary evidence"}])
        rt.context.compact(sid, count=1, summary=f"summary {index}", review_checkpoint=False)
        head = rt.context.compaction_message(rt.store.session(sid))
        assert f"revision {index}" in head["harnessDigest"]
        assert "Continual Harness State" not in head["summary"]
        text = convert_to_llm([head])[0]["content"]
        assert text.index("<harness_state>") < text.index("<summary>")
    shell_free = format_harness_state(
        rt.store.harness.merged(sid), include_ipython_examples=False, include_shell_examples=False
    )
    assert "without the Python REPL or shell access" in shell_free
    assert "await refine.run" not in shell_free and "<skill_import> ..." not in shell_free


@pytest.mark.parametrize("preparing", [False, True])
async def test_compaction_review_defers_to_queued_or_preparing_input(harness_runtime, preparing):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    if preparing:
        rt.tasks[sid] = asyncio.current_task()
    else:
        rt.message(None, sid, "real pending work")
    try:
        rt.refinement_compacted(sid)
        await rt.maybe_auto_refine(sid, "compact")
        assert not provider.requests and rt.refinement_state(sid).pending_compact
    finally:
        rt.tasks.pop(sid, None)
        rt.receive(sid)
    provider.decision = False
    await rt.maybe_auto_refine(sid, "compact")
    assert len(provider.requests) == 1


@pytest.mark.parametrize("background", [False, True])
async def test_public_plan_waits_for_serialized_plan_without_model_overlap(
    harness_runtime, background
):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    state = rt.refinement_state(sid)
    if background:
        active_turn(rt, sid)
        rt.request_refinement(sid, instructions="first")
        rt.store.update(sid, pending_turn={"context_committed": True})
    else:
        state.pending_request = {"source": "self", "instructions": "first"}
    checkpoint = asyncio.create_task(rt.refinement_checkpoint(sid))
    await provider.entered.wait()
    public = asyncio.create_task(rt.refine(sid, instructions="second"))
    await asyncio.sleep(0)
    assert len(provider.requests) == 1
    provider.release.set()
    await checkpoint
    rt.store.update(sid, pending_turn=None)
    await public
    assert len(provider.requests) == 2 and len(rt.store.harness.history(sid)) == 2


@pytest.mark.parametrize("serialized", [False, True])
async def test_public_apply_rechecks_operations_after_root_becomes_idle(
    harness_runtime, serialized
):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    )
    rt._admitted_turns.add(sid)
    entered = asyncio.Event()
    quiescence = rt._wait_refinement_quiescence

    async def waiting(session_id):
        entered.set()
        await quiescence(session_id)

    rt._wait_refinement_quiescence = waiting
    task = asyncio.create_task(rt.refine(sid))
    await entered.wait()
    rt._transitioning.add(sid)
    rt._admitted_turns.discard(sid)
    await asyncio.sleep(0.01)
    assert len(provider.requests) == 1 and not rt.store.harness.history(sid)
    barrier = asyncio.create_task(rt.wait_refinement_barrier(sid))
    rt.message(None, sid, "next legitimate request")
    assert not barrier.done()
    rt._transitioning.discard(sid)
    rt.store.update(sid, result="Concurrent root answer")
    await asyncio.gather(task, barrier)
    assert rt.store.session(sid).runnable and len(rt.store.messages(sid, pending=True)) == 1
    assert len(rt.store.harness.history(sid)) == 1
    assert rt.store.session(sid).result == "Concurrent root answer"
    rt.receive(sid)


@pytest.mark.parametrize("serialized", [False, True])
async def test_disposal_waits_for_owned_auto_operation(harness_runtime, serialized):
    rt, sid, _ = harness_runtime
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    )
    release = asyncio.Event()
    state = rt.refinement_state(sid)
    state.auto_task = asyncio.create_task(release.wait())
    drain = asyncio.create_task(rt.drain_refinement(sid))
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    await drain


@pytest.mark.parametrize("failure", [False, True])
async def test_disposal_compaction_and_disabled_compact_interval(harness_runtime, failure):
    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)
    state.pending_compact, state.turns_since_review = True, 25
    if failure:
        rt.review_refinement = AsyncMock(side_effect=ValueError("review failed"))
    else:
        rt.store.reconfigure(
            sid,
            rt.store.config(sid).model_copy(update={"refinement": RefinementPolicy(compact=False)}),
        )
    await rt.drain_refinement(sid)
    assert not state.pending_compact
    if failure:
        assert state.last_review_at > 0 and not provider.requests
    else:
        assert "turn_interval" in input_sections(provider.requests[0])["trigger"]
        assert len(rt.store.harness.history(sid)) == 1


async def test_disposal_retries_failed_explicit_background_once(harness_runtime):
    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)

    async def failed():
        return {
            "status": "failure",
            "branch": 0,
            "explicit": True,
            "options": {"source": "self", "instructions": "retry once"},
        }

    state.background = asyncio.create_task(failed())
    await rt.drain_refinement(sid)
    assert len(provider.requests) == 1 and len(rt.store.harness.history(sid)) == 1
    assert state.pending_request is None


async def test_interactive_approved_cancel_consumes_cooldown_like_prime(harness_runtime):
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    rt.review_refinement = AsyncMock(
        return_value={"shouldRefine": True, "rationale": "lesson", "instructions": ""}
    )

    async def cancel(*args, **kwargs):
        rt.invalidate_refinement(sid, branch_change=True)
        raise asyncio.CancelledError()

    rt.refine = cancel
    await rt.maybe_auto_refine(sid, "compact")
    assert state.last_review_at > 0 and not state.reviewing


@pytest.mark.parametrize("serialized", [False, True])
async def test_daemon_creation_preserves_serialized_setting(harness_runtime, serialized):
    rt, sid, _ = harness_runtime
    daemon = object.__new__(Daemon)
    daemon.runtime = rt
    cfg = rt.store.config(sid).model_dump(mode="json")
    cfg["serialized_refine"] = serialized
    created = await daemon.dispatch(
        "create",
        {"instruction": "new", "workspace": rt.store.session(sid).workspace.path, "config": cfg},
    )
    assert rt.serialized_refinement(created["id"]) == serialized


async def test_disposal_reloads_policy_between_compact_and_interval(harness_runtime, monkeypatch):
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    state.pending_compact, state.turns_since_review, state.last_review_at = True, 25, time.time()
    config = rt.store.config(sid)

    class ChangingPolicy:
        reads = 0

        def __getattr__(self, key):
            if key == "refinement":
                self.reads += 1
                return RefinementPolicy(cooldown_seconds=1200 if self.reads == 1 else 0)
            return getattr(config, key)

    changing = ChangingPolicy()
    monkeypatch.setattr(rt.store, "config", lambda *_: changing)
    rt.refinement_checkpoint = AsyncMock()
    await rt.drain_refinement(sid)
    assert not state.pending_compact and rt.refinement_checkpoint.await_count == 1
    state.turns_since_review = 0


async def test_due_interactive_disposal_does_not_wait_for_idle(harness_runtime):
    rt, sid, _ = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    rt.maybe_auto_refine = AsyncMock()
    rt._wait_refinement_quiescence = AsyncMock(side_effect=AssertionError("idle wait"))
    await rt.drain_refinement(sid)
    rt.maybe_auto_refine.assert_awaited_once_with(sid, "turn_interval")
    state.turns_since_review = 0


@pytest.mark.parametrize("phase", ["review", "plan"])
async def test_late_result_after_disposal_cannot_persist_or_plan(harness_runtime, phase):
    rt, sid, provider = harness_runtime
    entered, release = asyncio.Event(), asyncio.Event()
    original = provider.invoke

    async def blocked(request, emit):
        entered.set()
        await release.wait()
        return await original(request, emit)

    provider.invoke = blocked
    task = asyncio.create_task(
        rt.maybe_auto_refine(sid, "compact") if phase == "review" else rt.refine(sid)
    )
    await entered.wait()
    rt._closing = True
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert len(provider.requests) == 1 and not rt.store.harness.history(sid)
    assert not rt.store.events(sid, kind="refinement_notice")
    rt._closing = False


async def test_unknown_refine_host_request_is_rejected(harness_runtime):
    from threadweave.host_api import Request, dispatch
    from threadweave.tools import ToolContext

    rt, sid, _ = harness_runtime
    context = ToolContext(rt, sid, "action", rt.store.event(sid, "test", {}))
    with pytest.raises(ValueError, match="Unknown refinement operation"):
        await dispatch(context, Request(operation="refine.unknown"))


async def test_deferred_interval_is_consumed_once_below_threshold(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    state = rt.refinement_state(sid)
    state.turns_since_review = 1
    state.pending_interval = True
    rt._schedule_deferred_auto_refine(sid)
    scheduled = state.auto_task
    assert scheduled is not None and not state.pending_interval
    await scheduled
    await asyncio.sleep(0)
    assert state.auto_task is None and not state.pending_interval
    assert not provider.requests


@pytest.mark.parametrize("global_", [False, True])
@pytest.mark.parametrize("prefix", ["local:", "global:"])
async def test_planner_strips_display_prefix_without_changing_requested_scope(
    harness_runtime, global_, prefix
):
    rt, sid, provider = harness_runtime
    for scope in [False, True]:
        rt.store.harness.mutate(
            sid, "create", "memory", id="shared", title="Shared", content="original", global_=scope
        )
    provider.proposal = proposal(edit(action="update", id=prefix + "shared", content="updated"))
    result = await rt.refine(sid, global_=global_)
    assert result["appliedEdits"][0]["id"] == "shared"
    assert (
        rt.store.harness.load(None if global_ else sid)["entries"]["memory"]["shared"]["content"]
        == "updated"
    )
    assert (
        rt.store.harness.load(sid if global_ else None)["entries"]["memory"]["shared"]["content"]
        == "original"
    )


async def test_legacy_scope_less_global_rollback_stays_global(harness_runtime):
    from threadweave.harness import append_refinement_history

    rt, sid, _ = harness_runtime
    result = await rt.refine(sid, global_=True)
    result.pop("scope")
    result["appliedEdits"][0]["after"].pop("scope")
    append_refinement_history(rt.store.harness.path(), result)
    rt.store.record_harness_refinement(sid, result)
    reversed_ = await rt.refine(sid, rollback_id=result["id"])
    assert reversed_["scope"] == "global"
    assert not rt.store.harness.load()["entries"]["memory"]
    assert not rt.store.harness.load(sid)["entries"]["memory"]


async def test_interactive_real_turn_counter_deferred_review_and_navigation(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    state = rt.refinement_state(sid)
    state.turns_since_review = 24
    active_turn(rt, sid)
    rt.refinement_message_end(sid)
    assert state.turns_since_review == 25
    rt.store.update(sid, pending_turn=None)
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    reviewing = asyncio.create_task(rt.maybe_auto_refine(sid))
    await provider.entered.wait()
    rt.store.update(sid, paused=True)
    provider.release.set()
    await reviewing
    assert state.pending_review is not None
    await rt.refinement_branch_changed(sid)
    assert state.pending_review is None and state.turns_since_review == 0
    assert len(provider.requests) == 1 and not rt.store.harness.history(sid)


async def test_aborted_first_input_drops_digest_and_rearms_next_commit(harness_runtime):
    rt, sid, _ = harness_runtime
    rt.context.ensure_harness_digest(sid)
    assert sid in rt.context._pending_harness_digest
    await rt.pause(sid)
    assert sid not in rt.context._pending_harness_digest
    assert not rt.store.events(sid, kind="harness_digest")
    rt.resume(sid)
    rt.store.harness.mutate(sid, "create", "memory", title="fresh", content="AFTER_ABORT")
    rt.context.ensure_harness_digest(sid)
    messages = rt.context.messages(sid)
    rt.store.harness.mutate(sid, "update", "memory", id="fresh", title="fresh", content="AT_COMMIT")
    messages = rt.context.refresh_prepared_harness_digest(sid, messages)
    with rt.store.transaction():
        rt.context.commit_harness_digest(sid)
    rt.context.harness_digest_committed(sid)
    assert "AT_COMMIT" in str(messages) and "AFTER_ABORT" not in str(messages)
    assert len(rt.store.events(sid, kind="harness_digest")) == 1


async def test_broken_completion_listener_does_not_block_audit_consumers(
    harness_runtime, monkeypatch
):
    rt, sid, _ = harness_runtime

    def broken(*args):
        raise RuntimeError("listener failed")

    monkeypatch.setattr(rt.environment.adapter(sid), "refine_complete", broken, raising=False)
    result = await rt.refine(sid)
    assert rt.store.events(sid, kind="refine_complete")[0]["payload"]["id"] == result["id"]
    assert rt.store.harness.history(sid)[0]["id"] == result["id"]
    assert not rt.store.events(sid, kind="refine_failed")
