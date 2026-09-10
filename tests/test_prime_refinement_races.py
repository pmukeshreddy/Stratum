"""Focused ports of Prime serialized-refine branch/abort/disposal regressions."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from threadweave.refinement import RefineSkippedError

from .test_continual_harness import harness_runtime as harness_runtime
from .test_continual_harness_gaps import active_turn


@pytest.mark.parametrize("error", [ValueError("late failure"), RefineSkippedError("late skip")])
async def test_stale_background_rejection_is_invalidated(harness_runtime, error):
    # Prime serialized-refine:949; both catch branches obey branch identity.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)

    async def plan(*args):
        state.branch_version += 1
        raise error

    rt.plan_refinement = plan
    assert await rt.background_refinement_plan(sid, {"source": "self"}, 0) == {
        "status": "invalidated",
        "branch": 0,
    }
    assert not rt.store.harness.history(sid)


@pytest.mark.parametrize("path", ["background", "direct", "public", "review"])
@pytest.mark.parametrize("branch_change", [False, True])
async def test_abort_and_branch_change_cancel_owned_work_without_late_apply(
    harness_runtime,
    path,
    branch_change,
):
    # Prime serialized-refine:1760,2112,2145,2168,2237.
    rt, sid, provider = harness_runtime
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = provider.invoke

    async def invoke(request, emit):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # The provider deliberately ignores cancellation.
        return await original(request, emit)

    provider.invoke = invoke
    state = rt.refinement_state(sid)
    if path == "background":
        active_turn(rt, sid)
        rt.request_refinement(sid, instructions="obsolete")
        task = state.background
    elif path == "public":
        task = asyncio.create_task(rt.refine(sid))
    else:
        if path == "review":
            state.turns_since_review = 25
        else:
            state.pending_request = {"instructions": "obsolete", "source": "self"}
        task = asyncio.create_task(rt.refinement_checkpoint(sid))
    await entered.wait()
    if branch_change:
        invalidation = asyncio.create_task(rt.refinement_branch_changed(sid))
    else:
        rt.invalidate_refinement(sid)
        invalidation = None
    await cancelled.wait()
    release.set()
    await asyncio.gather(task, *([invalidation] if invalidation else []), return_exceptions=True)
    assert not rt.store.harness.history(sid)
    assert not rt.store.events(sid, kind="refine_failed")
    assert state.pending_request is None
    assert state.last_review_at == 0
    rt.store.update(sid, pending_turn=None)
    if branch_change:
        assert state.background is None and state.turns_since_review == 0


async def test_newer_pending_request_wins_over_old_background_failure(harness_runtime):
    # Prime serialized-refine:1005,2286; exact newer plan, one application.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)

    async def failed():
        return {
            "status": "failure",
            "branch": 0,
            "explicit": True,
            "options": {"source": "self", "instructions": "old"},
        }

    state.background = asyncio.create_task(failed())
    state.pending_request = {"source": "self", "instructions": "new"}
    original = rt.plan_refinement
    rt.plan_refinement = AsyncMock(wraps=original)
    await rt.refinement_checkpoint(sid)
    assert rt.plan_refinement.call_args.args[1]["instructions"] == "new"
    assert rt.plan_refinement.await_count == 1
    assert len(rt.store.harness.history(sid)) == 1 and state.pending_request is None


async def test_branch_change_during_auto_apply_does_not_stamp_new_cooldown(harness_runtime):
    # Prime serialized-refine:2264; completion listeners may yield after persistence.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    state.turns_since_review = 25

    async def apply(*args):
        state.branch_version += 1
        return {"appliedEdits": []}

    rt.apply_refinement_plan = apply
    await rt.refinement_checkpoint(sid)
    assert state.last_review_at == 0 and state.turns_since_review == 25
    state.turns_since_review = 0


async def test_disposal_waits_for_checkpoint_claimed_apply(harness_runtime):
    # Prime serialized-refine:2475.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    entered, release = asyncio.Event(), asyncio.Event()
    original = rt.apply_refinement_plan

    async def apply(*args):
        entered.set()
        await release.wait()
        return await original(*args)

    rt.apply_refinement_plan = AsyncMock(side_effect=apply)
    plan = await rt.plan_refinement(sid, {})

    async def ready():
        return {"status": "plan", "branch": 0, "plan": plan, "options": {}}

    state.background = asyncio.create_task(ready())
    checkpoint = asyncio.create_task(rt.refinement_checkpoint(sid))
    await entered.wait()
    drain = asyncio.create_task(rt.drain_refinement(sid))
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    await asyncio.gather(checkpoint, drain)
    assert rt.apply_refinement_plan.await_count == 1
    assert len(rt.store.harness.history(sid)) == 1


@pytest.mark.parametrize("serialized", [False, True])
async def test_disposal_drains_explicit_once_before_due_interval(harness_runtime, serialized):
    # Prime serialized-refine:214,416,1042.
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    )
    state = rt.refinement_state(sid)
    state.pending_request = {"source": "self"}
    state.turns_since_review = 25
    await rt.drain_refinement(sid)
    await rt.drain_refinement(sid)
    assert [r.metadata["purpose"] for r in provider.requests] == ["refinement"]
    assert len(rt.store.harness.history(sid)) == 1
    assert state.turns_since_review == 0 and state.last_review_at > 0


async def test_disposal_clears_compaction_during_cooldown(harness_runtime):
    # Prime serialized-refine:1487; a normal checkpoint preserves it.
    import time

    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)
    state.pending_compact, state.last_review_at = True, time.time()
    await rt.refinement_checkpoint(sid)
    assert state.pending_compact
    await rt.drain_refinement(sid)
    assert not state.pending_compact and not provider.requests


async def test_shutdown_is_memoized_while_refinement_drains(harness_runtime):
    # Prime serialized-refine:2555.
    rt, sid, _ = harness_runtime
    rt.refinement_state(sid)
    entered, release = asyncio.Event(), asyncio.Event()

    async def drain(*args):
        entered.set()
        await release.wait()

    rt.drain_refinement = AsyncMock(side_effect=drain)
    first = asyncio.create_task(rt.shutdown())
    await entered.wait()
    second = asyncio.create_task(rt.shutdown())
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    release.set()
    await asyncio.gather(first, second)
    assert rt.drain_refinement.await_count == 1


async def test_public_refine_discards_settled_background_left_by_aborted_turn(harness_runtime):
    # Prime serialized-refine:2572.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)

    async def skipped():
        return {"status": "skip", "branch": 0}

    state.background = asyncio.create_task(skipped())
    state.background_options = {"instructions": "stale", "global_": True}
    await rt.refine(sid, instructions="current")
    assert state.background is None and state.background_options is None
    assert len(rt.store.harness.history(sid)) == 1


@pytest.mark.parametrize("draining", [False, True])
@pytest.mark.parametrize("phase", ["plan_refinement", "apply_refinement_plan"])
async def test_explicit_boundary_failure_is_contained_and_disposal_is_best_effort(
    harness_runtime,
    draining,
    phase,
):
    # Prime serialized-refine:2398,2417; disposeAsync's explicit drain suppresses errors.
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    state.pending_request = {"source": "self"}
    setattr(rt, phase, AsyncMock(side_effect=ValueError("unavailable")))
    if draining:
        await rt.drain_refinement(sid)
    else:
        await rt.refinement_checkpoint(sid)
    assert len(rt.store.events(sid, kind="refine_failed")) == (0 if draining else 1)
    assert state.pending_request is None and state.last_review_at > 0


@pytest.mark.parametrize("serialized", [False, True])
async def test_aborted_response_clears_pending_explicit_without_host_abort(
    harness_runtime, serialized
):
    # Prime serialized-refine:2054,2322.
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"serialized_refine": serialized})
    )
    state = rt.refinement_state(sid)
    state.pending_request = {"source": "self", "instructions": "must not leak"}
    rt.store.update(
        sid,
        pending_turn={
            "context_committed": True,
            "response": {"metadata": {"stop_reason": "aborted"}},
        },
    )
    rt.refinement_message_end(sid)
    assert state.pending_request is None
    assert not await rt.refinement_checkpoint(sid)
    assert not provider.requests and not rt.store.harness.history(sid)
    rt.store.update(sid, pending_turn=None)
