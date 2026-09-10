"""Ported contracts from Prime refinement.test.ts and serialized-refine/skill suites."""

import asyncio
import copy
import json
import re
import time
from pathlib import Path

import pytest

from threadweave.context import Context
from threadweave.harness import (
    KINDS,
    append_refinement_history,
    apply_refinement_proposal,
    empty_harness_state,
    format_harness_state,
    load_harness_state,
    load_refinement_history,
    merge_harness_states,
    merge_refinement_history,
    rollback_proposal,
    save_harness_state,
)
from threadweave.models import ModelResponse, RefinementPolicy
from threadweave.runtime import Runtime
from threadweave.storage import Store

REF = {"type": "python", "import": "json", "callable": "loads", "call_pattern": "loads(text)"}


def input_sections(request):
    return dict(re.findall(r"<([a-z_]+)>\n([\s\S]*?)\n</\1>", request.messages[-1]["content"]))


def edit(kind="memory", action="create", id="lesson", **kwargs):
    return {
        "kind": kind,
        "action": action,
        "id": id,
        "title": "Lesson",
        "content": "Run checks in the project environment",
        **(
            {"reference": REF, "arguments": {"text": {"type": "string", "required": True}}}
            if kind == "skill"
            else {}
        ),
        **kwargs,
    }


def proposal(*edits):
    return {
        "summary": "Retain project lesson",
        "rationale": "Observed repeated error",
        "expectedOutcome": "Use the correct environment",
        "edits": list(edits),
    }


def apply(state, *edits, **options):
    return apply_refinement_proposal(
        state, proposal(*edits), id=options.pop("id", "refine_test"), **options
    )


@pytest.mark.parametrize("payload", [None, "not json", "null", "[]", '"string"', "123"])
def test_missing_and_corrupt_state(tmp_path, payload):
    if payload is not None:
        (tmp_path / "harness_state.json").write_text(payload)
    assert load_harness_state(tmp_path) == empty_harness_state()


def test_atomic_save_roundtrip_and_failure(tmp_path, monkeypatch):
    state = empty_harness_state()
    apply(state, edit())
    path = Path(save_harness_state(tmp_path, state))
    assert path.read_text().endswith("\n") and '\n  "schema"' in path.read_text()
    assert load_harness_state(tmp_path) == state
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("threadweave.harness.os.replace", fail)
    with pytest.raises(OSError):
        save_harness_state(tmp_path, empty_harness_state())
    assert path.read_bytes() == original
    assert {p.name for p in tmp_path.iterdir()} == {
        "harness_state.json",
        "state_changes.jsonl",
        ".harness.lock",
    }


@pytest.mark.parametrize("kind", KINDS)
def test_create_update_delete_stable_id_versions(kind):
    state = empty_harness_state()
    first = apply(state, edit(kind))
    before = copy.deepcopy(first["appliedEdits"][0]["after"])
    second = apply(state, edit(kind, "update", content="corrected"))
    after = second["appliedEdits"][0]["after"]
    assert after["id"] == before["id"] and after["version"] == 2
    assert after["created_at"] == before["created_at"]
    assert second["appliedEdits"][0]["before"] == before
    deleted = apply(state, edit(kind, "delete"))
    assert deleted["appliedEdits"][0]["before"] == after
    assert not state["entries"][kind]


def test_validation_generated_ids_and_sequential_conflicts():
    state = empty_harness_state()
    result = apply(state, edit("prompt", id=None, title="Base system prompt"))
    assert not result["appliedEdits"][0]["applied"]
    invalid = edit("skill")
    invalid.pop("arguments")
    assert not apply(state, invalid)["appliedEdits"][0]["applied"]
    invalid = edit("skill", reference={"type": "shell"})
    assert not apply(state, invalid)["appliedEdits"][0]["applied"]
    apply(state, edit(id=None, title="Native Check!"))
    assert "native_check" in state["entries"]["memory"]
    apply(state, edit())
    baseline = copy.deepcopy(state)
    result = apply(
        state,
        edit(action="update", content="first"),
        edit(action="update", content="second"),
        baseline_state=baseline,
    )
    assert all(e["applied"] for e in result["appliedEdits"])
    result = apply(state, edit(action="update"), baseline_state=baseline)
    assert result["appliedEdits"][0]["error"] == "entry changed during refinement planning"
    assert state["entries"]["memory"]["lesson"]["version"] == 3


def test_merge_preserves_collisions_without_mutating_inputs():
    global_state, local = empty_harness_state(), empty_harness_state()
    apply(global_state, edit(), scope="global")
    apply(local, edit(content="local override"))
    merged = merge_harness_states(global_state, local)
    assert merged["entries"]["memory"]["lesson"]["scope"] == "global"
    assert merged["entries"]["memory"]["local:lesson"]["content"] == "local override"
    merged["entries"]["memory"]["lesson"]["content"] = "changed"
    assert global_state["entries"]["memory"]["lesson"]["content"] != "changed"


def test_jsonl_reports_corruption_and_merges(tmp_path):
    result = apply(empty_harness_state(), edit(), scope="global")
    append_refinement_history(tmp_path, result)
    with (tmp_path / "refinements.jsonl").open("a") as f:
        f.write('bad\n{"id":"invalid"}\n')
    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_refinement_history(tmp_path)
    assert merge_refinement_history([result], [{**result, "summary": "new"}])[0]["summary"] == "new"


def test_inverse_rollback_restores_create_update_delete():
    state = empty_harness_state()
    apply(state, edit(), edit("skill", id="deleted"))
    result = apply(
        state,
        edit("prompt", id="created"),
        edit(action="update", content="wrong"),
        edit("skill", "delete", id="deleted"),
    )
    rollback = apply_refinement_proposal(
        state, rollback_proposal(result), id="undo", rollback_of=result["id"]
    )
    assert not state["entries"]["prompt"]
    assert state["entries"]["memory"]["lesson"]["content"] == edit()["content"]
    assert state["entries"]["memory"]["lesson"]["version"] == 3
    assert state["entries"]["skill"]["deleted"]["version"] == 1
    assert rollback["rollbackOf"] == result["id"]


def test_harness_digest_bounded_with_five_recent_refinements():
    state = empty_harness_state()
    for i in range(9):
        apply(state, edit(id=str(i), content="x" * 500), id=str(i))
    digest = format_harness_state(state)
    assert "[local:0]" in digest and "[local:6]" not in digest
    assert "+3 more memory" in digest
    assert "x" * 181 not in digest
    assert digest.count("Retain project lesson") == 5
    assert "+4 older refinement events" in digest
    assert "Observed repeated error" not in digest


class Refiner:
    def __init__(self, decision=True):
        self.requests = []
        self.decision = decision
        self.proposal = proposal(edit())
        self.entered = self.release = None
        self.fail = False

    async def invoke(self, request, emit):
        self.requests.append(request)
        if self.entered:
            self.entered.set()
            await self.release.wait()
        if self.fail:
            raise ValueError("provider unavailable")
        value = (
            {
                "shouldRefine": self.decision,
                "rationale": "reusable lesson",
                "instructions": "keep the lesson",
            }
            if request.metadata["purpose"] == "refinement_review"
            else self.proposal
        )
        return ModelResponse(text=json.dumps(value))


@pytest.fixture
async def harness_runtime(tmp_path, config):
    provider = Refiner()
    config.features.model_compaction = False
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("learn", tmp_path, config=config)
    try:
        yield runtime, root.id, provider
    finally:
        await runtime.shutdown()


def test_prime_defaults():
    assert RefinementPolicy().model_dump() == {
        "enabled": True,
        "turn_interval": 25,
        "compact": True,
        "cooldown_seconds": 1200,
    }


@pytest.mark.parametrize("override", [None, False])
async def test_schedule_status_coalesce_and_safe_boundary(harness_runtime, override):
    rt, sid, provider = harness_runtime
    assert not rt.request_refinement(sid)["scheduled"]
    rt.store.update(sid, pending_turn={"context_committed": False})
    assert rt.request_refinement(sid, instructions="first", global_=True)["scheduled"]
    rt.request_refinement(
        sid, instructions="last", **({} if override is None else {"global_": override})
    )
    assert rt.refinement_status(sid) == {"pending": True, "in_flight": True}
    assert not await rt.refinement_checkpoint(sid)
    assert not provider.requests and not rt.store.harness.entries(sid)
    rt.store.update(sid, pending_turn={"context_committed": True})
    assert await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == 1 and provider.requests[0].metadata["purpose"] == "refinement"
    evidence = input_sections(provider.requests[0])
    assert (
        evidence["user_refine_instructions"] == "last"
        and ("scope: global" if override is None else "scope: local") in evidence["scope_policy"]
    )
    assert rt.store.harness.load(None if override is None else sid)["entries"]["memory"]["lesson"]
    assert not rt.store.harness.load(sid if override is None else None)["entries"]["memory"]
    assert rt.refinement_status(sid) == {"pending": False, "in_flight": False}


async def test_local_cannot_update_or_delete_global_and_can_override(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.harness.apply(sid, proposal(edit()), id="global", global_=True)
    provider.proposal = proposal(
        edit(action="update", id="global:lesson", content="wrong"),
        edit(action="delete", id="global:lesson"),
    )
    result = await rt.refine(sid)
    assert not any(e["applied"] for e in result["appliedEdits"])
    assert rt.store.harness.load()["entries"]["memory"]["lesson"]["version"] == 1
    assert all(not e["applied"] for e in rt.store.harness.history(sid)[-1]["appliedEdits"])
    provider.proposal = proposal(edit(content="local override"))
    assert (await rt.refine(sid))["appliedEdits"][0]["applied"]
    assert len(rt.store.harness.merged(sid)["entries"]["memory"]) == 2


@pytest.mark.parametrize("decision", [False, True])
async def test_interval_gate_cooldown_and_no_duplicates(harness_runtime, decision):
    rt, sid, provider = harness_runtime
    provider.decision = decision
    for _ in range(24):
        assert not await rt.refinement_checkpoint(sid, completed_turn=True)
    assert not provider.requests
    assert await rt.refinement_checkpoint(sid, completed_turn=True) == decision
    assert len(provider.requests) == (2 if decision else 1)
    assert bool(rt.store.events(sid, kind="refinement_notice")) == decision
    for _ in range(25):
        await rt.refinement_checkpoint(sid, completed_turn=True)
    assert len(provider.requests) == (2 if decision else 1)
    assert rt.refinement_state(sid).pending_interval
    rt.refinement_state(sid).last_review_at = time.time() - 1201
    provider.decision = False
    await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == (3 if decision else 2)


async def test_compaction_pending_cooldown_disabled_fallback(harness_runtime):
    rt, sid, provider = harness_runtime
    state = rt.refinement_state(sid)
    state.last_review_at = time.time()
    rt.refinement_compacted(sid)
    await rt.refinement_checkpoint(sid)
    assert state.pending_compact and not provider.requests
    state.last_review_at = 0
    provider.decision = False
    await rt.refinement_checkpoint(sid)
    assert not state.pending_compact and len(provider.requests) == 1
    assert input_sections(provider.requests[0])["trigger"].startswith("compact;")


async def test_inflight_protection_branch_invalidation_and_recovery(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    rt.refinement_state(sid).pending_request = {"source": "self"}
    task = asyncio.create_task(rt.refinement_checkpoint(sid))
    await provider.entered.wait()
    assert rt.refinement_status(sid)["in_flight"]
    other = asyncio.create_task(rt.refinement_checkpoint(sid))
    await asyncio.sleep(0)
    assert not other.done()
    rt.invalidate_refinement(sid)
    provider.release.set()
    await asyncio.gather(task, other, return_exceptions=True)
    assert not rt.refinement_status(sid)["in_flight"]
    assert not rt.store.harness.entries(sid)
    rt.refinement_state(sid).pending_request = {"source": "self"}
    assert await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == 2


async def test_approved_review_deferred_until_safe_and_failure_unwedges(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.reconfigure(sid, rt.store.config(sid).model_copy(update={"serialized_refine": False}))
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    rt.refinement_compacted(sid)
    task = rt.refinement_state(sid).auto_task
    await provider.entered.wait()
    rt.store.update(sid, paused=True)
    provider.release.set()
    await task
    assert rt.refinement_state(sid).pending_review
    assert len(provider.requests) == 1
    rt.store.update(sid, paused=False)
    provider.fail = True
    await rt.maybe_auto_refine(sid)
    assert not rt.refinement_status(sid)["in_flight"]
    provider.fail = False
    rt.refinement_state(sid).last_review_at = 0
    await rt.maybe_auto_refine(sid)
    assert rt.store.harness.entries(sid)
    assert sum(r.metadata["purpose"] == "refinement_review" for r in provider.requests) == 1


async def test_planner_input_bounded_state_history_scope_and_empty(harness_runtime):
    rt, sid, provider = harness_runtime
    event = rt.store.event(sid, "observation", {})
    rt.store.add_context(sid, event, [{"role": "user", "content": "z" * 100000 + "RECENT"}])
    for review, bound in ((False, 80000), (True, 40000)):
        evidence = rt.refinement_input(sid, review=review)
        assert len(evidence["conversation"]) == bound
        assert "RECENT" in evidence["conversation"]
        assert evidence["current_harness_state"]
        assert set(evidence) == (
            {"current_harness_state", "refinement_history", "conversation", "trigger"}
            if review
            else {"current_harness_state", "refinement_history", "conversation", "scope_policy"}
        )
    provider.proposal = proposal()
    # Keep actual model request within the configured context cap.
    rt.store.update(sid, context=[], summary="Compacted reusable observation")
    assert "Compacted reusable observation" in rt.refinement_input(sid)["conversation"]
    assert not (await rt.refine(sid))["appliedEdits"]
    assert not rt.store.events(sid, kind="refinement_notice")
    assert len(rt.store.harness.history(sid)) == 1


@pytest.mark.parametrize("completed", [True, False])
async def test_idle_refinement_does_not_schedule_root_continuation(harness_runtime, completed):
    rt, sid, provider = harness_runtime
    rt.store.update(sid, runnable=False, outcome="completed" if completed else "active")
    await rt.refine(sid)
    root = rt.store.session(sid)
    assert root.outcome == ("completed" if completed else "active") and not root.runnable
    assert len(provider.requests) == 1
    assert not rt.store.events(sid, kind="refinement_continuation")


async def test_idle_compaction_is_serviced_when_safe(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.update(sid, runnable=False, outcome="completed")
    event = rt.store.event(sid, "observation", {})
    rt.store.add_context(sid, event, [{"role": "user", "content": "Repeated useful lesson"}])
    rt.context.compact(sid, count=1)
    assert rt.has_pending_refinement(sid)
    await rt._run_turn(sid)
    assert [r.metadata["purpose"] for r in provider.requests] == ["refinement_review", "refinement"]
    assert not rt.store.session(sid).runnable
    assert not rt.has_pending_refinement(sid)


def test_digest_cannot_loop_compaction_below_fixed_context_capacity(tmp_path, config):
    from threadweave.models import HarnessError, Workspace

    store = Store(tmp_path / "state")
    root = store.create("Tiny budget", Workspace(path=str(tmp_path)), config)
    try:
        with pytest.raises(HarnessError, match="context_capacity|Instructions and tool schemas"):
            Context(store).assemble(root.id, [], input_budget=1)
        assert len(store.events(root.id, kind="context_compaction")) < 4
    finally:
        store.close()


async def test_rollback_history_across_sessions(harness_runtime, tmp_path, config):
    rt, sid, provider = harness_runtime
    await rt.refine(sid, global_=True)
    target = rt.store.harness.history(sid)[-1]
    other = rt.create("new", tmp_path, config=config)
    assert (await rt.refine(other.id, rollback_id=target["id"]))["appliedEdits"][0]["applied"]
    assert not rt.store.harness.load()["entries"]["memory"]
    assert rt.store.harness.history(other.id)[-1]["rollbackOf"] == target["id"]


async def test_copied_local_history_rollback_targets_original_file(
    harness_runtime, tmp_path, config
):
    rt, sid, provider = harness_runtime
    await rt.refine(sid)
    target = rt.store.harness.history(sid)[-1]
    branch = rt.create("branch", tmp_path, config=config)
    rt.store.harness.apply(branch.id, proposal(edit(content="branch keeps this")), id="branch")
    rt.store.record_harness_refinement(branch.id, target)
    assert (await rt.refine(branch.id, rollback_id=target["id"]))["appliedEdits"][0]["applied"]
    assert not rt.store.harness.load(sid)["entries"]["memory"]
    assert rt.store.harness.get(branch.id, "memory", "lesson")["content"] == "branch keeps this"
    assert rt.store.harness.history(branch.id)[-1]["rollbackOf"] == target["id"]


async def test_only_prime_triggers_schedule_review(harness_runtime):
    rt, sid, provider = harness_runtime
    for kind in (
        "failure",
        "child_evidence_used",
        "experiment",
        "completion",
        "python_error",
        "verification_result",
        "verifier_result",
        "semantic_state_updated",
    ):
        rt.store.event(sid, kind, {"passed": False, "finding": "reusable lesson"})
        assert not await rt.refinement_checkpoint(sid)
    assert not provider.requests
    rt.store.reconfigure(
        sid, rt.store.config(sid).model_copy(update={"refinement": RefinementPolicy(enabled=False)})
    )
    rt.refinement_compacted(sid)
    assert not await rt.refinement_checkpoint(sid)
    assert not provider.requests
    assert (await rt.refine(sid))["appliedEdits"][0]["applied"]
    assert [r.metadata["purpose"] for r in provider.requests] == ["refinement"]
