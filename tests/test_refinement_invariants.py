"""Prime scheduling and actual stable-prefix/notice context delivery."""

import json

import pytest

from threadweave.models import ModelResponse, Outcome
from threadweave.runtime import Runtime

from .test_continual_harness import edit, input_sections, proposal
from .test_continual_harness import harness_runtime as harness_runtime

LESSON = "Validate each subtotal against the source rows."
AUDIT = "AUDIT_ONLY_REFINEMENT_RESULT"


def test_learning_conversation_does_not_inherit_root_system_contract(harness_runtime):
    rt, sid, _ = harness_runtime
    config = rt.store.config(sid)
    config.task.instruction_messages = [
        {"role": "system", "content": "ROOT_SYSTEM_ONLY"},
        {"role": "developer", "content": "ROOT_DEVELOPER_ONLY"},
        {"role": "user", "content": "Ordinary earlier user context"},
    ]
    rt.store.reconfigure(sid, config)
    for review in (False, True):
        conversation = rt.refinement_input(sid, review=review, reason="turn_interval")[
            "conversation"
        ]
        assert "ROOT_SYSTEM_ONLY" not in conversation
        assert "ROOT_DEVELOPER_ONLY" not in conversation
        assert "Ordinary earlier user context" in conversation


@pytest.mark.parametrize("restart", [False, True])
async def test_completed_turn_notice_survives_resume_without_manufacturing_continuation(
    tmp_path, python_config, restart
):
    requests = []

    class Provider:
        async def invoke(self, request, emit):
            requests.append(request)
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement_review":
                inputs = input_sections(request)
                assert (
                    inputs["trigger"]
                    == "turn_interval; 25 assistant turns since last auto-refine review"
                )
                assert set(inputs) == {
                    "trigger",
                    "conversation",
                    "current_harness_state",
                    "refinement_history",
                }
                return ModelResponse(text=json.dumps({"shouldRefine": True, "rationale": AUDIT}))
            if purpose == "refinement":
                inputs = input_sections(request)
                assert set(inputs) == {
                    "conversation",
                    "current_harness_state",
                    "refinement_history",
                    "scope_policy",
                    "user_refine_instructions",
                }
                return ModelResponse(
                    text=json.dumps(
                        {
                            **proposal(edit("prompt", content=LESSON)),
                            "summary": AUDIT,
                            "rationale": AUDIT,
                            "expectedOutcome": AUDIT,
                        }
                    )
                )
            return ModelResponse(text="Task finished.")

    directory = tmp_path / "state"
    rt = Runtime(directory, providers={"mock": Provider()})
    root = rt.create("Check the totals.", tmp_path, config=python_config)
    sid = root.id
    try:
        rt.refinement_state(sid).turns_since_review = 24
        await rt._run_turn(sid)
        assert [r.metadata.get("purpose", "agent") for r in requests] == [
            "agent",
            "refinement_review",
            "refinement",
        ]
        assert LESSON not in requests[0].messages[0]["content"]
        assert rt.store.session(sid).outcome == Outcome.COMPLETED
        assert not rt.store.session(sid).runnable
        assert rt.store.session(sid).turns == 1
        assert not rt.store.events(sid, kind="refinement_continuation")
        assert len(rt.store.events(sid, kind="refinement_notice")) == 1
        assert len(rt.store.events(sid, kind="harness_digest")) == 1
        assert len(rt.store.refinement_history(sid)) == 1
        assert rt.store.refinement_history(sid)[0]["summary"] == AUDIT
        saved = json.loads((rt.store.harness.path(sid) / "harness_state.json").read_text())
        assert saved["entries"]["prompt"]["lesson"]["content"] == LESSON
        conversation = rt.store.session(sid).context
        assert AUDIT in json.dumps(conversation)
        assert LESSON in json.dumps(conversation)
        assert AUDIT in json.dumps(rt.store.trajectory(sid))
        assert not rt.store.messages(sid)

        if restart:
            await rt.shutdown()
            rt = Runtime(directory, providers={"mock": Provider()})
        assert LESSON not in rt.context.system_prompt(sid)
        assert AUDIT not in rt.context.system_prompt(sid)
        rt.interact(sid, "Now check the next batch.")
        await rt._run_turn(sid)
        next_request = requests[-1]
        assert next_request.metadata.get("purpose", "agent") == "agent"
        assert next_request.messages[0]["role"] == "system"
        assert next_request.messages[0] == requests[0].messages[0]
        assert AUDIT in json.dumps(next_request.messages)
        assert "Refinement complete:" not in json.dumps(next_request.messages)
        ordinary = [m for m in next_request.messages if m["role"] != "system"]
        assert LESSON in json.dumps(ordinary)
        assert any(
            m["role"] == "user" and m["content"].startswith("[auto-refinement]") for m in ordinary
        )
        assert "Now check the next batch." in json.dumps(ordinary)
        assert len(rt.store.messages(sid)) == 1
        assert rt.store.session(sid).outcome == Outcome.COMPLETED
        assert AUDIT in json.dumps(rt.store.trajectory(sid))
        for review in (True, False):
            inputs = rt.refinement_input(sid, review=review)
            assert AUDIT in inputs["refinement_history"]
            assert AUDIT in inputs["conversation"]
            assert LESSON in inputs["conversation"]
            assert "Refinement complete:" not in inputs["conversation"]
        # The fresh digest is carried mechanically, separate from the summary.
        rt.context.compact(sid, count=len(rt.store.session(sid).context))
        assert LESSON in rt.store.session(sid).summary_harness_digest
        assert rt.context.messages(sid)[0] == requests[0].messages[0]
    finally:
        await rt.shutdown()


async def test_children_cannot_trigger_review_or_explicit_refine(harness_runtime):
    rt, sid, provider = harness_runtime
    child = rt.spawn(sid, "Investigate a subtask")
    rt.store.update(child.id, pending_turn={"context_committed": True})
    assert not rt.request_refinement(child.id)["scheduled"]
    rt.refinement_compacted(child.id)
    rt.refinement_message_end(child.id)
    for _ in range(30):
        assert not await rt.refinement_checkpoint(child.id, completed_turn=True)
    assert not rt.has_pending_refinement(child.id)
    assert not provider.requests


async def test_approved_review_may_produce_no_edits(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.proposal = proposal()
    rt.refinement_compacted(sid)
    assert not await rt.refinement_checkpoint(sid)
    assert [r.metadata.get("purpose", "agent") for r in provider.requests] == [
        "refinement_review",
        "refinement",
    ]
    assert rt.store.events(sid, kind="refinement_review")[0]["payload"]["shouldRefine"]
    assert rt.store.refinement_history(sid)[0]["appliedEdits"] == []
    assert all(
        m.get("customType") == "refinement_outcome"
        for block in rt.store.session(sid).context
        for m in block["messages"]
    )


async def test_direct_typed_edits_leave_system_stable_and_refresh_at_cold_context(harness_runtime):
    rt, sid, _ = harness_runtime
    before = rt.context.system_prompt(sid)
    for kind in ("prompt", "memory", "skill", "subagent"):
        rt.store.harness.apply(
            sid, proposal(edit(kind, content=f"NEW {kind}")), id=f"create-{kind}"
        )
        assert f"NEW {kind}" not in rt.context.system_prompt(sid)
    rt.store.harness.apply(sid, proposal(edit("prompt", "update", content="UPDATED")), id="update")
    assert "UPDATED" not in rt.context.system_prompt(sid)
    assert "NEW prompt" not in rt.context.system_prompt(sid)
    rt.store.harness.apply(sid, proposal(edit("prompt", "delete")), id="delete")
    assert "UPDATED" not in rt.context.system_prompt(sid)
    assert rt.context.system_prompt(sid) == before
    assert not rt.store.session(sid).context
    rt.context.ensure_harness_digest(sid)
    assert "NEW memory" in json.dumps(rt.context.messages(sid))


async def test_global_planner_sees_only_global_store(harness_runtime):
    rt, sid, _ = harness_runtime
    rt.store.harness.mutate(
        sid, "create", "memory", id="local", title="Local only", content="Session progress"
    )
    rt.store.harness.mutate(
        sid,
        "create",
        "memory",
        global_=True,
        id="global",
        title="Global only",
        content="Durable preference",
    )
    assert "Local only" not in rt.refinement_input(sid, global_=True)["current_harness_state"]
    assert "Global only" in rt.refinement_input(sid, global_=True)["current_harness_state"]
    assert "Local only" in rt.refinement_input(sid)["current_harness_state"]
    assert "Global only" in rt.refinement_input(sid)["current_harness_state"]


async def test_compaction_failure_consumes_trigger_and_sets_cooldown(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.fail = True
    rt.refinement_compacted(sid)
    assert not await rt.refinement_checkpoint(sid)
    state = rt.refinement_state(sid)
    assert not state.pending_compact and state.last_review_at > 0
    state.last_review_at = 0
    assert not await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == 1


def test_proposal_discards_unknown_fields_and_normalizes_optional_types():
    from threadweave.harness import apply_refinement_proposal, empty_harness_state

    state = empty_harness_state()
    result = apply_refinement_proposal(
        state,
        {
            "summary": 123,
            "rationale": None,
            "expectedOutcome": [],
            "unsupported": "discard",
            "edits": [{**edit("prompt"), "path": None, "metadata": [], "unsupported": "discard"}],
        },
        id="normalized",
    )
    assert result["summary"] == "Refined continual harness state"
    assert result["rationale"] == result["expectedOutcome"] == ""
    assert result["appliedEdits"][0]["applied"]
    assert state["entries"]["prompt"]["lesson"]["path"] == "general"
    assert "unsupported" not in json.dumps(state)
    assert "unsupported" not in json.dumps(result)
