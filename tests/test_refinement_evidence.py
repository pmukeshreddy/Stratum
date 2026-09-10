"""Evidence integrity and optional learning; scripted inference is not natural-use evidence."""

import json

import pytest

from threadweave.models import Action, ModelResponse, RunConfig, new_id
from threadweave.refinement_evidence import bounded_records
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext


async def test_original_roles_and_successful_candidate_survive_context_tail(tmp_path):
    original = [
        {"role": "system", "content": "Return only source code."},
        {"role": "developer", "content": "Use Python."},
        {
            "role": "user",
            "content": "Explain the result and annotate every parameter and return type.",
        },
    ]
    rt = Runtime(tmp_path / "state")
    try:
        root = rt.create(
            "Implement double.", tmp_path, config=RunConfig(task={"original_messages": original})
        )
        decision = rt.store.event(
            root.id,
            "model_response",
            {
                "text": "I will discard the user's entire request because prose is disallowed.",
                "actions": [],
                "metadata": {},
            },
        )
        result = await rt.execute_python(
            ToolContext(rt, root.id, new_id(), decision),
            "candidate = 'def double(x): return x * 2'\nexec(candidate)\nassert double(3) == 6\nprint(candidate)",
        )
        assert not result["error"]
        rt.store.update(root.id, context=[], summary="Earlier work compacted.")
        event = rt.store.event(root.id, "observation", {})
        rt.store.add_context(
            root.id, event, [{"role": "user", "content": "z" * 100_000 + "RECENT"}]
        )
        for review in (False, True):
            value = rt.refinement_input(root.id, review=review)
            assert value["original_task"]["messages"] == original
            assert value["conversation_omitted_chars"] > 0
            assert not value["in_task_opportunity"]["observed_failures"]
            evidence = value["trajectory_evidence"]["records"]
            assert any(
                r["id"] == decision and "entire request" in r["payload"]["text"] for r in evidence
            )
            assert any(
                r["type"] == "python_execution" and "def double(x)" in r["payload"]["code"]
                for r in evidence
            )
            assert any(
                r["type"] == "python_result" and "def double(x)" in r["payload"]["stdout"]
                for r in evidence
            )
    finally:
        await rt.shutdown()


async def test_full_existing_and_previous_lessons_are_available_in_both_scopes(tmp_path):
    rt = Runtime(tmp_path / "state")
    try:
        root = rt.create("Work.", tmp_path)
        content = "Context. " * 60 + "The differentiating rule is retained at the end."
        proposal = {
            "summary": "Earlier discovery",
            "rationale": "observed earlier",
            "edits": [
                {
                    "action": "create",
                    "kind": "memory",
                    "id": "known",
                    "title": "Known procedure",
                    "content": content,
                }
            ],
        }
        rt.store.harness.apply(root.id, proposal, id="prior")
        for global_ in (False, True):
            evidence = rt.refinement_input(root.id, global_=global_)
            assert evidence["current_harness_state"]["entries"][0]["content"] == content
            assert (
                evidence["refinement_history"][0]["appliedEdits"][0]["after"]["content"] == content
            )
            assert evidence["current_harness_state"]["coverage"] == {"omitted": 0, "truncated": []}
        corrected = rt.store.event(
            root.id,
            "model_response",
            {
                "text": "Already used the known procedure and checked it.",
                "metadata": {},
            },
        )
        rt.store.event(
            root.id,
            "model_response",
            {
                "text": "Auxiliary opinion must not become observed root behavior.",
                "metadata": {"purpose": "refinement"},
            },
        )
        evidence = rt.refinement_input(root.id)["trajectory_evidence"]["records"]
        assert any(r["id"] == corrected for r in evidence)
        assert "Auxiliary opinion" not in json.dumps(evidence)
    finally:
        await rt.shutdown()


def test_evidence_coverage_discloses_missing_middle_and_large_records():
    records = [{"id": str(i), "type": "observation", "payload": "x" * 100} for i in range(100)]
    kept, coverage = bounded_records(records, 1600)
    assert kept[0] == records[0] and kept[-1] == records[-1]
    assert coverage["omitted"] == len(records) - len(kept)
    kept, coverage = bounded_records([{"id": "large", "payload": "x" * 20_000}], 1000)
    assert coverage["truncated"] == ["large"] and kept[0]["truncated"]


async def test_global_refinement_full_notice_restart_and_fresh_session_use(tmp_path):
    rule = "For the Acorn fixture, catalog.lookup(name) returns a record; read its 'payload' field."

    class Provider:
        async def invoke(self, request, emit):
            if request.metadata.get("purpose") == "refinement":
                return ModelResponse(
                    text=json.dumps(
                        {
                            "summary": "Remember Acorn's observed API",
                            "learning_assessment": {
                                "decision": "reusable",
                                "new_observation": "lookup returned a record",
                            },
                            "edits": [
                                {
                                    "action": "create",
                                    "kind": "memory",
                                    "id": "acorn",
                                    "title": "Acorn lookup",
                                    "content": rule,
                                }
                            ],
                        }
                    )
                )
            if request.name == "root":
                if request.turn == 0:
                    code = "record = {'payload': 7}\nassert record != 7\nprint(await refine.run('Retain the observed Acorn API for later Acorn sessions', global_=True))"
                    return ModelResponse(actions=[Action(name="ipython", arguments={"code": code})])
                assert rule in str(request.messages)
                return ModelResponse(text="Observed and retained.")
            if request.turn == 0:
                assert "Acorn lookup" in str(request.messages)
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "entry = rlm.harness.get_memory('acorn', global_=True)\nprint(entry)"
                            },
                        )
                    ]
                )
            if request.turn == 1:
                assert rule in str(request.messages)
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "record = {'payload': 11}\nanswer = record['payload'] * 2\nassert answer == 22\nprint(answer)"
                            },
                        )
                    ]
                )
            return ModelResponse(text="22")

    config = RunConfig(provider={"name": "mock", "model": "deterministic"})
    directory = tmp_path / "state"
    rt = Runtime(directory, providers={"mock": Provider()})
    try:
        root = rt.create("Inspect Acorn.", tmp_path, config=config)
        await rt.start()
        assert (await rt.wait(root.id)).outcome == "completed"
        assert not rt.store.harness.load(root.id)["entries"]["memory"]
        event = rt.store.events(root.id, kind="refine_complete")[0]["payload"]
        assert event["scope"] == "global" and event["learning_assessment"]["decision"] == "reusable"
        assert rt.store.events(root.id, kind="refinement_notice")[0]["payload"]["expanded"]
    finally:
        await rt.shutdown()
    rt = Runtime(directory, providers={"mock": Provider()})
    try:
        fresh = rt.create("Use Acorn for the next record.", tmp_path, config=config, name="fresh")
        assert not rt.store.harness.load(fresh.id)["entries"]["memory"]
        await rt.start()
        assert (await rt.wait(fresh.id)).result == "22"
        assert not rt.store.events(fresh.id, kind="refine_scheduled")
    finally:
        await rt.shutdown()


@pytest.mark.parametrize(
    "candidate,status,approve,trigger",
    [
        ("def double(x): return x * 2", "violated", False, "turn_interval"),
        ("def double(x: int) -> int: return x * 2", "satisfied", False, "turn_interval"),
        ("def double(x): return x * 2", "violated", True, "compact"),
        ("def double(x): return x * 2", "violated", True, "background_interval"),
    ],
)
async def test_review_requirement_findings_without_runtime_failure(
    tmp_path, candidate, status, approve, trigger
):
    """Scripted review results verify evidence transport and independent learning decisions."""
    original = [
        {"role": "system", "content": "Return only Python source code."},
        {
            "role": "user",
            "content": "Implement double(x) and annotate every parameter and return type.",
        },
    ]
    assert not any(
        word in json.dumps(original).lower() for word in ("conflict", "contradict", "incompatible")
    )
    calls, findings = [], []

    class Provider:
        async def invoke(self, request, emit):
            purpose = request.metadata["purpose"]
            calls.append(purpose)
            evidence = json.loads(request.messages[-1]["content"])
            assert evidence["original_task"]["messages"] == original
            assert evidence["in_task_opportunity"] == {
                "observed_failures": [],
                "unresolved_work": [],
            }
            results = [
                r
                for r in evidence["trajectory_evidence"]["records"]
                if r["type"] == "python_result"
            ]
            assert results and results[-1]["payload"]["error"] is None
            assert results[-1]["payload"]["stdout"].strip() == candidate
            if purpose == "refinement_review":
                prompt = request.messages[0]["content"]
                assert "Compare each relevant active requirement" in prompt
                assert "A violated requirement can still yield shouldRefine=false" in prompt
                findings.append(
                    {
                        "requirement": "Annotate every parameter and return type.",
                        "source_message_indices": [1],
                        "resolution": "Active: annotations are compatible with source-only output in message 0.",
                        "status": status,
                        "evidence": {
                            "event_id": results[-1]["id"],
                            "candidate_excerpt": candidate,
                            "explanation": "Parameter x and the return annotation are missing."
                            if status == "violated"
                            else "Parameter x and the return both have int annotations.",
                        },
                    }
                )
                return ModelResponse(
                    text=json.dumps(
                        {
                            "requirement_findings": findings,
                            "shouldRefine": approve,
                            "rationale": "Consider a reusable lesson."
                            if approve
                            else "No useful new reusable lesson.",
                            "instructions": "Consider the cited evidence." if approve else "",
                            "learning_assessment": {
                                "decision": "reusable" if approve else "no_new_evidence"
                            },
                        }
                    )
                )
            assert purpose == "refinement"
            assert evidence["reviewer_assessment"]["requirement_findings"] == findings
            # An approved review does not oblige the planner to save an edit.
            return ModelResponse(
                text=json.dumps(
                    {
                        "summary": "No novel reusable rule to persist.",
                        "edits": [],
                        "learning_assessment": {"decision": "already_known"},
                    }
                )
            )

    rt = Runtime(tmp_path / "state", providers={"mock": Provider()})
    try:
        root = rt.create(
            original[-1]["content"],
            tmp_path,
            config=RunConfig(
                provider={"name": "mock", "model": "deterministic"},
                task={"original_messages": original},
            ),
        )
        source = rt.store.event(root.id, "observation", {"note": "Candidate under consideration"})
        result = await rt.execute_python(
            ToolContext(rt, root.id, new_id(), source),
            f"candidate = {candidate!r}\nexec(candidate)\nassert double(3) == 6\nprint(candidate)",
        )
        assert result["error"] is None
        assert not rt.store.events(root.id, kind="python_error")
        # Evidence alone does not schedule a review.
        assert not await rt.refinement_checkpoint(root.id)
        assert not calls
        if trigger == "compact":
            rt.refinement_compacted(root.id)
        else:
            for _ in range(24):
                assert not await rt.refinement_checkpoint(root.id, completed_turn=True)
            assert not calls
            if trigger == "background_interval":
                rt.store.update(
                    root.id, pending_turn={"event_id": source, "context_committed": True}
                )
                rt.refinement_message_end(root.id)
            else:
                rt.count_refinement_turn(root.id)
        assert not await rt.refinement_checkpoint(root.id)
        assert calls == ["refinement_review"] + (["refinement"] if approve else [])
        review = rt.store.events(root.id, kind="refinement_review")[0]["payload"]
        assert review["requirement_findings"] == findings
        assert review["shouldRefine"] is approve
        assert findings[0]["status"] == status
        assert not rt.store.harness.entries(root.id)
        assert not rt.store.events(root.id, kind="refinement_notice")
        assert not rt.store.events(root.id, kind="refinement_continuation")
        assert not rt.store.events(root.id, kind="refine_failed")
        later = rt.refinement_input(root.id)["trajectory_evidence"]["records"]
        assert not any(r["type"] == "model_response" for r in later)
    finally:
        await rt.shutdown()
