"""Learning assessment admission and the existing selective full-content path."""

import json

import pytest

from threadweave.models import ModelResponse, Outcome
from threadweave.runtime import Runtime

from .conftest import response
from .test_continual_harness import edit, input_sections, learning_assessment, proposal
from .test_continual_harness import harness_runtime as harness_runtime


@pytest.mark.parametrize(
    "content,earlier,assessment",
    [
        (
            "Parser finished; unit tests passed; worker A owns parser.py.",
            "Parser finished. Tests passed. Worker A owns parser.py.",
            learning_assessment(progressOnly=True, redundant=True, wouldActSame=True),
        ),
        (
            "Use a fresh cache when the compiler cache is corrupt.",
            "I already cleared the corrupt cache and added fresh-cache validation to my plan.",
            learning_assessment(redundant=True, wouldActSame=True),
        ),
        (
            "Use the existing validator before completion.",
            "My active plan already ends with running the validator.",
            learning_assessment(wouldActSame=True),
        ),
    ],
)
async def test_assessed_progress_and_preexisting_behavior_produce_no_edit(
    harness_runtime, content, earlier, assessment
):
    rt, sid, provider = harness_runtime
    event = rt.store.event(sid, "observation", {})
    rt.store.add_context(sid, event, [{"role": "assistant", "content": earlier}])
    assessment["priorEvidence"] = earlier
    provider.proposal = proposal(edit(content=content, metadata={"learningAssessment": assessment}))
    result = await rt.refine(sid)
    assert earlier in input_sections(provider.requests[0])["conversation"]
    assert result["appliedEdits"] == []
    assert not rt.store.harness.entries(sid)
    assert not rt.store.events(sid, kind="refinement_notice")
    plan = rt.store.events(sid, kind="refinement_plan")[0]["payload"]
    assert plan["status"] == "APPROVED_EMPTY_PLAN"
    assert plan["assessments"][0]["status"] == "REJECTED_REDUNDANT"
    assert plan["assessments"][0]["learningAssessment"]["priorEvidence"] == earlier
    # Explicit planning still bypasses the automatic reviewer.
    assert [r.metadata["purpose"] for r in provider.requests] == ["refinement"]


async def test_novel_procedure_persists_but_unassessed_candidate_does_not(harness_runtime):
    rt, sid, provider = harness_runtime
    candidate = edit("skill")
    provider.proposal = proposal(candidate, edit(id="unsupported", metadata={}))
    rt.refinement_compacted(sid)
    assert await rt.refinement_checkpoint(sid)
    assert [r.metadata["purpose"] for r in provider.requests] == [
        "refinement_review",
        "refinement",
    ]
    assert input_sections(provider.requests[0])["trigger"].startswith("compact;")
    entry = rt.store.harness.get(sid, "skill", "lesson")
    assert entry["version"] == 1
    for key in ("content", "metadata", "reference", "arguments"):
        assert entry[key] == candidate[key]
    persisted = json.loads((rt.store.harness.path(sid) / "harness_state.json").read_text())
    assert persisted["entries"]["skill"]["lesson"] == entry
    assert not rt.store.harness.get(sid, "memory", "unsupported")
    decisions = rt.store.events(sid, kind="refinement_plan")[0]["payload"]["assessments"]
    assert [d["status"] for d in decisions] == ["ACCEPTED", "REJECTED_UNSUPPORTED"]
    assert len(rt.store.events(sid, kind="refinement_notice")) == 1


async def test_full_entry_retrieval_reaches_next_root_request(tmp_path, python_config):
    tail = "FULL_ENTRY_TAIL: Compare both schema versions before committing a migration."
    content = "Detailed procedure and applicability conditions. " * 40 + tail
    requests = []

    class Provider:
        async def invoke(self, request, emit):
            requests.append(request)
            if request.metadata.get("purpose") == "refinement":
                return ModelResponse(text=json.dumps(proposal(edit(content=content))))
            assert request.metadata.get("purpose", "agent") == "agent"
            if request.turn == 0:
                return response("ipython", code="print(await refine.run())")
            if request.turn == 1:
                # The digest/notice intentionally contains only a preview.
                assert tail not in json.dumps(request.messages)
                assert "lesson" in json.dumps(request.messages)
                return response("ipython", code="print(harness.get('memory', 'lesson').content)")
            assert tail in json.dumps(request.messages)
            assert tail not in request.messages[0]["content"]
            return ModelResponse(text="Full procedure retrieved.")

    rt = Runtime(tmp_path / "state", providers={"mock": Provider()})
    root = rt.create("Exercise selective harness retrieval.", tmp_path, config=python_config)
    try:
        await rt.start()
        await rt.wait(root.id, timeout=25)
        assert rt.store.session(root.id).outcome == Outcome.COMPLETED
        assert len([r for r in requests if r.metadata.get("purpose", "agent") == "agent"]) == 3
        assert rt.store.harness.get(root.id, "memory", "lesson")["content"] == content
    finally:
        await rt.shutdown()
