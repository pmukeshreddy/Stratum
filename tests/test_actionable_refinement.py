"""Evidence-triggered refinement must change unfinished work, without a final-answer ritual."""

import json

import pytest

from threadweave.models import Action, ModelResponse, RunConfig
from threadweave.runtime import Runtime

LESSON = (
    "Check the actual failing input before committing to a correction. "
    "Keep the candidate and the expected result separately so a test cannot silently redefine "
    "the task. The observed failure is a missing negative-input branch: return abs(n), then "
    "validate negative, zero and positive inputs against their specified magnitudes."
)


@pytest.mark.parametrize("boundary", ["failed_execution", "pre_completion"])
async def test_refinement_changes_work_before_completion(tmp_path, boundary):
    purposes = []

    class Provider:
        async def invoke(self, request, emit):
            purpose = request.metadata.get("purpose", "agent")
            purposes.append(purpose)
            if purpose == "refinement_review":
                evidence = json.loads(request.messages[-1]["content"])
                assert any(evidence["in_task_opportunity"].values())
                return ModelResponse(
                    text=json.dumps(
                        {
                            "shouldRefine": True,
                            "rationale": "Negative inputs remain incorrect",
                            "instructions": "Correct the magnitude procedure before completion",
                        }
                    )
                )
            if purpose == "refinement":
                return ModelResponse(
                    text=json.dumps(
                        {
                            "summary": "Correct negative inputs before delivering",
                            "application": {
                                "status": "actionable",
                                "issue": "negative input",
                                "next_action": "Correct the candidate and check both signs",
                                "validation": "Check -3, 0, 3",
                            },
                            "edits": [
                                {
                                    "action": "create",
                                    "kind": "prompt",
                                    "id": "magnitude",
                                    "title": "Magnitude validation",
                                    "content": LESSON,
                                }
                            ],
                        }
                    )
                )
            if request.turn == 0:
                if boundary == "pre_completion":
                    return ModelResponse(text="def magnitude(n): return n")
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "candidate = 'def magnitude(n): return n'\nexec(candidate)\nassert magnitude(-3) == 3"
                            },
                        )
                    ]
                )
            if request.turn == 1:
                assert LESSON in "\n".join(str(m.get("content", "")) for m in request.messages)
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "candidate = 'def magnitude(n): return abs(n)'\nexec(candidate)\nassert [magnitude(n) for n in (-3, 0, 3)] == [3, 0, 3]\nprint(candidate)"
                            },
                        )
                    ]
                )
            return ModelResponse(text="def magnitude(n): return abs(n)")

    config = RunConfig(
        provider={"name": "mock", "model": "deterministic"}, limits={"wall_seconds": 30}
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": Provider()})
    try:
        root = runtime.create(
            "Implement magnitude and check negative inputs.", tmp_path, config=config
        )
        if boundary == "pre_completion":
            runtime.store.event(
                root.id,
                "semantic_state_updated",
                {
                    "id": "sign",
                    "kind": "decision",
                    "status": "open",
                    "text": "The candidate returns negative values unchanged; the required result is a magnitude.",
                },
            )
        await runtime.start()
        await runtime.wait(root.id, timeout=25)
        assert runtime.store.session(root.id).outcome == "completed"
        events = list(runtime.store.iter_events(root.id))
        complete = next(e for e in events if e["type"] == "refine_complete")
        notice = next(e for e in events if e["type"] == "refinement_notice")
        later = [e for e in events if e["type"] == "python_result" and e["seq"] > notice["seq"]]
        assert later and later[-1]["payload"]["result"]["error"] is None
        assert complete["payload"]["application"]["issue"] == "negative input"
        assert notice["payload"]["expanded"] and LESSON in notice["payload"]["content"]
        assert purposes.count("refinement") == purposes.count("refinement_review") == 1
        assert not any(e["type"] == "refine_failed" for e in events)
        if boundary == "pre_completion":
            attempt = next(e for e in events if e["type"] == "completion_attempt")
            assert attempt["seq"] < complete["seq"]
    finally:
        await runtime.shutdown()


async def test_clean_completion_has_no_refinement_ritual(tmp_path):
    class Provider:
        async def invoke(self, request, emit):
            assert request.metadata.get("purpose", "agent") == "agent"
            return ModelResponse(text="Done.")

    runtime = Runtime(tmp_path / "state", providers={"mock": Provider()})
    try:
        root = runtime.create(
            "Acknowledge.",
            tmp_path,
            config=RunConfig(provider={"name": "mock", "model": "deterministic"}),
        )
        await runtime.start()
        await runtime.wait(root.id, timeout=10)
        assert runtime.store.session(root.id).outcome == "completed"
        assert not runtime.store.events(root.id, kind="refinement_review")
    finally:
        await runtime.shutdown()


async def test_competing_requirements_review_can_decline_without_extra_work(tmp_path):
    purposes = []

    class Provider:
        async def invoke(self, request, emit):
            purpose = request.metadata.get("purpose", "agent")
            purposes.append(purpose)
            if purpose == "refinement_review":
                evidence = json.loads(request.messages[-1]["content"])
                assert evidence["in_task_opportunity"]["instruction_decision"]
                return ModelResponse(
                    text=json.dumps(
                        {
                            "shouldRefine": False,
                            "rationale": "The interpretation is consistent and checked.",
                        }
                    )
                )
            assert purpose == "agent"
            if request.turn == 0:
                return ModelResponse(
                    actions=[Action(name="ipython", arguments={"code": "assert 2 + 3 == 5"})]
                )
            return ModelResponse(text="5")

    runtime = Runtime(tmp_path / "state", providers={"mock": Provider()})
    try:
        root = runtime.create(
            "Resolve conflicting requirements by the stated authority. Compute 2 + 3.",
            tmp_path,
            config=RunConfig(provider={"name": "mock", "model": "deterministic"}),
        )
        await runtime.start()
        await runtime.wait(root.id, timeout=10)
        assert runtime.store.session(root.id).outcome == "completed"
        assert purposes == ["agent", "refinement_review", "agent"]
        assert not runtime.store.events(root.id, kind="refine_complete")
    finally:
        await runtime.shutdown()


def test_new_candidate_is_fresh_evidence_but_repeated_candidate_is_not():
    from threadweave.refinement_evidence import opportunity, opportunity_key

    class Store:
        def __init__(self):
            self.rows = []

        def iter_events(self, sid, kind=None):
            return iter([r for r in self.rows if not kind or r["type"] == kind])

    store = Store()
    messages = [{"role": "user", "content": "Resolve the conflicting width requirements."}]

    def record(code):
        store.rows.append(
            {"id": str(len(store.rows)), "type": "python_execution", "payload": {"code": code}}
        )
        return opportunity_key(opportunity(store, "root", messages))

    first = record("print('Inspecting the requirements')")
    candidate = record("draft = 'def width(): return 1'")
    assert candidate != first
    assert record("draft = 'def width(): return 1'") == candidate
    assert record("draft = 'def width(): return 2'") != candidate
