import json

import pytest

from threadweave.evals.manyih_reinforcement import INTERVAL_ALGORITHMS, scenario
from threadweave.evals.reinforcement_metrics import lifecycle
from threadweave.models import ModelResponse
from threadweave.runtime import Runtime

from .test_continual_harness import input_sections


@pytest.mark.parametrize("decision", ["decline", "empty", "edit", "failure"])
async def test_lifecycle_reports_real_boundaries_without_mutation(
    tmp_path, python_config, decision
):
    class Provider:
        async def invoke(self, request, emit):
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement_review":
                assert (
                    input_sections(request)["trigger"]
                    == "turn_interval; 25 assistant turns since last auto-refine review"
                )
                if decision == "failure":
                    return ModelResponse(text="not JSON")
                return ModelResponse(
                    text=json.dumps(
                        {"shouldRefine": decision != "decline", "rationale": "observed"}
                    )
                )
            if purpose == "refinement":
                edits = (
                    [{"action": "create", "kind": "memory", "title": "test", "content": "lesson"}]
                    if decision == "edit"
                    else []
                )
                return ModelResponse(
                    text=json.dumps(
                        {
                            "summary": "audit",
                            "rationale": "observed",
                            "expectedOutcome": "reuse",
                            "edits": edits,
                        }
                    )
                )
            return ModelResponse(text=str(request.turn**2))

    runtime = Runtime(tmp_path / "state", providers={"mock": Provider()})
    session = runtime.create("Calculate the first square", tmp_path, config=python_config)
    try:
        await runtime.start()
        for i in range(25):
            if i:
                runtime.interact(session.id, f"Calculate square {i}")
            await runtime.wait(session.id)
            if i == 23:
                before = lifecycle(tmp_path)
                assert before["root_assistant_turns"] == 24
                assert before["outcomes"] == ["NO_TRIGGER"]
                assert not before["interval_threshold_reached"]
        events = list(runtime.store.iter_events(session.id))
        result = lifecycle(tmp_path)
        assert list(runtime.store.iter_events(session.id)) == events
        assert result["root_assistant_turns"] == 25
        assert result["interval_threshold_reached"]
        assert result["interval_triggers"] == 1
        assert result["automatic_reviews"][0]["root_turn"] == 25
        assert result["planner_calls"] == int(decision in {"empty", "edit"})
        assert result["typed_edits"] == int(decision == "edit")
        assert result["outcomes"] == [
            {
                "decline": "TRIGGER_REVIEW_DECLINED",
                "empty": "TRIGGER_APPROVED_EMPTY_PLAN",
                "edit": "TRIGGER_APPLIED_EDIT",
                "failure": "REFINEMENT_FAILED",
            }[decision]
        ]
    finally:
        await runtime.shutdown()


def test_evaluation_has_no_trigger_overrides_or_refinement_commands(tmp_path):
    for tid in {0, *INTERVAL_ALGORITHMS}:
        path = tmp_path / str(tid)
        path.mkdir()
        (path / "task.json").write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "Python programmer"},
                        {"role": "user", "content": f"Implement algorithm {tid}"},
                    ]
                }
            )
        )
    plan = scenario(tmp_path, "interval")
    assert plan["policy"] == {
        "enabled": True,
        "turn_interval": 25,
        "compact": True,
        "cooldown_seconds": 1200,
    }
    assert len(plan["stages"]) == 2 * len(INTERVAL_ALGORITHMS)
    assert not plan["canonical_score"]
    for stage in plan["stages"]:
        assert "refine" not in stage["prompt"]
        assert "compaction" not in stage["prompt"]
