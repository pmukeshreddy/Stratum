"""Opt-in transport smoke on an unmodified released EvoCode project.

This scripted provider proves plumbing, not task-solving ability or learning.
It does not request rlm/refine or modify any benchmark input/solution/verifier.
"""

import json
import os
from pathlib import Path

import pytest

from threadweave.models import Action, ModelResponse, RunConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("BUFFALO_EVOCODE_SMOKE_TASK"),
    reason="requires released EvoCode task and Docker",
)


async def test_released_task_persistent_runtime_and_hidden_feedback(tmp_path):
    pytest.importorskip("harbor")
    from threadweave.evals.evocode import run_task

    class TransportProbe:
        def __init__(self):
            self.calls = 0

        async def invoke(self, request, emit):
            assert not request.parent_id
            index = self.calls
            self.calls += 1
            if index in {0, 5}:
                # First round is 1 cell + final + 3 continuations; its untouched
                # official verifier always fails on the unimplemented pipeline.
                code = (
                    "transport_probe = {'value': 937}\n"
                    if index == 0
                    else "assert transport_probe['value'] == 937\n"
                )
                code += (
                    "from pathlib import Path\n"
                    "assert not Path('/tests/test.sh').exists()\n"
                    "assert not Path('/logs/verifier/reward.txt').exists()\n"
                    "print(transport_probe)\n"
                )
                return ModelResponse(actions=[Action(name="ipython", arguments={"code": code})])
            return ModelResponse(text="Transport probe candidate boundary.")

    provider = TransportProbe()
    config = RunConfig(
        provider={"name": "transport_probe", "model": "fixture"},
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        limits={"max_turns": 100, "wall_seconds": 3600, "token_budget": 3000000},
    )
    output = Path(os.environ.get("BUFFALO_EVOCODE_SMOKE_OUTPUT", tmp_path))
    result, directory = await run_task(
        Path(os.environ["BUFFALO_EVOCODE_SMOKE_TASK"]),
        output,
        config,
        providers={"transport_probe": provider},
        max_steps=2,
    )
    assert result.exception_info is None, result.exception_info
    assert not any(s.exception_info for s in result.step_results), result.step_results
    report = json.loads((directory / "report.json").read_text())
    assert report["rounds_reached"] == 2
    first, second = report["rounds"]
    assert first["root_session_id"] == second["root_session_id"]
    assert (
        first["persistence"]["after"]["kernel_pid"] == second["persistence"]["after"]["kernel_pid"]
    )
    assert second["persistence"]["before"]["previous_history_retained"]
    assert all(r["autonomous"]["continuations"] == 3 for r in report["rounds"])
    assert all(r["verifier_attempts"] == 4 for r in report["rounds"])
    for round_name in ("round-1", "round-2"):
        audit = json.loads((directory / f"{round_name}-audit.json").read_text())
        assert not any(e["type"] == "python_error" for e in audit["events"])
        assert any(e["type"] == "python_result" and "937" in json.dumps(e) for e in audit["events"])
    assert "Autonomous quality gate failed" in (directory / "model-calls.jsonl").read_text()
    assert json.loads((directory / "manifest.json").read_text())["official_inputs_unchanged"]
