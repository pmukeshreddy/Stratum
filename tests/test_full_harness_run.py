"""No live inference: exercise the frozen 300-second configuration through real runtime turns."""

import asyncio
import json

import pytest

from threadweave.evals.manyih_full_harness import experiment_config as production_config
from threadweave.models import Action, ModelResponse, now
from threadweave.runtime import Runtime


def experiment_config():
    config = production_config()
    return config


@pytest.mark.parametrize("remaining,admitted", [(280, True), (120, True), (80, False), (20, False)])
async def test_bounded_refinement_admission(tmp_path, remaining, admitted):
    config = experiment_config()
    config.provider.name = "mock"
    runtime = Runtime(tmp_path / "state", providers={"mock": object()})
    try:
        root = runtime.create("task", tmp_path, config=config)
        runtime.store.update(root.id, started_at=now() - (300 - remaining))
        assert runtime.auxiliary_admitted(root.id, "refinement") is admitted
        assert config.limits.tool_timeout_seconds == 600
        allowance, estimate, reserve, available = runtime.refinement_allowance(
            root.id, "refinement_review"
        )
        assert allowance <= max(0, available - reserve - 5)
        if admitted:
            assert allowance >= estimate
    finally:
        await runtime.shutdown()


async def test_slow_review_leaves_planner_admissible_when_real_wall_time_remains(
    tmp_path, monkeypatch
):
    config = experiment_config()
    config.provider.name = "mock"
    runtime = Runtime(tmp_path / "state", providers={"mock": object()})
    try:
        root = runtime.create("Repair and verify a component", tmp_path, config=config)
        runtime.store.update(root.id, started_at=now())
        monkeypatch.setattr(runtime, "_elapsed", lambda root_id: 117)
        allowance = runtime.refinement_allowance(root.id, "refinement_review")[0]
        monkeypatch.setattr(runtime, "_elapsed", lambda root_id: 204)
        runtime._automatic_refinement_deadlines = {
            root.id: asyncio.get_running_loop().time() + allowance - 87
        }
        assert runtime.auxiliary_admitted(root.id, "refinement")
        remaining_pass, _, reserve, remaining_wall = runtime.refinement_allowance(
            root.id, "refinement"
        )
        assert remaining_pass <= remaining_wall - reserve - 5
    finally:
        await runtime.shutdown()


async def test_refinement_reserves_slow_root_work_after_short_inspection_calls(
    tmp_path, monkeypatch
):
    config = experiment_config()
    config.provider.name = "mock"
    runtime = Runtime(tmp_path / "state", providers={"mock": object()})
    try:
        root = runtime.create("Investigate and report observed findings", tmp_path, config=config)
        runtime.store.update(root.id, started_at=now())
        timestamp = 1000
        monkeypatch.setattr("threadweave.storage.now", lambda: timestamp)
        for duration in (2, 3, 55, 4, 5):
            start = runtime.store.event(root.id, "model_request", {"purpose": "agent"})
            timestamp += duration
            runtime.store.event(root.id, "model_response", {}, parent=start)
        monkeypatch.setattr(runtime, "_elapsed", lambda root_id: 180)
        allowance, _, reserve, remaining = runtime.refinement_allowance(
            root.id, "refinement_review"
        )
        assert reserve >= 2 * 55
        assert allowance == 0 and remaining == 120
        assert not runtime.auxiliary_admitted(root.id, "refinement_review")
        # More actual time admits optional learning; no task or action label is involved.
        monkeypatch.setattr(runtime, "_elapsed", lambda root_id: 50)
        assert runtime.auxiliary_admitted(root.id, "refinement_review")
    finally:
        await runtime.shutdown()


class CompletionProvider:
    def __init__(self, decision=True, slow=False):
        self.requests = []
        self.decision, self.slow = decision, slow

    async def invoke(self, request, emit):
        self.requests.append(request)
        assert request.config.model == "gpt-6-astra"
        assert request.config.parameters["reasoning_effort"] == "xhigh"
        assert request.reasoning_mode == "inherit"
        purpose = request.metadata.get("purpose", "agent")
        if purpose == "refinement_review":
            if self.slow:
                await asyncio.sleep(2)
            return ModelResponse(
                text=json.dumps(
                    {
                        "shouldRefine": self.decision,
                        "rationale": "Visible reusable evidence",
                        "instructions": "Retain the task observation",
                    }
                )
            )
        if purpose == "refinement":
            context = json.loads(request.messages[-1]["content"])
            return ModelResponse(
                text=json.dumps(
                    {
                        "proposals": [
                            {
                                "kind": "prompt_note",
                                "title": "Task observation",
                                "content": {"text": "RETRIEVED_LESSON"},
                                "source_events": [context["evidence"][0]["id"]],
                                "intended_effect": "Inform the next turn",
                                "select": True,
                            }
                        ]
                    }
                )
            )
        assert purpose == "agent"
        if request.turn:
            assert "RETRIEVED_LESSON" in json.dumps(request.messages)
            return ModelResponse(text="reconsidered answer")
        return ModelResponse(text="initial candidate")


@pytest.mark.parametrize("decision", [False, True])
async def test_completion_edits_reach_later_turn_without_forcing_refinement(
    tmp_path, repository, decision
):
    config = experiment_config()
    config.provider.name = "mock"
    config.control_plane = "python"
    provider = CompletionProvider(decision)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("task observation", repository, config=config)
        await runtime._run_turn(root.id)
        assert runtime.store.session(root.id).outcome == ("active" if decision else "completed")
        if decision:
            provider.decision = False
            await runtime._run_turn(root.id)
            assert runtime.store.session(root.id).result == "reconsidered answer"
            assert len(runtime.store.events(root.id, kind="refinement_continuation")) == 1
            assert runtime.store.events(root.id, kind="state_retrieved")
        else:
            assert runtime.store.session(root.id).result == "initial candidate"
            assert not runtime.store.states(root.id)
    finally:
        await runtime.shutdown()


async def test_review_and_planner_receive_complete_original_message_contract(tmp_path, repository):
    messages = [
        {"role": "system", "content": "Express quantities using English words."},
        {"role": "developer", "content": "Use lowercase without punctuation."},
    ]

    class ContractProvider(CompletionProvider):
        async def invoke(self, request, emit):
            if request.metadata.get("purpose") in {"refinement", "refinement_review"}:
                original = json.loads(request.messages[-1]["content"])["original_task"]
                assert original["messages"] == messages + [
                    {"role": "user", "content": "Return the number five as a digit."}
                ]
                assert original["current_assignment"] == "Return the number five as a digit."
            return await super().invoke(request, emit)

    config = experiment_config()
    config.provider.name = "mock"
    config.task.instruction_messages = messages
    provider = ContractProvider()
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("Return the number five as a digit.", repository, config=config)
        await runtime._run_turn(root.id)
        assert {r.metadata.get("purpose") for r in provider.requests} >= {
            "refinement_review",
            "refinement",
        }
        assert runtime.store.states(root.id)
    finally:
        await runtime.shutdown()


async def test_refinement_timeout_preserves_candidate_and_cleans_requests(tmp_path, repository):
    config = experiment_config()
    config.provider.name = "mock"
    config.control_plane = "python"
    config.refinement.automatic_budget_seconds = 0.05
    provider = CompletionProvider(slow=True)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("task", repository, config=config)
        await runtime._run_turn(root.id)
        assert runtime.store.session(root.id).result == "initial candidate"
        assert runtime.store.events(root.id, kind="refinement_budget_exhausted")
        assert not runtime.store.db.execute(
            "SELECT 1 FROM model_attempts WHERE status='running'"
        ).fetchone()
        assert not runtime.store.db.execute("SELECT 1 FROM reservations").fetchone()
    finally:
        await runtime.shutdown()


async def test_normal_ipython_path_reaches_persistence_state_skills_and_recursive_agents(
    tmp_path, repository
):
    from .fakes import ScriptedProvider

    def cell(code):
        return ModelResponse(actions=[Action(name="ipython", arguments={"code": code})])

    config = experiment_config()
    config.provider.name = "mock"
    config.control_plane = "python"
    config.refinement.automatic = False  # Automatic path exercised independently above.
    config.features.model_compaction = False
    provider = ScriptedProvider(
        {
            "root": [
                cell(
                    "x = 41\n"
                    "memory = harness.create('memory', 'Fact', 'x is 41', select=True)\n"
                    "note = harness.create('prompt_note', 'Check', 'Use observed facts', select=True)\n"
                    "spec = harness.create('subagent_spec', 'Worker', {'instruction':'Inspect the visible task'})\n"
                    "skill = harness.create('skill', 'Increment', {'name':'increment', 'description':'Add one', 'code':'skill_result = x + 1', 'required_permissions':['python']})\n"
                    "child = await rlm(harness.get('subagent_spec', spec.id).content['instruction'], name='child', isolate=False)\n"
                    "assert len(harness.list()) == 4\n"
                ),
                cell(
                    "assert x == 41\nassert await skills.run('increment') == 42\nassert harness.get('memory', memory.id).content['text'] == 'x is 41'\nprint('PERSISTENCE_OK')"
                ),
                ModelResponse(text="done"),
            ],
            "child": [
                cell(
                    "grandchild = await rlm('Return evidence', name='grandchild', isolate=False)\nawait agent_message.send('child evidence', receiver_role='parent')"
                ),
                ModelResponse(text="child done"),
            ],
            "grandchild": [
                cell("await agent_message.send('recursive evidence', receiver_role='parent')"),
                ModelResponse(text="grandchild done"),
            ],
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("task", repository, config=config)
        await runtime._run_turn(root.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == root.id)
        await runtime._run_turn(child.id)
        grandchild = next(s for s in runtime.store.sessions() if s.parent_id == child.id)
        await runtime._run_turn(grandchild.id)
        await runtime._run_turn(root.id)
        assert grandchild.depth == 2
        assert runtime.store.events(root.id, kind="agent_message_received")
        assert "PERSISTENCE_OK" in json.dumps(runtime.store.events(root.id, kind="python_result"))
        assert runtime.store.events(root.id, kind="skill_outcome")
        for session in (root, child, grandchild):
            assert (
                runtime.store.config(session.id).provider.parameters["reasoning_effort"] == "xhigh"
            )
        runtime.context.compact(root.id, summary="Retained task evidence")
        from threadweave.models import new_id
        from threadweave.tools import ToolContext

        event = runtime.store.event(root.id, "test", {})
        result = await runtime.execute_python(
            ToolContext(runtime, root.id, new_id(), event),
            "assert x == 41\nprint('AFTER_COMPACTION')",
        )
        assert "AFTER_COMPACTION" in json.dumps(result)
    finally:
        await runtime.shutdown()
