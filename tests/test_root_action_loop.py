"""Controlled actions exercise the production root loop; live behavior is validated separately."""

import asyncio
import json
import sqlite3

import pytest

from threadweave.evals.harness import run_buffalo
from threadweave.evals.manyih_full_harness import experiment_config
from threadweave.gitops import git
from threadweave.models import Action, ModelResponse, Usage


def response(text="", code=None):
    return ModelResponse(
        text=text,
        actions=[Action(name="ipython", arguments={"code": code})] if code else [],
        usage=Usage(input_tokens=40, output_tokens=20),
    )


class ActionPeer:
    def __init__(self, scenario):
        self.scenario, self.requests = scenario, []
        self.child_started, self.parent_progress = asyncio.Event(), asyncio.Event()
        self.stage = 0

    async def resolve(self, config, **kwargs):
        return config, {}

    async def invoke(self, request, emit):
        self.requests.append(request)
        assert request.config.model == "gpt-6-astra"
        assert request.config.parameters["reasoning_effort"] == "xhigh"
        purpose = request.metadata.get("purpose", "agent")
        if purpose == "refinement_review":
            assert "trajectory" in request.messages[-1]["content"]
            return response(
                json.dumps(
                    {
                        "shouldRefine": False,
                        "rationale": "One-off checks do not establish a reusable improvement",
                    }
                )
            )
        assert purpose == "agent", purpose
        assert [t["function"]["name"] for t in request.tools] == ["ipython"]
        if request.parent_id:
            if request.turn == 0:
                self.child_started.set()
                await self.parent_progress.wait()
                values = [1, 2, 3] if request.name == "left" else [4, 5]
                return response(
                    code=f"values = {values!r}\nsubtotal = sum(v*v for v in values)\nassert task.instructions and task.assignment != task.instructions"
                )
            if request.turn == 1:
                return response(
                    code=f"assert subtotal == sum(v*v for v in values)\n(workspace / '{request.name}.json').write_text(json.dumps(subtotal))\nawait agent_message.send('{request.name} squared subtotal is ready in {request.name}.json',receiver_role='parent')"
                )
            return response("Independent subtotal verified and delivered")
        if self.scenario == "trivial":
            return response("42")
        if self.scenario == "failure":
            if request.turn == 0:
                return response("Complete")
            if request.turn == 1:
                verifier = next(
                    m
                    for m in request.messages
                    if str(m.get("content", "")).startswith("Verifier: ")
                )
                exposed = json.loads(verifier["content"].removeprefix("Verifier: "))
                assert json.loads(exposed["preview"])["passed"] is False
                return response(code="(workspace / 'answer.txt').write_text('42')")
            return response("The requested answer file contains 42")
        if self.scenario == "inspect":
            if request.turn == 0:
                return response(
                    code="numbers = json.loads((workspace / 'numbers.json').read_text())\ntotal = sum(numbers)\nprint(total)"
                )
            return response("15")
        if self.stage == 0:
            self.stage = 1
            return response(
                code="left = await rlm('Compute squared subtotal for batch [1,2,3]',name='left',isolate=False)\nright = await rlm('Compute squared subtotal for batch [4,5]',name='right',isolate=False)\nroot_note = 'combine independently computed squared subtotals'"
            )
        if self.stage == 1:
            await self.child_started.wait()
            self.parent_progress.set()
            self.stage = 2
            return response(
                code="expected_count = 5\nprint('Combining two independent batches covering',expected_count,'items')"
            )
        if self.stage == 2:
            if len(request.metadata["execution_inputs"]["child_evidence"]) < 2:
                return response(code="await agents.wait(seconds=0.05)")
            self.stage = 3
            return response(
                code="assert root_note and expected_count == 5\nleft_total = json.loads((workspace / 'left.json').read_text())\nright_total = json.loads((workspace / 'right.json').read_text())\ncombined = left_total + right_total\nprint(combined)\nassert (await agent_observe.recent_messages(left.session_id))['messages']"
            )
        assert "55" in json.dumps(request.messages)
        return response("55")


@pytest.mark.parametrize("scenario", ["trivial", "inspect", "parallel", "failure"])
async def test_root_actions_run_without_feature_completion_requirements(
    tmp_path, repository, monkeypatch, scenario
):
    peer = ActionPeer(scenario)
    monkeypatch.setattr(
        "threadweave.evals.harness.default_providers", lambda: {"codex_subscription": peer}
    )
    config = experiment_config()
    task = {
        "messages": [
            {
                "role": "user",
                "content": {
                    "trivial": "What is six times seven?",
                    "inspect": "Sum the integers in numbers.json.",
                    "parallel": "Combine independently computed squared subtotals of the disjoint batches [1,2,3] and [4,5].",
                    "failure": "Create answer.txt containing 42.",
                }[scenario],
            }
        ]
    }
    (repository / "numbers.json").write_text("[1,2,3,4,5]")
    git(repository, "add", "numbers.json")
    git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "-qm",
        "Task input",
    )
    if scenario == "failure":
        config.task.adapter = "workspace"
        config.task.verifier = "file"
        config.task.verifier_options = {"path": "answer.txt", "equals": "42"}
        config.task.require_verifier = True
        config.task.verify_each_turn = False
    directory = tmp_path / "run"
    directory.mkdir()
    result = await run_buffalo(
        config, task, directory, workspace=repository, task_config=config.task
    )
    assert result["stop_reason"] == "completed", result
    assert {r.metadata.get("purpose", "agent") for r in peer.requests} <= {
        "agent",
        "refinement_review",
        "refinement",
    }
    first = peer.requests[0]
    assert first.metadata.get("purpose", "agent") == "agent"
    foundation = first.messages[0]["content"]
    assert foundation.startswith("You are Buffalo, a code-using agent")
    markers = [
        "Recursive mechanics:",
        "Independent work:",
        "Available subagent specifications:",
        "Harness state:",
        "Skills and project context:",
    ]
    assert [foundation.index(m) for m in markers] == sorted(foundation.index(m) for m in markers)
    with sqlite3.connect(directory / "state/history.sqlite3") as db:
        counts = dict(db.execute("SELECT type,count(*) FROM events GROUP BY type"))
        if scenario == "trivial":
            assert counts.get("python_execution", 0) == counts.get("rlm_admitted", 0) == 0
        if scenario == "inspect":
            assert counts["python_result"] > 0 and counts.get("rlm_admitted", 0) == 0
        if scenario == "parallel":
            assert peer.parent_progress.is_set() and counts["rlm_admitted"] == 2
            assert counts["agent_message_received"] >= 2 and counts["child_observation"] >= 1
            assert not counts.get("python_error")
        if scenario == "failure":
            assert counts["verifier_result"] >= 2 and counts.get("rlm_admitted", 0) == 0
        assert counts["refinement_review"] >= 1
        assert db.execute("SELECT count(*) FROM state_entries").fetchone()[0] == 0


async def test_original_contract_and_preloaded_api_survive_child_restart(tmp_path, python_config):
    from threadweave.models import new_id
    from threadweave.runtime import Runtime
    from threadweave.tools import ToolContext

    messages = [
        {"role": "system", "content": "Keep exact integer values."},
        {"role": "developer", "content": "Report evidence in order."},
        {"role": "user", "content": "Earlier requirement."},
        {"role": "assistant", "content": "Acknowledged."},
    ]
    instruction = "Inspect the supplied values."
    python_config.task.instruction_messages = messages
    python_config.task.specification = {"units": "cents"}
    runtime = Runtime(tmp_path / "state", providers={"mock": object()})

    async def cell(sid, code):
        event = runtime.store.event(sid, "test_cell", {})
        result = await runtime.execute_python(ToolContext(runtime, sid, new_id(), event), code)
        assert not result.get("error"), result
        return result

    try:
        root = runtime.create(instruction, tmp_path, config=python_config)
        expected = messages + [{"role": "user", "content": instruction}]
        active = runtime.context.messages(root.id)
        start = active.index(messages[0])
        assert active[start : start + len(expected)] == expected
        await cell(
            root.id,
            f"""
import inspect
assert list(task.messages) == {expected!r}
assert task.instructions == {instruction!r}
assert task.assignment == {instruction!r}
assert task.workspace == workspace
assert task.context['specification'] == {{'units': 'cents'}}
assert callable(rlm) and inspect.iscoroutinefunction(rlm.__call__)
assert all(name in globals() for name in ('task','workspace','rlm','agent_message','agent_observe','harness','skills','tools','bash','compact','refine'))
assert rlm.harness is harness
entry = harness.create('memory', 'Observed units', 'Amounts are integer cents')
assert not inspect.isawaitable(entry)
assert not inspect.isawaitable(harness.list())
assert not inspect.isawaitable(skills.list())
assert harness.get('memory', entry.id).version == 1
try:
    harness.get('memory', 'missing')
except KeyError:
    pass
else:
    raise AssertionError('missing entry must raise KeyError')
child = await rlm('Check a distinct boundary.', name='checker')
assert child.session_id and child.name == 'checker'
root_value = 41
""",
        )
        child = next(s for s in runtime.store.sessions() if s.parent_id == root.id)
        child_config = runtime.store.config(child.id)
        assert child_config.provider == python_config.provider
        assert child_config.limits == python_config.limits
        assert child_config.features == python_config.features
        assert child_config.task.instruction_messages == messages
        assert child_config.task.specification == {"units": "cents"}
        await cell(
            child.id,
            f"""
assert list(task.messages) == {expected!r}
assert task.instructions == {instruction!r}
assert task.assignment.startswith('Check a distinct boundary.')
assert task.workspace == workspace
assert 'root_value' not in globals()
child_value = 73
""",
        )
        kernel_id = child.kernel_id
        await runtime.shutdown()
        runtime = Runtime(tmp_path / "state", providers={"mock": object()})
        await cell(
            child.id, f"assert child_value == 73\nassert list(task.messages) == {expected!r}"
        )
        await cell(root.id, "assert root_value == 41\nassert child.session_id")
        assert runtime.store.session(child.id).kernel_id == kernel_id
    finally:
        await runtime.shutdown()


async def test_corrected_failure_refines_real_evidence_into_next_invocation(
    tmp_path, python_config
):
    from threadweave.runtime import Runtime

    class FailurePeer:
        def __init__(self):
            self.requests = []

        async def invoke(self, request, emit):
            self.requests.append(request)
            purpose = request.metadata.get("purpose", "agent")
            if purpose != "agent":
                data = json.loads(request.messages[-1]["content"])
                assert data["original_task"]["messages"] == expected
            if purpose == "refinement_review":
                # Review the actual assertion failure; later completion needs no second lesson.
                reusable = request.turn == 1
                if reusable:
                    assert "AssertionError" in json.dumps(data["trajectory"])
                return response(
                    json.dumps(
                        {
                            "shouldRefine": reusable,
                            "rationale": "Exact decimal conversion must not truncate binary floating-point products."
                            if reusable
                            else "No new reusable evidence.",
                        }
                    )
                )
            if purpose == "refinement":
                sources = [e["id"] for e in data["evidence"] if e["type"] == "python_error"]
                assert sources
                return response(
                    json.dumps(
                        {
                            "proposals": [
                                {
                                    "kind": "prompt_note",
                                    "title": "Exact decimal cents",
                                    "content": {
                                        "text": "For decimal amount strings, use exact decimal arithmetic before converting to integer cents. Binary floating-point truncation failed for 0.29."
                                    },
                                    "source_events": sources,
                                    "intended_effect": "Avoid the observed rounding defect in later conversions.",
                                }
                            ]
                        }
                    )
                )
            if request.turn == 0:
                return response(code="assert int(float('0.29') * 100) == 29")
            if request.turn == 1:
                assert "Binary floating-point truncation failed for 0.29" in json.dumps(
                    request.messages
                )
                assert request.metadata["execution_inputs"]["harness_state"]
                return response(
                    code="from decimal import Decimal\nassert int(Decimal('0.29') * 100) == 29\nassert int(Decimal('-0.29') * 100) == -29\nprint('exact conversion verified')"
                )
            return response("Exact conversion verified for both signs.")

    python_config.refinement.automatic = True
    python_config.refinement.continuation_reserve_seconds = 0
    python_config.limits.wall_seconds = 300
    python_config.task.instruction_messages = [
        {"role": "system", "content": "Keep decimal amounts exact."},
        {"role": "developer", "content": "Check both signs."},
    ]
    expected = python_config.task.instruction_messages + [
        {"role": "user", "content": "Check conversion to integer cents."}
    ]
    provider = FailurePeer()
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create(expected[-1]["content"], tmp_path, config=python_config)
        for _ in range(3):
            await runtime._run_turn(root.id)
        assert runtime.store.session(root.id).outcome == "completed"
        assert len(runtime.store.events(root.id, kind="python_error")) == 1
        states = runtime.store.states(root.id)
        assert len(states) == 1 and states[0]["version"] == 1
        assert runtime.store.events(root.id, kind="state_retrieved")
        assert {r.metadata.get("purpose", "agent") for r in provider.requests} == {
            "agent",
            "refinement_review",
            "refinement",
        }
    finally:
        await runtime.shutdown()


async def test_evaluation_wrapper_preserves_original_contract_for_python_and_review(
    tmp_path, monkeypatch
):
    original = [
        {"role": "system", "content": "Use English words."},
        {"role": "developer", "content": "Use lowercase."},
        {"role": "user", "content": "What is two plus three?"},
    ]

    class Peer:
        async def resolve(self, config, **kwargs):
            return config, {}

        async def invoke(self, request, emit):
            if request.metadata.get("purpose") == "refinement_review":
                assert (
                    json.loads(request.messages[-1]["content"])["original_task"]["messages"]
                    == original
                )
                return response(
                    json.dumps({"shouldRefine": False, "rationale": "One-off arithmetic."})
                )
            if not request.turn:
                return response(
                    code=f"assert list(task.messages) == {original!r}\nassert task.instructions == 'What is two plus three?'\nprint('contract retained')"
                )
            return response("five")

    monkeypatch.setattr(
        "threadweave.evals.harness.default_providers", lambda: {"codex_subscription": Peer()}
    )
    config = experiment_config()
    result = await run_buffalo(config, {"messages": original}, tmp_path / "run")
    assert result["response"] == "five"


async def test_child_result_arriving_during_preparation_reaches_next_root_request(
    tmp_path, python_config
):
    from threadweave.runtime import Runtime

    entered, release = asyncio.Event(), asyncio.Event()

    class Peer:
        async def invoke(self, request, emit):
            if request.metadata.get("purpose") == "refinement_review":
                entered.set()
                await release.wait()
                return response(json.dumps({"shouldRefine": False, "rationale": "No lesson."}))
            if request.parent_id:
                return response("The boundary case returns 17.")
            if request.turn == 0:
                return response(code="child = await rlm('Check the boundary.', name='reviewer')")
            assert "The boundary case returns 17." in json.dumps(request.messages)
            assert request.metadata["execution_inputs"]["child_evidence"]
            return response("Observed boundary result: 17.")

    python_config.refinement.automatic = False
    python_config.refinement.on_completion = False
    python_config.refinement.continuation_reserve_seconds = 0
    python_config.limits.wall_seconds = 300
    runtime = Runtime(tmp_path / "state", providers={"mock": Peer()})
    compact = runtime.semantic_compact

    async def delayed_compact(sid, **kwargs):
        if not runtime.store.session(sid).parent_id and runtime.store.session(sid).turns == 1:
            entered.set()
            await release.wait()
        return await compact(sid, **kwargs)

    runtime.semantic_compact = delayed_compact
    turn = None
    try:
        root = runtime.create("Check boundary behavior.", tmp_path, config=python_config)
        await runtime._run_turn(root.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == root.id)
        turn = asyncio.create_task(runtime._run_turn(root.id))
        await asyncio.wait_for(entered.wait(), 5)
        await runtime._run_turn(child.id)
        assert runtime.store.messages(root.id, pending=True)
        release.set()
        await turn
        assert runtime.store.session(root.id).outcome == "completed"
        assert runtime.store.session(root.id).turns == 2
        assert not runtime.store.messages(root.id, pending=True)
    finally:
        release.set()
        if turn:
            await turn
        await runtime.shutdown()


async def test_background_review_overlaps_root_and_applies_only_between_turns(
    tmp_path, python_config
):
    from threadweave.runtime import Runtime

    reviewing, root_running, release_root = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []

    class Peer:
        async def invoke(self, request, emit):
            requests.append(request)
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement_review":
                reviewing.set()
                await root_running.wait()
                return response(
                    json.dumps(
                        {"shouldRefine": True, "rationale": "A reusable check was established."}
                    )
                )
            if purpose == "refinement":
                evidence = json.loads(request.messages[-1]["content"])
                assert (
                    evidence["original_task"]["messages"][-1]["content"]
                    == "Check and retain the result."
                )
                return response(
                    json.dumps(
                        {
                            "proposals": [
                                {
                                    "kind": "prompt_note",
                                    "title": "Verified check",
                                    "content": {"text": "PERSISTED_VERIFIED_CHECK"},
                                    "source_events": [evidence["evidence"][0]["id"]],
                                    "intended_effect": "Reuse the verified check.",
                                }
                            ]
                        }
                    )
                )
            if request.turn == 0:
                return response(code="verified = 17\nassert verified == 17")
            if request.turn == 1:
                assert reviewing.is_set()
                root_running.set()
                await release_root.wait()
                return response(code="assert verified == 17")
            assert "PERSISTED_VERIFIED_CHECK" in json.dumps(request.messages)
            return response("Verified 17.")

    python_config.refinement.automatic = True
    python_config.refinement.every_turns = 1
    python_config.refinement.on_completion = False
    python_config.refinement.continuation_reserve_seconds = 0
    python_config.limits.wall_seconds = 300
    runtime = Runtime(tmp_path / "state", providers={"mock": Peer()})
    try:
        root = runtime.create("Check and retain the result.", tmp_path, config=python_config)
        await runtime.start()
        await asyncio.wait_for(root_running.wait(), 5)
        from .conftest import eventually

        await eventually(
            lambda: runtime.store.db.execute(
                "SELECT 1 FROM refinement_runs WHERE status='waiting_to_apply'"
            ).fetchone()
        )
        assert not runtime.store.states(root.id), "An in-flight root must keep its context version."
        assert len([r for r in requests if r.metadata.get("purpose") == "refinement_review"]) == 1
        release_root.set()
        result = await runtime.wait(root.id, timeout=10)
        assert result.outcome == "completed" and result.turns == 3
        assert runtime.store.states(root.id)[0]["version"] == 1
        assert runtime.store.events(root.id, kind="refinement_scheduled")
    finally:
        release_root.set()
        await runtime.shutdown()


async def test_completed_evaluation_settles_background_review_usage(
    tmp_path, repository, monkeypatch
):
    reviewing, cancelled = asyncio.Event(), asyncio.Event()

    class Peer:
        async def resolve(self, config, **kwargs):
            return config, {}

        async def invoke(self, request, emit):
            if request.metadata.get("purpose") == "refinement_review":
                reviewing.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if request.turn == 0:
                return response(code="checked = 17\nassert checked == 17")
            assert reviewing.is_set()
            return response("17")

    monkeypatch.setattr(
        "threadweave.evals.harness.default_providers", lambda: {"codex_subscription": Peer()}
    )
    config = experiment_config()
    config.refinement.every_turns = 1
    directory = tmp_path / "run"
    directory.mkdir()
    result = await run_buffalo(
        config,
        {"messages": [{"role": "user", "content": "Check 17."}]},
        directory,
        workspace=repository,
        task_config=config.task,
    )
    assert result["stop_reason"] == "completed" and result["response"] == "17"
    assert cancelled.is_set()
    with sqlite3.connect(directory / "state/history.sqlite3") as db:
        assert not db.execute("SELECT 1 FROM model_attempts WHERE status='running'").fetchone()
        assert not db.execute("SELECT 1 FROM reservations").fetchone()
        assert (
            db.execute("SELECT count(*) FROM model_requests").fetchone()[0]
            == result["usage"]["model_calls"]
            == 3
        )
        assert result["usage"]["estimated_calls"] == 1
        saved = json.loads((directory / "usage.json").read_text())
        assert saved["total_tokens"] == result["usage"]["total_tokens"]


async def test_late_child_findings_explain_that_complete_candidate_was_withheld(
    tmp_path, python_config
):
    from threadweave.runtime import Runtime

    from .conftest import eventually

    candidate_started = asyncio.Event()

    class Peer:
        async def invoke(self, request, emit):
            if request.parent_id:
                await candidate_started.wait()
                return response("The edge case also requires rejecting bool.")
            if request.turn == 0:
                return response(code="reviewer = await rlm('Check the edge case', name='reviewer')")
            if request.turn == 1:
                candidate_started.set()
                await eventually(
                    lambda: any(
                        s.parent_id and s.outcome == "completed" for s in runtime.store.sessions()
                    )
                )
                return response("Detailed candidate: validate all integer amounts. Observed 17.")
            text = json.dumps(request.messages)
            assert "has not been returned" in text and "complete, self-contained answer" in text
            assert "rejecting bool" in text and "Detailed candidate" in text
            return response(
                "Complete report: observed 17; validate integer amounts and reject bool."
            )

    python_config.refinement.automatic = False
    runtime = Runtime(tmp_path / "state", providers={"mock": Peer()})
    try:
        root = runtime.create(
            "Report the validation rule and observed result.", tmp_path, config=python_config
        )
        await runtime.start()
        result = await runtime.wait(root.id, timeout=10)
        assert result.outcome == "completed" and result.turns == 3
        assert result.result.startswith("Complete report: observed 17")
        assert len(runtime.store.events(root.id, kind="completion_continuation")) == 1
    finally:
        await runtime.shutdown()
