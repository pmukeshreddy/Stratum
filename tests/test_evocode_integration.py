"""Contract integration tests; scripted model choices are fixtures, not benchmark policy."""

import asyncio
import json
import subprocess
from dataclasses import asdict

import pytest

from threadweave.autonomous import (
    AutonomousCycle,
    AutonomousPolicy,
    GateResult,
    bounded_output,
    git_snapshot,
)
from threadweave.evals.evocode_worker import PersistentWorker
from threadweave.models import Action, ModelResponse, RunConfig, Usage

from .test_continual_harness import learning_assessment


def cell(code):
    return ModelResponse(actions=[Action(name="ipython", arguments={"code": code})])


def initialize_git(path):
    path.mkdir(exist_ok=True)
    for args in (
        ["init", "-q"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
    ):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


async def test_prime_snapshots_content_exclusions_and_untracked_symlinks(tmp_path):
    initialize_git(tmp_path)
    before = await git_snapshot(tmp_path)
    (tmp_path / "Cargo.lock").write_text("ignored")
    assert await git_snapshot(tmp_path) == before
    (tmp_path / "new.py").write_text("a")
    changed = await git_snapshot(tmp_path)
    (tmp_path / "new.py").write_text("b")
    assert await git_snapshot(tmp_path) != changed
    (tmp_path / "link").symlink_to("one")
    linked = await git_snapshot(tmp_path)
    (tmp_path / "link").unlink()
    (tmp_path / "link").symlink_to("two")
    assert await git_snapshot(tmp_path) != linked


async def test_sequential_gates_unchanged_candidate_and_retry_exhaustion(tmp_path):
    initialize_git(tmp_path)
    cycle = AutonomousCycle(AutonomousPolicy(), ["one", "two"])
    calls = []

    async def gate(name, timeout_seconds):
        assert timeout_seconds == 300
        calls.append(name)
        return GateResult(name == "one", "exited 1", "failure " + "x" * 9000)

    for attempt in range(1, 5):
        message = await cycle.next_message(
            None, snapshot=lambda: git_snapshot(tmp_path), run_gate=gate
        )
        if attempt < 4:
            assert f"attempt {attempt}/3" in message
            if attempt > 1:
                assert "not rerun" in message
        else:
            assert message is None and cycle.stop_reason == "retry_exhausted"
    assert calls.count("two") == 1
    assert cycle.continuations == 3
    assert bounded_output("a" * 6001) == "a" * 6000 + "\n... [truncated]"
    assert len(bounded_output("😀" * 4000).split("\n")[0]) == 3000


@pytest.mark.parametrize("stop", ["error", "aborted"])
async def test_no_gate_after_assistant_error_or_abort(stop):
    async def forbidden(*args):
        pytest.fail("Gate must not run")

    cycle = AutonomousCycle(AutonomousPolicy(), ["gate"])
    assert await cycle.next_message(stop, snapshot=forbidden, run_gate=forbidden) is None


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("continuations", 3, "maxContinuations"),
        ("turns", 12, "maxTurns"),
        ("tokens", 80000, "maxTokens"),
        ("started", 0, "timeoutMs"),
    ],
)
async def test_gate_before_continuation_limits(field, value, reason):
    cycle = AutonomousCycle(AutonomousPolicy(), ["gate"])
    setattr(cycle, field, value)
    ran = []

    async def snapshot():
        return None

    async def gate(*args):
        ran.append(True)
        return GateResult(False, "exited 1")

    assert await cycle.next_message(None, snapshot=snapshot, run_gate=gate) is None
    assert ran == [True] and cycle.stop_reason == reason


def test_prime_usage_excludes_cache_reads():
    cycle = AutonomousCycle(AutonomousPolicy(), [])
    cycle.record_response(
        ModelResponse(usage=Usage(input_tokens=100, cached_input_tokens=90, output_tokens=5))
    )
    assert (cycle.turns, cycle.tokens) == (1, 15)
    cycle.record_response(ModelResponse(metadata={"stop_reason": "error"}))
    assert cycle.turns == 1


class FixturePeer:
    def __init__(self, worker, scripts, *, fail_first=False):
        self.worker, self.scripts = worker, scripts
        self.index = 0
        self.requests = []
        self.gates = 0
        self.fail_first = fail_first

    async def call(self, method, **args):
        if method == "resolve":
            return {"config": args["config"], "details": {}}
        if method == "gate":
            self.gates += 1
            status = self.worker.runtime.refinement_status(self.worker.root_id)
            assert not status["pending"] and not status["in_flight"]
            assert not self.worker.runtime.tasks
            return {
                "passed": not (self.fail_first and self.gates == 1),
                "exit_text": "exited 1",
                "output": "fixture diagnostic",
            }
        assert method == "model"
        request = args["request"]
        self.requests.append(request)
        purpose = request.get("metadata", {}).get("purpose")
        if purpose == "refinement":
            response = ModelResponse(
                text=json.dumps(
                    {
                        "summary": "Record fixture procedure",
                        "rationale": "fixture evidence",
                        "expectedOutcome": "later visibility",
                        "edits": [
                            {
                                "action": "create",
                                "kind": "memory",
                                "id": "fixture-procedure",
                                "title": "Fixture procedure",
                                "content": "Read both persistent values before completion.",
                                "metadata": {"learningAssessment": learning_assessment()},
                            }
                        ],
                    }
                )
            )
        elif purpose == "refinement_review":
            response = ModelResponse(
                text=json.dumps(
                    {"shouldRefine": False, "reason": "No reusable lesson", "focus": ""}
                )
            )
        elif request["parent_id"]:
            response = (
                cell(
                    "from pathlib import Path\nPath('child-proof.txt').write_text('evidence')\nawait agent_message.send('fixture child evidence', receiver_role='parent')"
                )
                if request["turn"] == 0
                else ModelResponse(text="Child settled")
            )
        else:
            response = (
                self.scripts[self.index]
                if self.index < len(self.scripts)
                else ModelResponse(text="Candidate settled")
            )
            self.index += 1
            if callable(response):
                response = response(request)
        return response.model_dump(mode="json")


async def make_worker(tmp_path, scripts, **kwargs):
    workspace = tmp_path / "workspace"
    initialize_git(workspace)
    worker = PersistentWorker()
    worker.peer = FixturePeer(worker, scripts, **kwargs)
    config = RunConfig(
        provider={"name": "fixture", "model": "fixture", "model_metadata": {"maxTokens": 8192}},
        limits={"max_turns": 100, "token_budget": 3000000, "wall_seconds": 120},
    )
    await worker.setup(
        "Official first request fixture",
        str(workspace),
        str(tmp_path / "state"),
        config.model_dump(),
    )
    return worker


async def test_same_root_kernel_child_refinement_and_failure_feedback(tmp_path):
    def repair(request):
        if "Autonomous quality gate failed" not in str(request["messages"]):
            worker.peer.index -= 1
            return ModelResponse(text="Candidate settled after child delivery")
        assert "Autonomous quality gate failed" in str(request["messages"])
        assert "fixture diagnostic" in str(request["messages"])
        return cell("from pathlib import Path\nPath('repair.py').write_text('fixed')")

    worker = await make_worker(
        tmp_path,
        [
            cell(
                "persistent_value = 73\nchild = await rlm('Fixture audit task', name='auditor')\nprint(child)"
            ),
            cell("print(await refine.run('Record the fixture procedure'))"),
            ModelResponse(text="Candidate settled"),
            repair,
            ModelResponse(text="Candidate repaired"),
            cell(
                "assert persistent_value == 73\nassert len(await rlm.list_subagents()) == 1\nprint(persistent_value)"
            ),
            ModelResponse(text="Second request settled"),
        ],
        fail_first=True,
    )
    try:
        first = await asyncio.wait_for(
            worker.run(worker.first_instruction, "round-1", asdict(AutonomousPolicy())), 60
        )
        second = await asyncio.wait_for(
            worker.run("Official second request fixture", "round-2", asdict(AutonomousPolicy())), 60
        )
        a, b = first["after"], second["before"]
        for key in (
            "root_session_id",
            "runtime_id",
            "kernel_pid",
            "kernel_id",
            "local_harness",
            "assistant_turns_since_auto_refine",
        ):
            assert a[key] == b[key]
        assert b["previous_history_retained"] and a["kernel_pid"]
        assert len(a["children"]) == 1
        assert first["autonomous"]["continuations"] == 1
        assert first["autonomous"]["stop_reason"] == "passed"
        assert a["local_harness"]["entries"]["memory"]["fixture-procedure"]["source"] == "refine"
        assert any("[self-refinement]" in str(r["messages"]) for r in worker.peer.requests)
        assert not any(
            r.get("metadata", {}).get("purpose") == "refinement_review"
            for r in worker.peer.requests
        )
        assert (tmp_path / "workspace/child-proof.txt").read_text() == "evidence"
        root_requests = [
            r
            for r in worker.peer.requests
            if not r["parent_id"] and r.get("metadata", {}).get("purpose", "agent") == "agent"
        ]
        assert len({json.dumps(r["messages"][0]) for r in root_requests}) == 1
    finally:
        await worker.runtime.shutdown()


async def test_owned_apply_blocks_gate_and_next_primary_request(tmp_path):
    worker = await make_worker(
        tmp_path,
        [
            cell("await refine.run('Fixture request')"),
            ModelResponse(text="Candidate"),
        ],
    )
    applying, release = asyncio.Event(), asyncio.Event()
    original = worker.runtime.apply_refinement_plan

    async def apply(*args):
        applying.set()
        await release.wait()
        return await original(*args)

    worker.runtime.apply_refinement_plan = apply
    task = asyncio.create_task(
        worker.run(worker.first_instruction, "round-1", asdict(AutonomousPolicy()))
    )
    try:
        await asyncio.wait_for(applying.wait(), 30)
        assert worker.peer.gates == 0 and worker.peer.index == 1
        release.set()
        packet = await asyncio.wait_for(task, 30)
        assert packet["autonomous"]["stop_reason"] == "passed"
        assert len(packet["after"]["refinement_history"]) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await worker.runtime.shutdown()


async def test_independent_tasks_have_fresh_harness_root_and_kernel(tmp_path):
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first = await make_worker(
        first_path,
        [
            cell("value_from_other_task = 73\nawait refine.run('Fixture request')"),
            ModelResponse(text="Done"),
        ],
    )
    second = None
    try:
        await first.run(first.first_instruction, "round-1", asdict(AutonomousPolicy()))
        second = await make_worker(
            second_path,
            [cell("assert 'value_from_other_task' not in globals()"), ModelResponse(text="Done")],
        )
        packet = await second.run(second.first_instruction, "round-1", asdict(AutonomousPolicy()))
        assert first.root_id != second.root_id
        assert first.audit()["kernel_pid"] != packet["after"]["kernel_pid"]
        assert not packet["after"]["refinement_history"]
        assert not packet["after"]["local_harness"]["entries"]["memory"]
        assert not packet["after"]["global_harness"]["entries"]["memory"]
    finally:
        await first.runtime.shutdown()
        if second:
            await second.runtime.shutdown()


async def test_25_turn_counter_spans_rounds_without_failure_trigger(tmp_path):
    scripts = [cell(f"interval_value = {i}") for i in range(12)] + [ModelResponse(text="Round one")]
    scripts += [cell(f"interval_value = {i}") for i in range(12, 24)] + [
        ModelResponse(text="Round two")
    ]
    worker = await make_worker(tmp_path, scripts)
    try:
        first = await asyncio.wait_for(
            worker.run(worker.first_instruction, "round-1", asdict(AutonomousPolicy())), 60
        )
        assert first["after"]["assistant_turns_since_auto_refine"] == 13
        second = await asyncio.wait_for(
            worker.run("Second fixture request", "round-2", asdict(AutonomousPolicy())), 60
        )
        reviews = [
            r
            for r in worker.peer.requests
            if r.get("metadata", {}).get("purpose") == "refinement_review"
        ]
        assert len(reviews) == 1
        assert "25 assistant turns" in str(reviews[0]["messages"])
        assert second["after"]["assistant_turns_since_auto_refine"] == 1
        assert not second["after"]["refinement_history"]
        assert not any(
            r.get("metadata", {}).get("purpose") == "refinement" for r in worker.peer.requests
        )
    finally:
        await worker.runtime.shutdown()


async def test_verifier_infrastructure_failure_aborts_without_continuation(tmp_path):
    worker = await make_worker(tmp_path, [ModelResponse(text="Candidate")])
    original = worker.peer.call

    async def fail(method, **args):
        if method == "gate":
            raise ConnectionError("Fixture host transport failed")
        return await original(method, **args)

    worker.peer.call = fail
    try:
        with pytest.raises(ConnectionError):
            await worker.run(worker.first_instruction, "round-1", asdict(AutonomousPolicy()))
        assert worker.cycle.continuations == 0
        assert not worker.runtime.store.events(worker.root_id, kind="user_intervention")
        packet = await worker.dispatch("last_round", {})
        assert packet["round"] == "round-1"
        assert packet["autonomous"]["stop_reason"] == "error"
        assert packet["usage"]["model_calls"] == 1
        assert any(e["type"] == "model_response" for e in packet["events"])
    finally:
        await worker.runtime.shutdown()


def test_diagnostics_disclose_only_numeric_aggregates():
    pytest.importorskip("harbor")
    from threadweave.evals.evocode import diagnostic

    safe, cases = diagnostic(
        {"reward": 0},
        "hidden source\n/tests/test.sh\nCASE_SUMMARY total_cases=4 success_count=2 fail_count=2\nsecret expected value",
    )
    assert cases == {"total_cases": 4, "success_count": 2}
    assert safe["output"] == "Cumulative verification failed.\nCases passed: 2/4."
    assert "/tests" not in str(safe) and "secret" not in str(safe)


def test_official_rewards_survive_an_incomplete_instrumentation_export():
    from threadweave.evals.evocode_report import task_report

    report = task_report(
        "task",
        3,
        [],
        [],
        1.0,
        {"rewards": {"reward": 1}},
        official_steps=[{"round_name": "round-1", "rewards": {"reward": 1}}],
    )
    assert report["rounds_reached"] == 1
    assert report["rounds_with_instrumentation"] == 0
    assert report["rounds_passed"] == 1
    assert report["evocode_task_score"] == 1 / 3


async def test_worker_transport_failure_does_not_skip_container_cleanup(monkeypatch):
    pytest.importorskip("harbor")
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from harbor.trial.multi_step import MultiStepTrial

    from threadweave.evals.evocode import BuffaloEvoCodeTrial

    cleanup = AsyncMock()
    monkeypatch.setattr(MultiStepTrial, "_stop_agent_environment", cleanup)
    # super() requires the real subclass identity, without starting a container.
    trial = object.__new__(BuffaloEvoCodeTrial)
    trial._is_agent_environment_stopped = False
    trial.agent = SimpleNamespace(close=AsyncMock(side_effect=ConnectionError("transport closed")))
    trial._record_exception = Mock()
    await trial._stop_agent_environment()
    cleanup.assert_awaited_once()
    trial._record_exception.assert_called_once()
