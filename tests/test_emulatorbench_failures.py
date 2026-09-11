"""Failure injection at external boundaries; production scoring is never substituted."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.autonomous import AutonomousPolicy
from threadweave.evals.autonomous_host import EvaluationFailure, EvidenceJournal, ResidentClient
from threadweave.evals.emulatorbench import OfficialVerifier
from threadweave.evals.emulatorbench_report import read_jsonl, reviewed_claims, task_result
from threadweave.models import HarnessError, ModelResponse

from .test_emulatorbench import run_fixture, worker_fixture


async def test_provider_failure_before_attempt_stays_ungraded(tmp_path):
    class BrokenProvider:
        async def invoke(self, request, emit):
            raise ConnectionError("scripted provider unavailable")

    worker = await worker_fixture(tmp_path, [])
    peer = worker.peer
    client = ResidentClient(
        worker.runtime.store.config(worker.root_id),
        peer.journal,
        None,
        providers={"fixture": BrokenProvider()},
    )
    original = peer.call

    async def proxy(method, **args):
        if method == "model":
            return await client.handle(method, args)
        return await original(method, **args)

    peer.call = proxy
    try:
        with pytest.raises(HarnessError, match="Root stopped"):
            await run_fixture(worker)
        result = task_result("fixture", worker.last_packet, peer.journal, client.failure, 1)
        assert result["status"] == "PROVIDER_FAILURE"
        assert result["official_score"] is None and peer.gates == 0
        assert result["usage"]["root"]["calls_attempted"] >= 1
        usage = result["usage"]["root"]
        assert usage["model_calls"] == usage["calls_attempted"]
        assert usage["calls_completed"] == 0
        assert usage["calls_without_reported_usage"] == usage["calls_attempted"]
        assert not usage["usage_complete"]
        assert len(usage["unreported_call_attempts"]) == usage["calls_attempted"]
        assert "partial_provider_report" in usage["token_accounting"]
        assert all(c["error"] for c in read_jsonl(peer.journal.directory / "model-calls.jsonl"))
    finally:
        await worker.runtime.shutdown()


async def test_cancelled_trajectory_keeps_packet_and_stops_runtime_work(tmp_path):
    worker = await worker_fixture(tmp_path, [ModelResponse(text="Ready.")])
    entered = asyncio.Event()
    original = worker.peer.call

    async def stalled_gate(method, **args):
        if method == "gate":
            entered.set()
            await asyncio.Future()
        return await original(method, **args)

    worker.peer.call = stalled_gate
    task = asyncio.create_task(
        worker.run(worker.first_instruction, "fixture verifier", asdict(AutonomousPolicy()))
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.last_packet["autonomous"]["stop_reason"] == "aborted"
        assert not worker.runtime.tasks
        assert any(e["type"] == "evaluation_trajectory_error" for e in worker.peer.journal.events)
    finally:
        await worker.runtime.shutdown()


async def test_verifier_cannot_mutate_candidate_workspace(tmp_path):
    worker = await worker_fixture(tmp_path, [ModelResponse(text="Ready.")])
    original = worker.peer.call

    async def changed_gate(method, **args):
        result = await original(method, **args)
        if method == "gate":
            # A verifier that writes candidate code violates the boundary even
            # if its reported verdict says the candidate passed.
            (tmp_path / "workspace" / "emulator.py").write_text("print(2)\n")
        return result

    worker.peer.call = changed_gate
    try:
        with pytest.raises(RuntimeError, match="VERIFIER MUTATED CANDIDATE WORKSPACE"):
            await run_fixture(worker)
        assert worker.last_packet["autonomous"]["stop_reason"] == "error"
    finally:
        await worker.runtime.shutdown()


@pytest.mark.parametrize("mode", ["timeout", "untrusted", "partial"])
async def test_official_bridge_timeout_untrusted_and_numeric_disclosure(
    tmp_path, monkeypatch, mode
):
    runtime_module = pytest.importorskip("emulatorbench.emulator_common.runtime")

    async def snapshot(runtime):
        return SimpleNamespace(
            sanitized_archive=b"test snapshot", candidate_archive=b"test snapshot", metadata={}
        )

    async def controlled_external_boundary(*args, **kwargs):
        assert kwargs["expose_feedback"] is False
        if mode == "timeout":
            await asyncio.Future()
        score = {
            "score": 0.625,
            "passed": False,
            "trusted_oracle_verified": mode == "partial",
            "golden_output": "secret-answer",
            "components": {"overall_unit_pass_rate": 0.625, "secret": "secret-answer"},
        }
        return ({"score": 0.625, "passed": False} if mode == "partial" else {}), {
            "emulatorbench_score": score
        }

    monkeypatch.setattr(runtime_module, "export_submission_archive", snapshot)
    journal = EvidenceJournal(tmp_path / "evidence")
    policy = AutonomousPolicy(gate_timeout_seconds=0.05)
    bridge = OfficialVerifier(
        SimpleNamespace(_validate_controller_submission=controlled_external_boundary),
        object(),
        object(),
        journal,
        policy,
    )
    if mode == "partial":
        result = await bridge(bridge.command, 0.05, {"root": "fixture"}, None)
        assert "secret-answer" not in json.dumps(result)
        assert "0.625" not in result["output"]  # Official quantized candidate projection.
        assert journal.attempts[0]["score"] == 0.625
        assert not result["passed"]
    else:
        with pytest.raises(EvaluationFailure):
            await bridge(bridge.command, 0.05, {"root": "fixture"}, None)
        assert journal.attempts[0]["score"] is None
        assert journal.attempts[0]["status"] == (
            "TIMEOUT" if mode == "timeout" else "INFRASTRUCTURE_FAILURE"
        )
    assert len(read_jsonl(journal.directory / "verifier-attempts.jsonl")) == 1


def test_event_identity_conflicts_and_corrupt_completed_lines_are_rejected(tmp_path):
    journal = EvidenceJournal(tmp_path / "evidence")
    event = {"id": "e1", "type": "fixture", "payload": {}}
    journal.evidence([event, event])
    assert len(journal.events) == 1
    with pytest.raises(EvaluationFailure, match="Conflicting raw event"):
        journal.evidence([{**event, "payload": {"changed": True}}])
    path = tmp_path / "live.jsonl"
    path.write_text('{"complete": true}\n{"partial":')
    assert read_jsonl(path) == [{"complete": True}]
    path.write_text('{"complete": true}\n{bad}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


async def test_launcher_preflight_prevents_inference_and_preserves_null_aggregate(
    tmp_path, monkeypatch
):
    pytest.importorskip("emulatorbench")
    import threadweave.evals.emulatorbench as evaluator

    def forbidden(*args, **kwargs):
        raise AssertionError("No model client may be created before data preflight passes")

    monkeypatch.setattr(evaluator, "ResidentClient", forbidden)
    monkeypatch.delenv("EMULATORBENCH_RUNNER_V2_CORPUS_ROOT", raising=False)
    config = json.loads(await asyncio.to_thread(Path("configs/emulatorbench.json").read_text))
    output = tmp_path / "run"
    with pytest.raises(EvaluationFailure, match="Preflight failed before inference"):
        await evaluator.run_experiment(config, output)
    result = json.loads((output / "aggregate.json").read_text())
    assert result["official_mean_score"] is None
    assert result["tasks_planned"] == 4 and result["tasks_finished"] == 0
    assert all(r["official_score"] is None and r["status"] == "NOT_RUN" for r in result["tasks"])
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["provenance"]["buffalo_git_sha"]
    assert manifest["provenance"]["agent_source_archive_sha256"]
    assert manifest["provenance"]["untracked_archive_sha256"]
    assert manifest["ended"]


def test_causal_review_requires_quote_action_ancestry_and_official_progress():
    # A compact recorded-event graph for the annotation validator, not a grader.
    def event(identifier, seq, kind, payload, sid="root", parent=None):
        return {
            "id": identifier,
            "seq": seq,
            "type": kind,
            "payload": payload,
            "session_id": sid,
            "parent_event_id": parent,
        }

    events = [
        event("failed", 1, "evaluation_verifier_result", {"host_attempt_id": "v1"}),
        event(
            "child-message",
            2,
            "agent_message_sent",
            {"sender_id": "child", "body": "Fix sign extension of opcode 0x82"},
            "child",
        ),
        event(
            "received",
            3,
            "agent_message_received",
            {"source_event": "child-message", "sender_id": "child"},
        ),
        event("response", 4, "model_response", {}),
        event("edit", 5, "python_execution", {"code": "fix_opcode_82()"}, parent="response"),
        event("passed", 6, "evaluation_verifier_result", {"host_attempt_id": "v2"}),
    ]
    packet = {
        "after": {"root_session_id": "root"},
        "requests": [{"id": "request", "response_event": "response"}],
    }
    quote = (
        "The child identified opcode 0x82's sign-extension error; I will fix that decoder branch."
    )
    journal = SimpleNamespace(
        events=events,
        calls=[{"id": "request", "request": {"session_id": "root"}, "response": {"text": quote}}],
        attempts=[
            {"id": "v1", "status": "BENCHMARK_RESULT", "score": 0.61},
            {"id": "v2", "status": "BENCHMARK_RESULT", "score": 0.88},
        ],
    )
    review = {
        "claim": "ROOT_USED_RESULT_WITH_BENEFIT",
        "reviewer": "fixture reviewer",
        "evidence": "The cited response explicitly adopts the child's finding and its linked action changes that decoder branch.",
        "source_event": "child-message",
        "root_received_event": "received",
        "root_request": "request",
        "root_action_event": "edit",
        "response_quote": quote,
        "before_attempt": "v1",
        "after_attempt": "v2",
    }
    claims = reviewed_claims(packet, journal, [review])
    assert claims[0]["review_id"] and "reviewer_assessed" in claims[0]["assessment"]
    assert set(claims[0]["event_ids"]) <= {e["id"] for e in events}
    with pytest.raises(ValueError, match="quote"):
        reviewed_claims(packet, journal, [{**review, "response_quote": "invented reasoning"}])
    events[4]["parent_event_id"] = None
    with pytest.raises(ValueError, match="not linked"):
        reviewed_claims(packet, journal, [review])
    events[4]["parent_event_id"] = "response"
    journal.attempts[1]["score"] = 0.61
    with pytest.raises(ValueError, match="official improvement"):
        reviewed_claims(packet, journal, [review])
