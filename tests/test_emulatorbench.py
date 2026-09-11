"""Real Runtime continuation tests. Scripted choices are tests, never task policy."""

import asyncio
import io
import json
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest

from threadweave.autonomous import AutonomousPolicy
from threadweave.evals.autonomous_host import (
    EvaluationFailure,
    EvidenceJournal,
    agent_source_archive,
)
from threadweave.evals.autonomous_worker import AutonomousWorker, assert_identity
from threadweave.evals.emulatorbench import discover_tasks, pins, preflight
from threadweave.evals.emulatorbench_report import aggregate, audit_task, task_result
from threadweave.models import ModelResponse, RunConfig, Usage

from .test_evocode_integration import FixturePeer, cell, initialize_git


class ControlPeer(FixturePeer):
    def __init__(self, worker, scripts, directory):
        super().__init__(worker, scripts)
        self.journal = EvidenceJournal(directory)
        self.identities = []

    async def call(self, method, **args):
        if method == "evidence":
            self.journal.evidence(args["events"])
            return {}
        if method == "progress":
            self.journal.status(**args["status"])
            return {}
        if method == "gate":
            self.gates += 1
            self.identities.append(args["identity"])
            workspace = Path(args["identity"]["workspace"])
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(workspace / "emulator.py"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            passed = process.returncode == 0 and stdout.strip() == b"2"
            attempt = {
                "id": f"fixture-{self.gates}",
                "status": "BENCHMARK_RESULT",
                "score": 1.0 if passed else 0.0,
                "passed": passed,
            }
            self.journal.attempts.append(attempt)
            return {
                "passed": passed,
                "exit_text": "passed" if passed else "failed",
                "output": "Fixture emulator returned an incorrect value.",
                "host_attempt_id": attempt["id"],
            }
        result = await super().call(method, **args)
        if method == "model":
            self.journal.calls.append(
                {
                    "id": args["request"]["request_id"],
                    "request": args["request"],
                    "response": result,
                }
            )
        return result


async def worker_fixture(tmp_path, scripts):
    workspace = tmp_path / "workspace"
    initialize_git(workspace)
    (workspace / "emulator.py").write_text("print(1)\n")
    worker = AutonomousWorker()
    worker.peer = ControlPeer(worker, scripts, tmp_path / "evidence")
    config = RunConfig(
        provider={"name": "fixture", "model": "fixture", "model_metadata": {"maxTokens": 8192}},
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        limits={"max_turns": 100, "token_budget": 3000000, "wall_seconds": 120},
    )
    await worker.setup(
        "Implement the fixture emulator.",
        str(workspace),
        str(tmp_path / "state"),
        config.model_dump(),
    )
    return worker


async def run_fixture(worker, policy=None):
    return await asyncio.wait_for(
        worker.run(
            worker.first_instruction, "fixture verifier", asdict(policy or AutonomousPolicy())
        ),
        45,
    )


async def test_fail_feedback_fix_pass_same_trajectory_and_native_state(tmp_path):
    seen_feedback = []

    def repair(request):
        if "Autonomous quality gate failed" not in str(request["messages"]):
            worker.peer.index -= 1
            return ModelResponse(text="Ready for validation.")
        seen_feedback.append(request)
        assert any(
            m.get("role") == "user" and "Autonomous quality gate failed" in str(m)
            for m in request["messages"]
        )
        assert any(
            h["id"] == "fixture-procedure"
            for h in request["metadata"]["execution_inputs"]["harness_state"]
        )
        return cell(
            "assert persistent_value == 73\nassert len(await rlm.list_subagents()) == 1\nawait compact()\nfrom pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
        )

    worker = await worker_fixture(
        tmp_path,
        [
            cell("persistent_value = 73\nchild = await rlm('Fixture audit task', name='auditor')"),
            cell("await refine.run('Record fixture procedure')"),
            ModelResponse(text="Ready."),
            repair,
            cell("assert persistent_value == 73"),
            ModelResponse(text="Fixed."),
        ],
    )
    try:
        packet = await run_fixture(worker)
        assert packet["autonomous"]["stop_reason"] == "passed"
        assert worker.peer.gates == 2 and packet["autonomous"]["continuations"] == 1
        assert seen_feedback
        a, b = worker.peer.identities
        assert_identity(a, b)
        assert a["kernel_pid"] and a["kernel_pid"] == b["kernel_pid"]
        assert a["local_harness"] == b["local_harness"]
        assert [c["id"] for c in a["children"]] == [c["id"] for c in b["children"]]
        assert a["assistant_turns_since_auto_refine"] < b["assistant_turns_since_auto_refine"]
        result = task_result("fixture", packet, worker.peer.journal, None, 1)
        assert result["official_score"] == 1 and result["passed"]
        audit = audit_task(packet, worker.peer.journal)
        assert len(audit["children"]) == 1
        assert audit["children"][0]["root_used_result"] is None
        assert any(c["claim"] == "RLM_CALLED" for c in audit["claims"])
        assert audit["refinement_counts"]["applied_edits"] == 1
        # The preceding explicit refinement starts the native 1200s cooldown.
        assert audit["refinement_counts"]["compact_reviews"] == 0
        assert packet["after"]["compaction_events"]
        assert all(audit["persistence"]["invariants"].values())
        ids = {e["id"] for e in worker.peer.journal.events}
        assert all(set(c["event_ids"]) <= ids for c in audit["claims"])
        assert any(
            e["type"] == "evaluation_file_observation" and e["payload"]["changes"]
            for e in worker.peer.journal.events
        )
    finally:
        await worker.runtime.shutdown()


async def test_unchanged_workspace_skips_real_verifier_and_consumes_prime_retries(tmp_path):
    worker = await worker_fixture(tmp_path, [ModelResponse(text="Ready.")] * 5)
    try:
        packet = await run_fixture(worker)
        assert worker.peer.gates == 1
        assert packet["autonomous"]["stop_reason"] == "retry_exhausted"
        assert packet["autonomous"]["continuations"] == 3
        assert [c["rerun"] for c in packet["autonomous"]["checks"]] == [True, False, False, False]
        assert any("workspace has not changed" in str(r["messages"]) for r in worker.peer.requests)
        assert len({r["session_id"] for r in worker.peer.requests}) == 1
    finally:
        await worker.runtime.shutdown()


async def test_compaction_review_declines_naturally_without_resetting_cycle(tmp_path):
    worker = await worker_fixture(
        tmp_path,
        [
            cell("persistent_value = 73"),
            ModelResponse(text="Ready."),
            cell(
                "await compact()\nfrom pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
            ),
            cell("assert persistent_value == 73"),
            ModelResponse(text="Fixed."),
        ],
    )
    try:
        packet = await run_fixture(worker)
        assert packet["autonomous"]["continuations"] == 1
        assert packet["autonomous"]["turns"] >= 5
        assert packet["autonomous"]["stop_reason"] == "passed"
        assert packet["after"]["compaction_events"]
        counts = audit_task(packet, worker.peer.journal)["refinement_counts"]
        assert counts["compact_reviews"] == 1 and counts["declines"] == 1
        assert counts["explicit"] == 0 and counts["planner_calls"] == 0
        assert_identity(*worker.peer.identities)
    finally:
        await worker.runtime.shutdown()


async def test_exact_continuation_limit_and_no_cycle_reset(tmp_path):
    worker = await worker_fixture(tmp_path, [ModelResponse(text="Ready.")] * 4)
    try:
        packet = await run_fixture(worker, AutonomousPolicy(max_continuations=1, max_retries=20))
        assert packet["autonomous"]["stop_reason"] == "maxContinuations"
        assert packet["autonomous"]["continuations"] == 1
        with pytest.raises(RuntimeError, match="cannot be restarted"):
            await run_fixture(worker)
        result = task_result("fixture", packet, worker.peer.journal, None, 1)
        assert not result["passed"] and result["official_score"] == 0
    finally:
        await worker.runtime.shutdown()


async def test_task_isolation_and_fresh_native_harness(tmp_path):
    first_path, next_path = tmp_path / "a", tmp_path / "b"
    first_path.mkdir()
    next_path.mkdir()
    a = await worker_fixture(
        first_path,
        [
            cell(
                "old_value = 73\nchild = await rlm('Fixture child', name='old')\nawait refine.run('Fixture memory')"
            ),
            ModelResponse(text="Ready."),
        ],
    )
    b = None
    try:
        await run_fixture(a, AutonomousPolicy(max_continuations=1))
        b = await worker_fixture(
            next_path,
            [
                cell(
                    "assert 'old_value' not in globals()\nassert len(await rlm.list_subagents()) == 0\nfrom pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
                ),
                ModelResponse(text="Ready."),
            ],
        )
        packet = await run_fixture(b)
        assert a.root_id != b.root_id
        assert a.audit()["kernel_id"] != packet["after"]["kernel_id"]
        assert not packet["after"]["children"]
        assert not packet["after"]["refinement_history"]
        assert not packet["after"]["local_harness"]["entries"]["memory"]
        assert not packet["after"]["global_harness"]["entries"]["memory"]
        assert packet["autonomous"]["stop_reason"] == "passed"
    finally:
        await a.runtime.shutdown()
        if b:
            await b.runtime.shutdown()


def test_official_discovery_and_precommitted_selection():
    pytest.importorskip("emulatorbench")
    tasks = discover_tasks()
    assert len(tasks) == 16
    assert [t.data.task_id for t in tasks[:4]] == [
        "emulatorbench-chip8",
        "emulatorbench-i8080-space-invaders",
        "emulatorbench-gameboy-dmg",
        "emulatorbench-nes",
    ]
    assert [t.data.task_id for t in tasks] == pins()["benchmark_tasks"]


def test_official_preflight_rejects_provisional_corpora_before_inference(monkeypatch):
    pytest.importorskip("emulatorbench")
    monkeypatch.delenv("EMULATORBENCH_RUNNER_V2_CORPUS_ROOT", raising=False)
    result = preflight(json.loads(Path("configs/emulatorbench.json").read_text()))
    assert not result["ready"]
    assert len(result["tasks"]) == 4
    assert sum("upstream has no installed" in b for b in result["blockers"]) == 3


def test_no_benchmark_or_hidden_assets_in_agent_distribution():
    with tarfile.open(fileobj=io.BytesIO(agent_source_archive())) as archive:
        names = archive.getnames()
    assert "src/threadweave/runtime.py" in names
    assert "src/threadweave/evals/autonomous_worker.py" in names
    assert "src/threadweave/evals/autonomous_rehearsal.py" in names
    assert not any("emulatorbench" in n or "test_emulatorbench" in n for n in names)


def test_reporting_keeps_null_failures_and_numeric_partial_credit(tmp_path):
    journal = EvidenceJournal(tmp_path / "evidence")
    journal.attempts.append(
        {"id": "official-partial", "status": "BENCHMARK_RESULT", "score": 0.61, "passed": False}
    )
    packet = {
        "after": {"root_session_id": "R1", "kernel_id": "K1", "workspace": "W1"},
        "autonomous": {"stop_reason": "maxContinuations"},
    }
    result = task_result("a", packet, journal, None, 1)
    assert result["official_score"] == 0.61 and not result["passed"]
    for status in ("PROVIDER_FAILURE", "ADAPTER_FAILURE", "INFRASTRUCTURE_FAILURE", "TIMEOUT"):
        failed = task_result("b", packet, journal, EvaluationFailure(status, "reason"), 1)
        assert failed["official_score"] is None and failed["status"] == status
        assert aggregate([result, failed], ["a", "b"])["official_mean_score"] is None
    assert (
        aggregate([result, {**result, "task_id": "b", "official_score": 0.88}], ["a", "b"])[
            "official_mean_score"
        ]
        == (0.61 + 0.88) / 2
    )


def test_identity_checks_fail_loudly():
    a = dict.fromkeys(
        (
            "runtime_id",
            "runtime_pid",
            "root_session_id",
            "kernel_id",
            "workspace",
            "workspace_device",
            "workspace_inode",
            "kernel_pid",
        ),
        "same",
    )
    for key in a:
        with pytest.raises(RuntimeError, match="IDENTITY VIOLATION"):
            assert_identity(a, {**a, key: "changed"})


async def test_root_budget_excludes_native_child_and_cache_tokens(tmp_path):
    worker = await worker_fixture(
        tmp_path, [cell("child = await rlm('Fixture task')"), ModelResponse(text="Ready.")]
    )
    original = worker.peer.call

    async def with_usage(method, **args):
        response = await original(method, **args)
        if method == "model":
            response["usage"] = Usage(
                input_tokens=100, cached_input_tokens=90, output_tokens=5
            ).model_dump()
        return response

    worker.peer.call = with_usage
    try:
        packet = await run_fixture(worker, AutonomousPolicy(max_continuations=1))
        root_calls = [
            r
            for r in worker.peer.requests
            if not r["parent_id"] and r.get("metadata", {}).get("purpose", "agent") == "agent"
        ]
        assert packet["autonomous"]["turns"] == len(root_calls)
        assert packet["autonomous"]["tokens"] == len(root_calls) * 15
        assert any(r["parent_id"] for r in worker.peer.requests)
    finally:
        await worker.runtime.shutdown()
