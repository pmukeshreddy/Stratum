"""Rehearsal through the real Buffalo REPL, shell, resident socket and host bridge."""

import inspect
import json
import shlex
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.autonomous import AutonomousPolicy
from threadweave.evals.autonomous_rehearsal import RehearsalServer
from threadweave.evals.emulatorbench import discover_tasks
from threadweave.evals.emulatorbench_public import PublicSourceVerifier, public_contract_files
from threadweave.evals.emulatorbench_report import task_result
from threadweave.models import ModelResponse

from .test_emulatorbench import worker_fixture
from .test_evocode_integration import cell


def test_public_contract_uses_exact_source_and_only_public_descriptors():
    pytest.importorskip("emulatorbench")
    from emulatorbench.emulator_common import runners, suite_adapters

    for task in discover_tasks():
        files = public_contract_files(task)
        mapping = task.data.model_dump(mode="json")
        assert files["verification/public_manifest.json"] == runners.public_manifest_file(mapping)
        assert files["verification/public_tests.json"] == runners.public_tests_file(mapping)
        contract = files["verification/PUBLIC_VERIFIER_CONTRACT.md"]
        assert inspect.getsource(suite_adapters.invoke_case) in contract
        assert inspect.getsource(suite_adapters.replay_determinism) in contract
        manifest = json.loads(files["verification/public_manifest.json"])
        assert "private_cases" not in manifest
        assert not any(term in files["README.md"].lower() for term in ("rlm", "refine", "delegate"))


async def test_rehearse_fail_unchanged_fix_pass_then_final_same_runtime(tmp_path, monkeypatch):
    pytest.importorskip("emulatorbench")
    from emulatorbench.emulator_common import runtime as official_runtime
    from emulatorbench.emulator_common.runner_v2 import fail_closed_runner_v1_score

    with tempfile.TemporaryDirectory(prefix="rehearse-", dir="/tmp") as sockets:
        socket_path = str(Path(sockets) / "command.sock")
        command = shlex.join(
            [
                sys.executable,
                inspect.getfile(RehearsalServer),
                "--socket",
                socket_path,
                "--timeout",
                "20",
            ]
        )
        invoke = f"r = await bash({command!r})\nprint(r.stdout)\n"

        def fix(request):
            assert "input_file" in str(request["messages"])
            assert "workspace has not changed" in str(request["messages"])
            return cell(
                "assert persistent_value == 937\nfrom pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
            )

        worker = await worker_fixture(
            tmp_path,
            [
                cell("persistent_value = 937\nawait refine.run('Fixture procedure')"),
                cell(invoke + "assert r.exit_code == 1"),
                cell(invoke + "assert r.exit_code == 1"),
                fix,
                cell(invoke + "assert r.exit_code == 0\nassert persistent_value == 937"),
                ModelResponse(text="Final implementation rehearsed."),
            ],
        )
        journal = worker.peer.journal
        identities = []

        async def snapshot(runtime):
            return SimpleNamespace(
                sanitized_archive=b"fixture",
                metadata={"candidate_archive_sha256": "a" * 64, "sha256": "b" * 64},
            )

        monkeypatch.setattr(official_runtime, "export_submission_archive", snapshot)

        async def grader(trace, runtime, archive, *, score_info, **kwargs):
            identities.append(worker.audit())
            passed = (tmp_path / "workspace/emulator.py").read_text() == "print(2)\n"
            original = {
                "score": 1.0 if passed else 0.5,
                "passed": passed,
                "result_sha256": "d" * 64,
                "case_runs": [
                    {
                        "case_id": "fixture",
                        "artifact": "/isolated/input.json5",
                        "passed": passed,
                        "returncode": 0,
                        "output": {"cycles": 12, "frames": 1},
                    }
                ],
            }
            score_info.update(
                emulatorbench_score=fail_closed_runner_v1_score(original, task_id="fixture"),
                emulatorbench_verifier_stdout=json.dumps(original),
                emulatorbench_grader_lifecycle={
                    "status": "completed",
                    "stopped": True,
                    "egress_probe": "blocked",
                },
            )
            return 0.0

        task = SimpleNamespace(
            config=SimpleNamespace(anti_cheat_validation=False),
            _runner_v2_corpus_material=lambda: (None, None),
            _run_fresh_grader=grader,
        )
        policy = AutonomousPolicy(max_turns=1)
        bridge = PublicSourceVerifier(task, object(), object(), journal, policy)
        bridge.max_rehearsals = 4
        original_call = worker.peer.call

        async def call(method, **args):
            if method == "gate":
                return await bridge(**args)
            if method == "rehearse":
                return await bridge.rehearse(**args)
            return await original_call(method, **args)

        worker.peer.call = call
        worker.rehearsal_options = {"command": bridge.command, "timeout_seconds": 300}
        server = RehearsalServer(socket_path, worker.rehearse)
        await server.start()
        try:
            packet = await worker.run(worker.first_instruction, bridge.command, asdict(policy))
            assert packet["autonomous"]["turns"] >= 6
            assert packet["autonomous"]["continuations"] == 0
            assert packet["autonomous"]["stop_reason"] == "passed"
            assert len(journal.rehearsals) == 2 and len(journal.attempts) == 1
            assert len(identities) == 3
            for key in ("root_session_id", "kernel_id", "kernel_pid", "workspace"):
                assert len({identity[key] for identity in identities}) == 1
            assert identities[0]["local_harness"] == identities[-1]["local_harness"]
            assert identities[0]["local_harness"]
            assert (
                len([e for e in packet["events"] if e["type"] == "evaluation_rehearsal_result"])
                == 3
            )
            assert not any(e["type"] == "python_error" for e in packet["events"])
            result = task_result("fixture", packet, journal, None, 1)
            assert result["public_rehearsal_scores"] == [0.5, 1.0]
            assert result["score_attempt_id"] == journal.attempts[0]["id"]
        finally:
            await server.close()
            await worker.runtime.shutdown()


async def test_native_resource_exhaustion_is_not_an_adapter_or_benchmark_failure(tmp_path):
    from threadweave.evals.autonomous_host import EvaluationFailure
    from threadweave.models import HarnessError, Usage

    from .test_emulatorbench import run_fixture

    response = cell("persistent_value = 937")
    response.usage = Usage(input_tokens=3_000_000)
    worker = await worker_fixture(tmp_path, [response])
    try:
        with pytest.raises(HarnessError, match="Root stopped: limited:") as captured:
            await run_fixture(worker)
        result = task_result(
            "fixture",
            worker.last_packet,
            worker.peer.journal,
            EvaluationFailure("ADAPTER_FAILURE", str(captured.value)),
            1,
        )
        assert result["status"] == "RESOURCE_LIMIT"
        assert result["official_score"] is None and worker.peer.gates == 0
        assert result["resource_limit_event_id"]
        assert "token" in result["stop_reason"].lower()
    finally:
        await worker.runtime.shutdown()
