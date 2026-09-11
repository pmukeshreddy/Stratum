"""Public-mode bridge tests exercise the same native autonomous trajectory."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.autonomous import AutonomousPolicy
from threadweave.evals.autonomous_host import EvaluationFailure
from threadweave.evals.emulatorbench_public import PublicSourceVerifier, public_feedback
from threadweave.evals.emulatorbench_report import aggregate, task_result
from threadweave.models import ModelResponse

from .test_emulatorbench import run_fixture, worker_fixture
from .test_evocode_integration import cell


@pytest.mark.parametrize("broken", [False, True])
async def test_public_source_same_root_continuation_preserves_both_scores(
    tmp_path, monkeypatch, broken
):
    from emulatorbench.emulator_common import runtime as official_runtime
    from emulatorbench.emulator_common.runner_v2 import fail_closed_runner_v1_score

    def repair(request):
        assert "Autonomous quality gate failed" in str(request["messages"])
        return cell("from pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')")

    worker = await worker_fixture(
        tmp_path, [ModelResponse(text="Ready."), repair, ModelResponse(text="Ready.")]
    )
    journal = worker.peer.journal

    async def snapshot(runtime):
        return SimpleNamespace(
            sanitized_archive=b"fixture",
            metadata={"candidate_archive_sha256": "a" * 64, "sha256": "b" * 64},
        )

    monkeypatch.setattr(official_runtime, "export_submission_archive", snapshot)

    async def external_grader(trace, runtime, archive, *, score_info, **kwargs):
        passed = (tmp_path / "workspace/emulator.py").read_text() == "print(2)\n"
        original = {"score": 1.0 if passed else 0.5, "passed": passed, "result_sha256": "d" * 64}
        score_info.update(
            emulatorbench_score=fail_closed_runner_v1_score(original, task_id="fixture"),
            emulatorbench_verifier_stdout=json.dumps(original),
            emulatorbench_grader_lifecycle={
                "status": "completed",
                "stopped": not broken,
                "egress_probe": "blocked",
            },
        )
        return 0.0

    task = SimpleNamespace(
        config=SimpleNamespace(anti_cheat_validation=False),
        _runner_v2_corpus_material=lambda: (None, None),
        _run_fresh_grader=external_grader,
    )
    bridge = PublicSourceVerifier(task, object(), object(), journal, AutonomousPolicy())
    original_call = worker.peer.call

    async def call(method, **args):
        if method == "gate":
            return await bridge(
                bridge.command, args["timeout_seconds"], args["identity"], args["fingerprint"]
            )
        return await original_call(method, **args)

    worker.peer.call = call
    try:
        if broken:
            with pytest.raises(EvaluationFailure):
                await run_fixture(worker)
            assert journal.attempts[0]["score"] is None
            assert journal.attempts[0]["status"] == "INFRASTRUCTURE_FAILURE"
        else:
            packet = await run_fixture(worker)
            result = task_result("fixture", packet, journal, None, 1)
            assert len(journal.attempts) == 2
            assert packet["autonomous"]["continuations"] == 1
            assert packet["before"]["root_session_id"] == packet["after"]["root_session_id"]
            assert packet["before"]["kernel_id"] == packet["after"]["kernel_id"]
            assert result["passed"] is True and result["official_passed"] is False
            assert result["official_score"] == 0 and result["public_source_score"] == 1
            assert result["trusted_oracle_verified"] is False
            report = aggregate([result], ["fixture"])
            assert report["official_mean_score"] == 0 and report["public_source_mean_score"] == 1
    finally:
        await worker.runtime.shutdown()


def test_public_feedback_omits_expected_states():
    score = {
        "build": {"cargo_build": {"ok": False, "stderr": "compilation failed"}},
        "case_runs": [
            {
                "case_name": "public test",
                "passed": False,
                "expected_state": "hidden-answer",
                "output": {"expected": "hidden-answer"},
            }
        ],
    }
    feedback = public_feedback(score, {"score": 0.5, "passed": False})
    assert "hidden-answer" not in feedback
    assert "compilation failed" in feedback and "public test" in feedback


def test_public_configuration_preserves_experiment_and_native_mechanisms():
    signed = json.loads(Path("configs/emulatorbench.json").read_text())
    public = json.loads(Path("configs/emulatorbench-public.json").read_text())
    assert public["run"] == signed["run"]
    assert public["autonomous"] == signed["autonomous"]
    assert public["runtime"] == signed["runtime"]
    assert public["verification_mode"] == "public_source"


def test_runtime_patch_is_exact_idempotent_and_does_not_change_shared_files(tmp_path, monkeypatch):
    import os

    import verifiers

    from threadweave.evals.emulatorbench import pins
    from threadweave.evals.emulatorbench_setup import apply_runtime_patches
    from threadweave.evals.schema import file_digest

    package = tmp_path / "verifiers"
    destination = package / "v1/runtimes/modal.py"
    destination.parent.mkdir(parents=True)
    original = Path(".emulatorbench/sources/verifiers/verifiers/v1/runtimes/modal.py").read_bytes()
    shared = tmp_path / "shared-original.py"
    shared.write_bytes(original)
    os.link(shared, destination)
    monkeypatch.setattr(verifiers, "__file__", str(package / "__init__.py"))
    apply_runtime_patches()
    apply_runtime_patches()
    assert shared.read_bytes() == original
    assert file_digest(destination) == pins()["verifiers"]["patches"][0]["patched_sha256"]


@pytest.mark.parametrize("network", [False, True])
async def test_patched_official_modal_runtime_preserves_network_policy(monkeypatch, network):
    import modal
    from verifiers.v1.runtimes import ModalConfig
    from verifiers.v1.runtimes.modal import ModalRuntime

    observed = {}

    async def lookup(*args, **kwargs):
        return object()

    async def directory(*args, **kwargs):
        pass

    async def terminate():
        observed["stopped"] = True

    async def create(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            object_id="sb-fixture",
            filesystem=SimpleNamespace(make_directory=SimpleNamespace(aio=directory)),
            terminate=SimpleNamespace(aio=terminate),
        )

    monkeypatch.setattr(modal.App, "lookup", SimpleNamespace(aio=lookup))
    monkeypatch.setattr(modal.Sandbox, "create", SimpleNamespace(aio=create))
    runtime = ModalRuntime(ModalConfig(network_access=network, creates_per_sec=None))
    await runtime.start()
    await runtime.stop_confirmed()
    assert observed["block_network"] is not network
    assert bool(observed["encrypted_ports"]) is network
    assert observed["stopped"]
