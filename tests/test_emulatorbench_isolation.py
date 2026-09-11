"""Opt-in Linux isolation and mailbox integration; no paid provider is used."""

import asyncio
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.autonomous import AutonomousPolicy
from threadweave.evals.autonomous_host import (
    REMOTE_ROOT,
    EvaluationFailure,
    EvidenceJournal,
    ResidentClient,
    agent_source_archive,
)
from threadweave.models import ModelResponse, RunConfig

from .test_evocode_integration import cell

IMAGE = os.environ.get("BUFFALO_EMULATORBENCH_TEST_IMAGE")
pytestmark = pytest.mark.skipif(
    not IMAGE, reason="requires isolated EmulatorBench validation image"
)


class DockerFixture:
    """Test transport only. Official production controller rejects Docker."""

    def __init__(self):
        self.id = None

    async def command(self, *argv, data=None):
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(90):
                stdout, stderr = await process.communicate(data)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return process.returncode, stdout, stderr

    async def start(self):
        code, output, error = await self.command(
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            "1",
            "--memory",
            "2g",
            "--label",
            "buffalo.emulatorbench.validation=true",
            "--name",
            f"buffalo-emulatorbench-test-{uuid.uuid4().hex[:12]}",
            IMAGE,
            "sleep",
            "infinity",
        )
        assert code == 0, error
        self.id = output.decode().strip()
        await self.write(f"{REMOTE_ROOT}/source.tar.gz", agent_source_archive())
        result = await self.run(
            ["tar", "-xzf", f"{REMOTE_ROOT}/source.tar.gz", "-C", f"{REMOTE_ROOT}/source"], {}
        )
        assert result.exit_code == 0

    async def run(self, argv, env):
        args = ["docker", "exec", "-w", "/workspace"]
        for key, value in env.items():
            args.extend(["-e", f"{key}={value}"])
        code, out, err = await self.command(*args, self.id, *argv)
        return SimpleNamespace(exit_code=code, stdout=out.decode(), stderr=err.decode())

    async def write(self, path, data):
        code, _, error = await self.command(
            "docker",
            "exec",
            "-i",
            self.id,
            "python3",
            "-c",
            "import pathlib,sys;p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(sys.stdin.buffer.read())",
            path,
            data=data,
        )
        assert code == 0, error

    async def read(self, path):
        code, out, error = await self.command("docker", "exec", self.id, "cat", path)
        assert code == 0, error
        return out

    async def run_background(self, argv, env, log):
        import shlex

        result = await self.run(
            ["sh", "-c", f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"], env
        )
        assert result.exit_code == 0

    async def close(self):
        if self.id:
            code, _, error = await self.command("docker", "rm", "-f", self.id)
            assert code == 0, error
            self.id = None

    async def stop_confirmed(self):
        await self.close()


async def test_isolated_root_shell_python_same_kernel_fail_fix_pass(tmp_path):
    from emulatorbench.emulator_common.runtime import (
        export_submission_archive,
        stage_submission_archive,
    )

    candidate, verifier = DockerFixture(), DockerFixture()
    client = None
    journal = EvidenceJournal(tmp_path / "evidence")
    calls, attempts = [], []
    hidden_host_name = str(tmp_path / "secret-reference-answer.txt")
    (tmp_path / "secret-reference-answer.txt").write_text("2\n")

    class ScriptedProvider:
        repaired = False

        async def invoke(self, request, emit):
            calls.append(request)
            if len(calls) == 1:
                return cell(
                    "from pathlib import Path\nimport os\npersistent_value = 937\n"
                    f"assert not Path({hidden_host_name!r}).exists()\n"
                    "for base in ['/workspace', '/opt', '/root', '/tmp', '/home']:\n"
                    "    for folder, dirs, files in os.walk(base):\n"
                    "        assert 'secret-reference-answer.txt' not in files\n"
                    "assert not Path('/var/run/docker.sock').exists()\n"
                    "shell = await bash('find / -name secret-reference-answer.txt 2>/dev/null; true')\n"
                    "assert not shell.stdout.strip()\n"
                    "Path('emulator.py').write_text('print(1)\\n')\n"
                )
            if "Autonomous quality gate failed" in str(request.messages) and not self.repaired:
                self.repaired = True
                return cell(
                    "assert persistent_value == 937\nfrom pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
                )
            return ModelResponse(text="Ready for validation")

    provider = ScriptedProvider()
    policy = AutonomousPolicy()

    async def gate(**args):
        snapshot = await export_submission_archive(candidate)
        await stage_submission_archive(verifier, snapshot.sanitized_archive)
        observed = await verifier.run(
            [
                "python3",
                "-c",
                "import pathlib,subprocess;expected=pathlib.Path('/grader/secret-reference-answer.txt').read_text().strip();actual=subprocess.check_output(['python3','/workspace/emulator.py']).decode().strip();raise SystemExit(0 if actual==expected else 1)",
            ],
            {},
        )
        passed = observed.exit_code == 0
        attempts.append(args["identity"])
        return {
            "passed": passed,
            "exit_text": "passed" if passed else "failed",
            "output": "Emulator output is incorrect." if not passed else "Passed.",
            "host_attempt_id": str(len(attempts)),
        }

    try:
        await candidate.start()
        await verifier.start()
        await verifier.write("/grader/secret-reference-answer.txt", b"2\n")
        inspected = json.loads((await candidate.command("docker", "inspect", candidate.id))[1])[0]
        assert not inspected["Mounts"]
        assert inspected["HostConfig"]["NetworkMode"] == "none"
        assert candidate.id != verifier.id
        result = await candidate.run(
            [
                "sh",
                "-c",
                "git init -q && git -c user.name=Test -c user.email=test@example.invalid commit --allow-empty -qm initial",
            ],
            {},
        )
        assert result.exit_code == 0
        config = RunConfig(
            provider={
                "name": "scripted",
                "model": "fixture",
                "model_metadata": {"maxTokens": 8192},
            },
            permissions=[
                "workspace.read",
                "workspace.write",
                "python",
                "process",
                "agents",
                "state",
            ],
            limits={"wall_seconds": 120, "max_turns": 100, "token_budget": 3000000},
        )
        client = ResidentClient(config, journal, gate, providers={"scripted": provider})
        await client.start(candidate, "Implement the emulator.", install=False)
        async with asyncio.timeout(90):
            packet = await client.peer.call(
                "run",
                instruction="Implement the emulator.",
                round_name="fixture verifier",
                policy=asdict(policy),
            )
        assert len(attempts) == 2
        assert len({a["root_session_id"] for a in attempts}) == 1
        assert len({a["kernel_pid"] for a in attempts}) == 1
        assert packet["autonomous"]["continuations"] == 1
        assert packet["autonomous"]["stop_reason"] == "passed"
        assert not any(e["type"] == "python_error" for e in packet["events"])
        assert any(e["type"] == "evaluation_feedback" for e in journal.events)
        assert len(journal.calls) >= 4
    finally:
        if client:
            await client.close()
        await candidate.close()
        await verifier.close()


async def test_actual_official_verifier_on_official_starter_and_downloaded_sources(
    tmp_path, monkeypatch
):
    """Exercises the released verifier; untrusted runner-v1 must not earn credit."""
    from emulatorbench.emulator_common.runtime import score_workspace
    from verifiers.v1.trace import Trace

    from threadweave.evals.emulatorbench import OfficialVerifier, discover_tasks
    from threadweave.evals.emulatorbench_setup import controller_environment

    task = discover_tasks()[0]
    runtime = DockerFixture()
    cache = await asyncio.to_thread(Path(".emulatorbench/public-sources").resolve)
    if not cache.exists():
        pytest.skip("run official public source prefetch first")
    previous = os.environ.get("EMULATORBENCH_PUBLIC_SOURCE_CACHE")
    os.environ["EMULATORBENCH_PUBLIC_SOURCE_CACHE"] = str(cache)
    info = {}
    for key, value in controller_environment(tmp_path / "controller").items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EMULATORBENCH_RUNNER_V2_CORPUS_ROOT", raising=False)
    trace = Trace(task={"type": "emulatorbench:EmulatorBenchTask", "data": task.data})
    task.config.output_artifact_dir = str(tmp_path / "official-artifacts")
    try:
        await runtime.start()
        await task.setup(trace, runtime)
        # The release's starter has no Cargo.lock. Make the genuine std-only
        # starter buildable without solving it, so the real runner also executes.
        lock = await runtime.run(["cargo", "generate-lockfile", "--offline"], {})
        assert lock.exit_code == 0, lock.stderr
        async with asyncio.timeout(120):
            reward = await score_workspace(
                runtime,
                task.data,
                info,
                private_artifact_dir=None,
                require_private_artifacts=False,
                network_access=False,
            )
        assert reward == 0.0
        assert info["emulatorbench_score"]["trusted_oracle_verified"] is False
        assert info["emulatorbench_score"]["runner_protocol"] == "emulator-runner-v1"
        assert info["emulatorbench_score"]["public_expected_unit_count"] == 8
        assert info["emulatorbench_score"]["build"]["cargo_build"]["ok"]
        (tmp_path / "official-verifier-evidence.json").write_text(json.dumps(info, indent=2))
        if receipt := os.environ.get("BUFFALO_EMULATORBENCH_VERIFIER_RECEIPT"):
            await asyncio.to_thread(Path(receipt).write_text, json.dumps(info, indent=2))
        journal = EvidenceJournal(tmp_path / "bridge-evidence")
        bridge = OfficialVerifier(task, runtime, trace, journal, AutonomousPolicy())
        with pytest.raises(EvaluationFailure, match="withheld untrusted grading"):
            await bridge(bridge.command, 300, {"fixture_runtime": runtime.id}, None)
        assert journal.attempts[0]["status"] == "INFRASTRUCTURE_FAILURE"
        assert journal.attempts[0]["score"] is None
        assert "unsupported" in str(journal.attempts[0]["official_info"])
    finally:
        if previous is None:
            os.environ.pop("EMULATORBENCH_PUBLIC_SOURCE_CACHE", None)
        else:
            os.environ["EMULATORBENCH_PUBLIC_SOURCE_CACHE"] = previous
        await runtime.close()


async def test_complete_adapter_serializes_fixture_task_and_aggregate(tmp_path, monkeypatch):
    """Fixture grading only, through evaluate_task's real orchestration and transport."""
    import verifiers.v1.runtimes as runtimes
    from emulatorbench.emulator_common.runtime import stage_submission_archive

    from threadweave.evals.emulatorbench import discover_tasks, evaluate_task
    from threadweave.evals.emulatorbench_report import aggregate, audit_directory, monitor_snapshot

    task = discover_tasks()[0]
    task.data = task.data.model_copy(update={"task_id": "fixture-emulatorbench"})
    candidate, verifier = DockerFixture(), DockerFixture()
    candidate.info = SimpleNamespace(model_dump=lambda: {"id": candidate.id, "fixture": True})

    def runtime_factory(config, name):
        candidate.config = config
        return candidate

    async def fixture_setup(trace, runtime):
        await runtime.write("/workspace/emulator.py", b"print(1)\n")

    async def fixture_controller(trace, runtime, archive, metadata, **kwargs):
        assert kwargs["expose_feedback"] is False
        await stage_submission_archive(verifier, archive)
        result = await verifier.run(
            [
                "python3",
                "-c",
                "import subprocess;assert subprocess.check_output(['python3','/workspace/emulator.py']).strip()==b'2'",
            ],
            {},
        )
        passed = result.exit_code == 0
        record = {"score": 1.0 if passed else 0.0, "passed": passed}
        return record, {
            "emulatorbench_score": {**record, "trusted_oracle_verified": True},
            "fixture_only": True,
        }

    class Scripted:
        repaired = False

        async def invoke(self, request, emit):
            if "Autonomous quality gate failed" in str(request.messages) and not self.repaired:
                self.repaired = True
                return cell(
                    "from pathlib import Path\nPath('emulator.py').write_text('print(2)\\n')"
                )
            return ModelResponse(text="Ready for fixture validation")

    start = ResidentClient.start

    async def preinstalled(self, runtime, instruction):
        return await start(self, runtime, instruction, install=False)

    monkeypatch.setattr(runtimes, "make_runtime", runtime_factory)
    monkeypatch.setattr(task, "setup", fixture_setup)
    monkeypatch.setattr(task, "_validate_controller_submission", fixture_controller)
    monkeypatch.setattr(ResidentClient, "start", preinstalled)
    config = {
        "run": {
            "provider": {
                "name": "scripted",
                "model": "fixture",
                "model_metadata": {"maxTokens": 8192},
            },
            "limits": {"wall_seconds": 120},
        },
        "runtime": {"type": "prime"},
    }
    directory = tmp_path / "fixture-emulatorbench"
    try:
        await verifier.start()
        result = await evaluate_task(task, directory, config, providers={"scripted": Scripted()})
        assert result["status"] == "BENCHMARK_RESULT", result
        assert result["passed"] and result["verifier_attempts"] == 2
        assert result["autonomous"]["continuations"] == 1
        assert candidate.id is None  # Actual confirmed-cleanup path ran.
        for name in (
            "raw-events.jsonl",
            "model-calls.jsonl",
            "verifier-attempts.jsonl",
            "rlm-events.jsonl",
            "refinement-events.jsonl",
            "task-result.json",
            "task-audit.json",
            "initial-workspace.tar.gz",
        ):
            assert (directory / name).is_file()
        assert json.loads((directory / "manifest.json").read_text())["candidate_teardown_confirmed"]
        assert audit_directory(directory)["persistence"]["invariants"]["kernel_id"]
        assert aggregate([result], ["fixture-emulatorbench"])["official_mean_score"] == 1
        assert "NOT_RUN" in monitor_snapshot(tmp_path)  # Fixture is not one of the selected four.
    finally:
        await candidate.close()
        await verifier.close()
