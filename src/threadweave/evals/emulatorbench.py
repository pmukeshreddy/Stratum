"""Pinned official EmulatorBench adapter with isolated host-owned verification.

Run ``preflight`` before ``run``. Missing/provisional corpora never become scores.
The paid launch refuses to start any model until all four tasks pass preflight.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from ..autonomous import AutonomousPolicy, bounded_output
from ..models import RunConfig
from .autonomous_host import (
    EvaluationFailure,
    EvidenceJournal,
    ResidentClient,
    agent_source_archive,
    checked_run,
)
from .schema import file_digest, save, timestamp

PIN_PATH = Path(__file__).with_name("emulatorbench_sources.json")


def pins():
    return json.loads(PIN_PATH.read_text())


def discover_tasks():
    import verifiers
    from emulatorbench import EmulatorBenchConfig, EmulatorBenchTaskset
    from emulatorbench.taskset import PACKAGE_DIR

    pin = pins()
    verifier_dir = Path(verifiers.__file__).parent
    verifier_hashes = dict(pin["verifiers"]["files"])
    for patch in pin["verifiers"].get("patches", []):
        if verifier_hashes.get(patch["file"]) != patch["original_sha256"]:
            raise EvaluationFailure(
                "ADAPTER_FAILURE", "Verifiers compatibility patch baseline changed"
            )
        verifier_hashes[patch["file"]] = patch["patched_sha256"]
    mismatched_verifiers = [
        name
        for name, expected in verifier_hashes.items()
        if not (verifier_dir / name).is_file() or file_digest(verifier_dir / name) != expected
    ]
    if mismatched_verifiers:
        raise EvaluationFailure(
            "ADAPTER_FAILURE", f"Verifiers source pin mismatch: {mismatched_verifiers}"
        )
    mismatches = [
        name
        for name, expected in pin["benchmark"]["files"].items()
        if not (PACKAGE_DIR / name).is_file() or file_digest(PACKAGE_DIR / name) != expected
    ]
    if mismatches:
        raise EvaluationFailure(
            "ADAPTER_FAILURE", f"Official EmulatorBench source pin mismatch: {mismatches}"
        )
    tasks = EmulatorBenchTaskset(
        EmulatorBenchConfig(
            task={
                "public_feedback": False,
                "anti_cheat_validation": True,
            }
        )
    ).load()
    ids = [t.data.task_id for t in tasks]
    if len(ids) != 16 or len(set(ids)) != 16 or ids != pin["benchmark_tasks"]:
        raise EvaluationFailure("ADAPTER_FAILURE", "Canonical 16-task inventory changed")
    if pin["selection_rule"] != "first_four_canonical" or pin["selected_tasks"] != ids[:4]:
        raise EvaluationFailure("ADAPTER_FAILURE", "Precommitted four-task selection changed")
    return tasks


def input_hashes(task):
    from emulatorbench.taskset import TASKS_DIR

    root = TASKS_DIR / task.manifest.slug
    return {
        str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*")) if p.is_file()
    }


def preflight(config):
    from emulatorbench.emulator_common.runner_v2 import runner_v2_manifest_status
    from emulatorbench.emulator_common.runner_v2_scorer import open_installed_corpora_for_manifest

    from .emulatorbench_setup import configure_controller

    configure_controller(config)

    tasks = discover_tasks()[:4]
    mode = config.get("verification_mode", "signed_controller")
    result = {
        "checked_at": timestamp(),
        "sources": pins(),
        "tasks": [],
        "blockers": [],
        "verification_mode": mode,
    }
    if mode not in {"signed_controller", "public_source"}:
        result["blockers"].append(f"Unsupported verification mode: {mode}")
    if mode == "public_source":
        from .emulatorbench_public import public_source_preflight

        public_source_preflight(config, tasks, result)
    for task in tasks:
        if mode == "public_source":
            continue
        rows = runner_v2_manifest_status(task.data.manifest)
        row = {
            "task_id": task.data.task_id,
            "input_hashes": input_hashes(task),
            "declarations": rows,
            "corpus_authenticated": False,
        }
        ported = [r for r in rows if r["ported"]]
        if not ported:
            result["blockers"].append(
                f"{task.data.task_id}: upstream has no installed trusted corpus; provisional/blocked declarations cannot be promoted by this adapter"
            )
        try:
            root, key = task._runner_v2_corpus_material()
            if root is None:
                raise ValueError(
                    "EMULATORBENCH_RUNNER_V2_CORPUS_ROOT is unset; release does not publish a corpus download URL"
                )
            corpora = open_installed_corpora_for_manifest(
                manifest=task.data.manifest, root=root, public_key_pem=key
            )
            try:
                if not corpora:
                    raise ValueError("official loader found no authenticated, installed corpus")
                row["corpus_authenticated"] = True
                row["authenticated_declarations"] = [c.declaration for c in corpora]
            finally:
                for corpus in corpora:
                    corpus.close()
        except Exception as exc:
            result["blockers"].append(f"{task.data.task_id}: {exc}")
        result["tasks"].append(row)
    signer = os.environ.get("EMULATORBENCH_CONTROLLER_SIGNER", "")
    if mode != "public_source" and (
        not signer or not Path(signer).is_absolute() or not os.access(signer, os.X_OK)
    ):
        result["blockers"].append(
            "External controller signer is unavailable (EMULATORBENCH_CONTROLLER_SIGNER)"
        )
    try:
        from emulatorbench.emulator_common.controller_protocol import (
            load_controller_public_key_from_env,
        )

        if mode != "public_source":
            load_controller_public_key_from_env("EMULATORBENCH_CONTROLLER_PUBLIC_KEY")
    except Exception as exc:
        result["blockers"].append(f"Controller public key unavailable: {exc}")
    kind = config.get("runtime", {}).get("type", "prime")
    allowed_runtime = {
        "type",
        "network_access",
        "vm",
        "guaranteed",
        "region",
        "creates_per_min",
        "creates_per_sec",
    }
    if unexpected := set(config.get("runtime", {})) - allowed_runtime:
        result["blockers"].append(
            f"Runtime overrides may not replace official image/resources/workdir or add mounts: {sorted(unexpected)}"
        )
    if kind not in {"prime", "modal"}:
        result["blockers"].append(
            "Official controller supports Prime or Modal only; Docker is test-only"
        )
    if kind == "prime" and not os.environ.get("PRIME_API_KEY"):
        result["blockers"].append("Prime sandbox authentication unavailable: PRIME_API_KEY")
    if kind == "modal":
        try:
            from modal.config import Config

            modal = Config()
            if not modal.get("token_id") or not modal.get("token_secret"):
                result["blockers"].append(
                    "Modal sandbox authentication unavailable; use Modal's documented token setup"
                )
        except ImportError:
            result["blockers"].append("Optional Modal SDK is not installed")
    run = RunConfig.model_validate(config["run"])
    if (
        run.provider.model != "gpt-6-astra"
        or run.provider.parameters.get("reasoning_effort") != "xhigh"
    ):
        result["blockers"].append(
            "Experiment requires gpt-6-astra with the requested xhigh reasoning setting"
        )
    if config.get("task_concurrency", 1) != 1:
        result["blockers"].append("This precommitted experiment uses task_concurrency=1")
    try:
        policy = AutonomousPolicy(**config.get("autonomous", {}))
        if not policy.enabled:
            result["blockers"].append("Autonomous continuation must be enabled")
    except (TypeError, ValueError) as exc:
        result["blockers"].append(f"Invalid autonomous policy: {exc}")
    if (
        run.control_plane != "python"
        or not run.features.persistent_repl
        or not run.features.subagents
        or not run.refinement.enabled
        or not run.refinement.compact
        or run.refinement.turn_interval != 25
        or run.refinement.cooldown_seconds != 1200
    ):
        result["blockers"].append(
            "Resident Buffalo requires its native Python/RLM/refinement configuration"
        )
    result["ready"] = not result["blockers"]
    return result


def provenance(directory, project):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=project)

    patch = git("diff", "--binary", "HEAD")
    status = git("status", "--porcelain=v1", "-z").decode()
    archive = agent_source_archive()
    (directory / "buffalo.diff").write_bytes(patch)
    (directory / "buffalo-source.tar.gz").write_bytes(archive)
    untracked = {}
    untracked_payload = io.BytesIO()
    untracked_archive = tarfile.open(fileobj=untracked_payload, mode="w:gz")
    for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if name:
            path = project / os.fsdecode(name)
            if path.is_file():
                untracked[os.fsdecode(name)] = file_digest(path)
                untracked_archive.add(path, arcname=os.fsdecode(name), recursive=False)
    untracked_archive.close()
    (directory / "buffalo-untracked.tar.gz").write_bytes(untracked_payload.getvalue())
    return {
        "buffalo_git_sha": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(status),
        "git_status": status,
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_hashes": untracked,
        "untracked_archive_sha256": hashlib.sha256(untracked_payload.getvalue()).hexdigest(),
        "agent_source_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "python": sys.version,
        "host_platform": sys.platform,
        "installed_packages": sorted(
            f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
        ),
    }


class OfficialVerifier:
    """The official signed submission controller, with its feedback projection intact."""

    def __init__(self, task, runtime, trace, journal, policy):
        self.task, self.runtime, self.trace = task, runtime, trace
        self.journal, self.policy = journal, policy
        self.command = "official EmulatorBench signed controller"
        self.active = False

    async def verify(self, attempt):
        from emulatorbench.emulator_common.controller_protocol import project_controller_feedback
        from emulatorbench.emulator_common.runtime import export_submission_archive

        snapshot = await export_submission_archive(self.runtime)
        record, info = await self.task._validate_controller_submission(
            self.trace,
            self.runtime,
            snapshot.sanitized_archive,
            snapshot.metadata,
            candidate_archive=snapshot.candidate_archive,
            iteration=attempt["index"],
            expose_feedback=False,
        )
        attempt["official_record"] = record
        attempt["official_info"] = info
        score = info.get("emulatorbench_score", {})
        if not record or score.get("trusted_oracle_verified") is not True:
            raise EvaluationFailure(
                "INFRASTRUCTURE_FAILURE",
                "Official controller withheld untrusted grading; see official_info",
            )
        attempt.update(status="BENCHMARK_RESULT", score=record["score"], passed=record["passed"])
        feedback, _ = project_controller_feedback(score, "diagnostic_summary")
        return record["passed"], feedback

    async def __call__(self, command, timeout_seconds, identity, fingerprint):
        return await self._invoke(command, timeout_seconds, identity, fingerprint)

    async def _invoke(
        self, command, timeout_seconds, identity, fingerprint, *, purpose="final", event_id=None
    ):
        if command != self.command or timeout_seconds != self.policy.gate_timeout_seconds:
            raise EvaluationFailure(
                "ADAPTER_FAILURE", "Worker changed the authorized verifier or timeout"
            )
        rehearsal = purpose == "rehearsal"
        attempts = self.journal.rehearsals if rehearsal else self.journal.attempts
        limit = (
            self.max_rehearsals
            if rehearsal
            else min(self.policy.max_retries, self.policy.max_continuations) + 1
        )
        stream = "rehearsal" if rehearsal else "verifier"
        if self.active or len(attempts) >= limit:
            raise EvaluationFailure(
                "ADAPTER_FAILURE", "Unauthorized concurrent/excess verifier invocation"
            )
        self.active = True
        attempt = {
            "id": uuid.uuid4().hex,
            "index": len(attempts) + 1,
            "purpose": purpose,
            "request_event_id": event_id,
            "started": timestamp(),
            "fingerprint": fingerprint,
            "identity": identity,
            "status": None,
            "score": None,
            "passed": None,
        }
        self.journal.append(f"{stream}-attempt-starts", [attempt])
        try:
            async with asyncio.timeout(timeout_seconds):
                passed, feedback = await self.verify(attempt)
            return {
                "passed": passed,
                "exit_text": "passed" if passed else "failed",
                "output": bounded_output(feedback),
                "host_attempt_id": attempt["id"],
                **(
                    {
                        "exit_code": 0 if passed else 1,
                        "public_source_score": attempt.get("public_source_score"),
                        "submission_sha256": attempt.get("submission_sha256"),
                    }
                    if rehearsal
                    else {}
                ),
            }
        except TimeoutError as exc:
            attempt.update(
                status="TIMEOUT", score=None, passed=None, error="official_verifier_timeout"
            )
            raise EvaluationFailure("TIMEOUT", "official_verifier_timeout") from exc
        except asyncio.CancelledError:
            attempt.update(
                status="TIMEOUT", score=None, passed=None, error="verification_cancelled_by_host"
            )
            raise
        except EvaluationFailure as exc:
            attempt.update(status=exc.category, score=None, passed=None, error=str(exc))
            raise
        except Exception as exc:
            attempt.update(
                status="INFRASTRUCTURE_FAILURE",
                score=None,
                passed=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise EvaluationFailure(
                "INFRASTRUCTURE_FAILURE", "Official controller failed; see attempt log"
            ) from exc
        finally:
            self.active = False
            attempt["ended"] = timestamp()
            attempts.append(attempt)
            self.journal.append(f"{stream}-attempts", [attempt])
            self.journal.status(
                **{
                    f"{stream}_attempts": len(attempts),
                    "latest_rehearsal_score" if rehearsal else "latest_score": attempt.get(
                        "public_source_score", attempt["score"]
                    ),
                }
            )


async def evaluate_task(task, directory, config, *, providers=None):
    from emulatorbench.emulator_common.loader import build_instruction
    from emulatorbench.emulator_common.runtime import export_submission_archive
    from emulatorbench.emulator_common.sandbox import DEFAULT_TOOLCHAIN_IMAGE
    from verifiers.v1.runtimes import ModalConfig, PrimeConfig, make_runtime
    from verifiers.v1.trace import Trace

    from .emulatorbench_report import audit_task, task_result

    journal = EvidenceJournal(directory)
    run = RunConfig.model_validate(config["run"])
    policy = AutonomousPolicy(**config.get("autonomous", {}))
    runtime_config = {
        "image": DEFAULT_TOOLCHAIN_IMAGE,
        "workdir": "/workspace",
        "network_access": True,
        "cpu": task.data.resources.cpu,
        "memory": task.data.resources.memory,
        "disk": task.data.resources.disk,
        "timeout": task.data.timeout.harness + 1800,
        **config.get("runtime", {}),
    }
    cls = PrimeConfig if runtime_config.get("type", "prime") == "prime" else ModalConfig
    if cls is PrimeConfig:
        runtime_config["labels"] = ["emulatorbench"]
    runtime = make_runtime(
        cls(**runtime_config), name=f"buffalo-emulatorbench-{uuid.uuid4().hex[:12]}"
    )
    # Official grader names include this trace ID plus the controller suffix.
    # Keep the composed name below Modal's 64-character provider limit.
    trace = Trace(
        id=uuid.uuid4().hex[:24],
        task={"type": "emulatorbench:EmulatorBenchTask", "data": task.data},
    )
    # Taskset instances share their supplied config. Keep per-task artifact
    # routing independent even when another task's cleanup is still settling.
    task.config = task.config.model_copy(deep=True)
    task.config.output_artifact_dir = str(journal.directory.resolve() / "official-artifacts")
    public_source = config.get("verification_mode") == "public_source"
    rehearsal = config.get("public_rehearsal", {})
    if rehearsal.get("enabled") and not public_source:
        raise EvaluationFailure("ADAPTER_FAILURE", "Public rehearsal requires public-source mode")
    if public_source:
        from .emulatorbench_public import PublicSourceVerifier

        task.config.anti_cheat_validation = False
        task.config.public_feedback = False
        verifier = PublicSourceVerifier(task, runtime, trace, journal, policy)
        if rehearsal.get("enabled"):
            verifier.max_rehearsals = rehearsal["max_attempts"]
    else:
        verifier = OfficialVerifier(task, runtime, trace, journal, policy)
    client = ResidentClient(run, journal, verifier, providers=providers)
    instruction = (
        build_instruction(task.manifest)
        + (
            "\nThe host runs official verification when work settles and returns bounded feedback "
            "as a user message. Continue within this same workspace when feedback reports failure.\n"
        )
        + ("" if public_source else "Follow RUNNER_V2_CONTRACT.md. ")
        + (
            "Vendor non-std dependencies and preserve .cargo/config.toml, "
            "Cargo.lock, and the vendor tree: grading uses Cargo --offline --locked.\n"
        )
    )
    if rehearsal.get("enabled"):
        from .emulatorbench_public import REHEARSAL_GUIDANCE

        instruction += "\n" + REHEARSAL_GUIDANCE + "\n"
    manifest = {
        "task_id": task.data.task_id,
        "input_hashes": input_hashes(task),
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "run_config": run.model_dump(mode="json"),
        "autonomous": asdict(policy),
        "runtime_config": runtime.config.model_dump(),
        "sources": pins(),
        "started": timestamp(),
        "verifier": verifier.command,
        "verification_mode": config.get("verification_mode", "signed_controller"),
        "public_rehearsal": rehearsal,
        "outer_timeout_seconds": task.data.timeout.harness,
    }
    save(journal.directory / "manifest.json", manifest)
    started, packet, failure = time.monotonic(), None, None
    phase, cancelled = "setup", False
    try:
        async with asyncio.timeout(task.data.timeout.setup):
            await runtime.start()
            manifest["runtime_info"] = runtime.info.model_dump()
            save(journal.directory / "manifest.json", manifest)
            await task.setup(trace, runtime)
            if public_source:
                from emulatorbench.emulator_common.runner_contract import runner_contract

                await runtime.write(
                    "/workspace/RUNNER_CONTRACT.json",
                    json.dumps(runner_contract(), indent=2).encode(),
                )
            if rehearsal.get("enabled"):
                from .emulatorbench_public import stage_public_rehearsal

                manifest["public_rehearsal_files"] = await stage_public_rehearsal(
                    task, runtime, policy.gate_timeout_seconds
                )
            initial = await export_submission_archive(runtime)
            (journal.directory / "initial-workspace.tar.gz").write_bytes(initial.sanitized_archive)
            manifest["initial_workspace"] = initial.metadata
            manifest["initial_workspace_sha256"] = hashlib.sha256(
                initial.sanitized_archive
            ).hexdigest()
            save(journal.directory / "manifest.json", manifest)
            await checked_run(
                runtime,
                [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    "cd /workspace && git init -q && git -c user.name=Evaluator -c user.email=evaluator@example.invalid commit --allow-empty -qm initial",
                ],
            )
            await client.start(runtime, instruction)
            if rehearsal.get("enabled"):
                await client.peer.call(
                    "enable_rehearsal",
                    command=verifier.command,
                    timeout_seconds=policy.gate_timeout_seconds,
                )
        phase = "trajectory"
        async with asyncio.timeout(task.data.timeout.harness):
            packet = await client.peer.call(
                "run", instruction=instruction, round_name=verifier.command, policy=asdict(policy)
            )
    except TimeoutError:
        failure = EvaluationFailure("TIMEOUT", f"{phase}_wall_timeout")
    except asyncio.CancelledError:
        cancelled = True
        failure = EvaluationFailure("INFRASTRUCTURE_FAILURE", "host_cancelled")
    except EvaluationFailure as exc:
        failure = exc
    except Exception as exc:
        failure = client.failure or EvaluationFailure(
            "INFRASTRUCTURE_FAILURE"
            if phase == "setup" or isinstance(exc, ConnectionError)
            else "ADAPTER_FAILURE",
            f"{phase}: {type(exc).__name__}: {exc}",
        )
    finally:
        if packet is None and client.peer:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(20):
                    packet = await client.peer.call("last_round")
        if packet:
            journal.evidence(packet["events"])
            save(journal.directory / "trajectory.json", packet)
        try:
            await client.close()
        except Exception as exc:
            failure = EvaluationFailure("INFRASTRUCTURE_FAILURE", f"worker_teardown_failed: {exc}")
        try:
            async with asyncio.timeout(300):
                await runtime.stop_confirmed()
            manifest["candidate_teardown_confirmed"] = True
        except Exception as exc:
            failure = EvaluationFailure(
                "INFRASTRUCTURE_FAILURE", f"candidate_teardown_unconfirmed: {exc}"
            )
            manifest["candidate_teardown_confirmed"] = False
        manifest["ended"] = timestamp()
        save(journal.directory / "manifest.json", manifest)
        save(journal.directory / "official-trace.json", trace.model_dump(mode="json"))
    result = task_result(task.data.task_id, packet, journal, failure, time.monotonic() - started)
    save(journal.directory / "task-result.json", result)
    save(journal.directory / "task-audit.json", audit_task(packet, journal))
    if cancelled:
        raise asyncio.CancelledError()
    return result


async def run_experiment(config, output):
    from .emulatorbench_report import aggregate

    output.mkdir(parents=True, exist_ok=False)
    project = (await asyncio.to_thread(Path(__file__).resolve)).parents[3]
    run_manifest = {
        "started": timestamp(),
        "selection": {
            k: pins()[k] for k in ("benchmark_tasks", "selection_rule", "selected_tasks")
        },
        "config": config,
        "provenance": provenance(output, project),
        "protocol": "Official EmulatorBench + Buffalo Prime-style host continuation",
        "leaderboard_comparable": False,
    }
    save(output / "run-manifest.json", run_manifest)
    check = preflight(config)
    save(output / "preflight.json", check)
    if not check["ready"]:
        save(
            output / "aggregate.json",
            {
                **aggregate([], pins()["selected_tasks"]),
                "run_status": "INFRASTRUCTURE_FAILURE",
                "stop_reason": "preflight_failed",
                "blockers": check["blockers"],
            },
        )
        save(output / "run-manifest.json", {**run_manifest, "ended": timestamp()})
        raise EvaluationFailure(
            "INFRASTRUCTURE_FAILURE", "Preflight failed before inference; see preflight.json"
        )
    rows = []
    try:
        for task in discover_tasks()[:4]:
            rows.append(await evaluate_task(task, output / task.data.task_id, config))
            save(output / "aggregate.json", aggregate(rows, pins()["selected_tasks"]))
    finally:
        from .emulatorbench_report import report_directory

        report_directory(output)
        save(output / "run-manifest.json", {**run_manifest, "ended": timestamp()})
    return aggregate(rows, pins()["selected_tasks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("discover", "preflight", "run", "report", "monitor", "audit")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/emulatorbench.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        if args.command in {"run", "report", "monitor", "audit"}:
            parser.error(f"{args.command} requires an explicit --output directory")
        args.output = Path(f".emulatorbench/{args.command}.json")
    if args.command == "audit":
        from .emulatorbench_report import audit_directory

        value = audit_directory(args.output)
    elif args.command in {"report", "monitor"}:
        from .emulatorbench_report import monitor, report_directory

        value = monitor(args.output) if args.command == "monitor" else report_directory(args.output)
    elif args.command == "discover":
        discover_tasks()
        value = {k: pins()[k] for k in ("benchmark_tasks", "selection_rule", "selected_tasks")}
        save(args.output, value)
    else:
        config = json.loads(args.config.read_text())
        if args.command == "preflight":
            value = preflight(config)
            save(args.output, value)
        else:
            value = asyncio.run(run_experiment(config, args.output))
    print(json.dumps(value, indent=2))
    if args.command == "preflight" and not value["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
