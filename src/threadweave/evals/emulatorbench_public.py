"""Official public source-verifier mode, using the unchanged fresh-grader path."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import shlex
from pathlib import Path

from .autonomous_host import REMOTE_PYTHON, REMOTE_STATE, EvaluationFailure, checked_run
from .emulatorbench import OfficialVerifier, input_hashes, pins
from .schema import file_digest

REHEARSAL_GUIDANCE = """
Before completing the task, run ./verify-public.sh on the final implementation.
This command snapshots the workspace and rehearses the exact public verifier
used for final grading, with its actual pinned input assets, runner invocations,
build settings, output checks, and replay checks, in a separate environment.
Read verification/PUBLIC_VERIFIER_CONTRACT.md and the public manifest/source
descriptors for input formats, CLI flags, and frame/cycle budgets. Correct the
emulator and runner using the returned diagnostics, and rehearse again after
changes. Local component tests alone do not establish end-to-end compatibility.
Exit 0 means the snapshot passed, 1 means it failed, and 2 means unavailable.
Final grading always checks the final workspace independently. Rehearsals do not
reset or extend the model, task, or autonomous continuation budgets.
""".strip()


def public_contract_files(task):
    """Only source-authorized public descriptors and invocation code, no oracles."""
    from emulatorbench.emulator_common import runners, suite_adapters
    from emulatorbench.emulator_common.feedback import SYSTEM_PROMPT
    from emulatorbench.emulator_common.runner_contract import runner_contract_markdown

    mapping = task.data.model_dump(mode="json")
    functions = (suite_adapters.invoke_case, suite_adapters.replay_determinism)
    contract = REHEARSAL_GUIDANCE + "\n\n" + runner_contract_markdown()
    contract += "\n\nExact invocation/replay code from the pinned public release:\n"
    for function in functions:
        contract += f"\n```python\n{inspect.getsource(function)}```\n"
    return {
        "README.md": SYSTEM_PROMPT + "\n" + REHEARSAL_GUIDANCE + "\n",
        "verification/PUBLIC_VERIFIER_CONTRACT.md": contract,
        "verification/public_manifest.json": runners.public_manifest_file(mapping),
        "verification/public_tests.json": runners.public_tests_file(mapping),
    }


async def stage_public_rehearsal(task, runtime, timeout_seconds):
    files = public_contract_files(task)
    files["verify-public.sh"] = (
        "#!/bin/sh\nexec "
        + shlex.join(
            [
                REMOTE_PYTHON,
                "-m",
                "threadweave.evals.autonomous_rehearsal",
                "--socket",
                f"{REMOTE_STATE}/public-rehearsal.sock",
                "--timeout",
                str(timeout_seconds + 30),
            ]
        )
        + ' "$@"\n'
    )
    for name, value in files.items():
        await runtime.write(f"/workspace/{name}", value.encode())
    await checked_run(runtime, ["chmod", "+x", "/workspace/verify-public.sh"])
    import hashlib

    return {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}


def public_source_preflight(config, tasks, result):
    """Run the pinned release's source readiness checks before any model call."""
    source = Path(config.get("benchmark_checkout", ".emulatorbench/sources/prime-envs"))
    package = source / pins()["benchmark"]["package"]
    script = package / "scripts/prefetch_public_sources.py"
    if not script.is_file() or file_digest(script) != pins()["benchmark"]["setup_script_sha256"]:
        result["blockers"].append("Official source readiness script is unavailable or changed")
        return
    rehearsal = config.get("public_rehearsal", {})
    if rehearsal.get("enabled") and (
        type(rehearsal.get("max_attempts")) is not int or rehearsal["max_attempts"] <= 0
    ):
        result["blockers"].append("Public rehearsal requires a positive max_attempts")
    cache = Path(config.get("public_source_cache", ".emulatorbench/public-sources")).resolve()
    os.environ["EMULATORBENCH_PUBLIC_SOURCE_CACHE"] = str(cache)
    spec = importlib.util.spec_from_file_location("official_emulatorbench_source_readiness", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result["public_source_cache"] = str(cache)
    for task in tasks:
        row = {"task_id": task.data.task_id, "input_hashes": input_hashes(task)}
        try:
            if any(task._runner_v2_corpus_material()):
                raise ValueError(
                    "Public-source mode cannot also configure runner-v2 corpus material"
                )
            row["public_source_readiness"] = module.prepare_platform(
                package, task.manifest.slug, cache, 240
            )
        except Exception as exc:
            result["blockers"].append(f"{task.data.task_id}: public sources unavailable: {exc}")
        result["tasks"].append(row)


def public_feedback(score, summary):
    """Bound public diagnostics without exposing expected states or host files."""
    from emulatorbench.emulator_common.controller_protocol import project_controller_feedback

    # The original source-verifier summary has already been checked by the
    # official runtime against the full result digest. Never replace the saved
    # official reward with this separately named measurement.
    projection, _ = project_controller_feedback({**score, **summary}, "full")
    failures = []
    for row in score.get("case_runs", []):
        if isinstance(row, dict) and not row.get("passed"):
            failures.append(
                {
                    key: row[key]
                    for key in ("case_id", "case_name", "returncode", "error")
                    if key in row
                }
                | {
                    "input_file": Path(row.get("artifact", "")).name,
                    "observed": {
                        key: row.get("output", {}).get(key)
                        for key in ("cycles", "frames", "passed", "serial", "parse_error")
                        if key in row.get("output", {})
                    },
                    "stderr": str(row.get("stderr", ""))[-800:],
                }
            )
    details = {"failed_public_cases": failures[:12]}
    # Compiler diagnostics describe the submitted code, not benchmark oracles.
    build = score.get("build", {})
    details["build"] = {
        key: {name: value.get(name) for name in ("ok", "returncode", "stderr") if name in value}
        for key, value in build.items()
        if isinstance(value, dict) and not value.get("ok")
    }
    return projection + "\n" + json.dumps(details, ensure_ascii=True)


class PublicSourceVerifier(OfficialVerifier):
    def __init__(self, *args):
        super().__init__(*args)
        self.command = "official EmulatorBench public source verifier in fresh isolated grader"
        self.max_rehearsals = 0

    async def rehearse(self, command, timeout_seconds, identity, fingerprint, event_id):
        if not self.max_rehearsals:
            raise EvaluationFailure("ADAPTER_FAILURE", "Public rehearsal is disabled")
        if command != self.command or timeout_seconds != self.policy.gate_timeout_seconds:
            raise EvaluationFailure("ADAPTER_FAILURE", "Rehearsal contract changed")
        previous = self.journal.rehearsals[-1] if self.journal.rehearsals else None
        if (
            previous
            and previous.get("status") == "BENCHMARK_RESULT"
            and previous.get("passed") is False
            and fingerprint is not None
            and previous["fingerprint"] == fingerprint
        ):
            result = {
                "exit_code": 1,
                "passed": False,
                "suppressed": True,
                "host_attempt_id": previous["id"],
                "output": "The relevant workspace has not changed since the previous failed public rehearsal.",
            }
            self.journal.append("rehearsal-suppressed", [{"request_event_id": event_id, **result}])
            return result
        if len(self.journal.rehearsals) >= self.max_rehearsals:
            return {"exit_code": 2, "error": "Public rehearsal attempt limit reached."}
        return await self._invoke(
            command, timeout_seconds, identity, fingerprint, purpose="rehearsal", event_id=event_id
        )

    async def verify(self, attempt):
        from emulatorbench.emulator_common.runtime import (
            _verifier_summary,
            export_submission_archive,
        )

        if self.task.config.anti_cheat_validation or any(self.task._runner_v2_corpus_material()):
            raise EvaluationFailure(
                "ADAPTER_FAILURE", "Public source verifier configuration changed"
            )
        snapshot = await export_submission_archive(self.runtime)
        attempt["submission_sha256"] = snapshot.metadata["sha256"]
        attempt["candidate_archive_sha256"] = snapshot.metadata["candidate_archive_sha256"]
        info = {}
        attempt["official_info"] = info
        attempt["verification_mode"] = "public_source"
        reward = await self.task._run_fresh_grader(
            self.trace,
            self.runtime,
            snapshot.sanitized_archive,
            score_info=info,
            grader_suffix=f"public-{attempt.get('purpose', 'final')}-{attempt['index']}",
            candidate_archive_sha256=snapshot.metadata["candidate_archive_sha256"],
            submission_sha256=snapshot.metadata["sha256"],
        )
        score = info.get("emulatorbench_score", {})
        lifecycle = info.get("emulatorbench_grader_lifecycle", {})
        if (
            lifecycle.get("status") != "completed"
            or lifecycle.get("stopped") is not True
            or lifecycle.get("egress_probe") != "blocked"
            or score.get("error")
            or score.get("details_error")
            or any(not source.get("ok") for source in score.get("public_sources", []))
            or score.get("runner_protocol") != "emulator-runner-v1"
            or score.get("trusted_oracle_verified") is not False
        ):
            raise EvaluationFailure(
                "INFRASTRUCTURE_FAILURE",
                "Official source verification was incomplete; see official_info",
            )
        summary = _verifier_summary(info.get("emulatorbench_verifier_stdout", ""))
        value = score.get("untrusted_development_score")
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not 0 <= value <= 1
            or summary["score"] != value
            or summary["result_sha256"] != score.get("result_sha256")
            or type(summary["passed"]) is not bool
            or reward != score.get("score")
        ):
            raise EvaluationFailure(
                "ADAPTER_FAILURE", "Official source score/summary binding is inconsistent"
            )
        attempt.update(
            status="BENCHMARK_RESULT",
            score=reward,
            official_passed=score["passed"],
            passed=summary["passed"],
            public_source_score=value,
            trusted_oracle_verified=False,
            original_verifier_summary=summary,
        )
        return summary["passed"], public_feedback(score, summary)
