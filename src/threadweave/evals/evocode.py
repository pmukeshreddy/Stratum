"""Pinned Harbor multi-step integration with persistent Buffalo and isolated grading.

Use the module CLI, not stock ``harbor run``: the adapted trial owns verifier
callbacks and removes Harbor's agent-visible verifier log mount.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import ExceptionInfo, TimingInfo
from harbor.tasks.client import TaskDownloadResult
from harbor.trial.multi_step import MultiStepTrial
from harbor.verifier.verifier import Verifier

from ..autonomous import AutonomousPolicy
from ..models import ModelRequest, ProviderConfig, RunConfig
from ..providers import default_providers
from .evocode_report import round_report, task_report
from .evocode_rpc import Peer
from .schema import file_digest, save

HARBOR_COMMIT = "191d1b989bbba1d77c2db23e17aec308d7c08046"
REMOTE_ROOT = "/opt/buffalo-evaluation"
REMOTE_PYTHON = f"{REMOTE_ROOT}/venv/bin/python"
STATE_DIR = f"{REMOTE_ROOT}/state"


async def command(*argv, timeout_seconds=600):
    proc = await asyncio.create_subprocess_exec(
        *map(str, argv),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            stdout, stderr = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(
                f"{argv[0]} exited {proc.returncode}: {stderr.decode(errors='replace')[-4000:]}"
            )
        return stdout.decode()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def task_hashes(path):
    # Hash every official task input; never rewrite, append guidance, or stage solutions.
    return {
        str(p.relative_to(path)): file_digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and (p.relative_to(path).parts[0] in {"steps", "environment"} or p.name == "task.toml")
    }


def check_harbor_source():
    import harbor

    provenance = json.loads(Path(__file__).with_name("evocode_sources.json").read_text())
    root = Path(harbor.__file__).parent
    mismatches = [
        path
        for path, expected in provenance["harbor"]["files"].items()
        if not (root / path).is_file() or file_digest(root / path) != expected
    ]
    if mismatches:
        raise ValueError(
            f"Install the pinned Harbor commit {HARBOR_COMMIT}; source mismatch: {mismatches}"
        )
    return provenance


def diagnostic(rewards, raw):
    """Disclosure allowlist: reward status + numeric aggregate, never raw test output."""
    matches = re.findall(
        r"(?m)^CASE_SUMMARY total_cases=(\d+) success_count=(\d+)(?: fail_count=\d+)?[ \t]*$",
        raw,
    )
    summary = None
    if matches:
        total, success = map(int, matches[-1])
        if 0 <= success <= total and total <= 1_000_000:
            summary = {"total_cases": total, "success_count": success}
    passed = rewards == {"reward": 1} or rewards == {"reward": 1.0}
    output = "Cumulative verification passed." if passed else "Cumulative verification failed."
    if summary:
        output += f"\nCases passed: {summary['success_count']}/{summary['total_cases']}."
    return {
        "passed": passed,
        "exit_text": "passed" if passed else "returned reward 0",
        "output": output,
    }, summary


class BuffaloEvoCodeAgent(BaseAgent):
    capabilities = AgentCapabilities(resume=True)

    @staticmethod
    def name():
        return "buffalo-evocode"

    def version(self):
        return importlib.metadata.version("threadweave")

    async def setup(self, environment):
        if not hasattr(self, "controller"):
            raise ValueError(
                "Use BuffaloEvoCodeTrial: stock Harbor cannot supply isolated retry feedback"
            )
        if getattr(self, "peer", None):
            raise RuntimeError("Buffalo setup may only run once per trial")
        if not isinstance(environment, DockerEnvironment):
            raise ValueError("This pinned integration supports Linux Docker only")
        self.environment = environment
        self.container = await environment._platform._resolve_service_container(MAIN_SERVICE_NAME)
        inspection = json.loads(await command("docker", "inspect", self.container))[0]
        forbidden_mounts = [
            m
            for m in inspection["Mounts"]
            if m["Destination"] not in {"/logs/agent", "/logs/artifacts"}
        ]
        if forbidden_mounts or inspection["HostConfig"].get("Privileged"):
            raise ValueError(
                "Unsupported agent mounts or privileged container; grader isolation cannot be proven"
            )
        self.workspace = inspection["Config"]["WorkingDir"]
        if not self.workspace or self.workspace == "/":
            raise ValueError("The official task image must declare a dedicated WORKDIR")
        self.controller.manifest["container_id"] = inspection["Id"]
        self.controller.manifest["base_image_id"] = inspection["Image"]
        self.controller.manifest["mounts"] = inspection["Mounts"]
        self.controller.manifest["workspace"] = self.workspace
        task_path = next(
            (e[5:] for e in inspection["Config"]["Env"] if e.startswith("PATH=")), "/usr/bin:/bin"
        )
        self.controller.write_manifest()
        await self.install(environment)
        self.stderr_file = (self.controller.output / "worker-stderr.txt").open("wb")
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "--workdir",
            self.workspace,
            "--env",
            f"PATH={REMOTE_ROOT}/node/bin:{task_path}",
            self.container,
            REMOTE_PYTHON,
            "-u",
            "-m",
            "threadweave.evals.evocode_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr_file,
            limit=64 * 1024 * 1024,
        )

        async def send(data):
            self.process.stdin.write(data)
            await self.process.stdin.drain()

        self.providers = getattr(self.controller, "providers", None) or default_providers()
        self.peer = Peer(self.process.stdout, send, self.handle)
        self.root_audit = await self.peer.call(
            "setup",
            instruction=self.controller.first_instruction,
            workspace=self.workspace,
            directory=STATE_DIR,
            config=self.controller.run_config.model_dump(mode="json"),
        )
        self.root_id = self.root_audit["root_session_id"]
        save(self.controller.output / "setup-audit.json", self.root_audit)

    async def install(self, environment):
        # Tooling goes outside the evolving project. Only Buffalo source is uploaded.
        package = (await asyncio.to_thread(Path(__file__).resolve)).parents[1]
        project = package.parents[1]
        await environment.exec(f"mkdir -p {REMOTE_ROOT}/source/src", user="root")
        await environment.upload_dir(package, f"{REMOTE_ROOT}/source/src/threadweave")
        await environment.upload_file(
            project / "pyproject.toml", f"{REMOTE_ROOT}/source/pyproject.toml"
        )
        # uv's managed Python works on the release's Ubuntu 22.04 images. Node is
        # the actual formatter dependency; avoid changing the project's interpreter.
        uv = f"UV_PYTHON_INSTALL_DIR={REMOTE_ROOT}/python {REMOTE_ROOT}/uv/bin/uv"
        commands = [
            f"python3 -m pip install --target {REMOTE_ROOT}/uv uv==0.11.8",
            f"{uv} python install 3.12.12 --no-bin",
            f"{uv} venv --python 3.12.12 --managed-python {REMOTE_ROOT}/venv",
            f"{REMOTE_ROOT}/uv/bin/uv pip install --python {REMOTE_PYTHON} {REMOTE_ROOT}/source",
            f"curl -fsSL https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-$(uname -m | sed 's/x86_64/x64/;s/aarch64/arm64/').tar.gz -o {REMOTE_ROOT}/node.tar.gz",
            f"mkdir -p {REMOTE_ROOT}/node && tar -xzf {REMOTE_ROOT}/node.tar.gz --strip-components=1 -C {REMOTE_ROOT}/node",
        ]
        for index, cmd in enumerate(commands):
            result = await environment.exec(cmd, user="root", timeout_sec=600)
            save(
                self.controller.output / f"install-{index}.json",
                {
                    "command": cmd,
                    "return_code": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )
            if result.return_code:
                raise RuntimeError(f"Buffalo dependency setup failed; see install-{index}.json")

    def checked_provider(self, config):
        allowed = [self.controller.run_config.provider, *self.controller.run_config.models.values()]
        if not any(
            (config.name, config.base_url, config.api_key_env)
            == (p.name, p.base_url, p.api_key_env)
            for p in allowed
        ):
            raise ValueError("Worker requested an unconfigured provider endpoint")
        return self.providers[config.name]

    async def handle(self, method, args):
        if method == "models":
            provider = self.providers["codex_subscription"]
            async with provider.control_factory() as control:
                return await control.models()
        if method == "resolve":
            config = ProviderConfig.model_validate(args["config"])
            provider = self.checked_provider(config)
            if hasattr(provider, "resolve"):
                if args["reasoning_off"]:
                    config, details = await provider.resolve(config, reasoning_off=True)
                else:
                    config, details = await provider.resolve(config)
            else:
                details = {}
            return {"config": config.model_dump(mode="json"), "details": details}
        if method == "model":
            request = ModelRequest.model_validate(args["request"])
            provider = self.checked_provider(request.config)
            row = {
                "round": self.controller.step.name,
                "round_index": self.controller.index,
                "request": request.public_dump(),
                "started": time.monotonic(),
            }

            async def emit(delta):
                pass

            try:
                response = await provider.invoke(request, emit)
                row["response"] = response.model_dump(mode="json")
                if getattr(self, "current_context", None) is not None:
                    ctx = self.current_context
                    ctx.n_input_tokens = (ctx.n_input_tokens or 0) + response.usage.input_tokens
                    ctx.n_cache_tokens = (
                        ctx.n_cache_tokens or 0
                    ) + response.usage.cached_input_tokens
                    ctx.n_output_tokens = (ctx.n_output_tokens or 0) + response.usage.output_tokens
                # Opaque provider continuations are excluded from public dumps,
                # but must reach the resident Runtime for its next real request.
                return {**row["response"], "provider_items": response.provider_items}
            except BaseException as exc:
                row["error"] = type(exc).__name__
                raise
            finally:
                row["ended"] = time.monotonic()
                self.controller.calls.append(row)
                with (self.controller.output / "model-calls.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
        if method == "gate":
            if args["command"] != f"<{self.controller.step.name} verifier>":
                raise ValueError("Gate callback is restricted to the active official step")
            if args["timeout_seconds"] != self.controller.policy.gate_timeout_seconds:
                raise ValueError("Worker cannot change the host verifier budget")
            try:
                return await self.controller.verify_candidate(args["timeout_seconds"])
            except Exception as exc:
                from ..models import HarnessError

                save(
                    self.controller.output / f"{self.controller.step.name}-verifier-error.json",
                    {
                        "type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise HarnessError(
                    "environment",
                    "verifier_infrastructure",
                    "Host verifier failed; see host-only logs",
                ) from None
        raise ValueError(f"Unknown host operation: {method}")

    async def run(self, instruction, environment, context: AgentContext):
        if environment is not self.environment:
            raise RuntimeError("Environment changed inside a persistent trial")
        self.current_context = context
        context.metadata = {"root_session_id": self.root_id, "round": self.controller.step.name}
        try:
            packet = await self.peer.call(
                "run",
                instruction=instruction,
                round_name=self.controller.step.name,
                policy=asdict(self.controller.policy),
            )
        except BaseException:
            # Peer cancellation first stops the owned run and invalidates native
            # refinement. Export only after that run releases its worker lock.
            with contextlib.suppress(Exception):
                self.controller.packet = await asyncio.wait_for(
                    self.peer.call("last_round"), timeout=30
                )
            raise
        finally:
            self.current_context = None
        if packet["after"]["root_session_id"] != self.root_id:
            raise RuntimeError("Root session identity changed")
        self.controller.packet = packet
        usage = packet["usage"]
        previous = self.controller.previous_usage
        context.n_input_tokens = usage["input_tokens"] - previous.get("input_tokens", 0)
        context.n_cache_tokens = usage["cached_input_tokens"] - previous.get(
            "cached_input_tokens", 0
        )
        context.n_output_tokens = usage["output_tokens"] - previous.get("output_tokens", 0)
        context.cost_usd = (
            None if usage["cost"] is None else usage["cost"] - (previous.get("cost") or 0)
        )
        context.metadata = {
            "root_session_id": self.root_id,
            "round": self.controller.step.name,
            "protocol": self.controller.manifest["protocol"],
            "trajectory": "buffalo/model-calls.jsonl",
            "stop_reason": packet["autonomous"]["stop_reason"],
        }
        self.controller.previous_usage = usage

    async def resume(self, instruction, environment, context):
        await self.run(instruction, environment, context)

    async def close(self):
        if not getattr(self, "peer", None):
            return
        try:
            await asyncio.wait_for(self.peer.call("shutdown"), timeout=120)
        finally:
            try:
                await self.environment.download_dir(STATE_DIR, self.controller.output / "state")
            finally:
                await self.peer.close()
                self.process.stdin.close()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=10)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                self.stderr_file.close()
                self.peer = None


class BuffaloEvoCodeTrial(MultiStepTrial):
    def __init__(
        self, config, *, run_config, _task, _task_download_result, providers=None, max_steps=None
    ):
        provenance = check_harbor_source()
        if not _task.has_steps or _task.config.environment.os.value != "linux":
            raise ValueError("The integration requires an official Linux multi-step task")
        if max_steps is not None and not 1 <= max_steps <= len(_task.config.steps):
            raise ValueError("max_steps must select a nonempty prefix of the official steps")
        if run_config.task.verifier != "none" or run_config.task.adapter != "workspace":
            raise ValueError(
                "Use the standard workspace adapter; the host owns hidden verification"
            )
        if (
            run_config.execution.backend != "local"
            or run_config.extensions
            or run_config.skill_paths
        ):
            raise ValueError(
                "Use task-local execution with no imported extensions or learned skills"
            )
        if (
            run_config.task.instruction_messages
            or run_config.task.original_messages
            or run_config.task.specification
        ):
            raise ValueError("The official step instructions are the only task prompt")
        if (
            config.environment.type.value != "docker"
            or config.environment.mounts
            or config.environment.extra_docker_compose
        ):
            raise ValueError(
                "Only isolated Docker tasks without user mounts/sidecars are supported"
            )
        if (_task.paths.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Snapshot grading does not support sidecar services")
        if config.extra_instructions or config.extra_instruction_paths or config.verifier.disable:
            raise ValueError(
                "Task instruction overrides and disabled verification are not supported"
            )
        run_config = run_config.model_copy(deep=True)
        run_config.serialized_refine = True
        self.run_config = run_config
        self.providers = providers
        self.max_steps = max_steps
        self.rounds, self.calls, self.previous_usage = [], [], {}
        self.started = time.monotonic()
        self.source_hashes = task_hashes(_task.paths.task_dir)
        super().__init__(config, _task=_task, _task_download_result=_task_download_result)
        self.output = self.paths.trial_dir / "buffalo"
        self.output.mkdir()
        self.agent.controller = self
        self.first_instruction = self.task.step_instruction(self.task.config.steps[0].name)
        self.manifest = {
            "protocol": "EvoCode-Bench + Prime-style autonomous feedback",
            "leaderboard_comparable": False,
            "harbor_commit": HARBOR_COMMIT,
            "harbor_version": importlib.metadata.version("harbor"),
            "task": self.task.name,
            "task_input_sha256": self.source_hashes,
            "run_config": run_config.model_dump(mode="json"),
            "autonomous_defaults": asdict(AutonomousPolicy()),
            "round_policies": {},
            "refinement_policy": run_config.refinement.model_dump(mode="json"),
            "diagnostic_policy": "binary reward and numeric CASE_SUMMARY only; no raw stdout",
            "state_isolation": "fresh container and fresh /opt/buffalo-evaluation/state per task",
            "max_steps": max_steps,
            "score_kind": "smoke" if max_steps else "adapted_evaluation",
            "runtime_tooling": {"python": "3.12.12", "uv": "0.11.8", "node": "22.16.0"},
            "source_provenance": provenance,
        }
        self.write_manifest()

    def write_manifest(self):
        save(self.output / "manifest.json", self.manifest)

    @property
    def _agent_env_mounts(self):
        # Critical: Harbor's default host verifier log bind is otherwise readable
        # even when every verifier runs in a different container.
        return [
            m
            for m in super()._agent_env_mounts
            if m["target"] != str(self.agent_env_paths.verifier_dir)
        ]

    async def _run_step(self, step, step_result, *, index, total):
        self.step, self.index, self.step_result = step, index, step_result
        self.attempts, self.packet = [], None
        self._create_step_dirs(step)
        await self._prepare_step(step, step_result, resume=index > 1)
        if step_result.exception_info:
            return
        self.policy = replace(
            AutonomousPolicy(),
            timeout_seconds=self._step_agent_timeout_sec(step) or 1800,
            gate_timeout_seconds=self._step_verifier_timeout_sec(step) or 300,
        )
        self.manifest["round_policies"][step.name] = {
            **asdict(self.policy),
            "timeout_source": "resolved Harbor task/step agent and verifier timeout_sec",
        }
        self.write_manifest()
        await self._run_step_agent(step, step_result, resume=index > 1)
        if self.packet:
            save(self.output / f"{step.name}-audit.json", self.packet)
            row = round_report(self.task.name, index, self.packet, self.attempts, self.calls)
            self.rounds.append(row)
            save(self.output / f"{step.name}.json", row)
        if self.attempts:
            last = Path(self.attempts[-1]["logs"])
            shutil.copytree(last, self.paths.step_verifier_dir(step.name), dirs_exist_ok=True)
            if self.attempts[-1].get("timed_out") and not step_result.verifier_result:
                step_result.exception_info = ExceptionInfo.from_exception(
                    TimeoutError("Cumulative verifier exhausted retries without a reward")
                )
        self._copy_agent_dir_to_step(step)

    def _should_stop_after_step(self, step, step_result):
        return bool(
            step_result.exception_info
            or (self.packet and self.packet["autonomous"]["stop_reason"] in {"error", "aborted"})
            or (self.max_steps and self.index >= self.max_steps)
            or super()._should_stop_after_step(step, step_result)
        )

    async def verify_candidate(self, timeout_seconds):
        attempt = len(self.attempts) + 1
        image = f"buffalo-verifier-{uuid.uuid4().hex}:candidate"
        paths = TrialPaths(trial_dir=self.output / "attempts" / self.step.name / str(attempt))
        paths.mkdir()
        target = None
        created = False
        self.step_result.verifier = TimingInfo(started_at=self._now())
        try:
            # Pause briefly for an atomic filesystem capture, retaining every agent
            # process/kernel in its original container. The verifier has no agent process.
            await command("docker", "commit", "--pause=true", self.agent.container, image)
            created = True
            env_config = self.task.config.environment.model_copy(update={"docker_image": image})
            plan = self._network_plan(self.step)
            target = EnvironmentFactory.create_environment_from_config(
                config=self.config.environment,
                environment_dir=self.task.paths.environment_dir,
                environment_name=self.task.short_name,
                session_id=f"buffalo-verifier-{uuid.uuid4().hex[:16]}",
                trial_paths=paths,
                task_env_config=env_config,
                logger=self.logger,
                mounts=[
                    {
                        "type": "bind",
                        "source": str(paths.verifier_dir.resolve()),
                        "target": "/logs/verifier",
                    }
                ],
                network_policy=plan.verifier_phase,
                phase_network_policies=[plan.verifier_phase],
            )
            await asyncio.wait_for(
                target.start(force_build=False), self._environment_build_timeout_sec
            )
            try:
                # Original cumulative tests are uploaded ONLY to this disposable environment.
                async with asyncio.timeout(timeout_seconds):
                    with target.with_default_user(self._step_verifier_user(self.step)):
                        result = await Verifier(
                            task=self.task,
                            trial_paths=paths,
                            environment=target,
                            step_name=self.step.name,
                            verifier_env=self.step.verifier.env,
                            override_env=self.config.verifier.env,
                            logger=self.logger,
                        ).verify()
            except TimeoutError:
                safe = {
                    "passed": False,
                    "exit_text": "timed out",
                    "output": "Cumulative verifier exceeded its configured execution timeout.",
                }
                self.attempts.append(
                    {
                        "attempt": attempt,
                        "rewards": None,
                        "case_summary": None,
                        "logs": str(paths.verifier_dir),
                        "diagnostic": safe,
                        "timed_out": True,
                    }
                )
                self.step_result.verifier_result = None
                save(paths.trial_dir / "attempt.json", self.attempts[-1])
                return safe
            raw = (
                paths.test_stdout_path.read_text(errors="replace")
                if paths.test_stdout_path.exists()
                else ""
            )
            safe, cases = diagnostic(result.rewards, raw)
            self.attempts.append(
                {
                    "attempt": attempt,
                    "rewards": result.rewards,
                    "case_summary": cases,
                    "logs": str(paths.verifier_dir),
                    "diagnostic": safe,
                }
            )
            self.step_result.verifier_result = result
            save(paths.trial_dir / "attempt.json", self.attempts[-1])
            return safe
        finally:
            self.step_result.verifier.finished_at = self._now()
            try:
                if target:
                    await asyncio.shield(target.stop(delete=True))
            finally:
                if created:
                    await asyncio.shield(command("docker", "image", "rm", "--force", image))

    async def _stop_agent_environment(self):
        try:
            if not self._is_agent_environment_stopped:
                await self.agent.close()
        except Exception as exc:
            self._record_exception(exc)
        finally:
            await super()._stop_agent_environment()

    async def _finalize(self):
        try:
            await super()._finalize()
        finally:
            intact = task_hashes(self.task.paths.task_dir) == self.source_hashes
            self.manifest["official_inputs_unchanged"] = intact
            self.write_manifest()
            result = task_report(
                self.task.name,
                len(self.task.config.steps),
                self.rounds,
                self.calls,
                time.monotonic() - self.started,
                self.result.verifier_result.model_dump() if self.result.verifier_result else None,
                official_steps=[
                    {
                        "round_name": s.step_name,
                        "rewards": s.verifier_result.rewards if s.verifier_result else None,
                    }
                    for s in (self.result.step_results or [])
                ],
            )
            result["exception"] = (
                self.result.exception_info.model_dump(mode="json")
                if self.result.exception_info
                else None
            )
            result["step_exceptions"] = [
                s.exception_info.model_dump(mode="json")
                for s in (self.result.step_results or [])
                if s.exception_info
            ]
            save(self.output / "report.json", result)
            if not intact:
                raise RuntimeError("Official task inputs changed during evaluation")


async def run_task(task_path, output, config, *, providers=None, max_steps=None):
    task = Task(task_path)
    trial_config = TrialConfig(
        task={"path": task_path},
        trials_dir=output,
        trial_name=f"{task_path.name}__{uuid.uuid4().hex[:8]}",
        agent={
            "import_path": "threadweave.evals.evocode:BuffaloEvoCodeAgent",
            "resume_trajectory": True,
            "override_setup_timeout_sec": 900,
        },
    )
    trial = BuffaloEvoCodeTrial(
        trial_config,
        run_config=config,
        _task=task,
        _task_download_result=TaskDownloadResult(path=task_path, download_time_sec=0, cached=True),
        providers=providers,
        max_steps=max_steps,
    )
    result = await trial.run()
    return result, trial.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", type=Path)
    selection.add_argument(
        "--dataset", type=Path, help="Run every official task in this directory, in isolation"
    )
    parser.add_argument("--config", type=Path, required=True, help="Buffalo RunConfig JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-steps", type=int, help="Smoke prefix only; never a leaderboard result"
    )
    args = parser.parse_args()
    config = RunConfig.model_validate_json(args.config.read_text())
    tasks = [args.task] if args.task else sorted(p.parent for p in args.dataset.glob("*/task.toml"))
    if not tasks:
        parser.error("No official task.toml files found")

    async def run_all():
        reports, failed = [], False
        for task in tasks:
            result, output = await run_task(
                task.resolve(), args.output.resolve(), config, max_steps=args.max_steps
            )
            print(output / "report.json", flush=True)
            reports.append(json.loads((output / "report.json").read_text()))
            failed |= bool(
                result.exception_info or any(s.exception_info for s in result.step_results or [])
            )
        save(
            args.output / "dataset-report.json",
            {
                "protocol": "EvoCode-Bench + Prime-style autonomous feedback",
                "leaderboard_comparable": False,
                "tasks": len(reports),
                "task_scores": {r["task"]: r["evocode_task_score"] for r in reports},
                "dataset_score": sum(r["evocode_task_score"] for r in reports) / len(reports),
                "scope": "smoke prefix" if args.max_steps else "complete task chains",
            },
        )
        return failed

    if asyncio.run(run_all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
