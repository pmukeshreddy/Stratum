"""Matched model requests and the production Buffalo Runtime."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pydantic import Field

from ..auxiliary import refinement_provider_config
from ..models import (
    HarnessError,
    Record,
    TaskConfig,
)
from ..providers import default_providers
from ..refinement_model import refinement_output_limit
from ..runtime import Runtime
from ..tools import Tool
from .schema import NotRun, accounting, digest, save, timestamp


class EnvironmentAction(Record):
    action: str | None = None
    data: dict = Field(default_factory=dict)


class MatchedProvider:
    def __init__(self, provider, config, log, gate=None, owner=None, request_validator=None):
        self.provider, self.config, self.log = provider, config, Path(log)
        self.gate, self.owner = gate, owner
        self.primary_usages = []
        # Mirror the production auxiliary reservations without changing its policy.
        self.refinement_expected = {
            purpose: refinement_provider_config(config).model_copy(
                update={
                    "max_output_tokens": refinement_output_limit(
                        config, review=purpose == "refinement_review"
                    ),
                    "streaming": True,
                }
            )
            for purpose in ("refinement_review", "refinement")
        }
        self.refinement_configs = dict(self.refinement_expected)
        self.request_validator = request_validator

    async def resolve(self, config, *, reasoning_off=False):
        expected = (
            next((c for c in self.refinement_expected.values() if c == config), None)
            if reasoning_off
            else self.config
        )
        if config != expected:
            raise HarnessError(
                "provider",
                "evaluation_settings_changed",
                "Model/settings changed after comparison was pinned",
            )
        if reasoning_off:
            resolved, details = await self.provider.resolve(config, reasoning_off=True)
            compared = resolved
        else:
            resolved, details = await self.provider.resolve(config)
            compared = resolved
        if compared != expected:
            raise HarnessError(
                "provider",
                "evaluation_settings_changed",
                "Subscription model/settings changed after comparison was pinned",
            )
        if reasoning_off:
            for purpose, candidate in self.refinement_expected.items():
                if candidate == expected:
                    self.refinement_configs[purpose] = resolved
        return resolved, details

    async def invoke(self, request, emit):
        if self.gate is None:
            return await self._invoke(request, emit)
        async with self.gate.permit(self.owner):
            try:
                result = await self._invoke(request, emit)
            except HarnessError as exc:
                if exc.failure.retryable:
                    await self.gate.outcome(unstable=True)
                raise
            await self.gate.outcome()
            return result

    async def _invoke(self, request, emit):
        if self.request_validator:
            self.request_validator(request)
        structured = (
            request.request_kind == "auxiliary"
            and request.reasoning_mode == "off"
            and request.metadata.get("purpose") in {"refinement", "refinement_review"}
        )
        expected = (
            self.refinement_configs[request.metadata["purpose"]] if structured else self.config
        )
        if request.config != expected:
            raise HarnessError(
                "provider",
                "evaluation_settings_changed",
                "Model/provider/settings changed inside evaluation; comparison invalid",
            )
        started = timestamp()
        try:
            response = await self.provider.invoke(request, emit)
        except PermissionError as exc:
            if exc.errno != 1:
                raise
            raise HarnessError(
                "provider",
                "os_permission_retry",
                f"Operating system interrupted provider operation: {exc}",
                retryable=True,
                uncertain=True,
            ) from exc
        with self.log.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "start_time": started,
                        "end_time": timestamp(),
                        "session_id": request.session_id,
                        "root_id": request.root_id,
                        "parent_id": request.parent_id,
                        "request": request.public_dump(),
                        "response": response.model_dump(mode="json"),
                    }
                )
                + "\n"
            )
        if request.parent_id is None and request.metadata.get("purpose", "agent") == "agent":
            measured = response.usage.model_copy(deep=True)
            measured.model_calls += 1
            measured.estimated_calls += int(not response.usage_reported)
            self.primary_usages.append(measured)
        return response


async def discard(_):
    pass


async def run_buffalo(
    config,
    task,
    directory,
    *,
    action=None,
    persistent=False,
    long_context=False,
    gate=None,
    owner=None,
    controller=None,
    task_config=None,
    workspace=None,
    request_validator=None,
):
    directory = Path(directory)
    workspace = Path(workspace) if workspace is not None else directory / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    resolved = config.model_copy(deep=True)
    resolved.control_plane = "python"
    instruction = (
        task["messages"][-1]["content"]
        if long_context or controller or task_config is not None
        else "Complete the supplied official task and return its requested final answer."
    )
    resolved.task = TaskConfig(
        verify_each_turn=False,
        wait_for_children=True,
        instruction_messages=[] if long_context or controller else task["messages"],
    )
    if task_config is not None:
        resolved.task = task_config.model_copy(deep=True)
        resolved.task.instruction_messages = task["messages"][:-1]
    resolved.task.original_messages = task["messages"]
    if controller:
        resolved.task.adapter = "interactive_evaluation"
        resolved.task.verifier = "terminal"
        resolved.task.require_verifier = True
    if long_context:
        (workspace / "task.txt").write_text(instruction)
    if action:
        instruction += (
            '\nInteract using await tools.acall("benchmark_action", action=..., data=...)'
            " to interact with ARC."
        )
    # Every invocation, including child/compaction/refinement calls, checks the same provider contract.
    provider = MatchedProvider(
        default_providers()[resolved.provider.name],
        resolved.provider,
        directory / "provider-calls.jsonl",
        gate=gate,
        owner=owner,
        request_validator=request_validator,
    )
    runtime = Runtime(
        directory / "state",
        providers={resolved.provider.name: provider},
        concurrency=resolved.limits.concurrency,
        adapters={"interactive_evaluation": controller} if controller else None,
    )
    if action:

        async def execute_action(context, arguments):
            return await action(**arguments.model_dump())

        runtime.tools.register(
            Tool(
                "benchmark_action",
                "Interact with the official persistent environment.",
                EnvironmentAction,
                execute_action,
            )
        )
        if resolved.tool_allowlist is not None:
            resolved.tool_allowlist.append("benchmark_action")
    save(directory / "buffalo-config.json", resolved.model_dump(mode="json"))
    session = None
    terminal_watcher = None
    begin, started = time.monotonic(), timestamp()
    try:
        session = runtime.create(instruction, workspace, config=resolved)
        save(
            directory / "initial-harness-state.json",
            {
                "session_id": session.id,
                "states": runtime.store.harness.merged(session.id),
                "state_directory": str(runtime.store.directory.resolve()),
            },
        )
        if controller:
            controller.identity["session_id"] = session.id

            def identity():
                kernel = runtime.kernels.get(session.id)
                process = getattr(kernel, "process", None)
                return {**controller.identity, "python_pid": getattr(process, "pid", None)}

            controller.identity_reader = identity
            controller.usage_reader = lambda: accounting(
                [runtime.store.usage(session.id, tree=True)], time.monotonic() - begin
            )
            controller.primary_usage_reader = lambda: accounting(
                provider.primary_usages, time.monotonic() - begin
            )

            async def stop_terminal():
                await controller.done.wait()
                await runtime.stop(session.id, tree=True)

            terminal_watcher = asyncio.create_task(stop_terminal())
        await runtime.start()
        while True:
            remaining = config.limits.wall_seconds - (time.monotonic() - begin)
            if remaining <= 0:
                await runtime.stop(session.id, tree=True)
                break
            try:
                session = await runtime.wait(session.id, timeout=remaining)
            except TimeoutError:
                await runtime.stop(session.id, tree=True)
                break
            if persistent and session.outcome == "completed":
                runtime.interact(
                    session.id,
                    "Continue advancing factory research within the remaining run budget.",
                )
                continue
            break
        # Stop and settle the whole tree BEFORE capturing usage; no descendant is orphaned or omitted.
        if runtime.store.session(session.id).outcome == "active":
            await runtime.stop(session.id, tree=True)
        else:
            for child in runtime.store.sessions(root_id=session.root_id):
                if child.id != session.id and child.outcome == "active":
                    await runtime.stop(child.id, tree=True)
        active = list(runtime.tasks.values())
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        session = runtime.store.session(session.id)
        usage = runtime.store.usage(session.id, tree=True)
        result = {
            "response": (session.result or "") if session.outcome == "completed" else "",
            "stop_reason": str(session.outcome),
            "session_id": session.id,
            "start_time": started,
            "end_time": timestamp(),
            "usage": accounting([usage], time.monotonic() - begin),
            "trajectory_reference": str(directory.resolve()),
            "config_sha256": digest(resolved.model_dump(mode="json")),
        }
        save(directory / "usage.json", result["usage"])
        save(
            directory / "sessions.json",
            [s.model_dump(mode="json") for s in runtime.store.sessions(root_id=session.root_id)],
        )
        if controller:
            result.update(await controller.finish())
        if (
            session.outcome == "failed"
            and session.last_error
            and not (controller and controller.done.is_set() and not controller.error)
        ):
            raise NotRun(f"{session.last_error.code}: {session.last_error.message}")
        if not usage.model_calls:
            raise NotRun(f"Buffalo could not invoke the model: {session.result or session.outcome}")
        return result
    finally:
        if terminal_watcher:
            terminal_watcher.cancel()
            await asyncio.gather(terminal_watcher, return_exceptions=True)
        if session:
            pending = list(runtime.tasks.values())
            for member in runtime.store.sessions(root_id=session.root_id):
                if member.outcome == "active":
                    await runtime.stop(member.id, tree=True)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            save(
                directory / "usage.json",
                accounting([runtime.store.usage(session.id, tree=True)], time.monotonic() - begin),
            )
            save(
                directory / "sessions.json",
                [
                    s.model_dump(mode="json")
                    for s in runtime.store.sessions(root_id=session.root_id)
                ],
            )
        await runtime.shutdown()
