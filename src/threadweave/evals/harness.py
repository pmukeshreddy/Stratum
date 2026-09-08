"""Matched model requests and the production Buffalo Runtime; no custom BASE agent."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path

from pydantic import Field

from ..models import HarnessError, ModelRequest, Record, TaskConfig, Usage
from ..providers import default_providers
from ..runtime import Runtime
from ..tokenization import estimate
from ..tools import Tool
from .schema import NotRun, accounting, digest, save, timestamp


class EnvironmentAction(Record):
    action: str | None = None
    data: dict = Field(default_factory=dict)
    code: str | None = None


class MatchedProvider:
    def __init__(self, provider, config, log, gate=None, owner=None):
        self.provider, self.config, self.log = provider, config, Path(log)
        self.gate, self.owner = gate, owner

    async def resolve(self, config):
        resolved, details = await self.provider.resolve(config)
        if resolved.model_dump() != self.config.model_dump():
            raise HarnessError(
                "provider",
                "evaluation_settings_changed",
                "Subscription model/settings changed after comparison was pinned",
            )
        return resolved, details

    async def invoke(self, request, emit):
        if self.gate is None:
            return await self._invoke(request, emit)
        async with self.gate.permit(self.owner):
            try:
                result = await self._invoke(request, emit)
            except HarnessError as exc:
                if exc.failure.retryable and exc.failure.code != "observed_output_limit":
                    await self.gate.outcome(unstable=True)
                raise
            await self.gate.outcome()
            return result

    async def _invoke(self, request, emit):
        if request.config.model_dump() != self.config.model_dump():
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
                        "request": request.model_dump(mode="json"),
                        "response": response.model_dump(mode="json"),
                    }
                )
                + "\n"
            )
        return response


async def discard(_):
    pass


async def invoke_judge(config, request, directory, usages):
    provider = default_providers()[config.name]
    # The official evaluator selects judge temperature; the configured output cap is shared.
    resolved = config.model_copy(deep=True)
    resolved.max_output_tokens = min(config.max_output_tokens, request["max_tokens"])
    if config.name == "chat":
        resolved.parameters["temperature"] = request["temperature"]
    messages = [{"role": "user", "content": request["prompt"]}]
    measured = MatchedProvider(provider, resolved, Path(directory) / "judge-calls.jsonl")
    for attempt in range(3):
        try:
            response = await measured.invoke(
                ModelRequest(
                    session_id="official-judge",
                    root_id="official-judge",
                    parent_id=None,
                    name="official-judge",
                    turn=len(usages),
                    messages=messages,
                    tools=[],
                    config=resolved,
                    input_token_bound=estimate(messages, resolved.model),
                ),
                discard,
            )
            break
        except BaseException as exc:
            usages.append(
                Usage(
                    input_tokens=estimate(messages, resolved.model),
                    output_tokens=resolved.max_output_tokens,
                    cost=None,
                    estimated_calls=1,
                    model_calls=1,
                )
            )
            with (Path(directory) / "judge-errors.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "time": timestamp(),
                            "type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                        }
                    )
                    + "\n"
                )
            retryable = (isinstance(exc, HarnessError) and exc.failure.retryable) or (
                isinstance(exc, PermissionError) and exc.errno == 1
            )
            if not retryable or attempt == 2:
                raise
            await asyncio.sleep(2**attempt)
    usage = response.usage.model_copy(deep=True)
    usage.model_calls += 1
    usage.estimated_calls += int(not response.usage_reported)
    usages.append(usage)
    return response.text


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
):
    directory = Path(directory)
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    resolved = config.model_copy(deep=True)
    resolved.control_plane = "python"
    instruction = (
        task["messages"][-1]["content"]
        if long_context
        else "Complete the supplied official task and return its requested final answer."
    )
    resolved.task = TaskConfig(
        verify_each_turn=False,
        wait_for_children=True,
        instruction_messages=[] if long_context else task["messages"],
    )
    if long_context:
        (workspace / "task.txt").write_text(instruction)
    if action:
        instruction += (
            '\nInteract using await tools.acall("benchmark_action", action=..., data=...)'
            ' for ARC or await tools.acall("benchmark_action", code=...) for Factorio.'
        )
    # Every invocation, including child/compaction/refinement calls, checks the same provider contract.
    provider = MatchedProvider(
        default_providers()[resolved.provider.name],
        resolved.provider,
        directory / "provider-calls.jsonl",
        gate=gate,
        owner=owner,
    )
    runtime = Runtime(
        directory / "state",
        providers={resolved.provider.name: provider},
        concurrency=resolved.limits.concurrency,
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
    begin, started = time.monotonic(), timestamp()
    try:
        session = runtime.create(instruction, workspace, config=resolved)
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
        if session.outcome == "failed" and session.last_error:
            if session.last_error.code == "observed_output_limit":
                result["stop_reason"] = "observed_output_limit"
            else:
                raise NotRun(f"{session.last_error.code}: {session.last_error.message}")
        if not usage.model_calls:
            raise NotRun(f"Buffalo could not invoke the model: {session.result or session.outcome}")
        return result
    finally:
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
