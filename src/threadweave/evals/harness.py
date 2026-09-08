"""Same provider contract and budgets; direct chat baseline versus production Runtime."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
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


class PythonAction(Record):
    code: str


class MatchedProvider:
    def __init__(self, provider, config, log):
        self.provider, self.config, self.log = provider, config, Path(log)

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


async def scratch(code, workspace, seconds):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
        },
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), seconds)
    except BaseException:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        raise
    return {
        "stdout": out.decode(errors="replace"),
        "stderr": err.decode(errors="replace"),
        "returncode": process.returncode,
    }


async def run_base(config, task, directory, *, action=None, persistent=False, long_context=False):
    """Conventional full-history chat/tool loop, with fresh Python per call.

    The persistent external world is shared across steps; only Buffalo has its REPL,
    context compaction/retrieval, durable session state, and recursive agents.
    """
    directory = Path(directory)
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    messages = json.loads(json.dumps(task["messages"]))
    if long_context:
        (workspace / "task.txt").write_text(messages[-1]["content"])
        if (
            estimate(messages, config.provider.model) + config.provider.max_output_tokens
            > config.context.max_tokens
        ):
            messages = [
                {
                    "role": "user",
                    "content": "The complete official task is in task.txt. Read and reason over it with python; "
                    "return the final answer in the task's requested format.",
                }
            ]
    tools = [
        Tool(
            "python",
            "Run Python in a fresh process in the task workspace. Files persist.",
            PythonAction,
            None,
        ).schema()
    ]
    if action:
        tools.append(
            Tool(
                "benchmark_action",
                "Interact with the persistent official environment.",
                EnvironmentAction,
                None,
            ).schema()
        )
    provider = MatchedProvider(
        default_providers()[config.provider.name],
        config.provider,
        directory / "provider-calls.jsonl",
    )
    usages, response_text, tool_count, python_count = [], "", 0, 0
    started, begin = timestamp(), time.monotonic()
    reason = "max_turns"
    try:
        async with asyncio.timeout(config.limits.wall_seconds):
            for turn in range(min(config.limits.max_turns, config.limits.max_model_calls)):
                totals = accounting(usages, time.monotonic() - begin)
                bound = estimate({"messages": messages, "tools": tools}, config.provider.model)
                if bound + config.provider.max_output_tokens > config.context.max_tokens:
                    reason = "context_capacity"
                    break
                if (
                    totals["total_tokens"] + bound + config.provider.max_output_tokens
                    > config.limits.token_budget
                ):
                    reason = "token_budget"
                    break
                reserve_cost = (
                    bound * (config.provider.input_cost_per_million or 0)
                    + config.provider.max_output_tokens
                    * (config.provider.output_cost_per_million or 0)
                ) / 1e6
                if (
                    config.limits.cost_budget is not None
                    and (totals["api_cost"] or 0) + reserve_cost > config.limits.cost_budget
                ):
                    reason = "cost_budget"
                    break
                for attempt in range(3):
                    attempted = accounting(usages, time.monotonic() - begin)
                    exhausted = (
                        "max_model_calls"
                        if attempted["model_calls"] >= config.limits.max_model_calls
                        else "token_budget"
                        if attempted["total_tokens"] + bound + config.provider.max_output_tokens
                        > config.limits.token_budget
                        else None
                    )
                    if exhausted:
                        return _base_result(
                            directory, usages, begin, started, response_text, exhausted, messages
                        )
                    try:
                        response = await provider.invoke(
                            ModelRequest(
                                session_id="base",
                                root_id="base",
                                parent_id=None,
                                name="base",
                                turn=turn,
                                messages=messages,
                                tools=tools,
                                config=config.provider,
                                input_token_bound=bound,
                            ),
                            discard,
                        )
                        break
                    except BaseException as exc:
                        usages.append(
                            Usage(
                                input_tokens=bound,
                                output_tokens=config.provider.max_output_tokens,
                                model_calls=1,
                                estimated_calls=1,
                                cost=None,
                            )
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
                response_text = response.text
                message = {"role": "assistant", "content": response.text}
                if response.actions:
                    message["tool_calls"] = [
                        {
                            "id": a.id,
                            "type": "function",
                            "function": {"name": a.name, "arguments": json.dumps(a.arguments)},
                        }
                        for a in response.actions
                    ]
                messages.append(message)
                if not response.actions:
                    if persistent:
                        messages.append(
                            {
                                "role": "user",
                                "content": "Continue advancing factory research within the remaining run budget.",
                            }
                        )
                        continue
                    reason = "completed"
                    break
                for call in response.actions:
                    if tool_count >= config.limits.max_tool_calls:
                        reason = "max_tool_calls"
                        return _base_result(
                            directory, usages, begin, started, response_text, reason, messages
                        )
                    tool_count += 1
                    try:
                        if call.name == "benchmark_action" and action:
                            result = await action(
                                **EnvironmentAction.model_validate(call.arguments).model_dump()
                            )
                        elif call.name == "python":
                            if python_count >= config.limits.max_python_executions:
                                reason = "max_python_executions"
                                return _base_result(
                                    directory,
                                    usages,
                                    begin,
                                    started,
                                    response_text,
                                    reason,
                                    messages,
                                )
                            python_count += 1
                            result = await scratch(
                                PythonAction.model_validate(call.arguments).code,
                                workspace,
                                config.limits.python_timeout_seconds,
                            )
                        else:
                            result = {"error": "Unknown tool"}
                    except NotRun:
                        raise
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                    save(directory / f"tool-{tool_count}.json", result)
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
                    )
                save(directory / "messages.json", messages)
    except HarnessError as exc:
        if exc.failure.code != "observed_output_limit":
            raise
        reason = "observed_output_limit"
    except TimeoutError:
        reason = "wall_seconds"
    finally:
        save(directory / "usage.json", accounting(usages, time.monotonic() - begin))
    return _base_result(directory, usages, begin, started, response_text, reason, messages)


def _base_result(directory, usages, begin, started, response, reason, messages):
    save(directory / "messages.json", messages)
    if not usages and reason in {"context_capacity", "token_budget", "cost_budget"}:
        raise NotRun(f"Configured {reason} prevents even one model invocation")
    return {
        "response": response if reason == "completed" else "",
        "last_response": response,
        "stop_reason": reason,
        "start_time": started,
        "end_time": timestamp(),
        "usage": accounting(usages, time.monotonic() - begin),
        "trajectory_reference": str(directory.resolve()),
    }


async def run_buffalo(
    config, task, directory, *, action=None, persistent=False, long_context=False
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
                [s.model_dump(mode="json") for s in runtime.store.sessions(root_id=session.root_id)],
            )
        await runtime.shutdown()
