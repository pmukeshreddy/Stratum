"""Launch the installed Codex app-server, including its real agent loop and tools."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

from .schema import NotRun, save, timestamp


class CodexAgent:
    def __init__(self, directory, action):
        self.directory = Path(directory)
        self.action = action
        self.pending = {}
        self.sequence = 0
        self.notifications = asyncio.Queue()
        self.handlers = set()
        self.thread_ids = set()
        self.native_usage = {}
        self.tool_calls = 0

    async def send(self, payload):
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        await self.process.stdin.drain()

    async def call(self, method, **params):
        self.sequence += 1
        future = asyncio.get_running_loop().create_future()
        self.pending[self.sequence] = future
        await self.send({"id": self.sequence, "method": method, "params": params})
        return await asyncio.wait_for(future, 60)

    async def read(self):
        try:
            with (self.directory / "codex-events.jsonl").open("a") as journal:
                while line := await self.process.stdout.readline():
                    event = json.loads(line)
                    journal.write(json.dumps(event) + "\n")
                    journal.flush()
                    if "method" not in event and "id" in event:
                        future = self.pending.pop(event["id"], None)
                        if future and not future.done():
                            if "error" in event:
                                future.set_exception(NotRun(f"Codex app-server: {event['error']}"))
                            else:
                                future.set_result(event.get("result"))
                        continue
                    if "id" in event:
                        handler = asyncio.create_task(self.handle(event))
                        self.handlers.add(handler)
                        handler.add_done_callback(self.handlers.discard)
                    else:
                        params = event.get("params", {})
                        if event.get("method") == "thread/started":
                            self.thread_ids.add(params["thread"]["id"])
                        if event.get("method") == "thread/tokenUsage/updated":
                            self.native_usage[params["threadId"]] = params["tokenUsage"]
                            save(self.directory / "codex-native-usage.json", self.native_usage)
                        if event.get("method") == "item/started" and params.get("item", {}).get(
                            "type"
                        ) in {
                            "commandExecution",
                            "fileChange",
                            "mcpToolCall",
                            "dynamicToolCall",
                            "collabAgentToolCall",
                            "webSearch",
                            "imageView",
                            "imageGeneration",
                        }:
                            self.tool_calls += 1
                        await self.notifications.put(event)
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(NotRun("Native Codex app-server exited"))
            await self.notifications.put({"method": "server/exited"})

    async def handle(self, event):
        if event["method"] == "item/tool/call":
            params = event["params"]
            try:
                if params["tool"] != "benchmark_action":
                    raise ValueError("Unknown dynamic environment tool")
                arguments = params["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                result = await self.action(**arguments)
                reply = {
                    "contentItems": [{"type": "inputText", "text": json.dumps(result)}],
                    "success": True,
                }
            except Exception as exc:
                reply = {
                    "contentItems": [{"type": "inputText", "text": str(exc)}],
                    "success": False,
                }
            await self.send({"id": event["id"], "result": reply})
        else:
            # Never impersonate a user accepting an unexpected approval or elicitation.
            await self.send(
                {
                    "id": event["id"],
                    "error": {
                        "code": -32601,
                        "message": "No interactive user is attached to this evaluation",
                    },
                }
            )

    async def start(self, config, url):
        executable = shutil.which("codex")
        if executable is None:
            raise NotRun("The actual Codex executable is not installed")
        workspace = (self.directory / "workspace").resolve()
        workspace.mkdir(parents=True)
        options = {
            "model": config.provider.model,
            "model_reasoning_effort": "xhigh",
            "model_provider": "arc_subscription_transport",
            "model_providers.arc_subscription_transport": {
                "name": "OpenAI ChatGPT subscription through local concurrency gate",
                "base_url": url,
                "wire_api": "responses",
                "requires_openai_auth": True,
                "supports_websockets": False,
                "request_max_retries": 2,
                "stream_max_retries": 2,
                "stream_idle_timeout_ms": 180000,
            },
            "approval_policy": "never",
            "sandbox_mode": "workspace-write",
            "model_context_window": config.context.max_tokens,
            "agents.default_subagent_model": config.provider.model,
            "agents.default_subagent_reasoning_effort": "xhigh",
            "agents.max_concurrent_threads_per_session": config.limits.max_subagents,
            "features.enable_request_compression": False,
            "features.memories": False,
        }

        # TOML accepts JSON strings, scalars, and arrays; inline tables need TOML syntax.
        def toml(value):
            if isinstance(value, dict):
                return "{" + ", ".join(f"{k} = {toml(v)}" for k, v in value.items()) + "}"
            return json.dumps(value)

        command = [executable, "app-server"]
        for key, value in options.items():
            command += ["-c", f"{key}={toml(value)}"]
        save(
            self.directory / "codex-command.json",
            {
                "argv": command,
                "cwd": str(workspace),
                "baseline": "actual installed Codex app-server; normal prompt, agent loop, native tools and delegation",
            },
        )
        self.stderr = (self.directory / "codex-stderr.log").open("wb")
        environment = dict(os.environ)
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_ACCESS_TOKEN"):
            environment.pop(key, None)
        self.process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=True,
            limit=32 * 1024 * 1024,
        )
        self.reader = asyncio.create_task(self.read())
        await self.call(
            "initialize",
            clientInfo={"name": "buffalo-arc-evaluation", "version": "1"},
            capabilities={"experimentalApi": True},
        )
        await self.send({"method": "initialized", "params": {}})
        thread = await self.call(
            "thread/start",
            model=config.provider.model,
            modelProvider="arc_subscription_transport",
            cwd=str(workspace),
            approvalPolicy="never",
            sandbox="workspace-write",
            ephemeral=False,
            allowProviderModelFallback=False,
            dynamicTools=[
                {
                    "type": "function",
                    "name": "benchmark_action",
                    "description": "Interact with this worker's isolated official ARC-AGI-3 environment.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"action": {"type": "string"}, "data": {"type": "object"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }
            ],
        )
        self.thread_id = thread["thread"]["id"]
        self.thread_ids.add(self.thread_id)
        save(self.directory / "codex-thread.json", thread)

    async def close(self):
        for handler in list(self.handlers):
            handler.cancel()
        if self.handlers:
            await asyncio.gather(*self.handlers, return_exceptions=True)
        if hasattr(self, "process") and self.process.returncode is None:
            # The app-server owns its child agents and native tool subprocesses.
            import psutil

            with contextlib.suppress(psutil.NoSuchProcess):
                descendants = psutil.Process(self.process.pid).children(recursive=True)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                for child in descendants:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if hasattr(self, "reader"):
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if hasattr(self, "stderr"):
            self.stderr.close()


async def run_codex(config, task, directory, *, action, gate, owner):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    agent = CodexAgent(directory, action)
    url = gate.register(owner, config, directory)
    started, begin = timestamp(), time.monotonic()
    reason, answer = "wall_budget", ""
    try:
        await agent.start(config, url)
        begin, started = time.monotonic(), timestamp()
        gate.games[owner]["started"] = begin
        turn = await agent.call(
            "turn/start",
            threadId=agent.thread_id,
            model=config.provider.model,
            effort="xhigh",
            input=[{"type": "text", "text": task["messages"][-1]["content"]}],
        )
        turn_id = turn["turn"]["id"]
        while (
            time.monotonic() - begin < config.limits.wall_seconds
            and not gate.games[owner]["closed"]
        ):
            try:
                event = await asyncio.wait_for(
                    agent.notifications.get(),
                    min(1, config.limits.wall_seconds - (time.monotonic() - begin)),
                )
            except TimeoutError:
                continue
            params = event.get("params", {})
            if agent.tool_calls >= config.limits.max_tool_calls:
                reason = "tool_budget"
                break
            if event.get("method") == "server/exited":
                raise NotRun("Actual Codex app-server exited during the game")
            if (
                event.get("method") == "turn/completed"
                and params.get("turn", {}).get("id") == turn_id
            ):
                status = params["turn"]["status"]
                if status == "failed" and not gate.games[owner]["closed"]:
                    raise NotRun(f"Actual Codex turn failed: {params['turn'].get('error')}")
                reason = status
                break
        gate.games[owner]["closed"] = True
        with contextlib.suppress(Exception):
            await agent.call("turn/interrupt", threadId=agent.thread_id, turnId=turn_id)
    finally:
        await agent.close()
        await gate.close_owner(owner)
        usage = gate.usage(owner)
        usage["tool_calls"] = max(agent.tool_calls, gate.games[owner]["tool_calls"])
        usage["subagent_count"] = max(0, len(agent.thread_ids) - 1)
        usage["wall_seconds"] = time.monotonic() - begin
        save(directory / "usage.json", usage)
    if not usage["model_calls"]:
        raise NotRun("Actual Codex made no model requests")
    if any("different model" in error for error in gate.games[owner]["errors"]):
        raise NotRun("Codex requested a different model or reasoning level; comparison invalid")
    return {
        "response": answer,
        "stop_reason": reason,
        "start_time": started,
        "end_time": timestamp(),
        "usage": usage,
        "baseline": "actual Codex app-server",
        "thread_ids": sorted(agent.thread_ids),
        "transport_errors": gate.games[owner]["errors"],
        "trajectory_reference": str(directory.resolve()),
    }
