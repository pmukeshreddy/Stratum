"""Real MCP SDK transports, owned by daemon tasks, with per-session permissions.

Credential values are resolved only when opening transports. They never enter the
kernel, configuration snapshots or diagnostic logs. Server output is untrusted.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .execution import environment
from .models import HarnessError


class Connection:
    def __init__(self, context, name, config):
        self.context, self.name, self.config = context, name, config
        self.ready = asyncio.get_running_loop().create_future()
        self.stop = asyncio.Event()
        self.client, self.tools = None, {}
        self.secrets = []
        self.task = asyncio.create_task(self._own())

    def resolve(self, mappings):
        values = {}
        for destination, source in mappings.items():
            value = os.environ.get(source)
            if not value:
                raise ValueError("MCP credential environment variable is not set")
            values[destination] = value
            self.secrets.append(value)
        return values

    def clean(self, result):
        if isinstance(result, str):
            for secret in self.secrets:
                result = result.replace(secret, "[REDACTED]")
            return result
        if isinstance(result, list):
            return [self.clean(v) for v in result]
        if isinstance(result, dict):
            return {self.clean(k): self.clean(v) for k, v in result.items()}
        return result

    async def _own(self):
        # anyio transport cancel scopes must open and close on the same task.
        try:
            async with AsyncExitStack() as stack:
                c = self.config
                if c.type == "stdio":
                    execution = self.context.runtime.store.config(self.context.session_id).execution
                    if execution.backend != "local":
                        raise PermissionError(
                            "stdio MCP needs a trusted local server; use HTTP MCP with containers"
                        )
                    if (
                        execution.command_allowlist is not None
                        and c.command not in execution.command_allowlist
                    ):
                        raise PermissionError("MCP command is not allowed")
                    stderr = stack.enter_context(open(os.devnull, "w"))  # noqa: ASYNC230 - device open
                    env = {**environment(execution), **self.resolve(c.env_from)}
                    streams = await stack.enter_async_context(
                        stdio_client(
                            StdioServerParameters(
                                command=c.command,
                                args=c.args,
                                cwd=str(self.context.path(c.cwd)),
                                env=env,
                            ),
                            errlog=stderr,
                        )
                    )
                else:
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=self.resolve(c.headers_from), timeout=c.call_timeout_seconds
                        )
                    )
                    streams = await stack.enter_async_context(
                        streamable_http_client(c.url, http_client=client)
                    )
                self.client = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await self.client.initialize()
                cursor = None
                while True:
                    result = await self.client.list_tools(cursor=cursor)
                    for tool in result.tools:
                        if self.allowed(tool.name):
                            self.tools[tool.name] = self.clean(
                                tool.model_dump(mode="json", exclude_none=True)
                            )
                    cursor = result.nextCursor
                    if not cursor:
                        break
                if not self.ready.done():
                    self.ready.set_result(True)
                await self.stop.wait()
        except BaseException:
            if not self.ready.done():
                self.ready.set_exception(
                    HarnessError(
                        "environment",
                        "mcp_startup",
                        f"MCP server {self.name} could not initialize; check transport, credentials and server installation",
                    )
                )
        finally:
            self.client = None

    def allowed(self, tool):
        return (
            self.config.enabled_tools is None or tool in self.config.enabled_tools
        ) and tool not in self.config.disabled_tools

    async def call(self, tool, arguments):
        if not self.allowed(tool) or tool not in self.tools:
            raise PermissionError(f"MCP tool not discovered/enabled: {tool}")
        if not self.client:
            raise HarnessError(
                "environment",
                "mcp_disconnected",
                "MCP transport disconnected; reload before retrying",
                uncertain=True,
            )
        try:
            async with asyncio.timeout(self.config.call_timeout_seconds):
                result = await self.client.call_tool(tool, arguments)
            payload = self.clean(result.model_dump(mode="json", exclude_none=True))
            # Preserve structuredContent, all content blocks, and MCP isError.
            # A tool's error payload is evidence, not a transport exception.
            return payload
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise HarnessError(
                "environment",
                "mcp_call",
                "MCP invocation failed or timed out; external effects may be uncertain",
                uncertain=True,
            ) from exc

    async def close(self):
        self.stop.set()
        if not self.ready.done():
            self.task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), 5)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


class McpManager:
    def __init__(self, runtime):
        self.runtime, self.connections, self.locks = runtime, {}, {}

    async def connection(self, context, name):
        config = self.runtime.store.config(context.session_id).mcp_servers.get(name)
        if config is None or not config.enabled:
            raise PermissionError(f"MCP server is not configured/enabled: {name}")
        key = (context.session_id, name)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            connection = self.connections.get(key)
            if connection is None or connection.task.done():
                connection = self.connections[key] = Connection(context, name, config)
            try:
                await asyncio.wait_for(
                    asyncio.shield(connection.ready), config.startup_timeout_seconds
                )
            except BaseException:
                await connection.close()
                self.connections.pop(key, None)
                raise
            return connection

    async def call(self, context, operation, p):
        if operation == "servers":
            return [
                {"name": n, "type": c.type, "enabled": c.enabled}
                for n, c in self.runtime.store.config(context.session_id).mcp_servers.items()
            ]
        if operation in {"reload", "close"}:
            for key in list(self.connections):
                if key[0] == context.session_id and (not p.get("server") or key[1] == p["server"]):
                    await self.connections.pop(key).close()
            return {"closed": True, "reopens_lazily": operation == "reload"}
        connection = await self.connection(context, p["server"])
        if operation == "tools":
            return list(connection.tools.values())
        if operation == "call":
            return await connection.call(p["tool"], p.get("arguments", {}))
        raise ValueError("Unknown MCP operation")

    async def close(self):
        await asyncio.gather(
            *(c.close() for c in self.connections.values()), return_exceptions=True
        )
        self.connections.clear()

    async def close_session(self, sid):
        for key in list(self.connections):
            if key[0] == sid:
                await self.connections.pop(key).close()
