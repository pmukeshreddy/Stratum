"""Shared inference admission, including native Codex requests and Buffalo descendants.

The loopback proxy forwards native Codex request/response bytes. It never constructs
prompts, chooses actions, executes tools, or substitutes an agent/model implementation.
Credentials remain in memory and are never written to the request journal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from aiohttp import web

from ..models import HarnessError
from ..tokenization import estimate
from .schema import accounting, save, timestamp


class InferenceGate:
    def __init__(self, directory, capacity=16):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.initial_capacity = capacity
        self.active = self.peak = 0
        self.condition = asyncio.Condition()
        self.recent = deque(maxlen=20)
        self.adjustments = []
        self.games = {}
        self.sequence = 0
        self.last_reduction = float("-inf")
        self.stopped = set()

    async def stop_admission(self, owner):
        """Block new calls after a terminal observation without interrupting a response."""
        async with self.condition:
            self.stopped.add(owner)
            if owner in self.games:
                self.games[owner]["closed"] = True
                self.games[owner]["stop_reason"] = "environment_terminal"
            self.condition.notify_all()

    def journal(self, **event):
        with (self.directory / "inference-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": timestamp(), **event}) + "\n")

    @asynccontextmanager
    async def permit(self, owner):
        async with self.condition:
            await self.condition.wait_for(
                lambda: owner in self.stopped or self.active < self.capacity
            )
            if owner in self.stopped:
                raise HarnessError("environment", "environment_terminal", "Game already ended")
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.sequence += 1
            ticket = self.sequence
            self.journal(
                event="acquire",
                ticket=ticket,
                owner=owner,
                active=self.active,
                capacity=self.capacity,
            )
        try:
            yield ticket
        finally:
            async with self.condition:
                self.active -= 1
                self.journal(
                    event="release",
                    ticket=ticket,
                    owner=owner,
                    active=self.active,
                    capacity=self.capacity,
                )
                self.condition.notify_all()

    async def outcome(self, unstable=False):
        async with self.condition:
            self.recent.append(unstable)
            # A 429 reduces immediately; transport bursts use this same explicit event.
            if unstable and self.capacity > 1 and time.monotonic() - self.last_reduction >= 2:
                previous = self.capacity
                self.capacity -= 1
                self.last_reduction = time.monotonic()
                change = {
                    "from": previous,
                    "to": self.capacity,
                    "reason": "provider throttling or transport failure",
                    "time": timestamp(),
                }
                self.adjustments.append(change)
                self.journal(event="capacity_reduced", **change)
                self.condition.notify_all()

    def register(self, owner, config, directory):
        self.games[owner] = {
            "config": config,
            "directory": Path(directory),
            "usages": [],
            "primary_usages": [],
            "started": time.monotonic(),
            "closed": False,
            "errors": [],
            "inflight": 0,
            "reserved_tokens": 0,
            "reserved_output_tokens": 0,
            "tasks": set(),
            "tool_calls": 0,
            "stop_reason": None,
        }
        return f"{self.url}/{owner}"

    def usage(self, owner):
        game = self.games[owner]
        return accounting(game["usages"], time.monotonic() - game["started"])

    async def start(self):
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=180),
            auto_decompress=False,
        )

        @web.middleware
        async def terminal_response(request, handler):
            try:
                return await handler(request)
            except HarnessError as exc:
                if exc.failure.code != "environment_terminal":
                    raise
                return web.json_response({"error": {"message": "Game already ended"}}, status=400)

        app = web.Application(client_max_size=128 * 1024 * 1024, middlewares=[terminal_response])
        app.router.add_route("*", "/{owner}/{path:.*}", self.forward)
        self.server = web.AppRunner(app, access_log=None)
        await self.server.setup()
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def close(self):
        for owner in self.games:
            await self.close_owner(owner)
        if hasattr(self, "server"):
            await self.server.cleanup()
            await self.client.close()
        save(self.directory / "concurrency.json", self.summary())

    async def close_owner(self, owner):
        game = self.games[owner]
        game["closed"] = True
        tasks = list(game["tasks"])
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def summary(self):
        return {
            "initial_limit": self.initial_capacity,
            "final_limit": self.capacity,
            "peak_inflight": self.peak,
            "active": self.active,
            "adjustments": self.adjustments,
            "scope": "native Codex requests plus all Buffalo root/descendant/compaction calls",
        }

    async def forward(self, request):
        owner, path = request.match_info["owner"], request.match_info["path"]
        game = self.games.get(owner)
        if game is None or path not in {"responses", "responses/compact", "models"}:
            raise web.HTTPNotFound()
        body = await request.read()
        inference = request.method == "POST"
        config = game["config"]
        payload = json.loads(body) if inference else {}
        if inference and (
            payload.get("model") != config.provider.model
            or (path == "responses" and payload.get("reasoning", {}).get("effort") != "xhigh")
        ):
            game["errors"].append("Codex requested a different model or reasoning level")
            return web.json_response(
                {
                    "error": {
                        "message": "Matched evaluation requires gpt-6-astra/xhigh",
                        "type": "invalid_request_error",
                    }
                },
                status=400,
            )
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        headers["Accept-Encoding"] = "identity"
        upstream = "https://chatgpt.com/backend-api/codex/" + path
        if request.query_string:
            upstream += "?" + request.query_string
        if not inference:
            async with self.client.request(request.method, upstream, headers=headers) as response:
                return web.Response(
                    body=await response.read(),
                    status=response.status,
                    content_type=response.content_type,
                )
        bound = estimate(payload, config.provider.model)
        begin = time.monotonic()
        usage = None
        partial = b""
        response_id = None
        status = None
        completed = False
        sent = False
        outgoing = None
        tool_calls = 0
        json_parts = []
        async with self.permit(owner) as ticket:
            # Match Runtime's BudgetBusy semantics: another descendant's reservation
            # can be released without ending this logical game.
            while not game["closed"] and game["inflight"]:
                totals = self.usage(owner)
                output = config.provider.max_output_tokens
                output_busy = config.limits.output_token_budget is not None and (
                    totals["output_tokens"] + output
                    <= config.limits.output_token_budget
                    < totals["output_tokens"] + game["reserved_output_tokens"] + output
                )
                total_busy = (
                    totals["total_tokens"] + bound + output
                    <= config.limits.token_budget
                    < totals["total_tokens"] + game["reserved_tokens"] + bound + output
                )
                if not (output_busy or total_busy):
                    break
                await asyncio.sleep(0.01)
            totals = self.usage(owner)
            if (
                game["closed"]
                or time.monotonic() - game["started"] >= config.limits.wall_seconds
                or totals["model_calls"] + game["inflight"] >= config.limits.max_model_calls
                or totals["total_tokens"]
                + game["reserved_tokens"]
                + bound
                + config.provider.max_output_tokens
                > config.limits.token_budget
                or (
                    config.limits.output_token_budget is not None
                    and totals["output_tokens"]
                    + game["reserved_output_tokens"]
                    + config.provider.max_output_tokens
                    > config.limits.output_token_budget
                )
                or game["tool_calls"] >= config.limits.max_tool_calls
            ):
                game["closed"] = True
                game["stop_reason"] = "resource_budget"
                return web.json_response(
                    {
                        "error": {
                            "message": "Evaluation game budget exhausted",
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
            game["inflight"] += 1
            game["reserved_tokens"] += bound + config.provider.max_output_tokens
            game["reserved_output_tokens"] += config.provider.max_output_tokens
            current = asyncio.current_task()
            game["tasks"].add(current)
            try:
                async with self.client.request(
                    request.method, upstream, data=body, headers=headers
                ) as response:
                    status = response.status
                    if status == 429 or status >= 500:
                        await self.outcome(unstable=True)
                        game["errors"].append(f"http_{status}")
                    outgoing = web.StreamResponse(
                        status=status,
                        headers={
                            k: v
                            for k, v in response.headers.items()
                            if k.lower()
                            not in {
                                "transfer-encoding",
                                "content-length",
                                "connection",
                                "content-encoding",
                            }
                        },
                    )
                    await outgoing.prepare(request)
                    sent = True
                    async for chunk in response.content.iter_any():
                        if "application/json" in response.headers.get("Content-Type", ""):
                            json_parts.append(chunk)
                        partial += chunk
                        while b"\n" in partial:
                            line, partial = partial.split(b"\n", 1)
                            if line.startswith(b"data: "):
                                with contextlib.suppress(ValueError, TypeError):
                                    event = json.loads(line[6:])
                                    completed |= event.get("type") == "response.completed"
                                    data = event.get("response", {})
                                    response_id = data.get("id") or response_id
                                    if data.get("usage"):
                                        usage = data["usage"]
                                    if event.get(
                                        "type"
                                    ) == "response.output_item.done" and event.get("item", {}).get(
                                        "type"
                                    ) in {"function_call", "custom_tool_call"}:
                                        game["tool_calls"] += 1
                                        tool_calls += 1
                        await outgoing.write(chunk)
                    if json_parts:
                        with contextlib.suppress(ValueError, TypeError):
                            data = json.loads(b"".join(json_parts))
                            usage = data.get("usage") or usage
                            response_id = data.get("id") or response_id
                    await outgoing.write_eof()
                    await self.outcome()
                    return outgoing
            except (aiohttp.ClientError, TimeoutError) as exc:
                if isinstance(exc, (ConnectionResetError, aiohttp.ClientConnectionResetError)) and (
                    usage or game["closed"]
                ):
                    # Codex may close SSE after response.completed; this is not throttling.
                    return outgoing if outgoing is not None else web.Response(status=499)
                game["errors"].append(type(exc).__name__)
                await self.outcome(unstable=True)
                if not sent:
                    return web.json_response(
                        {"error": {"message": "Upstream transport interrupted"}}, status=502
                    )
                raise
            finally:
                game["inflight"] -= 1
                game["reserved_tokens"] -= bound + config.provider.max_output_tokens
                game["reserved_output_tokens"] -= config.provider.max_output_tokens
                game["tasks"].discard(current)
                measured = {
                    "input_tokens": usage["input_tokens"] if usage else bound,
                    "output_tokens": usage["output_tokens"]
                    if usage
                    else config.provider.max_output_tokens,
                    "cached_input_tokens": (usage or {})
                    .get("input_tokens_details", {})
                    .get("cached_tokens", 0),
                    "reasoning_output_tokens": (usage or {})
                    .get("output_tokens_details", {})
                    .get("reasoning_tokens", 0),
                    "estimated_calls": int(usage is None),
                    "model_calls": 1,
                    "tool_calls": tool_calls,
                    "api_cost": None,
                    "wall_seconds": time.monotonic() - begin,
                }
                game["usages"].append(measured)
                if path == "responses" and usage and status == 200 and completed:
                    game["primary_usages"].append(measured)
                self.journal(
                    event="codex_request_finished",
                    ticket=ticket,
                    owner=owner,
                    status=status,
                    path=path,
                    model=payload.get("model"),
                    reasoning=payload.get("reasoning", {}).get("effort", "native_compaction"),
                    response_id=response_id,
                    usage=measured,
                )
                save(game["directory"] / "transport-usage.json", self.usage(owner))
