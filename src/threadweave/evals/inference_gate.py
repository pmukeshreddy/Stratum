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

    def journal(self, **event):
        with (self.directory / "inference-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": timestamp(), **event}) + "\n")

    @asynccontextmanager
    async def permit(self, owner):
        async with self.condition:
            await self.condition.wait_for(lambda: self.active < self.capacity)
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
            if unstable and self.capacity > 1:
                previous = self.capacity
                self.capacity -= 1
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
            "started": time.monotonic(),
            "closed": False,
            "errors": [],
            "inflight": 0,
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
        app = web.Application(client_max_size=128 * 1024 * 1024)
        app.router.add_route("*", "/{owner}/{path:.*}", self.forward)
        self.server = web.AppRunner(app, access_log=None)
        await self.server.setup()
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def close(self):
        for game in self.games.values():
            game["closed"] = True
        if hasattr(self, "server"):
            await self.server.cleanup()
            await self.client.close()
        save(self.directory / "concurrency.json", self.summary())

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
            or payload.get("reasoning", {}).get("effort") != "xhigh"
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
        sent = False
        async with self.permit(owner) as ticket:
            totals = self.usage(owner)
            if (
                game["closed"]
                or time.monotonic() - game["started"] >= config.limits.wall_seconds
                or totals["model_calls"] + game["inflight"] >= config.limits.max_model_calls
                or totals["total_tokens"]
                + (game["inflight"] + 1) * (bound + config.provider.max_output_tokens)
                > config.limits.token_budget
            ):
                game["closed"] = True
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
            try:
                async with self.client.request(
                    request.method, upstream, data=body, headers=headers
                ) as response:
                    status = response.status
                    if status == 429 or status >= 500:
                        await self.outcome(unstable=True)
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
                        partial += chunk
                        while b"\n" in partial:
                            line, partial = partial.split(b"\n", 1)
                            if line.startswith(b"data: "):
                                with contextlib.suppress(ValueError, TypeError):
                                    event = json.loads(line[6:])
                                    data = event.get("response", {})
                                    response_id = data.get("id") or response_id
                                    if data.get("usage"):
                                        usage = data["usage"]
                        await outgoing.write(chunk)
                    await outgoing.write_eof()
                    await self.outcome()
                    return outgoing
            except (aiohttp.ClientError, TimeoutError) as exc:
                game["errors"].append(type(exc).__name__)
                await self.outcome(unstable=True)
                if not sent:
                    return web.json_response(
                        {"error": {"message": "Upstream transport interrupted"}}, status=502
                    )
                raise
            finally:
                game["inflight"] -= 1
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
                    "api_cost": None,
                    "wall_seconds": time.monotonic() - begin,
                }
                game["usages"].append(measured)
                self.journal(
                    event="codex_request_finished",
                    ticket=ticket,
                    owner=owner,
                    status=status,
                    response_id=response_id,
                    usage=measured,
                )
                save(game["directory"] / "transport-usage.json", self.usage(owner))
