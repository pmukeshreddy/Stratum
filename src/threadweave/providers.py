from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

import httpx

from .models import Action, HarnessError, ModelRequest, ModelResponse, Usage


class Provider(Protocol):
    async def invoke(
        self, request: ModelRequest, emit: Callable[[str], Awaitable[None]]
    ) -> ModelResponse:
        """Cancellation propagates via CancelledError; emit carries optional text deltas."""
        ...


class ChatProvider:
    """Chat-completions wire protocol, usable with hosted or compatible local endpoints."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    async def invoke(self, request: ModelRequest, emit) -> ModelResponse:
        config = request.config
        key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if config.api_key_env and not key:
            raise HarnessError(
                "provider",
                "missing_credentials",
                f"Set {config.api_key_env} in the daemon environment",
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        params = dict(config.parameters)
        # These fields are owned by the harness, including the reserved token maximum.
        for reserved in (
            "model",
            "messages",
            "tools",
            "stream",
            "stream_options",
            "n",
            "max_tokens",
            "max_completion_tokens",
        ):
            params.pop(reserved, None)
        body = {
            **params,
            "model": config.model,
            "messages": request.messages,
            "max_completion_tokens": config.max_output_tokens,
            "stream": config.streaming,
            "n": 1,
        }
        if request.tools:
            body["tools"] = request.tools
        if config.streaming:
            body["stream_options"] = {"include_usage": True}
        try:
            async with httpx.AsyncClient(
                timeout=config.timeout_seconds, transport=self.transport
            ) as client:
                url = config.base_url.rstrip("/") + "/chat/completions"
                if config.streaming:
                    async with client.stream("POST", url, json=body, headers=headers) as response:
                        if response.is_error:
                            await response.aread()
                        self._check(response)
                        data = await self._collect(response.aiter_lines(), emit)
                else:
                    response = await client.post(url, json=body, headers=headers)
                    self._check(response)
                    data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise HarnessError(
                    "model", "empty_response", "Provider returned no choices", retryable=True
                )
            message = choices[0]["message"]
            actions = []
            for call in message.get("tool_calls", []) or []:
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                actions.append(
                    Action(id=call["id"], name=call["function"]["name"], arguments=arguments)
                )
            raw_usage = data.get("usage") or {}
            input_tokens = raw_usage.get("prompt_tokens", request.input_token_bound)
            output_tokens = raw_usage.get("completion_tokens", config.max_output_tokens)
            cost = raw_usage.get(
                "cost",
                (
                    input_tokens * (config.input_cost_per_million or 0)
                    + output_tokens * (config.output_cost_per_million or 0)
                )
                / 1_000_000,
            )
            return ModelResponse(
                text=message.get("content") or "",
                actions=actions,
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost),
                usage_reported=bool(raw_usage),
                provider_id=data.get("id"),
            )
        except HarnessError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise HarnessError(
                "provider", type(exc).__name__, str(exc), retryable=True, uncertain=True
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise HarnessError("model", "invalid_response", str(exc), retryable=True) from exc

    @staticmethod
    def _check(response):
        if response.is_error:
            # Never persist request headers or credentials in provider failures.
            raise HarnessError(
                "provider",
                f"http_{response.status_code}",
                f"Provider returned HTTP {response.status_code}",
                retryable=response.status_code in (408, 409, 429) or response.status_code >= 500,
            )

    @staticmethod
    async def _collect(lines: AsyncIterator[str], emit):
        text, calls, usage, provider_id = [], {}, None, None
        finished = False
        async for line in lines:
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                finished = True
                break
            data = json.loads(payload)
            if data.get("error"):
                raise HarnessError(
                    "provider", "stream_error", "Provider stream reported an error", retryable=True
                )
            provider_id = data.get("id", provider_id)
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    text.append(delta["content"])
                    await emit(delta["content"])
                for item in delta.get("tool_calls", []):
                    call = calls.setdefault(
                        item["index"],
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if item.get("id"):
                        call["id"] = item["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += item.get("function", {}).get(key) or ""
        if not finished:
            raise HarnessError(
                "provider",
                "incomplete_stream",
                "Stream ended before [DONE]",
                retryable=True,
                uncertain=True,
            )
        return {
            "id": provider_id,
            "usage": usage,
            "choices": [
                {
                    "message": {
                        "content": "".join(text),
                        "tool_calls": [calls[i] for i in sorted(calls)],
                    }
                }
            ],
        }


class ScriptedProvider:
    """Deterministic test provider. The persisted turn index selects each response."""

    def __init__(self, scripts: dict[str, list], *, delay=0):
        self.scripts, self.delay = scripts, delay
        self.requests: list[ModelRequest] = []
        self.active = 0
        self.peak_active = 0

    async def invoke(self, request, emit):
        self.requests.append(request)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            sequence = self.scripts.get(request.name, self.scripts.get("*", []))
            item = (
                sequence[request.turn]
                if request.turn < len(sequence)
                else ModelResponse(
                    actions=[Action(name="finish", arguments={"result": "Script complete"})]
                )
            )
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item = item(request)
                if hasattr(item, "__await__"):
                    item = await item
            return item if isinstance(item, ModelResponse) else ModelResponse.model_validate(item)
        finally:
            self.active -= 1


class DemoProvider:
    """An offline executable example, not a model and not runtime planning logic."""

    async def invoke(self, request, emit):
        await asyncio.sleep(0.03)
        if request.parent_id:
            steps = [
                Action(
                    name="python", arguments={"code": "values = list(range(1000))\nsum(values)"}
                ),
                Action(
                    name="agent_message",
                    arguments={
                        "recipient_id": request.parent_id,
                        "body": "I computed sum(range(1000)) = 499500 in my persistent worker.",
                    },
                ),
                Action(name="finish", arguments={"result": "Independent computation verified."}),
            ]
            action = steps[min(request.turn, len(steps) - 1)]
        else:
            steps = [
                Action(
                    name="python", arguments={"code": "values = list(range(1000))\nlen(values)"}
                ),
                Action(
                    name="agent_spawn",
                    arguments={
                        "instruction": "Independently check sum(range(1000)).",
                        "name": "checker",
                    },
                ),
                Action(name="python", arguments={"code": "answer = sum(values)\nprint(answer)"}),
                Action(name="agent_wait", arguments={"seconds": 0.3}),
                Action(
                    name="python",
                    arguments={
                        "code": "tools.call('workspace_write', path='answer.txt', content=str(answer))\nanswer"
                    },
                ),
            ]
            action = (
                steps[request.turn]
                if request.turn < len(steps)
                else Action(
                    name="finish",
                    arguments={
                        "result": "Computed 499500; saved answer.txt. Child has its own history."
                    },
                )
            )
        return ModelResponse(actions=[action], usage=Usage(input_tokens=200, output_tokens=60))


def default_providers() -> dict[str, Provider]:
    return {"demo": DemoProvider(), "chat": ChatProvider()}
