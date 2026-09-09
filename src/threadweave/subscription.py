"""Single model requests using Codex's official auth and Responses client.

Only Threadweave's context and function schemas enter the request. There is no
Codex agent/session and no tool execution in this provider. All requests replay
Threadweave-owned history, so recovery does not depend on a Codex conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal

from .codex_auth import CodexControl
from .models import Action, HarnessError, ModelResponse, Usage
from .native_client import CODEX_REVISION, client_path


def responses_input(messages, *, provider=None):
    instructions, items = [], []
    for message in messages:
        role, content = message["role"], message.get("content")
        if message.get("provider_items") and (
            provider is None or message.get("provider_identity") == provider
        ):
            items.extend(message["provider_items"])
            continue
        if role == "system":
            instructions.append(content or "")
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": content or "",
                }
            )
        else:
            if content:
                items.append({"role": role, "content": content})
            for call in message.get("tool_calls", []):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
    return "\n\n".join(instructions), items


def responses_payload(messages, tools, config):
    """The same logical input is used for transmission and context estimation."""
    instructions, items = responses_input(messages, provider=[config.name, config.model])
    return {
        "instructions": instructions,
        "input": items,
        "tools": [{"type": "function", **tool["function"]} for tool in tools],
    }


class SubscriptionProvider:
    def __init__(self, *, control_factory=CodexControl, executable=None):
        self.control_factory = control_factory
        # Pin the installed client for this provider's lifetime. An in-place package
        # upgrade must not redirect a running daemon to a not-yet-built fingerprint.
        self.executable = executable or client_path()

    async def resolve(self, config):
        async with self.control_factory() as control:
            status = await control.status()
            if not status["logged_in"]:
                raise HarnessError(
                    "provider",
                    "AUTH_REQUIRED",
                    "Run threadweave auth login (ChatGPT authentication; no API key)",
                )
            settings, models = await control.settings(), await control.models()
        selected = config.model or settings["model"]
        if not selected:
            selected = next((model["model"] for model in models if model["isDefault"]), None)
        catalog = next((model for model in models if model["model"] == selected), None)
        if not catalog:
            raise HarnessError(
                "provider",
                "MODEL_UNAVAILABLE",
                "Selected model is not in the current Codex account catalog; run threadweave auth models",
            )
        parameters = dict(config.parameters)
        unknown = set(parameters) - {
            "reasoning_effort",
            "reasoning_summary",
            "verbosity",
            "parallel_tool_calls",
        }
        if unknown:
            raise HarnessError(
                "provider",
                "UNSUPPORTED_PARAMETER",
                "Subscription parameters supported: reasoning_effort, reasoning_summary, verbosity, parallel_tool_calls",
            )
        effort = parameters.setdefault(
            "reasoning_effort", settings["reasoning_effort"] or catalog["defaultReasoningEffort"]
        )
        if effort not in {item["reasoningEffort"] for item in catalog["supportedReasoningEfforts"]}:
            raise HarnessError(
                "provider",
                "UNSUPPORTED_REASONING",
                "Reasoning effort is not supported by the selected Codex model",
            )
        resolved = config.model_copy(update={"model": selected, "parameters": parameters})
        return resolved, settings

    async def invoke(self, request, emit):
        config, settings = await self.resolve(request.config)
        executable = self.executable
        if not executable.is_file():
            raise HarnessError(
                "provider",
                "CLIENT_NOT_INSTALLED",
                "Run threadweave auth install-client to build the official Codex inference client",
            )
        body = {
            "model": config.model,
            **responses_payload(request.messages, request.tools, config),
            "tool_choice": "auto",
            "parallel_tool_calls": config.parameters.get("parallel_tool_calls", True),
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": request.session_id,
            "reasoning": {
                "effort": config.parameters["reasoning_effort"],
                "summary": config.parameters.get("reasoning_summary", "auto"),
            },
        }
        if config.parameters.get("verbosity"):
            body["text"] = {"verbosity": config.parameters["verbosity"]}
        environment = dict(os.environ)
        for key in (
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT",
            "RUST_LOG",
        ):
            environment.pop(key, None)
        process = await asyncio.create_subprocess_exec(
            str(executable),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
            limit=8 * 1024 * 1024,
        )
        packet = {
            "operation": "invoke",
            "auth_settings": settings,
            "refresh_lock": str(executable.parent / "refresh.lock"),
            "body": body,
            "request_id": request.request_id,
        }
        try:
            async with asyncio.timeout(config.timeout_seconds):
                process.stdin.write((json.dumps(packet) + "\n").encode())
                await process.stdin.drain()
                process.stdin.close()
                result = await self.collect(process.stdout, request, config, emit)
                await process.wait()
                if process.returncode:
                    raise HarnessError(
                        "provider",
                        "CLIENT_EXIT",
                        "Codex inference client exited unsuccessfully",
                        uncertain=True,
                    )
                return result
        except TimeoutError as exc:
            raise HarnessError(
                "provider",
                "timeout",
                "Subscription inference timed out",
                retryable=True,
                uncertain=True,
            ) from exc
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 2)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    @staticmethod
    async def collect(lines, request, config, emit):
        text, actions, summaries, limits, items = [], [], [], [], []
        metadata = {
            "transport": "official_codex_responses_client",
            "client_revision": CODEX_REVISION,
            "model": config.model,
            "parameters": config.parameters,
            "billing": "subscription",
            "output_limit": "provider model token limit; cumulative runtime token budget",
            "auth_refreshes": 0,
        }
        allowed = {tool["function"]["name"] for tool in request.tools}
        try:
            async for line in lines:
                event = json.loads(line)
                kind = event.get("type")
                if kind in ("text_delta", "tool_delta", "reasoning_summary"):
                    delta = event.get("text", event.get("delta", ""))
                    if kind == "text_delta":
                        text.append(delta)
                        if config.streaming:
                            await emit(delta)
                    elif kind == "reasoning_summary" and sum(map(len, summaries)) < 8000:
                        summaries.append(delta)
                elif kind == "item":
                    item = event["item"]
                    if item.get("type") in {"reasoning", "message", "function_call"}:
                        items.append(item)
                    if item.get("type") == "function_call":
                        if item["name"] not in allowed:
                            raise ValueError("Unregistered tool returned by model")
                        arguments = json.loads(item["arguments"])
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be an object")
                        actions.append(
                            Action(id=item["call_id"], name=item["name"], arguments=arguments)
                        )
                elif kind == "rate_limits":
                    limits.append(event["limits"])
                elif kind == "auth_refresh":
                    metadata["auth_refreshes"] += 1
                elif kind == "model":
                    metadata["server_model"] = event["model"]
                elif kind == "error":
                    code = event.get("code", "provider_failure")
                    # Error codes are client-owned. Never relay server bodies, headers or credentials.
                    if (
                        not isinstance(code, str)
                        or len(code) > 60
                        or not code.replace("_", "").isalnum()
                    ):
                        code = "provider_failure"
                    message = (
                        "Run threadweave auth login"
                        if code == "AUTH_REQUIRED"
                        else f"Subscription inference failed: {code}"
                    )
                    raise HarnessError(
                        "provider",
                        code,
                        message,
                        retryable=bool(event.get("retryable")),
                        uncertain=True,
                    )
                elif kind == "completed":
                    raw = event.get("usage") or {}
                    usage = Usage(
                        input_tokens=raw.get("input_tokens", request.input_token_bound),
                        output_tokens=raw.get("output_tokens", config.max_output_tokens),
                        cached_input_tokens=raw.get("cached_input_tokens", 0),
                        reasoning_output_tokens=raw.get("reasoning_output_tokens", 0),
                        cost=None,
                    )
                    metadata.update(
                        rate_limits=limits,
                        reasoning_summary="".join(summaries),
                        end_turn=event.get("end_turn"),
                    )
                    return ModelResponse(
                        text="".join(text),
                        actions=actions,
                        usage=usage,
                        usage_reported=bool(raw),
                        provider_id=event.get("id"),
                        provider_items=items,
                        metadata=metadata,
                    )
        except (ValueError, KeyError, TypeError) as exc:
            raise HarnessError(
                "model",
                "invalid_response",
                "Invalid structured subscription response",
                retryable=True,
                uncertain=True,
            ) from exc
        raise HarnessError(
            "provider",
            "incomplete_stream",
            "Subscription stream ended before completion",
            retryable=True,
            uncertain=True,
        )
