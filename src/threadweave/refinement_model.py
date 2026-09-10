"""Prime's selected-model metadata and one-shot Chat request construction.

The bundled projection is from the supplied packages/ai/src/models.generated.ts,
restricted to Buffalo's supported transports. Custom definitions use Prime's
model-registry.ts defaults (maxTokens=16384, reasoning=false), not turn budgets.
"""

import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def model_catalog():
    return json.loads(Path(__file__).with_name("prime_refinement_models.json").read_text())


def selected_model(config):
    url = (
        "https://chatgpt.com/backend-api"
        if config.name == "codex_subscription"
        else config.base_url.rstrip("/")
    )
    registered = model_catalog().get(url, {}).get(config.model, {})
    overrides = config.model_metadata.model_dump(exclude_none=True)
    return {
        "maxTokens": 16384,
        "reasoning": False,
        **registered,
        **{k: v for k, v in overrides.items() if k != "compat"},
        "compat": {**registered.get("compat", {}), **overrides["compat"]},
    }


def refinement_output_limit(config, *, review):
    return min(selected_model(config)["maxTokens"], 4096 if review else 32000)


def chat_refinement_body(config, messages):
    """One-shot/no-tools slice of Prime openai-completions buildParams/getCompat."""
    url = config.base_url
    model = selected_model(config)
    zai, moonshot = "api.z.ai" in url, "api.moonshot." in url
    cloudflare = "api.cloudflare.com" in url or "gateway.ai.cloudflare.com" in url
    gateway, prime = "gateway.ai.cloudflare.com" in url, "api.pinference.ai" in url
    nonstandard = (
        any(
            part in url
            for part in (
                "cerebras.ai",
                "api.x.ai",
                "chutes.ai",
                "deepseek.com",
                "opencode.ai",
            )
        )
        or zai
        or moonshot
        or cloudflare
        or prime
    )
    compat = {
        "supportsStore": not nonstandard,
        "supportsDeveloperRole": not nonstandard,
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens"
        if ("chutes.ai" in url or moonshot or gateway or prime)
        else "max_completion_tokens",
        "thinkingFormat": "deepseek" if "deepseek.com" in url else "zai" if zai else "openai",
        **model["compat"],
    }
    body = {
        "model": config.model,
        "messages": [
            {**message, "role": "developer"}
            if message["role"] == "system"
            and model["reasoning"]
            and compat["supportsDeveloperRole"]
            else dict(message)
            for message in messages
        ],
        "stream": True,
        compat["maxTokensField"]: config.max_output_tokens,
    }
    if compat["supportsStore"]:
        body["store"] = False
    if compat["supportsUsageInStreaming"]:
        body["stream_options"] = {"include_usage": True}
    if model["reasoning"]:
        thinking = compat["thinkingFormat"]
        if thinking in {"zai", "qwen"}:
            body["enable_thinking"] = False
        elif thinking == "qwen-chat-template":
            body["chat_template_kwargs"] = {"enable_thinking": False, "preserve_thinking": True}
        elif thinking == "deepseek":
            body["thinking"] = {"type": "disabled"}
    if "openrouter.ai" in url and compat.get("openRouterRouting"):
        body["provider"] = compat["openRouterRouting"]
    cache_format = compat.get(
        "cacheControlFormat",
        "anthropic"
        if (config.model.startswith("anthropic/") and ("openrouter.ai" in url or prime))
        else None,
    )
    if cache_format == "anthropic":
        cache = {"type": "ephemeral"}
        if os.environ.get("PI_CACHE_RETENTION") == "long" and compat.get(
            "supportsLongCacheRetention", not cloudflare
        ):
            cache["ttl"] = "1h"
        for message in (body["messages"][0], body["messages"][-1]):
            if isinstance(message.get("content"), str) and message["content"]:
                message["content"] = [
                    {"type": "text", "text": message["content"], "cache_control": cache}
                ]
    return body
