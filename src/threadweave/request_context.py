"""Provider-shaped occupancy estimates, separate from cumulative usage accounting.

Anchors contain hashes and reported counts, never opaque continuation contents.
Unknown or rewritten windows use a conservative full-request estimate.
"""

import hashlib
import math

from .storage import encode
from .tokenization import estimate


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def projection(messages, tools, provider):
    if provider.name == "codex_subscription":
        from .subscription import responses_payload

        return responses_payload(messages, tools, provider)
    from .providers import chat_messages

    return {"input": chat_messages(messages), "tools": tools}


def parts(messages, tools, provider):
    stable = projection([m for m in messages if not m.get("context_status")], tools, provider)
    volatile = projection([m for m in messages if m.get("context_status")], [], provider)
    key = fingerprint(
        {
            "provider": {
                "name": provider.name,
                "model": provider.model,
                "endpoint": provider.base_url,
                "parameters": provider.parameters,
            },
            **{k: v for k, v in stable.items() if k != "input"},
        }
    )
    return key, stable["input"], volatile["input"]


def request_estimate(store, sid, messages, tools, provider):
    full = projection(messages, tools, provider)
    fallback = estimate(full, provider.model)
    if provider.name not in {"codex_subscription", "chat"}:
        return fallback, {"method": "provider_projection", "authoritative": False}
    key, items, volatile = parts(messages, tools, provider)
    rows = store.events(sid, kind="context_usage_anchor", limit=1)
    if rows:
        anchor = rows[0]["payload"]
        hashes = anchor["item_hashes"]
        # Messages received during an action may be committed before the
        # assistant/tools block. The previous items still occur in order, but
        # are no longer a strict prefix. Charge every insertion separately;
        # never tokenize unchanged opaque continuation as if it were prose.
        matched, additions = 0, []
        if key == anchor["key"]:
            for item in items:
                if matched < len(hashes) and fingerprint(item) == hashes[matched]:
                    matched += 1
                else:
                    additions.append(item)
        if key == anchor["key"] and matched == len(hashes):
            # Include the previous response's entire reported generation, including
            # reasoning, rather than tokenizing its encrypted wire representation.
            # Previous transient input is not subtracted: this is conservative.
            occupancy = math.ceil((anchor["input_tokens"] + anchor["output_tokens"]) * 1.12)
            occupancy += estimate([*additions, *volatile], provider.model)
            return occupancy, {
                "method": "reported_usage_plus_appended_input",
                "anchor_event": rows[0]["id"],
                "projection_tokens": fallback,
                "authoritative": False,
            }
    return fallback, {"method": "provider_projection", "authoritative": False}


def record_usage(store, sid, request, response, response_event):
    if not response.usage_reported or request.config.name not in {"codex_subscription", "chat"}:
        return
    key, items, _ = parts(request.messages, request.tools, request.config)
    assistant = {"role": "assistant", "content": response.text or None}
    if response.actions:
        assistant["tool_calls"] = [
            {
                "id": a.id,
                "type": "function",
                "function": {"name": a.name, "arguments": encode(a.arguments)},
            }
            for a in response.actions
        ]
    if response.provider_items:
        assistant.update(
            provider_items=response.provider_items,
            provider_identity=[request.config.name, request.config.model],
        )
    emitted = projection([assistant], [], request.config)["input"]
    store.event(
        sid,
        "context_usage_anchor",
        {
            "key": key,
            "item_hashes": [fingerprint(item) for item in [*items, *emitted]],
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "previous_estimate": request.input_token_bound,
        },
        parent=response_event,
    )
