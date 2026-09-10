"""Prime Codex SSE normalization before the native SDK can discard error details."""

import codecs
import json
import re
import time

from .refinement_retry import retry_after_ms


def codex_error(payload, *, status=None, headers=None, prefix="", fallback="Codex response failed"):
    has_error = payload is not None and payload is not False and payload != "" and payload != 0
    payload = payload if isinstance(payload, dict) else {}
    code = payload.get("code") or payload.get("type")
    message = payload.get("message") or fallback
    delay = retry_after_ms(headers or {})
    if has_error and (
        re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code or "", re.I)
        or status == 429
    ):
        reset = payload.get("resets_at")
        reset_delay = (
            max(0, reset * 1000 - time.time() * 1000) if isinstance(reset, (int, float)) else None
        )
        plan = f" ({payload['plan_type'].lower()} plan)" if payload.get("plan_type") else ""
        when = (
            f" Try again in ~{int(reset_delay / 60000 + 0.5)} min."
            if reset_delay is not None
            else ""
        )
        message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
        if reset_delay is not None:
            delay = max(delay or 0, reset_delay)
    else:
        message = prefix + message
    return {
        "type": "error",
        "code": code or "provider_failure",
        "provider_error_type": code,
        "error_message": message,
        "status": status,
        "retryAfterMs": delay,
        "retryable": True,
    }


def normalize_event(event):
    if event is None:
        raise ValueError("Invalid Codex event: null")
    if not isinstance(event, dict):
        return None
    kind = event.get("type")
    if kind == "response.output_text.delta":
        return {"type": "text_delta", "text": event.get("delta", "")}
    if kind == "response.output_item.done":
        return {"type": "item", "item": event["item"]}
    if kind == "error":
        nested = event.get("error")
        nested = nested if isinstance(nested, dict) else None
        payload = {
            "code": (event.get("code") if isinstance(event.get("code"), str) else None)
            or (nested or {}).get("code")
            or (nested or {}).get("type"),
            "message": (event.get("message") if isinstance(event.get("message"), str) else None)
            or (nested or {}).get("message"),
        }
        status = event.get("status_code")
        status = status if type(status) in (int, float) else None
        result = {
            "type": "error",
            "code": payload["code"] or "provider_failure",
            "provider_error_type": payload["code"],
            "status": status,
            "retryable": True,
            "error_message": "Codex error: "
            + (payload["message"] or payload["code"] or json.dumps(event, separators=(",", ":"))),
        }
        # Prime only expands usage-limit messages from the nested error payload.
        if nested is not None and (
            status == 429
            or re.search(
                r"usage_limit_reached|usage_not_included|rate_limit_exceeded",
                nested.get("code") or nested.get("type") or "",
                re.I,
            )
        ):
            usage = codex_error(nested, status=status)
            result.update(error_message=usage["error_message"], retryAfterMs=usage["retryAfterMs"])
        return result
    if kind == "response.failed":
        payload = (event.get("response") or {}).get("error") or {}
        # response.failed only uses code/message, unlike top-level usage errors.
        return {
            "type": "error",
            "code": payload.get("code") or "provider_failure",
            "provider_error_type": payload.get("code"),
            "error_message": payload.get("message") or "Codex response failed",
            "retryable": True,
        }
    if kind in {"response.completed", "response.done", "response.incomplete"}:
        response = event.get("response") or {}
        status = response.get("status")
        reason = (
            "length"
            if status == "incomplete"
            else "error"
            if status in {"failed", "cancelled"}
            else "stop"
        )
        usage = response.get("usage") or {}
        return {
            "type": "completed",
            "id": response.get("id"),
            "stop_reason": reason,
            "usage": {
                **usage,
                "cached_input_tokens": (usage.get("input_tokens_details") or {}).get(
                    "cached_tokens", 0
                ),
            }
            if usage
            else {},
        }
    return None


async def native_events(lines):
    """Ordinary native events pass through; one-shot requests use Prime's SSE framing."""
    decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
    buffer = ""
    async for line in lines:
        event = json.loads(line)
        if event.get("type") == "refinement_event":
            normalized = normalize_event(event["event"])
            if normalized:
                yield normalized
                if normalized["type"] in {"error", "completed"}:
                    return
        elif event.get("type") == "refinement_sse":
            buffer += decoder.decode(bytes(event["bytes"]))
            while "\n\n" in buffer:
                chunk, buffer = buffer.split("\n\n", 1)
                data = "\n".join(
                    line[5:].strip() for line in chunk.split("\n") if line.startswith("data:")
                ).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    raw = json.loads(
                        data,
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("Invalid JSON constant")
                        ),
                    )
                except ValueError as exc:
                    yield {
                        "type": "error",
                        "code": "provider_failure",
                        "error_message": f"Invalid Codex SSE JSON: {exc}",
                        "retryable": True,
                    }
                    return
                normalized = normalize_event(raw)
                if normalized:
                    yield normalized
                    if normalized["type"] in {"error", "completed"}:
                        return
        elif event.get("type") == "refinement_end":
            # Prime's SSE path accepts EOF, even without a terminal event.
            yield {"type": "completed"}
            return
        elif "provider_body" in event:
            body = event.get("provider_body") or ""
            try:
                payload = json.loads(body).get("error")
            except (ValueError, AttributeError):
                payload = None
            yield codex_error(
                payload,
                status=event.get("status"),
                headers=event.get("retry_headers"),
                fallback=body or event.get("status_text") or "Request failed",
            )
        else:
            yield event
