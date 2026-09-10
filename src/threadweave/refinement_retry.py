"""Prime's one-shot provider retry policy, separate from primary-turn recovery."""

import asyncio
import math
import re
import time
from email.utils import parsedate_to_datetime

from .models import HarnessError, Usage


def classify_failure(provider_type=None, status=None):
    """Prime ai/utils/stream-failure.ts classification, in precedence order."""
    kind = (provider_type or "").lower()
    if kind == "refusal":
        return "refusal"
    if re.search(
        r"sensitive|safety|prohibited_content|blocklist|spii|recitation|content.?filter|guardrail|flagged",
        kind,
    ):
        return "safety"
    if "overloaded" in kind or status == 529:
        return "overloaded"
    if re.search(r"rate_limit|usage_limit|usage_not_included|throttl", kind) or status == 429:
        return "rate_limit"
    if re.search(r"authentication|unauthorized", kind) or status == 401:
        return "auth"
    if re.search(r"permission|forbidden|access.?denied", kind) or status == 403:
        return "permission"
    if "invalid_request" in kind or "not_found_error" in kind or status in {400, 404}:
        return "invalid_request"
    if "malformed" in kind:
        return "malformed_response"
    if re.search(r"api_error|server_error|unavailable", kind) or (
        status is not None and status >= 500
    ):
        return "server_error"
    return "unknown"


def retry_after_ms(headers):
    headers = {key.lower(): value for key, value in headers.items()}
    for name, multiplier in (("retry-after-ms", 1), ("retry-after", 1000)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip() or "0")
            if math.isfinite(value) and value >= 0:
                return value * multiplier
        except (ValueError, TypeError):
            pass
    try:
        return max(
            0, (parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time()) * 1000
        )
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


def completion_metadata(reason):
    if reason in {None, "stop", "end"}:
        return {"stop_reason": "stop"}
    if reason == "length":
        return {"stop_reason": "length"}
    if reason in {"function_call", "tool_calls"}:
        return {"stop_reason": "toolUse"}
    return {
        "stop_reason": "error",
        "error_message": f"Provider finish_reason: {reason}",
        "diagnostics": [
            {
                "type": "provider_stream_failure",
                "details": {
                    "kind": classify_failure(reason),
                    "providerErrorType": reason,
                },
            }
        ],
    }


def retry_delay(attempt, retry_after, *, base_delay=2, max_retry_delay=60):
    if retry_after is not None and max_retry_delay > 0 and retry_after > max_retry_delay:
        return None
    return min(max(base_delay * 2 ** (attempt - 1), retry_after or 0), 2147483.647)


def response_failure(response):
    metadata = response.metadata
    if metadata.get("stop_reason") != "error":
        return None
    diagnostics = metadata.get("diagnostics", [])
    if any(d.get("type") == "agent_lifecycle_failure" for d in diagnostics):
        return {"permanent": True}
    if (
        metadata.get("provider") == "faux"
        and metadata.get("error_message") == "No more faux responses queued"
    ):
        return {"permanent": True}
    details = next(
        (d.get("details", {}) for d in diagnostics if d.get("type") == "provider_stream_failure"),
        {},
    )
    return details if isinstance(details, dict) else {}


async def complete_refinement(runtime, sid, request):
    configured = runtime.store.config(sid).provider_retry
    attempts = configured.max_retries + 1 if configured.enabled else 1
    base, cap = configured.base_delay, configured.max_retry_delay
    for attempt in range(attempts):
        error = None
        try:
            result = await runtime._model_call(sid, request, persist_turn=False, max_attempts=1)
            failure = response_failure(result[0])
            if failure is None:
                return result
        except HarnessError as exc:
            error = exc
            failure = getattr(exc, "provider_failure", None)
            if failure is None:
                if not exc.failure.retryable:
                    raise
                failure = {}
        kind = failure.get("kind")
        permanent = (
            failure.get("permanent")
            or kind in {"invalid_request", "refusal", "permission"}
            or (kind == "auth" and attempt > 0)
        )
        retry_after = failure.get("retryAfterMs")
        retry_after = (
            retry_after / 1000 if type(retry_after) in (int, float) and retry_after >= 0 else None
        )
        delay = retry_delay(attempt + 1, retry_after, base_delay=base, max_retry_delay=cap)
        if permanent or attempt + 1 >= attempts or delay is None:
            if error:
                raise error
            return result
        runtime.store.charge(sid, Usage(retries=1))
        runtime.store.event(
            sid, "retry", {"attempt": attempt + 2, "delay": delay, "category": "provider"}
        )
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            if error:
                raise
            # Prime changes the failed response to aborted when cancellation
            # wins during backoff; the refinement parser then rejects it.
            response, event = result
            return response.model_copy(
                update={
                    "metadata": {
                        **response.metadata,
                        "stop_reason": "aborted",
                    }
                }
            ), event
    raise AssertionError("Unreachable refinement retry loop")
