"""Prime model-budget, provider wire and failure contracts; no network calls."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from threadweave.models import ModelResponse, ProviderConfig
from threadweave.providers import ChatProvider
from threadweave.refinement_model import (
    chat_refinement_body,
    refinement_output_limit,
    selected_model,
)
from threadweave.refinement_retry import complete_refinement, completion_metadata, retry_after_ms
from threadweave.subscription import SubscriptionProvider

from .test_continual_harness import harness_runtime as harness_runtime
from .test_subscription import discard, model_request, stream


@pytest.mark.parametrize(
    "limit,review,expected",
    [
        (8192, False, 8192),
        (128000, False, 32000),
        (2048, False, 2048),
        (128000, True, 4096),
        (2048, True, 2048),
    ],
)
async def test_selected_model_cap_not_primary_turn_budget(harness_runtime, limit, review, expected):
    # Prime refinement.test.ts:985,1050; same rule for the automatic reviewer.
    rt, sid, provider = harness_runtime
    config = rt.store.config(sid)
    config.provider.max_output_tokens = 128
    config.provider.model_metadata.maxTokens = limit
    config.provider.parameters = {"reasoning_effort": "high", "temperature": 0.1}
    rt.store.reconfigure(sid, config)
    if review:
        await rt.review_refinement(sid, "turn_interval")
    else:
        await rt.plan_refinement(sid, {})
    request = provider.requests[-1]
    assert request.config.max_output_tokens == expected
    assert request.config.parameters == {}
    assert request.config.model == config.provider.model
    assert rt.store.config(sid).provider.max_output_tokens == 128


def test_catalog_metadata_and_custom_model_defaults_are_prime_values():
    codex = ProviderConfig(model="gpt-5.4", max_output_tokens=128)
    assert selected_model(codex)["maxTokens"] == 128000
    assert refinement_output_limit(codex, review=False) == 32000
    custom = ProviderConfig(name="chat", model="custom", base_url="https://local.test")
    assert selected_model(custom)["maxTokens"] == 16384
    assert selected_model(custom)["reasoning"] is False


@pytest.mark.parametrize(
    "url,meta,field,role,extra",
    [
        (
            "https://api.openai.com/v1",
            {"reasoning": True},
            "max_completion_tokens",
            "developer",
            {"store": False},
        ),
        ("https://api.pinference.ai/api/v1", {}, "max_tokens", "system", {}),
        (
            "https://api.deepseek.com",
            {"reasoning": True},
            "max_completion_tokens",
            "system",
            {"thinking": {"type": "disabled"}},
        ),
        (
            "https://api.z.ai/api/coding/paas/v4",
            {"reasoning": True},
            "max_completion_tokens",
            "system",
            {"enable_thinking": False},
        ),
        (
            "https://local.test",
            {"compat": {"maxTokensField": "max_tokens", "supportsStore": False}},
            "max_tokens",
            "system",
            {},
        ),
    ],
)
def test_prime_one_shot_chat_wire(url, meta, field, role, extra):
    config = ProviderConfig(name="chat", model="custom", base_url=url, model_metadata=meta)
    messages = [
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": "evidence"},
    ]
    body = chat_refinement_body(config, messages)
    assert body[field] == config.max_output_tokens
    assert body["messages"][0]["role"] == role
    assert body["stream"] is True and "n" not in body and "reasoning_effort" not in body
    assert all(body[key] == value for key, value in extra.items())
    assert messages[0]["role"] == "system"


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "stop"),
        ("end", "stop"),
        ("length", "length"),
        ("tool_calls", "toolUse"),
        ("content_filter", "error"),
        ("network_error", "error"),
        ("unknown_finish", "error"),
    ],
)
def test_prime_finish_reason_mapping(reason, expected):
    assert completion_metadata(reason)["stop_reason"] == expected


def test_retry_after_milliseconds_seconds_and_http_date(monkeypatch):
    monkeypatch.setattr("threadweave.refinement_retry.time.time", lambda: 0)
    assert retry_after_ms({"Retry-After-Ms": "1500", "Retry-After": "7"}) == 1500
    assert retry_after_ms({"Retry-After": "7"}) == 7000
    assert retry_after_ms({"Retry-After": "Thu, 01 Jan 1970 00:00:09 GMT"}) == 9000
    assert retry_after_ms({"Retry-After": "garbage"}) is None


@pytest.mark.parametrize(
    "status,kind",
    [
        (400, "invalid_request"),
        (401, "auth"),
        (403, "permission"),
        (404, "invalid_request"),
        (429, "rate_limit"),
        (529, "overloaded"),
    ],
)
def test_http_failures_preserve_prime_retry_classification(status, kind):
    from threadweave.models import HarnessError

    with pytest.raises(HarnessError) as raised:
        ChatProvider._check(httpx.Response(status, headers={"Retry-After-Ms": "1200"}))
    assert raised.value.provider_failure == {"kind": kind, "retryAfterMs": 1200}


async def test_retry_cancelled_during_backoff_returns_aborted(harness_runtime, monkeypatch):
    # Prime provider-retry.test.ts:27.
    rt, sid, _ = harness_runtime
    rt._model_call = AsyncMock(
        return_value=(ModelResponse(metadata={"stop_reason": "error"}), "event")
    )
    entered = asyncio.Event()

    async def sleep(_):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr("threadweave.refinement_retry.asyncio.sleep", sleep)
    task = asyncio.create_task(complete_refinement(rt, sid, model_request()))
    await entered.wait()
    task.cancel()
    response, event = await task
    assert response.metadata["stop_reason"] == "aborted" and event == "event"
    assert rt._model_call.await_count == 1


async def test_disabled_retry_makes_one_attempt(harness_runtime):
    # Prime provider-retry.test.ts:39.
    rt, sid, _ = harness_runtime
    config = rt.store.config(sid)
    config.provider_retry.enabled = False
    rt.store.reconfigure(sid, config)
    response = ModelResponse(metadata={"stop_reason": "error"})
    rt._model_call = AsyncMock(return_value=(response, None))
    assert (await complete_refinement(rt, sid, model_request()))[0] is response
    assert rt._model_call.await_count == 1


async def test_native_refinement_truncation_and_retry_metadata():
    from threadweave.models import HarnessError

    request = model_request().model_copy(update={"metadata": {"purpose": "refinement"}})
    response = await SubscriptionProvider.collect(
        stream(
            [
                {"type": "text_delta", "text": '{"edits":['},
                {"type": "completed", "stop_reason": "length"},
            ]
        ),
        request,
        request.config,
        discard,
    )
    assert response.metadata["stop_reason"] == "length" and response.text == '{"edits":['
    with pytest.raises(HarnessError) as raised:
        await SubscriptionProvider.collect(
            stream(
                [
                    {
                        "type": "error",
                        "code": "http_429",
                        "status": 429,
                        "retry_headers": {"retry-after-ms": "4500"},
                    }
                ]
            ),
            request,
            request.config,
            discard,
        )
    assert raised.value.provider_failure == {"kind": "rate_limit", "retryAfterMs": 4500}
    assert "retry_headers" not in str(raised.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numeric_constants_are_rejected(constant):
    from threadweave.refinement import parse_object

    with pytest.raises(ValueError, match="did not return valid JSON"):
        parse_object('{"summary": ' + constant + ', "edits": []}')


async def test_prime_chat_refinement_allows_stream_eof_without_done_marker():
    async def lines():
        yield 'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}'

    result = await ChatProvider._collect(lines(), discard, require_done=False)
    assert result["choices"][0]["message"]["content"] == "{}"


async def test_aborted_provider_tag_alone_does_not_replace_json_parsing(harness_runtime):
    # Prime planRefinement/reviewAutoRefine special-case error and length only.
    # Authoritative cancellation is enforced by session ownership, not a tag.
    rt, sid, provider = harness_runtime
    provider.invoke = AsyncMock(
        return_value=ModelResponse(
            text='{"edits": []}',
            metadata={"stop_reason": "aborted"},
        )
    )
    plan = await rt.plan_refinement(sid, {})
    assert plan["proposal"]["edits"] == []
    assert not rt.store.harness.history(sid)
