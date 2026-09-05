import asyncio
import json

import httpx
import pytest

from threadweave.models import HarnessError, ModelRequest, ProviderConfig
from threadweave.providers import ChatProvider


def request_config(**changes):
    return ModelRequest(
        session_id="root",
        root_id="root",
        parent_id=None,
        name="root",
        turn=0,
        messages=[{"role": "user", "content": "Do the task"}],
        tools=[],
        input_token_bound=500,
        config=ProviderConfig(
            model="user-selected-model",
            api_key_env="",
            base_url="https://provider.test/v1",
            **changes,
        ),
    )


async def discard(delta):
    pass


async def test_chat_nonstream_tool_calls_and_usage():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "user-selected-model"
        assert body["max_completion_tokens"] == 256
        assert body["temperature"] == 0.2
        assert "max_tokens" not in body
        return httpx.Response(
            200,
            json={
                "id": "response-id",
                "choices": [
                    {
                        "message": {
                            "content": "Let's compute",
                            "tool_calls": [
                                {
                                    "id": "action",
                                    "type": "function",
                                    "function": {"name": "python", "arguments": '{"code":"1+1"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    provider = ChatProvider(httpx.MockTransport(handler))
    result = await provider.invoke(
        request_config(
            streaming=False,
            max_output_tokens=256,
            parameters={"temperature": 0.2, "max_tokens": 10000},
            input_cost_per_million=1,
            output_cost_per_million=2,
        ),
        discard,
    )
    assert result.actions[0].name == "python"
    assert result.actions[0].arguments == {"code": "1+1"}
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20
    assert result.usage.cost == pytest.approx(0.00014)
    assert result.usage_reported and result.provider_id == "response-id"


async def test_streaming_fragments_actions_and_final_usage():
    chunks = [
        {"id": "stream", "choices": [{"delta": {"content": "Hello "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "content": "there",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a1",
                                "function": {"name": "python", "arguments": '{"co'},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'de":"2+2"}'}}]}}
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 12}},
    ]
    stream = "\n\n".join("data: " + json.dumps(c) for c in chunks) + "\n\ndata: [DONE]\n\n"
    provider = ChatProvider(httpx.MockTransport(lambda r: httpx.Response(200, text=stream)))
    emitted = []

    async def emit(delta):
        emitted.append(delta)

    response = await provider.invoke(request_config(), emit)
    assert response.text == "Hello there" and emitted == ["Hello ", "there"]
    assert response.actions[0].arguments == {"code": "2+2"}
    assert response.usage.output_tokens == 12


@pytest.mark.parametrize("status,retryable", [(401, False), (429, True), (500, True)])
async def test_http_errors_classified(status, retryable):
    provider = ChatProvider(
        httpx.MockTransport(lambda r: httpx.Response(status, json={"error": "failed"}))
    )
    with pytest.raises(HarnessError) as error:
        await provider.invoke(request_config(), discard)
    assert error.value.failure.category == "provider"
    assert error.value.failure.retryable == retryable


async def test_truncated_stream_is_retryable_provider_failure():
    provider = ChatProvider(
        httpx.MockTransport(lambda r: httpx.Response(200, text='data: {"choices":[]}\n\n'))
    )
    with pytest.raises(HarnessError) as error:
        await provider.invoke(request_config(), discard)
    assert error.value.failure.code == "incomplete_stream"
    assert error.value.failure.uncertain


async def test_malformed_tool_arguments_classified_as_model_failure():
    provider = ChatProvider(
        httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "a",
                                        "function": {"name": "python", "arguments": "invalid"},
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        )
    )
    with pytest.raises(HarnessError) as error:
        await provider.invoke(request_config(streaming=False), discard)
    assert error.value.failure.category == "model"


async def test_provider_timeout_and_cancellation():
    async def timeout(request):
        raise httpx.ReadTimeout("Slow service")

    provider = ChatProvider(httpx.MockTransport(timeout))
    with pytest.raises(HarnessError) as error:
        await provider.invoke(request_config(), discard)
    assert error.value.failure.category == "provider" and error.value.failure.retryable
    cancelled = asyncio.Event()

    async def slow(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    provider = ChatProvider(httpx.MockTransport(slow))
    task = asyncio.create_task(provider.invoke(request_config(), discard))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
