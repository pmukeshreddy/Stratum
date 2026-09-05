import asyncio
import json

import pytest

from threadweave.codex_auth import CodexControl
from threadweave.models import HarnessError, ModelRequest, ProviderConfig, RunConfig, Usage
from threadweave.storage import Store
from threadweave.subscription import SubscriptionProvider, responses_input


class AccountControl(CodexControl):
    """Test-only account protocol peer; never used by normal configuration."""

    def __init__(self, account=None):
        super().__init__()
        self.account = account
        self.calls = []

    async def call(self, method, params):
        self.calls.append((method, params))
        if method == "account/read":
            return {"account": self.account}
        if method == "account/logout":
            self.account = None
            return {}
        if method == "account/login/start":
            self.account = {"type": "chatgpt", "planType": "pro"}
            await self.notifications.put(
                {"method": "account/login/completed", "params": {"success": True}}
            )
            return {"authUrl": "https://auth.example.test/authorize", "loginId": "login"}
        raise AssertionError(method)


async def test_auth_state_redaction_existing_login_and_logout():
    peer = AccountControl(
        {
            "type": "chatgpt",
            "planType": "pro",
            "email": "private@example.test",
            "accessToken": "SECRET",
            "refreshToken": "SECRET",
        }
    )
    status = await peer.status()
    assert status["logged_in"] and status["authentication"] == "ChatGPT"
    assert "SECRET" not in json.dumps(status) and "private@" not in json.dumps(status)
    assert peer.calls[-1][1] == {"refreshToken": False}
    assert (await peer.login(lambda _: pytest.fail("Must reuse existing login")))["logged_in"]
    assert not (await peer.logout())["logged_in"]
    assert not (await AccountControl({"type": "apiKey"}).status())["logged_in"]


async def test_official_login_request_and_completion():
    peer, presented = AccountControl(), []
    assert not (await peer.status())["logged_in"]
    assert (await peer.login(presented.append))["logged_in"]
    assert presented and ("account/login/start", {"type": "chatgpt"}) in peer.calls
    await peer.logout()
    assert (await peer.login(presented.append, device=True))["logged_in"]
    assert ("account/login/start", {"type": "chatgptDeviceCode"}) in peer.calls


def model_request():
    return ModelRequest(
        session_id="root",
        root_id="root",
        parent_id=None,
        name="root",
        turn=0,
        messages=[{"role": "user", "content": "Search"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "repo_search", "parameters": {"type": "object"}},
            }
        ],
        input_token_bound=200,
        config=ProviderConfig(model="catalog-selected", parameters={"reasoning_effort": "low"}),
    )


async def stream(events):
    for event in events:
        yield (json.dumps(event) + "\n").encode()


async def discard(_):
    pass


async def test_stream_text_parallel_tools_usage_and_refresh_metadata():
    request, emitted = model_request(), []

    async def emit(text):
        emitted.append(text)

    events = [
        {"type": "text_delta", "text": "Inspecting "},
        {"type": "text_delta", "text": "code."},
        {"type": "auth_refresh", "succeeded": True},
        {"type": "reasoning_summary", "text": "Search the definitions."},
        {"type": "rate_limits", "limits": {"primary": {"used_percent": 20}}},
    ]
    for i in range(2):
        events.extend(
            [
                {"type": "tool_delta", "delta": '{"pattern":'},
                {"type": "tool_delta", "delta": '"symbol"}'},
                {
                    "type": "item",
                    "item": {
                        "type": "function_call",
                        "name": "repo_search",
                        "call_id": f"c{i}",
                        "arguments": '{"pattern":"symbol"}',
                    },
                },
            ]
        )
    events.append(
        {
            "type": "completed",
            "id": "response",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 30,
                "cached_input_tokens": 60,
                "reasoning_output_tokens": 10,
            },
        }
    )
    response = await SubscriptionProvider.collect(stream(events), request, request.config, emit)
    assert response.text == "Inspecting code." and len(response.actions) == 2
    assert "".join(emitted) == response.text
    assert response.usage.cost is None and response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 60 and response.usage.reasoning_output_tokens == 10
    assert response.metadata["auth_refreshes"] == 1 and response.metadata["rate_limits"]


@pytest.mark.parametrize(
    "code,retry",
    [
        ("AUTH_REQUIRED", False),
        ("http_429", True),
        ("usage_limit", False),
        ("transport_failure", True),
    ],
)
async def test_provider_errors(code, retry):
    request = model_request()
    with pytest.raises(HarnessError) as caught:
        await SubscriptionProvider.collect(
            stream(
                [
                    {
                        "type": "error",
                        "code": code,
                        "retryable": retry,
                        "body": "SECRET",
                        "headers": {"Authorization": "SECRET"},
                    }
                ]
            ),
            request,
            request.config,
            discard,
        )
    assert caught.value.failure.code == code and caught.value.failure.retryable == retry
    assert "SECRET" not in str(caught.value)


async def test_incomplete_malformed_and_unregistered_calls():
    request = model_request()
    with pytest.raises(HarnessError, match="ended before"):
        await SubscriptionProvider.collect(stream([]), request, request.config, discard)
    for arguments, name in [
        ("not-json", "repo_search"),
        ("[]", "repo_search"),
        ("{}", "exec_command"),
    ]:
        with pytest.raises(HarnessError, match="Invalid structured"):
            await SubscriptionProvider.collect(
                stream(
                    [
                        {
                            "type": "item",
                            "item": {
                                "type": "function_call",
                                "name": name,
                                "call_id": "x",
                                "arguments": arguments,
                            },
                        }
                    ]
                ),
                request,
                request.config,
                discard,
            )


async def test_output_guard_and_unreported_usage():
    request = model_request()
    with pytest.raises(HarnessError, match="byte limit"):
        await SubscriptionProvider.collect(
            stream(
                [{"type": "text_delta", "text": "x" * (request.config.max_output_tokens * 4 + 1)}]
            ),
            request,
            request.config,
            discard,
        )
    result = await SubscriptionProvider.collect(
        stream([{"type": "completed"}]), request, request.config, discard
    )
    assert not result.usage_reported and result.usage.cost is None


def test_context_uses_structured_function_inputs_and_outputs():
    instructions, items = responses_input(
        [
            {"role": "system", "content": "Threadweave policy"},
            {
                "role": "assistant",
                "content": "Search",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "repo_search", "arguments": '{"pattern":"abc"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"matches":[]}'},
        ]
    )
    assert instructions == "Threadweave policy"
    assert items[1]["type"] == "function_call"
    assert items[2]["type"] == "function_call_output" and items[2]["call_id"] == "c1"


def test_defaults_and_subscription_prices_are_not_api_prices(tmp_path):
    assert ProviderConfig().name == "codex_subscription"
    assert ProviderConfig().api_key_env == ""
    assert ProviderConfig(name="chat").api_key_env == "OPENAI_API_KEY"
    for field in (
        {"api_key_env": "SECRET"},
        {"input_cost_per_million": 1},
        {"base_url": "https://api.example.test"},
    ):
        with pytest.raises(ValueError, match="managed ChatGPT"):
            ProviderConfig(**field)
    with pytest.raises(ValueError, match="cost budgets"):
        RunConfig(limits={"cost_budget": 1})
    store = Store(tmp_path / "store")
    try:
        from threadweave.models import Workspace

        root = store.create("root", Workspace(path=str(tmp_path)), RunConfig(), name="root")
        store.charge(root.id, Usage(input_tokens=3, cost=None))
        assert store.usage(root.id, tree=True).cost is None
        assert store.usage(root.id, tree=True).input_tokens == 3
        old = root.config_id
        config = store.config(root.id)
        selected = config.provider.model_copy(update={"model": "resolved-account-model"})
        store.pin_provider(root.id, config.provider, selected)
        assert store.config(root.id).provider.model == "resolved-account-model"
        assert store.config(old).provider.model == ""
        assert store.events(root.id, kind="provider_config_resolved")
    finally:
        store.close()


async def test_cancellation_terminates_client_process(tmp_path, monkeypatch):
    executable = tmp_path / "client"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o700)
    provider = SubscriptionProvider(executable=executable)

    async def resolve(config):
        return config, {}

    monkeypatch.setattr(provider, "resolve", resolve)
    task = asyncio.create_task(provider.invoke(model_request(), discard))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)


async def test_account_model_resolution_and_unsupported_parameters():
    class Catalog(AccountControl):
        async def __aenter__(self):
            self.account = {"type": "chatgpt", "planType": "pro"}
            return self

        async def __aexit__(self, *_):
            pass

        async def settings(self):
            return {"model": "current-account-model", "reasoning_effort": "medium"}

        async def models(self):
            return [
                {
                    "model": "current-account-model",
                    "isDefault": True,
                    "defaultReasoningEffort": "low",
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": "low"},
                        {"reasoningEffort": "medium"},
                    ],
                }
            ]

    provider = SubscriptionProvider(control_factory=Catalog)
    config, _ = await provider.resolve(ProviderConfig())
    assert (
        config.model == "current-account-model"
        and config.parameters["reasoning_effort"] == "medium"
    )
    for config, code in [
        (ProviderConfig(model="unavailable"), "MODEL_UNAVAILABLE"),
        (ProviderConfig(parameters={"temperature": 0}), "UNSUPPORTED_PARAMETER"),
        (ProviderConfig(parameters={"reasoning_effort": "unsupported"}), "UNSUPPORTED_REASONING"),
    ]:
        with pytest.raises(HarnessError) as caught:
            await provider.resolve(config)
        assert caught.value.failure.code == code


def test_running_provider_pins_client_across_source_upgrade(tmp_path, monkeypatch):
    before, after = tmp_path / "before", tmp_path / "after"
    monkeypatch.setattr("threadweave.subscription.client_path", lambda: before)
    provider = SubscriptionProvider()
    monkeypatch.setattr("threadweave.subscription.client_path", lambda: after)
    assert provider.executable == before
    assert SubscriptionProvider().executable == after
