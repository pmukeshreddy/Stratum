"""Provider and historical request-graph contracts unaffected by harness replacement."""

import json

import httpx
import pytest

from threadweave.models import ModelResponse, ProviderConfig
from threadweave.providers import ChatProvider
from threadweave.runtime import Runtime
from threadweave.storage import encode
from threadweave.subscription import SubscriptionProvider

from .test_continual_harness import Refiner
from .test_runtime_contracts import seed


async def test_structured_refinement_omits_primary_thinking_options_on_wire(tmp_path, config):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        review = "shouldRefine" in body["messages"][0]["content"]
        content = (
            encode({"shouldRefine": True, "rationale": "useful"}) if review else '{"edits": []}'
        )
        if body["stream"]:
            return httpx.Response(
                200,
                text="data: "
                + json.dumps(
                    {
                        "choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                )
                + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    config.provider = ProviderConfig(
        name="chat",
        model="reasoning-model",
        base_url="https://example.test",
        api_key_env="",
        streaming=False,
        parameters={"reasoning_effort": "high"},
    )
    runtime = Runtime(
        tmp_path / "state", providers={"chat": ChatProvider(httpx.MockTransport(handle))}
    )
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        runtime.refinement_compacted(root.id)
        await runtime.refinement_checkpoint(root.id)
        assert len(bodies) == 2
        assert all("reasoning_effort" not in body for body in bodies)
        assert runtime.store.config(root.id).provider.parameters["reasoning_effort"] == "high"
        await runtime._invoke(root.id)
        assert bodies[-1]["reasoning_effort"] == "high"
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("supported", [["none", "low", "high"], ["low", "high"]])
async def test_subscription_refinement_omits_effort_instead_of_selecting_a_tier(supported):
    class Catalog:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def status(self):
            return {"logged_in": True}

        async def settings(self):
            return {"model": "m", "reasoning_effort": "high"}

        async def models(self):
            return [
                {
                    "model": "m",
                    "isDefault": True,
                    "defaultReasoningEffort": "high",
                    "supportedReasoningEfforts": [{"reasoningEffort": e} for e in supported],
                }
            ]

    provider = SubscriptionProvider(control_factory=Catalog)
    config = ProviderConfig(model="m", parameters={"reasoning_effort": "high"})
    omitted, _ = await provider.resolve(config, reasoning_off=True)
    assert "reasoning_effort" not in omitted.parameters
    unchanged, _ = await provider.resolve(config)
    assert unchanged.parameters["reasoning_effort"] == "high"


async def test_file_restart_preserves_auxiliary_isolation_and_usage(tmp_path, config):
    from .fakes import ScriptedProvider

    provider = ScriptedProvider(
        {"root": [ModelResponse(text="agent A"), ModelResponse(text="agent B")]}
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        await runtime._run_turn(root.id)
        review_provider = Refiner()
        runtime.providers["mock"] = review_provider
        seed(runtime, root.id)
        await runtime.refine(root.id)
        # Auxiliary requests remain outside the primary continuation chain.
        await runtime.auxiliary(
            root.id,
            "refinement_review",
            "review",
            runtime.refinement_input(root.id, review=True, reason="compact"),
        )
        runtime.providers["mock"] = provider
        runtime.store.update(root.id, runnable=True)
        await runtime._run_turn(root.id)
        a, b = [r.request_id for r in provider.requests]
        usage = runtime.store.usage(root.id)
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        graph = runtime.store.request_graph(root.id)
        assert {r["id"] for r in graph["requests"]} == {a, b}
        assert graph["edges"] == [{"source": a, "target": b, "kind": "continuation"}]
        assert next(r for r in graph["requests"] if r["id"] == b)["inbound"] == [
            {"source": a, "kind": "continuation"}
        ]
        assert len(runtime.store.request_history(root.id)) == 4
        assert all(
            not r["inbound"] for r in runtime.store.request_history(root.id, kind="auxiliary")
        )
        assert runtime.store.usage(root.id) == usage
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        assert runtime.store.request_graph(root.id) == graph
    finally:
        await runtime.shutdown()
