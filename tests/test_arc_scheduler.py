"""Scheduler and native transport regressions, with no capability evaluation/inference."""

import asyncio
import json
from contextlib import asynccontextmanager

from threadweave.evals.arc_scheduler import infrastructure_failure
from threadweave.evals.harness import MatchedProvider
from threadweave.evals.inference_gate import InferenceGate
from threadweave.evals.schema import NotRun
from threadweave.models import ModelResponse, Usage
from threadweave.runtime import Runtime

from .fakes import ScriptedProvider


async def test_global_gate_covers_root_descendant_and_auxiliary_calls(tmp_path, config):
    config.limits.max_subagents = 4
    config.limits.concurrency = 4
    gate = InferenceGate(tmp_path / "gate", 2)
    provider = ScriptedProvider(
        {"*": [ModelResponse(text="result", usage=Usage(input_tokens=20, output_tokens=10))]},
        delay=0.02,
    )
    matched = MatchedProvider(
        provider, config.provider, tmp_path / "requests.jsonl", gate=gate, owner="game"
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": matched})
    root = runtime.create("root", tmp_path, config=config)
    try:
        children = [runtime.spawn(root.id, "independent work", isolate=False) for _ in range(4)]
        await asyncio.gather(
            runtime._invoke(root.id),
            *[runtime._invoke(c.id) for c in children],
            runtime.auxiliary(root.id, "compaction", "Summarize", {"fact": "retained"}),
            runtime.auxiliary(root.id, "refinement", "Refine", {"fact": "retained"}),
        )
        assert gate.peak == provider.peak_active == 2 and gate.active == 0
        assert runtime.store.usage(root.id, tree=True).model_calls == 7
        assert sum(r.parent_id is not None for r in provider.requests) == 4
    finally:
        await runtime.shutdown()
        await gate.close()


async def test_gate_maximum_16_and_throttle_step_down(tmp_path):
    gate = InferenceGate(tmp_path, 16)

    async def invoke(i):
        async with gate.permit(str(i)):
            await asyncio.sleep(0.01)

    await asyncio.gather(*(invoke(i) for i in range(64)))
    assert gate.peak == 16 and gate.active == 0
    await gate.outcome(unstable=True)
    assert gate.capacity == 15
    await gate.outcome(unstable=True)
    assert gate.capacity == 15  # One transport burst is not sixteen capacity decisions.


async def test_native_proxy_preserves_payload_usage_and_releases_admission(tmp_path, config):
    import aiohttp
    from aiohttp import web

    config.provider.model = "gpt-6-astra"
    config.provider.parameters = {"reasoning_effort": "xhigh"}
    received = []

    class Response:
        status = 200
        headers = {"Content-Type": "text/event-stream"}

        def __init__(self):
            self.content = self

        async def iter_any(self):
            yield b'data: {"type":"response.completed","response":{"id":"resp-test","usage":{"input_tokens":123,"output_tokens":45}}}\n\n'

    gate = await InferenceGate(tmp_path, 2).start()
    try:
        url = gate.register("native", config, tmp_path)

        @asynccontextmanager
        async def upstream(method, url, *, data, headers):
            received.append(data)
            yield Response()

        gate.client.request = upstream
        payload = {
            "model": "gpt-6-astra",
            "reasoning": {"effort": "xhigh"},
            "input": [{"role": "user", "content": "protocol test"}],
        }
        async with aiohttp.ClientSession() as client:
            async with client.post(url + "/responses", data=json.dumps(payload)) as response:
                assert response.status == web.HTTPOk.status_code
                assert "response.completed" in await response.text()
        assert json.loads(received[0]) == payload
        assert gate.usage("native")["total_tokens"] == 168
        assert gate.usage("native")["model_calls"] == 1
        assert gate.games["native"]["reserved_tokens"] == gate.active == 0
    finally:
        await gate.close()


def test_only_infrastructure_failures_can_retry():
    for message in (
        "http_429",
        "http_503",
        "transport_failure",
        "Official worker exited during action",
        "ConnectionResetError",
    ):
        assert infrastructure_failure(NotRun(message))
    for message in (
        "score is zero",
        "Initial observation differs",
        "invalid_request",
        "evaluation_settings_changed",
        "incorrect game strategy",
    ):
        assert not infrastructure_failure(NotRun(message))
