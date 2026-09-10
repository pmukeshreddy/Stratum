"""Scheduler and inference admission regressions, with no capability evaluation/inference."""

import asyncio

import pytest

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
        assert len(matched.primary_usages) == 1
        assert matched.primary_usages[0].model_calls == 1
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


async def test_output_budget_reserves_all_buffalo_descendant_calls(tmp_path, config):
    from threadweave.runtime import BudgetBusy, LimitReached

    config.limits.output_token_budget = 200
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Shared output budget", tmp_path, config=config)
    child = runtime.spawn(root.id, "Child", isolate=False)
    try:
        runtime.store.records.insert(
            "reservations",
            {
                "id": "call",
                "session_id": child.id,
                "input_tokens": 10,
                "output_tokens": 128,
                "cost": 0,
            },
        )
        with pytest.raises(BudgetBusy):
            runtime._check_limits(root.id, resource="model_calls")
        runtime.store.records.delete("reservations")
        runtime.store.charge(child.id, Usage(output_tokens=100))
        with pytest.raises(LimitReached, match="output tokens"):
            runtime._check_limits(root.id, resource="model_calls")
    finally:
        await runtime.shutdown()
