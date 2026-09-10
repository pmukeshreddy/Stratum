"""One end-to-end regression for the evaluation wrapper's auxiliary contract."""

import json

import pytest

from threadweave.evals.harness import MatchedProvider, discard
from threadweave.models import Action, HarnessError, ModelRequest, ModelResponse, RunConfig
from threadweave.runtime import Runtime


async def test_refinement_reservations_notice_and_root_continuation(tmp_path):
    class Provider:
        def __init__(self):
            self.requests = []

        async def resolve(self, config, *, reasoning_off=False):
            if reasoning_off:
                config = config.model_copy(update={"parameters": {"reasoning_effort": "low"}})
            return config, {}

        async def invoke(self, request, emit):
            self.requests.append(request)
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement_review":
                return ModelResponse(text='{"shouldRefine": false, "rationale": "no new evidence"}')
            if purpose == "refinement":
                assert request.config.max_output_tokens == 32000
                assert request.config.parameters["reasoning_effort"] == "low"
                return ModelResponse(
                    text=json.dumps(
                        {
                            "summary": "Keep validated procedure",
                            "edits": [
                                {
                                    "action": "create",
                                    "kind": "memory",
                                    "id": "validated",
                                    "title": "Validated procedure",
                                    "content": "Compare results against an independent subtotal.",
                                }
                            ],
                        }
                    )
                )
            if request.turn == 0:
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "assert 11 + 13 == 24\nprint(await refine.run('Retain the checked subtotal procedure'))"
                            },
                        )
                    ]
                )
            assert "[self-refinement]" in str(request.messages)
            assert "Compare results against an independent subtotal." in str(request.messages)
            assert request.config.max_output_tokens == 32768
            assert request.config.parameters["reasoning_effort"] == "xhigh"
            return ModelResponse(text="24, checked")

    config = RunConfig(
        provider={
            "name": "codex_subscription",
            "model": "test-model",
            "max_output_tokens": 32768,
            "parameters": {"reasoning_effort": "xhigh"},
        },
        limits={"wall_seconds": 30},
    )
    provider = Provider()
    measured = MatchedProvider(provider, config.provider, tmp_path / "calls.jsonl")
    # Keep both stage identities pinned even when resolutions are interleaved.
    review = measured.refinement_expected["refinement_review"]
    resolved_review, _ = await measured.resolve(review, reasoning_off=True)
    assert resolved_review.max_output_tokens == 4096
    with pytest.raises(HarnessError):
        await measured.resolve(
            review.model_copy(update={"max_output_tokens": 4095}), reasoning_off=True
        )
    with pytest.raises(HarnessError):
        await measured.resolve(review.model_copy(update={"model": "other"}), reasoning_off=True)
    with pytest.raises(HarnessError):
        await measured.resolve(config.provider.model_copy(update={"max_output_tokens": 32000}))
    runtime = Runtime(tmp_path / "state", providers={"codex_subscription": measured})
    try:
        root = runtime.create(
            "Check the subtotal and retain the useful procedure", tmp_path, config=config
        )
        await runtime.start()
        await runtime.wait(root.id, timeout=25)
        assert runtime.store.session(root.id).outcome == "completed"
        assert [r.metadata.get("purpose", "agent") for r in provider.requests] == [
            "agent",
            "refinement",
            "agent",
        ]
        events = list(runtime.store.iter_events(root.id))
        scheduled = next(e for e in events if e["type"] == "refine_scheduled")
        complete = next(e for e in events if e["type"] == "refine_complete")
        notice = next(e for e in events if e["type"] == "refinement_notice")
        assert scheduled["seq"] < complete["seq"] < notice["seq"]
        assert complete["payload"]["appliedEdits"][0]["applied"]
        assert runtime.store.harness.get(root.id, "memory", "validated")
        assert not any(e["type"] == "refine_failed" for e in events)
        invocation = ModelRequest(
            session_id=root.id,
            root_id=root.id,
            parent_id=None,
            name="root",
            turn=2,
            messages=[],
            tools=[],
            config=resolved_review,
            input_token_bound=1,
            request_kind="auxiliary",
            reasoning_mode="off",
            metadata={"purpose": "refinement_review"},
        )
        await measured.invoke(invocation, discard)
        with pytest.raises(HarnessError):
            await measured.invoke(
                invocation.model_copy(update={"config": measured.refinement_configs["refinement"]}),
                discard,
            )
    finally:
        await runtime.shutdown()
