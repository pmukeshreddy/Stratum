"""Regressions for request occupancy, retained work and useful auxiliary admission."""

import asyncio
import json
from contextlib import closing

import pytest

from threadweave.context import Context
from threadweave.models import ModelRequest, ModelResponse, Usage, Workspace, now
from threadweave.runtime import Runtime
from threadweave.storage import Store, encode

from .fakes import ScriptedProvider


def add_block(store, sid, text):
    event = store.event(sid, "observation", {})
    store.add_context(sid, event, [{"role": "user", "content": text}])
    return event


def test_subscription_estimate_uses_native_representation_once(tmp_path, config):
    config.provider.name = "codex_subscription"
    config.provider.model = "gpt-6-astra"
    config.context.max_tokens = 500000
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Inspect evidence", Workspace(path=str(tmp_path)), config)
        event = store.event(root.id, "model_response", {})
        item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Native evidence."}],
        }
        store.db.execute(
            "INSERT INTO provider_continuations VALUES(?,?,?,?)",
            (event, "codex_subscription", "gpt-6-astra", encode([item])),
        )
        store.add_context(
            root.id,
            event,
            [
                {
                    "role": "assistant",
                    "content": "visible duplicate " * 2000,
                    "provider_response_event": event,
                }
            ],
        )
        context = Context(store)
        _, before = context.assemble(root.id, [])
        blocks = store.session(root.id).context
        blocks[0]["messages"][0]["content"] = "short duplicate"
        store.update(root.id, context=blocks)
        _, after = context.assemble(root.id, [])
        assert before == after


def test_dynamic_status_does_not_rewrite_instruction_prefix(tmp_path, config):
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Inspect evidence", Workspace(path=str(tmp_path)), config)
        context = Context(store)
        before = context.messages(root.id)
        store.charge(root.id, Usage(turns=1, input_tokens=120, output_tokens=30))
        store.update(root.id, turns=1)
        after = context.messages(root.id)
        assert before[:2] == after[:2]
        assert before[-1] != after[-1]
        assert after[-1]["role"] == "developer"
        assert "resources_remaining" in after[-1]["content"]


async def test_compaction_request_has_budget_and_retained_context_contract(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.summary_tokens = 1000
    config.context.recent_blocks = 1
    provider = ScriptedProvider(
        {"root": [ModelResponse(text=encode({"unresolved_requirements": ["Verify checksum."]}))]}
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Keep pending work", tmp_path, config=config)
    try:
        add_block(runtime.store, root.id, "Old evidence; still must verify checksum.")
        add_block(runtime.store, root.id, "RECENT_HOST_RETAINED_CONTEXT")
        await runtime.semantic_compact(root.id, force=True)
        request = provider.requests[0]
        assert "1000" in request.messages[0]["content"]
        assert "Do not copy" in request.messages[0]["content"]
        evidence = json.loads(request.messages[-1]["content"])
        assert evidence["output_contract"]["visible_summary_tokens"] == 1000
        assert "RECENT_HOST_RETAINED_CONTEXT" in encode(evidence)
        assert "RECENT_HOST_RETAINED_CONTEXT" not in runtime.store.session(root.id).summary
    finally:
        await runtime.shutdown()


def test_compaction_carries_previous_pending_items_when_model_omits_them(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.summary_tokens = 2000
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Complete work", Workspace(path=str(tmp_path)), config)
        pending = ["Verify the checksum.", "Validate the backup."]
        store.update(
            root.id,
            summary=encode(
                {
                    "unresolved_requirements": pending,
                    "active_hypotheses": ["Header bytes may be missing."],
                }
            ),
        )
        add_block(store, root.id, "Some completed work.")
        Context(store).compact(
            root.id, summary={"completed_work": ["Inspected files."], "unresolved_requirements": []}
        )
        summary = store.session(root.id).summary
        assert all(item in summary for item in pending)
        assert summary.index(pending[0]) < summary.index(pending[1])
        assert "Header bytes may be missing." in summary


def test_compaction_reference_cannot_embed_retained_conversation(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.summary_tokens = 4000
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Retain evidence", Workspace(path=str(tmp_path)), config)
        add_block(store, root.id, "Original evidence.")
        copied = {
            "recent_context": [
                {"messages": [{"role": "tool", "content": "REDUNDANT_COPIED_BODY" * 20}]}
            ]
        }
        Context(store).compact(
            root.id,
            summary={
                "unresolved_requirements": ["Verify evidence."],
                "important_references": [copied],
            },
        )
        summary = store.session(root.id).summary
        assert "REDUNDANT_COPIED_BODY" not in summary
        assert "artifacts.load" in summary


async def test_periodic_checkpoint_reviews_routine_work_before_deciding_to_refine(tmp_path, config):
    config.refinement.automatic = True
    config.refinement.every_turns = 20
    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    text='{"shouldRefine": false, "rationale": "Routine work has no reusable lesson"}'
                )
            ]
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Inspect evidence", tmp_path, config=config)
    try:
        runtime.store.update(root.id, turns=20)
        runtime.store.event(
            root.id, "python_result", {"stdout": "Routine observation: 12", "error": None}
        )
        await runtime.auto_refine(root.id)
        assert len(provider.requests) == 1
        assert provider.requests[0].metadata["purpose"] == "refinement_review"
        assert runtime.store.states(root.id) == []
    finally:
        await runtime.shutdown()


async def test_automatic_refinement_preserves_time_for_next_agent_action(tmp_path, config):
    config.refinement.automatic = True
    provider = ScriptedProvider({"root": [ModelResponse(text='{"proposals":[]}')]})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Inspect evidence", tmp_path, config=config)
    try:
        runtime.store.update(root.id, started_at=now() - 29, runnable=True)
        runtime.store.event(
            root.id, "experiment_conclusion", {"conclusion": "New reusable evidence."}
        )
        await runtime.auto_refine(root.id)
        assert provider.requests == []
        assert runtime.store.session(root.id).runnable
        assert runtime.store.events(root.id, kind="auxiliary_deferred")
    finally:
        await runtime.shutdown()


def test_reported_occupancy_anchor_restores_and_preserves_opaque_state(tmp_path, config):
    from threadweave.request_context import record_usage
    from threadweave.subscription import responses_input

    config.provider.name = "codex_subscription"
    config.provider.model = "gpt-6-astra"
    state = tmp_path / "state"
    with closing(Store(state)) as store:
        root = store.create("Continue work", Workspace(path=str(tmp_path)), config)
        request = ModelRequest(
            session_id=root.id,
            root_id=root.id,
            parent_id=None,
            name="root",
            turn=1,
            messages=Context(store).messages(root.id),
            tools=[],
            config=config.provider,
            input_token_bound=2000,
        )
        opaque = {
            "type": "reasoning",
            "id": "rs-native",
            "encrypted_content": "opaque-123-" * 12000,
            "summary": [],
        }
        native = {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Evidence retained."}],
        }
        response = ModelResponse(
            text="Evidence retained.",
            provider_items=[opaque, native],
            usage=Usage(input_tokens=1000, output_tokens=500),
        )
        event = store.event(root.id, "model_response", {})
        store.db.execute(
            "INSERT INTO provider_continuations VALUES(?,?,?,?)",
            (event, config.provider.name, config.provider.model, encode(response.provider_items)),
        )
        record_usage(store, root.id, request, response, event)
        store.add_context(
            root.id,
            event,
            [{"role": "assistant", "content": response.text, "provider_response_event": event}],
        )
        add_block(store, root.id, "New observation: verify the header.")
        size, method = Context(store).request_estimate(
            root.id, Context(store).messages(root.id), []
        )
        assert method["method"] == "reported_usage_plus_appended_input"
        assert 1500 < size < 5000
        assert method["projection_tokens"] > 15000
        assert "opaque-123-" not in encode(store.events(root.id))
    with closing(Store(state)) as store:
        context = Context(store)
        restored, method = context.request_estimate(root.id, context.messages(root.id), [])
        assert restored == size and method["method"] == "reported_usage_plus_appended_input"
        _, items = responses_input(
            context.messages(root.id), provider=[config.provider.name, config.provider.model]
        )
        offset = items.index(opaque)
        assert items[offset : offset + 2] == [opaque, native]
        # Any rewritten prefix or changed tool contract invalidates the anchor.
        messages = context.messages(root.id)
        messages[1]["content"] += " Changed objective."
        assert context.request_estimate(root.id, messages, [])[1]["method"] == "provider_projection"
        changed = config.provider.model_copy(update={"parameters": {"reasoning_effort": "xhigh"}})
        assert (
            context.request_estimate(root.id, context.messages(root.id), [], changed)[1]["method"]
            == "provider_projection"
        )
        tools = [
            {"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}
        ]
        assert (
            context.request_estimate(root.id, context.messages(root.id), tools)[1]["method"]
            == "provider_projection"
        )


@pytest.mark.parametrize("supported", [False, True])
def test_pending_resolution_requires_supplied_evidence(tmp_path, config, supported):
    from threadweave.context_budget import pending_ledger

    config.provider.model = "gpt-6-astra"
    config.context.summary_tokens = 1500
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Validate release", Workspace(path=str(tmp_path)), config)
        previous = {"unresolved_requirements": ["Verify checksum.", "Validate backup."]}
        store.update(root.id, summary=encode(previous))
        ledger = pending_ledger(previous)
        event = add_block(store, root.id, "Checksum verified successfully; backup still pending.")
        Context(store).compact(
            root.id,
            summary={
                "unresolved_requirements": [],
                "resolved_items": [
                    {
                        "id": ledger[0]["id"],
                        "reason": "Checksum verification succeeded",
                        "source_events": [event if supported else "unsupplied-evidence"],
                    }
                ],
            },
        )
        final = json.loads(store.session(root.id).summary)
        assert ("Verify checksum." in final["unresolved_requirements"]) is not supported
        assert "Validate backup." in final["unresolved_requirements"]
        assert pending_ledger(final)[-1]["id"] == ledger[-1]["id"]


async def test_interrupted_compaction_does_not_commit_partial_state(tmp_path, config):
    class Interrupted:
        async def invoke(self, request, emit):
            raise asyncio.CancelledError()

    config.provider.model = "gpt-6-astra"
    config.context.recent_blocks = 1
    runtime = Runtime(tmp_path / "state", providers={"mock": Interrupted()})
    root = runtime.create("Keep pending work", tmp_path, config=config)
    try:
        runtime.store.update(
            root.id, summary=encode({"unresolved_requirements": ["Verify checksum."]})
        )
        add_block(runtime.store, root.id, "Old evidence.")
        add_block(runtime.store, root.id, "Recent verbatim evidence.")
        before = runtime.store.session(root.id)
        with pytest.raises(asyncio.CancelledError):
            await runtime.semantic_compact(root.id, force=True)
        after = runtime.store.session(root.id)
        assert after.summary == before.summary and after.context == before.context
        assert not runtime.store.events(root.id, kind="context_compaction")
        assert runtime.store.usage(root.id).model_calls == 1
        assert runtime.store.usage(root.id).estimated_calls == 1
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("input_size,needs_compaction", [(9000, False), (11000, True)])
async def test_late_compaction_keeps_fitting_evidence_but_enforces_capacity(
    tmp_path, config, monkeypatch, input_size, needs_compaction
):
    config.context.max_tokens = 10000
    config.context.recent_blocks = 1
    config.limits.wall_seconds = 300
    config.refinement.automatic = False
    responses = [ModelResponse(text="Verified conclusion.")]
    if needs_compaction:
        responses.insert(
            0,
            ModelResponse(
                text=encode(
                    {
                        "unresolved_requirements": ["Retain the constraint."],
                        "established_facts": ["Old evidence verified."],
                    }
                )
            ),
        )
    provider = ScriptedProvider({"root": responses})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("Retain the constraint.", tmp_path, config=config)
        add_block(runtime.store, root.id, "OLD_FULL_EVIDENCE")
        add_block(runtime.store, root.id, "RECENT_FULL_EVIDENCE")
        runtime.store.update(root.id, started_at=now())
        monkeypatch.setattr(runtime, "_elapsed", lambda sid: 200)
        monkeypatch.setattr(
            runtime.context,
            "request_estimate",
            lambda sid, *args, **kwargs: (
                1000 if runtime.store.session(sid).summary else input_size,
                {},
            ),
        )
        await runtime._invoke(root.id)
        assert bool(runtime.store.events(root.id, kind="context_compaction")) is needs_compaction
        assert (
            bool(runtime.store.events(root.id, kind="compaction_deferred")) is not needs_compaction
        )
        assert len(provider.requests) == (2 if needs_compaction else 1)
        assert "RECENT_FULL_EVIDENCE" in encode(provider.requests[-1].messages)
        if not needs_compaction:
            assert "OLD_FULL_EVIDENCE" in encode(provider.requests[-1].messages)
        assert provider.requests[-1].input_token_bound < 10000 - config.provider.max_output_tokens
    finally:
        await runtime.shutdown()


def test_usage_anchor_counts_messages_inserted_before_committed_response(tmp_path, config):
    from threadweave.request_context import record_usage

    config.provider.name = "codex_subscription"
    config.provider.model = "gpt-6-astra"
    with closing(Store(tmp_path / "state")) as store:
        root = store.create("Inspect all evidence", Workspace(path=str(tmp_path)), config)
        context = Context(store)
        request = ModelRequest(
            session_id=root.id,
            root_id=root.id,
            parent_id=None,
            name="root",
            turn=0,
            messages=context.messages(root.id),
            tools=[],
            config=config.provider,
            input_token_bound=2000,
        )
        opaque = {
            "type": "reasoning",
            "id": "private",
            "encrypted_content": "opaque-123-" * 12000,
            "summary": [],
        }
        response = ModelResponse(
            text="Candidate checked.",
            provider_items=[
                opaque,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Candidate checked."}],
                },
            ],
            usage=Usage(input_tokens=1000, output_tokens=500),
        )
        event = store.event(root.id, "model_response", {})
        store.db.execute(
            "INSERT INTO provider_continuations VALUES(?,?,?,?)",
            (event, config.provider.name, config.provider.model, encode(response.provider_items)),
        )
        record_usage(store, root.id, request, response, event)
        add_block(store, root.id, "Child found a boundary failure while the action was executing.")
        store.add_context(
            root.id,
            event,
            [{"role": "assistant", "content": response.text, "provider_response_event": event}],
        )
        add_block(store, root.id, "Action result: verified input.")
        messages = context.messages(root.id)
        size, method = context.request_estimate(root.id, messages, [])
        assert method["method"] == "reported_usage_plus_appended_input"
        assert 1500 < size < 5000 and method["projection_tokens"] > 15000
        # Additional inserted evidence is charged, even before the old response.
        messages.insert(-2, {"role": "user", "content": "Additional child evidence. " * 100})
        larger, method = context.request_estimate(root.id, messages, [])
        assert larger > size and method["method"] == "reported_usage_plus_appended_input"
        # Rewriting/removing an old item invalidates the reported occupancy.
        messages[1]["content"] = "Different task"
        assert context.request_estimate(root.id, messages, [])[1]["method"] == "provider_projection"
