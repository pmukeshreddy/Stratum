"""Ports of Prime lifecycle regressions, using production paths and deterministic providers."""

import asyncio
import copy
import json

import pytest

from threadweave.context import Context
from threadweave.harness import apply_refinement_proposal, empty_harness_state, normalize_proposal
from threadweave.models import ModelResponse, RefinementPolicy
from threadweave.refinement import RefineSkippedError, parse_object
from threadweave.refinement_context import (
    convert_to_llm,
    refinement_messages,
    serialize_conversation,
)

from .test_continual_harness import edit, proposal
from .test_continual_harness import harness_runtime as harness_runtime
from .test_continual_harness_gaps import active_turn


@pytest.mark.parametrize(
    "value,interval,cooldown",
    [(None, 25, 1200), ("2", 25, 1200), (float("inf"), 25, 1200), (-1, 1, 0), (2.5, 2.5, 2.5)],
)
def test_settings_normalize_like_prime(value, interval, cooldown):
    policy = RefinementPolicy(turn_interval=value, cooldown_seconds=value)
    assert policy.turn_interval == interval and policy.cooldown_seconds == cooldown


@pytest.mark.parametrize(
    "text", ['{"edits": []}', '```json\n{"edits": []}\n```', 'Plan follows: {"edits": []} Done.']
)
def test_model_json_accepts_prime_wrappers(text):
    assert parse_object(text) == {"edits": []}


@pytest.mark.parametrize(
    "text", ['{"edits": [', '{"edits": [{"kind":"memory"}', 'Plan: {"edits":[{"kind":"memory"}']
)
def test_unreported_truncation_is_diagnosed(text):
    with pytest.raises(ValueError, match="output budget was exhausted"):
        parse_object(text)


def test_balanced_malformed_json_is_not_truncation():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        parse_object('{"edits": nope}')


@pytest.mark.parametrize("reason", ["length", "error"])
async def test_stop_reason_wins_over_valid_json(harness_runtime, reason):
    rt, sid, provider = harness_runtime

    async def invoke(request, emit):
        return ModelResponse(text='{"edits": []}', metadata={"stop_reason": reason})

    provider.invoke = invoke
    with pytest.raises(ValueError):
        await rt.plan_refinement(sid, {})
    assert not rt.store.harness.history(sid)


def test_conversion_and_serializer_match_prime_categories():
    messages = [
        {"role": "system", "content": "SYSTEM_PRIVATE"},
        {"role": "developer", "content": "DEVELOPER_PRIVATE"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
                {"type": "toolCall", "name": "ipython", "arguments": {"code": "print(1)"}},
            ],
        },
        {"role": "toolResult", "content": [{"type": "text", "text": "x" * 2001}]},
        {"role": "custom", "customType": "refinement_outcome", "content": "OUTCOME_PRIVATE"},
        {
            "role": "custom",
            "customType": "refinement_notice",
            "content": "[self-refinement]\n\nlesson",
        },
        {"role": "compactionSummary", "summary": "summary", "harnessDigest": "digest"},
    ]
    text = serialize_conversation(messages)
    assert text.startswith(
        '[User]: question\n\n[Assistant thinking]: reason\n\n[Assistant]: answer\n\n[Assistant tool calls]: ipython(code="print(1)")'
    )
    assert "[... 1 more characters truncated]" in text and "x" * 2001 not in text
    assert "[User]: [self-refinement]" in text and "<harness_state>\ndigest" in text
    assert "<summary>\nsummary" in text
    assert "PRIVATE" not in text


@pytest.mark.parametrize("source", ["user", "self", "auto"])
def test_only_applied_edits_create_notice_and_outcome_is_not_llm_visible(source):
    result = apply_refinement_proposal(empty_harness_state(), proposal(edit()), id="r")
    raw = refinement_messages(result, source)
    assert [m["display"] for m in raw] == [True, False]
    converted = convert_to_llm(raw)
    assert len(converted) == 1 and converted[0]["role"] == "user"
    assert converted[0]["content"].startswith(f"[{source}-refinement]\n\n")
    assert "create memory [local:lesson] Lesson:" in converted[0]["content"]
    result["appliedEdits"] = []
    assert not convert_to_llm(refinement_messages(result, source))


async def test_global_omission_false_and_empty_instructions_are_distinct(harness_runtime):
    rt, sid, _ = harness_runtime
    config = rt.store.config(sid).model_copy(update={"serialized_refine": False})
    rt.store.reconfigure(sid, config)
    active_turn(rt, sid)
    rt.request_refinement(sid, instructions="first", global_=True)
    rt.request_refinement(sid)
    assert rt.refinement_state(sid).pending_request == {
        "source": "self",
        "instructions": "first",
        "global_": True,
    }
    rt.request_refinement(sid, instructions="", global_=False)
    assert rt.refinement_state(sid).pending_request == {
        "source": "self",
        "instructions": "",
        "global_": False,
    }
    with pytest.raises(TypeError):
        rt.request_refinement(sid, instructions=None)
    rt.invalidate_refinement(sid)


async def test_automatic_review_alone_is_not_status_in_flight(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(rt.maybe_auto_refine(sid, "compact"))
    await provider.entered.wait()
    assert rt.refinement_status(sid) == {"pending": False, "in_flight": False}
    provider.decision = False
    provider.release.set()
    await task


async def test_extension_plan_is_normalized_without_model_and_notice_keeps_system(
    harness_runtime, monkeypatch
):
    rt, sid, provider = harness_runtime
    seen = []

    async def hook(preparation):
        seen.append(preparation)
        return {"proposal": proposal(edit())}

    monkeypatch.setattr(rt.environment.adapter(sid), "session_before_refine", hook, raising=False)
    before = rt.context.system_prompt(sid)
    result = await rt.refine(sid, instructions="capture")
    assert result["appliedEdits"][0]["applied"] and not provider.requests
    assert seen[0]["trigger"] == "manual" and seen[0]["scope"] == "local"
    assert set(seen[0]) == {
        "trigger",
        "instructions",
        "scope",
        "planningState",
        "history",
        "conversationText",
    }
    assert rt.context.system_prompt(sid) == before
    assert "[user-refinement]" in json.dumps(rt.context.messages(sid))


async def test_extension_skip_is_not_an_automatic_failure(harness_runtime, monkeypatch):
    rt, sid, provider = harness_runtime
    monkeypatch.setattr(
        rt.environment.adapter(sid),
        "session_before_refine",
        lambda preparation: {"skip": True},
        raising=False,
    )
    with pytest.raises(RefineSkippedError):
        await rt.refine(sid)
    rt.refinement_compacted(sid)
    assert not await rt.refinement_checkpoint(sid)
    assert not rt.store.events(sid, kind="refine_failed")
    assert not rt.store.harness.history(sid)
    assert len(provider.requests) == 1


async def test_public_refine_calls_serialize_without_coalescing(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(rt.refine(sid))
    await provider.entered.wait()
    second = asyncio.create_task(rt.refine(sid))
    await asyncio.sleep(0)
    assert len(provider.requests) == 1
    provider.release.set()
    results = await asyncio.gather(first, second)
    assert len(results) == 2 and len(provider.requests) == 2
    assert results[0]["appliedEdits"][0]["applied"]
    assert not results[1]["appliedEdits"][0]["applied"]
    assert rt.refinement_state(sid).last_review_at == 0


async def test_public_apply_waits_for_compaction_and_blocks_turn_entry(harness_runtime):
    rt, sid, provider = harness_runtime
    rt._transitioning.add(sid)
    task = asyncio.create_task(rt.refine(sid))
    await asyncio.sleep(0.02)
    assert len(provider.requests) == 1 and not rt.store.harness.entries(sid)
    barrier = asyncio.create_task(rt.wait_refinement_barrier(sid))
    await asyncio.sleep(0)
    assert not barrier.done()
    rt._transitioning.discard(sid)
    await asyncio.gather(task, barrier)
    assert rt.store.harness.entries(sid)


async def test_abort_public_plan_that_ignores_cancel_cannot_write(harness_runtime):
    rt, sid, provider = harness_runtime
    entered, release = asyncio.Event(), asyncio.Event()
    original = provider.invoke

    async def invoke(request, emit):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return await original(request, emit)

    provider.invoke = invoke
    task = asyncio.create_task(rt.refine(sid))
    await entered.wait()
    rt.invalidate_refinement(sid)
    release.set()
    await asyncio.gather(task, return_exceptions=True)
    assert not rt.store.harness.history(sid)
    assert rt.refinement_state(sid).last_review_at == 0


async def test_disposal_drain_awaits_valid_background_plan_and_consumes_once(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await provider.entered.wait()
    drain = asyncio.create_task(rt.drain_refinement(sid))
    await asyncio.sleep(0)
    assert not drain.done()
    provider.release.set()
    await drain
    assert len(rt.store.harness.history(sid)) == 1
    await rt.drain_refinement(sid)
    assert len(rt.store.harness.history(sid)) == 1 and len(provider.requests) == 1
    rt.store.update(sid, pending_turn=None)


async def test_digest_is_cold_only_and_latest_compaction_digest_deduplicates(harness_runtime):
    rt, sid, _ = harness_runtime
    rt.context.ensure_harness_digest(sid)
    original = rt.context.messages(sid)
    rt.store.harness.mutate(sid, "create", "memory", title="Cold", content="COLD_ONLY", id="cold")
    assert "COLD_ONLY" not in json.dumps(rt.context.messages(sid))
    resumed = Context(rt.store)
    resumed.ensure_harness_digest(sid)
    assert "COLD_ONLY" in json.dumps(resumed.messages(sid))
    assert resumed.messages(sid)[0] == original[0]
    count = len(rt.store.events(sid, kind="harness_digest"))
    Context(rt.store).ensure_harness_digest(sid)
    assert len(rt.store.events(sid, kind="harness_digest")) == count
    rt.context.compact(sid, count=len(rt.store.session(sid).context), review_checkpoint=False)
    Context(rt.store).ensure_harness_digest(sid)
    assert len(rt.store.events(sid, kind="harness_digest")) == count


def test_partial_conflict_and_validation_failures_do_not_block_other_edits():
    state = empty_harness_state()
    apply_refinement_proposal(state, proposal(edit()), id="before")
    baseline = copy.deepcopy(state)
    state["entries"]["memory"]["lesson"]["content"] = "external"
    result = apply_refinement_proposal(
        state,
        proposal(edit(action="update"), edit("skill", reference={}), edit("prompt", id="other")),
        id="after",
        baseline_state=baseline,
    )
    assert [e["applied"] for e in result["appliedEdits"]] == [False, False, True]
    assert "before" not in result["appliedEdits"][1]
    assert state["entries"]["memory"]["lesson"]["content"] == "external"
    assert normalize_proposal(None)["edits"] == []
    assert len(normalize_proposal({"edits": [None, [], "invalid"]})["edits"]) == 1


async def test_audit_failure_preserves_outcome_but_does_not_emit_notice(
    harness_runtime, monkeypatch
):
    rt, sid, _ = harness_runtime

    def fail(*args):
        raise OSError("audit failed")

    monkeypatch.setattr(rt.store, "record_harness_refinement", fail)
    with pytest.raises(OSError, match="audit failed"):
        await rt.refine(sid)
    assert rt.store.harness.entries(sid)
    assert rt.store.events(sid, kind="refinement_outcome")
    assert not rt.store.events(sid, kind="refinement_notice")
    assert not rt.store.events(sid, kind="refine_complete")


async def test_unpersisted_outcome_survives_live_context_rebuild(harness_runtime, monkeypatch):
    rt, sid, _ = harness_runtime
    original = rt.store.add_context

    def fail_outcome(session, event, messages):
        if messages[0].get("customType") == "refinement_outcome":
            raise OSError("outcome failed")
        return original(session, event, messages)

    monkeypatch.setattr(rt.store, "add_context", fail_outcome)
    await rt.refine(sid)
    assert rt.context.unpersisted_refinement_messages[sid][0]["customType"] == "refinement_outcome"
    assert "Refinement complete:" not in json.dumps(rt.context.messages(sid))
    assert "[user-refinement]" in json.dumps(rt.context.messages(sid))


@pytest.mark.parametrize(
    "kind,attempts",
    [("unknown", 4), ("auth", 2), ("invalid_request", 1), ("permission", 1), ("refusal", 1)],
)
async def test_provider_error_retry_policy_matches_prime(harness_runtime, kind, attempts):
    rt, sid, provider = harness_runtime
    calls = []

    async def invoke(request, emit):
        calls.append(request)
        return ModelResponse(
            text="",
            metadata={
                "stop_reason": "error",
                "error_message": "failed",
                "diagnostics": [{"type": "provider_stream_failure", "details": {"kind": kind}}],
            },
        )

    provider.invoke = invoke
    with pytest.raises(ValueError, match="Refinement failed"):
        await rt.plan_refinement(sid, {})
    assert len(calls) == attempts


def test_retry_after_cap_and_exponential_delays():
    from threadweave.refinement_retry import retry_delay

    assert [retry_delay(i, None) for i in (1, 2, 3)] == [2, 4, 8]
    assert retry_delay(1, 10) == 10
    assert retry_delay(1, 61) is None
    assert retry_delay(1, 61, max_retry_delay=0) == 61


async def test_refinement_uses_active_root_model_not_auxiliary_role_override(harness_runtime):
    rt, sid, provider = harness_runtime
    config = rt.store.config(sid)
    config.models = {
        "root": config.provider.model_copy(update={"model": "root-model"}),
        "other": config.provider.model_copy(update={"model": "wrong-model"}),
    }
    config.routing.policy = "role_based"
    config.routing.roles = {"agent": "root", "refinement": "other", "refinement_review": "other"}
    rt.store.reconfigure(sid, config)
    await rt.review_refinement(sid, "compact")
    await rt.plan_refinement(sid, {})
    assert [r.config.model for r in provider.requests] == ["root-model", "root-model"]


@pytest.mark.parametrize("reason", ["error", "aborted"])
async def test_failed_assistant_does_not_count_or_start_background_plan(harness_runtime, reason):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    pending = rt.store.session(sid).pending_turn
    pending["response"]["metadata"] = {"stop_reason": reason}
    rt.store.update(sid, pending_turn=pending)
    rt.refinement_state(sid).turns_since_review = 24
    rt.refinement_message_end(sid)
    assert rt.refinement_state(sid).turns_since_review == 24
    assert rt.refinement_state(sid).background is None and not provider.requests
    rt.invalidate_refinement(sid)


async def test_abort_preserves_automatic_counters_but_clears_explicit_request(harness_runtime):
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    state.turns_since_review, state.last_review_at = 24, 123
    state.pending_request = {"instructions": "pending"}
    rt.invalidate_refinement(sid)
    assert state.pending_request is None
    assert state.turns_since_review == 24 and state.last_review_at == 123


async def test_branch_invalidation_resets_auto_state_without_dropping_requested_options(
    harness_runtime,
):
    rt, sid, _ = harness_runtime
    state = rt.refinement_state(sid)
    state.turns_since_review, state.last_review_at = 24, 123
    state.pending_request = {"instructions": "pending"}
    state.pending_compact = state.pending_interval = True
    state.pending_review = ("compact", {"shouldRefine": True})
    await rt.refinement_branch_changed(sid)
    assert state.pending_request == {"instructions": "pending"}
    assert state.turns_since_review == 0 and state.last_review_at == 123
    assert not state.pending_compact and not state.pending_interval and not state.pending_review
    rt.invalidate_refinement(sid)


async def test_disposal_background_failure_consumes_interval_counter(harness_runtime):
    rt, sid, provider = harness_runtime
    provider.review = {"shouldRefine": True}

    async def fail(*args, **kwargs):
        raise ValueError("planner unavailable")

    rt.plan_refinement = fail
    active_turn(rt, sid)
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    state.pending_interval = True
    rt.maybe_start_refinement_plan(sid)
    assert state.background
    await rt.drain_refinement(sid)
    assert state.turns_since_review == 0 and state.last_review_at > 0
    assert not rt.store.harness.history(sid)


async def test_daemon_manual_refine_returns_result_without_resetting_auto_state(harness_runtime):
    from threadweave.daemon import REFINE_REQUEST_TIMEOUT_SECONDS, Daemon

    rt, sid, provider = harness_runtime
    daemon = object.__new__(Daemon)
    daemon.runtime = rt
    state = rt.refinement_state(sid)
    state.turns_since_review = 24
    result = await daemon.dispatch("refine", {"session_id": sid})
    assert result["appliedEdits"][0]["applied"]
    assert len(provider.requests) == 1
    assert state.turns_since_review == 24 and not state.last_review_at
    assert not state.pending_request
    assert REFINE_REQUEST_TIMEOUT_SECONDS == 600


def test_save_returns_caller_path_while_atomically_preserving_symlink(tmp_path):
    from threadweave.harness import save_harness_state

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert save_harness_state(alias, empty_harness_state()) == str(alias / "harness_state.json")
    assert alias.is_symlink() and (real / "harness_state.json").is_file()


async def test_python_helpers_do_not_apply_the_planner_validation_contract(harness_runtime):
    from .test_continual_harness_gaps import kernel_harness

    rt, sid, _ = harness_runtime
    h = kernel_harness(rt, sid)
    assert h.create("prompt", "", "", id="system").id == "system"
    assert h.create("skill", "Generic", "Generic CRUD").reference == {}
    with pytest.raises(ValueError, match="Python reference"):
        h.create_skill("Specific", "Dedicated helper")
    record = h.create_skill(
        "Specific",
        "Dedicated helper",
        reference={"type": "python", "import": "json", "callable": "loads"},
    )
    assert record.arguments == {}
    assert h.update_skill(record.id, "Specific", "Changed").reference == record.reference
    assert not hasattr(h, "create_prompt")


async def test_python_loader_discards_unknown_fields_and_restores_entry_identity(harness_runtime):
    from threadweave.harness import save_harness_state

    from .test_continual_harness_gaps import kernel_harness

    rt, sid, _ = harness_runtime
    state = empty_harness_state()
    state["entries"]["memory"] = {
        "known": {
            "id": "wrong",
            "kind": "skill",
            "title": "Known",
            "content": "data",
            "path": 123,
            "version": "2",
            "source": None,
            "metadata": "invalid",
            "unexpected": True,
        },
        "incomplete": {"title": "Incomplete"},
    }
    state["refinements"] = [
        {"id": "extra", "trigger": "test", "changes": [1, "loaded"], "ignored": True},
        {"id": "missing", "trigger": "test"},
    ]
    save_harness_state(rt.store.harness.path(sid), state)
    h = kernel_harness(rt, sid)
    record = h.get("memory", "known")
    assert (record.id, record.kind, record.path, record.source, record.version) == (
        "known",
        "memory",
        "general",
        "agent",
        2,
    )
    assert record.metadata == {} and "unexpected" not in record
    assert h.get("memory", "incomplete") is None
    assert h.snapshot()["refinements"][0]["changes"] == ["1", "loaded"]
    assert len(h.snapshot()["refinements"]) == 1
    assert h.update_memory("known", "Known", "changed").version == 3
    assert "unexpected" not in rt.store.harness.load(sid)["entries"]["memory"]["known"]


@pytest.mark.parametrize("native", [True, False])
def test_atomic_writer_matches_host_and_python_umask_semantics(tmp_path, native):
    import os

    from threadweave.harness import save_harness_state

    previous = os.umask(0o777)
    try:
        path = save_harness_state(tmp_path, empty_harness_state(), python=native)
        assert os.stat(path).st_mode & 0o777 == (0 if native else 0o600)
        os.chmod(path, 0o640)
        save_harness_state(tmp_path, empty_harness_state(), python=native)
        assert os.stat(path).st_mode & 0o777 == 0o640
    finally:
        os.umask(previous)


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ("rollback refine_123", {"rollback_id": "refine_123", "global_": False}),
        ("rollback refine_456 --global", {"rollback_id": "refine_456", "global_": True}),
        ("--global rollback refine_789", {"rollback_id": "refine_789", "global_": True}),
        ("--global focus on validation", {"instructions": "focus on validation", "global_": True}),
        (
            "update docs to explain --global",
            {"instructions": "update docs to explain --global", "global_": False},
        ),
        ("--local", {"instructions": "--local", "global_": False}),
    ],
)
def test_manual_refine_argument_parser_matches_prime(arguments, expected):
    from threadweave.refinement import refine_command_options

    assert refine_command_options(arguments) == expected


@pytest.mark.parametrize(
    "text", ["prefix /refine", "/refinement", "/refine\nwork", "/refine\t\u2028work"]
)
def test_only_exact_single_line_refine_commands_are_control_input(text):
    from threadweave.refinement import refine_command

    assert refine_command(text) is None


async def test_queued_refine_commands_are_independent_and_model_invisible(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.update(sid, runnable=False)
    provider.entered, provider.release = asyncio.Event(), asyncio.Event()
    rt.message(None, sid, "/refine first request")
    await provider.entered.wait()
    rt.message(None, sid, "/refine second request")
    state = rt.refinement_state(sid)
    barrier = asyncio.create_task(rt.wait_refinement_barrier(sid))
    await asyncio.sleep(0)
    assert len(provider.requests) == 1 and not barrier.done()
    provider.release.set()
    await state.command_task
    await barrier
    assert len(provider.requests) == 2
    assert "first request" in provider.requests[0].messages[-1]["content"]
    assert "second request" in provider.requests[1].messages[-1]["content"]
    assert not state.pending_request and not state.command_task
    assert not rt.store.session(sid).runnable
    assert len(rt.store.events(sid, kind="session_slash_command_result")) == 2
    assert "/refine first request" not in str(rt.context.messages(sid))
    assert "Refined continual harness state:" not in rt.refinement_input(sid)["conversation"]


async def test_missing_rollback_id_emits_command_failure_without_calling_planner(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.message(None, sid, "/refine rollback")
    await rt.refinement_state(sid).command_task
    assert not provider.requests
    assert (
        rt.store.events(sid, kind="refine_failed")[-1]["payload"]["error"]
        == "Usage: /refine rollback <refinement-id>"
    )
    assert "Command failed:" not in str(rt.context.messages(sid))


@pytest.mark.parametrize("decision,nonempty", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("creation", ["runtime", "daemon"])
async def test_twenty_five_real_runtime_turn_boundaries(
    tmp_path, python_config, decision, nonempty, creation
):
    """Deterministic provider, real turn/tool scheduler: no direct checkpoint calls."""
    from threadweave.models import Action
    from threadweave.runtime import Runtime

    calls = []
    primary_turns = reviews = planners = 0

    class Provider:
        async def invoke(self, request, emit):
            nonlocal primary_turns, reviews, planners
            purpose = request.metadata.get("purpose", "agent")
            calls.append(purpose)
            if purpose == "refinement_review":
                reviews += 1
                assert primary_turns == 25
                assert "queued boundary input" not in str(request.messages)
                return ModelResponse(
                    text=json.dumps({"shouldRefine": decision, "rationale": "review"})
                )
            if purpose == "refinement":
                planners += 1
                assert primary_turns == 25 and decision
                return ModelResponse(text=json.dumps(proposal(*([edit()] if nonempty else []))))
            primary_turns += 1
            if primary_turns == 25:
                runtime.message(None, root.id, "queued boundary input")
            if primary_turns <= 25:
                assert reviews == 0
                return ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={"code": f"step = {primary_turns}\nstep * step"},
                        )
                    ]
                )
            assert reviews == 1 and planners == int(decision)
            assert "queued boundary input" in str(request.messages)
            assert ("[auto-refinement]" in str(request.messages)) == nonempty
            return ModelResponse(text="Done")

    config = python_config.model_copy(deep=True)
    config.context.max_tokens = 128000
    config.context.compact_at = 0.95
    runtime = Runtime(tmp_path / "state", providers={"mock": Provider()})
    try:
        if creation == "daemon":
            from threadweave.daemon import Daemon

            daemon = object.__new__(Daemon)
            daemon.runtime = runtime
            created = await daemon.dispatch(
                "create",
                {
                    "instruction": "Execute the deterministic scheduler regression",
                    "workspace": str(tmp_path),
                    "config": config.model_dump(mode="json"),
                },
            )
            root = runtime.store.session(created["id"])
        else:
            root = runtime.create(
                "Execute the deterministic scheduler regression", tmp_path, config=config
            )
        await runtime.start()
        await runtime.wait(root.id, timeout=30)
        assert primary_turns == 26
        assert calls[24:27] == (
            ["agent", "refinement_review", "refinement"]
            if decision
            else ["agent", "refinement_review", "agent"]
        )
        assert bool(runtime.store.harness.entries(root.id)) == nonempty
        assert runtime.refinement_state(root.id).turns_since_review == 1
    finally:
        await runtime.shutdown()


async def test_subscription_refinement_wire_omits_reasoning_and_primary_options(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from threadweave.subscription import SubscriptionProvider

    from .test_subscription import discard, model_request, stream

    captured = []

    class Sink:
        def write(self, data):
            captured.append(json.loads(data))

        async def drain(self):
            pass

        def close(self):
            pass

    async def wait():
        return 0

    async def process(*args, **kwargs):
        return SimpleNamespace(
            stdin=Sink(),
            stdout=stream(
                [
                    {
                        "type": "completed",
                        "id": "response",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ]
            ),
            returncode=0,
            wait=wait,
        )

    executable = tmp_path / "client"
    executable.touch()
    provider = SubscriptionProvider(executable=executable)

    async def resolve(config, *, reasoning_off=False):
        assert reasoning_off
        return config.model_copy(update={"parameters": {}}), {}

    monkeypatch.setattr(provider, "resolve", resolve)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", process)
    request = model_request().model_copy(update={"reasoning_mode": "off", "tools": []})
    await provider.invoke(request, discard)
    body = captured[0]["body"]
    assert not {"reasoning", "prompt_cache_key", "tools"} & body.keys()
    assert body["text"] == {"verbosity": "low"}


async def test_ready_background_apply_failure_consumes_counter_and_emits_failure(
    harness_runtime, monkeypatch
):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    rt.maybe_start_refinement_plan(sid)
    await state.background
    rt.store.update(sid, pending_turn={"context_committed": True})

    def fail(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(rt.store.harness, "apply", fail)
    assert not await rt.refinement_checkpoint(sid)
    assert state.turns_since_review == 0 and state.last_review_at > 0
    assert len(rt.store.events(sid, kind="refine_failed")) == 1
    assert len(provider.requests) == 2


async def test_request_received_during_background_apply_is_consumed_at_same_boundary(
    harness_runtime,
):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    rt.request_refinement(sid)
    await rt.refinement_state(sid).background
    rt.store.update(sid, pending_turn={"context_committed": True})
    original = rt.apply_refinement_plan
    applied = 0

    async def apply(*args):
        nonlocal applied
        result = await original(*args)
        applied += 1
        if applied == 1:
            provider.proposal = proposal(edit(id="second"))
            assert rt.request_refinement(sid, instructions="new request")["scheduled"]
        return result

    rt.apply_refinement_plan = apply
    assert await rt.refinement_checkpoint(sid)
    assert applied == 2 and len(provider.requests) == 2
    assert not rt.refinement_state(sid).pending_request
    assert set(rt.store.harness.load(sid)["entries"]["memory"]) == {"lesson", "second"}
