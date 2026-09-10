"""Exact remaining Prime state, resource, input-commit and transport variants."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from threadweave.context import Context
from threadweave.harness import (
    KINDS,
    append_refinement_history,
    apply_refinement_proposal,
    atomic_json,
    empty_harness_state,
    format_harness_state,
    infer_scope,
    js_json,
    load_harness_state,
    load_refinement_history,
    merge_refinement_history,
    save_harness_state,
)
from threadweave.refinement_context import convert_to_llm, custom_message, serialize_conversation
from threadweave.refinement_transport import native_events
from threadweave.skills import refine_skill

from .test_continual_harness import edit, proposal
from .test_continual_harness import harness_runtime as harness_runtime
from .test_continual_harness_gaps import active_turn


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "variant",
    ["duplicate", "missing_update", "missing_delete", "title", "content", "id", "action", "kind"],
)
def test_invalid_edit_variants_are_independent(kind, variant):
    state = empty_harness_state()
    item = edit(kind)
    if variant == "duplicate":
        apply_refinement_proposal(state, proposal(item), id="seed")
    elif variant.startswith("missing_"):
        item["action"] = variant.removeprefix("missing_")
    elif variant == "id":
        item.update(action="delete", id="")
    elif variant in {"title", "content"}:
        item.pop(variant)
    else:
        item[variant] = "unsupported"
    before = copy.deepcopy(state["entries"])
    result = apply_refinement_proposal(state, proposal(item), id="invalid")
    assert not result["appliedEdits"][0]["applied"]
    assert result["appliedEdits"][0]["error"]
    assert state["entries"] == before


@pytest.mark.parametrize("identifier", ["base_system_prompt", None])
def test_base_prompt_is_immutable_including_derived_id(identifier):
    result = apply_refinement_proposal(
        empty_harness_state(),
        proposal(edit("prompt", id=identifier, title="Base system prompt")),
        id="invalid",
    )
    assert not result["appliedEdits"][0]["applied"]


@pytest.mark.parametrize("scope", [None, "global", "local"])
def test_legacy_history_scope_inference_and_shadowing(tmp_path, scope):
    result = {"id": "legacy", "appliedEdits": [], "harnessStatePath": "state.json"}
    if scope:
        result["appliedEdits"] = [{"after": {"scope": scope}}]
    assert infer_scope(result, "global") == (scope or "global")
    append_refinement_history(tmp_path, result)
    loaded = load_refinement_history(tmp_path)
    assert loaded[0]["scope"] == (scope or "global")
    assert merge_refinement_history(loaded, [{**result, "summary": "new"}])[0]["scope"] == (
        scope or "global"
    )


@pytest.mark.parametrize("global_", [False, True])
async def test_rollback_planning_only_and_missing_target(harness_runtime, global_):
    rt, sid, provider = harness_runtime
    result = await rt.refine(sid, global_=global_)
    state = rt.store.harness.load(None if global_ else sid)
    plan = await rt.plan_refinement(sid, {"rollback_id": result["id"]})
    assert plan["global_"] == global_
    assert rt.store.harness.load(None if global_ else sid) == state
    assert len(provider.requests) == 1
    with pytest.raises(ValueError, match="not found"):
        await rt.plan_refinement(sid, {"rollback_id": "missing"})
    await rt.apply_refinement_plan(sid, plan, {})
    assert not rt.store.harness.load(None if global_ else sid)["entries"]["memory"]


@pytest.mark.parametrize("field", ["reference", "arguments", "metadata"])
async def test_python_dictionary_coercion_and_omission(harness_runtime, field):
    rt, sid, _ = harness_runtime
    store = rt.store.harness
    created = store.mutate(
        sid, "create", "memory", title="entry", content="x", **{field: [("x", 1)]}
    )
    assert created[field] == {"x": 1}
    changed = store.mutate(
        sid, "update", "memory", id="entry", title="entry", content="y", **{field: None}
    )
    assert changed[field] == {"x": 1}
    with pytest.raises(TypeError):
        store.mutate(sid, "update", "memory", id="entry", **{field: 0})
    assert store.get(sid, "memory", "entry") == changed
    assert (
        store.mutate(
            sid, "update", "memory", id="entry", title="entry", content="y", **{field: {}}
        )[field]
        == {}
    )
    assert (
        store.mutate(sid, "create", "memory", title="empty", content="", **{field: 0})[field] == {}
    )


@pytest.mark.parametrize("action", ["get", "create", "update", "delete", "upsert"])
async def test_python_unknown_kind_and_external_create(harness_runtime, action):
    rt, sid, _ = harness_runtime
    store = rt.store.harness
    with pytest.raises(ValueError, match="[Uu]nknown harness kind"):
        store.get(sid, "unknown", "entry") if action == "get" else store.mutate(
            sid, action, "unknown", id="entry"
        )
    store.mutate(sid, "create", "memory", title="entry", content="external")
    with pytest.raises(ValueError, match="already exists"):
        store.mutate(sid, "create", "memory", id="entry", title="entry", content="overwrite")
    assert store.get(sid, "memory", "entry")["content"] == "external"


def test_ecmascript_json_numbers_key_order_and_locale_formatting(tmp_path):
    value = {"10": -0.0, "2": 1e-6, "x": [1e20, 1e21, 1e-7, 9007199254740993, float("inf")]}
    assert (
        js_json(value)
        == '{"2":0.000001,"10":0,"x":[100000000000000000000,1e+21,1e-7,9007199254740992,null]}'
    )
    state = empty_harness_state()
    for title in ["Z", "a", "A", "ä"]:
        apply_refinement_proposal(state, proposal(edit(id=title, title=title)), id=title)
    formatted = format_harness_state(state)
    assert [formatted.index(f"[local:{title}]") for title in ["a", "A", "ä", "Z"]] == sorted(
        formatted.index(f"[local:{title}]") for title in ["a", "A", "ä", "Z"]
    )
    save_harness_state(tmp_path, value)
    assert "1e+21" in (tmp_path / "harness_state.json").read_text()


def test_zero_history_limit_retains_prime_slice_semantics():
    state = empty_harness_state()
    state["refinements"] = [{"id": "old", "trigger": "KEEP_HISTORY", "changes": []}]
    assert "KEEP_HISTORY" in format_harness_state(state, refinement_limit=0)


@pytest.mark.parametrize("channel", ["refinement_sse", "refinement_event"])
async def test_native_null_is_protocol_error_and_scalars_are_ignored(channel):
    async def lines(raw):
        yield json.dumps(
            {"type": channel, "bytes": list(f"data: {raw}\n\n".encode())}
            if channel == "refinement_sse"
            else {"type": channel, "event": json.loads(raw)}
        )
        yield json.dumps({"type": "refinement_end"})

    assert [event async for event in native_events(lines("42"))] == [{"type": "completed"}]
    with pytest.raises(ValueError, match="null"):
        _ = [event async for event in native_events(lines("null"))]


def test_native_usage_limit_expansion_requires_nested_error():
    from threadweave.refinement_transport import normalize_event

    flat = {"type": "error", "code": "usage_limit_reached", "message": "Original"}
    assert normalize_event(flat)["error_message"] == "Codex error: Original"
    nested = {"type": "error", "error": {"code": "usage_limit_reached", "message": "Original"}}
    assert normalize_event(nested)["error_message"] == "You have hit your ChatGPT usage limit."


@pytest.mark.parametrize("python", [False, True])
def test_atomic_failure_preserves_destination_and_temp_permissions(tmp_path, monkeypatch, python):
    path = tmp_path / "harness_state.json"
    path.write_text("previous")
    path.chmod(0o640)

    def fail(source, destination):
        assert Path(source).stat().st_mode & 0o777 == 0o640
        raise OSError("replace failed")

    monkeypatch.setattr("threadweave.harness.os.replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, empty_harness_state(), python=python)
    assert path.read_text() == "previous" and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("raw", ["broken", "null", "[]", '"value"'])
def test_python_corrupt_state_is_empty(tmp_path, raw):
    (tmp_path / "harness_state.json").write_text(raw)
    assert load_harness_state(tmp_path, python=True) == empty_harness_state()


async def test_first_digest_staged_rearmed_and_transactionally_committed(harness_runtime):
    rt, sid, _ = harness_runtime
    context = rt.context
    assert not rt.store.session(sid).context
    context.ensure_harness_digest(sid)
    assert not rt.store.session(sid).context
    assert "Continual Harness State" in str(context.messages(sid))
    with pytest.raises(RuntimeError):
        with rt.store.transaction():
            context.commit_harness_digest(sid)
            raise RuntimeError("request admission rejected")
    assert not rt.store.session(sid).context
    assert not rt.store.events(sid, kind="harness_digest")
    rt.store.harness.mutate(sid, "create", "memory", title="fresh", content="AFTER_FAILED_COMMIT")
    context.ensure_harness_digest(sid)
    assert "AFTER_FAILED_COMMIT" in str(context.messages(sid))
    with rt.store.transaction():
        context.commit_harness_digest(sid)
    context.harness_digest_committed(sid)
    context.ensure_harness_digest(sid)
    assert len(rt.store.events(sid, kind="harness_digest")) == 1
    assert str(context.messages(sid)).count("# Continual Harness State") == 1


async def test_cold_navigation_deduplicates_timestamp_ties(harness_runtime):
    rt, sid, _ = harness_runtime
    digest = rt.context.harness_digest(sid)
    for value in ["obsolete", digest]:
        msg = custom_message("harness_digest", value, details={"digest": value})
        msg["timestamp"] = 100
        rt.store.add_context(sid, rt.store.event(sid, "seed", {}), [msg])
    Context(rt.store).ensure_harness_digest(sid)
    assert not rt.store.events(sid, kind="harness_digest")


async def test_refine_command_fifo_and_recovery_without_synthetic_user_input(harness_runtime):
    rt, sid, provider = harness_runtime
    rt.store.update(sid, runnable=False)
    earlier = rt.message(None, sid, "first legitimate input")
    command = rt.message(None, sid, "/refine requested edit")
    later = rt.message(None, sid, "later legitimate input")
    assert rt.refinement_state(sid).command_task is None
    assert [m["id"] for m in rt.receive(sid)] == [earlier]
    # A new in-memory checkpoint (restart) still finds the command in the durable message records.
    rt._refinement_states.pop(sid, None)
    task = rt.start_queued_refine_command(sid)
    assert task is not None
    await task
    assert len(provider.requests) == 1
    assert "first legitimate input" in str(provider.requests[0].messages)
    assert "later legitimate input" not in str(provider.requests[0].messages)
    assert [m["id"] for m in rt.receive(sid)] == [later]
    assert "/refine requested edit" not in rt.refinement_input(sid)["conversation"]
    assert next(m for m in rt.store.messages(sid) if m["id"] == command)["received_at"]


async def test_settled_plan_replacement_and_explicit_interval_priority(harness_runtime):
    rt, sid, provider = harness_runtime
    active_turn(rt, sid)
    state = rt.refinement_state(sid)
    state.turns_since_review = 25
    rt.request_refinement(sid, instructions="old")
    await state.background
    rt.request_refinement(sid, instructions="replacement")
    rt.store.update(sid, pending_turn={"context_committed": True})
    await rt.refinement_checkpoint(sid)
    assert len(provider.requests) == 2
    assert all(r.metadata["purpose"] == "refinement" for r in provider.requests)
    assert "replacement" in str(provider.requests[-1].messages)
    assert len(rt.store.harness.history(sid)) == 1


def test_refine_user_override_and_disable_model_invocation(tmp_path):
    path = tmp_path / ".agents/skills/refine/SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: refine\ndescription: Custom guide\ndisable-model-invocation: true\n---\nprivate"
    )
    skill = refine_skill(tmp_path)
    assert skill["path"] == str(path) and skill["disable_model_invocation"]
    assert refine_skill(tmp_path, child=True) == skill
    assert refine_skill(tmp_path / "empty", child=True) is None


@pytest.mark.parametrize("role", ["assistant", "user", "tool"])
def test_conversation_serializer_short_tool_and_unbounded_ordinary_text(role):
    content = "kept" if role == "tool" else "x" * 3000
    assert content in serialize_conversation([{"role": role, "content": content, "name": "read"}])
    assert convert_to_llm([custom_message("ordinary", "custom evidence", {})]) == [
        {"role": "user", "content": "custom evidence"}
    ]


async def wire_events(*events, chunk=7):
    raw = "".join(
        "data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events
    ).encode()
    for offset in range(0, len(raw), chunk):
        yield json.dumps({"type": "refinement_sse", "bytes": list(raw[offset : offset + chunk])})
    yield json.dumps({"type": "refinement_end"})


@pytest.mark.parametrize("terminal", ["response.completed", "response.done", "response.incomplete"])
@pytest.mark.parametrize(
    "status,reason",
    [("completed", "stop"), ("incomplete", "length"), ("cancelled", "error"), ("unknown", "stop")],
)
async def test_native_raw_sse_preserves_codex_status_and_utf8(terminal, status, reason):
    events = [
        event
        async for event in native_events(
            wire_events(
                {"type": "response.output_text.delta", "delta": "é🦬"},
                {"type": terminal, "response": {"status": status}},
            )
        )
    ]
    assert events[0] == {"type": "text_delta", "text": "é🦬"}
    assert events[1]["stop_reason"] == reason


@pytest.mark.parametrize(
    "code",
    [
        "permission_denied",
        "invalid_prompt",
        "overloaded_error",
        "usage_not_included",
        "custom_error",
    ],
)
async def test_native_raw_error_retains_provider_code_not_sdk_classification(code):
    events = [
        event
        async for event in native_events(
            wire_events(
                {
                    "type": "response.failed",
                    "response": {"error": {"code": code, "message": "original provider message"}},
                }
            )
        )
    ]
    assert len(events) == 1
    assert events[0]["provider_error_type"] == code
    assert events[0]["error_message"] == "original provider message"


async def test_native_eof_is_valid_and_never_adds_sdk_retry():
    assert [event async for event in native_events(wire_events())] == [{"type": "completed"}]
    source = await asyncio.to_thread(Path("src/threadweave/native/inference.rs").read_text)
    branch = source.split("if refinement {\n        // Prime's one-shot path", 1)[1].split(
        "let client =", 1
    )[0]
    assert "recovery.next" not in branch and "return Ok(())" in branch
