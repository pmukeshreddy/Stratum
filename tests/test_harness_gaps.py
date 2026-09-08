"""Production regressions for continuation, compaction, recovery and durable context."""

import json

import pytest

from threadweave.models import HarnessError, ModelResponse, StateEdit, Usage, new_id
from threadweave.runtime import Runtime
from threadweave.storage import encode
from threadweave.subscription import SubscriptionProvider, responses_input
from threadweave.tools import ToolContext

from .fakes import ScriptedProvider
from .test_subscription import discard, model_request, stream


async def test_reasoning_continuation_replayed_after_restore(tmp_path, config):
    opaque = {
        "type": "reasoning",
        "id": "rs1",
        "summary": [],
        "encrypted_content": "opaque-ciphertext",
    }
    call = {
        "type": "function_call",
        "call_id": "native-call",
        "name": "workspace_list",
        "arguments": "{}",
    }
    request = model_request()
    request.tools[0]["function"]["name"] = "workspace_list"
    result = await SubscriptionProvider.collect(
        stream(
            [
                {"type": "item", "item": opaque},
                {"type": "item", "item": call},
                {"type": "completed", "usage": {"input_tokens": 30, "output_tokens": 20}},
            ]
        ),
        request,
        request.config,
        discard,
    )
    provider = ScriptedProvider({"root": [result]})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Inspect files", tmp_path, config=config)
    try:
        await runtime._run_turn(root.id)
        assert "opaque-ciphertext" not in encode(runtime.store.events(root.id))
    finally:
        await runtime.shutdown()
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        await runtime._invoke(root.id)
        _, items = responses_input(provider.requests[-1].messages)
        position = items.index(opaque)
        assert items[position + 1] == call
        assert items[position + 2]["type"] == "function_call_output"
        assert items[position + 2]["call_id"] == "native-call"
    finally:
        await runtime.shutdown()


async def test_compaction_covers_material_beyond_24000_and_keeps_recent(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.max_tokens = 10000
    config.context.recent_blocks = 1
    config.features.model_compaction = True
    seen = []

    class Compactor:
        async def invoke(self, request, emit):
            seen.append(request.messages[-1]["content"])
            return ModelResponse(
                text=json.dumps({"unresolved_work": ["Review archived evidence"]}),
                usage=Usage(input_tokens=30, output_tokens=20),
            )

    runtime = Runtime(tmp_path / "state", providers={"mock": Compactor()})
    root = runtime.create("Recover facts", tmp_path, config=config)
    try:
        source = (
            "old line with evidence\n" * 5000
            + "\nCritical fact: release requires checksum ZEBRA-937."
        )
        eid = runtime.store.event(root.id, "observation", {"source": "original"})
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": source}])
        recent = [{"role": "user", "content": "Recent exact context stays verbatim."}]
        runtime.store.add_context(root.id, runtime.store.event(root.id, "observation", {}), recent)
        await runtime.semantic_compact(root.id)
        assert "ZEBRA-937" in "".join(seen)
        compaction = runtime.store.events(root.id, kind="context_compaction")[-1]["payload"]
        archive = runtime.artifacts.load(root.id, compaction["archive_artifact"])
        assert "ZEBRA-937" in encode(archive)
        assert compaction["archive_artifact"] in runtime.store.session(root.id).summary
        assert runtime.store.session(root.id).context[-1]["messages"] == recent
    finally:
        await runtime.shutdown()


async def test_context_overflow_compacts_retries_same_turn_and_accounts(tmp_path, config):
    calls = []

    class OverflowOnce:
        async def invoke(self, request, emit):
            calls.append(request)
            if len(calls) == 1:
                raise HarnessError("provider", "context_overflow", "Context capacity reached")
            return ModelResponse(text="Recovered", usage=Usage(input_tokens=20, output_tokens=10))

    config.features.model_compaction = False
    runtime = Runtime(tmp_path / "state", providers={"mock": OverflowOnce()})
    root = runtime.create("Continue the same logical task", tmp_path, config=config)
    try:
        for _ in range(8):
            eid = runtime.store.event(root.id, "observation", {})
            runtime.store.add_context(
                root.id, eid, [{"role": "user", "content": "old evidence " * 100}]
            )
        result, _ = await runtime._invoke(root.id)
        assert result.text == "Recovered"
        assert len(calls) == 2 and calls[0].turn == calls[1].turn
        assert calls[1].input_token_bound < calls[0].input_token_bound
        assert runtime.store.events(root.id, kind="context_compaction")
        usage = runtime.store.usage(root.id)
        assert usage.model_calls == 2 and usage.retries == 1
        assert usage.input_tokens >= 20 and usage.output_tokens >= 10
    finally:
        await runtime.shutdown()


async def test_response_beyond_16384_bytes_is_not_a_terminal_failure():
    request = model_request()
    request.config.max_output_tokens = 4096
    text = "x" * 20000
    result = await SubscriptionProvider.collect(
        stream(
            [
                {"type": "text_delta", "text": text},
                {"type": "completed", "usage": {"input_tokens": 100, "output_tokens": 3000}},
            ]
        ),
        request,
        request.config,
        discard,
    )
    assert result.text == text and result.usage.output_tokens == 3000


@pytest.mark.parametrize("leaf", ["src", "src/foo"])
async def test_repository_instructions_are_loaded_in_hierarchy(tmp_path, config, leaf):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src/foo").mkdir(parents=True)
    (repo / "other").mkdir()
    (repo / "AGENTS.md").write_text("ROOT repository instruction")
    (repo / "src/AGENTS.md").write_text("NESTED source instruction")
    (repo / "src/CLAUDE.md").write_text("CLAUDE source instruction")
    (repo / "other/AGENTS.md").write_text("UNRELATED instruction")
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("Implement change", repo / leaf, config=config)
        messages = encode(runtime.context.messages(root.id))
        assert (
            messages.index("ROOT repository")
            < messages.index("NESTED source")
            < messages.index("CLAUDE source")
        )
        assert "UNRELATED instruction" not in messages
        loaded = runtime.store.events(root.id, kind="repository_instructions_loaded")[-1][
            "payload"
        ]["files"]
        assert [x["path"] for x in loaded] == [
            str(repo / p) for p in ("AGENTS.md", "src/AGENTS.md", "src/CLAUDE.md")
        ]
        assert all(x["sha256"] for x in loaded)
    finally:
        await runtime.shutdown()


async def test_refiner_knows_existing_memory_and_later_task_retrieves_it(tmp_path, config):
    config.refinement.automatic = True
    config.refinement.allow_global_writes = True
    provider = ScriptedProvider({})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Repair zeta checksum validation", tmp_path, config=config)
    try:
        eid = runtime.store.event(
            root.id, "verifier_result", {"passed": False, "reason": "zeta checksum"}
        )
        runtime.store.queue_refinement(
            root.id,
            StateEdit(
                scope="global",
                title="Zeta checksum",
                content={"text": "Zeta checksum must include header bytes."},
                source_events=[eid],
                intended_effect="Avoid checksum corruption",
            ),
        )
        entry = runtime.store.apply_refinements(root.id)[0]

        class Refiner:
            async def invoke(self, request, emit):
                evidence = json.loads(request.messages[-1]["content"])
                assert entry in encode(evidence["existing_state"])
                return ModelResponse(
                    text=json.dumps(
                        {
                            "proposals": [
                                {
                                    "entry_id": entry,
                                    "expected_version": 1,
                                    "scope": "global",
                                    "kind": "memory",
                                    "title": "Zeta checksum",
                                    "content": {
                                        "text": "Zeta checksum includes header bytes and payload."
                                    },
                                    "source_events": [eid],
                                    "intended_effect": "Update existing lesson",
                                }
                            ]
                        }
                    )
                )

        runtime.providers["mock"] = Refiner()
        await runtime.auto_refine(root.id, trigger="completion")
        assert runtime.store.state(root.id, entry)["version"] == 2
        later = runtime.create("Fix the zeta checksum parser", tmp_path, config=config)
        assert "includes header bytes and payload" in encode(runtime.context.messages(later.id))
        unrelated = runtime.create("Draw a cat", tmp_path, config=config)
        assert "includes header bytes and payload" not in encode(
            runtime.context.messages(unrelated.id)
        )
    finally:
        await runtime.shutdown()


async def test_deep_tool_stdout_visible_and_full_output_retrievable(tmp_path, python_config):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Inspect command output", tmp_path, config=python_config)
    try:
        eid = runtime.store.event(root.id, "test_source", {})
        context = ToolContext(runtime, root.id, new_id(), eid)
        result = await runtime.execute_python(
            context, "print('x' * 40000 + 'DEEP_EVIDENCE' + 'y' * 80000)"
        )
        exposed = runtime.artifacts.expose(root.id, result)
        assert "DEEP_EVIDENCE" in exposed["preview"]
        assert "DEEP_EVIDENCE" in runtime.artifacts.load(root.id, result["stdout_artifact"])
        assert len(runtime.artifacts.load(root.id, result["stdout_artifact"])) > 120000
        assert result["stdout_artifact"] in exposed["preview"] or exposed.get("inspection")
    finally:
        await runtime.shutdown()


async def test_compaction_failure_preserves_complete_archive(tmp_path, config):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Preserve unknown facts", tmp_path, config=config)
    try:
        eid = runtime.store.event(root.id, "observation", {})
        body = "x" * 30000 + "RESTORABLE-END-FACT"
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": body}])
        runtime.context.compact(root.id)
        first = runtime.store.session(root.id).summary
        artifact = runtime.store.events(root.id, kind="context_compaction")[-1]["payload"][
            "archive_artifact"
        ]
        assert body in encode(runtime.artifacts.load(root.id, artifact))
        eid = runtime.store.event(root.id, "observation", {})
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": "another fact"}])
        runtime.context.compact(root.id)
        second = runtime.store.events(root.id, kind="context_compaction")[-1]["payload"][
            "archive_artifact"
        ]
        assert runtime.artifacts.load(root.id, second)["previous_summary"] == first
    finally:
        await runtime.shutdown()


async def test_large_tool_arguments_are_preserved_without_byte_guard():
    request = model_request()
    request.config.max_output_tokens = 4096
    arguments = json.dumps({"pattern": "x" * 20000})
    result = await SubscriptionProvider.collect(
        stream(
            [
                {"type": "tool_delta", "delta": arguments},
                {
                    "type": "item",
                    "item": {
                        "type": "function_call",
                        "call_id": "large",
                        "name": "repo_search",
                        "arguments": arguments,
                    },
                },
                {"type": "completed", "usage": {"input_tokens": 10, "output_tokens": 3000}},
            ]
        ),
        request,
        request.config,
        discard,
    )
    assert result.actions[0].arguments["pattern"] == "x" * 20000
