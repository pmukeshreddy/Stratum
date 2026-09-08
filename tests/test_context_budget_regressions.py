"""Regressions reproduced against the pre-scaling production implementation."""

import asyncio
import json

import pytest

from threadweave.context import token_bound
from threadweave.models import HarnessError, ModelResponse, StateEdit, Usage
from threadweave.runtime import Runtime
from threadweave.storage import encode

from .fakes import ScriptedProvider


@pytest.mark.parametrize(
    "requirements",
    [
        ["Still must verify the release checksum."],
        [
            "Still must verify the release checksum.",
            "Still must preserve the pending migration.",
            "Still must resolve the failed backup.",
        ],
    ],
)
async def test_oversized_structured_summary_retains_late_unresolved_work(
    tmp_path, config, requirements
):
    config.provider.model = "gpt-6-astra"
    config.context.summary_chars = 6000
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Finish every requirement", tmp_path, config=config)
    try:
        body = "Completed background detail.\n" * 2000 + "\n".join(requirements)
        eid = runtime.store.event(root.id, "observation", {"body": body})
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": body}])
        summary = {
            "objective": "Finish every requirement",
            "established_facts": [
                f"Background fact {i}: completed descriptive material." for i in range(600)
            ],
            "unresolved_requirements": requirements,
            "blockers": ["Backup not verified"],
            "next_actions": ["Verify backup before release"],
        }
        runtime.context.compact(root.id, summary=encode(summary), provenance=eid)
        active = runtime.store.session(root.id).summary
        assert all(item in active for item in requirements)
        assert [active.index(item) for item in requirements] == sorted(
            active.index(item) for item in requirements
        )
        parsed = json.loads(active)
        assert parsed["unresolved_requirements"] == requirements
        assert "Backup not verified" in active
        assert "Verify backup before release" in active
        assert token_bound(active, config.provider.model) <= 1500
        assert "artifacts.load" in active
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("count", [1, 3])
async def test_selected_large_state_gets_semantic_excerpt_and_reference(tmp_path, config, count):
    config.provider.model = "gpt-6-astra"
    config.context.supplemental_chars = 1800
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Inspect data", tmp_path, config=config)
    try:
        event = runtime.store.event(root.id, "observation", {})
        ids = []
        for i in range(count):
            body = f"LESSON-{i}: Verify payload checksums before accepting data. " * 1200
            content = (
                {"text": body}
                if i != 1
                else {
                    "name": "checksum",
                    "description": "Verify checksums",
                    "code": "# " + body,
                }
            )
            runtime.store.queue_refinement(
                root.id,
                StateEdit(
                    title=f"Selected lesson {i}",
                    kind=["memory", "skill", "prompt_note"][i],
                    content=content,
                    source_events=[event],
                    intended_effect="Remember useful evidence",
                ),
            )
            ids.extend(runtime.store.apply_refinements(root.id))
        runtime.store.update(root.id, selected_state=ids)
        first = runtime.context.supplemental(root.id)
        assert all(f"LESSON-{i}" in first for i in range(count))
        assert all(f"://{entry}" in first for entry in ids)
        assert "selected" in first.lower() and "additional" in first.lower()
        assert first == runtime.context.supplemental(root.id)
        assert [first.index(entry) for entry in ids] == sorted(first.index(entry) for entry in ids)
    finally:
        await runtime.shutdown()


class EvidenceReducer:
    def __init__(self, interrupt=False):
        self.requests = []
        self.interrupt = interrupt

    async def invoke(self, request, emit):
        self.requests.append(request)
        if self.interrupt and len(self.requests) == 2:
            raise asyncio.CancelledError()
        evidence = request.messages[-1]["content"]
        if '"proposals"' in request.messages[0]["content"]:
            records = json.loads(evidence)["evidence"]
            text = {
                "proposals": [
                    {
                        "kind": "memory",
                        "title": "Combined evidence lesson",
                        "content": {
                            "text": " ".join(
                                marker
                                for marker in ("EARLY-LESSON", "LATE-LESSON")
                                if marker in evidence
                            )
                        },
                        "source_events": [record["id"] for record in records],
                        "intended_effect": "Retain both ends",
                    }
                ]
            }
        else:
            text = {
                "findings": [
                    marker
                    for marker in ("EARLY-LESSON", "LATE-LESSON", "NORMAL-LESSON")
                    if marker in evidence
                ]
            }
        return ModelResponse(text=encode(text), usage=Usage(input_tokens=30, output_tokens=10))


@pytest.mark.parametrize("sizes", [[20000], [20, 20000], [20000, 24000]])
async def test_refinement_chunks_complete_records_and_accounts(tmp_path, config, sizes):
    config.provider.model = "gpt-6-astra"
    config.context.max_tokens = 4096
    provider = EvidenceReducer()
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Learn from observed failures", tmp_path, config=config)
    try:
        ids = []
        for size in sizes:
            ids.append(
                runtime.store.event(
                    root.id,
                    "python_result",
                    {
                        "stdout": "EARLY-LESSON "
                        + "padding evidence " * size
                        + " LATE-LESSON NORMAL-LESSON",
                    },
                )
            )
        await runtime._refinement_pass(root.id, "completion")
        assert len(provider.requests) > 1
        assert all(request.input_token_bound <= 4096 - 128 for request in provider.requests)
        final = provider.requests[-1].messages[-1]["content"]
        assert "EARLY-LESSON" in final and "LATE-LESSON" in final
        assert all(eid in final for eid in ids)
        chunks = [r for r in provider.requests if '"chunk_index"' in r.messages[-1]["content"]]
        assert chunks
        assert all(
            '"total_chunks"' in r.messages[-1]["content"]
            and '"timestamp"' in r.messages[-1]["content"]
            for r in chunks
        )
        usage = runtime.store.usage(root.id, tree=True)
        assert usage.model_calls == len(provider.requests)
        assert usage.input_tokens == 30 * len(provider.requests)
        assert usage.output_tokens == 10 * len(provider.requests)
        assert not runtime.store.events(root.id, kind="refinement_failed")
        assert all(
            marker in encode(runtime.store.states(root.id))
            for marker in ("EARLY-LESSON", "LATE-LESSON")
        )
    finally:
        await runtime.shutdown()


async def test_interrupted_refinement_chunks_keep_provenance_and_usage(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.max_tokens = 4096
    provider = EvidenceReducer(interrupt=True)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Learn safely", tmp_path, config=config)
    try:
        runtime.store.event(root.id, "python_result", {"stdout": "evidence " * 40000})
        with pytest.raises(asyncio.CancelledError):
            await runtime._refinement_pass(root.id, "completion")
        assert len(provider.requests) == 2
        assert runtime.store.usage(root.id, tree=True).model_calls == 2
        assert runtime.store.events(root.id, kind="refinement_evidence_chunk")
        assert runtime.store.events(root.id, kind="refinement_status")[-1]["payload"]["uncertain"]
        assert runtime.store.states(root.id) == []
    finally:
        await runtime.shutdown()


async def test_unresolved_summary_overflow_retains_source_and_previous_digest(tmp_path, config):
    config.context.summary_tokens = 256
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Retain pending work", tmp_path, config=config)
    try:
        eid = runtime.store.event(root.id, "observation", {})
        runtime.store.add_context(
            root.id, eid, [{"role": "user", "content": "Still must verify checksum."}]
        )
        before = runtime.store.session(root.id)
        with pytest.raises(HarnessError, match="Source context retained"):
            runtime.context.compact(
                root.id,
                summary=encode({"unresolved_requirements": ["Still must verify checksum. " * 200]}),
            )
        after = runtime.store.session(root.id)
        assert after.context == before.context and after.summary == before.summary
    finally:
        await runtime.shutdown()


async def test_extractive_compaction_parses_json_envelopes_without_false_errors(tmp_path, config):
    config.context.summary_chars = 600
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Keep pending work", tmp_path, config=config)
    try:
        eid = runtime.store.event(root.id, "observation", {})
        content = "Environment: " + encode(
            {"error": None, "detail": "routine " * 10000, "pending": "Still must verify checksum."}
        )
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": content}])
        runtime.context.compact(root.id)
        assert "Still must verify checksum." in runtime.store.session(root.id).summary
        assert token_bound(runtime.store.session(root.id).summary, config.provider.model) <= 600
    finally:
        await runtime.shutdown()


async def test_model_compaction_budgets_oversized_valid_response(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.summary_tokens = 1000
    config.context.recent_blocks = 1
    requirement = "Still must verify checksum ZEBRA-937 before release."
    summary = {
        "established_facts": [f"Descriptive background {i}" for i in range(2000)],
        "unresolved_requirements": [requirement],
    }
    provider = ScriptedProvider({"root": [ModelResponse(text=encode(summary))]})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    root = runtime.create("Preserve pending work", tmp_path, config=config)
    try:
        for body in ["background detail\n" * 2500 + requirement, "Recent context stays verbatim."]:
            eid = runtime.store.event(root.id, "observation", {})
            runtime.store.add_context(root.id, eid, [{"role": "user", "content": body}])
        await runtime.semantic_compact(root.id, force=True)
        assert requirement in provider.requests[0].messages[-1]["content"]
        assert requirement in runtime.store.session(root.id).summary
        assert token_bound(runtime.store.session(root.id).summary, config.provider.model) <= 1000
        assert (
            runtime.store.events(root.id, kind="context_compaction")[-1]["payload"]["method"]
            == "model_structured"
        )
    finally:
        await runtime.shutdown()


async def test_refinement_hierarchical_reduction_keeps_logical_identity(tmp_path, config):
    config.provider.model = "gpt-6-astra"
    config.context.max_tokens = 4096
    calls = []

    class Reducer:
        async def invoke(self, request, emit):
            body = json.loads(request.messages[-1]["content"])
            calls.append(body)
            findings = ["EARLY-LESSON", "LATE-LESSON"]
            if body["level"] == 0:
                findings += [
                    f"Routine extracted description {i}: same supporting observation"
                    for i in range(150)
                ]
            return ModelResponse(
                text=encode({"findings": findings}), usage=Usage(input_tokens=30, output_tokens=10)
            )

    runtime = Runtime(tmp_path / "state", providers={"mock": Reducer()})
    root = runtime.create("Learn from one record", tmp_path, config=config)
    try:
        eid = runtime.store.event(root.id, "python_result", {"stdout": "evidence " * 15000})
        record = runtime.store.event_by_id(eid)
        archive = runtime.artifacts.put(root.id, record)
        reduced = await runtime.reduce_refinement_record(
            root.id, record, budget=600, archive=archive, position=0
        )
        assert {body["level"] for body in calls} >= {0, 1}
        assert reduced["id"] == eid and reduced["logical_record"] is True
        assert all(lesson in encode(reduced) for lesson in ("EARLY-LESSON", "LATE-LESSON"))
        assert runtime.store.usage(root.id).model_calls == len(calls)
    finally:
        await runtime.shutdown()
