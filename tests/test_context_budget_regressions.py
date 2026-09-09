"""Regressions reproduced against the pre-scaling production implementation."""

import json

import pytest

from threadweave.context import token_bound
from threadweave.models import HarnessError, ModelResponse
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
