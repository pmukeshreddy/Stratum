"""Refinement contracts through runtime triggers, model calls and durable state."""

import asyncio
import json

import httpx
import pytest

from threadweave.context import token_bound
from threadweave.models import HarnessError, ModelResponse, Outcome, ProviderConfig, StateEdit
from threadweave.providers import ChatProvider
from threadweave.runtime import Runtime
from threadweave.state_retrieval import relevant_state, state_overview
from threadweave.storage import Store, encode
from threadweave.subscription import SubscriptionProvider

from .conftest import eventually, response
from .test_runtime_contracts import ReviewingProvider, seed


def add_state(runtime, sid, source, *, kind="memory", title="existing", content=None):
    runtime.store.queue_refinement(
        sid,
        StateEdit(
            kind=kind,
            title=title,
            content=content or {"text": title},
            source_events=[source],
            intended_effect="retain tested knowledge",
        ),
    )
    return runtime.store.apply_refinements(sid)[0]


async def test_manual_refine_bypasses_declining_reviewer_with_full_context(tmp_path, config):
    provider = ReviewingProvider(decision=False)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config, mode="interactive")
        source = seed(runtime, root.id)
        existing = add_state(runtime, root.id, source, title="learn discovery")
        rid = runtime.interact(root.id, "/refine")
        await runtime.auto_refine(root.id)
        assert [r.metadata["purpose"] for r in provider.requests] == ["refinement"]
        value = json.loads(provider.requests[0].messages[-1]["content"])
        assert len(encode(value["trajectory"])) > 40000
        assert value["review"] is None and value["trigger"] == "manual"
        assert existing in {e["id"] for e in value["existing_state"]}
        assert value["previous_refinements"] and value["scope_policy"]
        assert not runtime.store.events(root.id, kind="refinement_review")
        assert runtime.store.refinement_request(root.id, rid)["status"] == "applied", (
            runtime.store.refinement_request(root.id, rid)
        )
        run = json.loads(
            runtime.store.db.execute(
                "SELECT body FROM refinement_runs WHERE id=?", (rid,)
            ).fetchone()[0]
        )
        assert run["trigger"] == "manual" and existing in run["baseline"]
        assert runtime.store.events(root.id, kind="manual_refinement")
    finally:
        await runtime.shutdown()


async def test_manual_batch_conflict_rejects_entire_plan(tmp_path, config):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = ReviewingProvider(decision=False, planner_gate=(entered, release))
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    task = None
    try:
        root = runtime.create("learn", tmp_path, config=config)
        source = seed(runtime, root.id)
        ids = [add_state(runtime, root.id, source, title=title) for title in ("one", "two")]
        provider.planner = lambda evidence: [
            dict(
                entry_id=entry,
                title="updated",
                content={"text": "planned"},
                source_events=[source],
                intended_effect="improve",
            )
            for entry in ids
        ]
        rid = runtime.interact(root.id, "/refine")
        task = asyncio.create_task(runtime.auto_refine(root.id))
        await asyncio.wait_for(entered.wait(), 5)
        other = Store(tmp_path / "state")
        try:
            other.queue_refinement(
                root.id,
                StateEdit(
                    entry_id=ids[1],
                    title="concurrent",
                    content={"text": "newer"},
                    source_events=[source],
                    intended_effect="preserve newer work",
                ),
            )
            other.apply_refinements(root.id)
        finally:
            other.close()
        release.set()
        await task
        assert runtime.store.refinement_request(root.id, rid)["status"] == "conflicted"
        assert runtime.store.state(root.id, ids[0])["version"] == 1
        assert runtime.store.state(root.id, ids[1])["content"]["text"] == "newer"
        assert [r.metadata["purpose"] for r in provider.requests] == ["refinement"]
    finally:
        release.set()
        if task:
            await task
        await runtime.shutdown()


class CompactionProvider(ReviewingProvider):
    def __init__(self, *, failed=False, **kwargs):
        super().__init__(**kwargs)
        self.failed = failed

    async def invoke(self, request, emit):
        if request.metadata.get("purpose") == "compaction":
            self.requests.append(request)
            if self.failed == "cancel":
                raise asyncio.CancelledError()
            if self.failed:
                raise ValueError("summary failed")
            return ModelResponse(
                text=encode(
                    {
                        "unresolved_requirements": [
                            "COMPACTION-CONTEXT retain established constraint"
                        ],
                        "established_facts": ["Computed reliable evidence"],
                    }
                )
            )
        return await super().invoke(request, emit)


@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("restart", [False, "queued", "claimed"])
async def test_compaction_checkpoint_review_once_after_safe_boundary(
    tmp_path, config, decision, restart
):
    config.refinement.automatic = True
    config.limits.wall_seconds = 10000
    provider = CompactionProvider(decision=decision)
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config, mode="interactive")
        seed(runtime, root.id)
        await runtime.semantic_compact(root.id, force=True)
        compaction = runtime.store.events(root.id, kind="context_compaction")[0]
        pending = runtime.store.pending_refinement_requests(root.id)
        assert len(pending) == 1 and pending[0]["trigger"] == "compaction"
        rid = pending[0]["id"]
        assert len(provider.requests) == 1  # Summarization did not run review inline.
        runtime._transitioning.add(root.id)
        await runtime.auto_refine(root.id)
        assert len(provider.requests) == 1
        runtime._transitioning.remove(root.id)
        if restart:
            if restart == "claimed":
                runtime.store.db.execute(
                    "UPDATE refinement_requests SET status='running' WHERE id=?", (rid,)
                )
            await runtime.shutdown()
            runtime = Runtime(data, providers={"mock": provider})
        await runtime.start()  # The production scheduler consumes the persisted checkpoint.
        await eventually(
            lambda: (
                runtime.store.refinement_request(root.id, rid)["status"] in {"applied", "skipped"}
            )
        )
        for _ in range(3):
            await runtime.auto_refine(root.id)
        purposes = [r.metadata["purpose"] for r in provider.requests]
        assert purposes.count("refinement_review") == 1
        assert purposes.count("refinement") == int(decision)
        review = next(r for r in provider.requests if r.metadata["purpose"] == "refinement_review")
        value = json.loads(review.messages[-1]["content"])
        assert value["trigger"] == "compaction" and value["checkpoint"]["id"] == compaction["id"]
        assert "COMPACTION-CONTEXT" in encode(value)
        assert {
            "trajectory",
            "state_overview",
            "existing_state",
            "previous_refinements",
        } <= value.keys()
        assert bool(runtime.store.states(root.id)) == decision
        assert runtime.store.session(root.id).turns == 0
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("failure", [True, "cancel"])
async def test_failed_compaction_does_not_queue_review(tmp_path, config, failure):
    config.refinement.automatic = True
    provider = CompactionProvider(failed=failure)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await runtime.semantic_compact(root.id, force=True)
        else:
            await runtime.semantic_compact(root.id, force=True)
            assert runtime.store.events(root.id, kind="compaction_fallback")
        assert not runtime.store.pending_refinement_requests(root.id)
        await runtime.auto_refine(root.id)
        assert [r.metadata["purpose"] for r in provider.requests] == ["compaction"]
        assert root.id not in runtime._transitioning
    finally:
        await runtime.shutdown()


class GraphProvider(ReviewingProvider):
    async def invoke(self, request, emit):
        purpose = request.metadata.get("purpose", "agent")
        if purpose != "agent":
            if purpose == "refinement_review" and not any(
                r.metadata.get("purpose") == purpose for r in self.requests
            ):
                self.requests.append(request)
                raise HarnessError("provider", "transient", "retry auxiliary", retryable=True)
            return await super().invoke(request, emit)
        self.requests.append(request)
        if request.name == "child":
            return response("finish", result="child evidence")
        if (
            sum(
                r.metadata.get("purpose", "agent") == "agent" and r.name == "root"
                for r in self.requests
            )
            == 1
        ):
            return response("agent_spawn", instruction="work", name="child", isolate=False)
        return response("finish", result="parent done")


async def test_auxiliary_accounting_does_not_break_delegation_graph_and_recovers(tmp_path, config):
    config.limits.wall_seconds = 10000
    provider = GraphProvider()
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        await runtime._run_turn(root.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == root.id)
        await runtime._run_turn(child.id)
        seed(runtime, root.id)
        await runtime._refinement_pass(root.id, "completion")
        await runtime._run_turn(root.id)
        trajectory = [r for r in provider.requests if r.request_kind == "trajectory"]
        a, c, b = trajectory
        graph = runtime.store.request_graph(root.id)
        assert {r["id"] for r in graph["requests"]} == {a.request_id, c.request_id, b.request_id}
        assert {(e["source"], e["target"], e["kind"]) for e in graph["edges"]} == {
            (a.request_id, c.request_id, "subagent_call"),
            (c.request_id, b.request_id, "subagent_return"),
            (a.request_id, b.request_id, "continuation"),
        }
        auxiliary = runtime.store.request_history(root.id, kind="auxiliary")
        assert {r["purpose"] for r in auxiliary} == {"refinement_review", "refinement"}
        assert all(r["inbound"] == [] for r in auxiliary)
        review = next(r for r in auxiliary if r["purpose"] == "refinement_review")
        assert len(review["attempts"]) == 2
        attempts = [
            r for r in provider.requests if r.metadata.get("purpose") == "refinement_review"
        ]
        assert attempts[0].request_id == attempts[1].request_id
        assert runtime.store.usage(root.id, tree=True).model_calls == 6
        assert runtime.store.request_usage(review["id"]).model_calls == 2
        history = runtime.store.request_history(root.id)
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        assert runtime.store.request_history(root.id) == history
        assert runtime.store.request_graph(root.id) == graph
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("kind", ["memory", "prompt_note", "skill", "subagent_spec"])
async def test_overview_exposes_existing_entry_outside_top_relevance(tmp_path, config, kind):
    provider = ReviewingProvider(decision=False)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("frequent retrieval keyword", tmp_path, config=config)
        source = runtime.store.event(
            root.id, "python_result", {"stdout": "observed reusable evidence"}
        )
        for index in range(15):
            add_state(runtime, root.id, source, title=f"frequent retrieval keyword {index}")
        content = {
            "memory": {"text": "UNIQUE-LESSON deduplicate this entry"},
            "prompt_note": {"text": "UNIQUE-LESSON deduplicate this entry"},
            "skill": {
                "name": "UNIQUE-LESSON",
                "description": "reusable procedure",
                "code": "answer = 42",
            },
            "subagent_spec": {"instruction": "UNIQUE-LESSON retain protocol", "name": "reviewer"},
        }[kind]
        target = add_state(runtime, root.id, source, kind=kind, title="obscure", content=content)
        assert target not in {e["id"] for e in relevant_state(runtime.store, root.id, limit=12)}

        def plan(value):
            item = next(
                e for e in value["state_overview"]["entries"] if "UNIQUE-LESSON" in encode(e)
            )
            assert item["id"] == target and item["kind"] == kind and item["version"] == 1
            assert item["scope"] == "session" and "metadata" in item
            return [
                {
                    "entry_id": item["id"],
                    "kind": kind,
                    "title": "updated existing",
                    "content": content,
                    "source_events": [source],
                    "intended_effect": "remove overlap",
                }
            ]

        provider.planner = plan
        rid = runtime.interact(root.id, "/refine")
        await runtime.auto_refine(root.id)
        assert runtime.store.refinement_request(root.id, rid)["status"] == "applied", (
            runtime.store.refinement_request(root.id, rid)
        )
        assert len(runtime.store.states(root.id)) == 16
        assert runtime.store.state(root.id, target)["version"] == 2
    finally:
        await runtime.shutdown()


def test_overview_truncation_is_bounded_balanced_and_deterministic():
    entries = [
        dict(
            id=f"{kind}-{index:03}",
            kind=kind,
            version=3,
            title="Title",
            owner_id="owner",
            content={"text": "preview " * 200, "metadata": {"tags": "x" * 2000}},
            created_at=index,
        )
        for kind in ("memory", "prompt_note", "skill", "subagent_spec")
        for index in range(45)
    ]
    large = state_overview(entries, token_budget=200000)
    assert len(large["entries"]) == 160
    assert all(value == 5 for value in large["omitted"].values())
    assert all(len(e["content_preview"]) <= 240 for e in large["entries"])
    small = state_overview(entries, token_budget=4096)
    assert small == state_overview(list(reversed(entries)), token_budget=4096)
    assert token_bound(small) <= 4096
    assert {e["kind"] for e in small["entries"]} == {
        "memory",
        "prompt_note",
        "skill",
        "subagent_spec",
    }
    assert 0 < len(small["entries"]) < 160


async def test_structured_refinement_disables_thinking_on_wire_only(tmp_path, config):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        review = "shouldRefine" in body["messages"][0]["content"]
        content = (
            encode({"shouldRefine": True, "rationale": "useful"}) if review else '{"proposals": []}'
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
        await runtime._refinement_pass(root.id, "completion")
        assert len(bodies) == 2
        assert all(body["reasoning_effort"] == "none" for body in bodies)
        assert runtime.store.config(root.id).provider.parameters["reasoning_effort"] == "high"
        await runtime._invoke(root.id)
        assert bodies[-1]["reasoning_effort"] == "high"
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize(
    "supported,expected", [(["none", "low", "high"], "none"), (["low", "high"], "low")]
)
async def test_subscription_refinement_uses_least_supported_effort(supported, expected):
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
    low, _ = await provider.resolve(config, reasoning_off=True)
    assert low.parameters["reasoning_effort"] == expected
    unchanged, _ = await provider.resolve(config)
    assert unchanged.parameters["reasoning_effort"] == "high"


async def test_schema_11_migration_repairs_auxiliary_chains_without_losing_usage(tmp_path, config):
    from .fakes import ScriptedProvider

    provider = ScriptedProvider(
        {"root": [ModelResponse(text="agent A"), ModelResponse(text="agent B")]}
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        await runtime._run_turn(root.id)
        review_provider = ReviewingProvider()
        runtime.providers["mock"] = review_provider
        seed(runtime, root.id)
        await runtime._refinement_pass(root.id, "manual")
        # Two auxiliary calls reproduce the buggy v11 chain, including an old reviewer.
        await runtime.auxiliary(root.id, "refinement_review", "review", {})
        runtime.providers["mock"] = provider
        runtime.store.update(root.id, runnable=True)
        await runtime._run_turn(root.id)
        a, b = [r.request_id for r in provider.requests]
        planner, review = [r.request_id for r in review_provider.requests]
        usage = runtime.store.usage(root.id)
        db = runtime.store.db
        db.execute("DELETE FROM request_edges")
        for source, target in ((a, planner), (planner, review), (review, b)):
            db.execute("INSERT INTO request_edges VALUES(?,?,?)", (source, target, "continuation"))
            db.execute(
                "UPDATE model_requests SET inbound=? WHERE id=?",
                (encode([{"source": source, "kind": "continuation"}]), target),
            )
        db.execute("ALTER TABLE model_requests DROP COLUMN request_kind")
        db.execute("ALTER TABLE refinement_requests DROP COLUMN trigger")
        db.execute("DELETE FROM schema_migrations WHERE version=12")
        db.execute("PRAGMA user_version=11")
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        assert runtime.store.db.execute("PRAGMA user_version").fetchone()[0] == 12
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


async def test_extractive_compaction_on_completed_session_schedules_control_only_review(
    tmp_path, config
):
    config.refinement.automatic = True
    config.limits.wall_seconds = 10000
    provider = ReviewingProvider(decision=False)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        runtime.store.finish(root.id, Outcome.COMPLETED, "done")
        compaction = runtime.context.compact(root.id)
        rid = runtime.store.pending_refinement_requests(root.id)[0]["id"]
        await runtime.start()
        await eventually(
            lambda: runtime.store.refinement_request(root.id, rid)["status"] == "skipped"
        )
        assert runtime.store.session(root.id).outcome == "completed"
        assert [r.metadata["purpose"] for r in provider.requests] == ["refinement_review"]
        assert (
            json.loads(provider.requests[0].messages[-1]["content"])["checkpoint"]["id"]
            == compaction
        )
    finally:
        await runtime.shutdown()


async def test_compaction_and_review_checkpoint_commit_atomically(tmp_path, config, monkeypatch):
    config.refinement.automatic = True
    runtime = Runtime(tmp_path / "state", providers={"mock": ReviewingProvider()})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        previous = runtime.store.session(root.id)
        enqueue = runtime.store.enqueue_refinement_request

        def interrupted(*args, **kwargs):
            enqueue(*args, **kwargs)
            raise RuntimeError("interrupted commit")

        monkeypatch.setattr(runtime.store, "enqueue_refinement_request", interrupted)
        with pytest.raises(RuntimeError, match="interrupted commit"):
            runtime.context.compact(root.id)
        restored = runtime.store.session(root.id)
        assert restored.context == previous.context and restored.summary == previous.summary
        assert not runtime.store.events(root.id, kind="context_compaction")
        assert not runtime.store.pending_refinement_requests(root.id)
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("provider_name", ["mock", "codex_subscription"])
async def test_pinned_evaluation_allows_only_structured_auxiliary_reasoning_change(
    tmp_path, config, provider_name
):
    from threadweave.evals.harness import MatchedProvider, discard

    class Provider(ReviewingProvider):
        async def resolve(self, candidate, *, reasoning_off=False):
            return candidate.model_copy(
                update={"parameters": {"reasoning_effort": "low"}}
            ) if reasoning_off else candidate, {}

        async def invoke(self, request, emit):
            if request.request_kind == "auxiliary":
                return await super().invoke(request, emit)
            self.requests.append(request)
            return response("finish", result="done")

    config.provider = ProviderConfig(
        name=provider_name, model="m", parameters={"reasoning_effort": "high"}
    )
    provider = Provider(planner=lambda _: [])
    measured = MatchedProvider(provider, config.provider, tmp_path / "calls.jsonl")
    runtime = Runtime(tmp_path / "state", providers={provider_name: measured})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        await runtime._refinement_pass(root.id, "completion")
        assert len(provider.requests) == 2
        assert all(r.request_kind == "auxiliary" for r in provider.requests)
        assert all(
            r.config.parameters["reasoning_effort"]
            == ("none" if provider_name == "mock" else "low")
            for r in provider.requests
        )
        await runtime._run_turn(root.id)
        assert runtime.store.config(root.id).provider.parameters["reasoning_effort"] == "high"
        assert provider.requests[-1].config.parameters["reasoning_effort"] == "high"
        wrong = provider.requests[-1].model_copy(update={"config": provider.requests[0].config})
        with pytest.raises(HarnessError, match="settings changed"):
            await measured.invoke(wrong, discard)
    finally:
        await runtime.shutdown()
