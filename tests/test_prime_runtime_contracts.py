"""Runtime contracts exercised through real turns, tools and durable SQLite state."""

import asyncio
import json

import pytest

from threadweave.host_api import Request, dispatch
from threadweave.models import Action, HarnessError, ModelResponse, StateEdit, Usage, new_id
from threadweave.runtime import Runtime
from threadweave.storage import Store, encode
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider


async def observe(runtime, parent, child, **options):
    event = runtime.store.event(parent, "test_observe", {})
    return await dispatch(
        ToolContext(runtime, parent, new_id(), event),
        Request(
            operation="agent_observe.recent",
            payload={"target": child, "limit": 50, "max_chars": 2000, **options},
        ),
    )


@pytest.mark.parametrize(
    "adapter,isolate",
    [("coding", False), ("coding", True), ("workspace", False), ("workspace", True)],
)
async def test_child_inherits_capabilities_independently_of_workspace(
    tmp_path, repository, python_config, adapter, isolate
):
    python_config.task.adapter = adapter
    python_config.models["active"] = python_config.provider.model_copy(
        update={"model": "parent-active", "parameters": {"reasoning_effort": "high"}}
    )
    python_config.routing.policy = "role_based"
    python_config.routing.roles["agent"] = "active"
    python_config.tool_allowlist = ["ipython", "workspace_read", "host_request", "rlm.run"]
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        parent = runtime.create("parent", repository, config=python_config)
        child = runtime.spawn(parent.id, "work", isolate=isolate)
        config = runtime.store.config(child.id)
        assert config.task.adapter == adapter
        assert (child.workspace.path != parent.workspace.path) == isolate
        assert config.provider.model == "parent-active"
        assert config.provider.parameters["reasoning_effort"] == "high"
        assert config.tool_allowlist == python_config.tool_allowlist
        assert config.permissions == python_config.permissions
        assert config.execution == python_config.execution
        assert config.features == python_config.features
        assert child.depth == 1 and child.parent_id == parent.id
        assert runtime.tools.schemas(config) == runtime.tools.schemas(
            runtime.store.config(parent.id)
        )
        explicit = runtime.spawn(
            parent.id,
            "override",
            isolate=False,
            adapter="workspace",
            model="other-model",
            thinking="low",
        )
        override = runtime.store.config(explicit.id)
        assert override.task.adapter == "workspace"
        assert override.provider.model == "other-model"
        assert override.provider.parameters["reasoning_effort"] == "low"
        assert (
            runtime.store.config(parent.id).models["active"].parameters["reasoning_effort"]
            == "high"
        )
    finally:
        await runtime.shutdown()


async def test_public_rlm_inherits_effective_route_and_accepts_explicit_options(
    tmp_path, python_config
):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        parent = runtime.create("parent", tmp_path, config=python_config)
        eid = runtime.store.event(parent.id, "test", {})
        context = ToolContext(runtime, parent.id, new_id(), eid)
        result = await runtime.execute_python(
            context,
            'child = await rlm("work", thinking="low", isolate=False)\nprint(child.session_id)',
        )
        assert not result.get("error"), result
        children = [s for s in runtime.store.sessions() if s.parent_id == parent.id]
        assert len(children) == 1
        assert runtime.store.config(children[0].id).provider.parameters["reasoning_effort"] == "low"
    finally:
        await runtime.shutdown()


async def test_child_trajectory_without_mailbox_survives_compaction_and_restart(tmp_path, config):
    (tmp_path / "source.txt").write_text("SOURCE-EVIDENCE")
    provider = ScriptedProvider(
        {
            "child": [
                ModelResponse(
                    text="INTERMEDIATE-WORK",
                    actions=[Action(name="workspace_read", arguments={"path": "source.txt"})],
                ),
                response("finish", result="child finished"),
            ]
        }
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        parent = runtime.create("parent", tmp_path, config=config)
        child = runtime.spawn(parent.id, "inspect source", name="child", isolate=False)
        await runtime._run_turn(child.id)
        records = (await observe(runtime, parent.id, child.id))["messages"]
        assert runtime.store.session(child.id).outcome == "active"
        assert not runtime.store.messages(parent.id)
        assert "INTERMEDIATE-WORK" in encode(records) and "SOURCE-EVIDENCE" in encode(records)
        assert {"assistant", "tool", "tool-result"} <= {r["role"] for r in records}
        assert [r["seq"] for r in records] == sorted(r["seq"] for r in records)
        runtime.store.send(parent.id, child.id, "MAILBOX-ONLY-UNDELIVERED")
        assert "MAILBOX-ONLY-UNDELIVERED" not in encode(
            (await observe(runtime, parent.id, child.id))["messages"]
        )
        assert runtime.store.messages(child.id)[0]["body"] == "MAILBOX-ONLY-UNDELIVERED"
        runtime.context.compact(child.id, summary="Inspected source; SOURCE-EVIDENCE established.")
        assert any(
            r["role"] == "summary"
            for r in (await observe(runtime, parent.id, child.id))["messages"]
        )
        await runtime._run_turn(child.id)
        assert runtime.store.session(child.id).outcome == "completed"
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        records = (await observe(runtime, parent.id, child.id))["messages"]
        assert "INTERMEDIATE-WORK" in encode(records) and "SOURCE-EVIDENCE" in encode(records)
        assert any(r["role"] == "summary" for r in records)
        assert "child finished" in encode(records)
    finally:
        await runtime.shutdown()


async def test_observation_hides_uncommitted_response_and_tools(tmp_path, config):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = ScriptedProvider(
        {
            "child": [
                ModelResponse(
                    text="HALF-WRITTEN",
                    actions=[Action(name="workspace_read", arguments={"path": "source.txt"})],
                )
            ]
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    (tmp_path / "source.txt").write_text("HALF-TOOL")
    parent = runtime.create("parent", tmp_path, config=config)
    child = runtime.spawn(parent.id, "work", name="child", isolate=False)
    execute = runtime._execute_action

    async def gated(*args, **kwargs):
        result = await execute(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    runtime._execute_action = gated
    task = asyncio.create_task(runtime._run_turn(child.id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        snapshot = encode((await observe(runtime, parent.id, child.id))["messages"])
        assert "HALF-WRITTEN" not in snapshot and "HALF-TOOL" not in snapshot
        release.set()
        await task
        snapshot = encode((await observe(runtime, parent.id, child.id))["messages"])
        assert "HALF-WRITTEN" in snapshot and "HALF-TOOL" in snapshot
    finally:
        release.set()
        await task
        await runtime.shutdown()


async def test_request_graph_spawn_return_retries_usage_and_restart(tmp_path, config):
    (tmp_path / "source.txt").write_text("source")
    attempts = []

    async def retry(request):
        attempts.append(request.request_id)
        assert (
            runtime.store.db.execute(
                "SELECT status FROM model_requests WHERE id=?", (request.request_id,)
            ).fetchone()[0]
            == "running"
        )
        if len(attempts) == 1:
            raise HarnessError("provider", "transient", "retry", retryable=True)
        return ModelResponse(
            text="child intermediate",
            actions=[Action(name="workspace_read", arguments={"path": "source.txt"})],
            usage=Usage(input_tokens=7, output_tokens=3),
        )

    provider = ScriptedProvider(
        {
            "root": [
                response("workspace_read", path="source.txt"),
                response("agent_spawn", instruction="work", name="child", isolate=False),
                response("finish", result="done"),
            ],
            "child": [retry, response("finish", result="child result")],
        }
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    try:
        parent = runtime.create("parent", tmp_path, config=config)
        await runtime._run_turn(parent.id)
        await runtime._run_turn(parent.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == parent.id)
        await runtime._run_turn(child.id)
        await runtime._run_turn(child.id)
        await runtime._run_turn(parent.id)
        a, b, c, retry_c, d, e = provider.requests
        assert c.request_id == retry_c.request_id and attempts[0] == attempts[1]
        assert child.spawned_by_request_id == b.request_id
        graph = runtime.store.request_graph(parent.id)
        edges = {(r["source"], r["target"], r["kind"]) for r in graph["edges"]}
        assert {
            (a.request_id, b.request_id, "continuation"),
            (b.request_id, c.request_id, "subagent_call"),
            (c.request_id, d.request_id, "continuation"),
            (d.request_id, e.request_id, "subagent_return"),
            (b.request_id, e.request_id, "continuation"),
        } <= edges
        usage = runtime.store.request_usage(b.request_id, delegated=True)
        assert usage.model_calls == 4 and usage.retries == 1
        assert usage.input_tokens == c.input_token_bound + 47
        assert runtime.store.request_usage(c.request_id).estimated_calls == 1
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        assert runtime.store.request_graph(parent.id) == graph
        assert runtime.store.request_usage(b.request_id, delegated=True) == usage
    finally:
        await runtime.shutdown()


class ReviewingProvider:
    def __init__(self, *, decision=True, planner=None, review_gate=None):
        self.decision, self.planner, self.review_gate = decision, planner, review_gate
        self.requests = []

    async def invoke(self, request, emit):
        self.requests.append(request)
        if request.metadata.get("purpose") == "refinement_review":
            if self.review_gate:
                self.review_gate[0].set()
                await self.review_gate[1].wait()
            if self.decision == "error":
                raise ValueError("review failed")
            return ModelResponse(
                text=encode(
                    {
                        "shouldRefine": self.decision,
                        "rationale": "review rationale",
                        "instructions": "retain the discovered constraint",
                    }
                )
            )
        assert request.metadata["purpose"] == "refinement"
        evidence = json.loads(request.messages[-1]["content"])
        return ModelResponse(
            text=encode(
                {
                    "proposals": self.planner(evidence)
                    if self.planner
                    else [
                        {
                            "title": "lesson",
                            "content": {"text": "learned constraint"},
                            "source_events": [evidence["evidence"][0]["id"]],
                            "intended_effect": "improve later turns",
                        }
                    ]
                }
            )
        )


def seed(runtime, sid):
    event = runtime.store.event(sid, "python_result", {"stdout": "computed evidence"})
    runtime.store.add_context(
        sid, event, [{"role": "assistant", "content": "REASONING about discovery " * 2000}]
    )
    return event


@pytest.mark.parametrize(
    "decision,expected", [(False, "skipped"), (True, "applied"), ("error", "failed")]
)
async def test_reviewer_gates_planner_with_broad_trajectory_and_audit(
    tmp_path, config, decision, expected
):
    provider = ReviewingProvider(decision=decision)
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        rid = runtime.request_refinement(root.id, source="test")["request_id"]
        await runtime.auto_refine(root.id)
        result = runtime.store.refinement_request(root.id, rid)
        assert result["status"] == expected
        assert len(provider.requests) == (2 if decision is True else 1)
        review = json.loads(provider.requests[0].messages[-1]["content"])
        assert len(encode(review["trajectory"])) > 35000
        assert "REASONING about discovery" in encode(review["trajectory"])
        assert {
            "existing_state",
            "previous_refinements",
            "state_catalog",
            "trigger",
        } <= review.keys()
        if decision is True:
            planner = json.loads(provider.requests[1].messages[-1]["content"])
            assert len(encode(planner["trajectory"])) > len(encode(review["trajectory"]))
            assert planner["review"]["instructions"] == "retain the discovered constraint"
            assert runtime.store.states(root.id)
        else:
            assert not runtime.store.states(root.id)
        if decision != "error":
            assert (
                runtime.store.events(root.id, kind="refinement_review")[0]["payload"]["rationale"]
                == "review rationale"
            )
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("same_entry", [True, False])
async def test_host_baseline_conflicts_without_model_expected_version(tmp_path, config, same_entry):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = ReviewingProvider(review_gate=(entered, release))
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        event = seed(runtime, root.id)
        edit = StateEdit(
            title="existing",
            content={"text": "old"},
            source_events=[event],
            intended_effect="retain",
        )
        runtime.store.queue_refinement(root.id, edit)
        entry = runtime.store.apply_refinements(root.id)[0]
        provider.planner = lambda evidence: [
            {
                **edit.model_dump(),
                "entry_id": entry,
                "content": {"text": "planned"},
                "expected_version": None,
            }
        ]
        rid = runtime.request_refinement(root.id, source="test")["request_id"]
        task = asyncio.create_task(runtime.auto_refine(root.id))
        await asyncio.wait_for(entered.wait(), 5)
        other = Store(tmp_path / "state")
        try:
            other.queue_refinement(
                root.id,
                edit.model_copy(
                    update={
                        "entry_id": entry if same_entry else None,
                        "content": {"text": "concurrent"},
                    }
                ),
            )
            other.apply_refinements(root.id)
        finally:
            other.close()
        release.set()
        await task
        assert runtime.store.state(root.id, entry)["content"]["text"] == (
            "concurrent" if same_entry else "planned"
        )
        assert runtime.store.refinement_request(root.id, rid)["status"] == (
            "conflicted" if same_entry else "applied"
        )
        if same_entry:
            assert runtime.store.events(root.id, kind="refinement_conflict")
    finally:
        release.set()
        await runtime.shutdown()


async def test_child_completion_steers_active_parent_at_committed_boundary(tmp_path, config):
    entered, release = asyncio.Event(), asyncio.Event()

    async def in_flight(request):
        entered.set()
        await release.wait()
        return response("finish", result="provisional parent answer")

    provider = ScriptedProvider(
        {
            "root": [
                response("agent_spawn", instruction="work", name="child", isolate=False),
                in_flight,
                response("finish", result="incorporated child"),
            ],
            "child": [response("finish", result="USEFUL-CHILD-RESULT")],
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    task = None
    try:
        parent = runtime.create("parent", tmp_path, config=config)
        await runtime._run_turn(parent.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == parent.id)
        task = asyncio.create_task(runtime._run_turn(parent.id))
        await asyncio.wait_for(entered.wait(), 5)
        frozen = provider.requests[-1].model_dump_json()
        await runtime._run_turn(child.id)
        assert provider.requests[1].model_dump_json() == frozen
        assert "USEFUL-CHILD-RESULT" not in encode(runtime.store.session(parent.id).context)
        assert runtime.store.messages(parent.id, pending=True)
        release.set()
        await task
        assert runtime.store.session(parent.id).outcome == "active"
        await runtime._run_turn(parent.id)
        assert "USEFUL-CHILD-RESULT" in encode(provider.requests[-1].messages)
        assert runtime.store.session(parent.id).outcome == "completed"
    finally:
        release.set()
        if task:
            await task
        await runtime.shutdown()


async def test_background_completion_queues_result_and_idle_followup_waits(tmp_path, config):
    from threadweave.background import BackgroundProcesses

    config.permissions.append("process")
    entered, release = asyncio.Event(), asyncio.Event()
    (tmp_path / "source.txt").write_text("source")

    async def in_flight(request):
        entered.set()
        await release.wait()
        return response("workspace_read", path="source.txt")

    provider = ScriptedProvider(
        {
            "root": [
                in_flight,
                response("finish", result="first run done"),
                response("finish", result="followup done"),
            ]
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    task = None
    try:
        root = runtime.create("root", tmp_path, config=config)
        runtime.background = BackgroundProcesses(runtime)
        task = asyncio.create_task(runtime._run_turn(root.id))
        await asyncio.wait_for(entered.wait(), 5)
        eid = runtime.store.event(root.id, "test_background", {})
        handle = await runtime.background.start(
            ToolContext(runtime, root.id, new_id(), eid), "printf BACKGROUND-OUTPUT"
        )
        job = handle["id"]
        await runtime.background.tasks[job]
        runtime.message(None, root.id, "ORDINARY-FOLLOWUP", delivery="idle")
        assert "BACKGROUND-OUTPUT" not in encode(runtime.store.session(root.id).context)
        release.set()
        await task
        assert "ORDINARY-FOLLOWUP" not in encode(runtime.store.session(root.id).context)
        await runtime._run_turn(root.id)
        assert "Background process completed" in encode(provider.requests[-1].messages)
        assert "ORDINARY-FOLLOWUP" not in encode(provider.requests[-1].messages)
        await runtime._run_turn(root.id)
        assert "ORDINARY-FOLLOWUP" in encode(provider.requests[-1].messages)
    finally:
        release.set()
        if task:
            await task
        await runtime.shutdown()


async def test_compaction_request_edges_and_committed_summaries(tmp_path, config):
    class Provider:
        def __init__(self):
            self.requests = []

        async def invoke(self, request, emit):
            self.requests.append(request)
            if request.metadata.get("purpose") == "compaction":
                return ModelResponse(
                    text=encode(
                        {
                            "unresolved_requirements": ["Verify source"],
                            "established_facts": ["Read file"],
                        }
                    )
                )
            return response("workspace_read", path="source.txt")

    config.context.recent_blocks = 1
    (tmp_path / "source.txt").write_text("source")
    provider = Provider()
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("work", tmp_path, config=config)
        await runtime._run_turn(root.id)
        await runtime.semantic_compact(root.id, force=True)
        summary = provider.requests[-1]
        assert summary.metadata["purpose"] == "compaction"
        await runtime._run_turn(root.id)
        edges = runtime.store.request_graph(root.id)["edges"]
        assert {
            "source": summary.request_id,
            "target": provider.requests[-1].request_id,
            "kind": "compaction",
        } in edges
        assert any(r["role"] == "summary" for r in runtime.store.trajectory(root.id))
    finally:
        await runtime.shutdown()


async def test_planned_refinement_waits_for_owner_and_recovers_without_duplicate_apply(
    tmp_path, config
):
    provider = ReviewingProvider()
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    blocker = asyncio.create_task(asyncio.Event().wait())
    try:
        root = runtime.create("learn", tmp_path, config=config, mode="interactive")
        seed(runtime, root.id)
        runtime.tasks[root.id] = blocker
        rid = runtime.request_refinement(root.id, source="test")["request_id"]
        await runtime.auto_refine(root.id)
        assert runtime.store.refinement_request(root.id, rid)["status"] == "waiting_to_apply"
        assert not runtime.store.states(root.id)
        assert runtime.store.apply_refinements(root.id) == []
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        await runtime.start()
        await eventually(
            lambda: runtime.store.refinement_request(root.id, rid)["status"] == "applied"
        )
        await eventually(lambda: root.id not in runtime.tasks)
        entries = runtime.store.states(root.id)
        assert len(entries) == 1 and entries[0]["version"] == 1
        assert len(provider.requests) == 2
        runtime.apply_pending_refinements(root.id)
        assert runtime.store.states(root.id) == entries
    finally:
        blocker.cancel()
        await runtime.shutdown()


@pytest.mark.parametrize("phase", ["reviewing", "planning"])
async def test_restart_interrupted_refinement_retains_baseline_without_mutation(
    tmp_path, config, phase
):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("learn", tmp_path, config=config, mode="interactive")
        rid = runtime.request_refinement(root.id, source="test")["request_id"]
        runtime.store.db.execute(
            "INSERT INTO refinement_runs VALUES(?,?,?,?)",
            (rid, root.id, phase, encode({"baseline": {}, "trigger": "test"})),
        )
        runtime.store.refinement_request_result(root.id, rid, phase)
        await runtime.shutdown()
        runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
        await runtime.start()
        assert runtime.store.refinement_request(root.id, rid)["status"] == "failed"
        assert runtime.store.refinement_request(root.id, rid)["uncertain"]
        assert not runtime.store.states(root.id)
        assert (
            json.loads(
                runtime.store.db.execute(
                    "SELECT body FROM refinement_runs WHERE id=?", (rid,)
                ).fetchone()[0]
            )["baseline"]
            == {}
        )
    finally:
        await runtime.shutdown()


async def test_atomic_batch_rolls_back_all_state_when_one_edit_fails(tmp_path, config):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        event = seed(runtime, root.id)
        for content in [{"text": "valid"}, {"text": ""}]:
            runtime.store.queue_refinement(
                root.id,
                StateEdit(
                    title="test",
                    content=content,
                    source_events=[event],
                    intended_effect="test atomicity",
                ),
            )
        assert runtime.store.apply_refinements(root.id) == []
        assert runtime.store.states(root.id) == []
        assert runtime.store.events(root.id, kind="refinement") == []
    finally:
        await runtime.shutdown()


async def test_crash_during_atomic_apply_rolls_back_then_recovers_plan(tmp_path, config):
    import sys

    provider = ReviewingProvider()
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})
    blocker = asyncio.create_task(asyncio.Event().wait())
    try:
        root = runtime.create("learn", tmp_path, config=config, mode="interactive")
        seed(runtime, root.id)
        runtime.tasks[root.id] = blocker
        rid = runtime.request_refinement(root.id, source="test")["request_id"]
        await runtime.auto_refine(root.id)
        assert runtime.store.refinement_request(root.id, rid)["status"] == "waiting_to_apply"
        await runtime.shutdown()
        script = """
import os, sys
from threadweave.runtime import Runtime
runtime = Runtime(sys.argv[1])
apply = runtime.store._apply_edit
def crash(*args, **kwargs):
    apply(*args, **kwargs)
    os._exit(19)
runtime.store._apply_edit = crash
runtime.apply_pending_refinements(sys.argv[2])
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, str(data), root.id
        )
        assert await proc.wait() == 19
        runtime = Runtime(data, providers={"mock": provider})
        assert runtime.store.states(root.id) == []
        assert runtime.store.refinement_request(root.id, rid)["status"] == "waiting_to_apply"
        await runtime.start()
        await eventually(
            lambda: runtime.store.refinement_request(root.id, rid)["status"] == "applied"
        )
        assert runtime.store.states(root.id)[0]["version"] == 1
        assert len(runtime.store.events(root.id, kind="refinement")) == 1
        assert len(provider.requests) == 2
    finally:
        blocker.cancel()
        await runtime.shutdown()


async def test_chat_wire_retry_identity_is_durable_before_transmission(tmp_path, config):
    import httpx

    from threadweave.models import ProviderConfig
    from threadweave.providers import ChatProvider

    wire = []

    def handler(request):
        identifier = request.headers["Idempotency-Key"]
        assert request.headers["X-Client-Request-Id"] == identifier
        assert (
            runtime.store.db.execute(
                "SELECT status FROM model_requests WHERE id=?", (identifier,)
            ).fetchone()[0]
            == "running"
        )
        wire.append((identifier, request.content))
        if len(wire) == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    config.control_plane = "python"
    config.provider = ProviderConfig(
        name="chat",
        model="model",
        api_key_env="",
        base_url="https://provider.test/v1",
        streaming=False,
        max_output_tokens=128,
    )
    runtime = Runtime(
        tmp_path / "state", providers={"chat": ChatProvider(httpx.MockTransport(handler))}
    )
    try:
        root = runtime.create("work", tmp_path, config=config)
        await runtime._run_turn(root.id)
        assert len(wire) == 2 and wire[0] == wire[1]
        graph = runtime.store.request_graph(root.id)
        assert len(graph["requests"]) == 1
        assert [a["status"] for a in graph["requests"][0]["attempts"]] == ["failed", "completed"]
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("operation", ["delete", "rollback"])
async def test_stale_deletes_and_rollbacks_conflict_without_expected_version(
    tmp_path, config, operation
):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        event = seed(runtime, root.id)
        edit = StateEdit(
            title="lesson", content={"text": "old"}, source_events=[event], intended_effect="learn"
        )
        runtime.store.queue_refinement(root.id, edit)
        entry = runtime.store.apply_refinements(root.id)[0]
        baseline = {entry: runtime.store.state(root.id, entry)}
        runtime.store.queue_refinement(
            root.id, edit.model_copy(update={"entry_id": entry, "content": {"text": "new"}})
        )
        runtime.store.apply_refinements(root.id)
        runtime.store.queue_refinement(
            root.id,
            edit.model_copy(
                update={"entry_id": entry, "operation": operation, "rollback_version": 1}
            ),
            baseline=baseline,
        )
        assert runtime.store.apply_refinements(root.id) == []
        assert runtime.store.state(root.id, entry)["content"] == {"text": "new"}
        assert runtime.store.events(root.id, kind="refinement_conflict")
    finally:
        await runtime.shutdown()


async def test_shared_coding_child_preserves_original_baseline_after_parent_edits(
    tmp_path, repository, coding_config
):
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    try:
        parent = runtime.create("fix arithmetic", repository, config=coding_config)
        await runtime.environment.prepare(parent.id)
        original = runtime.store.db.execute(
            "SELECT body FROM coding_baselines WHERE session_id=?", (parent.id,)
        ).fetchone()[0]
        (repository / "mathops.py").write_text(
            "def add(a, b):\n    return a + b\n\ndef twice(value):\n    return add(value, value)\n"
        )
        child = runtime.spawn(parent.id, "inspect and verify current fix", isolate=False)
        await runtime.environment.prepare(child.id)
        inherited = runtime.store.db.execute(
            "SELECT body FROM coding_baselines WHERE session_id=?", (child.id,)
        ).fetchone()[0]
        assert inherited == original
        assert runtime.store.config(child.id).task.require_verifier
        assert runtime.store.config(child.id).task.protect_tests
        assert runtime.store.events(child.id, kind="coding_baseline_inherited")
        event = runtime.store.event(child.id, "test_verify", {})
        verification, error = await runtime._verify(child.id, event)
        assert not error and verification.passed
    finally:
        await runtime.shutdown()


async def test_failed_review_can_be_retried_on_same_evidence_with_fresh_request(tmp_path, config):
    provider = ReviewingProvider(decision="error")
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("learn", tmp_path, config=config)
        seed(runtime, root.id)
        failed = runtime.request_refinement(root.id, source="test")["request_id"]
        await runtime.auto_refine(root.id)
        assert runtime.store.refinement_request(root.id, failed)["status"] == "failed"
        provider.decision = True
        retry = runtime.request_refinement(root.id, source="retry")["request_id"]
        await runtime.auto_refine(root.id)
        assert retry != failed
        assert runtime.store.refinement_request(root.id, retry)["status"] == "applied"
        assert len(runtime.store.states(root.id)) == 1
    finally:
        await runtime.shutdown()


async def test_parent_effective_role_and_child_model_override_reach_provider(tmp_path, config):
    config.models["special"] = config.provider.model_copy(
        update={"model": "role-model", "parameters": {"reasoning_effort": "high"}}
    )
    config.routing.policy = "role_based"
    config.routing.roles["review"] = "special"
    provider = ScriptedProvider({})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("parent", tmp_path, config=config)
        runtime.store.update(root.id, role="review")
        child = runtime.spawn(root.id, "child", isolate=False)
        await runtime._run_turn(child.id)
        assert provider.requests[-1].config.model == "role-model"
        assert provider.requests[-1].config.parameters["reasoning_effort"] == "high"
        override = runtime.spawn(
            root.id,
            "explicit child",
            isolate=False,
            role="review",
            model="override-model",
            thinking="low",
        )
        await runtime._run_turn(override.id)
        assert provider.requests[-1].config.model == "override-model"
        assert provider.requests[-1].config.parameters["reasoning_effort"] == "low"
    finally:
        await runtime.shutdown()


async def test_failed_child_with_committed_work_returns_causal_error_edge(tmp_path, config):
    (tmp_path / "source.txt").write_text("source")
    provider = ScriptedProvider(
        {
            "root": [
                response("agent_spawn", instruction="work", name="child", isolate=False),
                response("finish", result="handled failure"),
            ],
            "child": [
                response("workspace_read", path="source.txt"),
                HarnessError("provider", "permanent", "failed"),
            ],
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("root", tmp_path, config=config)
        await runtime._run_turn(root.id)
        child = next(s for s in runtime.store.sessions() if s.parent_id == root.id)
        await runtime._run_turn(child.id)
        committed = provider.requests[-1].request_id
        await runtime._run_turn(child.id)
        assert runtime.store.session(child.id).outcome == "failed"
        await runtime._run_turn(root.id)
        assert {
            "source": committed,
            "target": provider.requests[-1].request_id,
            "kind": "subagent_return",
        } in runtime.store.request_graph(root.id)["edges"]
        assert "failed (provider)" in encode(provider.requests[-1].messages)
    finally:
        await runtime.shutdown()


async def test_custom_allowed_tool_executes_in_child_and_disallowed_tool_stays_hidden(
    tmp_path, config
):
    from threadweave.models import Record
    from threadweave.tools import Tool

    class Arguments(Record):
        value: str

    async def lookup(context, arguments):
        return {"value": arguments.value, "session": context.session_id}

    config.tool_allowlist = ["custom_lookup", "finish"]
    provider = ScriptedProvider({"child": [response("custom_lookup", value="custom result")]})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    runtime.tools.register(
        Tool("custom_lookup", "custom lookup", Arguments, lookup, ("workspace.read",))
    )
    try:
        root = runtime.create("parent", tmp_path, config=config)
        child = runtime.spawn(root.id, "work", name="child", isolate=False)
        await runtime._run_turn(child.id)
        assert {t["function"]["name"] for t in provider.requests[0].tools} == {
            "custom_lookup",
            "finish",
        }
        assert "custom result" in encode(runtime.store.trajectory(child.id))
        assert not runtime.store.events(child.id, kind="failure")
    finally:
        await runtime.shutdown()


async def test_evaluation_judge_uses_durable_runtime_retry_path(tmp_path, config, monkeypatch):
    from threadweave.evals import harness

    class Judge:
        def __init__(self):
            self.requests = []

        async def invoke(self, request, emit):
            self.requests.append(request)
            assert request.messages == [{"role": "user", "content": "Official judge prompt"}]
            assert request.tools == []
            if len(self.requests) == 1:
                raise HarnessError("provider", "transient", "retry", retryable=True)
            return ModelResponse(text="judgment", usage=Usage(input_tokens=4, output_tokens=2))

    judge = Judge()
    monkeypatch.setattr(harness, "default_providers", lambda: {"mock": judge})
    usages = []
    result = await harness.invoke_judge(
        config.provider,
        {"prompt": "Official judge prompt", "max_tokens": 128, "temperature": 0},
        tmp_path,
        usages,
    )
    assert result == "judgment"
    assert len(judge.requests) == 2
    assert judge.requests[0].request_id == judge.requests[1].request_id
    history = Store(next((tmp_path / "judge-state").iterdir()))
    try:
        graph = history.request_graph(judge.requests[0].session_id)
        assert len(graph["requests"]) == 1
        assert len(graph["requests"][0]["attempts"]) == 2
        assert sum(u.model_calls for u in usages) == 2
        assert sum(u.retries for u in usages) == 1
    finally:
        history.close()


async def test_child_tool_snapshot_does_not_expand_when_another_session_loads_tools(
    tmp_path, config
):
    from threadweave.models import Record
    from threadweave.tools import Tool

    async def custom(context, arguments):
        return {"value": "available at admission"}

    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        runtime.tools.register(Tool("existing_custom", "existing", Record, custom))
        root = runtime.create("parent", tmp_path, config=config)
        child = runtime.spawn(root.id, "child", isolate=False)
        runtime.tools.register(Tool("later_custom", "loaded later", Record, custom))
        child_config = runtime.store.config(child.id)
        assert runtime.tools.allowed("existing_custom", child_config)
        assert not runtime.tools.allowed("later_custom", child_config)
        assert runtime.tools.allowed("later_custom", runtime.store.config(root.id))
        event = runtime.store.event(child.id, "test_tool", {})
        result = await runtime._execute_action(
            child.id, new_id(), Action(name="later_custom"), event
        )
        assert result["error"]["code"] == "permission_denied"
        grandchild = runtime.spawn(child.id, "nested", isolate=False)
        assert not runtime.tools.allowed("later_custom", runtime.store.config(grandchild.id))
    finally:
        await runtime.shutdown()


async def test_idle_followup_is_in_first_request_when_session_is_already_idle(tmp_path, config):
    provider = ScriptedProvider(
        {"root": [response("finish", result="done"), response("finish", result="done again")]}
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("root", tmp_path, config=config, mode="interactive")
        runtime.interact(root.id, "initial task")
        await runtime._run_turn(root.id)
        assert not runtime.store.session(root.id).runnable
        runtime.message(None, root.id, "IDLE-FOLLOWUP", delivery="idle")
        await runtime._run_turn(root.id)
        assert len(provider.requests) == 2
        assert "IDLE-FOLLOWUP" in encode(provider.requests[-1].messages)
        assert not runtime.store.session(root.id).runnable
    finally:
        await runtime.shutdown()
