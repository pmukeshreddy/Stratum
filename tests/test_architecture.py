"""Every diagram connection: real workers/storage, with test-only model responses."""

import json
import sys
from collections import deque

import pytest

from threadweave.chat import Chat, chat_config
from threadweave.coding_config import update_coding_options
from threadweave.context import FOUNDATION
from threadweave.daemon import Daemon
from threadweave.guardrails import observe
from threadweave.models import ModelResponse, StateEdit
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider
from .test_chat import InputTerminal


async def settle(runtime, sid, minimum=0):
    await eventually(
        lambda: (
            runtime.store.session(sid).turns >= minimum
            and not runtime.store.session(sid).runnable
            and sid not in runtime.tasks
        )
    )


async def test_agents_view_greeting_in_non_repository_has_no_coding_admission(tmp_path, config):
    workspace = tmp_path / "ordinary-files"
    workspace.mkdir()
    daemon = Daemon(tmp_path / "state")
    terminal = InputTerminal(tmp_path / "view")

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    view = Chat(tmp_path / "state", workspace, terminal, rpc=rpc)
    daemon.runtime.providers["mock"] = ScriptedProvider({"root": [ModelResponse(text="Hello!")]})
    try:
        assert await view.open(config=config)
        root = view.session["id"]
        await daemon.runtime.start()
        await view.submit("hi")
        await settle(daemon.runtime, root, 1)
        await view.poll()
        for command in ("/status", "/state", "/states", "/tree", "/usage", "/history"):
            assert await view.submit(command)
        assert "Hello!" in terminal.output.getvalue()
        assert "Opening environment" in terminal.output.getvalue()
        assert "baseline" not in terminal.output.getvalue()
        assert "L1" in terminal.output.getvalue() and "L3" in terminal.output.getvalue()
        assert daemon.runtime.store.usage(root).model_calls == 1
        assert daemon.runtime.store.usage(root).tool_calls == 0
        assert not daemon.runtime.store.events(root, kind="coding_baseline")
        assert not daemon.runtime.store.events(root, kind="coding_command")
        assert not (workspace / ".git").exists()
        await rpc(None, "chat_detach", session_id=root)
        assert daemon.runtime.store.session(root).outcome == "active"
    finally:
        await daemon.runtime.shutdown()
        daemon.lock.close()


async def test_explicit_coding_greeting_does_not_run_baseline(runtime, repository, coding_config):
    runtime.providers["mock"] = ScriptedProvider({"root": [ModelResponse(text="Hi.")]})
    coding_config.provider.name = "mock"
    root = runtime.create("Conversation", repository, config=coding_config, mode="interactive")
    await runtime.start()
    runtime.interact(root.id, "hi")
    await settle(runtime, root.id, 1)
    assert not runtime.store.events(root.id, kind="coding_baseline")
    assert not runtime.store.events(root.id, kind="coding_command")


async def test_environment_tools_without_coding_task(runtime, repository, config):
    config.capabilities = ["coding"]
    config.permissions.append("process")
    update_coding_options(
        config.task, test_commands=[[sys.executable, "-m", "unittest", "discover", "-s", "tests"]]
    )
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                response("repo_search", query="def add"),
                response("run_tests"),
                response("finish", result="Observed real test evidence"),
            ]
        }
    )
    root = runtime.create("Inspect environment", repository, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == "completed"
    assert runtime.store.events(root.id, kind="coding_command")
    assert not runtime.store.events(root.id, kind="coding_baseline")
    assert not runtime.store.events(root.id, kind="failure")


async def test_root_rlm_messages_layers_refinement_detach_recovery(tmp_path, config):
    workspace = tmp_path / "world"
    workspace.mkdir()
    data = tmp_path / "state"
    config.features.model_compaction = False
    pending = deque()

    def root_reply(request):
        return pending.popleft() if pending else ModelResponse(text="Received in the same session.")

    def child_python(request):
        return response(
            "python",
            code=(
                f"child_value = {request.name!r}\n"
                "assert 'x' not in globals()\nimport time\n"
                "deadline = time.monotonic() + 5\n"
                "while not (workspace / 'root-continued').exists() and time.monotonic() < deadline:\n"
                "    time.sleep(0.01)\n"
                "assert (workspace / 'root-continued').exists(), 'Parent waited for child answer'\n"
                "tools.call('workspace_write', path=child_value + '.txt', content=child_value)\n"
                "time.sleep(0.15)\nchild_value"
            ),
        )

    def report(request):
        return response(
            "agent_message", recipient_id=request.parent_id, body=f"Result from {request.name}"
        )

    provider = ScriptedProvider(
        {
            "root": [root_reply] * 50,
            "left": [child_python, report, response("finish", result="left complete")],
            "right": [child_python, report, response("finish", result="right complete")],
        },
        delay=0.04,
    )
    runtime = Runtime(data, providers={"mock": provider})

    async def say(body, actions=()):
        previous = runtime.store.session(root.id).turns
        pending.extend([*actions, ModelResponse(text="Boundary reached.")])
        runtime.interact(root.id, body)
        await settle(runtime, root.id, previous + len(actions) + 1)

    try:
        root = runtime.create("Persistent session", workspace, config=config, mode="interactive")
        await runtime.start()
        await say("hi")
        await say(
            "Remember x",
            [response("python", code="x = 123\nlarge_value = 'L2_ONLY_SECRET' * 10000")],
        )
        assert "L2_ONLY_SECRET" * 100 not in json.dumps(provider.requests[-1].messages)
        await say(
            "Create children and continue locally",
            [
                response(
                    "python",
                    code=(
                        "assert x == 123\nleft = await rlm('Left work', name='left')\n"
                        "right = await rlm('Right work', name='right')\n"
                        "assert left.session_id != right.session_id\n"
                        "tools.call('workspace_write', path='root-continued', content='continued immediately')\nprint(x)"
                    ),
                )
            ],
        )
        children = [s for s in runtime.store.sessions(root_id=root.id) if s.parent_id]
        assert len(children) == 2
        for child in children:
            assert (await runtime.wait(child.id)).outcome == "completed"
        await settle(runtime, root.id)
        assert provider.peak_active >= 2
        assert len({s.kernel_id for s in [root, *children]}) == 3
        assert all(
            (workspace / f"{child.name}.txt").read_text() == child.name for child in children
        )
        assert {m["body"] for m in runtime.store.messages(root.id)} >= {
            "Result from left",
            "Result from right",
        }
        assert all(m["received_at"] for m in runtime.store.messages(root.id))
        evidence = runtime.store.events(root.id, kind="python_result")[0]["id"]
        edit = StateEdit(
            kind="skill",
            title="Inspect retained x",
            content={
                "name": "inspect_x",
                "description": "Select a scalar without printing large values",
                "code": "print(x)",
                "required_permissions": ["python"],
            },
            source_events=[evidence],
            intended_effect="Reuse bounded selection of persistent state",
        )
        runtime.store.queue_refinement(root.id, edit)
        assert not runtime.store.states(root.id)
        await say("Apply at boundary")
        entry = runtime.store.states(root.id)[0]
        assert entry["version"] == 1
        assert entry["content"]["validation_status"] == "syntax_and_permissions_validated"
        runtime.store.queue_refinement(
            root.id,
            edit.model_copy(
                update={
                    "entry_id": entry["id"],
                    "expected_version": 1,
                    "content": {**edit.content, "code": "assert x == 123\nprint(x)"},
                }
            ),
        )
        await say("Apply skill update")
        assert runtime.store.state(root.id, entry["id"])["version"] == 2
        runtime.context.compact(root.id, count=len(runtime.store.session(root.id).context))
        assert runtime.store.event_by_id(evidence)["type"] == "python_result"
        assert runtime.context.messages(root.id)[0] == {"role": "system", "content": FOUNDATION}
        info = runtime.information(root.id)
        assert info["L1"]["blocks"] == 0 and "x" in info["L2"]["checkpointed_names"]
        assert len(info["L2"]["children"]) == 2 and info["L3"]["compactions"] > 0
        assert "L2_ONLY_SECRET" not in json.dumps(info)
        await runtime.pause(root.id)
        runtime.message(children[0].id, root.id, "Queued while detached/inactive")
        schedule = runtime.schedule(
            root.id, interval_seconds=3600, instruction="Persistent heartbeat"
        )
        usage = runtime.store.usage(root.id, tree=True)
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        await runtime.start()
        assert runtime.store.session(root.id).kernel_id == root.kernel_id
        assert {s.id for s in runtime.store.sessions(root_id=root.id)} == {
            root.id,
            *[s.id for s in children],
        }
        assert (
            runtime.store.messages(root.id, pending=True)[0]["body"]
            == "Queued while detached/inactive"
        )
        assert runtime.store.state(root.id, entry["id"])["version"] == 2
        assert runtime.store.db.execute(
            "SELECT enabled FROM schedules WHERE id=?", (schedule,)
        ).fetchone()[0]
        assert runtime.store.usage(root.id, tree=True).model_calls == usage.model_calls
        await say("Reattach and inspect x", [response("skill_run", entry_id=entry["id"])])
        assert runtime.store.events(root.id, kind="kernel_recovery")[-1]["payload"]["restored"]
        assert runtime.store.events(root.id, kind="skill_outcome")[-1]["payload"]["passed"]
        assert not runtime.store.events(root.id, kind="python_error", tree=True)
        assert (
            runtime.store.usage(root.id, tree=True).python_executions
            > runtime.store.usage(root.id).python_executions
        )
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("mode", ["autonomous", "goal", "heartbeat"])
async def test_agents_view_attach_preserves_execution_controls(tmp_path, config, mode):
    daemon = Daemon(tmp_path / "state")
    daemon.runtime.providers["mock"] = ScriptedProvider({})
    try:
        root = daemon.runtime.create("Persistent objective", tmp_path, config=config, mode=mode)
        await daemon.dispatch("chat_open", {"session_id": root.id})
        await daemon.dispatch("chat_detach", {"session_id": root.id})
        after = daemon.runtime.store.session(root.id)
        assert after.mode == mode and after.runnable == root.runnable
        assert after.lifecycle == root.lifecycle
        if mode == "goal":
            assert daemon.runtime.store.goal(root.id)["objective"] == "Persistent objective"
    finally:
        await daemon.runtime.shutdown()
        daemon.lock.close()


@pytest.mark.parametrize(
    "kind,content",
    [
        ("memory", {"text": "A fact"}),
        ("prompt_note", {"text": "An instruction"}),
        ("skill", {"name": "procedure", "description": "Compute", "code": "1 + 1"}),
        ("subagent_spec", {"instruction": "Investigate independently"}),
    ],
)
async def test_continual_categories_at_daemon_boundaries(tmp_path, config, kind, content):
    daemon = Daemon(tmp_path / "state")
    runtime = daemon.runtime
    runtime.providers["mock"] = ScriptedProvider({"root": [ModelResponse(text="Boundary")] * 10})
    try:
        root = runtime.create("Reusable state", tmp_path, config=config, mode="interactive")
        evidence = runtime.store.event(root.id, "observation", {"fact": "source"})
        edit = StateEdit(
            kind=kind, content=content, source_events=[evidence], intended_effect="Reuse evidence"
        )
        await runtime.start()
        for turn, operation in enumerate(["upsert", "delete", "rollback"], 1):
            if turn > 1:
                entry = runtime.store.states(root.id, include_deleted=True)[0]
                edit.entry_id = entry["id"]
                edit.expected_version = entry["version"]
            edit.operation = operation
            edit.rollback_version = 1 if operation == "rollback" else None
            await daemon.dispatch("refine", {"session_id": root.id, "edit": edit.model_dump()})
            runtime.interact(root.id, f"Apply queued change {turn}")
            await settle(runtime, root.id, turn)
            entry = (await daemon.dispatch("states", {"session_id": root.id}))[0]
            assert entry["version"] == turn and entry["deleted"] == (operation == "delete")
            assert entry["provenance"]["source_events"] == [evidence]
        assert runtime.store.state(root.id, entry["id"], 1)["version"] == 1
    finally:
        await runtime.shutdown()
        daemon.lock.close()


async def test_no_progress_accounts_for_l2_changes(runtime, tmp_path, config):
    root = runtime.create("Incremental computation", tmp_path, config=config, mode="interactive")
    event = runtime.store.event(root.id, "test", {})
    for number in range(15):
        await runtime.execute_python(
            ToolContext(runtime, root.id, f"execution-{number}", event), f"x = {number}"
        )
        assert not observe(runtime, root.id, "python", {"code": "x += 1"})
    assert not runtime.store.events(root.id, kind="no_progress")


async def test_unverified_finish_is_not_reported_as_verified(runtime, tmp_path, config):
    runtime.providers["mock"] = ScriptedProvider({"root": [response("finish", result="Done")]})
    root = runtime.create("Ordinary task", tmp_path, config=config, mode="interactive")
    await runtime.start()
    runtime.interact(root.id, "Complete this request")
    await settle(runtime, root.id, 1)
    assert (
        runtime.store.events(root.id, kind="conversation_completed")[-1]["payload"]["verified"]
        is False
    )


def test_explicit_environment_config_is_not_rewritten(tmp_path):
    path = tmp_path / "environment.json"
    path.write_text('{"task":{"adapter":"external_simulation"}}')
    assert chat_config(tmp_path, path).task.adapter == "external_simulation"


async def test_model_compaction_in_plain_environment_preserves_l2_l3(runtime, tmp_path, config):
    config.context.max_tokens = 18000
    config.context.recent_blocks = 1
    config.tool_allowlist = ["python", "rlm", "finish"]
    summary = ModelResponse(
        text=json.dumps(
            {
                "retained_repl_names": ["x"],
                "unresolved_work": ["Continue the objective"],
                "evidence_ids": [],
            }
        )
    )
    runtime.providers["mock"] = ScriptedProvider({"root": [summary]})
    root = runtime.create("Generic compaction", tmp_path, config=config, mode="interactive")
    child = runtime.spawn(root.id, "Independent objective")
    source = runtime.store.event(root.id, "observation", {})
    await runtime.execute_python(ToolContext(runtime, root.id, "initialize", source), "x = 123")
    events = []
    for i in range(8):
        eid = runtime.store.event(root.id, "observation", {"text": str(i) + "z" * 4000})
        runtime.store.add_context(root.id, eid, [{"role": "user", "content": "z" * 4000}])
        events.append(eid)
    worker = runtime.kernels[root.id].process
    await runtime.semantic_compact(root.id)
    compact = runtime.store.events(root.id, kind="context_compaction")[-1]
    assert compact["payload"]["method"] == "model_structured"
    assert compact["payload"]["model_response_event"]
    assert runtime.kernels[root.id].process is worker
    assert runtime.store.session(child.id).parent_id == root.id
    assert all(runtime.store.event_by_id(eid) for eid in events)
    result = await runtime.execute_python(ToolContext(runtime, root.id, "inspect", source), "x")
    assert result["value"] == "123" and not result["error"]


async def test_rlm_recurses_and_can_message_sibling(runtime, tmp_path, config):
    from threadweave.models import Action

    def nested(request):
        return response(
            "python",
            code=(
                "leaf = await rlm('Nested objective', name='leaf')\n"
                "sibling = next(s['id'] for s in tools.call('agent_sessions') if s['name'] == 'right')\n"
                "tools.call('agent_message', recipient_id=sibling, body='persistent sibling message')"
            ),
        )

    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    actions=[
                        Action(name="rlm", arguments={"instruction": "left", "name": "left"}),
                        Action(name="rlm", arguments={"instruction": "right", "name": "right"}),
                    ]
                ),
                response("agent_wait", seconds=0.2),
                response("finish", result="Done"),
            ],
            "left": [nested, response("finish", result="Nested work done")],
            "right": [response("agent_wait", seconds=0.2), response("finish", result="Received")],
            "leaf": [
                response("python", code="leaf_value = 7"),
                response("finish", result="Leaf done"),
            ],
        }
    )
    root = runtime.create("Recursive work", tmp_path, config=config, mode="goal")
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == "completed"
    sessions = {s.name: s for s in runtime.store.sessions(root_id=root.id)}
    assert sessions["leaf"].parent_id == sessions["left"].id
    assert runtime.store.usage(root.id, tree=True).subagent_count == 3
    assert any(
        m["body"] == "persistent sibling message"
        for m in runtime.store.messages(sessions["right"].id)
    )


async def test_rlm_respects_feature_permissions(runtime, tmp_path, config):
    config.features.subagents = False
    root = runtime.create("Disabled delegation", tmp_path, config=config)
    assert not runtime.tools.allowed("rlm", config)
    assert not runtime.tools.allowed("agent_spawn", config)
    with pytest.raises(PermissionError):
        runtime.spawn(root.id, "Not allowed")
