import asyncio
import sys
from datetime import UTC, datetime

import pytest

from threadweave.models import HarnessError, Outcome, StateEdit
from threadweave.runtime import LimitReached, Runtime
from threadweave.tools import Empty, Tool, ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider


async def test_process_tool_retains_large_output_and_command_verifier(runtime, tmp_path, config):
    config.permissions.append("process")
    config.task.verifier = "command"
    config.task.verifier_options = {"command": [sys.executable, "-c", "print('verified')"]}
    config.task.require_verifier = True
    config.task.verify_each_turn = False
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                response("process_run", command=[sys.executable, "-c", "print('x'*100000)"]),
                response("finish", result="Verified command"),
            ],
        }
    )
    root = runtime.create("Command", tmp_path, config=config)
    await runtime.start()
    result = await runtime.wait(root.id)
    assert result.outcome == Outcome.COMPLETED
    tool_event = runtime.store.events(root.id, kind="tool_result")[0]
    raw = runtime.artifacts.load(root.id, tool_event["payload"]["result"]["artifact_id"])
    assert len(raw["stdout"]) == 4000
    full = runtime.artifacts.load(root.id, raw["stdout_artifact"])
    assert len(full) == 100001
    assert runtime.store.usage(root.id).verifier_calls == 1


async def test_command_timeout_reaps_process_group(runtime, tmp_path, config):
    config.permissions.append("process")
    config.limits.tool_timeout_seconds = 0.2
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                response(
                    "process_run", command=[sys.executable, "-c", "import time; time.sleep(60)"]
                ),
                response("finish", result="Timeout handled"),
            ],
        }
    )
    root = runtime.create("Timeout", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    failure = runtime.store.events(root.id, kind="failure")[0]["payload"]
    assert failure["category"] == "tool" and failure["uncertain"]


async def test_symlink_escape_and_typed_validation(runtime, tmp_path):
    root = runtime.create("Workspace", tmp_path)
    event = runtime.store.event(root.id, "test", {})
    context = ToolContext(runtime, root.id, "action", event)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(HarnessError, match="within"):
        await runtime.tools.call(context, "workspace_read", {"path": "escape/anything"})
    with pytest.raises(HarnessError) as error:
        await runtime.tools.call(context, "workspace_write", {"path": "x", "content": 123})
    assert error.value.failure.code == "invalid_arguments"


async def test_artifact_retention_explicit_selection_and_tree_permissions(runtime, tmp_path):
    root = runtime.create("Root", tmp_path)
    child = runtime.spawn(root.id, "Child")
    other = runtime.create("Unrelated", tmp_path)
    value = {"records": list(range(10000))}
    exposed = runtime.artifacts.expose(root.id, value)
    assert exposed["truncated"] and len(exposed["preview"]) <= 1800
    assert runtime.artifacts.load(child.id, exposed["artifact_id"]) == value
    assert runtime.store.session(root.id).context == []
    with pytest.raises(PermissionError):
        runtime.artifacts.load(other.id, exposed["artifact_id"])
    page = runtime.artifacts.read(root.id, exposed["artifact_id"], limit=100)
    assert page["next_offset"] == 100 and page["total_bytes"] > 100
    branch = await runtime.fork(root.id)
    assert runtime.artifacts.load(branch.id, exposed["artifact_id"]) == value


async def test_model_artifact_read_tool_maps_public_id_to_storage(runtime, tmp_path):
    root = runtime.create("Inspect retained output", tmp_path)
    artifact = runtime.artifacts.put_bytes(root.id, b"abcdefgh", "text/plain")
    event = runtime.store.event(root.id, "test", {})
    result = await runtime.tools.call(
        ToolContext(runtime, root.id, "action", event),
        "artifact_read",
        {"artifact_id": artifact, "offset": 2, "limit": 3},
    )
    assert result["text"] == "cde"
    assert result["next_offset"] == 5


async def test_pause_keeps_messages_queued_until_resume(runtime, tmp_path, config):
    runtime.providers["mock"] = ScriptedProvider(
        {"root": [response("agent_wait", seconds=300), response("finish", result="Resumed")]}
    )
    root = runtime.create("Pause", tmp_path, config=config)
    await runtime.start()
    await eventually(
        lambda: runtime.store.session(root.id).turns == 1 and root.id not in runtime.tasks
    )
    await runtime.pause(root.id)
    runtime.message(None, root.id, "Queued while paused")
    await asyncio.sleep(0.15)
    assert runtime.store.session(root.id).paused
    assert runtime.store.session(root.id).turns == 1
    assert runtime.store.messages(root.id, pending=True)
    runtime.resume(root.id)
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED


async def test_cancelled_model_reservations_are_settled_before_same_process_resume(
    runtime, tmp_path, config
):
    entered = asyncio.Event()

    async def slow(request):
        entered.set()
        await asyncio.Event().wait()

    provider = ScriptedProvider({"root": [slow]})
    runtime.providers["mock"] = provider
    root = runtime.create("Cancel model", tmp_path, config=config)
    await runtime.start()
    await entered.wait()
    assert runtime.store.reserved(root.id)[0] > 0
    await runtime.pause(root.id)
    assert runtime.store.reserved(root.id) == (0, 0)
    assert runtime.store.usage(root.id).estimated_calls == 1
    provider.scripts["root"] = [response("finish", result="Resumed")]
    runtime.resume(root.id)
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED


async def test_cost_budget_and_subtree_limits_are_not_reset(runtime, tmp_path, config):
    config.provider.input_cost_per_million = 1
    config.provider.output_cost_per_million = 2
    config.limits.cost_budget = 0.00001
    runtime.providers["mock"] = ScriptedProvider({})
    root = runtime.create("Expensive", tmp_path, config=config)
    child = runtime.spawn(root.id, "Child")
    with pytest.raises(LimitReached, match="cost budget"):
        runtime._check_limits(child.id, resource="model_calls", input_bound=1000)


async def test_limited_partial_turn_can_fork_without_replaying_source_actions(
    runtime, tmp_path, config
):
    config.limits.max_python_executions = 1
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [response("python", code="value = 41"), response("python", code="value += 1")],
            "branch": [
                response("finish", result="unused"),
                response("finish", result="unused"),
                response("python", code="assert value == 41\nvalue"),
                response("finish", result="Recovered branch"),
            ],
        }
    )
    root = runtime.create("Fork partial", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.LIMITED
    assert runtime.store.session(root.id).pending_turn
    branch = await runtime.fork(root.id, name="branch")
    assert (await runtime.wait(branch.id)).outcome == Outcome.COMPLETED
    assert runtime.store.session(root.id).outcome == Outcome.LIMITED
    assert runtime.store.usage(branch.id).python_executions == 1


async def test_global_refinement_read_update_and_delete(runtime, tmp_path):
    from .fakes import TestConfig as RunConfig

    config = RunConfig(refinement={"allow_global_writes": True})
    root = runtime.create("Global", tmp_path, config=config)
    other = runtime.create("Consumer", tmp_path)
    evidence = runtime.store.event(root.id, "observation", {})
    edit = StateEdit(
        kind="memory",
        scope="global",
        content={"text": "Reusable fact"},
        source_events=[evidence],
        intended_effect="Share across tasks",
    )
    runtime.store.queue_refinement(root.id, edit)
    entry = runtime.store.apply_refinements(root.id)[0]
    assert runtime.store.state(other.id, entry)["owner_id"] is None
    other_event = runtime.store.event(other.id, "observation", {})
    with pytest.raises(PermissionError):
        runtime.store.queue_refinement(
            other.id,
            edit.model_copy(
                update={
                    "entry_id": entry,
                    "scope": "session",
                    "source_events": [other_event],
                    "operation": "delete",
                }
            ),
        )


async def test_extension_tools_are_typed_and_registered_once(runtime, tmp_path):
    async def tool(context, arguments):
        return {"session": context.session_id}

    definition = Tool("custom", "Custom tool", Empty, tool)
    runtime.tools.register(definition)
    with pytest.raises(ValueError, match="already registered"):
        runtime.tools.register(definition)
    root = runtime.create("Extension", tmp_path)
    event = runtime.store.event(root.id, "test", {})
    result = await runtime.tools.call(ToolContext(runtime, root.id, "action", event), "custom", {})
    assert result["session"] == root.id


async def test_cron_uses_utc_and_validates_inputs(runtime, tmp_path):
    root = runtime.create("Cron", tmp_path)
    timestamp = datetime(2026, 9, 5, 12, 1, tzinfo=UTC).timestamp()
    result = runtime._next_schedule(None, "*/5 * * * *", timestamp)
    assert datetime.fromtimestamp(result, UTC).minute == 5
    with pytest.raises(ValueError):
        runtime.schedule(root.id, cron="bad cron")
    with pytest.raises(ValueError):
        runtime.schedule(root.id, cron="* * * * *", interval_seconds=1)
    schedule = runtime.schedule(root.id, cron="*/5 * * * *")
    assert (
        runtime.store.db.execute("SELECT cron FROM schedules WHERE id=?", (schedule,)).fetchone()[0]
        == "*/5 * * * *"
    )


async def test_embedded_runtime_has_single_owner(runtime):
    with pytest.raises(RuntimeError, match="already owns"):
        Runtime(runtime.store.directory)


async def test_fork_children_can_retrieve_ancestral_artifacts_and_history(runtime, tmp_path):
    root = runtime.create("Original", tmp_path)
    artifact = runtime.artifacts.put(root.id, {"evidence": "retained"})
    event = runtime.store.event(root.id, "observation", {"fact": 42})
    branch = await runtime.fork(root.id)
    child = runtime.spawn(branch.id, "Inspect ancestry")
    context = ToolContext(runtime, child.id, "read", event)
    result = await runtime.tools.call(
        context, "history_read", {"session_id": root.id, "kind": "observation"}
    )
    assert result[0]["payload"] == {"fact": 42}
    assert runtime.artifacts.load(child.id, artifact) == {"evidence": "retained"}
    tree_events = runtime.store.events(child.id, tree=True)
    assert any(e["session_id"] == branch.id for e in tree_events)


async def test_interrupted_turn_time_is_recovered_once(runtime, tmp_path):
    from threadweave.models import Lifecycle, now

    root = runtime.create("Interrupted timing", tmp_path)
    runtime.store.transition(root.id, Lifecycle.RUNNING, running_since=now() - 2)
    await runtime.recover()
    charged = runtime.store.usage(root.id).wall_seconds
    assert charged >= 2
    assert runtime.store.session(root.id).running_since is None
    await runtime.recover()
    assert runtime.store.usage(root.id).wall_seconds == charged


@pytest.mark.parametrize("mode", ["autonomous", "heartbeat"])
async def test_input_arriving_during_model_turn_is_not_lost_to_wait(
    runtime, tmp_path, config, mode
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def model_wait(request):
        entered.set()
        await release.wait()
        return response("agent_wait", seconds=300)

    runtime.providers["mock"] = ScriptedProvider(
        {"root": [model_wait, response("finish", result="Received input")]}
    )
    root = runtime.create("Concurrent input", tmp_path, config=config, mode=mode)
    await runtime.start()
    await entered.wait()
    runtime.message(None, root.id, "New information during the invocation")
    release.set()
    assert (await runtime.wait(root.id, timeout=5)).outcome == Outcome.COMPLETED
    assert runtime.store.messages(root.id)[0]["received_at"]
