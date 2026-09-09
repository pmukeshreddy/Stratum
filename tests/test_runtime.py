import json

import pytest

from threadweave.models import (
    Action,
    HarnessError,
    Lifecycle,
    ModelResponse,
    Outcome,
)
from threadweave.runtime import BudgetBusy, LimitReached, Runtime

from .conftest import eventually, response
from .fakes import ScriptedProvider


async def test_model_action_result_and_repl_roundtrip(runtime, tmp_path, config):
    provider = ScriptedProvider(
        {
            "root": [
                response("python", code="values = list(range(1000))\nlen(values)"),
                response("python", code="sum(values)"),
                response("finish", result="499500"),
            ]
        }
    )
    runtime.providers["mock"] = provider
    root = runtime.create("Compute", tmp_path, config=config)
    await runtime.start()
    result = await runtime.wait(root.id)
    assert result.outcome == Outcome.COMPLETED
    assert result.lifecycle == Lifecycle.INACTIVE
    assert "1000" in str(provider.requests[1].messages)
    assert "499500" in str(provider.requests[2].messages)
    assert len(runtime.store.events(root.id, kind="python_execution")) == 2
    assert runtime.store.usage(root.id).python_executions == 2
    assert not runtime.kernels


async def test_recursive_concurrent_sessions_sibling_messages_and_failure_isolation(
    runtime, tmp_path, config
):
    config.limits.concurrency = 4
    root_actions = ModelResponse(
        actions=[
            Action(name="agent_spawn", arguments={"instruction": "Left", "name": "left"}),
            Action(name="agent_spawn", arguments={"instruction": "Right", "name": "right"}),
        ]
    )
    left_code = """
peers = tools.call('agent_sessions')
right = next(p['id'] for p in peers if p['name'] == 'right')
tools.call('agent_message', recipient_id=right, body='hello sibling')
grandchild = tools.call('agent_spawn', instruction='Nested work', name='leaf')
"""
    provider = ScriptedProvider(
        {
            "root": [
                root_actions,
                response("agent_wait", seconds=0.4),
                response("finish", result="Root success"),
            ],
            "left": [response("python", code=left_code), response("finish", result="Left success")],
            "right": [
                response("agent_wait", seconds=0.2),
                response("finish", result="Right success"),
            ],
            "leaf": [HarnessError("provider", "deliberate", "A failed child is isolated")],
        },
        delay=0.05,
    )
    runtime.providers["mock"] = provider
    root = runtime.create("Coordinate", tmp_path, config=config, mode="goal")
    await runtime.start()
    result = await runtime.wait(root.id)
    assert result.outcome == Outcome.COMPLETED
    assert provider.peak_active >= 2
    sessions = {s.name: s for s in runtime.store.sessions(root_id=root.id)}
    assert sessions["leaf"].parent_id == sessions["left"].id
    assert sessions["leaf"].outcome == Outcome.FAILED
    assert sessions["right"].outcome == Outcome.COMPLETED
    assert any(m["body"] == "hello sibling" for m in runtime.store.messages(sessions["right"].id))
    assert len({s.kernel_id for s in sessions.values()}) == 4
    usage = runtime.store.usage(root.id, tree=True)
    assert usage.subagent_count == 3 and usage.model_calls >= 8
    assert runtime.store.goal(root.id)["status"] == "completed"


async def test_provider_retry_and_accounting(runtime, tmp_path, config):
    attempts = 0

    def flaky(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HarnessError("provider", "http_429", "Rate limit", retryable=True)
        return response("finish", result="Recovered")

    runtime.providers["mock"] = ScriptedProvider({"root": [flaky]})
    root = runtime.create("Retry", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    usage = runtime.store.usage(root.id)
    assert usage.retries == 2 and usage.model_calls == 3 and usage.estimated_calls == 2
    assert usage.input_tokens > 20
    assert len(runtime.store.events(root.id, kind="retry")) == 2


async def test_bad_tool_and_python_error_return_to_model(runtime, tmp_path, config):
    provider = ScriptedProvider(
        {
            "root": [
                response("nonexistent"),
                response("python", code="raise ValueError('bad input')"),
                response("workspace_read", path="../outside"),
                response("finish", result="Handled"),
            ]
        }
    )
    runtime.providers["mock"] = provider
    root = runtime.create("Errors", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    assert "unknown_tool" in str(provider.requests[1].messages)
    assert "ValueError" in str(provider.requests[2].messages)
    assert "PermissionError" in str(provider.requests[3].messages)


@pytest.mark.parametrize(
    "limit,value,expected",
    [
        ("max_turns", 1, "turn limit"),
        ("max_tool_calls", 1, "tool_calls"),
        ("max_python_executions", 1, "python_executions"),
        ("max_model_calls", 1, "model_calls"),
        ("token_budget", 1, "token budget"),
    ],
)
async def test_tree_limits_stop_execution(runtime, tmp_path, config, limit, value, expected):
    setattr(config.limits, limit, value)
    runtime.providers["mock"] = ScriptedProvider(
        {"root": [response("python", code="x=1"), response("python", code="x+=1")]}
    )
    root = runtime.create("Bounded", tmp_path, config=config)
    await runtime.start()
    result = await runtime.wait(root.id)
    assert result.outcome == Outcome.LIMITED
    assert expected in result.result
    with pytest.raises(ValueError, match="cannot reset"):
        runtime.resume(root.id)


async def test_concurrency_limit_and_resource_reservations(runtime, tmp_path, config):
    config.limits.concurrency = 1
    runtime.providers["mock"] = ScriptedProvider(
        {"*": [response("agent_wait", seconds=0.2), response("finish", result="done")]}, delay=0.05
    )
    root = runtime.create("Root", tmp_path, config=config)
    runtime.spawn(root.id, "Child", name="child")
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    assert runtime.providers["mock"].peak_active == 1
    assert runtime.store.reserved(root.id) == (0, 0)


async def test_budget_reservations_wait_without_false_tree_termination(runtime, tmp_path, config):
    config.limits.token_budget = 1000
    runtime.providers["mock"] = ScriptedProvider({})
    root = runtime.create("Root", tmp_path, config=config)
    eid = runtime.store.event(root.id, "model_invocation_started", {})
    runtime.store.db.execute(
        "INSERT INTO reservations VALUES(?,?,?,?,?)", (eid, root.id, 800, 100, 0)
    )
    with pytest.raises(BudgetBusy):
        runtime._check_limits(root.id, resource="model_calls", input_bound=100)
    runtime._check_limits(root.id)  # Reserved tokens are not spent tokens.
    runtime.store.db.execute("DELETE FROM reservations")
    runtime._check_limits(root.id, resource="model_calls", input_bound=100)
    with pytest.raises(LimitReached):
        runtime._check_limits(root.id, resource="model_calls", input_bound=1000)


async def test_wall_limit_cancels_busy_worker_and_tree(runtime, tmp_path, config):
    config.limits.wall_seconds = 0.4
    runtime.providers["mock"] = ScriptedProvider(
        {"*": [response("python", code="while True: pass")]}
    )
    root = runtime.create("Bounded", tmp_path, config=config)
    child = runtime.spawn(root.id, "Busy", name="child")
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.LIMITED
    assert (await runtime.wait(child.id)).outcome == Outcome.LIMITED
    assert not runtime.kernels


async def test_permission_and_subagent_depth_limits(runtime, tmp_path, config):
    config.limits.max_depth = 1
    config.limits.max_subagents = 2
    config.permissions.remove("python")
    runtime.providers["mock"] = ScriptedProvider(
        {"root": [response("python", code="x=1"), response("finish", result="done")]}
    )
    root = runtime.create("Root", tmp_path, config=config)
    child = runtime.spawn(root.id, "Child")
    with pytest.raises(HarnessError, match="depth limit"):
        runtime.spawn(child.id, "Too deep")
    runtime.spawn(root.id, "Second")
    with pytest.raises(HarnessError, match="subagent limit"):
        runtime.spawn(root.id, "Third")
    unrelated = runtime.create("Other", tmp_path, config=config)
    with pytest.raises(HarnessError, match="parent, child"):
        runtime.message(child.id, unrelated.id, "denied")
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    assert "permission_denied" in str(
        runtime.providers["mock"].requests[-1].messages
    ) or runtime.store.events(root.id, kind="failure")
    assert runtime.store.usage(root.id).python_executions == 0


async def test_verifier_failure_is_not_a_task_failure_and_completion_gate(
    runtime, tmp_path, config
):
    from threadweave.models import Verification

    class Adapter:
        calls = 0

        async def prepare(self, context, task):
            return {"ready": True}

        async def verify(self, context, task):
            self.calls += 1
            if self.calls == 1:
                raise OSError("Verifier service unavailable")
            return Verification(passed=self.calls >= 3, details="x" * 100000)

    config.task.adapter = "custom"
    config.task.verifier = "file"
    config.task.require_verifier = True
    runtime.adapters["custom"] = Adapter()
    provider = ScriptedProvider(
        {"root": [response("finish", result="Premature"), response("finish", result="Verified")]}
    )
    runtime.providers["mock"] = provider
    root = runtime.create("Verify", tmp_path, config=config)
    await runtime.start()
    completed = await runtime.wait(root.id)
    assert completed.result == "Verified" and completed.turns == 2
    assert runtime.store.usage(root.id).verifier_calls == 3
    assert runtime.store.usage(root.id).retries == 1
    failure = runtime.store.events(root.id, kind="failure")[0]
    assert failure["payload"]["category"] == "verifier"
    assert "x" * 5000 not in str(provider.requests[-1].messages)
    assert "full_verifier_artifact" in str(provider.requests[-1].messages)
    receipts = runtime.store.events(root.id, kind="verifier_result")
    exposed = receipts[0]["payload"]["result"]
    detail = (
        runtime.artifacts.load(root.id, exposed["artifact_id"])
        if "artifact_id" in exposed
        else json.loads(exposed["preview"])
    )
    assert (
        runtime.artifacts.load(root.id, detail["full_verifier_artifact"])["details"] == "x" * 100000
    )
    assert "x" * 100000 not in str(provider.requests[-1].messages)


async def test_refinement_applied_at_next_turn_then_skill_executes(runtime, tmp_path, config):
    def refine(request):
        event = runtime.store.events(request.session_id, kind="python_result", limit=1)[0]["id"]
        return response(
            "refine",
            edit={
                "kind": "skill",
                "title": "Compute",
                "content": {
                    "name": "compute",
                    "description": "Reuse calculation",
                    "code": "answer = x * 2\nanswer",
                },
                "source_events": [event],
                "intended_effect": "Reuse calculation",
                "select": True,
            },
        )

    def skill(request):
        entry = runtime.store.states(request.session_id)[0]
        supplemental = next(
            m["content"]
            for m in request.messages
            if (m.get("content") or "").startswith("Selected supplemental state:")
        )
        assert "answer = x * 2" in supplemental
        return response("skill_run", entry_id=entry["id"])

    runtime.providers["mock"] = ScriptedProvider(
        {"root": [response("python", code="x=21"), refine, skill, response("finish", result="42")]}
    )
    root = runtime.create("Refine", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    assert len(runtime.store.events(root.id, kind="refinement")) == 1
    assert runtime.store.usage(root.id).python_executions == 2


async def test_heartbeat_persistence_and_missed_ticks_coalesce(tmp_path, config):
    config.provider.name = "mock"
    directory = tmp_path / "data"
    runtime = Runtime(
        directory,
        providers={
            "mock": ScriptedProvider(
                {"*": [response("python", code="x=1"), response("finish", result="tick")]}
            )
        },
    )
    root = runtime.create("Scheduled", tmp_path, config=config, mode="heartbeat")
    runtime.schedule(root.id, interval_seconds=60)
    await runtime.start()
    await eventually(
        lambda: runtime.store.session(root.id).turns == 1 and root.id not in runtime.tasks
    )
    assert not runtime.store.session(root.id).runnable
    runtime.store.db.execute("UPDATE schedules SET next_at=0")
    await runtime.shutdown()
    restored = Runtime(directory, providers={"mock": ScriptedProvider({})})
    try:
        await restored.start()
        assert (await restored.wait(root.id)).outcome == Outcome.COMPLETED
        assert len(restored.store.events(root.id, kind="heartbeat")) == 1
        assert restored.store.session(root.id).kernel_id == root.kernel_id
    finally:
        await restored.shutdown()


async def test_graceful_restart_restores_context_messages_goal_and_python(tmp_path, config):
    scripts = {
        "root": [
            response("python", code="retained = 123"),
            response("agent_wait", seconds=300),
            response("python", code="assert retained == 123\nretained + 1"),
            response("finish", result="Restored"),
        ]
    }
    directory = tmp_path / "data"
    runtime = Runtime(directory, providers={"mock": ScriptedProvider(scripts)})
    root = runtime.create("Persist objective", tmp_path, config=config, mode="goal")
    await runtime.start()
    await eventually(
        lambda: runtime.store.session(root.id).turns == 2 and root.id not in runtime.tasks
    )
    runtime.context.compact(root.id)
    runtime.store.send(None, root.id, "Continue after restart")
    before_history = runtime.store.events(root.id, limit=500)
    await runtime.shutdown()
    restored = Runtime(directory, providers={"mock": ScriptedProvider(scripts)})
    try:
        restored.resume(root.id)
        await restored.start()
        result = await restored.wait(root.id)
        assert result.outcome == Outcome.COMPLETED and result.id == root.id
        assert result.kernel_id == root.kernel_id
        assert restored.store.goal(root.id)["objective"] == "Persist objective"
        assert len(restored.store.events(root.id, limit=500)) > len(before_history)
        assert restored.store.messages(root.id)[0]["received_at"]
        assert "124" in str(restored.store.events(root.id, kind="python_result")[-1])
        assert restored.store.events(root.id, kind="context_compaction")
    finally:
        await restored.shutdown()


async def test_branch_is_new_identity_and_preserves_history_and_checkpoint(
    runtime, tmp_path, config
):
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [response("python", code="x = 40"), response("agent_wait", seconds=300)],
            "branch": [
                response("finish", result="unused"),
                response("finish", result="unused"),
                response("python", code="x += 2\nx"),
                response("finish", result="42"),
            ],
        }
    )
    root = runtime.create("Branch", tmp_path, config=config)
    await runtime.start()
    await eventually(
        lambda: runtime.store.session(root.id).turns == 2 and root.id not in runtime.tasks
    )
    previous = runtime.store.events(root.id, limit=500)
    branch = await runtime.fork(root.id, name="branch")
    assert branch.id != root.id and branch.root_id == branch.id
    assert branch.branch_from == root.id and branch.branch_event
    assert branch.kernel_id != root.kernel_id
    assert (await runtime.wait(branch.id)).outcome == Outcome.COMPLETED
    after = runtime.store.events(root.id, limit=500)
    assert after[: len(previous)] == previous
    assert "42" in str(runtime.store.events(branch.id, kind="python_result"))
    assert runtime.store.session(root.id).outcome == Outcome.ACTIVE


async def test_terminal_parent_can_leave_child_running_when_gate_disabled(
    runtime, tmp_path, config
):
    config.task.wait_for_children = False
    runtime.providers["mock"] = ScriptedProvider(
        {
            "root": [
                response("agent_spawn", instruction="Continue independently", name="child"),
                response("finish", result="Parent done"),
            ],
            "child": [response("agent_wait", seconds=0.2), response("finish", result="Child done")],
        }
    )
    root = runtime.create("Root", tmp_path, config=config)
    await runtime.start()
    assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
    child = runtime.store.sessions(root_id=root.id)[1]
    assert (await runtime.wait(child.id)).outcome == Outcome.COMPLETED


async def test_recovery_of_action_receipt_crash_window(tmp_path, config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Receipt", tmp_path, config=config)
    action_id, event = "action_receipt_test", runtime.store.event(root.id, "tool_call", {})
    runtime.store.db.execute(
        "INSERT INTO actions VALUES(?,?,?,?,?,?,?,?)",
        (action_id, root.id, "python", "{}", "running", None, event, None),
    )
    kernel = runtime._kernel(root.id)
    await kernel.execute(action_id, "marker = workspace / 'once'\nmarker.write_text('once')", 5)
    await runtime.shutdown()
    recovered = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        await recovered.recover()
        row = recovered.store.db.execute(
            "SELECT * FROM actions WHERE id=?", (action_id,)
        ).fetchone()
        assert row["status"] == "done"
        assert "error" not in json.loads(row["result"])
        assert recovered.store.events(root.id, kind="tool_result")[-1]["payload"]["recovered"]
        assert (tmp_path / "once").read_text() == "once"
    finally:
        await recovered.shutdown()


async def test_recovery_marks_external_actions_uncertain_and_accounts_unfinished_model(
    tmp_path, config
):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Recovery", tmp_path, config=config)
    event = runtime.store.event(root.id, "tool_call", {})
    runtime.store.db.execute(
        "INSERT INTO actions VALUES(?,?,?,?,?,?,?,?)",
        ("external", root.id, "workspace_write", "{}", "running", None, event, None),
    )
    call = runtime.store.event(root.id, "model_invocation_started", {})
    runtime.store.db.execute(
        "INSERT INTO reservations VALUES(?,?,?,?,?)", (call, root.id, 123, 45, 0.02)
    )
    await runtime.shutdown()
    recovered = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        await recovered.recover()
        result = json.loads(
            recovered.store.db.execute("SELECT result FROM actions WHERE id='external'").fetchone()[
                0
            ]
        )
        assert result["error"]["uncertain"]
        assert result["error"]["category"] == "runtime"
        assert recovered.store.usage(root.id).input_tokens == 123
        assert recovered.store.usage(root.id).estimated_calls == 1
        await recovered.recover()
        assert recovered.store.usage(root.id).input_tokens == 123
    finally:
        await recovered.shutdown()
