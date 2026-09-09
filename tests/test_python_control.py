"""Execution semantics, captured provider requests, real workers/processes and MCP."""

import asyncio
import json
import sys
from collections import deque
from pathlib import Path

import pytest

from threadweave.models import ModelResponse, RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext, builtins

from .conftest import eventually, response
from .fakes import ScriptedProvider


async def cell(runtime, sid, code):
    event = runtime.store.event(sid, "test_cell", {})
    result = await runtime.execute_python(ToolContext(runtime, sid, new_id(), event), code)
    assert not result.get("error"), result
    return result


def test_default_schema_has_one_control_plane():
    schema = builtins().schemas(RunConfig())
    snapshot = Path(__file__).resolve().parent / "fixtures/ipython-schema.json"
    assert schema == json.loads(snapshot.read_text())
    assert [s["function"]["name"] for s in schema] == ["ipython"]
    assert schema[0]["function"]["parameters"]["required"] == ["code"]
    assert not RunConfig().task.wait_for_children
    assert RunConfig(task={"wait_for_children": True}).task.wait_for_children
    direct_config = RunConfig(control_plane="direct")
    direct_config.permissions.append("process")
    direct = builtins().schemas(direct_config)
    assert {"python", "rlm"} <= {s["function"]["name"] for s in direct}
    assert not {"repo_search", "run_tests"} & {s["function"]["name"] for s in direct}


async def test_provider_boundary_rejects_direct_model_tool(tmp_path, python_config):
    provider = ScriptedProvider(
        {
            "root": [
                response("workspace_write", path="bad", content="bad"),
                ModelResponse(text="done"),
            ]
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        session = runtime.create("hi", tmp_path, config=python_config)
        await runtime.start()
        assert (await runtime.wait(session.id)).outcome == "completed"
        assert all(
            [t["function"]["name"] for t in r.tools] == ["ipython"] for r in provider.requests
        )
        assert not (tmp_path / "bad").exists()
        assert (
            runtime.store.events(session.id, kind="failure")[0]["payload"]["code"]
            == "tool_not_exposed"
        )
    finally:
        await runtime.shutdown()


async def test_python_heartbeat_text_yields_without_terminating(tmp_path, python_config):
    provider = ScriptedProvider({"root": [ModelResponse(text="Heartbeat observed")] * 2})
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("periodic", tmp_path, config=python_config, mode="heartbeat")
        await runtime.start()
        await eventually(
            lambda: runtime.store.session(root.id).turns == 1 and root.id not in runtime.tasks
        )
        assert runtime.store.session(root.id).outcome == "active"
        assert not runtime.store.session(root.id).runnable
        runtime.message(None, root.id, "next heartbeat")
        await eventually(
            lambda: runtime.store.session(root.id).turns == 2 and root.id not in runtime.tasks
        )
        assert runtime.store.session(root.id).outcome == "active"
    finally:
        await runtime.shutdown()


async def test_python_children_share_workspace_unless_isolation_explicit(
    repository, tmp_path, python_config
):
    python_config.task.adapter = "coding"
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("candidates", repository, config=python_config)
        shared = runtime.spawn(root.id, "shared")
        isolated = runtime.spawn(root.id, "isolated", isolate=True)
        assert shared.workspace.path == root.workspace.path
        assert isolated.workspace.path != root.workspace.path
        assert runtime.store.config(shared.id).task.adapter == "coding"
        assert runtime.store.config(shared.id).task.verifier == "coding"
        assert runtime.store.config(isolated.id).task.verifier == "coding"
        assert await asyncio.to_thread(Path(isolated.workspace.path, "mathops.py").is_file)
    finally:
        await runtime.shutdown()


async def test_programmable_multiturn_children_context_and_recovery(
    tmp_path, python_config, repository
):
    sequence = deque()

    def root_reply(request):
        return sequence.popleft() if sequence else ModelResponse(text="Ready")

    def child_reply(request):
        return response(
            "ipython",
            code=(
                "assert 'x' not in globals()\n"
                f"child_value = {request.name!r}\n"
                "while not (workspace / 'parent-continued').exists():\n    await asyncio.sleep(0.01)\n"
                "observation = (workspace / 'mathops.py').read_text()\n"
                "await agent_message.send(child_value + ' inspected ' + observation[:20], receiver_role='parent')"
            ),
        )

    provider = ScriptedProvider(
        {
            "root": [root_reply] * 30,
            "persistence": [child_reply, ModelResponse(text="Persistence inspected")],
            "messaging": [child_reply, ModelResponse(text="Messaging inspected")],
        },
        delay=0.02,
    )
    data = tmp_path / "state"
    runtime = Runtime(data, providers={"mock": provider})

    async def say(text, code=None):
        before = runtime.store.session(root.id).turns
        if code:
            sequence.append(response("ipython", code=code))
        sequence.append(ModelResponse(text="Ready"))
        runtime.interact(root.id, text)
        await eventually(
            lambda: (
                runtime.store.session(root.id).turns > before
                and not runtime.store.session(root.id).runnable
                and root.id not in runtime.tasks
            )
        )

    try:
        root = runtime.create(
            "long task " * 10000, repository, config=python_config, mode="interactive"
        )
        await runtime.start()
        await say("hi")
        assert runtime.store.usage(root.id).tool_calls == 0
        await say(
            "retain",
            "x = 123\nfiles = list(workspace.rglob('*.py'))\nsearch_result = repo.search('def add')\nassert len(context['task']) > 6000",
        )
        await say(
            "delegate",
            "left = await rlm('inspect persistence', name='persistence')\nright = await rlm('inspect messaging', name='messaging')\n(workspace / 'parent-continued').write_text('yes')",
        )
        await eventually(
            lambda: (
                len(runtime.store.sessions(root_id=root.id)) == 3
                and all(
                    s.outcome == "completed"
                    for s in runtime.store.sessions(root_id=root.id)
                    if s.parent_id
                )
            )
        )
        children = [s for s in runtime.store.sessions(root_id=root.id) if s.parent_id]
        assert all(s.workspace.path == root.workspace.path for s in children)
        assert len({s.kernel_id for s in [root, *children]}) == 3
        assert {s.id for s in children} <= {m["sender_id"] for m in runtime.store.messages(root.id)}
        await say(
            "precise edit and real tests",
            "await edit('mathops.py', 'return a - b', 'return a + b')\nresult = await bash("
            + repr(f"{sys.executable} -m pytest -q")
            + ")\nassert result.exit_code == 0, result.output",
        )
        await say(
            "retain procedure",
            "memory = harness.create_memory('Observation', 'x is a retained scalar', id='scalar')\nassert harness.get('memory', 'scalar').version == 1\n"
            + "memory = harness.update_memory('scalar', 'Observation', 'x equals 123')\nassert memory.version == 2\n"
            + "skill = harness.create_skill('Read x', {'name':'read_x','description':'Read retained x','code':'skill_result = x','required_permissions':['python']})\nassert await skills.run('read_x') == 123",
        )
        runtime.context.compact(root.id)
        await say(
            "after compaction",
            "assert x == 123 and files\nassert left.session_id != right.session_id\nassert history.messages()",
        )
        assert all(
            {t["function"]["name"] for t in r.tools} == {"ipython"} for r in provider.requests
        )
        assert all("Current coding evidence" not in str(r.messages) for r in provider.requests)
        assert not runtime.store.events(root.id, kind="coding_baseline")
        assert not runtime.store.events(root.id, kind="verifier_started")
        assert not runtime.store.events(root.id, kind="failure")
        identities = {s.id: s.kernel_id for s in runtime.store.sessions(root_id=root.id)}
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": provider})
        await runtime.start()
        await say(
            "same logical session",
            "assert x == 123 and files\nassert left.session_id and right.session_id\nassert harness.get('memory','scalar').version == 2",
        )
        assert identities == {s.id: s.kernel_id for s in runtime.store.sessions(root_id=root.id)}
    finally:
        await runtime.shutdown()


async def test_ipython_and_background_async_survive_between_cells(tmp_path, python_config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("state", tmp_path, config=python_config, mode="interactive")
        await cell(
            runtime,
            root.id,
            "async def later():\n    await asyncio.sleep(.1)\n    globals()['after_idle'] = 7\n    print('background output')\nbackground = asyncio.create_task(later())\n%precision 3",
        )
        await asyncio.sleep(0.2)
        await cell(runtime, root.id, "assert after_idle == 7\nassert get_ipython() is not None")
        await cell(runtime, root.id, "job = bash('sleep .2; echo complete')\nassert job.pid > 0")
        await asyncio.sleep(0.3)
        result = await cell(
            runtime,
            root.id,
            "result = await job\nassert result.exit_code == 0\nassert 'complete' in result.output\nprint(result.stdout_artifact)",
        )
        assert result["stdout"].strip()
    finally:
        await runtime.shutdown()


async def test_shell_cancel_timeout_and_handle_recovery(tmp_path, python_config):
    data = tmp_path / "data"
    runtime = Runtime(data, providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("process", tmp_path, config=python_config, mode="interactive")
        await cell(
            runtime,
            root.id,
            "job = bash('sleep 10')\nassert job.running\njob.kill()\nassert not job.running\nshort = await bash('sleep 10', timeout=.1)\nassert short.timed_out",
        )
        await runtime.shutdown()
        runtime = Runtime(data, providers={"mock": ScriptedProvider({})})
        await cell(runtime, root.id, "assert not job.running\nassert job.poll().interrupted")
    finally:
        await runtime.shutdown()


async def test_real_mcp_stdio_discovery_call_errors_permissions_and_redaction(
    tmp_path, python_config, monkeypatch
):
    secret = "test-private-value-4935"
    monkeypatch.setenv("THREADWEAVE_MCP_TEST_CREDENTIAL", secret)
    python_config.mcp_servers = {
        "fixture": {
            "command": sys.executable,
            "args": [str(Path(__file__).with_name("mcp_server.py"))],
            "env_from": {"FIXTURE_CREDENTIAL": "THREADWEAVE_MCP_TEST_CREDENTIAL"},
            "disabled_tools": ["forbidden"],
        }
    }
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("MCP", tmp_path, config=python_config, mode="interactive")
        result = await cell(
            runtime,
            root.id,
            "servers = await mcp.list_servers()\nschema = await mcp.list_tools('fixture')\nassert {t['name'] for t in schema} == {'echo', 'rejected'}\nresult = await mcp.call_tool('fixture', 'echo', {'text':'genuine transport'})\nif isinstance(result, str): result = json.loads(result)\nassert result['text'] == 'genuine transport'\nassert result['credential'] == '[REDACTED]'\ntry:\n    await mcp.call_tool('fixture', 'rejected')\n    raise AssertionError('Expected MCP tool error')\nexcept RuntimeError as error:\n    assert 'deliberate diagnostic' in str(error)\nawait mcp.reload('fixture')\nassert await mcp.list_tools('fixture')",
        )
        assert secret not in json.dumps(result)
        assert secret not in json.dumps(runtime.store.events(root.id, limit=500))
        with pytest.raises(Exception, match="not discovered/enabled"):
            await runtime.mcp.call(
                ToolContext(runtime, root.id, new_id(), "test"),
                "call",
                {"server": "fixture", "tool": "forbidden"},
            )
    finally:
        await runtime.shutdown()
