"""Actual kernel/bootstrap, process lifecycle, durable goals and SDK transport tests."""

import asyncio
import json
import socket
import sys
from pathlib import Path

import httpx
import pytest

from threadweave.context import Context
from threadweave.models import Usage, new_id
from threadweave.runtime import GoalLimitReached, Runtime
from threadweave.skills import discover, load_module
from threadweave.tools import ToolContext

from .fakes import ScriptedProvider
from .test_python_control import cell


async def test_complete_history_pagination_and_readable_log(tmp_path, python_config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("long original " * 10000, tmp_path, config=python_config)
        for i in range(1250):
            runtime.store.event(root.id, "observation", {"ordinal": i})
        events = list(runtime.store.iter_events(root.id, kind="observation"))
        assert [e["payload"]["ordinal"] for e in events] == list(range(1250))
        projection = runtime.context.history_file(root.id)
        assert len(projection.read_text().splitlines()) >= 1250
        runtime.context.compact(root.id)
        runtime.context = Context(runtime.store)  # projection rebuild after restart
        assert runtime.context.history_file(root.id).read_text().count('"ordinal"') == 1250
        await cell(
            runtime,
            root.id,
            "assert len(context['task']) > 100000\nassert Path(context['messages_path']).is_file()\npage = history.read(after=-1, limit=100)\nassert page",
        )
    finally:
        await runtime.shutdown()


async def test_edit_uses_kernel_cwd_and_stop_cleans_background(tmp_path, python_config):
    python_config.capabilities = ["coding"]
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/source.txt").write_text("unique line\n")
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("edit", tmp_path, config=python_config, mode="interactive")
        await cell(
            runtime,
            root.id,
            "os.chdir(workspace / 'nested')\nawait edit('source.txt','unique line','updated line')\njob = bash('sleep 60')\nassert job.running",
        )
        assert (tmp_path / "nested/source.txt").read_text() == "updated line\n"
        edits = runtime.store.events(root.id, kind="code_edit")
        assert "nested/source.txt" in edits[0]["payload"]["files"]
        await runtime.stop(root.id)
        assert all(t.done() for t in runtime.background.tasks.values())
        assert not runtime.background.processes
    finally:
        await runtime.shutdown()


async def test_goal_budget_is_durable_and_covers_descendants(tmp_path, python_config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("goal", tmp_path, config=python_config, mode="interactive")
        await cell(
            runtime, root.id, "await goal.create('Persistent objective', token_budget=1000000)"
        )
        child = runtime.spawn(root.id, "work")
        runtime.store.charge(child.id, Usage(input_tokens=1000000))
        with pytest.raises(GoalLimitReached):
            runtime._check_limits(root.id)
        with pytest.raises(GoalLimitReached):
            runtime._check_limits(child.id)
        assert runtime.store.goal(root.id)["tokens_used"] == 1000000
        await runtime.shutdown()
        runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
        assert runtime.store.goal(root.id)["token_budget"] == 1000000
        with pytest.raises(GoalLimitReached):
            runtime._check_limits(root.id)
    finally:
        await runtime.shutdown()


def package(workspace, name, code, permissions="[python]"):
    directory = workspace / ".agents/skills" / name
    module = directory / "src" / name.replace("-", "_")
    module.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Executable test procedure\npermissions: {permissions}\n---\nRead and run this procedure.\n"
    )
    (directory / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.0.1"\n')
    (module / "__init__.py").write_text(code)
    return module


async def test_skills_discovery_bootstrap_execution_provenance_and_rollback(
    tmp_path, python_config
):
    package(tmp_path, "parity-procedure", "async def run(value=1):\n    return value + 10\n")
    package(tmp_path, "parity-library", "print('skill import log')\nconstant = 23\n")
    package(tmp_path, "parity-unavailable", "import missing_dependency_937\n")
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("skills", tmp_path, config=python_config)
        await cell(
            runtime,
            root.id,
            "assert await parity_procedure(2) == 12\nassert parity_library.constant == 23\nassert await skills.run('parity-procedure', value=3) == 13\nassert skills.load('parity-library').constant == 23\ntry:\n    await parity_unavailable()\nexcept RuntimeError as error:\n    assert 'missing_dependency' in str(error)\nelse:\n    raise AssertionError('Expected import diagnostic')\nentry = harness.create_skill('Skill', {'name':'parity_stored','description':'retained code','code':'skill_result = 17'}, id='stable-skill')\nassert harness.get('skill','stable-skill').version == 1\nassert await skills.run('parity_stored') == 17\nharness.delete_skill('stable-skill')\nassert harness.get('skill','stable-skill').deleted\nharness.rollback('skill','stable-skill',1)\nassert not harness.get('skill','stable-skill').deleted\nassert harness.get('skill','stable-skill').version == 3",
        )
        assert runtime.store.db.execute("SELECT COUNT(*) FROM skill_outcomes").fetchone()[0] == 2
        states = runtime.store.states(root.id)
        assert states[0]["provenance"]["source_events"] and states[0]["intended_effect"]
        await cell(
            runtime,
            root.id,
            "ref = harness.create_skill('Reference procedure', 'Add ten', reference={'type':'python','import':'parity_procedure','callable':'run'}, arguments={'value':7})\nassert await skills.run('Reference procedure') == 17",
        )
    finally:
        await runtime.shutdown()


def test_skill_fingerprint_and_permissions_validation(tmp_path):
    module = package(tmp_path, "parity-fingerprint", "value = 1\n")
    before = next(e for e in discover(tmp_path) if e["name"] == "parity-fingerprint")
    assert load_module(before).value == 1
    (module / "__init__.py").write_text("value = 200\n")
    after = next(e for e in discover(tmp_path) if e["name"] == "parity-fingerprint")
    assert before["sha256"] != after["sha256"]
    assert load_module(after).value == 200
    package(tmp_path, "parity-invalid", "", permissions="python")
    assert (
        next(e for e in discover(tmp_path) if e["name"] == "parity-invalid")["kind"]
        == "unavailable"
    )


async def test_real_mcp_http_transport_and_unconfigured_server(tmp_path, python_config):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name("mcp_server.py")),
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    runtime = None
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        async with asyncio.timeout(15), httpx.AsyncClient() as client:
            while True:
                try:
                    await client.get(url)
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.05)
        python_config.mcp_servers = {
            "local-http": {"type": "http", "url": url, "enabled_tools": ["echo"]}
        }
        runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
        root = runtime.create("http", tmp_path, config=python_config)
        await cell(
            runtime,
            root.id,
            "assert [t['name'] for t in await mcp.list_tools('local-http')] == ['echo']\nvalue = await mcp.call_tool('local-http','echo',{'text':'HTTP works'})\nif isinstance(value,str): value=json.loads(value)\nassert value['text'] == 'HTTP works'\nawait mcp.close()",
        )
        with pytest.raises(PermissionError, match="not configured"):
            await runtime.mcp.call(
                ToolContext(runtime, root.id, new_id(), "test"), "tools", {"server": "absent"}
            )
        assert all(
            not json.loads(row[0]).get("access_token")
            for row in runtime.store.db.execute("SELECT body FROM configs")
        )
    finally:
        if runtime:
            await runtime.shutdown()
        server.terminate()
        await server.wait()


async def test_observation_is_related_bounded_and_durable(tmp_path, python_config):
    runtime = Runtime(tmp_path / "data", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("observe", tmp_path, config=python_config)
        child = runtime.spawn(root.id, "work", name="observer")
        runtime.message(root.id, child.id, "evidence" * 100)
        await cell(
            runtime,
            root.id,
            "assert await agent_observe.list_agents()\nchild_status = await agent_observe.get_agent('observer')\npreviews = await agent_observe.recent_messages('observer', limit=1, max_chars=80)\nassert len(previews['messages'][0]['body']) == 80",
        )
        stranger = runtime.create("unrelated", tmp_path, config=python_config)
        from threadweave.host_api import Request, dispatch

        with pytest.raises(ValueError, match="not visible"):
            await dispatch(
                ToolContext(runtime, root.id, new_id(), "test"),
                Request(operation="agent_observe.get", payload={"target": stranger.id}),
            )
    finally:
        await runtime.shutdown()
