"""Real temporary Git trees/workers; model and external-MCP doubles only in tests."""

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

from threadweave.coding import CodingTask
from threadweave.coding_config import update_coding_options
from threadweave.editing import Editor
from threadweave.gitops import candidate_result
from threadweave.models import Action, HarnessError, McpServerConfig, Outcome, RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext, builtins

from .conftest import eventually
from .fakes import ScriptedProvider


@pytest.fixture
async def coding_python(tmp_path, repository, coding_config):
    coding_config.control_plane = "python"
    update_coding_options(coding_config.task, capture_baseline=False)
    update_coding_options(coding_config.task, require_clean_baseline=False)
    coding_config.permissions.append("mcp")
    coding_config.mcp_servers = {
        "fixture": McpServerConfig(
            command=sys.executable, args=[str(Path(__file__).with_name("mcp_server.py"))]
        )
    }
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    session = runtime.create("Observe Python", repository, config=coding_config, mode="interactive")
    context = ToolContext(
        runtime, session.id, new_id(), runtime.store.event(session.id, "test_setup", {})
    )
    await runtime.environment.prepare(session.id, force=True)
    try:
        yield runtime, session, context
    finally:
        await runtime.shutdown()


async def cell(runtime, session, code):
    event = runtime.store.event(session.id, "test_cell", {})
    return await runtime._execute_action(
        session.id, new_id(), Action(name="ipython", arguments={"code": code}), event
    )


def effects(runtime, sid, *, reason="action"):
    return [
        e["payload"]
        for e in runtime.store.events(sid, kind="workspace_effects", limit=100)
        if e["payload"].get("kind") == "externally_observed" and e["payload"]["reason"] == reason
    ]


@pytest.mark.parametrize(
    "code,expected",
    [
        ("Path('mathops.py').write_text('changed')", {"mathops.py"}),
        ("open('new.txt', 'w').write('new')", {"new.txt"}),
        ("Path('mathops.py').unlink()", {"mathops.py"}),
        ("Path('mathops.py').rename('renamed.py')", {"mathops.py", "renamed.py"}),
        (
            "import shutil\nshutil.copyfile('mathops.py', 'copy.py')\nPath('mathops.py').write_text('two changes')",
            {"mathops.py", "copy.py"},
        ),
        ("Path('binary.dat').write_bytes(bytes(range(256)))", {"binary.dat"}),
        ("Path('mathops.py').chmod(0o755)", {"mathops.py"}),
        (
            "Path('mathops.py').write_text('partial')\nraise ValueError('after write')",
            {"mathops.py"},
        ),
        (
            "Path('.pytest_cache').mkdir(exist_ok=True)\nPath('.pytest_cache/ignored.dat').write_text('observed')",
            {".pytest_cache/ignored.dat"},
        ),
    ],
)
async def test_arbitrary_python_mutations_are_observed(coding_python, code, expected):
    runtime, session, _ = coding_python
    result = await cell(runtime, session, code)
    assert bool(result.get("error")) == ("raise ValueError" in code)
    evidence = effects(runtime, session.id)[-1]
    assert set(evidence["files"]) == expected
    assert evidence["before_state"] != evidence["after_state"]
    assert not evidence["transactional"] and not evidence["rollback_performed"]
    assert (
        runtime.artifacts.load(session.id, evidence["observation_artifact"])["files"]
        == evidence["files"]
    )
    event = runtime.store.events(session.id, kind="tool_call", limit=1)[0]
    assert evidence["execution_id"] == event["payload"]["action_id"]
    if "rename" in code:
        assert evidence["renames"] == [
            {"from": "mathops.py", "to": "renamed.py", "basis": "identical_content_and_mode"}
        ]
    assert not runtime.store.events(session.id, kind="code_edit")


@pytest.mark.parametrize(
    "code",
    [
        "x = 123",
        "Path('transient').write_text('x')\nPath('transient').unlink()",
        "p=Path('mathops.py')\np.write_bytes(p.read_bytes())",
    ],
)
async def test_no_net_change_is_not_reported_as_mutation(coding_python, code):
    runtime, session, _ = coding_python
    assert not (await cell(runtime, session, code)).get("error")
    assert not effects(runtime, session.id)
    if code == "x = 123":
        assert runtime.environment.adapters["coding"].mutations.stats["hashed_bytes"] == 0


async def test_dirty_state_is_compared_to_cell_start_not_git_head(coding_python, repository):
    runtime, session, _ = coding_python
    (repository / "mathops.py").write_text("existing user change")
    (repository / "untouched.txt").write_text("existing untracked")
    await cell(runtime, session, "Path('mathops.py').write_text('agent change')")
    evidence = effects(runtime, session.id)[-1]
    assert set(evidence["files"]) == {"mathops.py"}
    assert (
        evidence["files"]["mathops.py"]["before"]["hash"]
        == hashlib.sha256(b"existing user change").hexdigest()
    )
    assert effects(runtime, session.id, reason="between_actions")


async def test_rpc_capabilities_receive_common_hooks_and_audit(coding_python, monkeypatch):
    runtime, session, _ = coding_python
    before, after = [], []
    original_before, original_after = (
        runtime.environment.before_action,
        runtime.environment.after_action,
    )

    async def entering(context, name):
        before.append((name, context.from_python, context.action_id))
        return await original_before(context, name)

    def leaving(context, token):
        after.append((context.capability.name, context.from_python, context.action_id))
        original_after(context, token)

    monkeypatch.setattr(runtime.environment, "before_action", entering)
    monkeypatch.setattr(runtime.environment, "after_action", leaving)
    result = await cell(
        runtime,
        session,
        """
matches = repo.search('add')
await edit('mathops.py', 'a - b', 'a + b')
tested = tests.run()
assert tested['passed']
child = await rlm('observe only', name='observer')
assert child.session_id
full = await tools.acall('workspace_read', path='mathops.py')
""",
    )
    assert not result.get("error"), result
    assert set(before) == set(after)
    assert {n for n, python, _ in before if python} >= {
        "repo_search",
        "edit",
        "run_tests",
        "rlm.run",
        "workspace_read",
    }
    assert len(runtime.store.events(session.id, kind="tool_result", limit=100)) == len(before)
    assert len(
        runtime.store.events(session.id, kind="environment_action_finished", limit=100)
    ) == len(before)
    assert len(runtime.store.events(session.id, kind="code_edit")) == 1
    assert not runtime.environment.adapters["coding"].mutations.active


@pytest.mark.parametrize("from_python", [False, True])
async def test_direct_and_python_registry_calls_share_policy(
    coding_python, from_python, monkeypatch
):
    runtime, session, context = coding_python
    config = runtime.store.config(session.id).model_copy(deep=True)
    config.control_plane = "direct"
    config.permissions.remove("workspace.write")
    monkeypatch.setattr(runtime.store, "config", lambda sid: config)
    calls = []

    async def forbidden_hook(*args):
        calls.append(args)
        raise AssertionError("Permission checks must precede hooks")

    monkeypatch.setattr(runtime.environment, "before_action", forbidden_hook)
    invocation = runtime._execute_action(
        session.id,
        new_id(),
        Action(name="workspace_write", arguments={"path": "denied", "content": "no"}),
        context.source_event,
        from_python=from_python,
    )
    if from_python:
        with pytest.raises(HarnessError, match="permission_denied"):
            await invocation
    else:
        assert (await invocation)["error"]["code"] == "permission_denied"
    assert not calls and not (context.path("denied")).exists()
    assert runtime.store.events(session.id, kind="tool_result")


@pytest.mark.parametrize("case", ["read_only", "permission", "allowlist"])
async def test_host_rpc_edit_resolves_policy_before_environment(coding_python, monkeypatch, case):
    runtime, session, context = coding_python
    config = runtime.store.config(session.id).model_copy(deep=True)
    if case == "read_only":
        config.execution.read_only = True
    elif case == "permission":
        config.permissions.remove("workspace.write")
    else:
        config.tool_allowlist = ["ipython", "host_request"]  # not the edit capability
    monkeypatch.setattr(runtime.store, "config", lambda sid: config)
    with pytest.raises(HarnessError, match="permission_denied"):
        await runtime._execute_action(
            session.id,
            new_id(),
            Action(
                name="host_request",
                arguments={
                    "operation": "edit",
                    "payload": {"path": "mathops.py", "old_str": "a - b", "new_str": "a + b"},
                },
            ),
            context.source_event,
            from_python=True,
        )
    assert not runtime.store.events(session.id, kind="code_edit")


@pytest.mark.parametrize("phase", ["before", "after", "host"])
async def test_hook_and_rpc_failures_are_audited_without_claiming_rollback(
    coding_python, monkeypatch, phase
):
    runtime, session, context = coding_python
    original = runtime.environment.after_action
    if phase == "before":

        async def reject(c, name):
            raise PermissionError("Environment rejected")

        monkeypatch.setattr(runtime.environment, "before_action", reject)
    elif phase == "after":

        def reject(c, token):
            original(c, token)
            raise ValueError("After-action failure")

        monkeypatch.setattr(runtime.environment, "after_action", reject)
    result = await cell(
        runtime,
        session,
        "Path('partial').write_text('retained')"
        if phase != "host"
        else "Path('partial').write_text('retained')\nawait edit('missing', 'x', 'y')",
    )
    assert result.get("error"), result
    assert context.path("partial").exists() == (phase != "before")
    if phase != "before":
        assert effects(runtime, session.id)[-1]["files"]["partial"]
    if phase == "after":
        assert result["error"]["code"] == "after_action_failed"
        assert runtime.store.session(session.id).paused
    assert all(r[0] == "done" for r in runtime.store.db.execute("SELECT status FROM actions"))


@pytest.mark.parametrize("interruption", ["cancel", "kernel_exit"])
async def test_interrupted_cell_retains_partial_mutation_evidence(
    coding_python, repository, interruption
):
    runtime, session, _ = coding_python
    task = asyncio.create_task(
        cell(
            runtime,
            session,
            "Path('partial').write_text('before interruption')\nimport time\ntime.sleep(30)",
        )
    )
    await eventually(lambda: (repository / "partial").exists())
    if interruption == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        runtime.kernels[session.id].process.kill()
        assert (await task).get("error")
    assert (repository / "partial").read_text() == "before interruption"
    assert effects(runtime, session.id)[-1]["files"]["partial"]
    assert not runtime.environment.adapters["coding"].mutations.active


async def test_recovery_observes_open_window_without_replaying_python(coding_python, repository):
    runtime, session, context = coding_python
    observer = runtime.environment.adapters["coding"].mutations
    wid = observer.begin(context)
    (repository / "after-crash").write_bytes(b"partial")
    # Persisted open window simulates daemon loss before its after hook.
    observer.cache.clear()
    observer.active.clear()
    observer.baselines.clear()
    await runtime.recover()
    evidence = effects(runtime, session.id, reason="recovery")[-1]
    assert evidence["recovered"] and not evidence["rollback_performed"]
    assert "after-crash" in evidence["files"]
    assert (
        runtime.store.db.execute(
            "SELECT status FROM mutation_windows WHERE id=?", (wid,)
        ).fetchone()[0]
        == "observed"
    )
    assert not runtime.store.events(session.id, kind="model_started")


async def test_delayed_unmanaged_process_is_observed_while_session_idle(coding_python, repository):
    runtime, session, _ = coding_python
    script = "import time; from pathlib import Path; time.sleep(.3); Path('later').write_text('background')"
    await cell(
        runtime,
        session,
        f"import subprocess, sys\nproc = subprocess.Popen([sys.executable, '-c', {script!r}], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
    )
    await runtime.start()
    await eventually(
        lambda: effects(runtime, session.id, reason="background_reconciliation"), seconds=8
    )
    assert (repository / "later").exists()
    assert not runtime.store.session(session.id).runnable


async def test_cached_large_files_are_not_rehashed_for_noop_or_small_edit(
    coding_python, repository
):
    runtime, session, context = coding_python
    large = repository / "large.bin"
    with large.open("wb") as stream:
        stream.truncate(70 * 1024 * 1024)  # Observer hashes streaming; no 64MB snapshot limit.
    observer = runtime.environment.adapters["coding"].mutations
    observer.reconcile(context, reason="test_setup")
    assert observer.stats["hashed_bytes"] >= 70 * 1024 * 1024
    await cell(runtime, session, "x=1")
    assert observer.stats["hashed_bytes"] == 0
    await cell(runtime, session, "Path('tiny').write_text('ok')")
    assert observer.stats["hashed_bytes"] == 2 and observer.stats["hashed_files"] == 1


async def test_symlink_target_is_not_read_or_mutated_by_observer(
    coding_python, tmp_path, repository
):
    runtime, session, _ = coding_python
    outside = tmp_path / "outside.py"
    outside.write_text("private")
    await cell(
        runtime,
        session,
        f"Path('mathops.py').unlink()\nPath('mathops.py').symlink_to({str(outside)!r})",
    )
    evidence = effects(runtime, session.id)[-1]
    assert evidence["files"]["mathops.py"]["after"]["kind"] == "symlink"
    assert outside.read_text() == "private"


async def test_verifier_sees_python_shell_and_candidate_edits(coding_python, repository):
    runtime, session, context = coding_python
    assert not (
        await cell(
            runtime,
            session,
            "Path('mathops.py').write_text(Path('mathops.py').read_text().replace('a - b', 'a + b'))",
        )
    ).get("error")
    assert (await CodingTask().verify(context, runtime.store.config(session.id).task)).passed
    child = runtime.spawn(session.id, "candidate", isolate=True)
    child_context = ToolContext(runtime, child.id, new_id(), context.source_event)
    Editor(child_context).apply(
        {"mathops.py": (repository / "mathops.py").read_bytes() + b"\n# candidate\n"}
    )
    runtime.store.finish(child.id, Outcome.COMPLETED)
    candidate_result(context, child.id, accept=True)
    result = await cell(
        runtime,
        session,
        "result = await bash('printf shell > shell-output.txt')\nassert result.exit_code == 0",
    )
    assert not result.get("error"), result
    assert (await CodingTask().verify(context, runtime.store.config(session.id).task)).passed
    assert "# candidate" in (repository / "mathops.py").read_text()


def test_default_surface_and_explicit_direct_surface_unchanged():
    assert [t["function"]["name"] for t in builtins().schemas(RunConfig())] == ["ipython"]
    assert "workspace_write" in [
        t["function"]["name"] for t in builtins().schemas(RunConfig(control_plane="direct"))
    ]


async def test_real_mcp_rpc_uses_environment_hooks(coding_python):
    runtime, session, _ = coding_python
    result = await cell(
        runtime,
        session,
        "schema = await mcp.list_tools('fixture')\nassert schema\nanswer = await mcp.call_tool('fixture', 'echo', {'text': 'through Environment'})\nif isinstance(answer, str): answer = json.loads(answer)\nassert answer['text'] == 'through Environment'",
    )
    assert not result.get("error"), result
    started = runtime.store.events(session.id, kind="environment_action_started", limit=100)
    finished = runtime.store.events(session.id, kind="environment_action_finished", limit=100)
    assert {e["payload"]["capability"] for e in started} >= {"mcp.tools", "mcp.call"}
    assert {e["payload"]["action_id"] for e in started} == {
        e["payload"]["action_id"] for e in finished
    }


async def test_same_cell_write_then_tests_invalidates_stale_bytecode(coding_python):
    runtime, session, _ = coding_python
    result = await cell(
        runtime,
        session,
        """
import py_compile, os
py_compile.compile('mathops.py')
p = Path('mathops.py')
stamp = p.stat()
p.write_text(p.read_text().replace('a - b', 'a + b'))
os.utime(p, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
assert tests.run()['passed']
""",
    )
    assert not result.get("error"), result
    assert effects(runtime, session.id, reason="before_process")


async def test_failed_observation_recovers_without_hiding_partial_write(
    coding_python, monkeypatch, repository
):
    runtime, session, _ = coding_python
    observer, original = (
        runtime.environment.adapters["coding"].mutations,
        runtime.environment.adapters["coding"].mutations.report,
    )

    def fail(context, before, after, **kwargs):
        if kwargs.get("reason") == "action":
            raise OSError("audit store unavailable")
        return original(context, before, after, **kwargs)

    monkeypatch.setattr(observer, "report", fail)
    result = await cell(runtime, session, "Path('partial').write_text('not rolled back')")
    assert result["error"]["code"] == "observation_failed"
    assert runtime.store.session(session.id).paused
    assert (repository / "partial").read_text() == "not rolled back"
    monkeypatch.setattr(observer, "report", original)
    await runtime.recover()
    assert effects(runtime, session.id, reason="recovery")[-1]["files"]["partial"]


async def test_actual_daemon_process_loss_recovers_mutation(tmp_path, repository, coding_config):
    import json

    coding_config.control_plane = "python"
    update_coding_options(coding_config.task, capture_baseline=False)
    update_coding_options(coding_config.task, require_clean_baseline=False)
    data = tmp_path / "crashed-state"
    script = f"""
import asyncio, os
from pathlib import Path
from threadweave.runtime import Runtime
from threadweave.models import RunConfig, Action, new_id
from tests.fakes import ScriptedProvider
async def main():
    r = Runtime({str(data)!r}, providers={{'test': ScriptedProvider({{}})}})
    s = r.create('crash observation', {str(repository)!r}, config=RunConfig.model_validate_json({coding_config.model_dump_json()!r}), mode='interactive')
    event = r.store.event(s.id, 'crash_test', {{}})
    asyncio.create_task(r._execute_action(s.id, new_id(), Action(name='ipython', arguments={{'code': "Path('survived-crash').write_text('retained')\\nimport time\\ntime.sleep(30)"}}), event))
    while not Path({str(repository / "survived-crash")!r}).exists():
        await asyncio.sleep(.01)
    print(s.id, flush=True)
    os._exit(17)
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        output, error = await asyncio.wait_for(process.communicate(), 15)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 17, error.decode()
    sid = output.decode().strip()
    runtime = Runtime(data, providers={"test": ScriptedProvider({})})
    try:
        await runtime.recover()
        assert runtime.store.session(sid).id == sid
        assert effects(runtime, sid, reason="recovery")[-1]["files"]["survived-crash"]
        assert (repository / "survived-crash").read_text() == "retained"
        assert not json.loads(
            runtime.store.db.execute("SELECT result FROM actions WHERE name='ipython'").fetchone()[
                0
            ]
        ).get("rollback_performed")
    finally:
        await runtime.shutdown()
