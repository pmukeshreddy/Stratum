import asyncio
import base64
import os
import shutil
import sys

import pytest

from threadweave.coding import CodingTask, run_checks
from threadweave.diagnostics import parse_diagnostics, profiler_metrics
from threadweave.editing import Editor, make_diff, unified_changes
from threadweave.execution import ContainerExecutor, recover_containers
from threadweave.gitops import GitWorkspace, git
from threadweave.repository import digest

from .test_coding import FIX, setup_runtime


@pytest.mark.parametrize(
    "text,file,line",
    [
        ("src/parser.py:142: AssertionError: invalid input", "src/parser.py", 142),
        ("src/lib.rs:8:3: error[E0308]: mismatched types", "src/lib.rs", 8),
        ('  File "src/parser.py", line 42, in parse', "src/parser.py", 42),
        ("src/index.ts:12:5: error TS2322: incompatible", "src/index.ts", 12),
        ("main.cu:28: error: invalid conversion", "main.cu", 28),
    ],
)
def test_failure_locations(text, file, line):
    diagnostic = parse_diagnostics(text)["diagnostics"][0]
    assert diagnostic["file"] == file and diagnostic["line"] == line


def test_profiler_metrics_preserve_observed_units_and_values():
    text = '"ID","Metric Name","Metric Unit","Metric Value"\n"1","gpu__time_duration.sum","us","1,234.5"\n'
    parsed = profiler_metrics(text)
    assert parsed == [
        {
            "metric": "gpu__time_duration.sum",
            "unit": "us",
            "value": 1234.5,
            "source": "profiler_csv",
        }
    ]
    assert profiler_metrics("latency_ms=1.25 ms")[0]["value"] == 1.25


def test_newline_free_diff_roundtrip_and_empty_creation():
    for before, after in [
        (b"old", b"new"),
        (None, b""),
        (b"", None),
        (None, b"content"),
        (b"content", None),
    ]:
        assert unified_changes(
            make_diff("file", before, after), lambda _, before=before: before
        ) == {"file": after}


async def test_create_move_hash_range_delete_and_mode_rollback(tmp_path, repository, coding_config):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:

        def call(name, args):
            return runtime.tools.call(context, name, args)

        await call("create_file", {"path": "script.py", "content": "first\nsecond\n"})
        (repository / "script.py").chmod(0o755)
        await call("move_file", {"source": "script.py", "destination": "moved.py"})
        assert (repository / "moved.py").stat().st_mode & 0o777 == 0o755
        await call(
            "replace_range",
            {
                "path": "moved.py",
                "start_line": 2,
                "end_line": 2,
                "content": "changed\n",
                "expected_hash": digest((repository / "moved.py").read_bytes()),
            },
        )
        assert (repository / "moved.py").read_text() == "first\nchanged\n"
        deleted = await call(
            "delete_file",
            {"path": "moved.py", "expected_hash": digest((repository / "moved.py").read_bytes())},
        )
        await call("edit_rollback", {"edit_id": deleted["edit_id"]})
        assert (repository / "moved.py").stat().st_mode & 0o777 == 0o755
        (repository / "alias.py").symlink_to(repository / "mathops.py")
        with pytest.raises(PermissionError):
            Editor(context).apply({"alias.py": b"x"})
    finally:
        await runtime.shutdown()


async def test_independent_verifier_can_inspect_protected_tests_outside_agent_allowed_paths(
    tmp_path, repository, coding_config
):
    coding_config.task.allowed_paths = ["mathops.py"]
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        Editor(context).apply_patch(FIX)
        assert (await CodingTask().verify(context, coding_config.task)).passed
        with pytest.raises(PermissionError):
            context.path("tests/test_mathops.py")
    finally:
        await runtime.shutdown()


async def test_real_c_build_typecheck_and_python_lint_commands(tmp_path, repository, coding_config):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    (repository / "compute.c").write_text("int add(int a, int b) { return a + b; }\n")
    with (repository / ".gitignore").open("a") as stream:
        stream.write("*.o\n")
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "-qm",
        "C source",
    )
    coding_config.task.build_commands = [[compiler, "-c", "compute.c", "-o", "compute.o"]]
    coding_config.task.typecheck_commands = [[compiler, "-Werror", "-fsyntax-only", "compute.c"]]
    coding_config.task.lint_commands = [[sys.executable, "-m", "ruff", "check", "mathops.py"]]
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        for kind in ("build", "lint", "typecheck"):
            assert (await run_checks(context, kind))["passed"]
        assert (repository / "compute.o").exists()
    finally:
        await runtime.shutdown()


async def test_binary_artifact_import_and_secret_event_redaction(
    tmp_path, repository, coding_config, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-provider-value")
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        binary = b"\0\xff\x81profiler-report"
        (repository / "profile.bin").write_bytes(binary)
        result = await runtime.tools.call(context, "artifact_import", {"path": "profile.bin"})
        retained = runtime.artifacts.load(session.id, result["id"])
        assert base64.b64decode(retained["data"]) == binary
        eid = runtime.store.event(
            session.id, "observation", {"text": "sensitive-provider-value", "password": "hidden"}
        )
        assert runtime.store.event_by_id(eid)["payload"] == {
            "text": "[REDACTED]",
            "password": "[REDACTED]",
        }
        assert os.stat(runtime.store.directory / result["path"]).st_mode & 0o777 == 0o600
    finally:
        await runtime.shutdown()


async def test_container_arguments_and_recovery_cleanup_are_scoped(
    tmp_path, repository, coding_config, monkeypatch
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        config = coding_config.execution.model_copy(
            update={"backend": "container", "image": "toolchain:test", "read_only": True}
        )
        monkeypatch.setattr("threadweave.execution.shutil.which", lambda name: "/bin/" + name)
        argv, cwd, name = ContainerExecutor(config).invocation(
            context, ["python", "--version"], repository
        )
        assert "--read-only" in argv and "--cap-drop=ALL" in argv and "--network" in argv
        assert argv[argv.index("--network") + 1] == "none"
        assert str(repository) + ":/workspace:ro" in argv
        assert name.startswith("tw-") and cwd == repository
        runtime.store.event(
            session.id, "container_lease_started", {"name": name, "engine": "docker"}
        )
        cleaned = []

        async def cleanup(self, target):
            cleaned.append(target)
            return True

        monkeypatch.setattr(ContainerExecutor, "cleanup", cleanup)
        await recover_containers(runtime)
        await recover_containers(runtime)
        assert cleaned == [name]
        assert runtime.store.events(session.id, kind="container_lease_closed")
    finally:
        await runtime.shutdown()


async def test_optional_real_container_execution(tmp_path, repository, coding_config):
    engine = shutil.which("docker")
    if not engine:
        pytest.skip("Docker unavailable")
    process = await asyncio.create_subprocess_exec(
        engine,
        "image",
        "inspect",
        "python:3.12-slim",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(10):
            await process.wait()
    except TimeoutError:
        process.kill()
        await process.wait()
        pytest.skip("Docker engine unavailable")
    if process.returncode:
        pytest.skip(
            "Docker engine or preinstalled python:3.12-slim image unavailable; no implicit pulls"
        )
    coding_config.permissions.remove("python")
    coding_config.execution.image = "python:3.12-slim"
    coding_config.execution.backend = "container"
    coding_config.task.test_commands = [
        ["python", "-c", "import mathops; assert mathops.add(2,3) == 5"]
    ]
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        result = await ContainerExecutor(coding_config.execution).run(
            context, ["python", "-c", "print('real container')"]
        )
        assert result["passed"] and "real container" in result["stdout"]
        assert result["network_policy_enforced"]
    finally:
        await runtime.shutdown()


async def test_direct_python_edits_have_checkpoint_hashes_and_restore(
    tmp_path, repository, coding_config
):
    from threadweave.models import Action, new_id

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        result = await runtime._execute_action(
            session.id,
            new_id(),
            Action(
                name="python",
                arguments={
                    "code": "p = workspace / 'mathops.py'\np.write_text(p.read_text().replace('a - b', 'a + b'))"
                },
            ),
            context.source_event,
        )
        assert not result.get("error")
        evidence = runtime.store.events(session.id, kind="workspace_effects")[-1]["payload"]
        assert (
            evidence["files"]["mathops.py"]["before"]["hash"]
            != evidence["files"]["mathops.py"]["after"]["hash"]
        )
        GitWorkspace(context).restore(evidence["before_checkpoint"])
        assert "a - b" in (repository / "mathops.py").read_text()
    finally:
        await runtime.shutdown()


async def test_coding_fork_rebinds_python_paths_without_modifying_source(
    tmp_path, repository, coding_config
):
    from threadweave.models import new_id
    from threadweave.tools import ToolContext

    runtime, root, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        await runtime.execute_python(context, "source = workspace / 'mathops.py'\nsource")
        branch = await runtime.fork(root.id)
        branch_context = ToolContext(runtime, branch.id, new_id(), context.source_event)
        result = await runtime.execute_python(
            branch_context,
            "assert source.parent == workspace\nsource.write_text(source.read_text().replace('a - b', 'a + b'))",
        )
        assert not result["error"]
        assert "a - b" in (repository / "mathops.py").read_text()
        branch_context.action_id = new_id()
        imported = await runtime.execute_python(
            branch_context, "import mathops\nassert mathops.add(2, 3) == 5\nmathops.__file__"
        )
        assert not imported["error"] and branch.id != root.id
    finally:
        await runtime.shutdown()
