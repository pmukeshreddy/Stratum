"""Current production capabilities: component tests, not coding solve-rate evidence."""

import asyncio
import json
import sys

import pytest

from threadweave.kernel import Kernel
from threadweave.models import new_id
from threadweave.repository import RepositoryIndex
from threadweave.snapshots import SnapshotBlobs
from threadweave.storage import Store

from .conftest import eventually
from .test_kernel import bridge


@pytest.mark.parametrize("wait", ["while True: pass", "await asyncio.sleep(20)"])
async def test_targeted_interrupt_preserves_live_identity_and_receipt(tmp_path, wait):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        await kernel.execute(new_id(), "values = [123]", 5)
        pid, execution = kernel.process.pid, new_id()
        job = asyncio.create_task(
            kernel.execute(
                execution,
                "values.append(456)\nprint('MUTATION_COMMITTED', flush=True)\n" + wait,
                30,
            )
        )
        stdout = tmp_path / "kernel" / f"{execution}.stdout"
        await eventually(lambda: stdout.exists() and "MUTATION_COMMITTED" in stdout.read_text())
        assert await kernel.interrupt(execution)
        assert (await job)["error"]
        assert kernel.process.pid == pid
        assert kernel.receipt(execution)
        assert not await kernel.interrupt(execution)
        assert (await kernel.execute(new_id(), "values", 5))["value"] == "[123, 456]"
    finally:
        await kernel.close()


async def test_interrupt_blocked_host_rpc_and_next_cell(tmp_path):
    entered = asyncio.Event()

    async def blocked(name, arguments):
        entered.set()
        await asyncio.Event().wait()

    kernel = Kernel(tmp_path / "kernel", tmp_path, blocked)
    try:
        await kernel.execute(new_id(), "kept = 123", 5)
        pid, execution = kernel.process.pid, new_id()
        job = asyncio.create_task(kernel.execute(execution, "await tools.acall('wait')", 20))
        await asyncio.wait_for(entered.wait(), 5)
        assert await kernel.interrupt(execution)
        assert (await job)["error"]
        assert kernel.process.pid == pid
        assert (await kernel.execute(new_id(), "kept", 5))["value"] == "123"
    finally:
        await kernel.close()


async def test_duplicate_interrupt_does_not_restart_or_signal_next_execution(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        await kernel.execute(new_id(), "kept = 123", 5)
        pid, execution = kernel.process.pid, new_id()
        job = asyncio.create_task(kernel.execute(execution, "await asyncio.sleep(30)", 40))
        while kernel.active_execution != execution:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        assert any(await asyncio.gather(kernel.interrupt(execution), kernel.interrupt(execution)))
        assert (await job)["error"]
        assert (await kernel.execute(new_id(), "kept", 5))["value"] == "123"
        assert kernel.process.pid == pid
    finally:
        await kernel.close()


async def test_kernel_cleanup_finishes_when_owner_is_cancelled(tmp_path, monkeypatch):
    from threadweave.runtime import Runtime

    from .fakes import ScriptedProvider, TestConfig

    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Cleanup", tmp_path, config=TestConfig(), mode="interactive")
    kernel = runtime._kernel(root.id)
    await kernel.start()
    process = kernel.process
    entered, release = asyncio.Event(), asyncio.Event()
    original = kernel.close

    async def closing():
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(kernel, "close", closing)
    owner = asyncio.create_task(runtime._close_kernel(root.id))
    try:
        await entered.wait()
        owner.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert process.returncode is not None
        assert root.id not in runtime.kernels
    finally:
        release.set()
        await runtime.shutdown()


async def test_raw_python_detached_process_is_owned_and_cleaned(tmp_path):
    import psutil

    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        result = await kernel.execute(
            new_id(),
            "import subprocess, sys\nchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\nchild.pid",
            5,
        )
        pid = int(result["value"])
        assert any(
            p["pid"] == pid and p["source"] == "python_popen" for p in result["process_effects"]
        )
        assert psutil.pid_exists(pid)
    finally:
        await kernel.close()
    if psutil.pid_exists(pid):
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_numpy_snapshot_unchanged_and_inplace_edit(tmp_path):
    np = pytest.importorskip("numpy")
    from threadweave.kernel_worker import pack, unpack

    blobs = SnapshotBlobs(tmp_path)
    array = np.arange(100000, dtype=np.float64)
    blobs.begin()
    encoded, size = blobs.encode("array", array, pack)
    blobs.begin()
    assert blobs.encode("array", array, pack) == (encoded, size)
    assert blobs.stats["serialized_bytes"] == 0
    array[123] = 456
    changed, _ = blobs.encode("array", array, pack)
    assert changed != encoded
    assert blobs.decode(changed, unpack)[123] == 456


async def test_actual_clangd_semantic_definition(tmp_path, repository, coding_config):
    import shutil

    if not shutil.which("clangd"):
        pytest.skip("clangd not installed")
    from threadweave.lsp import query

    from .test_coding import setup_runtime

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        (repository / "main.cpp").write_text(
            "int answer() { return 42; }\nint main() { return answer(); }\n"
        )
        result = await query(context, "main.cpp", 2, 22)
        assert result["matches"]
        assert result["matches"][0]["range"]["start"]["line"] == 0
        assert result["matches"][0]["quality"] == "compiler/LSP semantic"
        assert runtime.store.events(session.id, kind="lsp_query")
    finally:
        await runtime.shutdown()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS OS-sandbox acceptance")
async def test_research_os_denies_python_and_subprocess_writes(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "input.py").write_text("value = 123\n")
    kernel = Kernel(tmp_path / "kernel", workspace, bridge, bootstrap={"research_read_only": True})
    try:
        assert (await kernel.execute(new_id(), "Path('input.py').read_text()", 5))[
            "value"
        ] == "'value = 123\\n'"
        for code in [
            "Path('input.py').write_text('bad')",
            "open('new.py', 'w')",
            "os.unlink('input.py')",
        ]:
            assert (await kernel.execute(new_id(), code, 5))["error"]["code"] == "PermissionError"
        result = await kernel.execute(
            new_id(),
            "import subprocess\nsubprocess.run(['sh', '-c', 'echo bad > input.py']).returncode",
            5,
        )
        assert result["value"] != "0"
        assert (workspace / "input.py").read_text() == "value = 123\n"
    finally:
        await kernel.close()


@pytest.mark.parametrize("transport", ["stdio", "http"])
async def test_readonly_mcp_cannot_launch_unconfined_transports(tmp_path, python_config, transport):
    from threadweave.mcp_client import McpManager
    from threadweave.models import Action, HarnessError, McpServerConfig
    from threadweave.runtime import Runtime

    from .fakes import ScriptedProvider

    python_config.execution.read_only = True
    python_config.permissions.append("mcp")
    marker = tmp_path / "server-started"
    python_config.mcp_servers = {
        "external": McpServerConfig(
            type=transport,
            command=sys.executable if transport == "stdio" else None,
            args=["-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            url="http://127.0.0.1:1/mcp" if transport == "http" else None,
        )
    }
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("Research", tmp_path, config=python_config)
        runtime.mcp = McpManager(runtime)
        event = runtime.store.event(root.id, "test_cell", {})
        for operation in ("mcp.tools", "mcp.call"):
            with pytest.raises(HarnessError, match="read-only research"):
                await runtime._execute_action(
                    root.id,
                    new_id(),
                    Action(
                        name="host_request",
                        arguments={
                            "operation": operation,
                            "payload": {"server": "external", "tool": "write"},
                        },
                    ),
                    event,
                    from_python=True,
                )
        assert not marker.exists()
        assert not runtime.mcp.connections
        assert len(runtime.store.events(root.id, kind="tool_result")) == 2
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize("container", [list, dict])
def test_mutable_checkpoint_cache_checks_values_and_types(tmp_path, container):
    from threadweave.kernel_worker import pack, unpack

    blob = SnapshotBlobs(tmp_path)
    value = list(range(100000)) if container is list else {i: i for i in range(100000)}
    blob.begin()
    first, size = blob.encode("value", value, pack)
    blob.begin()
    assert blob.encode("value", value, pack) == (first, size)
    assert blob.stats["serialized_bytes"] == 0
    assert blob.stats["mutable_cache_hits"] == 1
    value[1] = True  # Python equality would incorrectly conflate this with 1.
    blob.begin()
    changed, _ = blob.encode("value", value, pack)
    assert changed != first
    assert type(blob.decode(changed, unpack)[1]) is bool
    value[1] = 456
    assert blob.decode(blob.encode("value", value, pack)[0], unpack)[1] == 456


def test_python_import_aliases_jedi_and_persistent_relationships(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg/__init__.py").write_text("")
    (root / "pkg/core.py").write_text("def compute(x):\n    return x + 1\n")
    (root / "pkg/client.py").write_text(
        "from .core import compute as run\ndef use():\n    return run(2)\n"
    )
    store = Store(tmp_path / "data")
    index = RepositoryIndex(store, root)
    try:
        rows = index.callers("pkg/core.py::compute")["matches"]
        assert rows[0]["path"] == "pkg/client.py"
        assert rows[0]["quality"] == "resolved structural"
        assert index.dependencies("pkg/client.py")["likely_local_modules"] == ["pkg/core.py"]
        assert index.dependents("pkg/core.py")["matches"][0]["path"] == "pkg/client.py"
        inferred = index.resolve("pkg/client.py", 3, 13)["matches"]
        assert any(m["path"].endswith("pkg/core.py") and m["name"] == "compute" for m in inferred)
        # Removing a target invalidates its importers without rewriting their files.
        (root / "pkg/core.py").unlink()
        index.refresh(["pkg/core.py"])
        assert not index.dependencies("pkg/client.py")["likely_local_modules"]
    finally:
        index.close()
        store.close()


@pytest.mark.parametrize(
    "files,source,target",
    [
        (
            {"src/lib.rs": "mod engine;", "src/engine.rs": "pub fn run() {}"},
            "src/lib.rs",
            "src/engine.rs",
        ),
        (
            {
                "include/api.h": "int compute(int);",
                "src/main.cpp": '#include "api.h"\nint main() { return compute(1); }',
            },
            "src/main.cpp",
            "include/api.h",
        ),
        (
            {
                "tsconfig.json": json.dumps(
                    {"compilerOptions": {"baseUrl": ".", "paths": {"@core/*": ["src/*"]}}}
                ),
                "src/core.ts": "export function compute() {}",
                "app.ts": 'import {compute as run} from "@core/core"; run();',
            },
            "app.ts",
            "src/core.ts",
        ),
        (
            {
                "go.mod": "module example.org/project\n",
                "main.go": 'package main\nimport "example.org/project/pkg"',
                "pkg/a.go": "package pkg\nfunc Run() {}",
            },
            "main.go",
            "package:pkg",
        ),
    ],
)
def test_resolved_module_rules(tmp_path, files, source, target):
    root = tmp_path / "repo"
    root.mkdir()
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    store = Store(tmp_path / "data")
    index = RepositoryIndex(store, root)
    try:
        assert target in index.dependencies(source)["likely_local_modules"]
    finally:
        index.close()
        store.close()


async def test_parallel_candidate_admission_real_worktrees(
    tmp_path, repository, coding_config, monkeypatch
):
    import threading

    from .test_coding import setup_runtime

    runtime, parent, _ = await setup_runtime(tmp_path, repository, coding_config)
    barrier = threading.Barrier(2, timeout=10)
    original = runtime.environment.candidate_workspace
    entered = []

    def coordinated(*args):
        entered.append(args[1])
        barrier.wait()  # Both real admissions must be in-flight, not timing luck.
        return original(*args)

    monkeypatch.setattr(runtime.environment, "candidate_workspace", coordinated)
    try:
        children = await asyncio.gather(
            *(
                runtime.spawn_async(
                    parent.id,
                    "Investigate independently",
                    name=f"candidate{i}",
                    purpose="candidate",
                )
                for i in range(2)
            )
        )
        assert len(set(entered)) == 2
        assert len({c.workspace.path for c in children}) == 2
        for child in children:
            from pathlib import Path

            assert (Path(child.workspace.path) / ".git").is_file()
            assert runtime.store.config(child.id).task.require_verifier
            assert runtime.store.events(child.id, kind="candidate_ready")
            assert runtime.store.events(child.id, kind="child_context_package")
    finally:
        await runtime.shutdown()


async def test_coverage_relationships_and_discoverable_python_api(
    tmp_path, repository, coding_config
):
    from threadweave.test_selection import import_coverage, related, selection_reason

    from .test_coding import setup_runtime

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        (repository / "tests/conftest.py").write_text("from mathops import add\n")
        (repository / "coverage.json").write_text(
            json.dumps(
                {
                    "files": {
                        "mathops.py": {"contexts": {"2": ["tests/test_mathops.py::test_add|run"]}}
                    }
                }
            )
        )
        assert import_coverage(context, "coverage.json")["imported"] == 1
        selected = related(context, files=["mathops.py"])
        assert not any(s["target"] == "tests/conftest.py" for s in selected["selections"])
        assert selected["selections"][0]["quality"] == "runtime coverage"
        assert selection_reason(context, "tests/test_mathops.py::test_add")["event_id"]
        from .test_python_environment import cell

        result = await cell(
            runtime,
            session,
            "print(repo.help()); print(tests.help()); context.focus(files=['mathops.py'], hypothesis='Check addition')",
        )
        assert result.get("error") is None
        assert runtime.store.events(session.id, kind="working_focus")
    finally:
        await runtime.shutdown()


async def test_framework_ancestor_rootdir_yields_runnable_workspace_ids(
    tmp_path, repository, coding_config
):
    from threadweave.coding import run_command

    from .test_coding import setup_runtime

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        result = await run_command(
            context,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--rootdir",
                str(tmp_path),
                "tests/test_mathops.py",
            ],
            kind="test",
        )
        assert not result["passed"]
        failure = next(t for t in result["failures"] if t["test_id"].endswith("::test_add"))
        assert failure["file"] == "tests/test_mathops.py"
        assert failure["test_id"] == "tests/test_mathops.py::test_add"
        assert failure["reported_test_id"] != failure["test_id"]
        retained = runtime.artifacts.load(session.id, result["structured_artifact"])
        assert failure["reported_test_id"] in {t.get("reported_test_id") for t in retained["tests"]}
    finally:
        await runtime.shutdown()


def test_diagnostic_path_resolution_does_not_guess_or_escape(tmp_path):
    from threadweave.test_evidence import resolve_locations

    root = tmp_path / "repo"
    root.mkdir()
    (root / "source.py").write_text("pass\n")
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    evidence = {
        "diagnostics": [
            {"file": str(outside)},
            {"file": "elsewhere/source.py"},
            {"file": "/workspace/source.py"},
        ]
    }
    result = resolve_locations(evidence, root, container=True)["diagnostics"]
    assert [r["path_resolution"] for r in result] == ["unresolved", "unresolved", "workspace file"]
    assert result[2]["file"] == "source.py"


def test_message_limits_tied_order_and_receipt_metadata(tmp_path, monkeypatch):
    from .test_storage import make

    store = Store(tmp_path / "state")
    try:
        root = make(store, tmp_path)
        monkeypatch.setattr("threadweave.storage.now", lambda: 1000.0)
        ids = [store.send(None, root.id, str(i)) for i in range(110)]
        for invalid in (-1, 0, 1.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                store.messages(root.id, limit=invalid)
        assert len(store.messages(root.id, limit=10000)) == 100
        assert [m["id"] for m in store.messages(root.id, pending=True, limit=3)] == ids[:3]
        assert [m["id"] for m in store.messages(root.id, limit=3)] == list(reversed(ids[-3:]))
        delivered = store.receive(root.id, lambda m: m["body"], limit=3)
        assert all(m["received_at"] == 1000.0 for m in delivered)
        assert [m["id"] for m in store.messages(root.id, pending=True, limit=3)] == ids[3:6]
    finally:
        store.close()


async def test_verifier_cached_identity_does_not_hide_external_test_tampering(
    tmp_path, repository, coding_config
):
    from threadweave.coding import CodingTask

    from .test_coding import setup_runtime

    coding_config.task.capture_baseline = False
    coding_config.task.protect_tests = True
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        source = repository / "mathops.py"
        source.write_text(source.read_text().replace("return a - b", "return a + b"))
        task = CodingTask()
        assert (await task.verify(context, coding_config.task)).passed
        assert (await task.verify(context, coding_config.task)).passed
        # Deliberately lose watcher events: a verifier trust boundary still checks
        # metadata and must invalidate its cached changed-path set.
        observer = runtime.environment.mutations
        observer.trackers[str(repository.resolve())].close()
        (repository / "tests/test_mathops.py").write_text("def test_add():\n    assert True\n")
        verdict = await task.verify(context, coding_config.task)
        assert not verdict.passed
        assert any("Protected test modified" in v for v in verdict.details["violations"])
    finally:
        await runtime.shutdown()


async def test_python_context_surfaces_focus_and_keeps_negative_evidence(
    tmp_path, repository, coding_config
):
    from threadweave.context import python_instructions

    from .test_coding import setup_runtime

    coding_config.control_plane = "python"
    runtime, session, _ = await setup_runtime(tmp_path, repository, coding_config)
    try:
        runtime.store.event(
            session.id,
            "working_focus",
            {"files": ["mathops.py"], "hypothesis": "addition is not subtraction"},
        )
        eid = runtime.store.event(session.id, "failure", {"message": "expected 5, got -1"})
        runtime.store.update(
            session.id,
            context=[
                {
                    "event_id": eid,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hypothesis failed: expected 5, got -1. Must not weaken tests.",
                        }
                    ],
                }
            ],
        )
        runtime.context.compact(session.id, summary="Investigating arithmetic", provenance=eid)
        messages = runtime.context.messages(session.id)
        text = json.dumps(messages)
        assert "expected 5, got -1" in text
        assert "source_evidence" in text and "def add(a, b)" in text
        runtime.store.event(
            session.id,
            "coding_command",
            {
                "passed": False,
                "structured_artifact": "external-full-evidence",
                "failures": [
                    {
                        "test_id": "test_add",
                        "file": "mathops.py",
                        "line": 2,
                        "message": "expected 5, got -1",
                        "stack": "UNNEEDED_RAW_STACK" * 1000,
                    }
                ],
            },
        )
        focused = json.dumps(runtime.context.messages(session.id))
        assert "expected 5, got -1" in focused
        assert "external-full-evidence" in focused
        assert "UNNEEDED_RAW_STACK" not in focused
        basic = coding_config.model_copy(deep=True)
        basic.features.subagents = False
        basic.features.enhanced_code_index = False
        base = python_instructions(basic)
        assert "await edit(" not in base and "await rlm(" not in base
        assert "Path.read_text()/write_text()" in base
        readonly = coding_config.model_copy(deep=True)
        readonly.execution.read_only = True
        assert "Host shell execution is not" in python_instructions(readonly)
        assert "or await bash(command)" not in python_instructions(readonly)
        assert runtime.store.event_by_id(eid)["payload"]["message"] == "expected 5, got -1"
    finally:
        await runtime.shutdown()


def test_changed_module_settings_rebind_without_reparsing_importer(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "first.ts").write_text("export const answer = 1;")
    (root / "second.ts").write_text("export const answer = 2;")
    (root / "app.ts").write_text('import {answer} from "@answer"; console.log(answer);')
    config = root / "tsconfig.json"
    config.write_text(json.dumps({"compilerOptions": {"paths": {"@answer": ["first.ts"]}}}))
    store = Store(tmp_path / "state")
    index = RepositoryIndex(store, root)
    try:
        assert index.dependencies("app.ts")["likely_local_modules"] == ["first.ts"]
        config.write_text(json.dumps({"compilerOptions": {"paths": {"@answer": ["second.ts"]}}}))
        changed = index.refresh(["tsconfig.json"])
        assert "app.ts" not in changed
        assert index.dependencies("app.ts")["likely_local_modules"] == ["second.ts"]
    finally:
        index.close()
        store.close()


def test_import_syntax_ignores_comments_and_handles_multiline_aliases():
    from threadweave.module_imports import imports

    assert imports(
        '/* import {bad} from "hidden"; */\nimport Default, {\n value as local, type Kind\n} from "./core";',
        "typescript",
    ) == [
        ("./core", "Default", "default"),
        ("./core", "local", "value"),
        ("./core", "Kind", "Kind"),
    ]
    assert imports(
        "use crate::engine::{self, nested::{run as go, Thing}}; // use secret::answer;", "rust"
    ) == [
        ("crate::engine", "engine", ""),
        ("crate::engine::nested", "go", "run"),
        ("crate::engine::nested", "Thing", "Thing"),
    ]
    assert imports('// #include "secret.h"\n#include "api.h"', "cpp") == [("api.h", "", "*")]


def test_pytest_collection_diagnostic_retains_module_identity():
    from threadweave.test_evidence import junit

    xml = '<testsuite><testcase name="tests.test_parser" time="0"><error message="collection failure">ImportError: Parser unavailable</error></testcase></testsuite>'
    row = junit(xml, "pytest")[0]
    assert row["file"] == "tests/test_parser.py"
    assert row["test_id"] == "tests/test_parser.py"
    assert "Parser unavailable" in row["stack"]


async def test_last_admitted_child_turn_not_cancelled_by_exhausted_sibling(tmp_path):
    from threadweave.models import ModelResponse, Outcome
    from threadweave.runtime import Runtime

    from .conftest import eventually, response
    from .fakes import ScriptedProvider, TestConfig

    entered, release = asyncio.Event(), asyncio.Event()

    async def last_turn(request):
        entered.set()
        await release.wait()
        return ModelResponse(text="Child findings retained")

    provider = ScriptedProvider({"root": [response("ipython", code="x=123")], "child": [last_turn]})
    config = TestConfig(
        control_plane="python",
        limits={"max_turns": 2, "token_budget": 100000, "wall_seconds": 30, "concurrency": 2},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("Root", tmp_path, config=config)
        child = runtime.spawn(root.id, "Child", name="child", purpose="shared")
        await runtime.start()
        await asyncio.wait_for(entered.wait(), 5)
        # The parent reaches the admission gate while the last permitted model
        # invocation remains blocked. No timing assumption establishes overlap.
        await eventually(
            lambda: runtime.store.session(root.id).turns == 1 and root.id not in runtime.tasks
        )
        await runtime._run_turn(root.id)
        assert runtime.store.session(child.id).outcome == Outcome.ACTIVE
        release.set()
        result = await runtime.wait(root.id, timeout=10)
        assert result.outcome == Outcome.LIMITED
        assert runtime.store.session(child.id).outcome == Outcome.COMPLETED
        assert runtime.store.session(child.id).result == "Child findings retained"
        assert runtime.store.usage(root.id, tree=True).turns == 2
        assert len(provider.requests) == 2
    finally:
        release.set()
        await runtime.shutdown()


async def test_live_probe_api_errors_have_local_actionable_help(
    tmp_path, repository, coding_config
):
    from .test_coding import setup_runtime
    from .test_python_environment import cell

    coding_config.control_plane = "python"
    runtime, session, _ = await setup_runtime(tmp_path, repository, coding_config)
    try:
        result = await cell(
            runtime, session, "print(agent_message.help()); print(repo.help('search'))"
        )
        result = runtime.artifacts.load(session.id, result["artifact_id"])
        assert result.get("error") is None
        assert "receiver_role" in result["stdout"] and "path=" in result["stdout"]
        result = await cell(runtime, session, "repo.search('add', glob='*.py')")
        result = runtime.artifacts.load(session.id, result["artifact_id"])
        assert "Use path=" in result["error"]["message"]
        result = await cell(
            runtime,
            session,
            "packet = repo.context_for_symbol('add'); print(packet['sources'][0]['source'])",
        )
        result = runtime.artifacts.load(session.id, result["artifact_id"])
        assert "return a - b" in result["stdout"]
        result = await cell(
            runtime,
            session,
            "matches = await repo.search('add', path='mathops.py'); print(matches['total'])",
        )
        result = runtime.artifacts.load(session.id, result["artifact_id"])
        assert result["error"] is None
        calls = [
            e
            for e in runtime.store.events(session.id, kind="tool_call", limit=100)
            if e["payload"]["name"] == "repo_search"
        ]
        assert len(calls) == 1  # Invalid glob rejected locally; await executes once.
        assert [s["function"]["name"] for s in runtime.tools.schemas(coding_config)] == ["ipython"]
    finally:
        await runtime.shutdown()
