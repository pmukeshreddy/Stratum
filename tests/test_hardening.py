"""Production components on real local repositories; model doubles, not model evaluation."""

import asyncio
import json

import pytest

from threadweave.gitops import GitWorkspace, git
from threadweave.kernel import Kernel
from threadweave.models import new_id
from threadweave.repository import RepositoryIndex, symbols
from threadweave.storage import Store
from threadweave.test_evidence import machine_command, structured
from threadweave.test_selection import related
from threadweave.tokenization import estimate

from .test_coding import setup_runtime
from .test_kernel import bridge
from .test_python_environment import cell, coding_python, effects  # noqa: F401 - pytest fixture


async def test_warm_observation_no_enumeration_and_overflow_reconciles(coding_python, monkeypatch):  # noqa: F811
    runtime, session, context = coding_python
    await cell(runtime, session, "x = 1")
    observer = runtime.environment.mutations
    tracker = next(iter(observer.trackers.values()))
    import threadweave.mutations as module

    original = module.git
    enumerated = []

    def monitored(*args, **kwargs):
        if "ls-files" in args:
            enumerated.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "git", monitored)
    await cell(runtime, session, "x += 1")
    assert not enumerated
    tracker.uncertain = True  # overflow/lost stream is reconciled before continuing
    context.path("out-of-stream.txt").write_text("recovered")
    observer.reconcile(context, reason="background_reconciliation")
    assert enumerated
    assert (
        "out-of-stream.txt"
        in effects(runtime, session.id, reason="background_reconciliation")[-1]["files"]
    )


@pytest.mark.parametrize(
    "language,code,name",
    [
        ("python", "def outer():\n    return helper()", "outer"),
        ("rust", "pub fn outer() { helper(); }", "outer"),
        ("go", "package p\nfunc Outer() { helper() }", "Outer"),
        ("c", "int outer(void) { return helper(); }", "outer"),
        ("cpp", "class A { int outer() { return helper(); } };", "A.outer"),
        ("cuda", "__global__ void outer() { helper(); }", "outer"),
        ("javascript", "function outer() { return helper(); }", "outer"),
        ("typescript", "class A { outer(): number { return helper(); } }", "A.outer"),
    ],
)
def test_real_syntax_parsers(language, code, name):
    body = symbols(code, language)
    assert body["quality"] == "syntax-derived"
    assert name in [s["name"] for s in body["symbols"]]
    assert any(c["name"] == "helper" for c in body["calls"])
    assert any(r["name"] == "helper" for r in body["references"])
    if language == "cuda":
        assert body["parser"] == "tree_sitter_cpp"
        assert body["dialect"] == "cuda_cpp_syntax_subset"


def test_incremental_index_rename_delete_parse_error_and_recovery(
    tmp_path, repository, monkeypatch
):
    store = Store(tmp_path / "index")
    index = RepositoryIndex(store, repository)
    try:
        index.entries()
        monkeypatch.setattr(index, "paths", lambda: pytest.fail("warm lookup traversed repository"))
        assert index.definition("add")["matches"]
        (repository / "new.rs").write_text("fn compute() { helper(); }")
        assert index.callers("helper")["matches"][0]["path"] == "new.rs"
        (repository / "new.rs").rename(repository / "moved.rs")
        assert index.definition("compute")["matches"][0]["path"] == "moved.rs"
        (repository / "moved.rs").write_text("fn compute() { helper(;")
        assert index.outline("moved.rs")["parse_error"]
        (repository / "moved.rs").unlink()
        assert not index.definition("compute")["matches"]
    finally:
        index.close()
        store.close()
    store = Store(tmp_path / "index")
    index = RepositoryIndex(store, repository)
    try:
        assert index.definition("add")["matches"]
    finally:
        index.close()
        store.close()


async def test_worktree_dirty_parent_parallel_candidates_and_cleanup(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        (repository / "mathops.py").write_text("def add(a,b): return a+b\n")
        (repository / "binary").write_bytes(bytes(range(256)))
        git(repository, "add", "binary")
        before = git(repository, "diff", "--cached", "--binary")
        parent = GitWorkspace(context)
        checkpoint = parent.snapshot_tree()
        children = [parent.isolate(checkpoint) for _ in range(4)]
        assert all((p / ".git").is_file() for p in children)
        assert all((p / "binary").read_bytes() == bytes(range(256)) for p in children)
        (children[0] / "mathops.py").write_text("candidate")
        assert (children[1] / "mathops.py").read_text() != "candidate"
        assert (repository / "mathops.py").read_text() != "candidate"
        assert git(repository, "diff", "--cached", "--binary") == before
        for child in children:
            parent.cleanup_isolation(child)
            assert not child.exists()
    finally:
        await runtime.shutdown()


async def test_explicit_child_purposes(tmp_path, repository, coding_config):
    runtime, session, _ = await setup_runtime(tmp_path, repository, coding_config)
    try:
        coding_config.control_plane = "python"
        researcher = runtime.spawn(session.id, "Read only", purpose="research")
        candidate = runtime.spawn(session.id, "Make a candidate", purpose="candidate")
        shared = runtime.spawn(session.id, "Collaborate", purpose="shared")
        assert researcher.workspace.path == shared.workspace.path == session.workspace.path
        assert runtime.store.config(researcher.id).execution.read_only
        assert candidate.workspace.path != session.workspace.path
        assert runtime.store.config(candidate.id).task.adapter == "coding"
        assert runtime.store.events(candidate.id, kind="child_purpose")
    finally:
        await runtime.shutdown()


def test_model_tokenization_is_not_byte_length():
    text = "def parse(value): return value.strip()\n" * 500
    assert estimate(text, "gpt-6-astra") < len(text.encode()) / 2
    assert estimate(text, "unknown-provider-model") >= len(text.encode())
    assert estimate("你好" * 500, "gpt-6-astra") > 0


@pytest.mark.parametrize(
    "name,text,xml,expected",
    [
        (
            "pytest",
            "",
            '<testsuite><testcase classname="tests.test_p" name="test_x" time=".2"><failure message="wrong">AssertionError</failure></testcase></testsuite>',
            "failed",
        ),
        ("go", '{"Action":"fail","Package":"p","Test":"TestX","Elapsed":0.1}', None, "failed"),
        (
            "jest",
            '{"testResults":[{"name":"a.test.ts","assertionResults":[{"fullName":"x","status":"passed"}]}]}',
            None,
            "passed",
        ),
        (
            "vitest",
            '{"testResults":[{"assertionResults":[{"title":"x","status":"failed"}]}]}',
            None,
            "failed",
        ),
        ("ctest", "", '<testsuite><testcase name="test_x"/></testsuite>', "passed"),
    ],
)
def test_framework_machine_evidence(name, text, xml, expected):
    result = structured(text, name, xml)
    assert result["tests"][0]["status"] == expected
    assert result["tests"][0]["provenance"] != "text_location"


def test_rust_compiler_json_and_malformed_report():
    text = json.dumps(
        {
            "reason": "compiler-message",
            "message": {
                "level": "error",
                "message": "bad type",
                "code": {"code": "E0308"},
                "spans": [
                    {
                        "is_primary": True,
                        "file_name": "src/lib.rs",
                        "line_start": 4,
                        "column_start": 2,
                    }
                ],
            },
        }
    )
    assert structured(text, "cargo")["diagnostics"][0]["diagnostic_code"] == "E0308"
    assert structured("", "pytest", "broken xml")["parse_errors"]
    assert machine_command(["go", "test", "./..."], None)[1] == ["go", "test", "-json", "./..."]


async def test_related_tests_have_evidence_and_verifier_remains_full(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        result = related(context, files=["mathops.py"])
        assert any(
            s["target"] == "tests/test_mathops.py" and s["reasons"] for s in result["selections"]
        )
        assert result["final_verifier_unchanged"]
        assert (
            runtime.store.config(session.id).task.test_commands == coding_config.task.test_commands
        )
    finally:
        await runtime.shutdown()


async def test_procedure_blob_restore_corruption_isolation_and_immutable_cache(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        result = await kernel.execute(
            new_id(),
            "def double(x): return 2*x\nclass Counter:\n    def value(self): return 123\ncounter = Counter()\nlarge = b'x' * 1000000\nhandle = open('temporary', 'w')",
            10,
        )
        assert "handle" in result["not_checkpointed"]
        second = await kernel.execute(new_id(), "small = 1", 10)
        assert second["snapshot_metrics"]["cache_hits"] >= 1
        assert second["snapshot_metrics"]["written_bytes"] < 10000
        assert second["snapshot_metrics"]["serialized_bytes"] < 10000
    finally:
        await kernel.close()
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        assert (await kernel.execute(new_id(), "double(counter.value())", 10))["value"] == "246"
        manifest = json.loads((tmp_path / "kernel/checkpoint.json").read_text())
        blob = manifest["values"]["double"][1]["sha256"]
    finally:
        await kernel.close()
    (tmp_path / "kernel/values" / blob).write_bytes(b"corrupted")
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        await kernel.start()
        assert "double" in kernel.recovery["missing"]
        assert (await kernel.execute(new_id(), "counter.value()", 10))["value"] == "123"
    finally:
        await kernel.close()


async def test_stale_interrupt_cannot_hit_next_cell(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        first = new_id()
        await kernel.execute(first, "x = 123", 5)
        assert not await kernel.interrupt(first)
        assert (await kernel.execute(new_id(), "x", 5))["value"] == "123"
        current = new_id()
        pid = kernel.process.pid
        running = asyncio.create_task(kernel.execute(current, "await asyncio.sleep(30)", 40))
        while kernel.active_execution != current:  # noqa: ASYNC110 - inspect actual production admission
            await asyncio.sleep(0.01)
        assert await kernel.interrupt(current)
        assert (await running)["error"]["code"] in {"CancelledError", "KeyboardInterrupt"}
        assert kernel.process.pid == pid
        assert (await kernel.execute(new_id(), "x", 5))["value"] == "123"
    finally:
        await kernel.close()


def test_index_preserved_mtime_and_corrupt_metadata(tmp_path, repository):
    import os

    store = Store(tmp_path / "index")
    index = RepositoryIndex(store, repository)
    try:
        index.entries()
        path = repository / "mathops.py"
        before = path.stat()
        path.write_text(path.read_text().replace("add", "sum"))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert index.definition("sum")["matches"]
        store.db.execute("UPDATE repository_files SET body='broken' WHERE path='mathops.py'")
        index.refresh()
        assert index.outline("mathops.py")["symbols"]
    finally:
        index.close()
        store.close()


def test_jsonl_and_text_framework_diagnostics():
    stream = "\n".join(
        [
            json.dumps(
                {"Action": "output", "Package": "p", "Test": "TestX", "Output": "expected 1, got 2"}
            ),
            json.dumps({"Action": "fail", "Package": "p", "Test": "TestX"}),
        ]
    )
    result = structured(stream, "go")
    assert result["failures"][0]["message"] == "expected 1, got 2"
    assert structured("test_a (module.Case) ... FAIL", "unittest")["tests"][0]["status"] == "failed"
    assert (
        structured("src/a.ts(4,5): error TS123: wrong", "compiler")["diagnostics"][0]["column"] == 5
    )


async def test_remaining_budget_compaction_keeps_policy_and_durable_evidence(tmp_path):
    from threadweave.models import ModelResponse, RunConfig, Usage
    from threadweave.runtime import Runtime

    from .fakes import ScriptedProvider

    provider = ScriptedProvider(
        {"root": [ModelResponse(text="done", usage=Usage(input_tokens=10, output_tokens=5))]}
    )
    config = RunConfig(
        provider={"name": "mock", "model": "gpt-6-astra", "max_output_tokens": 512},
        limits={"token_budget": 20000},
        features={"model_compaction": False},
    )
    runtime = Runtime(tmp_path / "data", providers={"mock": provider})
    try:
        session = runtime.create("retain constraints", tmp_path, config=config, mode="interactive")
        ids = []
        for i in range(15):
            event = runtime.store.event(session.id, "observation", {"detail": i})
            ids.append(event)
            runtime.store.add_context(
                session.id,
                event,
                [{"role": "user", "content": " ".join(str(j * i) for j in range(700))}],
            )
        runtime.store.charge(session.id, Usage(input_tokens=12000))
        await runtime._invoke(session.id)
        assert runtime.store.events(session.id, kind="budget_context_pressure")
        assert runtime.store.session(session.id).summary
        assert all(runtime.store.event_by_id(e) for e in ids)
        assert runtime.store.config(session.id).limits.token_budget == 20000
        assert runtime.store.usage(session.id).input_tokens == 12010
        assert [s["function"]["name"] for s in provider.requests[0].tools] == ["ipython"]
    finally:
        await runtime.shutdown()


async def test_frozen_comparison_and_tamper_detection(tmp_path, repository, coding_config):
    import sys

    from threadweave.evaluation import profile_config
    from threadweave.frozen_eval import compare, freeze, validate
    from threadweave.models import ModelResponse

    from .conftest import response
    from .fakes import ScriptedProvider

    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {
                    "id": "component-fixture",
                    "adapter": "repository_issue",
                    "repository": str(repository),
                    "objective": "Fix addition",
                    "test_commands": [
                        [sys.executable, "-c", "from mathops import add; assert add(2,3)==5"]
                    ],
                }
            ]
        )
    )
    coding_config.control_plane = "python"
    coding_config.features.model_compaction = False
    bundle = tmp_path / "frozen"
    freeze(tasks, coding_config, bundle)
    provider = ScriptedProvider(
        {
            "root": [
                response(
                    "ipython", code="Path('mathops.py').write_text('def add(a,b): return a+b\\n')"
                ),
                ModelResponse(text="done"),
            ]
        }
    )
    result = await compare(bundle, tmp_path / "runs", providers={"test": provider})
    assert all(r["solved"] == 1 for r in result["profiles"].values())
    assert profile_config(coding_config, "base").limits == coding_config.limits
    (bundle / "tasks.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen input changed"):
        validate(bundle)


async def test_background_recovery_marks_lost_without_replaying(
    tmp_path, repository, coding_config
):
    from threadweave.background import BackgroundProcesses

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        service = BackgroundProcesses(runtime)
        service.save("lost", session.id, {"running": True, "command": "never replay", "id": "lost"})
        service.recover()
        assert service.status(context, "lost")["state"] == "lost"
        assert not service.tasks
        assert runtime.store.events(session.id, kind="process_recovery")
    finally:
        await runtime.shutdown()


async def test_external_adapter_large_response_retains_logs(tmp_path, repository, coding_config):
    import sys

    from threadweave.evaluation import run_external

    runtime, session, _ = await setup_runtime(tmp_path, repository, coding_config)
    try:
        with pytest.raises(ValueError, match="exceeds 4 MB"):
            await run_external(
                runtime,
                session,
                "Response-size boundary",
                coding_config,
                [sys.executable, "-c", "import sys; sys.stdin.read(); print('x'*4_000_001)"],
            )
        assert (
            runtime.store.db.execute(
                "SELECT count(*) FROM artifacts WHERE session_id=? AND size>4000000", (session.id,)
            ).fetchone()[0]
            == 1
        )
    finally:
        await runtime.shutdown()


async def test_native_array_snapshot_restores_and_isolated_resource_failure(tmp_path):
    kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        result = await kernel.execute(
            new_id(),
            "import array\nvalues = array.array('d', range(10000))\nresource = open('opaque', 'w')",
            10,
        )
        assert not result.get("error")
        assert "resource" in result["not_checkpointed"]
        await kernel.close()
        kernel = Kernel(tmp_path / "kernel", tmp_path, bridge)
        result = await kernel.execute(new_id(), "print(len(values), values[-1])", 10)
        assert "10000 9999.0" in result["stdout"]
    finally:
        await kernel.close()


def test_trajectory_errors_are_read_from_structured_tool_result():
    from threadweave.trajectory_analysis import analyze_events

    result = analyze_events(
        [{"id": "failed", "type": "tool_result", "payload": {"result": {"error": "denied"}}}]
    )
    assert result["failed_actions"] == [{"event_id": "failed", "error": "denied"}]


def test_distribution_and_failed_correctness_cannot_claim_improvement():
    import statistics

    from threadweave.benchmarks import compare, summarize
    from threadweave.models import BenchmarkConfig

    values = list(range(1, 101))
    result = summarize(values)
    assert result["p50"] == result["median"] == 50.5
    assert result["p95"] == 95 and result["p99"] == 99
    assert result["variance"] == statistics.variance(values)
    assert result["stdev"] == statistics.stdev(values)
    with pytest.raises(ValueError, match="correctness"):
        compare(
            {"median": 100, "correct": True},
            {"median": 1, "correct": False},
            BenchmarkConfig(command=["unused"], metric_regex=r"metric=(\d+)"),
        )


async def test_managed_observed_detached_descendant_is_reaped(tmp_path, repository, coding_config):
    import sys

    import psutil

    from threadweave.execution import LocalExecutor

    from .conftest import eventually

    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        # Keep the intermediary alive until the production supervisor has observed
        # its detached child; fast unobserved double-fork escapes remain a limit.
        script = (
            "import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
            "print(p.pid,flush=True); time.sleep(0.5)"
        )
        result = await LocalExecutor().run(context, [sys.executable, "-c", script])
        assert result["passed"]
        pid = int(result["stdout"].strip())

        def ended():
            try:
                return (
                    not psutil.Process(pid).is_running()
                    or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                return True

        await eventually(ended)
    finally:
        await runtime.shutdown()
