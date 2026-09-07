import asyncio
import json
import os
import sqlite3
import sys

import pytest

from threadweave.benchmarks import compare, run_benchmark, summarize
from threadweave.coding import CodingTask, baseline, run_checks
from threadweave.diagnostics import localize
from threadweave.editing import Editor, recover_edits, unified_changes
from threadweave.execution import LocalExecutor, environment
from threadweave.experiments import Experiments
from threadweave.gitops import GitWorkspace, candidate_result, git
from threadweave.guardrails import observe
from threadweave.models import BenchmarkConfig, new_id
from threadweave.refinement import validate_skill
from threadweave.repository import RepositoryIndex, confined, symbols
from threadweave.retrieval import search
from threadweave.runtime import Runtime
from threadweave.storage import Store, encode
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider


async def setup_runtime(tmp_path, repository, config):
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    session = runtime.create("Fix arithmetic without weakening tests", repository, config=config)
    event = runtime.store.event(session.id, "test_setup", {})
    context = ToolContext(runtime, session.id, new_id(), event)
    await runtime._prepare(session.id)
    return runtime, session, context


FIX = "--- a/mathops.py\n+++ b/mathops.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"


def test_incremental_index_ast_navigation_and_fallback(repository, tmp_path, monkeypatch):
    store = Store(tmp_path / "db")
    index = RepositoryIndex(store, repository)
    assert "mathops.py" in index.refresh()
    assert index.refresh() == []
    assert index.symbol_search("add")["matches"][0]["line"] == 1
    assert index.outline("mathops.py")["parser"] == "python_ast"
    assert index.dependencies("tests/test_mathops.py")["likely_local_modules"] == ["mathops.py"]
    (repository / "node_modules").mkdir()
    (repository / "node_modules/ignored.py").write_text("def ignored(): pass")
    assert not index.symbol_search("ignored")["matches"]
    path = repository / "mathops.py"
    path.write_text(path.read_text() + "\nclass Calculator:\n    def evaluate(self): return 1\n")
    assert index.refresh() == ["mathops.py"]
    assert index.symbol_search("Calculator.evaluate")["matches"]
    assert index.search("return", language="python", context=1, limit=1)["total"] == 3
    monkeypatch.setattr("threadweave.repository.shutil.which", lambda _: None)
    assert index.search("add", path="mathops.py")["engine"] == "python"
    store.close()
    reopened = Store(tmp_path / "db")
    assert RepositoryIndex(reopened, repository).refresh() == []
    reopened.close()


@pytest.mark.parametrize(
    "language,code,name",
    [
        ("rust", "pub fn parse() {}", "parse"),
        ("go", "func (p Parser) Parse() {}", "Parse"),
        ("cpp", "int sum(int a) { return a; }", "sum"),
        ("typescript", "export class Parser {}", "Parser"),
        ("cuda", "struct Kernel {};", "Kernel"),
    ],
)
def test_multilanguage_lexical_navigation(language, code, name):
    assert symbols(code, language)["symbols"][0]["name"] == name


@pytest.mark.parametrize(
    "patch",
    [
        "not a patch",
        "--- a/mathops.py\n+++ b/mathops.py\n",
        FIX.replace("@@ -1,2 +1,2", "@@ -1,4 +1,2"),
        FIX.replace("return a - b", "return a * b"),
        FIX.replace("@@ -1,2", "@@ -90,2"),
    ],
)
def test_patch_rejects_malformed_and_stale_input(repository, patch):
    original = (repository / "mathops.py").read_bytes()
    with pytest.raises(ValueError):
        unified_changes(patch, lambda p: (repository / p).read_bytes())
    assert (repository / "mathops.py").read_bytes() == original


def test_patch_creation_deletion_and_no_final_newline():
    patch = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x=1\n\\ No newline at end of file\n"
    assert unified_changes(patch, lambda _: None) == {"new.py": b"x=1"}
    delete = "--- a/new.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x=1\n\\ No newline at end of file\n"
    assert unified_changes(delete, lambda _: b"x=1") == {"new.py": None}


async def test_edits_hashes_checkpoints_rollback_and_prepared_recovery(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        original_head = git(repository, "rev-parse", "HEAD")
        checkpoint = GitWorkspace(context).snapshot("before-fix")
        result = Editor(context).apply_patch(FIX)
        assert (
            result["files"]["mathops.py"]["before_hash"]
            != result["files"]["mathops.py"]["after_hash"]
        )
        assert "+    return a + b" in GitWorkspace(context).diff(checkpoint)
        assert runtime.store.events(session.id, kind="code_edit")
        Editor(context).rollback(result["edit_id"])
        assert not GitWorkspace(context).diff(checkpoint)
        Editor(context).apply_patch(FIX)
        restored = GitWorkspace(context).restore(checkpoint)
        assert (
            restored["safety_checkpoint"]
            and "return a - b" in (repository / "mathops.py").read_text()
        )
        assert git(repository, "rev-parse", "HEAD") == original_head
        assert not git(repository, "diff", "--cached")
        pending = Editor(context).apply_patch(FIX)
        body = {k: v for k, v in pending.items() if k != "edit_id"}
        body["status"] = "prepared"
        runtime.store.db.execute(
            "UPDATE edits SET body=? WHERE id=?", (encode(body), pending["edit_id"])
        )
        recover_edits(runtime)
        assert "return a - b" in (repository / "mathops.py").read_text()
        assert runtime.store.events(session.id, kind="edit_recovery")
    finally:
        await runtime.shutdown()


async def test_atomic_patch_and_rollback_conflict_preserves_human_changes(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        with pytest.raises(ValueError):
            Editor(context).apply_patch(
                FIX + "--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-x\n+y\n"
            )
        assert "a - b" in (repository / "mathops.py").read_text()
        result = Editor(context).apply_patch(FIX)
        (repository / "mathops.py").write_text("human_edit = True\n")
        with pytest.raises(ValueError, match="Rollback conflict"):
            Editor(context).rollback(result["edit_id"])
        assert (repository / "mathops.py").read_text() == "human_edit = True\n"
    finally:
        await runtime.shutdown()


async def test_baseline_real_test_failure_localization_and_independent_verifier(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        recorded = baseline(context)
        assert recorded["results"]["test"][0]["passed"] is False
        failed = await CodingTask().verify(context, coding_config.task)
        assert not failed.passed
        actual = await run_checks(context, "test", targets=["tests/test_mathops.py::test_add"])
        assert not actual["passed"] and actual["failures"]
        assert localize(context, actual)["evidence"]
        Editor(context).apply_patch(FIX)
        passed = await CodingTask().verify(context, coding_config.task)
        assert passed.passed and passed.metrics["diff_bytes"] > 0
        Editor(context).apply({"tests/test_mathops.py": b"def test_nothing(): pass\n"})
        rejected = await CodingTask().verify(context, coding_config.task)
        assert not rejected.passed
        assert any(
            "Protected test" in v or "definitions deleted" in v
            for v in rejected.details["violations"]
        )
    finally:
        await runtime.shutdown()


async def test_isolated_candidate_patch_transfer_and_accounting(
    tmp_path, repository, coding_config
):
    runtime, root, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        child = runtime.spawn(root.id, "Repair arithmetic", name="candidate")
        assert child.workspace.path != root.workspace.path
        assert child.workspace.metadata["source_session"] == root.id
        child_context = ToolContext(runtime, child.id, new_id(), context.source_event)
        await runtime._prepare(child.id)
        Editor(child_context).apply_patch(FIX)
        assert "a - b" in (repository / "mathops.py").read_text()
        await runtime.pause(child.id)
        result = candidate_result(context, child.id)
        assert "+    return a + b" in result["patch"]
        candidate_result(context, child.id, accept=True)
        assert (await CodingTask().verify(context, coding_config.task)).passed
        row = json.loads(
            runtime.store.db.execute(
                "SELECT body FROM candidates WHERE child_id=?", (child.id,)
            ).fetchone()[0]
        )
        assert row["accepted"] and row["consumed"] and row["useful"]
        assert runtime.store.usage(root.id, tree=True).subagent_count == 1
    finally:
        await runtime.shutdown()


async def test_full_coding_model_loop_recursive_parallel_recovery(
    tmp_path, repository, coding_config
):
    directory = tmp_path / "runtime"
    runtime = Runtime(directory)
    child_patch = '--- a/mathops.py\n+++ b/mathops.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n+    """Add two numbers."""\n     return a + b\n'

    def consume(request):
        child = next(s for s in runtime.store.sessions(root_id=request.root_id) if s.parent_id)
        if child.outcome == "active":
            return response("agent_wait", seconds=0.15)
        return response("candidate_apply", child_id=child.id)

    scripts = {
        "root": [
            response("repo_search", query="return"),
            response("symbol_search", query="add"),
            response("apply_patch", patch=FIX),
            response("run_targeted_tests", targets=["tests/test_mathops.py::test_add"]),
            response("git_checkpoint", label="fixed"),
            response(
                "agent_spawn", instruction="Add a useful docstring and test it", name="candidate"
            ),
            response("python", code="retained = {'fixed': True}\nretained"),
            response("agent_wait", seconds=0.5),
            consume,
            response("agent_wait", seconds=0.5),
            consume,
            response("finish", result="Arithmetic fixed and verified"),
        ],
        "candidate": [
            response("apply_patch", patch=child_patch),
            response("run_tests"),
            response("finish", result="Docstring added; tests pass"),
        ],
    }
    provider = ScriptedProvider(scripts, delay=0.3)
    runtime.providers["test"] = provider
    root = runtime.create(
        "Repair arithmetic and retain verification evidence", repository, config=coding_config
    )
    try:
        await runtime.start()
        await eventually(lambda: runtime.store.session(root.id).turns >= 7, seconds=20)
        await runtime.pause(root.id)
        await eventually(
            lambda: not any(s.id in runtime.tasks for s in runtime.store.sessions(root_id=root.id))
        )
        runtime.context.compact(root.id)
        runtime.message(None, root.id, "Continue the same run after restart")
        tree = {
            s.id: (s.parent_id, s.kernel_id, s.workspace.path)
            for s in runtime.store.sessions(root_id=root.id)
        }
        count = runtime.store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        await runtime.shutdown()
        runtime = Runtime(directory, providers={"test": provider})
        runtime.resume(root.id)
        await runtime.start()
        result = await runtime.wait(root.id, timeout=30)
        assert result.outcome == "completed", runtime.inspect(root.id)
        assert tree == {
            s.id: (s.parent_id, s.kernel_id, s.workspace.path)
            for s in runtime.store.sessions(root_id=root.id)
        }
        assert runtime.store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] > count
        assert runtime.store.events(root.id, kind="context_compaction")
        assert runtime.store.events(root.id, kind="verifier_result")[-1]["payload"]["passed"]
        assert search(runtime.store, root.id, "arithmetic")
        assert runtime.index(root.id).symbol_search("add")["matches"]
        assert runtime.store.messages(root.id)[-1]["received_at"]
        assert provider.peak_active >= 2
        assert "return a + b" in (repository / "mathops.py").read_text()
        assert "Add two numbers" in (repository / "mathops.py").read_text()
    finally:
        await runtime.shutdown()


def test_statistics_and_noise_thresholds():
    config = BenchmarkConfig(
        command=["benchmark"], metric_regex=r"time=(\d+)", required_improvement=0.02
    )
    summary = summarize([1, 5, 2, 4, 3])
    assert summary["median"] == 3 and summary["p95"] == 5
    assert compare({"median": 100}, {"median": 97}, config)["passed"]
    assert not compare({"median": 100}, {"median": 102}, config)["passed"]
    with pytest.raises(ValueError):
        summarize([float("nan")])


async def test_real_benchmark_and_durable_experiments(tmp_path, repository, coding_config):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        Editor(context).apply_patch(FIX)
        # The metric is real elapsed computation time; no assumed/fabricated timing.
        metric = BenchmarkConfig(
            command=[
                sys.executable,
                "-c",
                "import time; t=time.perf_counter(); sum(i*i for i in range(10000)); print('seconds='+str(time.perf_counter()-t))",
            ],
            correctness_commands=[
                [sys.executable, "-c", "from mathops import add; assert add(2,3)==5"]
            ],
            metric_regex=r"seconds=([0-9.e-]+)",
            repetitions=3,
            warmups=1,
        )
        result = await run_benchmark(context, metric)
        assert result["passed"] and len(result["runs"]) == 4 and result["median"] > 0
        experiments = Experiments(context)
        experiment = experiments.create(
            "Documented arithmetic remains correct",
            "Add docstring",
            metric.model_dump(),
            metric.correctness_commands,
        )
        measured = await experiments.run(
            experiment["experiment_id"],
            "Correctness passed; timing retained without a speedup claim",
        )
        assert measured["passed"] and measured["metrics"]["count"] == 3
        assert experiments.get(experiment["experiment_id"])["status"] == "concluded"
        assert len(experiments.list()) == 1
        assert (
            runtime.store.db.execute("SELECT COUNT(*) FROM benchmark_measurements").fetchone()[0]
            == 2
        )
        assert search(runtime.store, session.id, "Documented")
    finally:
        await runtime.shutdown()


async def test_executor_artifacts_env_timeout_and_process_group_cancellation(
    tmp_path, repository, coding_config, monkeypatch
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-for-environment-test")
    try:
        result = await LocalExecutor().run(
            context,
            [
                sys.executable,
                "-c",
                "import os; assert 'OPENAI_API_KEY' not in os.environ; print('x'*100000)",
            ],
        )
        assert result["passed"] and len(result["stdout"]) == 4000
        assert len(runtime.artifacts.load(session.id, result["stdout_artifact"])) == 100001
        timeout = await LocalExecutor().run(
            context, [sys.executable, "-c", "import time; time.sleep(60)"], timeout_seconds=0.1
        )
        assert timeout["timed_out"] and not timeout["passed"]
        code = "import os,time; open('child.pid','w').write(str(os.getpid())); time.sleep(60)"
        task = asyncio.create_task(LocalExecutor().run(context, [sys.executable, "-c", code]))
        await eventually(lambda: (repository / "child.pid").exists())
        pid = int((repository / "child.pid").read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert runtime.store.events(session.id, kind="execution_capture")
    finally:
        await runtime.shutdown()


def test_path_confinement_and_container_policy(repository, tmp_path, coding_config):
    with pytest.raises(PermissionError):
        confined(repository, ".git/config")
    with pytest.raises(PermissionError):
        confined(repository, "../outside")
    (repository / "escape").symlink_to(tmp_path)
    with pytest.raises(PermissionError):
        confined(repository, "escape/outside")
    with pytest.raises(PermissionError):
        confined(repository, "mathops.py", allowed=["tests/*"])
    coding_config.execution.command_allowlist = ["pytest"]
    coding_config.execution.environment_allowlist.append("OPENAI_API_KEY")
    assert "OPENAI_API_KEY" not in environment(coding_config.execution)


async def test_fts_scope_refinement_validation_and_loop_guard(tmp_path, repository, coding_config):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        event = runtime.store.event(
            session.id, "observation", {"text": "Parsing negative operands failed"}
        )
        aid = runtime.artifacts.put(session.id, {"metric": "operand analysis"})
        assert search(runtime.store, session.id, "negative operands")[0]["id"] == event
        assert any(
            r["id"] == aid for r in search(runtime.store, session.id, "operand", kind="artifact")
        )
        other = runtime.create("Other", repository, config=coding_config)
        assert not search(runtime.store, other.id, "negative operands")
        with pytest.raises(ValueError):
            validate_skill(
                {"name": "bad", "description": "bad", "code": "if broken"},
                coding_config.permissions,
            )
        with pytest.raises(ValueError):
            validate_skill(
                {
                    "name": "bad",
                    "description": "bad",
                    "code": "import subprocess",
                    "required_permissions": ["python"],
                },
                coding_config.permissions,
            )
        for _ in range(3):
            assert not observe(runtime, session.id, "run_tests", {})
        assert runtime.store.events(session.id, kind="no_progress")
        Editor(context).apply_patch(FIX)
        assert not observe(runtime, session.id, "run_tests", {})
        assert (
            runtime.store.events(session.id, kind="action_fingerprint")[-1]["payload"][
                "repetitions"
            ]
            == 1
        )
    finally:
        await runtime.shutdown()


def test_migration_preserves_v1_history(tmp_path):
    from threadweave.storage import SCHEMA

    directory = tmp_path / "old"
    directory.mkdir()
    connection = sqlite3.connect(directory / "history.sqlite3")
    connection.executescript(SCHEMA + "PRAGMA user_version=1;")
    connection.close()
    store = Store(directory)
    from threadweave.migrations import VERSION

    assert store.db.execute("PRAGMA user_version").fetchone()[0] == VERSION
    assert store.db.execute("SELECT name FROM sqlite_master WHERE name='history_fts'").fetchone()
    store.close()
