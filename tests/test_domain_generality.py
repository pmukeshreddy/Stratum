"""Production admission, provider requests, worker restore and recursive specialization."""

import json
from pathlib import Path

import pytest

from threadweave.coding_config import coding_options, update_coding_options
from threadweave.models import ModelResponse, TaskConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext, builtins

from .conftest import response
from .fakes import ScriptedProvider
from .test_python_control import cell

CODING_TOOLS = {
    "repo_map",
    "repo_search",
    "symbol_search",
    "references_search",
    "git_status",
    "git_diff",
    "apply_patch",
    "run_tests",
    "run_build",
    "run_lint",
    "run_typecheck",
    "run_benchmark",
    "related_tests",
    "failure_localize",
    "candidate_apply",
}
CODING_NAMES = {"repo", "git", "edit", "tests", "build", "lint", "typecheck", "bench", "experiment"}


def admitted(runtime, sid):
    config = runtime.store.config(sid)
    return {name for name in runtime.tools.entries if runtime.tools.allowed(name, config)}


def inspect_code(coding):
    return f"""names = {sorted(CODING_NAMES)!r}
assert all((name in globals()) == {coding!r} for name in names)
assert all(name in globals() for name in ['workspace','bash','rlm','agents','history','artifacts','harness','skills','mcp'])
catalog = {{entry['function']['name'] for entry in tools.catalog()}}
assert ({sorted(CODING_TOOLS)!r} and set({sorted(CODING_TOOLS)!r}).issubset(catalog)) == {coding!r}
print('specialization verified')"""


@pytest.mark.parametrize("adapter", ["workspace", "coding"])
async def test_public_session_provider_and_kernel_are_adapter_owned(
    tmp_path, repository, python_config, adapter
):
    python_config.task.adapter = adapter
    python_config.context.max_tokens = 96000
    update_coding_options(
        python_config.task,
        capture_baseline=False,
        require_clean_baseline=False,
        require_tests=False,
        require_change=False,
    )
    provider = ScriptedProvider(
        {
            "root": [
                response("ipython", code=inspect_code(adapter == "coding")),
                ModelResponse(text="Inspected capabilities"),
            ]
        }
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("Inspect available capabilities", repository, config=python_config)
        await runtime.start()
        assert (await runtime.wait(root.id)).outcome == "completed"
        assert not runtime.store.events(root.id, kind="python_error")
        assert "specialization verified" in json.dumps(
            runtime.store.events(root.id, kind="python_result")
        )
        text = json.dumps(provider.requests[0].messages)
        if adapter == "workspace":
            assert not CODING_TOOLS & admitted(runtime, root.id)
            assert all(
                term not in text
                for term in [
                    "Coding APIs",
                    "repository intelligence",
                    "tests.related_to",
                    "worktrees",
                    "Coding decision support",
                ]
            )
            assert runtime.store.config(root.id).task.verifier == "none"
            assert not runtime.store.events(root.id, kind="coding_baseline")
        else:
            assert CODING_TOOLS <= admitted(runtime, root.id)
            assert "Coding APIs" in text and "Coding decision support" in text
            assert runtime.store.config(root.id).task.verifier == "coding"
            assert runtime.store.events(root.id, kind="coding_baseline")
    finally:
        await runtime.shutdown()


async def test_coding_registry_never_leaks_to_workspace_and_host_bridge(
    tmp_path, repository, config
):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        assert not CODING_TOOLS & builtins().entries.keys()
        coding = config.model_copy(deep=True)
        coding.task.adapter = "coding"
        coding.permissions.append("process")
        code = runtime.create("coding", repository, config=coding, mode="interactive")
        general = runtime.create("general", repository, config=config, mode="interactive")
        assert CODING_TOOLS <= {
            t["function"]["name"] for t in runtime.tools.schemas(runtime.store.config(code.id))
        }
        assert not CODING_TOOLS & {
            t["function"]["name"] for t in runtime.tools.schemas(runtime.store.config(general.id))
        }
        event = runtime.store.event(general.id, "test", {})
        context = ToolContext(runtime, general.id, new_id(), event)
        for name, arguments in [
            ("repo_map", {}),
            ("run_tests", {}),
            (
                "host_request",
                {
                    "operation": "edit",
                    "payload": {"path": "mathops.py", "old_str": "a - b", "new_str": "a + b"},
                },
            ),
            ("host_request", {"operation": "context.focus", "payload": {}}),
        ]:
            with pytest.raises(Exception, match="not permitted"):
                runtime.tools.resolve(context, name, arguments)
        assert "a - b" in (repository / "mathops.py").read_text()
    finally:
        await runtime.shutdown()


async def test_explicit_grant_and_disabling_remove_executable_capabilities(tmp_path, python_config):
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        python_config.capabilities = ["coding"]
        granted = runtime.create(
            "explicit capability", tmp_path, config=python_config, mode="interactive"
        )
        await cell(runtime, granted.id, inspect_code(True))
        python_config.disabled_capabilities = ["coding"]
        disabled = runtime.create(
            "disabled capability", tmp_path, config=python_config, mode="interactive"
        )
        await cell(runtime, disabled.id, inspect_code(False))
        assert not CODING_TOOLS & admitted(runtime, disabled.id)
        python_config.disabled_capabilities = []
        python_config.tool_allowlist = [
            n for n in admitted(runtime, granted.id) if n != "repo_search"
        ]
        limited = runtime.create(
            "one disabled tool", tmp_path, config=python_config, mode="interactive"
        )
        await cell(
            runtime, limited.id, "assert 'search' not in dir(repo)\nassert 'map' in dir(repo)"
        )
        assert "repo_search" not in admitted(runtime, limited.id)
    finally:
        await runtime.shutdown()


@pytest.mark.parametrize(
    "adapter,isolate",
    [("workspace", False), ("workspace", True), ("coding", False), ("coding", True)],
)
async def test_recursive_admission_and_restored_namespaces(
    tmp_path, repository, python_config, adapter, isolate
):
    python_config.task.adapter = adapter
    update_coding_options(
        python_config.task,
        capture_baseline=False,
        require_clean_baseline=False,
        require_tests=False,
        require_change=False,
    )
    directory = tmp_path / "state"
    runtime = Runtime(directory, providers={"mock": ScriptedProvider({})})
    try:
        parent = runtime.create("parent", repository, config=python_config, mode="interactive")
        child = await runtime.spawn_async(
            parent.id, "child", isolate=isolate, model="child-model", thinking="low"
        )
        runtime.store.update(child.id, runnable=False)
        cfg = runtime.store.config(child.id)
        assert cfg.task.adapter == adapter
        assert (
            cfg.provider.model == "child-model"
            and cfg.provider.parameters["reasoning_effort"] == "low"
        )
        assert (child.workspace.path != parent.workspace.path) == isolate
        assert admitted(runtime, child.id) == admitted(runtime, parent.id)
        await cell(runtime, child.id, inspect_code(adapter == "coding") + "\nretained_value = 42")
        await runtime._close_kernel(child.id)
        await cell(
            runtime, child.id, inspect_code(adapter == "coding") + "\nassert retained_value == 42"
        )
        await runtime.shutdown()
        runtime = Runtime(directory, providers={"mock": ScriptedProvider({})})
        await runtime.recover()
        await cell(
            runtime, child.id, inspect_code(adapter == "coding") + "\nassert retained_value == 42"
        )
        assert runtime.store.config(child.id).task.adapter == adapter
        if adapter == "workspace":
            assert not runtime.store.db.execute(
                "SELECT 1 FROM candidates WHERE child_id=?", (child.id,)
            ).fetchone()
    finally:
        await runtime.shutdown()


async def test_custom_adapter_extends_tools_namespaces_and_child_profiles(tmp_path, python_config):
    python_config.execution.environment_allowlist.append("PYTHONPATH")
    python_config.execution.environment["PYTHONPATH"] = str(Path(__file__).parents[1])
    python_config.extensions = ["tests.generality_extension:install"]
    python_config.task.adapter = "observatory"
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        parent = runtime.create("Observe", tmp_path, config=python_config, mode="interactive")
        child = runtime.spawn(parent.id, "Survey", purpose="survey")
        assert "Return measured observations." in child.instruction
        await cell(
            runtime,
            child.id,
            "sample = observatory.sample()\nassert sample['observation'] == 'measured'\nassert 'repo' not in globals()",
        )
        assert "observatory.sample()" in json.dumps(runtime.context.messages(child.id))
        with pytest.raises(ValueError, match="Unknown child profile"):
            runtime.spawn(parent.id, "Not coding", purpose="candidate")
    finally:
        await runtime.shutdown()


def test_legacy_task_configuration_migrates_to_adapter_options():
    legacy = {
        "adapter": "coding",
        "repository": "/example",
        "test_commands": [["pytest"]],
        "protect_tests": True,
    }
    task = TaskConfig.model_validate(legacy)
    assert coding_options(task).protect_tests
    assert coding_options(task).test_commands == [["pytest"]]
    assert "test_commands" not in TaskConfig.model_fields
    assert "repository" not in task.model_dump()
    assert TaskConfig.model_validate(task.model_dump()) == task
    with pytest.raises(ValueError, match="Conflicting legacy"):
        TaskConfig.model_validate({**legacy, "options": {"coding": {"repository": "/other"}}})


async def test_legacy_persisted_config_recovers_admission_and_options(
    tmp_path, repository, python_config
):
    python_config.task.adapter = "coding"
    update_coding_options(
        python_config.task,
        capture_baseline=False,
        require_clean_baseline=False,
        require_tests=False,
        require_change=False,
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("legacy coding", repository, config=python_config, mode="interactive")
        old = runtime.store.config(root.id).model_dump()
        old["task"].update(old["task"].pop("options")["coding"])
        for key in ["capabilities", "effective_capabilities", "disabled_capabilities"]:
            old.pop(key)
        runtime.store.db.execute(
            "UPDATE configs SET body=? WHERE id=?", (json.dumps(old), root.config_id)
        )
        await runtime.shutdown()
        runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
        await runtime.recover()
        assert runtime.store.config(root.id).effective_capabilities == ["coding"]
        assert not coding_options(runtime.store.config(root.id).task).require_change
        await cell(runtime, root.id, inspect_code(True))
    finally:
        await runtime.shutdown()


async def test_coding_adapter_capability_disabled_without_weakening_verifier(
    tmp_path, repository, python_config
):
    python_config.task.adapter = "coding"
    python_config.disabled_capabilities = ["coding"]
    update_coding_options(
        python_config.task,
        capture_baseline=False,
        require_clean_baseline=False,
        require_tests=False,
        require_change=True,
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create(
            "disabled tools", repository, config=python_config, mode="interactive"
        )
        assert not CODING_TOOLS & admitted(runtime, root.id)
        await cell(runtime, root.id, inspect_code(False))
        result, error = await runtime._verify(root.id, runtime.store.event(root.id, "test", {}))
        assert result and not result.passed
        assert "A nonempty change is required" in result.details["violations"]
    finally:
        await runtime.shutdown()
