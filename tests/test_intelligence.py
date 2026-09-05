import json
import sys

import pytest

from threadweave.configuration import doctor, load_config
from threadweave.evaluation import Instance, analyze, evaluate, prepare_instance
from threadweave.models import ModelResponse, RunConfig, StateEdit, Usage, new_id
from threadweave.refinement import run_skill
from threadweave.runtime import Runtime

from .conftest import response
from .fakes import ScriptedProvider
from .test_coding import FIX, setup_runtime


async def test_automatic_refinement_uses_evidence_routing_and_versioned_validation(
    tmp_path, repository, coding_config
):
    coding_config.refinement.automatic = True
    coding_config.models = {
        "fast": coding_config.provider.model_copy(update={"model": "fast-test"})
    }
    coding_config.routing.policy = "role_based"
    coding_config.routing.roles = {"refinement": "fast"}
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)

    class Refiner:
        async def invoke(self, request, emit):
            assert request.metadata["purpose"] == "refinement"
            assert request.config.model == "fast-test"
            evidence = json.loads(request.messages[-1]["content"])
            proposal = {
                "kind": "memory",
                "title": "Arithmetic failure",
                "content": {"text": "Subtraction was used in addition."},
                "source_events": [evidence[-1]["id"]],
                "intended_effect": "Check operator semantics",
            }
            invalid = {**proposal, "source_events": ["fabricated-event"]}
            return ModelResponse(
                text=json.dumps({"proposals": [proposal, invalid]}),
                usage=Usage(input_tokens=50, output_tokens=40),
            )

    runtime.providers["test"] = Refiner()
    try:
        await runtime.auto_refine(session.id, trigger="completion")
        states = runtime.store.states(session.id)
        assert len(states) == 1 and states[0]["version"] == 1
        assert states[0]["provenance"]["source_events"]
        assert runtime.store.events(session.id, kind="refinement_rejected")
        assert runtime.store.usage(session.id).model_calls == 1
        assert runtime.store.usage(session.id).input_tokens == 50
        assert not runtime.store.session(session.id).pending_turn
        decision = runtime.store.events(session.id, kind="model_routing")[-1]["payload"]
        assert decision["role"] == "refinement" and decision["alias"] == "fast"
        await runtime.auto_refine(session.id, trigger="completion")
        assert runtime.store.usage(session.id).model_calls == 1
    finally:
        await runtime.shutdown()


async def test_model_compaction_records_provenance_and_preserves_full_events(
    tmp_path, repository, coding_config
):
    coding_config.features.model_compaction = True
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)

    class Compactor:
        async def invoke(self, request, emit):
            assert request.metadata["purpose"] == "compaction"
            return ModelResponse(
                text=json.dumps(
                    {
                        "attempted_approaches": ["operator correction"],
                        "unresolved_work": ["run tests"],
                        "evidence_ids": [],
                    }
                ),
                usage=Usage(input_tokens=20, output_tokens=20),
            )

    runtime.providers["test"] = Compactor()
    try:
        ids = []
        for i in range(12):
            eid = runtime.store.event(session.id, "observation", {"text": str(i) + "x" * 10000})
            runtime.store.add_context(
                session.id, eid, [{"role": "user", "content": str(i) + "x" * 10000}]
            )
            ids.append(eid)
        await runtime.semantic_compact(session.id)
        event = runtime.store.events(session.id, kind="context_compaction")[-1]
        assert (
            event["payload"]["method"] == "model_structured"
            and event["payload"]["model_response_event"]
        )
        assert "run tests" in runtime.store.session(session.id).summary
        assert all(runtime.store.event_by_id(i) for i in ids)
        assert runtime.store.usage(session.id).model_calls == 1
        assert runtime.store.session(session.id).pending_turn is None
    finally:
        await runtime.shutdown()


async def test_skills_inputs_outcomes_quarantine_and_rollback(tmp_path, repository, coding_config):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        edit = StateEdit(
            kind="skill",
            title="A real executable procedure",
            content={
                "name": "double",
                "description": "Double the input",
                "inputs": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                "required_permissions": ["python"],
                "code": "skill_inputs['value'] * 2",
            },
            source_events=[context.source_event],
            intended_effect="Reuse validated computation",
        )
        runtime.store.queue_refinement(session.id, edit)
        entry = runtime.store.apply_refinements(session.id)[0]
        result = await run_skill(context, entry, {"value": 4})
        assert result["value"] == "8"
        with pytest.raises(ValueError):
            await run_skill(context, entry, {"value": "not a number"})
        broken = edit.model_copy(deep=True)
        broken.entry_id = entry
        broken.content["code"] = "raise ValueError('unsuitable procedure')"
        runtime.store.queue_refinement(session.id, broken)
        runtime.store.apply_refinements(session.id)
        for _ in range(3):
            context.action_id = new_id()
            assert (await run_skill(context, entry, {"value": 4}))["error"]
        with pytest.raises(ValueError, match="quarantined"):
            await run_skill(context, entry, {"value": 4})
        runtime.store.queue_refinement(
            session.id,
            edit.model_copy(
                update={"entry_id": entry, "operation": "rollback", "rollback_version": 1}
            ),
        )
        runtime.store.apply_refinements(session.id)
        context.action_id = new_id()
        assert (await run_skill(context, entry, {"value": 5}))["value"] == "10"
        assert runtime.store.state(session.id, entry)["version"] == 3
    finally:
        await runtime.shutdown()


async def test_ablation_flags_disable_capabilities_and_persistent_working_values(
    tmp_path, repository, coding_config
):
    coding_config.features.persistent_repl = False
    coding_config.features.subagents = False
    coding_config.features.history_retrieval = False
    coding_config.features.experiments = False
    coding_config.features.enhanced_code_index = False
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    try:
        names = {s["function"]["name"] for s in runtime.tools.schemas(coding_config)}
        assert (
            "agent_spawn" not in names
            and "history_search" not in names
            and "experiment_create" not in names
        )
        assert runtime.index(session.id).outline("mathops.py")["parser"] == "disabled"
        assert not (await runtime.execute_python(context, "value = 42"))["error"]
        context.action_id = new_id()
        assert (await runtime.execute_python(context, "value"))["error"]["code"] == "NameError"
        with pytest.raises(PermissionError):
            runtime.spawn(session.id, "Disabled")
    finally:
        await runtime.shutdown()


async def test_real_external_issue_evaluation_and_machine_readable_analysis(
    tmp_path, repository, coding_config
):
    tasks, output = tmp_path / "tasks.jsonl", tmp_path / "results.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "arithmetic-issue",
                "adapter": "repository_issue",
                "repository": str(repository),
                "objective": "Correct addition",
                "test_commands": [[sys.executable, "-m", "pytest", "-q"]],
            }
        )
        + "\n"
    )
    providers = {
        "test": ScriptedProvider(
            {"root": [response("apply_patch", patch=FIX), response("finish", result="Fixed")]}
        )
    }
    result = await evaluate(
        tasks, coding_config, tmp_path / "eval", output=output, providers=providers
    )
    assert result["solved"] == 1
    row = json.loads(output.read_text())
    assert row["metrics"]["tests_runs"] >= 2 and row["verifier_score"] == 1
    assert row["config_id"] and row["resolved_config"]["features"]["experiments"]
    assert row["metrics"]["final_diff_size"] > 0
    assert "a - b" in (repository / "mathops.py").read_text()  # Source instance was not modified.
    summary = analyze(output)
    assert summary["success_rate"] == 1 and summary["cost_per_solved"] == 0
    assert summary["totals"]["tool_calls"] > 0


def test_external_long_context_and_kernel_configuration_errors(tmp_path, repository, coding_config):
    bundle = tmp_path / "task.txt"
    bundle.write_text("Behavioral constraint: preserve signed arithmetic")
    instance = Instance(
        id="context",
        adapter="long_context",
        repository=str(repository),
        objective="Read task context",
        context_bundle=[str(bundle)],
        test_commands=[[sys.executable, "-m", "pytest"]],
    )
    workspace, instruction, config = prepare_instance(
        instance, tmp_path / "external", coding_config, base_directory=tmp_path
    )
    assert "Additional task context" in instruction
    assert (workspace / ".task_context/0-task.txt").exists()
    assert not config.task.require_clean_baseline
    missing = Instance(id="gpu", adapter="kernel", repository=str(repository), objective="Optimize")
    with pytest.raises(ValueError, match="Kernel instances require"):
        prepare_instance(missing, tmp_path / "gpu", coding_config, base_directory=tmp_path)


async def test_production_rejects_missing_model_and_demo_provider(tmp_path, repository):
    runtime = Runtime(tmp_path / "state")
    try:
        assert set(runtime.providers) == {"chat"}
        with pytest.raises(ValueError, match="Explicit provider.model"):
            runtime.create("Fix", repository)
        with pytest.raises(ValueError, match="Unknown provider"):
            runtime.create(
                "Fix", repository, config=RunConfig(provider={"name": "demo", "model": "not-real"})
            )
        assert not runtime.store.sessions()
    finally:
        await runtime.shutdown()


def test_config_environment_resolution_doctor_and_no_secret_printing(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"provider": {"model": "${TEST_MODEL}", "api_key_env": "TEST_KEY"}})
    )
    with pytest.raises(ValueError, match="Set TEST_MODEL"):
        load_config(config)
    monkeypatch.setenv("TEST_MODEL", "explicit-model")
    monkeypatch.setenv("TEST_KEY", "never-log-this-secret")
    loaded = load_config(config)
    report = doctor(tmp_path / "data", loaded)
    assert report["providers"][0]["credential_present"]
    assert "never-log-this-secret" not in json.dumps(report)
    assert report["capabilities"]["git"]


async def test_explicit_refinement_request_runs_at_boundary_without_periodic_policy(
    tmp_path, repository, coding_config
):
    runtime, session, context = await setup_runtime(tmp_path, repository, coding_config)
    runtime.providers["test"] = ScriptedProvider(
        {"root": [ModelResponse(text='{"proposals": []}')]}
    )
    try:
        runtime.message(None, session.id, "/refine")
        assert runtime.store.usage(session.id).model_calls == 0
        await runtime.auto_refine(session.id)
        assert runtime.store.usage(session.id).model_calls == 1
        assert (
            runtime.store.events(session.id, kind="automatic_refinement")[-1]["payload"]["trigger"]
            == "manual"
        )
    finally:
        await runtime.shutdown()
