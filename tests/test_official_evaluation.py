"""Integration-contract regressions. Test doubles are never capability evidence."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.evals.bridge import OfficialWorker
from threadweave.evals.harness import MatchedProvider, discard, run_buffalo
from threadweave.evals.official_worker import Official, check_docker
from threadweave.evals.runner import contract, resolve
from threadweave.evals.schema import (
    BenchmarkSetup,
    EvaluationConfig,
    NotRun,
    accounting,
    digest,
)
from threadweave.models import (
    Action,
    HarnessError,
    ModelRequest,
    ModelResponse,
    RunConfig,
    TaskConfig,
    Usage,
)
from threadweave.runtime import Runtime

from .fakes import ScriptedProvider


def eval_config():
    return RunConfig(
        provider={"name": "chat", "model": "gpt-test", "api_key_env": "", "max_output_tokens": 128},
        features={"model_compaction": False},
        refinement={"enabled": False},
        limits={"wall_seconds": 20, "max_turns": 5, "token_budget": 100000},
    )


async def test_late_environment_reply_cannot_become_a_scorecard(tmp_path):
    import asyncio

    replies = asyncio.StreamReader()
    replies.feed_data(b'{"request_id": 0, "result": {"old_action": true}}\n')
    replies.feed_data(b'{"request_id": 1, "result": {"score": 25}}\n')

    class Writer:
        def write(self, data):
            assert json.loads(data)["request_id"] == 1

        async def drain(self):
            pass

    worker = OfficialWorker(BenchmarkSetup(), "arc-agi-3", tmp_path)
    worker.process = SimpleNamespace(stdout=replies, stdin=Writer())
    result = await worker.call("finish_profile")
    assert result == {"score": 25}
    assert "old_action" in (tmp_path / "late-replies.jsonl").read_text()


def test_accounting_preserves_unknown_cost_and_descendant_totals():
    result = accounting(
        [
            Usage(input_tokens=10, output_tokens=3, cost=1),
            Usage(input_tokens=20, output_tokens=5, cost=None, reasoning_output_tokens=2),
        ],
        7,
    )
    assert result["input_tokens"] == 30 and result["output_tokens"] == 8
    assert result["total_tokens"] == 38  # Reasoning tokens already belong to output tokens.
    assert result["api_cost"] is None and result["wall_seconds"] == 7
    assert accounting([result], 9)["api_cost"] is None


async def test_verbatim_hierarchy_survives_compaction_and_child_inherits(tmp_path):
    config = eval_config()
    messages = [
        {"role": "system", "content": "Exact hierarchy \u00e9 <privilege>"},
        {"role": "user", "content": "Exact task\nwith conflicting instructions"},
    ]
    config.task = TaskConfig(instruction_messages=messages)
    runtime = Runtime(tmp_path / "state", providers={"chat": ScriptedProvider({})})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        root = runtime.create("Do the official task", workspace, config=config)
        event = runtime.store.event(root.id, "unit-test-context", {})
        runtime.store.add_context(root.id, event, [{"role": "assistant", "content": "old history"}])
        runtime.context.compact(root.id)
        assembled = runtime.context.messages(root.id)
        assert all(m in assembled for m in messages)
        child = runtime.spawn(root.id, "Inspect instructions", name="child")
        assert runtime.store.config(child.id).task.instruction_messages == messages
    finally:
        await runtime.shutdown()


async def test_provider_contract_rejects_changed_child_model(tmp_path):
    config = eval_config().provider
    provider = ScriptedProvider({})
    measured = MatchedProvider(provider, config, tmp_path / "calls.jsonl")
    changed = config.model_copy(update={"model": "different-model"})
    request = ModelRequest(
        session_id="child",
        root_id="root",
        parent_id="root",
        name="child",
        turn=0,
        messages=[],
        tools=[],
        config=changed,
        input_token_bound=1,
    )
    with pytest.raises(HarnessError, match="settings changed"):
        await measured.invoke(request, discard)
    assert not provider.requests


async def test_subscription_resolution_is_forwarded_and_pinned(tmp_path):
    config = eval_config().provider

    class SubscriptionDouble:
        async def resolve(self, candidate):
            return candidate, {"resolved": True}

    measured = MatchedProvider(SubscriptionDouble(), config, tmp_path / "calls.jsonl")
    actual, details = await measured.resolve(config)
    assert actual == config and details["resolved"]
    with pytest.raises(HarnessError, match="changed after comparison"):
        await measured.resolve(config.model_copy(update={"model": "another-model"}))


async def test_runtime_failure_is_not_scored_as_a_model_answer(tmp_path, monkeypatch):
    provider = ScriptedProvider({"root": [RuntimeError("integration broke")]})
    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": provider})
    with pytest.raises(NotRun, match="integration broke"):
        await run_buffalo(
            eval_config(), {"messages": [{"role": "user", "content": "task"}]}, tmp_path
        )
    assert (tmp_path / "usage.json").is_file()


async def test_single_model_configuration_rejects_routing():
    cfg = EvaluationConfig(run=eval_config())
    cfg.run.models["other"] = cfg.run.provider.model_copy()
    with pytest.raises(NotRun, match="routing overrides"):
        await resolve(cfg)


async def test_buffalo_final_answer_is_not_overwritten_by_cleanup(tmp_path, monkeypatch):
    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    text="final exact answer", usage=Usage(input_tokens=13, output_tokens=7)
                )
            ]
        }
    )
    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": provider})
    result = await run_buffalo(
        eval_config(), {"messages": [{"role": "user", "content": "task"}]}, tmp_path
    )
    assert result["response"] == "final exact answer"
    assert result["stop_reason"] == "completed"
    assert result["usage"]["total_tokens"] == 20


async def test_buffalo_recursive_usage_is_settled_and_counted(tmp_path, monkeypatch):
    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "child = await rlm('child work', name='child', purpose='shared')\nprint(child)"
                            },
                        )
                    ],
                    usage=Usage(input_tokens=11, output_tokens=4),
                ),
                ModelResponse(text="root result", usage=Usage(input_tokens=17, output_tokens=6)),
            ],
            "child": [
                ModelResponse(text="child result", usage=Usage(input_tokens=23, output_tokens=8))
            ],
        }
    )
    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": provider})
    result = await run_buffalo(
        eval_config(), {"messages": [{"role": "user", "content": "task"}]}, tmp_path
    )
    assert any(request.parent_id for request in provider.requests)
    # The root may take another turn to consume child completion; use the full recorded provider journal.
    rows = [
        json.loads(line) for line in (tmp_path / "provider-calls.jsonl").read_text().splitlines()
    ]
    assert result["usage"]["input_tokens"] == sum(
        r["response"]["usage"]["input_tokens"] for r in rows
    )
    assert result["usage"]["output_tokens"] == sum(
        r["response"]["usage"]["output_tokens"] for r in rows
    )
    sessions = json.loads((tmp_path / "sessions.json").read_text())
    assert len(sessions) == 2 and all(s["outcome"] != "active" for s in sessions)


async def test_cancelled_buffalo_run_counts_inflight_child_usage(tmp_path, monkeypatch):
    child_started = asyncio.Event()

    class Provider:
        async def invoke(self, request, emit):
            if request.parent_id:
                child_started.set()
                await asyncio.Future()
            return ModelResponse(
                actions=[
                    Action(
                        name="ipython",
                        arguments={
                            "code": "await rlm('child work', name='child', purpose='shared')"
                        },
                    )
                ],
                usage=Usage(input_tokens=5, output_tokens=2),
            )

    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": Provider()})
    task = asyncio.create_task(
        run_buffalo(eval_config(), {"messages": [{"role": "user", "content": "task"}]}, tmp_path)
    )
    await asyncio.wait_for(child_started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["model_calls"] == 2
    assert usage["estimated_calls"] == 1
    assert usage["subagent_count"] == 1
    sessions = json.loads((tmp_path / "sessions.json").read_text())
    assert len(sessions) == 2 and all(s["outcome"] != "active" for s in sessions)


async def test_long_context_is_exactly_preserved_in_repl_task(tmp_path, monkeypatch):
    content = "prefix\n" + "context material " * 2000 + "\nunique suffix"
    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "assert context['task'] == Path('task.txt').read_text()\nprint(context['task'][-13:])"
                            },
                        )
                    ]
                ),
                ModelResponse(text="answer"),
            ]
        }
    )
    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": provider})
    result = await run_buffalo(
        eval_config(),
        {"messages": [{"role": "user", "content": content}]},
        tmp_path,
        long_context=True,
    )
    assert (tmp_path / "workspace/task.txt").read_text() == content
    assert result["response"] == "answer"
    assert "unique suffix" in json.dumps(provider.requests[1].messages)


@pytest.mark.parametrize("error", ["Empty response", "Missing checker dependency"])
def test_if_preserves_official_empty_response_nulls_but_rejects_setup_errors(tmp_path, error):
    worker = Official()
    worker.benchmark = "manyih-if"
    worker.output = tmp_path
    worker.rows = {"0": {"constraints": [{"score": None, "eval_details": [{"error": error}]}]}}
    worker.if_evaluator = SimpleNamespace(_evaluate_single=lambda args: args[0])
    if error == "Empty response":
        result = worker.grade("0", "")
        assert result["constraints"][0]["score"] is None
        assert result["constraints"][0]["eval_details"][0]["error"] == error
    else:
        with pytest.raises(RuntimeError, match="Missing checker dependency"):
            worker.grade("0", "")


def test_if_judge_recovery_keeps_scored_checks_and_the_original_answer(tmp_path):
    worker = Official()
    worker.benchmark = "manyih-if"
    worker.output = tmp_path
    worker.rows = {"0": {"constraints": [{"name": "first"}, {"name": "second"}]}}
    prior = {
        "output": {"content": "original answer"},
        "constraints": [
            {"name": "first", "score": True, "eval_details": [{"error": None}]},
            {"name": "second", "score": None, "eval_details": [{"error": "transport_failure"}]},
        ],
    }

    def official_evaluator(args):
        entry = args[0]
        assert entry["constraints"][0]["score"] is True
        assert "score" not in entry["constraints"][1]
        entry["constraints"][1]["score"] = False
        return entry

    worker.if_evaluator = SimpleNamespace(_evaluate_single=official_evaluator)
    result = worker.grade("0", "original answer", previous_grade=prior)
    assert [c["score"] for c in result["constraints"]] == [True, False]
    with pytest.raises(ValueError, match="Cannot change an agent answer"):
        worker.grade("0", "a different answer", previous_grade=prior)


def test_arc_frame_pixels_are_included_even_when_sdk_uses_private_arrays():
    class Frame:
        def tolist(self):
            return [[1, 2], [3, 4]]

    obs = SimpleNamespace(
        frame=[Frame()], model_dump=lambda **_: {"guid": "random", "state": "PLAYING"}
    )
    worker = Official()
    worker.env = SimpleNamespace(
        observation_space=obs, action_space=[SimpleNamespace(name="ACTION1")]
    )
    result = worker.arc_observation()
    frame = result["observation"]["frame_rle"][0]
    assert frame["height"] == frame["width"] == 2
    assert [pixel for pixel, count in frame["runs"] for _ in range(count)] == [1, 2, 3, 4]
    assert "guid" not in result["observation"]


def test_arc_uses_official_scorecard_percent_without_recomputing():
    worker = Official()
    worker.benchmark = "arc-agi-3"
    worker.card_id = "scorecard"
    card = SimpleNamespace(score=31.25, model_dump=lambda **_: {"score": 31.25, "games": []})
    worker.arc = SimpleNamespace(
        close_scorecard=lambda _: card,
        arc_api_key=None,
        scorecard_manager=SimpleNamespace(get_scorecard=lambda *_: None),
    )
    assert worker.finish_profile()["primary_score"] == 31.25


def test_factorio_research_uses_authoritative_environment_state():
    state = SimpleNamespace(
        technologies={
            "a": SimpleNamespace(researched=True),
            "b": SimpleNamespace(researched=False),
        },
        current_research="b",
        research_progress=0.4,
    )
    # Real FLE types are dataclasses. A local dataclass verifies our boundary without a game substitute.
    from dataclasses import make_dataclass

    Research = make_dataclass("Research", ["technologies", "current_research", "research_progress"])
    state = Research(state.technologies, state.current_research, state.research_progress)
    worker = Official()
    worker.instance = SimpleNamespace(
        first_namespace=SimpleNamespace(_save_research_state=lambda: state)
    )
    assert worker.research()["technologies_completed"] == 1
    assert worker.research()["current_research_progress_pct"] == 40


def test_factorio_disconnected_docker_with_zero_exit_is_not_available(monkeypatch):
    monkeypatch.setattr("threadweave.evals.official_worker.shutil.which", lambda _: "/docker")
    monkeypatch.setattr(
        "threadweave.evals.official_worker.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="", stderr="Cannot connect to Docker daemon"
        ),
    )
    with pytest.raises(ValueError, match="Docker daemon unavailable: Cannot connect"):
        check_docker()


def test_factorio_changed_starting_save_is_rejected_before_server_mutation(tmp_path):
    worker = Official()
    worker.benchmark = "factorio"
    worker.world = tmp_path / "save.zip"
    worker.world.write_bytes(b"changed bytes")
    worker.provenance = {"dataset_environment_version": {"world_sha256": "original hash"}}
    with pytest.raises(ValueError, match="starting world changed"):
        worker.start_task("world")


def test_comparison_contract_changes_when_any_required_setting_changes():
    config = EvaluationConfig(run=eval_config())
    provenance = {
        "benchmark_version": "commit",
        "dataset_environment_version": "data",
        "starting_state": "world",
    }
    first = contract(config, provenance, ["id"], 0)
    config.run.provider.parameters["reasoning_effort"] = "high"
    assert digest(first) != digest(contract(config, provenance, ["id"], 0))
    assert first["task_ids"] == ["id"]


@pytest.mark.skipif(
    not os.environ.get("BUFFALO_TEST_MANYIH_SOURCE"),
    reason="Official ManyIH checkout not configured",
)
async def test_real_official_manyih_formatter_and_coding_analyzer(tmp_path):
    setup = BenchmarkSetup(
        source=Path(os.environ["BUFFALO_TEST_MANYIH_SOURCE"]),
        python=os.environ.get("BUFFALO_TEST_OFFICIAL_PYTHON", sys.executable),
    )
    worker = OfficialWorker(setup, "manyih-coding", tmp_path)
    try:
        provenance = await worker.start()
        tid = provenance["task_ids"][0]
        task = await worker.call("task", task_id=tid)
        assert [m["role"] for m in task["messages"]] == ["system", "user"]
        assert "test_code" not in task and "expected_styles" not in task
        # Empty model output tests scorer plumbing; no agent capability run occurs.
        grade = await worker.call("grade", task_id=tid, response="")
        assert grade["evaluation"]["overall_passed"] is False
        result = await worker.call("summarize", grades=[grade], profile="unit-test")
        assert result["primary_score"] == 0
        assert result["raw"]["failure_stats"]["test_failures"] == 1
    finally:
        await worker.close()


@pytest.mark.skipif(
    not os.environ.get("BUFFALO_TEST_MANYIH_SOURCE"),
    reason="Official ManyIH checkout not configured",
)
async def test_real_official_manyih_if_inputs_are_unchanged(tmp_path):
    setup = BenchmarkSetup(
        source=Path(os.environ["BUFFALO_TEST_MANYIH_SOURCE"]),
        python=os.environ.get("BUFFALO_TEST_OFFICIAL_PYTHON", sys.executable),
    )
    worker = OfficialWorker(setup, "manyih-if", tmp_path)
    try:
        provenance = await worker.start()
        tid = provenance["task_ids"][0]
        raw = json.loads((setup.source / "manyih/data/instruction_following.json").read_text())
        assert len(provenance["task_ids"]) == len(raw) == 426
        row = raw[provenance["task_identity_map"][tid]["row_index"]]
        task = await worker.call("task", task_id=tid)
        assert task["messages"] == row["input"]
        assert "constraints" not in task
    finally:
        await worker.close()


async def test_evaluation_can_use_production_coding_adapter_without_rewriting_task(
    tmp_path, repository, monkeypatch
):
    from threadweave.coding_config import update_coding_options

    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(
                    text="exact generated answer", usage=Usage(input_tokens=13, output_tokens=7)
                )
            ]
        }
    )
    monkeypatch.setattr("threadweave.evals.harness.default_providers", lambda: {"chat": provider})
    config = eval_config()
    config.permissions.append("process")
    config.task.adapter = "coding"
    update_coding_options(
        config.task, capture_baseline=False, require_change=False, require_tests=False
    )
    result = await run_buffalo(
        config,
        {
            "messages": [
                {"role": "system", "content": "Official system instruction"},
                {"role": "user", "content": "Official task verbatim"},
            ]
        },
        tmp_path / "run",
        workspace=repository,
        task_config=config.task,
    )
    assert result["response"] == "exact generated answer"
    request = provider.requests[0]
    assert any(
        m["role"] == "system" and m["content"] == "Official system instruction"
        for m in request.messages
    )
    assert "Official task verbatim" in json.dumps(request.messages)
    assert "Coding APIs" in json.dumps(request.messages)
    sessions = json.loads((tmp_path / "run/sessions.json").read_text())
    assert sessions[0]["instruction"] == "Official task verbatim"
    saved = RunConfig.model_validate_json((tmp_path / "run/buffalo-config.json").read_text())
    assert saved.task.adapter == "coding"
