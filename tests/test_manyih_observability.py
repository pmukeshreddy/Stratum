import json

from threadweave.evals.observability import operations, summarize, task_observability
from threadweave.models import Action, ModelResponse, RunConfig
from threadweave.runtime import Runtime


class TraceProvider:
    async def invoke(self, request, emit):
        if request.parent_id:
            return ModelResponse(text="Independent evidence: 6 * 7 = 42")
        if request.turn == 0:
            return ModelResponse(
                actions=[
                    Action(
                        name="ipython",
                        arguments={
                            "code": "child = await rlm('Compute 6 * 7 independently', name='check')\n"
                            "assert 6 * 7 == 42\nprint('checked 42')\nawait agents.wait(seconds=1)"
                        },
                    )
                ]
            )
        return ModelResponse(text="42")


async def test_projection_links_actual_requests_and_does_not_change_traces(tmp_path):
    config = RunConfig(
        provider={"name": "mock", "model": "test"},
        refinement={"enabled": False},
        task={"wait_for_children": True},
        limits={"wall_seconds": 30},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": TraceProvider()})
    try:
        root = runtime.create("Compute 6 * 7", tmp_path, config=config)
        await runtime.start()
        await runtime.wait(root.id, timeout=25)
        before = list(runtime.store.iter_events(root.id, tree=True))
        record = {
            "task_id": "0",
            "functional_pass": True,
            "style_pass": True,
            "overall_pass": True,
            "stop_reason": "completed",
            "wall_time_seconds": 2.0,
        }
        row, trace, calls, refinement = task_observability(tmp_path, record)
        assert row["REPL"]["entered_repl"]
        assert row["REPL"]["python_executions"] == 1
        assert row["REPL"]["first_root_turn"] == 1
        assert row["REPL"]["executions_with_output_seen_by_root"] == 1
        assert row["RLM"]["rlm_calls"] == row["RLM"]["children_spawned"] == 1
        assert row["RLM"]["child_completed"] == 1
        assert row["RLM"]["child_failed"] == row["RLM"]["child_cancelled"] == 0
        assert calls[0]["prompt_excerpt"] == "Compute 6 * 7 independently"
        assert calls[0]["root_turn"] == 1
        assert calls[0]["model"] == "test"
        assert calls[0]["completion_timestamp"] >= calls[0]["spawn_timestamp"]
        assert calls[0]["evidence_reached_root"]
        assert calls[0]["later_root_invocations"]
        assert row["child_evidence"]["explicit_evidence_consumption_events"] == 0
        assert trace["repl_trace"][0]["later_root_invocations_receiving_output"]
        assert refinement == []
        assert before == list(runtime.store.iter_events(root.id, tree=True))
        summary = summarize([row], [record], 3)
        assert summary["totals"]["REPL"]["entered_repl"] == 1
        assert summary["runtime"]["p95_task_seconds"] == 2
        assert summary["observational_cohorts"]["RLM"]["used"]["new_overall_pass"] == 1
    finally:
        await runtime.shutdown()


async def test_restore_sidecar_does_not_enter_tool_result(tmp_path):
    from threadweave.kernel import Kernel
    from threadweave.models import new_id

    async def bridge(*args, **kwargs):
        raise AssertionError("No host calls")

    worker = Kernel(
        tmp_path / "kernel",
        tmp_path,
        bridge,
        bootstrap={"session_id": "root", "kernel_state": {"variable_bytes": 4096}},
    )
    try:
        await worker.execute(new_id(), "large = 'x' * 9000", 10)
        execution = new_id()
        result = await worker.execute(
            execution, "assert len(repl_state.rehydrate('large')) == 9000", 10
        )
        assert result["error"] is None
        rows = [
            json.loads(line)
            for line in (tmp_path / "kernel/restore-events.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["execution_id"] == execution
        assert rows[0]["name"] == "large"
        assert "restore-events" not in json.dumps(result)
        assert "timestamp" not in result
    finally:
        await worker.close()


def test_operations_are_syntax_only():
    assert operations("assert x == 1; print(x)") == ["assertions", "call:print"]
    assert operations("%time x") == ["unparsed Python/IPython; see code excerpt"]


async def test_refinement_counts_and_notice_visibility(tmp_path):
    from .test_continual_harness import edit, proposal

    class RefineProvider:
        async def invoke(self, request, emit):
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement_review":
                return ModelResponse(
                    text=json.dumps({"shouldRefine": True, "rationale": "observed"})
                )
            if purpose == "refinement":
                return ModelResponse(text=json.dumps(proposal(edit())))
            if request.turn == 0:
                return ModelResponse(
                    actions=[
                        Action(name="ipython", arguments={"code": "await refine.run()\nprint(42)"})
                    ]
                )
            return ModelResponse(text="42")

    config = RunConfig(
        provider={"name": "mock", "model": "test"},
        refinement={"enabled": True},
        limits={"wall_seconds": 30},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": RefineProvider()})
    try:
        root = runtime.create("Compute 6 * 7", tmp_path, config=config)
        await runtime.start()
        await runtime.wait(root.id, timeout=25)
        record = {
            "task_id": "0",
            "functional_pass": True,
            "style_pass": True,
            "overall_pass": True,
            "wall_time_seconds": 2.0,
            "stop_reason": "completed",
        }
        row, _, _, events = task_observability(tmp_path, record)
        assert row["continual_harness"]["explicit_refine_calls"] == 1
        assert row["continual_harness"]["refinements_applied"] == 1
        assert row["continual_harness"]["memory_edits"] == 1
        assert row["continual_harness"]["later_root_inputs_receiving_refinement_notice"] == 1
        assert any(e["later_root_invocations_receiving_notice"] for e in events)
        # A second task's complete local AND global stores remain empty.
        other = Runtime(tmp_path / "other-state", providers={"mock": RefineProvider()})
        try:
            fresh = other.create("new task", tmp_path, config=config)
            assert not any(other.store.harness.merged(fresh.id)["entries"].values())
        finally:
            await other.shutdown()
    finally:
        await runtime.shutdown()
