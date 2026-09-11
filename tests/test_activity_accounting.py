from tests.diagnostics.activity import aggregate


def test_unknown_usage_stays_unknown_and_peaks_are_not_summed():
    rows = [
        {
            "input_tokens": 12,
            "rlm_calls": 2,
            "parallel_subagents_peak": 2,
            "max_recursive_depth": 1,
            "wall_seconds": 1.5,
        },
        {
            "input_tokens": None,
            "rlm_calls": 1,
            "parallel_subagents_peak": 1,
            "max_recursive_depth": 2,
            "wall_seconds": 2.0,
        },
    ]
    result = aggregate(rows)
    assert result["totals"]["input_tokens"] is None
    assert result["totals"]["rlm_calls"] == 3
    assert result["totals"]["parallel_subagents_peak"] == 2
    assert result["totals"]["max_recursive_depth"] == 2
    assert result["totals"]["wall_seconds"] == 3.5
    assert result["tasks_using"]["input_tokens"] == 1


async def test_peak_counts_resumed_children_without_counting_completed_idle_time(tmp_path, config):
    from tests.diagnostics.activity import activity
    from threadweave.models import Outcome
    from threadweave.runtime import Runtime

    runtime = Runtime(tmp_path / "state", providers={"mock": object()})
    try:
        root = runtime.create("root", tmp_path, config=config)
        first = runtime.spawn(root.id, "first", isolate=False)
        runtime.store.finish(first.id, Outcome.COMPLETED, "first result")
        second = runtime.spawn(root.id, "second", isolate=False)
        third = runtime.spawn(root.id, "third", isolate=False)
        runtime.store.finish(second.id, Outcome.COMPLETED, "second result")
        runtime.store.finish(third.id, Outcome.COMPLETED, "third result")
        runtime.message(root.id, first.id, "follow-up")
        assert runtime.store.session(first.id).outcome == "active"
        assert activity(tmp_path)["parallel_subagents_peak"] == 2
    finally:
        await runtime.shutdown()


async def test_child_receipts_only_describe_sources_present_in_actual_model_input(tmp_path, config):
    from threadweave.runtime import Runtime
    from threadweave.storage import encode

    runtime = Runtime(tmp_path / "state", providers={"mock": object()})
    try:
        root = runtime.create("root", tmp_path, config=config)
        child = runtime.spawn(root.id, "child", isolate=False)
        runtime.message(child.id, root.id, "useful evidence")
        delivered = runtime.receive(root.id)
        source = delivered[0]["source_event"]
        visible = runtime.context.messages(root.id)
        assert runtime.context.execution_inputs(root.id, visible)["child_evidence"] == {
            child.id: [source]
        }
        assert (
            runtime.context.execution_inputs(
                root.id, [{"role": "user", "content": "unrelated summary"}]
            )["child_evidence"]
            == {}
        )
        observed = runtime.store.event(child.id, "python_result", {"result": "kept only in Python"})
        runtime.store.event(
            root.id, "child_observation", {"child_id": child.id, "source_events": [observed]}
        )
        assert observed not in encode(runtime.context.execution_inputs(root.id, visible))
    finally:
        await runtime.shutdown()
