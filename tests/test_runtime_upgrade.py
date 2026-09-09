"""Lifecycle integration over real workers, durable state and production orchestration.

Scripted inference tests transport/causality. The separate live test evaluates the
unforced model policy against real repositories and independent acceptance checks.
"""

import json
import os
import sys

import pytest

from threadweave.context import python_instructions
from threadweave.evals.runtime_upgrade import evaluate
from threadweave.kernel import Kernel
from threadweave.models import Action, ModelResponse, Outcome, RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.tools import ToolContext


async def bridge(*args, **kwargs):
    raise AssertionError("No host calls expected")


def test_repl_is_the_operating_doctrine():
    config = RunConfig()
    text = python_instructions(config)
    assert "IPython is the default working and control environment" in text
    for item in (
        "coding and repository",
        "conflicting instructions",
        "long context",
        "multi-step reasoning",
        "structured extraction",
        "search/filtering",
        "uncertain intermediate",
        "mechanically checkable",
    ):
        assert item in text
    for forbidden in (
        "executing code when useful",
        "Direct reasoning and a direct answer are valid",
        "solve and answer directly",
        "execution is optional",
    ):
        assert forbidden not in text
    assert config.refinement.enabled and not config.task.verify_each_turn
    assert "tool_choice" not in config.provider.parameters


def kernel(tmp_path, owner="root"):
    return Kernel(
        tmp_path / owner,
        tmp_path,
        bridge,
        bootstrap={
            "session_id": owner,
            "kernel_state": {
                "variable_bytes": 4096,
                "memory_bytes": 64000,
                "stale_cells": 2,
                "checkpoint_cells": 1,
                "artifact_bytes": 1000000,
            },
        },
    )


async def test_l2_full_lifecycle_and_restart(tmp_path):
    root = kernel(tmp_path)
    try:
        result = await root.execute(
            new_id(),
            """small = {'answer': 42}
useful = 'u' * 9000
stale = 's' * 9000
source = workspace / 'source.txt'
source.write_text('r' * 9000)
rebuild = source.read_text()
remember_recipe('rebuild', 'rebuild = source.read_text()', dependencies=['source'])
opaque = iter([1,2,3])
""",
            10,
        )
        assert result["error"] is None
        assert {"useful", "stale"} <= set(result["snapshot_metrics"]["offloaded"])
        assert result["kernel_state"]["rebuild"]["action"] == "reconstruct"
        assert result["kernel_state"]["opaque"]["action"] == "skip"
        await root.execute(new_id(), 'assert small["answer"] == 42\nuseful', 10)
        result = await root.execute(new_id(), 'assert small["answer"] == 42\nuseful', 10)
        assert "stale" in result["snapshot_metrics"]["pruned"]
        assert "stale" not in result["variables"]
        pid = root.process.pid
        result = await root.checkpoint_state("l1_compaction")
        assert root.process.pid == pid
        assert result["snapshot_metrics"]["reason"] == "l1_compaction"
    finally:
        await root.close()
    root = kernel(tmp_path)
    try:
        await root.start()
        assert "small" in root.recovery["restored"] and "opaque" in root.recovery["missing"]
        result = await root.execute(
            new_id(),
            """assert small == {'answer':42}
assert len(repl_state.rehydrate('useful')) == 9000
assert len(repl_state.rehydrate('rebuild')) == 9000
assert len(repl_state.rehydrate('stale')) == 9000
""",
            10,
        )
        assert result["error"] is None
    finally:
        await root.close()


async def test_child_state_isolation_and_corruption(tmp_path):
    for owner in ("root", "child_a", "child_b"):
        worker = kernel(tmp_path, owner)
        try:
            await worker.execute(new_id(), f"marker = {owner!r}\nlarge = 'x' * 9000", 10)
        finally:
            await worker.close()
    for owner in ("child_b", "root", "child_a"):
        worker = kernel(tmp_path, owner)
        try:
            result = await worker.execute(new_id(), f"assert marker == {owner!r}", 10)
            assert result["error"] is None
        finally:
            await worker.close()
    checkpoint = tmp_path / "child_a/checkpoint.json"
    data = json.loads(checkpoint.read_text())
    digest = data["manifest"]["large"]["record"][1]["sha256"]
    (tmp_path / "child_a/values" / digest).write_bytes(b"corrupt")
    worker = kernel(tmp_path, "child_a")
    try:
        result = await worker.execute(
            new_id(), "assert marker == 'child_a'\nrepl_state.rehydrate('large')", 10
        )
        assert result["error"]["code"] == "ValueError"
    finally:
        await worker.close()
    checkpoint.write_text("{bad")
    worker = kernel(tmp_path, "child_a")
    try:
        await worker.start()
        assert "__checkpoint__" in worker.recovery["missing"]
        assert (await worker.execute(new_id(), "assert marker == 'child_a'", 10))["error"] is None
    finally:
        await worker.close()


async def test_snapshot_write_failure_never_evicts_live_value(tmp_path):
    worker = kernel(tmp_path)
    try:
        result = await worker.execute(
            new_id(),
            """import threadweave.kernel_state as ks
original_write = ks.atomic_write
def broken(*args, **kwargs):
    raise OSError('disk unavailable')
ks.atomic_write = broken
important = 'z' * 9000
""",
            10,
        )
        assert result["snapshot_metrics"]["commit_failed"]
        result = await worker.execute(
            new_id(), "assert isinstance(important, str)\nks.atomic_write = original_write", 10
        )
        assert result["error"] is None and not result["snapshot_metrics"].get("commit_failed")
    finally:
        await worker.close()


class RecursiveLearningProvider:
    def __init__(self):
        self.requests = []

    async def invoke(self, request, emit):
        self.requests.append(request)
        purpose = request.metadata.get("purpose", "agent")
        if purpose == "refinement_review":
            return ModelResponse(
                text=json.dumps(
                    {
                        "shouldRefine": True,
                        "rationale": "Validated independent child debugging procedure is reusable",
                        "instructions": "Record the procedure",
                    }
                )
            )
        if purpose == "refinement":
            evidence = json.loads(request.messages[-1]["content"])
            sources = [r["id"] for r in evidence["trajectory"] if r["type"] == "python_result"]
            return ModelResponse(
                text=json.dumps(
                    {
                        "proposals": [
                            {
                                "kind": "memory",
                                "title": "Validated debugging procedure",
                                "content": {
                                    "text": "Verify integer aggregation with independent subtotal checks before combining results."
                                },
                                "source_events": sources[:1],
                                "intended_effect": "Reuse independent subtotal checks",
                                "select": True,
                            }
                        ]
                    }
                )
            )
        if request.parent_id:
            return (
                ModelResponse(
                    actions=[
                        Action(
                            name="ipython",
                            arguments={
                                "code": "subtotal = sum([11, 13])\nassert subtotal == 24\nprint(subtotal)"
                            },
                        )
                    ]
                )
                if request.turn == 0
                else ModelResponse(
                    text="Independent subtotal is 24; checked by assertion in the child REPL."
                )
            )
        if request.turn == 0:
            return ModelResponse(
                actions=[
                    Action(
                        name="ipython",
                        arguments={
                            "code": "h = await rlm('Independently verify the subtotal of 11 and 13', name='subtotal')\nvalues = [11, 13]\nassert sum(values) == 24\nprint('Reusable check: assert independent subtotal before combining')"
                        },
                    )
                ]
            )
        assert request.metadata["execution_inputs"]["child_evidence"]
        assert request.metadata["execution_inputs"]["harness_state"]
        assert "independent subtotal checks" in str(request.messages)
        return ModelResponse(
            actions=[
                Action(
                    name="ipython",
                    arguments={
                        "code": "assert sum(values) == 24\nprint('Consumed verified child subtotal and validated harness guidance')"
                    },
                )
            ]
        )


async def test_l1_tree_compaction_and_recovery(tmp_path):
    config = RunConfig(
        provider={"name": "mock", "model": "deterministic", "max_output_tokens": 128},
        features={"model_compaction": False},
        kernel_state={"variable_bytes": 4096, "memory_bytes": 64000},
        context={"summary_tokens": 3000},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": RecursiveLearningProvider()})
    try:
        root = runtime.create(
            "Keep unresolved requirements and REPL state", tmp_path, config=config
        )
        ctx = ToolContext(runtime, root.id, new_id(), runtime.store.events(root.id)[0]["id"])
        result = await runtime.execute_python(
            ctx,
            "small = 42\nbig = 'x' * 9000\nitem = context.track('Still must validate the migration checksum')",
        )
        assert result["error"] is None
        for i in range(8):
            eid = runtime.store.event(root.id, "observation", {"iteration": i})
            runtime.store.add_context(
                root.id, eid, [{"role": "assistant", "content": "Observed log data. " * 500}]
            )
        pid = runtime.kernels[root.id].process.pid
        compaction = runtime.context.compact(root.id)
        await runtime._sync_l2_compaction(root.id)
        assert runtime.kernels[root.id].process.pid == pid
        assert (
            "Still must validate the migration checksum" in runtime.store.session(root.id).summary
        )
        tree = runtime.artifacts.load(
            root.id, runtime.store.event_by_id(compaction)["payload"]["semantic_tree_artifact"]
        )
        assert tree["tracked_work"][0]["status"] == "open" and tree["kernel_snapshot"]
        root_id = root.id
    finally:
        await runtime.shutdown()
    runtime = Runtime(tmp_path / "state", providers={"mock": RecursiveLearningProvider()})
    try:
        await runtime.recover()
        ctx = ToolContext(runtime, root_id, new_id(), compaction)
        result = await runtime.execute_python(
            ctx, "assert small==42\nassert len(repl_state.rehydrate('big'))==9000"
        )
        assert result["error"] is None
    finally:
        await runtime.shutdown()


async def test_verification_levels_and_no_repeated_full_gate(tmp_path, repository, coding_config):
    coding_config.control_plane = "python"
    coding_config.refinement.enabled = False
    coding_config.verification.targeted_commands = [
        [sys.executable, "-m", "pytest", "-q", "tests/test_mathops.py"]
    ]
    runtime = Runtime(tmp_path / "state", providers={"test": RecursiveLearningProvider()})
    try:
        root = runtime.create("Fix addition", repository, config=coding_config)
        await runtime._prepare(root.id)
        ctx = ToolContext(runtime, root.id, new_id(), runtime.store.events(root.id)[0]["id"])
        await runtime.execute_python(
            ctx, "(workspace/'mathops.py').write_text('def add(a,b):\\n    return a +\\n')"
        )
        result = await runtime.verification.run(root.id, ctx.source_event)
        assert result["level"] == 1 and not result["passed"]
        assert not runtime.store.events(root.id, kind="verifier_started")
        ctx = ToolContext(runtime, root.id, new_id(), ctx.source_event)
        await runtime.execute_python(
            ctx,
            "(workspace/'mathops.py').write_text('def add(a,b):\\n    return a + b\\n\\ndef twice(value):\\n    return add(value,value)\\n')",
        )
        result = await runtime.verification.run(root.id, ctx.source_event, level=2)
        assert result["level"] == 2 and result["passed"]
        count = len(runtime.store.events(root.id, kind="verification_targeted"))
        await runtime.verification.run(root.id, ctx.source_event)
        assert len(runtime.store.events(root.id, kind="verification_targeted")) == count
        full, error = await runtime._verify(root.id, ctx.source_event)
        assert full.passed and not error
        assert runtime.store.events(root.id, kind="verifier_started")[-1]["payload"]["level"] == 3
        verifier = runtime.store.events(root.id, kind="verifier_result")[-1]
        runtime.context.compact(root.id, count=len(runtime.store.session(root.id).context))
        visible = next(
            m["content"]
            for m in runtime.context.messages(root.id)
            if m.get("content", "").startswith("Live completion evidence")
        )
        assert verifier["id"] in visible and '"passed":true' in visible
        assert "stdout" not in visible
        # Actual test-runner artifacts must not create a verification feedback loop.
        runtime.store.update(root.id, turns=20)
        await runtime.execute_python(
            ToolContext(runtime, root.id, new_id(), ctx.source_event),
            "(workspace/'.pytest_cache').mkdir(exist_ok=True)\n"
            "(workspace/'.pytest_cache'/'runtime-cache').write_text('updated cache')",
        )
        await runtime.verification.run(root.id, ctx.source_event)
        assert len(runtime.store.events(root.id, kind="verification_targeted")) == count
    finally:
        await runtime.shutdown()


@pytest.mark.skipif(
    os.environ.get("BUFFALO_BEHAVIORAL_LIVE") != "1", reason="explicit real model evaluation"
)
async def test_unforced_repl_policy_on_real_coding_tasks(tmp_path):
    reports = await evaluate(tmp_path / "live", names=["identifier_repair"])
    assert all(r["root_entered_repl"] and r["acceptance_passed"] for r in reports)


async def test_real_oversized_string_survives_old_16mib_boundary(tmp_path):
    worker = Kernel(tmp_path / "kernel", tmp_path, bridge)
    try:
        result = await worker.execute(new_id(), "payload = 'valuable' * (3 * 1024 * 1024)", 15)
        assert result["error"] is None
        assert result["kernel_state"]["payload"]["action"] == "offload"
        assert result["kernel_state"]["payload"]["serialized_bytes"] > 16 * 1024 * 1024
        assert "payload" not in result["not_checkpointed"]
        result = await worker.execute(
            new_id(), "assert len(payload.load()) == 8 * 3 * 1024 * 1024", 15
        )
        assert result["error"] is None
    finally:
        await worker.close()


async def test_child_reads_global_harness_and_followup_preserves_kernel(tmp_path):

    runtime = Runtime(tmp_path / "state", providers={"mock": RecursiveLearningProvider()})
    try:
        root = runtime.create(
            "Investigate",
            tmp_path,
            config=RunConfig(provider={"name": "mock", "model": "deterministic"}),
        )
        from .test_continual_harness import edit, proposal

        source = runtime.store.events(root.id)[0]["id"]
        runtime.store.harness.apply(root.id, proposal(edit()), id="global", global_=True)
        child = runtime.spawn(root.id, "Check independent behavior", name="worker")
        assert runtime.store.harness.entries(child.id) == runtime.store.harness.entries(root.id)
        ctx = ToolContext(runtime, child.id, new_id(), source)
        await runtime.execute_python(ctx, "marker = 17")
        runtime.store.finish(child.id, Outcome.COMPLETED, "Checked")
        ctx = ToolContext(runtime, root.id, new_id(), source)
        result = await runtime.execute_python(
            ctx, f"await agents.followup({child.id!r}, 'Continue checking')"
        )
        assert result["error"] is None
        assert runtime.store.session(child.id).kernel_id == child.kernel_id
        assert runtime.store.session(child.id).outcome == "active"
        result = await runtime.execute_python(
            ToolContext(runtime, child.id, new_id(), source), "assert marker == 17"
        )
        assert result["error"] is None
    finally:
        await runtime.shutdown()


async def test_targeted_pytest_preserves_option_operands(tmp_path, repository, coding_config):
    from threadweave.coding_config import update_coding_options

    coding_config.control_plane = "python"
    update_coding_options(
        coding_config.task,
        test_commands=[
            [sys.executable, "-m", "pytest", "-q", "--ignore", "tests/ignored", "tests"]
        ],
    )
    runtime = Runtime(tmp_path / "state", providers={"test": RecursiveLearningProvider()})
    try:
        root = runtime.create("Fix arithmetic", repository, config=coding_config)
        await runtime._prepare(root.id)
        ctx = ToolContext(runtime, root.id, new_id(), runtime.store.events(root.id)[0]["id"])
        await runtime.execute_python(
            ctx,
            "(workspace/'tests/ignored').mkdir()\np = workspace/'mathops.py'\np.write_text(p.read_text().replace('a - b','a + b'))",
        )
        result = await runtime.verification.run(root.id, ctx.source_event, level=2)
        assert result["level"] == 2 and result["passed"]
        command = runtime.store.events(root.id, kind="verification_targeted")[-1]["payload"][
            "commands"
        ][0]
        assert command[command.index("--ignore") + 1] == "tests/ignored"
    finally:
        await runtime.shutdown()


async def test_full_gate_waits_for_pending_child_evidence(tmp_path):
    class Finish:
        async def invoke(self, request, emit):
            return ModelResponse(text="The answer is ready.")

    (tmp_path / "answer.txt").write_text("42")
    config = RunConfig(
        provider={"name": "mock", "model": "deterministic"},
        task={
            "verifier": "file",
            "verifier_options": {"path": "answer.txt", "equals": "42"},
            "require_verifier": True,
        },
        refinement={"enabled": False},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": Finish()})
    try:
        root = runtime.create("Write answer.txt", tmp_path, config=config)
        child = runtime.spawn(root.id, "Inspect the answer")
        await runtime._run_turn(root.id)
        assert runtime.store.session(root.id).outcome == "active"
        assert runtime.store.events(root.id, kind="completion_deferred")
        assert not runtime.store.events(root.id, kind="verifier_started")
        await runtime._run_turn(child.id)
        await runtime._run_turn(root.id)
        assert len(runtime.store.events(root.id, kind="verifier_started")) == 1
        assert runtime.store.session(root.id).outcome == "completed"
    finally:
        await runtime.shutdown()


async def test_durable_repl_tree_compacts_without_auxiliary_inference(tmp_path):
    class NoInference:
        async def invoke(self, request, emit):
            raise AssertionError("Structured checkpoint should not require inference")

    config = RunConfig(
        provider={"name": "mock", "model": "gpt-6-astra", "max_output_tokens": 128},
        context={"max_tokens": 12000, "recent_blocks": 1},
        refinement={"enabled": False},
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": NoInference()})
    try:
        root = runtime.create("Keep evidence", tmp_path, config=config)
        ctx = ToolContext(runtime, root.id, new_id(), runtime.store.events(root.id)[0]["id"])
        await runtime.execute_python(ctx, "answer = 42")
        for i in range(6):
            event = runtime.store.event(root.id, "observation", {"i": i})
            runtime.store.add_context(
                root.id,
                event,
                [
                    {
                        "role": "assistant",
                        "content": "Observed data and successful computation. " * 2500,
                    }
                ],
            )
        await runtime.semantic_compact(root.id)
        assert runtime.store.events(root.id, kind="semantic_compaction_projected")
        await runtime._sync_l2_compaction(root.id)
        assert (
            await runtime.execute_python(
                ToolContext(runtime, root.id, new_id(), ctx.source_event), "assert answer == 42"
            )
        )["error"] is None
    finally:
        await runtime.shutdown()


async def test_compacted_work_resolution_is_authoritative(tmp_path):
    from threadweave.context_budget import pending_ledger

    runtime = Runtime(tmp_path / "state", providers={"mock": RecursiveLearningProvider()})
    try:
        root = runtime.create(
            "Preserve unresolved work, retire completed branches",
            tmp_path,
            config=RunConfig(
                provider={"name": "mock", "model": "deterministic"}, refinement={"enabled": False}
            ),
        )
        child = runtime.spawn(root.id, "Inspect an independent requirement")

        def observation():
            event = runtime.store.event(root.id, "observation", {"checked": True})
            runtime.store.add_context(
                root.id, event, [{"role": "user", "content": "Observed result"}]
            )
            return event

        observation()
        runtime.context.compact(root.id)
        assert "child-" + child.id in {
            i["id"] for i in pending_ledger(runtime.store.session(root.id).summary)
        }
        runtime.store.finish(child.id, Outcome.COMPLETED, "Inspected; requirement met")
        observation()
        runtime.context.compact(root.id)
        assert "child-" + child.id not in {
            i["id"] for i in pending_ledger(runtime.store.session(root.id).summary)
        }

        # Compaction-created requirements are resolvable through the public REPL API.
        runtime.store.update(
            root.id,
            summary=json.dumps(
                {
                    "unresolved_requirements": [
                        {"id": "compacted-check", "text": "Check the observed result"}
                    ]
                }
            ),
        )
        source = observation()
        result = await runtime.execute_python(
            ToolContext(runtime, root.id, new_id(), source),
            f"resolved = context.resolve('compacted-check', evidence_events=[{source!r}])\n"
            "assert resolved['status'] == 'resolved'",
        )
        assert result["error"] is None
        assert not pending_ledger(runtime.store.session(root.id).summary)
        observation()
        runtime.context.compact(root.id)
        assert not pending_ledger(runtime.store.session(root.id).summary)
    finally:
        await runtime.shutdown()
