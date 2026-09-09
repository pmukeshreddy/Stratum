"""Focused runtime regressions: real workers, real checks, no model benchmark."""

import array
import asyncio
import json
import sys
import tracemalloc

import pytest

from threadweave.coding_config import update_coding_options
from threadweave.kernel import Kernel, fork_checkpoint
from threadweave.kernel_worker import pack, unpack
from threadweave.models import ModelResponse, Outcome, RunConfig, new_id
from threadweave.runtime import Runtime
from threadweave.snapshot_io import SnapshotLimit
from threadweave.snapshots import SnapshotBlobs
from threadweave.tools import ToolContext

from .conftest import eventually, response
from .fakes import ScriptedProvider


def context(runtime, sid):
    return ToolContext(runtime, sid, new_id(), runtime.store.event(sid, "test_observation", {}))


def test_streaming_limits_peak_memory_and_removes_partial_writes(tmp_path):
    from threadweave.models import KernelStatePolicy

    value = "x\r\n☃" * 2_000_000
    policy = KernelStatePolicy(artifact_bytes=32 * 1024 * 1024, stream_chunk_bytes=4096)
    blobs = SnapshotBlobs(tmp_path, policy)
    blobs.begin()
    oversized = value + "different"
    tracemalloc.start()
    try:
        record, size = blobs.encode("large", value, pack)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert size == len(value.encode())
    assert peak < 2 * 1024 * 1024  # Encoding does not allocate another value-sized buffer.
    assert blobs.stats["max_write_bytes"] <= policy.stream_chunk_bytes
    assert blobs.decode(record, unpack) == value
    prior = set(blobs.directory.iterdir())
    blobs.policy.artifact_bytes = 1024 * 1024
    blobs.begin()
    tracemalloc.start()
    try:
        with pytest.raises(SnapshotLimit):
            blobs.encode("too-large", oversized, pack)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024
    assert set(blobs.directory.iterdir()) == prior
    binary = b"z" * (8 * 1024 * 1024)
    tracemalloc.start()
    try:
        with pytest.raises(SnapshotLimit):
            blobs.encode("binary", binary, pack)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
    assert set(blobs.directory.iterdir()) == prior


def test_streamed_containers_procedures_and_native_arrays(tmp_path):
    blobs = SnapshotBlobs(tmp_path)
    value = {"rows": [(i, b"bytes", "line\r\n") for i in range(10000)]}
    blobs.begin()
    record, _ = blobs.encode("rows", value, pack)
    assert blobs.decode(record, unpack) == value
    native = array.array("Q", range(500000))
    blobs.begin()
    tracemalloc.start()
    try:
        record, _ = blobs.encode("array", native, pack)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
    assert record[1]["codec"] == "array"
    assert blobs.decode(record, unpack) == native
    namespace = {"__name__": "__session__", "large": "capture" * 300000}
    exec("def result(): return len(large)", namespace)
    blobs.begin()
    record, _ = blobs.encode("function", namespace["result"], pack)
    assert record[1]["dependencies"]
    assert blobs.decode(record, unpack)() == len(namespace["large"])
    dependency = record[1]["dependencies"][0][1]["sha256"]
    (blobs.directory / dependency).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size|checksum"):
        blobs.decode(record, unpack)
    prior = set(blobs.directory.iterdir())
    with (tmp_path / "opaque").open("w") as stream:
        namespace["stream"] = stream
        exec("def bad(): return stream, large", namespace)
        blobs.begin()
        with pytest.raises(ValueError, match="Opaque"):
            blobs.encode("bad", namespace["bad"], pack)
    assert set(blobs.directory.iterdir()) == prior


async def test_aliases_large_state_and_fork_survive_worker_restart(tmp_path):
    async def bridge(*args, **kwargs):
        raise AssertionError("No host call required")

    policy = {"variable_bytes": 1024 * 1024, "artifact_bytes": 16 * 1024 * 1024}
    directory = tmp_path / "kernel"
    kernel = Kernel(directory, tmp_path, bridge, bootstrap={"kernel_state": policy})
    try:
        result = await kernel.execute(
            new_id(),
            "a = [1, 2]\nb = a\ntext = 'a\\r\\nb' * 1000000\n"
            "nested = []\nleft = [nested]\nright = [nested]\n"
            "paths = [workspace / 'file' for _ in range(2000)]\n"
            "literal = str(workspace / 'file')",
            20,
        )
        assert not result["error"]
        assert "text" in result["snapshot_metrics"]["offloaded"]
        manifest = json.loads((directory / "checkpoint.json").read_text())
        assert manifest["reference_identity"]["top_level_aliases"] == "preserved"
        assert "not guaranteed" in manifest["reference_identity"]["cross_variable_nested_aliases"]
        await kernel.close()
        result = await kernel.execute(
            new_id(),
            "assert a is b\na.append(3)\nassert b[-1] == 3\n"
            "assert repl_state.rehydrate('text') == 'a\\r\\nb' * 1000000\n"
            "assert left[0] is not right[0]",
            20,
        )
        assert not result["error"]
        await kernel.close()
        child_directory = tmp_path / "child"
        child_directory.mkdir()
        candidate = tmp_path / "candidate"
        candidate.mkdir()
        tracemalloc.start()
        try:
            fork_checkpoint(
                directory / "checkpoint.json",
                child_directory / "checkpoint.json",
                tmp_path,
                candidate,
                owner="child",
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2 * 1024 * 1024
        child = Kernel(
            child_directory,
            candidate,
            bridge,
            bootstrap={"session_id": "child", "kernel_state": policy},
        )
        try:
            result = await child.execute(
                new_id(),
                "assert a is b\nassert len(repl_state.rehydrate('text')) == 4000000\n"
                "assert paths == [workspace / 'file'] * 2000\n"
                f"assert literal == {str(tmp_path / 'file')!r}",
                20,
            )
            assert not result["error"]
        finally:
            await child.close()
    finally:
        await kernel.close()


async def test_full_verification_receipts_revision_config_environment_and_resume(
    tmp_path, repository, coding_config
):
    update_coding_options(coding_config.task, capture_baseline=False)
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    sid = None
    try:
        root = runtime.create("Fix addition", repository, config=coding_config)
        sid = root.id
        await runtime._prepare(sid)
        source = context(runtime, sid).source_event
        failed, _ = await runtime._verify(sid, source)
        assert not failed.passed
        failed_again, _ = await runtime._verify(sid, source)
        assert not failed_again.passed
        assert not runtime.store.events(sid, kind="verification_receipt_saved")
        (repository / "mathops.py").write_text(
            "def add(a,b): return a+b\ndef twice(x): return add(x,x)\n"
        )
        passed, error = await runtime._verify(sid, source)
        assert passed.passed and not error
        count = len(runtime.store.events(sid, kind="verification_executed"))
        both = await asyncio.gather(runtime._verify(sid, source), runtime._verify(sid, source))
        assert all(result.passed and not error for result, error in both)
        assert len(runtime.store.events(sid, kind="verification_executed")) == count
        assert len(runtime.store.events(sid, kind="verification_receipt_reused")) == 2
        (repository / "mathops.py").write_text(
            "def add(a,b): return b+a\ndef twice(x): return add(x,x)\n"
        )
        assert (await runtime._verify(sid, source))[0].passed
        assert len(runtime.store.events(sid, kind="verification_executed")) == count + 1
        (repository / "pytest.ini").write_text("[pytest]\naddopts = --strict-config\n")
        assert (await runtime._verify(sid, source))[0].passed
        config = runtime.store.config(sid)
        update_coding_options(
            config.task,
            test_commands=[[sys.executable, "-m", "pytest", "-q", "--disable-warnings"]],
        )
        config.execution.environment["LC_ALL"] = "C"
        runtime.store.reconfigure(sid, config)
        assert (await runtime._verify(sid, source))[0].passed
        commands = runtime.store.events(sid, kind="coding_command")
        assert "--disable-warnings" in commands[-1]["payload"]["requested_command"]
        assert runtime.store.events(sid, kind="verification_receipt_invalidated")
        count = len(runtime.store.events(sid, kind="verification_executed"))
    finally:
        await runtime.shutdown()
    restored = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    try:
        await restored.recover()
        result, error = await restored._verify(sid, context(restored, sid).source_event)
        assert result.passed and not error
        assert len(restored.store.events(sid, kind="verification_executed")) == count
    finally:
        await restored.shutdown()


async def test_different_worktree_never_reuses_parent_receipt(tmp_path, repository, coding_config):
    update_coding_options(coding_config.task, capture_baseline=False)
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    try:
        root = runtime.create("Fix addition", repository, config=coding_config)
        await runtime._prepare(root.id)
        (repository / "mathops.py").write_text(
            "def add(a,b): return a+b\ndef twice(x): return add(x,x)\n"
        )
        assert (await runtime._verify(root.id, context(runtime, root.id).source_event))[0].passed
        child = await runtime.spawn_async(root.id, "Check candidate", purpose="candidate")
        await runtime._prepare(child.id)
        result, error = await runtime._verify(child.id, context(runtime, child.id).source_event)
        assert result.passed and not error
        assert runtime.store.events(child.id, kind="verification_executed")
        assert not runtime.store.events(child.id, kind="verification_receipt_reused")
    finally:
        await runtime.shutdown()


async def test_receipt_invalidates_changed_external_python_dependency(
    tmp_path, repository, coding_config
):
    update_coding_options(coding_config.task, capture_baseline=False)
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    source = dependency / "external_add.py"
    source.write_text("def add(a,b): return a+b\n")
    coding_config.execution.environment_allowlist.append("PYTHONPATH")
    coding_config.execution.environment["PYTHONPATH"] = str(dependency)
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})
    try:
        root = runtime.create("Use the configured dependency", repository, config=coding_config)
        await runtime._prepare(root.id)
        (repository / "mathops.py").write_text(
            "from external_add import add\ndef twice(x): return add(x,x)\n"
        )
        assert (await runtime._verify(root.id, context(runtime, root.id).source_event))[0].passed
        assert runtime.store.events(root.id, kind="verification_receipt_saved")
        source.write_text("def add(a,b): return a-b-1\n")
        result, error = await runtime._verify(root.id, context(runtime, root.id).source_event)
        assert not result.passed and not error
        assert runtime.store.events(root.id, kind="verification_receipt_invalidated")
        assert not runtime.store.events(root.id, kind="verification_receipt_reused")
    finally:
        await runtime.shutdown()


async def test_receipt_fingerprint_failure_runs_gate_once(
    tmp_path, repository, coding_config, monkeypatch
):
    from threadweave.verification_receipts import VerificationReceipts

    update_coding_options(coding_config.task, capture_baseline=False)
    runtime = Runtime(tmp_path / "state", providers={"test": ScriptedProvider({})})

    def unavailable(*args):
        raise OSError("Dependency disappeared during reconciliation")

    try:
        root = runtime.create("Fix addition", repository, config=coding_config)
        await runtime._prepare(root.id)
        (repository / "mathops.py").write_text(
            "def add(a,b): return a+b\ndef twice(x): return add(x,x)\n"
        )
        monkeypatch.setattr(VerificationReceipts, "environment_files", unavailable)
        result, error = await runtime._verify(root.id, context(runtime, root.id).source_event)
        assert result.passed and not error
        assert len(runtime.store.events(root.id, kind="verification_executed")) == 1
        assert not runtime.store.events(root.id, kind="verification_receipt_saved")
        assert runtime.store.events(root.id, kind="verification_receipt_invalidated")
    finally:
        await runtime.shutdown()


async def test_failed_child_followup_recovers_own_repl_and_stable_id_messaging(tmp_path):
    provider = ScriptedProvider(
        {
            "root": [
                response("ipython", code="await agents.wait(seconds=0.05)"),
                *[ModelResponse(text="Done") for _ in range(10)],
            ],
            "worker": [
                response(
                    "ipython",
                    code="assert state == 41\nstate += 1\nawait agent_message.send(str(state), receiver_role='parent')",
                ),
                ModelResponse(text="Recovered 42"),
            ],
        }
    )
    config = RunConfig(
        provider={"name": "mock", "model": "deterministic"}, refinement={"enabled": False}
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": provider})
    try:
        root = runtime.create("Recover a failed branch", tmp_path, config=config)
        child = runtime.spawn(root.id, "Work independently", name="worker")
        result = await runtime.execute_python(context(runtime, child.id), "state = 41")
        assert not result["error"]
        runtime.store.finish(child.id, Outcome.FAILED, "Interrupted investigation")
        await runtime._close_kernel(child.id)
        result = await runtime.execute_python(
            context(runtime, root.id),
            f"await agents.followup({child.id!r}, 'Retry from retained state')\n"
            f"await agent_message.send('Use your retained state', receiver_id={child.id!r})",
        )
        assert not result["error"]
        assert runtime.store.session(child.id).kernel_id == child.kernel_id
        await runtime.start()
        assert (await runtime.wait(root.id)).outcome == Outcome.COMPLETED
        assert runtime.store.session(child.id).depth == 1
        assert runtime.store.session(child.id).result == "Recovered 42"
        assert any(m["body"] == "42" for m in runtime.store.messages(root.id))
        assert runtime.store.usage(root.id, tree=True).subagent_count == 1
    finally:
        await runtime.shutdown()


async def test_parent_cancellation_cleans_completed_child_background_process(tmp_path):
    config = RunConfig(
        provider={"name": "mock", "model": "deterministic"}, refinement={"enabled": False}
    )
    config.permissions.append("process")
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        root = runtime.create("Cancel all owned execution", tmp_path, config=config)
        child = runtime.spawn(root.id, "Background computation")
        result = await runtime.execute_python(context(runtime, child.id), "job = bash('sleep 60')")
        assert not result["error"]
        await eventually(lambda: runtime.background.processes)
        process = next(iter(runtime.background.processes.values()))
        runtime.store.finish(child.id, Outcome.COMPLETED, "Background process started")
        await runtime.stop(root.id)
        assert process.returncode is not None
        assert all(task.done() for task in runtime.background.tasks.values())
        assert runtime.store.session(child.id).outcome == Outcome.COMPLETED
        with pytest.raises(ValueError, match="stopped parent"):
            runtime.resume(child.id)
    finally:
        await runtime.shutdown()


async def test_restart_does_not_admit_orphan_after_interrupted_parent_stop(tmp_path):
    config = RunConfig(
        provider={"name": "mock", "model": "deterministic"}, refinement={"enabled": False}
    )
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    root = runtime.create("Interrupted cancellation", tmp_path, config=config)
    child = runtime.spawn(root.id, "Child")
    grandchild = runtime.spawn(child.id, "Grandchild")
    runtime.store.finish(root.id, Outcome.CANCELLED, "Cancelled before child cleanup committed")
    await runtime.shutdown()
    runtime = Runtime(tmp_path / "state", providers={"mock": ScriptedProvider({})})
    try:
        await runtime.recover()
        assert runtime.store.session(child.id).outcome == Outcome.CANCELLED
        assert runtime.store.session(grandchild.id).outcome == Outcome.CANCELLED
        assert runtime.store.session(grandchild.id).depth == 2
        assert runtime.store.events(child.id, kind="orphan_child_cancelled")
    finally:
        await runtime.shutdown()
