"""Resident evaluation trajectory using Buffalo's existing worker and Prime controller.

The host owns grading. This module has no benchmark imports, hidden assets, model
credentials, solving hints, or alternate RLM/refinement implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import os
import sys
from dataclasses import asdict
from pathlib import Path

from ..autonomous import AutonomousCycle, AutonomousPolicy, GateResult, git_snapshot
from .autonomous_rehearsal import RehearsalServer
from .evocode_rpc import Peer
from .evocode_worker import PersistentWorker

IDENTITY_FIELDS = (
    "runtime_id",
    "runtime_pid",
    "root_session_id",
    "kernel_id",
    "workspace",
    "workspace_device",
    "workspace_inode",
)
RETAINED_FIELDS = (
    "kernel_pid",
    "local_harness",
    "global_harness",
    "refinement_history",
    "assistant_turns_since_auto_refine",
    "children",
    "compaction_events",
)


def assert_identity(anchor, current):
    for key in IDENTITY_FIELDS:
        if anchor[key] != current[key]:
            raise RuntimeError(f"TRAJECTORY IDENTITY VIOLATION: {key}")
    if anchor.get("kernel_pid") and anchor["kernel_pid"] != current["kernel_pid"]:
        raise RuntimeError("TRAJECTORY IDENTITY VIOLATION: kernel_pid")


def source_files(workspace):
    """Observation only. Attribution is withheld for overlapping execution spans."""
    result = {}
    for directory, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "target", "verification", "__pycache__"}]
        for name in files:
            path = Path(directory) / name
            relative = str(path.relative_to(workspace))
            try:
                if path.is_symlink():
                    result[relative] = {"link": os.readlink(path)}
                elif path.is_file():
                    with path.open("rb") as stream:
                        result[relative] = {
                            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                            "mode": path.stat().st_mode & 0o777,
                        }
            except OSError as exc:
                result[relative] = {"observation_error": type(exc).__name__}
    return result


class AutonomousWorker(PersistentWorker):
    """One setup, one cycle, any number of ordinary Runtime.interact continuations."""

    def __init__(self):
        super().__init__()
        self.exported_seq = 0
        self.export_lock = asyncio.Lock()
        self.running_once = False
        self.rehearsal_server = None
        self.rehearsal_options = None

    async def dispatch(self, method, args):
        if method == "enable_rehearsal":
            if self.runtime is None or self.running_once or self.rehearsal_server:
                raise RuntimeError("Rehearsal must be configured once before the trajectory")
            self.rehearsal_options = args
            path = self.runtime.store.directory / "public-rehearsal.sock"
            self.rehearsal_server = RehearsalServer(path, self.rehearse)
            await self.rehearsal_server.start()
            return {"socket": str(path)}
        if method == "shutdown" and self.rehearsal_server:
            await self.rehearsal_server.close()
        return await super().dispatch(method, args)

    async def rehearse(self):
        if not self.running_once or self.last_packet is not None:
            raise RuntimeError("No active trajectory")
        before = self.audit()
        workspace = Path(before["workspace"])
        fingerprint = await git_snapshot(workspace)
        event = self.runtime.store.event(
            self.root_id,
            "evaluation_rehearsal_started",
            {"identity": before, "fingerprint": fingerprint},
        )
        await self.export_events()
        result = await self.peer.call(
            "rehearse",
            **self.rehearsal_options,
            identity=before,
            fingerprint=fingerprint,
            event_id=event,
        )
        assert_identity(before, self.audit())
        # Native children may keep working during a user-invoked tool call.
        # This is a snapshot measurement, never permission to accept later edits.
        result["workspace_changed_during_rehearsal"] = fingerprint != await git_snapshot(workspace)
        self.runtime.store.event(self.root_id, "evaluation_rehearsal_result", result, parent=event)
        await self.export_events()
        return result

    async def setup(self, instruction, workspace, directory, config):
        result = await super().setup(instruction, workspace, directory, config)
        original = self.runtime.store.event
        spans = {}

        def observe_event(sid, kind, payload, **kwargs):
            event_id = original(sid, kind, payload, **kwargs)
            if kind == "python_execution":
                sessions = {span["session_id"] for span in spans.values()} | {sid}
                for span in spans.values():
                    span["overlap"].update(sessions - {span["session_id"]})
                spans[event_id] = {
                    "session_id": sid,
                    "workspace": Path(self.runtime.store.session(sid).workspace.path),
                    "before": source_files(Path(self.runtime.store.session(sid).workspace.path)),
                    "overlap": sessions - {sid},
                }
            elif kind in {"python_result", "python_error"}:
                start = kwargs.get("parent")
                span = spans.pop(start, None)
                if span:
                    after = source_files(span["workspace"])
                    paths = sorted(set(span["before"]) | set(after))
                    changes = [
                        {"path": p, "before": span["before"].get(p), "after": after.get(p)}
                        for p in paths
                        if span["before"].get(p) != after.get(p)
                    ]
                    original(
                        sid,
                        "evaluation_file_observation",
                        {
                            "start_event": start,
                            "end_event": event_id,
                            "changes": changes,
                            "workspace": str(span["workspace"]),
                            "overlapping_sessions": sorted(span["overlap"]),
                            "attribution": "execution_span_only; background writers may overlap",
                        },
                        parent=event_id,
                    )
            return event_id

        self.runtime.store.event = observe_event
        return result

    async def export_events(self):
        async with self.export_lock:
            events = list(
                self.runtime.store.iter_events(
                    self.root_id,
                    tree=True,
                    after=self.exported_seq,
                )
            )
            for start in range(0, len(events), 100):
                batch = events[start : start + 100]
                await self.peer.call("evidence", events=batch)
                self.exported_seq = max(e["seq"] for e in batch)

    async def observe(self):
        while True:
            await self.export_events()
            session = self.runtime.store.session(self.root_id)
            await self.peer.call(
                "progress",
                status={
                    "root_id": self.root_id,
                    "kernel_id": session.kernel_id,
                    "workspace": session.workspace.path,
                    "root_turns": self.cycle.turns,
                    "noncached_budget_tokens": self.cycle.tokens,
                    "continuations": self.cycle.continuations,
                    "checks": len(self.cycle.checks),
                    "outcome": str(session.outcome),
                },
            )
            await asyncio.sleep(1)

    async def run(self, instruction, round_name, policy):
        async with self.lock:
            if self.running_once:
                raise RuntimeError("A trajectory cannot be restarted or have its budgets reset")
            self.running_once = True
            if instruction != self.first_instruction:
                raise ValueError("Initial instruction differs from the staged task")
            runtime, sid = self.runtime, self.root_id
            before = self.audit()
            anchor = dict(before)
            self.cycle = AutonomousCycle(AutonomousPolicy(**policy), [round_name])
            runtime.store.event(
                sid,
                "evaluation_trajectory_start",
                {
                    "identity": before,
                    "policy": policy,
                    "command": round_name,
                },
            )
            observer = asyncio.create_task(self.observe())
            settling = None
            # The official instruction is already staged by Runtime.create.
            runtime.store.update(sid, runnable=True)
            runtime._wake.set()

            async def snapshot():
                return await git_snapshot(Path(anchor["workspace"]))

            async def gate(command, timeout_seconds):
                gate_before = self.audit()
                assert_identity(anchor, gate_before)
                fingerprint = await snapshot()
                await self.export_events()
                result = await self.peer.call(
                    "gate",
                    command=command,
                    timeout_seconds=timeout_seconds,
                    identity=gate_before,
                    fingerprint=fingerprint,
                )
                gate_after = self.audit()
                assert_identity(gate_before, gate_after)
                for key in RETAINED_FIELDS:
                    if gate_before[key] != gate_after[key]:
                        raise RuntimeError(f"VERIFIER MUTATED RESIDENT STATE: {key}")
                if fingerprint != await snapshot():
                    raise RuntimeError("VERIFIER MUTATED CANDIDATE WORKSPACE")
                runtime.store.event(
                    sid,
                    "evaluation_verifier_result",
                    {
                        "host_attempt_id": result.pop("host_attempt_id", None),
                        "fingerprint": fingerprint,
                        "identity": gate_after,
                        **result,
                    },
                )
                return GateResult(**result)

            try:
                while True:
                    settling = asyncio.create_task(self.settle())
                    done, _ = await asyncio.wait(
                        {settling, observer},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if observer in done:
                        settling.cancel()
                        await asyncio.gather(settling, return_exceptions=True)
                        await observer  # Export/transport failures invalidate the trajectory.
                        raise RuntimeError("Evidence observer stopped unexpectedly")
                    await settling
                    current = self.audit()
                    assert_identity(anchor, current)
                    if not anchor.get("kernel_pid"):
                        anchor["kernel_pid"] = current["kernel_pid"]
                    responses = [
                        e
                        for e in runtime.store.iter_events(sid, kind="model_response")
                        if e["payload"].get("metadata", {}).get("purpose", "agent") == "agent"
                    ]
                    stop = (
                        responses[-1]["payload"].get("metadata", {}).get("stop_reason")
                        if responses
                        else None
                    )
                    feedback = await self.cycle.next_message(stop, snapshot=snapshot, run_gate=gate)
                    boundary = runtime.store.event(
                        sid,
                        "autonomous_gate_boundary",
                        {
                            "stop_reason": self.cycle.stop_reason,
                            "check": self.cycle.checks[-1] if self.cycle.checks else None,
                            "fingerprint": await snapshot(),
                            "identity": self.audit(),
                            "turns": self.cycle.turns,
                            "tokens": self.cycle.tokens,
                            "continuations": self.cycle.continuations,
                        },
                    )
                    if feedback is None:
                        break
                    runtime.store.event(
                        sid,
                        "evaluation_feedback",
                        {
                            "boundary_event": boundary,
                            "role": "user",
                            "text": feedback,
                        },
                    )
                    runtime.interact(sid, feedback)
            except BaseException as exc:
                self.cycle.stop_reason = (
                    "aborted" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                runtime.store.event(
                    sid,
                    "evaluation_trajectory_error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                await runtime.stop(sid)
                raise
            finally:
                if settling and not settling.done():
                    settling.cancel()
                    await asyncio.gather(settling, return_exceptions=True)
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
                self.last_packet = {
                    "before": before,
                    "after": self.audit(),
                    "autonomous": asdict(self.cycle),
                    "usage": runtime.store.usage(sid, tree=True).model_dump(mode="json"),
                    "events": list(runtime.store.iter_events(sid, tree=True)),
                    "requests": runtime.store.request_history(sid),
                    "messages": runtime.store.records.select("messages"),
                    "artifacts": runtime.store.records.select("artifacts"),
                }
                await self.export_events()
            return self.last_packet


async def serve(mailbox: Path | None = None):
    reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
    input_task = None
    if mailbox is None:
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            sys.stdin.buffer,
        )

        async def send(data):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    else:
        await asyncio.to_thread(mailbox.mkdir, parents=True, exist_ok=True)
        incoming, outgoing = mailbox / "in.jsonl", mailbox / "out.jsonl"
        incoming.touch(exist_ok=True)
        outgoing.touch(exist_ok=True)

        async def read_mailbox():
            with incoming.open("rb") as stream:
                while True:
                    data = await asyncio.to_thread(stream.read, 1024 * 1024)
                    if data:
                        reader.feed_data(data)
                    else:
                        await asyncio.sleep(0.05)

        async def send(data):
            with outgoing.open("ab") as stream:
                stream.write(data)
                stream.flush()

        input_task = asyncio.create_task(read_mailbox())
    worker = AutonomousWorker()
    worker.peer = Peer(reader, send, worker.dispatch)
    try:
        await worker.peer.pump
    finally:
        if input_task:
            input_task.cancel()
            await asyncio.gather(input_task, return_exceptions=True)
        if worker.runtime:
            if worker.rehearsal_server:
                await worker.rehearsal_server.close()
            with contextlib.suppress(Exception):
                await worker.runtime.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mailbox", type=Path)
    asyncio.run(serve(parser.parse_args().mailbox))


if __name__ == "__main__":
    main()
