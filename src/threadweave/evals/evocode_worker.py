"""One resident production Runtime inside one Harbor task environment.

No Harbor package, task dataset, hidden verifier or credential is loaded here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from ..autonomous import AutonomousCycle, AutonomousPolicy, GateResult, git_snapshot
from ..models import HarnessError, ModelResponse, Outcome, ProviderConfig, RunConfig
from ..runtime import Runtime
from .evocode_rpc import Peer
from .schema import digest


class RemoteProvider:
    def __init__(self, worker):
        self.worker = worker

    async def resolve(self, config, *, reasoning_off=False):
        result = await self.worker.peer.call(
            "resolve",
            config=config.model_dump(mode="json"),
            reasoning_off=reasoning_off,
        )
        return ProviderConfig.model_validate(result["config"]), result["details"]

    @contextlib.asynccontextmanager
    async def control_factory(self):
        provider = self

        class Control:
            async def models(self):
                return await provider.worker.peer.call("models")

        yield Control()

    async def invoke(self, request, emit):
        result = await self.worker.peer.call("model", request=request.model_dump(mode="json"))
        response = ModelResponse.model_validate(result)
        if (
            self.worker.cycle
            and request.session_id == self.worker.root_id
            and request.metadata.get("purpose", "agent") == "agent"
        ):
            self.worker.cycle.record_response(response)
        return response


class PersistentWorker:
    def __init__(self):
        self.runtime = None
        self.root_id = None
        self.peer = None
        self.cycle = None
        self.rounds = 0
        self.lock = asyncio.Lock()
        self.identity = uuid.uuid4().hex
        self.previous_audit = None
        self.last_packet = None

    async def setup(self, instruction, workspace, directory, config):
        if self.runtime is not None or await asyncio.to_thread(Path(directory).exists):
            raise RuntimeError("Trial setup requires a fresh state directory and one Runtime")
        run = RunConfig.model_validate(config)
        if run.control_plane != "python" or not run.features.persistent_repl:
            raise ValueError("Persistent EvoCode execution requires Buffalo's Python control plane")
        if (
            run.refinement.turn_interval != 25
            or not run.refinement.enabled
            or not run.refinement.compact
            or run.refinement.cooldown_seconds != 1200
            or not run.features.subagents
        ):
            raise ValueError("This experiment requires unchanged enabled 25-turn refinement")
        run.serialized_refine = True
        # Runtime validates the presence of chat credentials before admission.
        # Actual credentials remain at the host proxy; this is never sent to an API.
        for provider in [run.provider, *run.models.values()]:
            if provider.name == "chat" and provider.api_key_env:
                os.environ.setdefault(provider.api_key_env, "host-managed-credential")
        proxy = RemoteProvider(self)
        self.runtime = Runtime(
            directory,
            providers={p.name: proxy for p in [run.provider, *run.models.values()]},
            idle_seconds=run.limits.wall_seconds + 60,
        )
        self.root_id = self.runtime.create(
            instruction,
            workspace,
            config=run,
            mode="interactive",
        ).id
        self.first_instruction = instruction
        await self.runtime.start()
        return self.audit()

    def audit(self):
        runtime, sid = self.runtime, self.root_id
        session = runtime.store.session(sid)
        events = list(runtime.store.iter_events(sid, tree=True))
        state = runtime.refinement_state(sid)
        kernel = runtime.kernels.get(sid)
        stat = Path(session.workspace.path).stat()
        checkpoint = runtime.store.directory / "kernels" / session.kernel_id / "checkpoint.json"
        value = {
            "runtime_id": self.identity,
            "runtime_pid": os.getpid(),
            "root_session_id": sid,
            "workspace": session.workspace.path,
            "workspace_device": stat.st_dev,
            "workspace_inode": stat.st_ino,
            "kernel_id": session.kernel_id,
            "kernel_pid": kernel.process.pid if kernel and kernel.process else None,
            "kernel_checkpoint": json.loads(checkpoint.read_text())
            if checkpoint.exists()
            else None,
            "event_count": len(events),
            "event_prefix_hash": digest([e["id"] for e in events]),
            "assistant_turns_since_auto_refine": state.turns_since_review,
            "last_refinement_review_at": state.last_review_at,
            "refinement_branch_version": state.branch_version,
            "refinement_status": runtime.refinement_status(sid),
            "refinement_history": runtime.store.harness.history(sid),
            "local_harness": runtime.store.harness.load(sid),
            "global_harness": runtime.store.harness.load(),
            "children": [
                s.model_dump(mode="json")
                for s in runtime.store.sessions(root_id=sid)
                if s.parent_id
            ],
            "compaction_events": [e["id"] for e in events if e["type"] == "context_compaction"],
        }
        if self.previous_audit:
            previous = self.previous_audit
            value["previous_history_retained"] = (
                digest([e["id"] for e in events[: previous["event_count"]]])
                == previous["event_prefix_hash"]
            )
        return value

    async def settle(self):
        runtime, sid = self.runtime, self.root_id
        while True:
            root = runtime.store.session(sid)
            if root.outcome != Outcome.ACTIVE or root.paused:
                raise HarnessError(
                    "environment",
                    "root_stopped",
                    f"Root stopped: {root.outcome}: {root.last_error}",
                )
            sessions = runtime.store.sessions(root_id=sid)
            busy = runtime.tasks or any(
                s.outcome == Outcome.ACTIVE and (s.runnable or s.pending_turn or s.wake_at)
                for s in sessions
            )
            processes = runtime.store.records.select("process_jobs")
            busy = busy or any(p["body"]["state"] == "running" for p in processes)
            if not busy:
                # The production turn checkpoint has already applied its exact plan.
                # Draining here also covers independently owned safe-boundary commands.
                await runtime.drain_refinement(sid)
                await runtime.wait_refinement_barrier(sid)
                if not runtime.tasks and not runtime.store.session(sid).runnable:
                    return
            await asyncio.sleep(0.05)

    async def run(self, instruction, round_name, policy):
        async with self.lock:
            runtime, sid = self.runtime, self.root_id
            self.last_packet = None
            before = self.audit()
            if self.previous_audit:
                for key in (
                    "runtime_id",
                    "root_session_id",
                    "workspace_device",
                    "workspace_inode",
                    "workspace",
                    "kernel_id",
                    "kernel_pid",
                    "assistant_turns_since_auto_refine",
                    "refinement_history",
                    "local_harness",
                    "global_harness",
                    "compaction_events",
                ):
                    if before[key] != self.previous_audit[key]:
                        raise RuntimeError(f"Round persistence violated: {key}")
                if not before["previous_history_retained"]:
                    raise RuntimeError("Previous conversation history was lost")
                if [s["id"] for s in before["children"]] != [
                    s["id"] for s in self.previous_audit["children"]
                ]:
                    raise RuntimeError("Retained child registry changed between rounds")
            self.cycle = AutonomousCycle(AutonomousPolicy(**policy), [f"<{round_name} verifier>"])
            start_seq = max(
                (e["seq"] for e in runtime.store.iter_events(sid, tree=True)), default=0
            )
            runtime.store.event(sid, "evaluation_round_start", {"round": round_name})
            if self.rounds == 0:
                if instruction != self.first_instruction:
                    raise ValueError(
                        "First instruction differs from the staged official instruction"
                    )
                runtime.store.update(sid, runnable=True)
                runtime._wake.set()
            else:
                runtime.interact(sid, instruction)

            async def gate(command, timeout_seconds):
                result = await self.peer.call(
                    "gate", command=command, timeout_seconds=timeout_seconds
                )
                return GateResult(**result)

            try:
                while True:
                    await self.settle()
                    responses = [
                        e
                        for e in runtime.store.iter_events(sid, kind="model_response")
                        if e["payload"].get("metadata", {}).get("purpose", "agent") == "agent"
                    ]
                    stop = responses[-1]["payload"].get("metadata", {}).get("stop_reason")
                    feedback = await self.cycle.next_message(
                        stop,
                        snapshot=lambda: git_snapshot(
                            Path(runtime.store.session(sid).workspace.path)
                        ),
                        run_gate=gate,
                    )
                    runtime.store.event(
                        sid,
                        "autonomous_gate_boundary",
                        {
                            "round": round_name,
                            "stop_reason": self.cycle.stop_reason,
                            "check": self.cycle.checks[-1] if self.cycle.checks else None,
                        },
                    )
                    if feedback is None:
                        break
                    # An ordinary user message, with no second Runtime/root/resume.
                    runtime.interact(sid, feedback)
            except BaseException as exc:
                self.cycle.stop_reason = (
                    "aborted" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                await runtime.stop(sid)
                raise
            finally:
                # Preserve real usage and applied-edit evidence even when a host
                # timeout or transport error ends the round without a reward.
                self.last_packet = {
                    "round": round_name,
                    "before": before,
                    "after": self.audit(),
                    "autonomous": asdict(self.cycle),
                    "usage": runtime.store.usage(sid, tree=True).model_dump(mode="json"),
                    "events": [
                        e
                        for e in runtime.store.iter_events(sid, tree=True, after=start_seq)
                        if e["type"] != "model_stream"
                    ],
                    "requests": runtime.store.request_history(sid),
                    "messages": runtime.store.records.select("messages"),
                    "artifacts": runtime.store.records.select("artifacts"),
                }
            self.rounds += 1
            self.previous_audit = self.last_packet["after"]
            return self.last_packet

    async def dispatch(self, method, args):
        if method == "setup":
            return await self.setup(**args)
        if self.runtime is None:
            raise RuntimeError("Worker has not been set up")
        if method == "run":
            return await self.run(**args)
        if method == "audit":
            return self.audit()
        if method == "last_round":
            async with self.lock:
                return self.last_packet
        if method == "shutdown":
            await self.runtime.shutdown()
            return {"closed": True}
        raise ValueError(f"Unknown worker operation: {method}")


async def main():
    reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        sys.stdin.buffer,
    )

    async def send(data):
        def write():
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

        await asyncio.to_thread(write)

    worker = PersistentWorker()
    worker.peer = Peer(reader, send, worker.dispatch)
    try:
        await worker.peer.pump
    finally:
        if worker.runtime:
            await worker.runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
