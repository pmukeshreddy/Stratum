"""Daemon-owned shell handles, retained streams, timeout and process-group cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections import Counter

from .execution import environment, executor
from .models import new_id
from .storage import encode


class BackgroundProcesses:
    def __init__(self, runtime):
        self.runtime, self.tasks, self.processes = runtime, {}, {}
        self.waiters = Counter()

    def save(self, id, sid, body):
        self.runtime.store.records.insert(
            "process_jobs", {"id": id, "session_id": sid, "body": body}, on_conflict="replace"
        )

    def status(self, context, id):
        row = self.runtime.store.records.first("process_jobs", id=id)
        if not row or row["session_id"] != context.session_id:
            raise PermissionError("Process handle is not owned by this session")
        result = row["body"]
        result.pop("completion_message_id", None)
        if result["running"] and id not in self.tasks:
            # An orphan worker kills its group after daemon loss. Never replay a command.
            result.update(
                running=False,
                interrupted=True,
                exit_code=None,
                passed=False,
                state="lost",
                recovery_note="Daemon lost ownership; no command replay or unsafe PID-based adoption",
            )
            with self.runtime.store.transaction():
                self.save(id, row["session_id"], result)
                self.runtime.message(
                    context.session_id,
                    context.session_id,
                    "Background process lost during restart: "
                    + encode({"id": id, "state": "lost", "recovery_note": result["recovery_note"]}),
                )
        directory = self.runtime.store.directory / "processes" / id
        cap = self.runtime.store.config(context.session_id).execution.output_chars
        for stream in ("stdout", "stderr", "output"):
            path = directory / stream
            if path.is_file():
                with path.open("rb") as f:
                    size = path.stat().st_size
                    if size > cap:
                        head = f.read(cap // 2)
                        f.seek(max(cap // 2, size - cap // 2))
                        data = head + b"\n[omitted; full stream retained]\n" + f.read(cap // 2)
                    else:
                        data = f.read(cap)
                result[stream] = data.decode(errors="replace")
        return result

    async def call(self, context, operation, p):
        if operation == "start":
            return await self.start(context, **p)
        status = self.status(context, p["id"])
        if operation == "status":
            return status
        if operation == "kill":
            task = self.tasks.get(p["id"])
            if task and not task.done():
                sig, grace = p.get("sig", signal.SIGTERM), p.get("grace", 0.5)
                if sig not in signal.valid_signals() or not 0 <= grace <= 30:
                    raise ValueError("Invalid signal or grace period (0..30 seconds)")
                proc = self.processes.get(p["id"])
                if proc:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, sig)
                try:
                    await asyncio.wait_for(asyncio.shield(task), grace)
                except TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            return self.status(context, p["id"])
        if operation == "wait":
            task = self.tasks.get(p["id"])
            self.waiters[p["id"]] += 1
            try:
                if task:
                    await asyncio.shield(task)
            except asyncio.CancelledError:
                if task and p.get("owned"):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
            finally:
                self.waiters[p["id"]] -= 1
                if not self.waiters[p["id"]]:
                    del self.waiters[p["id"]]
            # A very short process can finish before its handle is awaited.
            # Consume only its still-pending notification: this call returns
            # the same complete result. Keep the message/event in the ledger.
            row = self.runtime.store.records.first("process_jobs", id=p["id"], fields=("body",))
            notification = row["body"].get("completion_message_id")
            if notification:
                consumed = self.runtime.store.records.update(
                    "messages",
                    {"received_at": time.time()},
                    id=notification,
                    recipient_id=context.session_id,
                    received_at=None,
                )
                if consumed:
                    self.runtime.store.event(
                        context.session_id,
                        "process_result_consumed",
                        {"process_id": p["id"], "message_id": notification},
                        parent=context.source_event,
                    )
            return self.status(context, p["id"])
        raise ValueError("Unknown process operation")

    async def start(self, context, command, cwd=None, timeout=None):  # noqa: ASYNC109 - public shell API
        config = self.runtime.store.config(context.session_id)
        if not isinstance(command, str) or not command.strip() or "\0" in command:
            raise ValueError("bash requires a nonempty command string")
        if config.execution.read_only and config.execution.backend == "local":
            raise PermissionError("Read-only shell requires a container")
        if (
            config.execution.command_allowlist is not None
            and "bash" not in config.execution.command_allowlist
        ):
            raise PermissionError("Shell requires explicit bash permission in command_allowlist")
        timeout = min(
            config.limits.tool_timeout_seconds if timeout is None else timeout,
            config.limits.tool_timeout_seconds,
            self.runtime.store.config(context.session.root_id).limits.wall_seconds
            - self.runtime._elapsed(context.session.root_id),
        )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        cwd = context.path(cwd or ".")
        engine = executor(config.execution)
        argv, directory, container = engine.invocation(context, ["bash", "-c", command], cwd)
        id = new_id()
        destination = self.runtime.store.directory / "processes" / id
        destination.mkdir(parents=True, mode=0o700)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=directory,
            env=environment(config.execution),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        body = {
            "id": id,
            "pid": proc.pid,
            "running": True,
            "state": "running",
            "started_at": time.time(),
            "owner_pid": os.getpid(),
            "command": command,
            "cwd": str(cwd),
            "exit_code": None,
            "duration": 0,
            "output": "",
            "stdout": "",
            "stderr": "",
        }
        self.save(id, context.session_id, body)
        self.processes[id] = proc
        self.tasks[id] = asyncio.create_task(
            self.monitor(context, proc, body, timeout, destination, engine, container)
        )
        self.runtime.store.event(
            context.session_id, "process_started", body, parent=context.source_event
        )
        if container:
            self.runtime.store.event(
                context.session_id,
                "container_lease_started",
                {"name": container, "engine": config.execution.engine},
                parent=context.source_event,
            )
        return body

    async def monitor(self, context, proc, body, timeout, destination, engine, container):  # noqa: ASYNC109
        start, timed_out, interrupted = time.monotonic(), False, False
        with contextlib.ExitStack() as stack:
            files = {
                s: stack.enter_context((destination / s).open("wb"))
                for s in ("stdout", "stderr", "output")
            }
            for s in files:
                (destination / s).chmod(0o600)

            async def drain(stream, name):
                announced = 0
                while chunk := await stream.read(32768):
                    files[name].write(chunk)
                    files["output"].write(chunk)
                    files[name].flush()
                    files["output"].flush()
                    if announced < 2000:
                        self.runtime.store.event(
                            context.session_id,
                            "execution_output",
                            {
                                "stream": name,
                                "text": chunk[: 2000 - announced].decode(errors="replace"),
                            },
                            parent=context.source_event,
                        )
                        announced += len(chunk)

            readers = [
                asyncio.create_task(drain(proc.stdout, "stdout")),
                asyncio.create_task(drain(proc.stderr, "stderr")),
            ]
            try:
                async with asyncio.timeout(timeout):
                    await proc.wait()
                    await asyncio.gather(*readers)
            except TimeoutError:
                timed_out = True
            except asyncio.CancelledError:
                interrupted = True
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), 0.3)
                except TimeoutError:
                    pass
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
                try:
                    await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), 1)
                except TimeoutError:
                    body["output_incomplete"] = True
                await engine.cleanup(container)
                if container:
                    self.runtime.store.event(
                        context.session_id,
                        "container_lease_closed",
                        {"name": container},
                        parent=context.source_event,
                    )
        body.update(
            running=False,
            exit_code=proc.returncode,
            duration=time.monotonic() - start,
            timed_out=timed_out,
            interrupted=interrupted or proc.returncode < 0,
            passed=proc.returncode == 0 and not timed_out and not interrupted,
            state="timed_out"
            if timed_out
            else "cancelled"
            if interrupted
            else "killed"
            if proc.returncode < 0
            else "completed"
            if proc.returncode == 0
            else "failed",
        )
        for stream in ("stdout", "stderr", "output"):
            with (destination / stream).open("rb") as f:
                body[stream + "_artifact"] = self.runtime.artifacts.put_stream(
                    context.session_id, f, source_event=context.source_event
                )
        with self.runtime.store.transaction():
            self.save(body["id"], context.session_id, body)
            self.runtime.store.event(
                context.session_id, "execution_result", body, parent=context.source_event
            )
            if not self.waiters[body["id"]]:
                body["completion_message_id"] = self.runtime.message(
                    context.session_id,
                    context.session_id,
                    "Background process completed: "
                    + encode(
                        {
                            key: value[:2000] if key in {"stdout", "stderr", "output"} else value
                            for key, value in self.status(context, body["id"]).items()
                        }
                    ),
                )
                self.save(body["id"], context.session_id, body)
        self.runtime.environment.call(context.session_id, "process_completed", context)
        self.processes.pop(body["id"], None)

    def recover(self):
        from .tools import ToolContext

        for row in self.runtime.store.records.select("process_jobs"):
            if row["body"].get("running"):
                event = self.runtime.store.event(
                    row["session_id"], "process_recovery", {"id": row["id"], "state": "lost"}
                )
                self.status(
                    ToolContext(self.runtime, row["session_id"], new_id(), event), row["id"]
                )

    async def close_session(self, sid):
        ids = [
            r["id"]
            for r in self.runtime.store.records.select(
                "process_jobs", session_id=sid, fields=("id",)
            )
        ]
        tasks = [self.tasks[id] for id in ids if id in self.tasks]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
