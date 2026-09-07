"""Cancellable execution with retained streams. Local execution is trusted-host, not a sandbox."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Protocol

from .models import new_id


def environment(config):
    denied = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    names = [n for n in config.environment_allowlist if not any(s in n.upper() for s in denied)]
    result = {k: os.environ[k] for k in names if k in os.environ}
    for key, value in config.environment.items():
        if key not in names:
            raise PermissionError(f"Environment override is not allowlisted: {key}")
        result[key] = value
    return result


class Executor(Protocol):
    async def run(self, context, command, *, cwd=".", timeout_seconds=60): ...


class LocalExecutor:
    def invocation(self, context, command, cwd):
        bootstrap = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r}); from threadweave.process_worker import main; main()"
        return [sys.executable, "-c", bootstrap, *command], cwd, None

    async def cleanup(self, name):
        return True

    async def run(self, context, command, *, cwd=".", timeout_seconds=60):
        config = context.runtime.store.config(context.session_id).execution
        if config.backend == "local" and config.read_only:
            raise PermissionError("Read-only process execution requires a container backend")
        if not command or not all(isinstance(a, str) and "\0" not in a for a in command):
            raise ValueError("command must be a nonempty argv array without NULs")
        if config.command_allowlist is not None and command[0] not in config.command_allowlist:
            raise PermissionError(f"Executable is not allowed: {command[0]}")
        directory = context.path(cwd)
        argv, directory, name = self.invocation(context, command, directory)
        if name:
            context.runtime.store.event(
                context.session_id,
                "container_lease_started",
                {"name": name, "engine": config.engine},
                parent=context.source_event,
            )
        start, proc, timed_out = time.monotonic(), None, False
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            streams = {"stdout": out, "stderr": err}

            async def consume(reader, kind):
                announced = 0
                while chunk := await reader.read(32768):
                    streams[kind].write(chunk)
                    # Stream a bounded amount to durable events; all bytes remain artifacts.
                    if announced < config.output_chars:
                        text = chunk[: config.output_chars - announced].decode(errors="replace")
                        context.runtime.store.event(
                            context.session_id,
                            "execution_output",
                            {"stream": kind, "text": text},
                            parent=context.source_event,
                        )
                        announced += len(chunk)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=directory,
                    env=environment(config),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                readers = [
                    asyncio.create_task(consume(proc.stdout, "stdout")),
                    asyncio.create_task(consume(proc.stderr, "stderr")),
                ]
                try:
                    async with asyncio.timeout(timeout_seconds):
                        await proc.wait()
                        await asyncio.gather(*readers)
                except TimeoutError:
                    timed_out = True
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), 0.3)
                    except TimeoutError:
                        pass
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                    try:
                        await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), 1)
                    except TimeoutError:
                        context.runtime.store.event(
                            context.session_id,
                            "process_output_incomplete",
                            {"reason": "A detached writer retained the output pipe"},
                            parent=context.source_event,
                        )
            finally:
                try:
                    cleaned = await self.cleanup(name)
                except OSError:
                    cleaned = False
                if name:
                    context.runtime.store.event(
                        context.session_id,
                        "container_lease_closed" if cleaned else "container_cleanup_failed",
                        {"name": name},
                        parent=context.source_event,
                    )
                    if not cleaned:
                        context.runtime.store.update(
                            context.session_id, paused=True, runnable=False
                        )
                output = {}
                for kind, stream in streams.items():
                    stream.seek(0)
                    output[kind] = stream.read(config.output_chars).decode(errors="replace")
                    stream.seek(0)
                    output[kind + "_artifact"] = context.runtime.artifacts.put_stream(
                        context.session_id, stream, source_event=context.source_event
                    )
                context.runtime.store.event(
                    context.session_id, "execution_capture", output, parent=context.source_event
                )
        result = {
            "command": command,
            "cwd": cwd,
            "exit_code": proc.returncode,
            "returncode": proc.returncode,
            "duration": time.monotonic() - start,
            "timed_out": timed_out,
            "state": "timed_out"
            if timed_out
            else "completed"
            if proc.returncode == 0
            else "failed",
            "passed": proc.returncode == 0 and not timed_out,
            "backend": config.backend,
            "network_policy_enforced": config.backend == "container",
            **output,
        }
        context.runtime.store.event(
            context.session_id, "execution_result", result, parent=context.source_event
        )
        return result


class ContainerExecutor(LocalExecutor):
    def __init__(self, config):
        self.config = config

    def invocation(self, context, command, cwd):
        config = self.config
        if not shutil.which(config.engine):
            raise ValueError(
                f"{config.engine} is not installed; configure/install the requested container engine"
            )
        root = Path(context.session.workspace.path).resolve()
        name = "tw-" + new_id()
        mount = f"{root}:/workspace" + (":ro" if config.read_only else ":rw")
        argv = [
            config.engine,
            "run",
            "--rm",
            "--name",
            name,
            "--init",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            "--memory",
            config.memory,
            "--cpus",
            str(config.cpus),
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            mount,
            "-w",
            "/workspace/" + cwd.relative_to(root).as_posix(),
        ]
        if not config.network:
            argv += ["--network", "none"]
        for key, value in environment(config).items():
            if key != "PATH":
                argv += ["--env", f"{key}={value}"]
        return [*argv, config.image, *command], root, name

    async def cleanup(self, name):
        if name:
            proc = await asyncio.create_subprocess_exec(
                self.config.engine,
                "rm",
                "--force",
                name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(10):
                    _, error = await proc.communicate()
                absent = any(
                    text in error.decode(errors="replace").lower()
                    for text in ("no such container", "no container with name")
                )
                return proc.returncode == 0 or absent
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return False
        return True


async def recover_containers(runtime):
    import json

    rows = runtime.store.db.execute(
        "SELECT session_id,type,payload FROM events WHERE type IN ('container_lease_started','container_lease_closed') ORDER BY seq"
    ).fetchall()
    pending = {}
    for row in rows:
        body = json.loads(row["payload"])
        if row["type"] == "container_lease_started":
            pending[body["name"]] = (row["session_id"], body)
        else:
            pending.pop(body["name"], None)
    for name, (sid, _body) in pending.items():
        config = runtime.store.config(sid).execution
        try:
            cleaned = await ContainerExecutor(config).cleanup(name)
        except OSError:
            cleaned = False
        runtime.store.event(
            sid,
            "container_lease_closed" if cleaned else "container_cleanup_failed",
            {"name": name, "recovery": True},
        )
        if not cleaned:
            runtime.store.update(sid, paused=True, runnable=False)


def executor(config):
    return LocalExecutor() if config.backend == "local" else ContainerExecutor(config)
