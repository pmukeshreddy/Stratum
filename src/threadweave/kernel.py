from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from .models import HarnessError

SOURCE_ROOT = str(Path(__file__).resolve().parent.parent)


def fork_checkpoint(source, destination, old_workspace, new_workspace):
    """Rebind explicit Path codecs; do not replay source-bound reconstruction recipes."""
    from .artifacts import atomic_write

    checkpoint = json.loads(source.read_text())
    old, new = Path(old_workspace).resolve(), Path(new_workspace).resolve()

    def rebind(value):
        if isinstance(value, list):
            if len(value) == 2 and value[0] == "path" and isinstance(value[1], str):
                path = Path(value[1])
                if path.is_absolute() and path.is_relative_to(old):
                    return ["path", str(new / path.relative_to(old))]
            return [rebind(item) for item in value]
        if isinstance(value, dict):
            return {key: rebind(item) for key, item in value.items()}
        return value

    checkpoint["values"] = rebind(checkpoint["values"])
    # Blob values may capture source-workspace paths. Do not silently rebind opaque
    # procedure closures into a different environment.
    for name, record in list(checkpoint["values"].items()):
        if record[0] == "blob":
            if record[1]["codec"] == "cloudpickle":
                checkpoint.setdefault("missing", {})[name] = (
                    "Procedure blob requires explicit reconstruction in isolated fork"
                )
                del checkpoint["values"][name]
            else:
                blob = source.parent / "values" / record[1]["sha256"]
                data = json.loads(blob.read_bytes())
                checkpoint["values"][name] = rebind(data)
    missing = checkpoint.setdefault("missing", {})
    for name in checkpoint.get("recipes", {}):
        missing[name] = (
            "Reconstruction recipe not inherited into an isolated fork; review workspace bindings before registering it again"
        )
    checkpoint["recipes"] = {}
    checkpoint["receipt"] = None
    atomic_write(destination, json.dumps(checkpoint).encode())


class Kernel:
    def __init__(self, directory: Path, workspace: Path, bridge, *, env=None, bootstrap=None):
        self.directory, self.workspace, self.bridge = directory, workspace, bridge
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.recovery = {}
        self._stderr = None
        self.env = env
        self.bootstrap = bootstrap or {}
        self.reader = None
        self.calls, self.results = set(), {}
        self.write_lock = asyncio.Lock()
        self.active_execution = None

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        from .artifacts import atomic_write

        atomic_write(self.directory / "bootstrap.json", json.dumps(self.bootstrap).encode())
        self._stderr = (self.directory / "worker.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-c",
            f"import sys; sys.path.insert(0, {SOURCE_ROOT!r}); from threadweave.kernel_worker import main; main()",
            str(self.directory),
            str(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr,
            start_new_session=True,
            limit=8 * 1024 * 1024,
            env=self.env,
        )
        try:
            async with asyncio.timeout(20):
                ready = await self._read()
            if ready["type"] != "ready":
                raise HarnessError(
                    "environment", "kernel_restore", ready.get("error", "Invalid handshake")
                )
            self.recovery = ready["recovery"]
            self.reader = asyncio.create_task(self._pump())
        except BaseException:
            await self.close()
            raise

    async def _read(self):
        line = await self.process.stdout.readline()
        if not line:
            raise HarnessError(
                "environment",
                "kernel_exited",
                "Python worker exited; effects may be uncertain",
                uncertain=True,
            )
        try:
            return json.loads(line)
        except ValueError as exc:
            raise HarnessError(
                "environment", "kernel_protocol", "Invalid worker response", uncertain=True
            ) from exc

    async def _send(self, value):
        async with self.write_lock:
            self.process.stdin.write((json.dumps(value, allow_nan=False) + "\n").encode())
            await self.process.stdin.drain()

    async def _reply(self, packet):
        try:
            result = await self.bridge(packet["name"], packet["arguments"])
            await self._send({"id": packet["id"], "result": result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.process and self.process.returncode is None:
                await self._send({"id": packet["id"], "error": str(exc)[:2000]})

    async def _pump(self):
        try:
            while True:
                packet = await self._read()
                if packet["type"] == "result":
                    result = self.results.get(packet["id"])
                    if result and not result.done():
                        result.set_result(packet["result"])
                elif packet["type"] == "tool_request":
                    task = asyncio.create_task(self._reply(packet))
                    self.calls.add(task)
                    task.add_done_callback(self.calls.discard)
                else:
                    raise HarnessError(
                        "environment", "kernel_protocol", "Unexpected worker packet", uncertain=True
                    )
        except Exception as exc:
            for result in list(self.results.values()):
                if not result.done():
                    result.set_exception(exc)

    async def execute(self, execution_id: str, code: str, timeout: float):  # noqa: ASYNC109
        async with self.lock:
            await self.start()
            try:
                async with asyncio.timeout(timeout):
                    result = asyncio.get_running_loop().create_future()
                    self.results[execution_id] = result
                    self.active_execution = execution_id
                    await self._send({"type": "execute", "id": execution_id, "code": code})
                    return await result
            except TimeoutError as exc:
                await self.close()
                raise HarnessError(
                    "environment",
                    "python_timeout",
                    "Python execution timed out; worker was stopped",
                    uncertain=True,
                ) from exc
            except asyncio.CancelledError:
                await self.close()
                raise
            except (BrokenPipeError, ConnectionError) as exc:
                await self.close()
                raise HarnessError(
                    "environment", "kernel_disconnected", str(exc), uncertain=True
                ) from exc
            finally:
                self.results.pop(execution_id, None)
                self.active_execution = None

    async def interrupt(self, execution_id):
        """Identity-fenced hard interruption. A stale interrupt never targets a later cell.

        Worker termination preserves the last committed snapshot, not partial live
        namespace changes. It also cancels pending host RPCs before kernel replacement.
        """
        if execution_id != self.active_execution:
            return False
        await self.close()
        return True

    async def close(self):
        jobs = [
            t
            for t in [self.reader, *self.calls]
            if t is not None and t is not asyncio.current_task()
        ]
        for task in jobs:
            task.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self.reader = None
        for result in self.results.values():
            if not result.done():
                result.set_exception(
                    HarnessError(
                        "environment", "kernel_closed", "Kernel interrupted", uncertain=True
                    )
                )
        if self.process:
            # Reap the process group even if the direct child already exited.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
            self.process = None
        if self._stderr:
            self._stderr.close()
            self._stderr = None

    def receipt(self, execution_id: str):
        path = self.directory / "checkpoint.json"
        if not path.exists():
            return None
        try:
            checkpoint = json.loads(path.read_text())
            receipt = checkpoint.get("receipt")
            return receipt["result"] if receipt and receipt["id"] == execution_id else None
        except (ValueError, KeyError):
            return None
