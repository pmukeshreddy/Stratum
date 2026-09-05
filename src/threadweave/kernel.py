from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from .models import HarnessError


class Kernel:
    def __init__(self, directory: Path, workspace: Path, bridge):
        self.directory, self.workspace, self.bridge = directory, workspace, bridge
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.recovery = {}
        self._stderr = None

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stderr = (self.directory / "worker.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "threadweave.kernel_worker",
            str(self.directory),
            str(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr,
            start_new_session=True,
            limit=8 * 1024 * 1024,
        )
        try:
            async with asyncio.timeout(20):
                ready = await self._read()
            if ready["type"] != "ready":
                raise HarnessError(
                    "environment", "kernel_restore", ready.get("error", "Invalid handshake")
                )
            self.recovery = ready["recovery"]
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
        self.process.stdin.write((json.dumps(value, allow_nan=False) + "\n").encode())
        await self.process.stdin.drain()

    async def execute(self, execution_id: str, code: str, timeout: float):  # noqa: ASYNC109
        async with self.lock:
            await self.start()
            try:
                async with asyncio.timeout(timeout):
                    await self._send({"type": "execute", "id": execution_id, "code": code})
                    while True:
                        packet = await self._read()
                        if packet["type"] == "result":
                            return packet["result"]
                        if packet["type"] == "tool_request":
                            try:
                                result = await self.bridge(packet["name"], packet["arguments"])
                                await self._send({"result": result})
                            except Exception as exc:
                                await self._send({"error": str(exc)[:2000]})
                        else:
                            raise HarnessError(
                                "environment",
                                "kernel_protocol",
                                "Unexpected packet",
                                uncertain=True,
                            )
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

    async def close(self):
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
