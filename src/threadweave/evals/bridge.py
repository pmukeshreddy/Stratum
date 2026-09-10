"""Bounded JSON-lines connection to an official dependency environment."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from pathlib import Path

from .schema import NotRun


class OfficialWorker:
    def __init__(self, setup, benchmark, directory):
        self.setup, self.benchmark, self.directory = setup, benchmark, Path(directory)
        self.process = None
        self.lock = asyncio.Lock()
        self.sequence = 0

    async def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "official-stderr.log").open("wb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.setup.python,
                "-u",
                str(Path(__file__).with_name("official_worker.py")),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
                start_new_session=True,
                limit=128 * 1024 * 1024,
                # Official code/test workers do not need the agent's credentials.
                env={
                    k: v
                    for k, v in os.environ.items()
                    if k
                    in {
                        "PATH",
                        "LANG",
                        "LC_ALL",
                        "TMPDIR",
                        "SYSTEMROOT",
                        "ARC_API_KEY",
                        "FACTORIO_RCON_PASSWORD",
                    }
                },
            )
        except OSError as exc:
            self.log.close()
            raise NotRun(f"Cannot start benchmark Python {self.setup.python}: {exc}") from exc
        return await self.call(
            "prepare",
            benchmark=self.benchmark,
            setup=self.setup.model_dump(mode="json"),
            output=str(self.directory.resolve()),
        )

    async def call(self, operation, *, timeout=180, **payload):  # noqa: ASYNC109 - bounded external RPC
        async with self.lock:
            self.sequence += 1
            request_id = self.sequence
            try:
                async with asyncio.timeout(timeout):
                    await self.send({"operation": operation, "request_id": request_id, **payload})
                    while True:
                        line = await self.process.stdout.readline()
                        if not line:
                            raise NotRun(
                                f"Official worker exited during {operation}; see {self.directory / 'official-stderr.log'}"
                            )
                        result = json.loads(line)
                        if result.get("request_id") != request_id:
                            # An action can settle after its caller's wall budget expires.
                            # Drain its reply without mistaking it for a later scorecard/checkpoint.
                            with (self.directory / "late-replies.jsonl").open("a") as stream:
                                stream.write(json.dumps(result) + "\n")
                            continue
                        if "error" in result:
                            raise NotRun(result["error"])
                        return result["result"]
            except TimeoutError as exc:
                await self.close()
                raise NotRun(
                    f"Official {self.benchmark} {operation} timed out after {timeout}s"
                ) from exc

    async def send(self, payload):
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        await self.process.stdin.drain()

    async def close(self):
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
        if hasattr(self, "log"):
            self.log.close()
