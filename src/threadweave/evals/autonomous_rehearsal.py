"""Agent-invoked evaluation command; all grading remains at the host boundary.

This transport exposes one fixed operation, not arbitrary host commands/files.
It uses the resident worker's existing RPC peer and ordinary shell tool results.
It neither resumes the root nor changes autonomous/RLM/refinement state.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
from pathlib import Path


class RehearsalServer:
    def __init__(self, path, callback):
        self.path, self.callback = Path(path), callback
        self.server = None
        self.active = False
        self.connections = set()

    async def start(self):
        self.server = await asyncio.start_unix_server(self.handle, str(self.path), limit=1024)
        os.chmod(self.path, 0o600)

    async def handle(self, reader, writer):
        current = asyncio.current_task()
        self.connections.add(current)
        acquired = False
        try:
            async with asyncio.timeout(5):
                request = json.loads(await reader.readline())
            if request != {"operation": "rehearse", "version": 1}:
                raise ValueError("Unsupported rehearsal request")
            if self.active:
                result = {"exit_code": 2, "error": "A public rehearsal is already running."}
            else:
                self.active = acquired = True
                result = await self.callback()
        except Exception as exc:
            result = {"exit_code": 2, "error": f"Rehearsal unavailable: {type(exc).__name__}"}
        except asyncio.CancelledError:
            writer.close()
            raise
        finally:
            if acquired:
                self.active = False
            self.connections.discard(current)
        try:
            writer.write(json.dumps(result, ensure_ascii=True).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        connections = tuple(self.connections)
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)
        self.path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Rehearse the host's exact public verifier.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--output", default="verification/public_rehearsal.json")
    args = parser.parse_args()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(args.timeout)
            client.connect(args.socket)
            client.sendall(b'{"operation":"rehearse","version":1}\n')
            data = bytearray()
            while b"\n" not in data:
                chunk = client.recv(8192)
                if not chunk or len(data) + len(chunk) > 128 * 1024:
                    raise ConnectionError("Missing or oversized rehearsal response")
                data.extend(chunk)
        result = json.loads(data)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "output"}, indent=2))
        if result.get("output"):
            print(result["output"])
        return int(result["exit_code"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"Public rehearsal unavailable: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
