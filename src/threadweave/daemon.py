from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import signal
import tempfile
from pathlib import Path

from .artifacts import atomic_write
from .models import RunConfig, StateEdit
from .runtime import Runtime

MAX_PACKET = 8 * 1024 * 1024


def socket_path(directory: str | Path) -> Path:
    # POSIX sockets have short path limits, especially on macOS.
    digest = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:24]
    return Path(tempfile.gettempdir()) / f"tw-{os.getuid()}-{digest}.sock"


async def request(directory, method, **arguments):
    reader, writer = await asyncio.open_unix_connection(
        str(socket_path(directory)), limit=MAX_PACKET
    )
    try:
        writer.write((json.dumps({"method": method, "arguments": arguments}) + "\n").encode())
        await writer.drain()
        async with asyncio.timeout(60):
            line = await reader.readline()
        if not line:
            raise ConnectionError("Daemon disconnected before responding")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]
    finally:
        writer.close()
        await writer.wait_closed()


def bounded_session(session):
    result = session.model_dump(mode="json")
    result.pop("context")
    pending = result.pop("pending_turn")
    result["pending_turn"] = bool(pending)
    result["instruction"] = result["instruction"][:2000]
    result["summary"] = result["summary"][:3000]
    if result["result"]:
        result["result"] = result["result"][:3000]
    return result


class Daemon:
    def __init__(self, directory: Path, *, concurrency=8, idle_seconds=60):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.directory / "daemon.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise RuntimeError("A daemon already owns this data directory") from exc
        self.runtime = Runtime(directory, concurrency=concurrency, idle_seconds=idle_seconds)
        self.stop_event = asyncio.Event()
        self.server = None
        self.clients: set[asyncio.Task] = set()

    async def dispatch(self, method, args):
        runtime, store = self.runtime, self.runtime.store
        if method == "ping":
            return {"pid": os.getpid(), "data": str(self.directory), "schema_version": 1}
        if method == "create":
            config = RunConfig.model_validate(args.pop("config", {}))
            return bounded_session(runtime.create(config=config, **args))
        if method == "list":
            return [bounded_session(s) for s in store.sessions(roots_only=True)]
        if method == "tree":
            sid = args["session_id"]
            root = store.session(sid).root_id
            return [bounded_session(s) for s in store.sessions(root_id=root)]
        if method == "status":
            sid = args["session_id"]
            return {
                "session": bounded_session(store.session(sid)),
                "goal": store.goal(sid),
                "usage": store.usage(sid).model_dump(),
                "tree_usage": store.usage(sid, tree=True).model_dump(),
                "elapsed_seconds": runtime._elapsed(store.session(sid).root_id),
            }
        if method == "config":
            return store.config(args["session_id"]).model_dump(mode="json")
        if method == "history":
            sid = args.pop("session_id")
            rows = store.events(sid, **args)
            for row in rows:
                raw = json.dumps(row["payload"], ensure_ascii=False)
                if len(raw) > 4000:
                    row["payload"] = {
                        "preview": raw[:4000],
                        "truncated": True,
                        "retrieve_with": "history_read(event_id=...)",
                    }
            return rows
        if method == "input":
            return {"message_id": runtime.message(None, args["session_id"], args["body"])}
        if method == "resume":
            runtime.resume(args["session_id"])
            return {"resumed": args["session_id"]}
        if method == "pause":
            await runtime.pause(args["session_id"])
            return {"paused": args["session_id"]}
        if method == "stop":
            await runtime.stop(args["session_id"], tree=args.get("tree", True))
            return {"stopped": args["session_id"]}
        if method == "fork":
            return bounded_session(await runtime.fork(args["session_id"], name=args.get("name")))
        if method == "schedule":
            return {"schedule_id": runtime.schedule(**args)}
        if method == "schedules":
            return [
                dict(row)
                for row in store.db.execute(
                    "SELECT * FROM schedules WHERE session_id=?", (args["session_id"],)
                )
            ]
        if method == "unschedule":
            with store.transaction():
                row = store.db.execute(
                    "SELECT * FROM schedules WHERE id=?", (args["schedule_id"],)
                ).fetchone()
                if not row:
                    raise KeyError("Unknown schedule")
                store.db.execute(
                    "UPDATE schedules SET enabled=0 WHERE id=?", (args["schedule_id"],)
                )
                store.event(row["session_id"], "schedule_disabled", args)
            return {"disabled": args["schedule_id"]}
        if method == "states":
            return store.states(args["session_id"], include_deleted=True)
        if method == "refine":
            return {
                "refinement_id": store.queue_refinement(
                    args["session_id"], StateEdit.model_validate(args["edit"])
                )
            }
        if method == "compact":
            sid = args["session_id"]
            if sid in runtime.tasks:
                raise ValueError("Pause the session before manual compaction")
            return {"event_id": runtime.context.compact(sid)}
        if method == "artifact":
            return runtime.artifacts.read(
                args["session_id"],
                args["artifact_id"],
                offset=args.get("offset", 0),
                limit=args.get("limit", 16000),
            )
        if method == "shutdown":
            self.stop_event.set()
            return {"stopping": True}
        raise ValueError(f"Unknown method: {method}")

    async def client(self, reader, writer):
        task = asyncio.current_task()
        self.clients.add(task)
        try:
            async with asyncio.timeout(30):
                line = await reader.readline()
            packet = json.loads(line)
            result = await self.dispatch(packet["method"], packet.get("arguments", {}))
            response = {"result": result}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Session work belongs to the runtime, not this socket.
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.clients.discard(task)

    async def serve(self):
        path = socket_path(self.directory)
        # The exclusive data-directory lock proves a prior socket is stale.
        path.unlink(missing_ok=True)
        try:
            await self.runtime.start()
            self.server = await asyncio.start_unix_server(
                self.client, path=str(path), limit=MAX_PACKET
            )
            path.chmod(0o600)
            atomic_write(
                self.directory / "daemon.json",
                json.dumps(
                    {"pid": os.getpid(), "socket": str(path), "data": str(self.directory)}
                ).encode(),
            )
            loop = asyncio.get_running_loop()
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signum, self.stop_event.set)
            await self.stop_event.wait()
        finally:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            if self.clients:
                await asyncio.gather(*list(self.clients), return_exceptions=True)
            await self.runtime.shutdown()
            path.unlink(missing_ok=True)
            self.lock.close()


def main():
    parser = argparse.ArgumentParser(description="Threadweave session daemon")
    parser.add_argument("--data", type=Path, default=Path.cwd() / ".threadweave")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--idle-seconds", type=float, default=60)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(
            Daemon(args.data, concurrency=args.concurrency, idle_seconds=args.idle_seconds).serve()
        )
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
