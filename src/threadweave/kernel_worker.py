"""Private subprocess protocol. Run only via Kernel, never in the daemon process.

Python is trusted local code, not an OS sandbox. Checkpoints use explicit codecs,
 explicit codecs plus trusted procedure blobs; never replay execution history. Registered recovery recipes
are executed during reconstruction.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import fcntl
import importlib
import inspect
import json
import os
import queue
import reprlib
import signal
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path

from .artifacts import atomic_write


def pack(value, seen=None):
    from .kernel_api import Record, snapshot_handle

    handle = snapshot_handle(value)
    if handle:
        return handle
    if isinstance(value, Record):
        return ["record", pack(dict(value), seen)]
    seen = set() if seen is None else seen
    if value is None or type(value) in (str, int, float, bool):
        return ["scalar", value]
    if id(value) in seen:
        raise ValueError("cyclic value needs an explicit recovery recipe")
    seen = seen | {id(value)}
    if type(value) is dict:
        return ["dict", [[pack(k, seen), pack(v, seen)] for k, v in value.items()]]
    for typ in (list, tuple, set, frozenset):
        if type(value) is typ:
            return [typ.__name__, [pack(v, seen) for v in value]]
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode()]
    if isinstance(value, Path):
        return ["path", str(value)]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    raise TypeError(f"{type(value).__module__}.{type(value).__name__} requires a recovery recipe")


def unpack(record, host=None):
    kind, value = record
    if kind == "agent_handle":
        from .kernel_api import AgentHandle

        return AgentHandle(**value)
    if kind == "bash_handle":
        from .kernel_api import BashHandle

        if host is None:
            raise ValueError("Process handle requires the owning session bridge")
        return BashHandle(host, value)
    if kind == "scalar":
        return value
    if kind == "record":
        from .kernel_api import Record

        return Record(unpack(value, host))
    if kind == "dict":
        return {unpack(k, host): unpack(v, host) for k, v in value}
    if kind in ("list", "tuple", "set", "frozenset"):
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](
            unpack(v, host) for v in value
        )
    if kind == "bytes":
        return base64.b64decode(value)
    if kind == "path":
        return Path(value)
    if kind == "module":
        return importlib.import_module(value)
    raise ValueError(f"Unknown checkpoint codec: {kind}")


class Bridge:
    def __init__(self, emit, receive):
        self.emit, self.receive = emit, receive

    def call(self, tool_name: str, /, **arguments):
        self.emit({"type": "tool_request", "name": tool_name, "arguments": arguments})
        answer = self.receive()
        if answer.get("error"):
            raise RuntimeError(f"Tool {tool_name}: {answer['error']}")
        return answer["result"]

    async def acall(self, tool_name: str, /, **arguments):
        return await asyncio.to_thread(self.call, tool_name, **arguments)

    def catalog(self):
        return self.call("host_request", operation="catalog", payload={})


class Worker:
    def __init__(self, directory: Path, workspace: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        # A replacement worker must not race an orphan's final checkpoint write.
        self.ownership = (directory / "worker.lock").open("a+")
        fcntl.flock(self.ownership, fcntl.LOCK_EX)
        self.checkpoint = directory / "checkpoint.json"
        self.protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        # Protocol has its own descriptor. Imports and asynchronous work may print
        # outside a cell; retain those bytes without corrupting the RPC channel.
        self.console = (directory / "background.log").open("a")
        (directory / "background.log").chmod(0o600)
        os.dup2(self.console.fileno(), 1)
        os.dup2(self.console.fileno(), 2)
        self.input = sys.stdin
        self.recipes = {}
        self.pending_replies, self.requests = {}, queue.Queue()
        self.write_lock = threading.Lock()
        self.execution_lock = threading.RLock()
        self.active_id = None
        self.interrupt_id = None
        self.phase = "idle"
        self.loop = None
        self.cell_task = None
        self.queued = set()
        self.cancelled = set()
        signal.signal(signal.SIGUSR1, self.on_interrupt)
        from IPython.core.interactiveshell import InteractiveShell

        self.shell = InteractiveShell()
        self.values = {
            "__name__": "__session__",
            "tools": self.make_bridge(),
            "workspace": workspace,
            "remember_recipe": self.remember_recipe,
            "forget": self.forget,
        }
        metadata_path = directory / "bootstrap.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        metadata["console_log"] = str(directory / "background.log")
        os.chdir(workspace)
        sys.path.insert(0, str(workspace))
        from .kernel_api import bootstrap

        self.host = bootstrap(self.values["tools"], self.values, metadata)
        self.values["get_ipython"] = lambda: self.shell
        self.shell.user_ns = self.values
        self.protected = set(self.values)
        self.receipt = None
        from .snapshots import SnapshotBlobs

        self.blobs = SnapshotBlobs(directory)
        from .process_family import ProcessFamily, install_shutdown

        self.family = ProcessFamily(directory)
        install_shutdown(self.family)

    def emit(self, value):
        with self.write_lock:
            self.protocol.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            self.protocol.flush()

    def make_bridge(self):
        worker = self

        class MultiplexBridge(Bridge):
            def call(self, tool_name, /, **arguments):
                identifier, replies = uuid.uuid4().hex, queue.Queue()
                worker.pending_replies[identifier] = replies
                try:
                    worker.emit(
                        {
                            "type": "tool_request",
                            "id": identifier,
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    )
                    answer = replies.get()
                    if answer.get("error"):
                        raise RuntimeError(answer["error"])
                    return answer["result"]
                finally:
                    worker.pending_replies.pop(identifier, None)

        return MultiplexBridge(self.emit, self.receive)

    def read_packets(self):
        try:
            while True:
                packet = self.receive()
                if packet.get("type") == "interrupt":
                    with self.execution_lock:
                        active = self.active_id == packet["id"] and self.phase == "executing"
                        accepted = active or packet["id"] in self.queued
                        if accepted and not active:
                            self.cancelled.add(packet["id"])
                        if active:
                            self.interrupt_id = packet["id"]
                            for replies in list(self.pending_replies.values()):
                                replies.put(
                                    {
                                        "error": "Cell interrupted; inspect side effects before retrying"
                                    }
                                )
                            os.kill(os.getpid(), signal.SIGUSR1)
                    self.emit({"type": "interrupt_ack", "id": packet["id"], "accepted": accepted})
                elif packet.get("type") in {"execute", "shutdown"}:
                    if packet["type"] == "execute":
                        self.queued.add(packet["id"])
                    self.requests.put(packet)
                elif packet.get("id") in self.pending_replies:
                    self.pending_replies[packet["id"]].put(packet)
        except (EOFError, ValueError):
            for replies in list(self.pending_replies.values()):
                replies.put({"error": "Runtime disconnected"})
            self.requests.put({"type": "shutdown"})

    def on_interrupt(self, *_):
        # Signal delivery can lag the RPC. Re-check identity AND phase on the
        # main thread; never interrupt a snapshot or a subsequent execution.
        if self.phase != "executing" or self.active_id != self.interrupt_id:
            return
        self.interrupt_id = None
        if self.cell_task and self.cell_task._fut_waiter is not None:
            self.loop.call_soon_threadsafe(self.cell_task.cancel, "Cell interrupted")
        else:
            raise KeyboardInterrupt("Cell interrupted")

    def receive(self):
        line = self.input.readline()
        if not line:
            raise EOFError("Runtime disconnected")
        return json.loads(line)

    def remember_recipe(self, name: str, code: str):
        """Opt in to re-running this reconstruction code on restart (not the original action)."""
        if not name.isidentifier() or name in self.protected:
            raise ValueError("Recipe needs a non-reserved Python variable name")
        compile(code, "<recovery-recipe>", "exec")
        self.recipes[name] = code

    def forget(self, *names):
        for name in names:
            if name in self.protected:
                raise ValueError(f"Reserved name: {name}")
            self.values.pop(name, None)
            self.recipes.pop(name, None)

    def restore(self):
        restored, missing, reconstructed = [], {}, []
        if self.checkpoint.exists():
            data = json.loads(self.checkpoint.read_text())
            if data.get("version") != 1:
                raise ValueError("Unsupported checkpoint version")
            self.recipes = data.get("recipes", {})
            self.receipt = data.get("receipt")
            missing.update(data.get("missing", {}))
            for name, value in data["values"].items():
                try:
                    self.values[name] = self.blobs.decode(value, lambda v: unpack(v, self.host))
                    restored.append(name)
                except Exception as exc:
                    missing[name] = str(exc)
            # Explicit recipes run after serializable artifacts have been restored.
            # Redirect their output too, so it cannot corrupt the protocol.
            with (self.directory / "recovery.log").open("a") as output:
                import contextlib

                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    for name, code in self.recipes.items():
                        try:
                            exec(compile(code, "<recovery-recipe>", "exec"), self.values)
                            if name not in self.values:
                                raise ValueError(f"Recipe did not recreate {name}")
                            reconstructed.append(name)
                            missing.pop(name, None)
                        except BaseException as exc:
                            missing[name] = f"Recipe failed: {exc}"
        return {"restored": restored, "reconstructed": reconstructed, "missing": missing}

    def snapshot(self, receipt):
        from .snapshots import deadline

        started = self.blobs.begin()
        values, missing, used = {}, {}, 0
        for name, value in list(self.values.items()):
            if name in self.protected or name.startswith("__") or name in self.recipes:
                continue
            try:
                if time.monotonic() - started > 5:
                    raise ValueError("Namespace snapshot exceeded 5 seconds; variable not saved")
                with deadline(min(1, 5 - (time.monotonic() - started))):
                    encoded, size = self.blobs.encode(name, value, pack)
                if size > 16 * 1024 * 1024 or used + size > 64 * 1024 * 1024:
                    raise ValueError(
                        "Checkpoint size cap reached; persist an artifact and a recipe"
                    )
                values[name] = encoded
                used += size
            except Exception as exc:
                missing[name] = str(exc)
        self.blobs.cache = {k: v for k, v in self.blobs.cache.items() if k in values}
        self.blobs.shadows = {k: v for k, v in self.blobs.shadows.items() if k in values}
        self.blobs.array_shadows = {
            k: v for k, v in self.blobs.array_shadows.items() if k in values
        }
        receipt["result"]["snapshot_metrics"] = {
            **self.blobs.stats,
            "seconds": time.monotonic() - started,
            "saved_variables": len(values),
            "missing_variables": len(missing),
        }
        atomic_write(
            self.checkpoint,
            json.dumps(
                {
                    "version": 1,
                    "values": values,
                    "missing": missing,
                    "recipes": self.recipes,
                    "receipt": receipt,
                },
                allow_nan=False,
            ).encode(),
        )
        return missing

    async def evaluate(self, code):
        code = self.shell.transform_cell(code)
        tree = ast.parse(code, mode="exec")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        compiled = compile(tree, "<session>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        outcome = eval(compiled, self.values)
        if inspect.isawaitable(outcome):
            await outcome
        if last:
            expression = ast.Expression(last.value)
            outcome = eval(
                compile(expression, "<session>", "eval", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
                self.values,
            )
            if inspect.isawaitable(outcome):
                outcome = await outcome
            self.values["_"] = outcome
            return outcome

    async def execute(self, request):
        execution_id = request["id"]
        if self.receipt and self.receipt["id"] == execution_id:
            return self.receipt["result"]
        stdout = self.directory / f"{execution_id}.stdout"
        stderr = self.directory / f"{execution_id}.stderr"
        saved_out, saved_err = os.dup(1), os.dup(2)
        result = {
            "stdout_path": str(stdout),
            "stderr_path": str(stderr),
            "error": None,
            "value": None,
        }
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            with stdout.open("w") as out, stderr.open("w") as err:
                os.dup2(out.fileno(), 1)
                os.dup2(err.fileno(), 2)
                try:
                    with self.execution_lock:
                        self.queued.discard(execution_id)
                        self.active_id, self.phase = execution_id, "executing"
                        self.cell_task = asyncio.current_task()
                        self.family.execution_id = execution_id
                        if execution_id in self.cancelled:
                            self.cancelled.discard(execution_id)
                            raise KeyboardInterrupt("Cell interrupted before execution")
                    value = await self.evaluate(request["code"])
                    result["value"] = reprlib.repr(value)
                except BaseException as exc:
                    result["error"] = {
                        "category": "environment",
                        "code": type(exc).__name__,
                        "message": str(exc)[:4000],
                        "traceback": traceback.format_exc()[-8000:],
                    }
                finally:
                    with self.execution_lock:
                        self.phase, self.active_id, self.interrupt_id = "snapshot", None, None
                    self.emit({"type": "phase", "id": execution_id, "phase": "snapshot"})
                    self.family.execution_id = None
                    sys.stdout.flush()
                    sys.stderr.flush()
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
        for key, path in (("stdout", stdout), ("stderr", stderr)):
            with path.open() as stream:
                result[key] = stream.read(65536)
            result[f"{key}_bytes"] = path.stat().st_size
        result["variables"] = sorted(n for n in self.values if not n.startswith("__"))[:100]
        result["process_effects"] = self.family.observation()
        self.receipt = {"id": execution_id, "result": result}
        result["not_checkpointed"] = self.snapshot(self.receipt)
        self.phase = "idle"
        return result

    async def run(self):
        self.loop = asyncio.get_running_loop()
        threading.Thread(target=self.read_packets, daemon=True).start()
        try:
            saved_out, saved_err = os.dup(1), os.dup(2)
            try:
                with (self.directory / "recovery.log").open("a") as output:
                    os.dup2(output.fileno(), 1)
                    os.dup2(output.fileno(), 2)
                    restored = self.restore()
                    sys.stdout.flush()
                    sys.stderr.flush()
            finally:
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
            self.emit({"type": "ready", "recovery": restored})
        except BaseException as exc:
            self.emit({"type": "fatal", "error": str(exc)})
            return
        while True:
            request = await asyncio.to_thread(self.requests.get)
            if request["type"] == "shutdown":
                self.family.close()
                return
            if request["type"] == "execute":
                self.emit(
                    {"type": "result", "id": request["id"], "result": await self.execute(request)}
                )


def main():
    owner = os.getppid()

    def watch_owner():
        while True:
            time.sleep(0.25)
            if os.getppid() != owner:
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_owner, daemon=True).start()
    asyncio.run(Worker(Path(sys.argv[1]), Path(sys.argv[2])).run())


if __name__ == "__main__":
    main()
