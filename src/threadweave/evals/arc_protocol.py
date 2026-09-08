"""Fixed-game transport, retained continuations, and diagnostic score snapshots.

Snapshots record observed expenditure at action/continuation boundaries. They are
not an inferred token-checkpoint sequence or a reproduction of a published curve.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from ..models import Verification
from .schema import NotRun, accounting, digest, save, timestamp


def load_validation(path, limit, seed):
    if not limit or limit > 2 or seed != 0:
        raise NotRun("Protocol validation requires --limit 1 or 2 and --seed 0")
    policy = json.loads(Path(path).read_text())
    for key in ("instructions", "guidance", "continuation_prompt"):
        if not isinstance(policy.get(key), str) or not policy[key].strip():
            raise NotRun(f"Protocol validation requires {key}")
    for key in ("max_continuations", "max_turns", "max_tokens", "wall_seconds"):
        if type(policy.get(key)) is not int or policy[key] <= 0:
            raise NotRun(f"Protocol validation requires positive integer {key}")
    return policy


class FixedGameControl:
    def __init__(self, worker, directory, policy, gate, owner):
        self.worker, self.directory, self.policy = worker, Path(directory), policy
        self.gate, self.owner = gate, owner
        self.lock = asyncio.Lock()
        self.done = asyncio.Event()
        self.phase = self.continuations = self.snapshot_count = 0
        self.stop_reason = None
        self.error = None
        self.identity = {}
        self.identity_reader = lambda: self.identity
        self.usage_reader = lambda: accounting([], 0)
        self.primary_usage_reader = lambda: accounting([], 0)
        self.started = time.monotonic()
        self.handlers = set()
        self.token = secrets.token_hex(32)

    async def start(self):
        workspace = self.directory / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            Path(__file__).with_name("game_client.py"), workspace / "fixed_broker_client.py"
        )
        (workspace / "AGENTS.md").write_text(self.policy["guidance"])
        self.socket_directory = tempfile.TemporaryDirectory(prefix="arc-io-")
        socket_path = Path(self.socket_directory.name) / "game.sock"
        self.server = await asyncio.start_unix_server(self.handle, path=socket_path, limit=1 << 20)
        save(workspace / ".game-connection.json", {"socket": str(socket_path), "token": self.token})
        (workspace / ".game-connection.json").chmod(0o600)
        self.last_observation = await self.worker.call("game_query", op="observe")
        save(self.directory / "initial-observation.json", self.last_observation)
        save(self.directory / "protocol.json", self.policy)
        return {"messages": [{"role": "user", "content": self.policy["instructions"]}]}

    @staticmethod
    def single_action(query):
        actions = query.get("actions")
        return (
            isinstance(actions, list)
            and len(actions) == 1
            and isinstance(actions[0], dict)
            and set(actions[0]) == {"name", "data"}
            and isinstance(actions[0]["name"], str)
            and actions[0]["name"] != "RESET"
            and isinstance(actions[0]["data"], dict)
        )

    async def query(self, query):
        async with self.lock:
            op = query.get("op")
            allowed = {"observe": {"op"}, "status": {"op"}, "act": {"op", "actions"}}
            if op not in allowed or set(query) != allowed[op]:
                raise ValueError("Fixed-game operation refused")
            if self.phase == 0 and op != "observe":
                raise ValueError("First game operation must be observe")
            if self.phase in {1, 2} and (op != "act" or not self.single_action(query)):
                raise ValueError("Immediate single genuine non-RESET action required")
            before = await self.worker.call("game_query", op="status")
            if digest(before) != digest(self.last_observation):
                raise NotRun("Isolated game changed without an acknowledged action")
            value = await self.worker.call("game_query", **query)
            self.last_observation = value
            if self.phase == 0:
                self.phase = 1
            elif self.phase == 1:
                self.phase = 2 if value["terminal"] == "ACTIVE" else 3
            elif self.phase == 2:
                self.phase = 3
            if op == "act":
                await self._snapshot("action", query)
            if value["terminal"] != "ACTIVE":
                self.stop_reason = value["terminal"]
                await self.gate.stop_admission(self.owner)
                self.done.set()
            return value

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        try:
            query = json.loads(await reader.readline())
            if not isinstance(query, dict):
                raise ValueError("Game request must be an object")
            token = query.pop("token", "")
            if not isinstance(token, str) or not hmac.compare_digest(token, self.token):
                raise ValueError("Wrong game capability")
            try:
                reply = {"ok": True, "value": await self.query(query)}
            except NotRun as exc:
                # Worker errors after mutating requests have ambiguous effects. Never replay.
                if not str(exc).startswith(("ValueError:", "KeyError:")):
                    self.error = str(exc)
                    await self.gate.stop_admission(self.owner)
                    self.done.set()
                raise
            writer.write((json.dumps(reply) + "\n").encode())
            await writer.drain()
        except (ValueError, NotRun, OSError) as exc:
            with contextlib.suppress(OSError):
                writer.write((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            self.handlers.discard(task)

    async def _snapshot(self, kind, action=None):
        self.snapshot_count += 1
        value = await self.worker.call("snapshot_arc")
        value.update(
            kind=kind,
            sequence=self.snapshot_count,
            time=timestamp(),
            elapsed_seconds=time.monotonic() - self.started,
            usage=self.usage_reader(),
            primary_usage=self.primary_usage_reader(),
            agent_identity=dict(self.identity_reader()),
            observation_sha256=digest(self.last_observation),
            game_state=self.last_observation,
            action=action,
            semantics="Diagnostic boundary snapshot; actual cumulative usage, no token threshold",
        )
        path = self.directory / "snapshots" / f"{self.snapshot_count:06d}.json"
        save(path, value)
        save(self.directory / "latest-snapshot.json", value)
        return value

    async def boundary(self):
        async with self.lock:
            self.last_observation = await self.worker.call("game_query", op="status")
            await self._snapshot("agent_boundary")
            if self.last_observation["terminal"] != "ACTIVE":
                self.stop_reason = self.last_observation["terminal"]
                return False
            usage = self.primary_usage_reader()
            tokens = (
                usage["input_tokens"] - usage.get("cached_input_tokens", 0) + usage["output_tokens"]
            )
            for exhausted, reason in (
                (self.continuations >= self.policy["max_continuations"], "continuation_limit"),
                (usage["model_calls"] >= self.policy["max_turns"], "continuation_turn_limit"),
                (tokens >= self.policy["max_tokens"], "continuation_token_limit"),
                (
                    time.monotonic() - self.started >= self.policy["wall_seconds"],
                    "continuation_wall_limit",
                ),
            ):
                if exhausted:
                    self.stop_reason = reason
                    return False
            self.continuations += 1
            with (self.directory / "continuations.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "number": self.continuations,
                            "identity": self.identity_reader(),
                            "usage": usage,
                            "time": timestamp(),
                        }
                    )
                    + "\n"
                )
            return True

    async def prepare(self, context, config):
        return {"workspace": str(context.path(".")), "interface": "fixed-game Python client"}

    async def verify(self, context, config):
        more = await self.boundary()
        return Verification(
            passed=not more,
            details=self.policy["continuation_prompt"] if more else self.stop_reason,
        )

    async def finish(self):
        if self.error:
            raise NotRun(self.error)
        async with self.lock:
            self.last_observation = await self.worker.call("game_query", op="status")
            await self._snapshot("final")
        return {
            "game_state": self.last_observation,
            "environment_terminal": self.last_observation["terminal"],
            "continuations": self.continuations,
            "snapshot_count": self.snapshot_count,
            "protocol_stop_reason": self.stop_reason,
        }

    async def close(self):
        if hasattr(self, "server"):
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.handlers)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if hasattr(self, "socket_directory"):
            self.socket_directory.cleanup()
