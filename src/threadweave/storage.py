"""Inspectable SQLite persistence. Mutations and their audit events share transactions."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .migrations import VERSION, migrate
from .models import (
    HarnessError,
    Lifecycle,
    Outcome,
    RunConfig,
    Session,
    Usage,
    Workspace,
    new_id,
    now,
)
from .provenance import RequestHistory
from .trajectory import TrajectoryHistory

log = logging.getLogger("threadweave.events")


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS configs(id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
 id TEXT PRIMARY KEY, parent_id TEXT REFERENCES sessions(id), root_id TEXT NOT NULL,
 lifecycle TEXT NOT NULL, outcome TEXT NOT NULL, runnable INTEGER NOT NULL, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_root ON sessions(root_id);
CREATE TABLE IF NOT EXISTS events(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
 session_id TEXT NOT NULL REFERENCES sessions(id), root_id TEXT NOT NULL,
 timestamp REAL NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL,
 parent_event_id TEXT, usage TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS events_root ON events(root_id, seq);
CREATE TRIGGER IF NOT EXISTS events_immutable_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_immutable_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TABLE IF NOT EXISTS usage(session_id TEXT PRIMARY KEY REFERENCES sessions(id), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reservations(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages(
 id TEXT PRIMARY KEY, sender_id TEXT REFERENCES sessions(id),
 recipient_id TEXT NOT NULL REFERENCES sessions(id), body TEXT NOT NULL,
 created_at REAL NOT NULL, received_at REAL, source_event TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_pending ON messages(recipient_id, received_at);
CREATE TABLE IF NOT EXISTS actions(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), name TEXT NOT NULL,
 arguments TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
 source_event TEXT NOT NULL, result_event TEXT
);
CREATE TABLE IF NOT EXISTS artifacts(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 path TEXT NOT NULL, media_type TEXT NOT NULL, size INTEGER NOT NULL,
 sha256 TEXT NOT NULL, created_at REAL NOT NULL, source_event TEXT
);
CREATE INDEX IF NOT EXISTS artifacts_content ON artifacts(session_id,sha256,media_type,size);
CREATE TABLE IF NOT EXISTS compactions(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 source_events TEXT NOT NULL, summary TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_continuations(
 event_id TEXT PRIMARY KEY REFERENCES events(id), provider TEXT NOT NULL, model TEXT NOT NULL,
 items TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goals(
 session_id TEXT PRIMARY KEY REFERENCES sessions(id), objective TEXT NOT NULL,
 status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS schedules(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 interval_seconds REAL, cron TEXT, next_at REAL NOT NULL, enabled INTEGER NOT NULL,
 instruction TEXT NOT NULL
);
"""


class Store(RequestHistory, TrajectoryHistory):
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.directory / "history.sqlite3", isolation_level=None)
        (self.directory / "history.sqlite3").chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=5000")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > VERSION:
            raise RuntimeError(
                f"Database schema {version} is newer than supported schema {VERSION}"
            )
        self.db.executescript(SCHEMA)
        if version < 11:
            from .migrations import LEGACY_HARNESS_SCHEMA

            self.db.executescript(LEGACY_HARNESS_SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(1, ?)", (now(),))
        migrate(self.db, version, now())
        self._depth = 0
        from .harness import HarnessStore
        from .harness_migration import migrate_session_refinement_history, migrate_sqlite_harness

        self.harness = HarnessStore(self.directory, session_history=self)
        migrate_sqlite_harness(self.db, self.harness)
        migrate_session_refinement_history(self)

    def refinement_history(self, sid):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT payload FROM events WHERE session_id=? AND type='harness_refinement' ORDER BY seq",
                (sid,),
            )
        ]

    def record_harness_refinement(self, sid, result):
        self.event(sid, "harness_refinement", result)

    @contextmanager
    def transaction(self):
        # No await is allowed inside this synchronous transaction.
        name = f"tx_{self._depth}"
        self.db.execute(f"SAVEPOINT {name}")
        self._depth += 1
        try:
            yield
        except BaseException:
            self.db.execute(f"ROLLBACK TO {name}")
            raise
        finally:
            self._depth -= 1
            self.db.execute(f"RELEASE {name}")

    def close(self):
        self.db.close()

    def config(self, session_or_config_id: str) -> RunConfig:
        row = self.db.execute(
            "SELECT body FROM configs WHERE id=?", (session_or_config_id,)
        ).fetchone()
        if row is None:
            session = self.session(session_or_config_id)
            row = self.db.execute(
                "SELECT body FROM configs WHERE id=?", (session.config_id,)
            ).fetchone()
        return RunConfig.model_validate_json(row[0])

    def reconfigure(self, sid, config):
        """Persist a new immutable config identity at an explicit host boundary."""
        body = encode(config.model_dump(mode="json"))
        identifier = hashlib.sha256(body.encode()).hexdigest()
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO configs VALUES(?,?)", (identifier, body))
            self.update(sid, config_id=identifier)
            self.event(sid, "session_configured", {"config_id": identifier})
        return identifier

    def session(self, session_id: str) -> Session:
        row = self.db.execute("SELECT body FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return Session.model_validate_json(row[0])

    def sessions(self, *, root_id: str | None = None, roots_only=False) -> list[Session]:
        query, args = "SELECT body FROM sessions", ()
        if root_id:
            query += " WHERE root_id=?"
            args = (root_id,)
        elif roots_only:
            query += " WHERE parent_id IS NULL"
        return [Session.model_validate_json(r[0]) for r in self.db.execute(query, args)]

    def _save(self, session: Session):
        self.db.execute(
            "INSERT INTO sessions VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "lifecycle=excluded.lifecycle,outcome=excluded.outcome,runnable=excluded.runnable,"
            "body=excluded.body",
            (
                session.id,
                session.parent_id,
                session.root_id,
                session.lifecycle,
                session.outcome,
                session.runnable,
                session.model_dump_json(),
            ),
        )

    def update(self, session_id: str, **changes) -> Session:
        session = self.session(session_id)
        data = session.model_dump()
        data.update(changes, updated_at=now())
        updated = Session.model_validate(data)
        self._save(updated)
        return updated

    def create(
        self,
        instruction: str,
        workspace: Workspace,
        config: RunConfig,
        *,
        name="root",
        parent_id=None,
        spawned_by_request_id=None,
        depth=0,
        mode="autonomous",
        branch_from=None,
        branch_event=None,
        role="agent",
    ) -> Session:
        sid = new_id()
        root_id = self.session(parent_id).root_id if parent_id else sid
        body = json.dumps(config.model_dump(mode="json"), sort_keys=True)
        config_id = hashlib.sha256(body.encode()).hexdigest()
        session = Session(
            id=sid,
            parent_id=parent_id,
            spawned_by_request_id=spawned_by_request_id,
            depth=depth,
            root_id=root_id,
            name=name,
            role=role,
            instruction=instruction,
            workspace=workspace,
            config_id=config_id,
            kernel_id=new_id(),
            mode=mode,
            created_at=now(),
            updated_at=now(),
            branch_from=branch_from,
            branch_event=branch_event,
        )
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO configs VALUES(?,?)", (config_id, body))
            self._save(session)
            subscription = any(
                p.name == "codex_subscription"
                for p in [
                    *config.models.values(),
                    *([config.provider] if not config.routing.default else []),
                ]
            )
            self.db.execute(
                "INSERT INTO usage VALUES(?,?)",
                (sid, Usage(cost=None if subscription else 0).model_dump_json()),
            )
            self.event(sid, "session_transition", {"from": None, "to": Lifecycle.ADMITTED})
            if mode == "goal":
                self.db.execute(
                    "INSERT INTO goals VALUES(?,?,?,?,?)",
                    (sid, instruction, "active", now(), now()),
                )
            if parent_id:
                self.event(parent_id, "subagent_created", {"child_id": sid, "name": name})
                self.charge(parent_id, Usage(subagent_count=1))
        return session

    def transition(self, sid: str, lifecycle: Lifecycle, **changes):
        with self.transaction():
            before = self.session(sid)
            allowed = {
                Lifecycle.ADMITTED: {Lifecycle.RUNNING, Lifecycle.IDLE, Lifecycle.INACTIVE},
                Lifecycle.RUNNING: {Lifecycle.IDLE, Lifecycle.INACTIVE},
                Lifecycle.IDLE: {Lifecycle.RUNNING, Lifecycle.INACTIVE},
                Lifecycle.INACTIVE: {Lifecycle.IDLE},
            }
            if lifecycle != before.lifecycle and lifecycle not in allowed[before.lifecycle]:
                raise ValueError(f"Invalid lifecycle transition {before.lifecycle} -> {lifecycle}")
            result = self.update(sid, lifecycle=lifecycle, **changes)
            if lifecycle != before.lifecycle:
                self.event(sid, "session_transition", {"from": before.lifecycle, "to": lifecycle})
            return result

    def event(
        self, sid: str, kind: str, payload: dict[str, Any], *, parent=None, usage=None
    ) -> str:
        session = self.session(sid)
        from .security import redact

        config = self.config(sid)
        payload = redact(
            payload, [p.api_key_env for p in [config.provider, *config.models.values()]]
        )
        eid, timestamp = new_id(), now()
        self.db.execute(
            "INSERT INTO events(id,session_id,root_id,timestamp,type,payload,"
            "parent_event_id,usage) VALUES(?,?,?,?,?,?,?,?)",
            (
                eid,
                sid,
                session.root_id,
                timestamp,
                kind,
                encode(payload),
                parent,
                encode(usage or {}),
            ),
        )
        log.info(
            encode(
                {
                    "event_id": eid,
                    "session_id": sid,
                    "root_id": session.root_id,
                    "timestamp": timestamp,
                    "type": kind,
                    "parent_event_id": parent,
                }
            )
        )
        return eid

    @staticmethod
    def _event_row(row) -> dict:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["usage"] = json.loads(result["usage"])
        return result

    def events(self, sid: str, *, limit=30, after=0, kind=None, tree=False) -> list[dict]:
        session = self.session(sid)
        column = "root_id" if tree else "session_id"
        args: list = [session.root_id if tree else sid, after]
        condition = f"{column}=? AND seq>?"
        if kind:
            condition += " AND type=?"
            args.append(kind)
        # Tail on initial inspection; forward cursor pagination for attaching.
        direction = "ASC" if after else "DESC"
        args.append(min(max(limit, 1), 500))
        rows = self.db.execute(
            f"SELECT * FROM events WHERE {condition} ORDER BY seq {direction} LIMIT ?", args
        ).fetchall()
        if not after:
            rows.reverse()
        return [self._event_row(r) for r in rows]

    def iter_events(self, sid: str, *, tree=False, kind=None, after=-1):
        """Forward pagination over the complete trajectory, unlike bounded tail inspection."""
        cursor = after
        while page := self.events(sid, tree=tree, kind=kind, after=cursor, limit=500):
            yield from page
            cursor = page[-1]["seq"]

    def history_roots(self, sid: str) -> set[str]:
        """Readable tree ancestry, including children admitted under a forked root."""
        roots, visited, pending = set(), set(), [self.session(sid)]
        while pending:
            session = pending.pop()
            if session.id in visited:
                continue
            visited.add(session.id)
            roots.add(session.root_id)
            if session.parent_id:
                pending.append(self.session(session.root_id))
            if session.branch_from:
                pending.append(self.session(session.branch_from))
        return roots

    def event_by_id(self, event_id: str) -> dict:
        row = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown event: {event_id}")
        return self._event_row(row)

    def add_context(self, sid: str, event_id: str, messages: list[dict]):
        with self.transaction():
            session = self.session(sid)
            session.context.append({"event_id": event_id, "messages": messages})
            self.db.execute(
                "INSERT OR IGNORE INTO conversation_blocks VALUES(?,?,?)",
                (sid, event_id, encode(messages)),
            )
            self.update(sid, context=session.context)

    def charge(self, sid: str, usage: Usage, *, parent=None):
        with self.transaction():
            old = self.usage(sid)
            values = {
                k: None if getattr(old, k) is None or v is None else getattr(old, k) + v
                for k, v in usage.model_dump().items()
            }
            self.db.execute(
                "UPDATE usage SET body=? WHERE session_id=?",
                (Usage(**values).model_dump_json(), sid),
            )
            self.event(
                sid, "resource_usage", usage.model_dump(), parent=parent, usage=usage.model_dump()
            )

    def pin_provider(self, sid, previous, resolved):
        """Persist account-default resolution without overwriting an old config snapshot."""
        if previous == resolved:
            return
        config = self.config(sid)
        if config.provider == previous:
            config.provider = resolved
        for alias, provider in config.models.items():
            if provider == previous:
                config.models[alias] = resolved
        body = encode(config.model_dump(mode="json"))
        identifier = hashlib.sha256(body.encode()).hexdigest()
        with self.transaction():
            old = self.session(sid).config_id
            self.db.execute("INSERT OR IGNORE INTO configs VALUES(?,?)", (identifier, body))
            self.update(sid, config_id=identifier)
            self.event(
                sid,
                "provider_config_resolved",
                {
                    "previous_config_id": old,
                    "config_id": identifier,
                    "provider": resolved.name,
                    "model": resolved.model,
                    "parameters": resolved.parameters,
                },
            )

    def usage(self, sid: str, *, tree=False) -> Usage:
        if not tree:
            row = self.db.execute("SELECT body FROM usage WHERE session_id=?", (sid,)).fetchone()
            if row is None:
                raise KeyError(sid)
            return Usage.model_validate_json(row[0])
        sessions = self.sessions(root_id=self.session(sid).root_id)
        total = Usage().model_dump()
        for session in sessions:
            for key, value in self.usage(session.id).model_dump().items():
                total[key] = None if total[key] is None or value is None else total[key] + value
        return Usage(**total)

    def reserved(self, root_id: str) -> tuple[int, float]:
        row = self.db.execute(
            "SELECT COALESCE(SUM(r.input_tokens+r.output_tokens),0),COALESCE(SUM(r.cost),0) "
            "FROM reservations r JOIN sessions s ON s.id=r.session_id WHERE s.root_id=?",
            (root_id,),
        ).fetchone()
        return row[0], row[1]

    def send(
        self,
        sender_id: str | None,
        recipient_id: str,
        body: str,
        *,
        delivery="boundary",
        causal_request_id=None,
    ) -> str:
        if delivery not in {"boundary", "idle"}:
            raise ValueError("delivery must be boundary or idle")
        if len(body.encode()) > 256000:
            raise ValueError("Messages are limited to 256 KB; use artifacts for larger values")
        self.session(recipient_id)
        with self.transaction():
            mid = new_id()
            kind = "agent_message_sent" if sender_id else "user_intervention"
            eid = self.event(
                sender_id or recipient_id,
                kind,
                {
                    "message_id": mid,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "delivery": delivery,
                    "causal_request_id": causal_request_id,
                    "body": body,
                },
            )
            self.db.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                (mid, sender_id, recipient_id, body, now(), None, eid, delivery, causal_request_id),
            )
        return mid

    def messages(self, sid: str, *, pending=False, limit=30, include_followups=True) -> list[dict]:
        self.session(sid)
        if type(limit) is not int or limit < 1:
            raise ValueError("Message limit must be a positive integer; at most 100 are returned")
        condition = " AND received_at IS NULL" if pending else ""
        if not include_followups:
            condition += " AND delivery='boundary'"
        order = "ASC" if pending else "DESC"
        rows = self.db.execute(
            f"SELECT * FROM messages WHERE recipient_id=?{condition} "
            f"ORDER BY created_at {order}, rowid {order} LIMIT ?",
            (sid, min(limit, 100)),
        )
        return [dict(r) for r in rows]

    def receive(self, sid: str, render, *, limit=30, include_followups=True) -> list[dict]:
        with self.transaction():
            messages = self.messages(
                sid, pending=True, limit=limit, include_followups=include_followups
            )
            for message in messages:
                message["received_at"] = now()
                eid = self.event(
                    sid, "agent_message_received", message, parent=message["source_event"]
                )
                self.add_context(sid, eid, [{"role": "user", "content": render(message)}])
                if message.get("causal_request_id"):
                    self.queue_request_edge(sid, message["causal_request_id"], "subagent_return")
                self.db.execute(
                    "UPDATE messages SET received_at=? WHERE id=?",
                    (message["received_at"], message["id"]),
                )
            return messages

    def goal(self, sid: str) -> dict | None:
        row = self.db.execute("SELECT * FROM goals WHERE session_id=?", (sid,)).fetchone()
        if not row:
            return None
        result = dict(row)
        budget = self.db.execute("SELECT * FROM goal_budgets WHERE session_id=?", (sid,)).fetchone()
        if budget:
            used, reserved = self.subtree_tokens(sid)
            result.update(
                token_budget=budget["token_budget"],
                tokens_used=max(0, used - budget["starting_tokens"]),
                tokens_reserved=reserved,
            )
        return result

    def subtree_tokens(self, sid):
        ids = [
            r[0]
            for r in self.db.execute(
                "WITH RECURSIVE descendants(id) AS (SELECT ? UNION ALL "
                "SELECT s.id FROM sessions s JOIN descendants d ON s.parent_id=d.id) "
                "SELECT id FROM descendants",
                (sid,),
            )
        ]
        used = sum(self.usage(id).input_tokens + self.usage(id).output_tokens for id in ids)
        reserved = sum(
            self.db.execute(
                "SELECT COALESCE(SUM(input_tokens+output_tokens),0) FROM reservations WHERE session_id=?",
                (id,),
            ).fetchone()[0]
            for id in ids
        )
        return used, reserved

    def finish(self, sid: str, outcome: Outcome, result: str | None = None):
        with self.transaction():
            self.update(sid, outcome=outcome, runnable=False, result=result)
            self.db.execute(
                "UPDATE goals SET status=?,updated_at=? WHERE session_id=?",
                (outcome.value, now(), sid),
            )
            self.db.execute("UPDATE schedules SET enabled=0 WHERE session_id=?", (sid,))
            self.event(
                sid,
                "completion" if outcome == Outcome.COMPLETED else "termination",
                {"outcome": outcome, "result": result},
            )

    def ensure_related(self, sender_id: str, recipient_id: str):
        sender, recipient = self.session(sender_id), self.session(recipient_id)
        siblings = sender.parent_id is not None and sender.parent_id == recipient.parent_id
        if (
            sender.id == recipient.id
            or sender.parent_id == recipient.id
            or recipient.parent_id == sender.id
        ):
            return
        if siblings and self.config(sender_id).allow_sibling_messages:
            return
        raise HarnessError(
            "tool", "unrelated_session", "Target must be a parent, child, or permitted sibling"
        )
