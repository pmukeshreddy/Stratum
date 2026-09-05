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
    StateEdit,
    Usage,
    Workspace,
    new_id,
    now,
)

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
CREATE TABLE IF NOT EXISTS state_entries(
 id TEXT PRIMARY KEY, owner_id TEXT REFERENCES sessions(id), kind TEXT NOT NULL,
 current_version INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS state_versions(
 entry_id TEXT NOT NULL REFERENCES state_entries(id), version INTEGER NOT NULL,
 body TEXT NOT NULL, PRIMARY KEY(entry_id, version)
);
CREATE TRIGGER IF NOT EXISTS versions_immutable_update BEFORE UPDATE ON state_versions
BEGIN SELECT RAISE(ABORT, 'state versions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS versions_immutable_delete BEFORE DELETE ON state_versions
BEGIN SELECT RAISE(ABORT, 'state versions are immutable'); END;
CREATE TABLE IF NOT EXISTS refinements(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 edit TEXT NOT NULL, status TEXT NOT NULL, source_event TEXT NOT NULL, error TEXT
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


class Store:
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
        self.db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(1, ?)", (now(),))
        migrate(self.db, version, now())
        self._depth = 0

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
            selected_state=config.refinement.selected_entries,
        )
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO configs VALUES(?,?)", (config_id, body))
            self._save(session)
            self.db.execute("INSERT INTO usage VALUES(?,?)", (sid, Usage().model_dump_json()))
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
        session = self.session(sid)
        session.context.append({"event_id": event_id, "messages": messages})
        self.update(sid, context=session.context)

    def charge(self, sid: str, usage: Usage, *, parent=None):
        with self.transaction():
            old = self.usage(sid)
            values = {k: getattr(old, k) + v for k, v in usage.model_dump().items()}
            self.db.execute(
                "UPDATE usage SET body=? WHERE session_id=?",
                (Usage(**values).model_dump_json(), sid),
            )
            self.event(
                sid, "resource_usage", usage.model_dump(), parent=parent, usage=usage.model_dump()
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
                total[key] += value
        return Usage(**total)

    def reserved(self, root_id: str) -> tuple[int, float]:
        row = self.db.execute(
            "SELECT COALESCE(SUM(r.input_tokens+r.output_tokens),0),COALESCE(SUM(r.cost),0) "
            "FROM reservations r JOIN sessions s ON s.id=r.session_id WHERE s.root_id=?",
            (root_id,),
        ).fetchone()
        return row[0], row[1]

    def send(self, sender_id: str | None, recipient_id: str, body: str) -> str:
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
                    "body": body,
                },
            )
            self.db.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
                (mid, sender_id, recipient_id, body, now(), None, eid),
            )
        return mid

    def messages(self, sid: str, *, pending=False, limit=30) -> list[dict]:
        self.session(sid)
        condition = " AND received_at IS NULL" if pending else ""
        order = "ASC" if pending else "DESC"
        rows = self.db.execute(
            f"SELECT * FROM messages WHERE recipient_id=?{condition} "
            f"ORDER BY created_at {order} LIMIT ?",
            (sid, min(limit, 100)),
        )
        return [dict(r) for r in rows]

    def receive(self, sid: str, render, *, limit=30) -> list[dict]:
        with self.transaction():
            messages = self.messages(sid, pending=True, limit=limit)
            for message in messages:
                eid = self.event(
                    sid, "agent_message_received", message, parent=message["source_event"]
                )
                self.add_context(sid, eid, [{"role": "user", "content": render(message)}])
                self.db.execute(
                    "UPDATE messages SET received_at=? WHERE id=?", (now(), message["id"])
                )
            return messages

    def state(self, sid: str, entry_id: str, version: int | None = None) -> dict:
        entry = self.db.execute("SELECT * FROM state_entries WHERE id=?", (entry_id,)).fetchone()
        if not entry or entry["owner_id"] not in (None, sid):
            raise KeyError(f"State entry not accessible: {entry_id}")
        version = version or entry["current_version"]
        row = self.db.execute(
            "SELECT body FROM state_versions WHERE entry_id=? AND version=?", (entry_id, version)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown version: {version}")
        return {**dict(entry), **json.loads(row[0])}

    def states(self, sid: str, *, include_deleted=False) -> list[dict]:
        clause = "" if include_deleted else " AND deleted=0"
        rows = self.db.execute(
            "SELECT id FROM state_entries WHERE (owner_id IS NULL OR owner_id=?)" + clause, (sid,)
        ).fetchall()
        return [self.state(sid, r[0]) for r in rows]

    def queue_refinement(self, sid: str, edit: StateEdit) -> str:
        config = self.config(sid)
        if not config.refinement.enabled:
            raise PermissionError("Refinement is disabled")
        if edit.scope == "global" and not config.refinement.allow_global_writes:
            raise PermissionError("Global state writes are disabled")
        for eid in edit.source_events:
            source = self.event_by_id(eid)
            if source["root_id"] != self.session(sid).root_id:
                raise PermissionError("Refinement evidence must belong to this session tree")
        if edit.entry_id:
            existing = self.state(sid, edit.entry_id)
            if existing["owner_id"] is None and not config.refinement.allow_global_writes:
                raise PermissionError("Global state writes are disabled")
        elif edit.operation != "upsert":
            raise ValueError("Delete and rollback require entry_id")
        with self.transaction():
            rid = new_id()
            eid = self.event(
                sid,
                "refinement_requested",
                {"refinement_id": rid, "edit": edit.model_dump()},
                parent=edit.source_events[-1],
            )
            self.db.execute(
                "INSERT INTO refinements VALUES(?,?,?,?,?,NULL)",
                (rid, sid, edit.model_dump_json(), "pending", eid),
            )
        return rid

    def apply_refinements(self, sid: str) -> list[str]:
        rows = self.db.execute(
            "SELECT * FROM refinements WHERE session_id=? AND status='pending'", (sid,)
        ).fetchall()
        applied = []
        for row in rows:
            try:
                with self.transaction():
                    edit = StateEdit.model_validate_json(row["edit"])
                    entry_id = self._apply_edit(sid, edit, row["source_event"])
                    self.db.execute(
                        "UPDATE refinements SET status='applied' WHERE id=?", (row["id"],)
                    )
                    applied.append(entry_id)
            except (ValueError, KeyError, PermissionError) as exc:
                self.db.execute(
                    "UPDATE refinements SET status='rejected',error=? WHERE id=?",
                    (str(exc), row["id"]),
                )
                self.event(
                    sid,
                    "refinement_rejected",
                    {"id": row["id"], "error": str(exc)},
                    parent=row["source_event"],
                )
        return applied

    def _apply_edit(self, sid: str, edit: StateEdit, source: str) -> str:
        eid = edit.entry_id or new_id()
        current = self.state(sid, eid) if edit.entry_id else None
        owner = current["owner_id"] if current else (sid if edit.scope == "session" else None)
        config = self.config(sid)
        if owner is None and not config.refinement.allow_global_writes:
            raise PermissionError("Global state writes are disabled")
        if current and edit.expected_version and edit.expected_version != current["version"]:
            raise ValueError("State changed since the expected version; retrieve and retry")
        version = current["version"] + 1 if current else 1
        kind = current["kind"] if current else edit.kind
        title, content, deleted = edit.title, edit.content, edit.operation == "delete"
        if edit.operation == "rollback":
            if not edit.rollback_version:
                raise ValueError("rollback_version is required")
            previous = self.state(sid, eid, edit.rollback_version)
            title, content, deleted = previous["title"], previous["content"], previous["deleted"]
        if edit.operation == "delete" and current:
            title, content = current["title"], current["content"]
        if not deleted:
            if kind == "skill":
                from .refinement import validate_skill

                content = validate_skill(content, config.permissions)
            required = {
                "memory": "text",
                "prompt_note": "text",
                "skill": "code",
                "subagent_spec": "instruction",
            }[kind]
            if not isinstance(content.get(required), str) or not content[required].strip():
                raise ValueError(f"{kind} content requires a nonempty {required} string")
        body = {
            "version": version,
            "title": title,
            "content": content,
            "deleted": bool(deleted),
            "provenance": {
                "author_session": sid,
                "source_events": edit.source_events,
                "trigger": source,
                "operation": edit.operation,
                "rollback_version": edit.rollback_version,
            },
            "intended_effect": edit.intended_effect,
            "created_at": now(),
        }
        if not current:
            self.db.execute(
                "INSERT INTO state_entries VALUES(?,?,?,?,?)", (eid, owner, kind, version, deleted)
            )
        else:
            self.db.execute(
                "UPDATE state_entries SET current_version=?,deleted=? WHERE id=?",
                (version, deleted, eid),
            )
        self.db.execute("INSERT INTO state_versions VALUES(?,?,?)", (eid, version, encode(body)))
        self.event(sid, "refinement", {"entry_id": eid, "kind": kind, **body}, parent=source)
        if edit.select:
            selected = list(dict.fromkeys([*self.session(sid).selected_state, eid]))
            self.update(sid, selected_state=selected)
        return eid

    def goal(self, sid: str) -> dict | None:
        row = self.db.execute("SELECT * FROM goals WHERE session_id=?", (sid,)).fetchone()
        return dict(row) if row else None

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
