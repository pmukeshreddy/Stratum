"""File-backed sessions, history, queues, and recoverable runtime transactions."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .file_store import FileStore
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


class Store(RequestHistory, TrajectoryHistory):
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        if (self.directory / "history.sqlite3").exists() and not (
            self.directory / ".legacy-imported.json"
        ).exists():
            from .legacy_import import import_if_needed

            import_if_needed(self.directory)
        self.records = FileStore(self.directory)
        from .harness import HarnessStore
        from .history_search import HistorySearch

        self.harness = HarnessStore(self.directory, session_history=self)
        self.search_index = HistorySearch(self)

    def refinement_history(self, sid):
        from .harness import load_refinement_history

        return load_refinement_history(self.harness.path(sid), scope="local")

    def record_harness_refinement(self, sid, result):
        from .harness import append_refinement_history

        append_refinement_history(self.harness.path(sid), result)

    def transaction(self):
        return self.records.transaction()

    def close(self):
        self.records.close()

    def config(self, session_or_config_id: str) -> RunConfig:
        row = self.records.first("configs", id=session_or_config_id, fields=("body",))
        if row is None:
            session = self.session(session_or_config_id)
            row = self.records.first("configs", id=session.config_id, fields=("body",))
        return RunConfig.model_validate(row["body"])

    def reconfigure(self, sid, config):
        """Persist a new immutable config identity at an explicit host boundary."""
        body = encode(config.model_dump(mode="json"))
        identifier = hashlib.sha256(body.encode()).hexdigest()
        with self.transaction():
            self.records.insert(
                "configs", {"id": identifier, "body": json.loads(body)}, on_conflict="ignore"
            )
            self.update(sid, config_id=identifier)
            self.event(sid, "session_configured", {"config_id": identifier})
        return identifier

    def session(self, session_id: str) -> Session:
        row = self.records.first("sessions", id=session_id, fields=("body",))
        if row is None:
            raise KeyError(f"Unknown session: {session_id}")
        return Session.model_validate(row["body"])

    def sessions(self, *, root_id: str | None = None, roots_only=False) -> list[Session]:
        match = {"root_id": root_id} if root_id else {"parent_id": None} if roots_only else {}
        return [
            Session.model_validate(row["body"]) for row in self.records.select("sessions", **match)
        ]

    def _save(self, session: Session):
        self.records.insert(
            "sessions",
            {
                "id": session.id,
                "parent_id": session.parent_id,
                "root_id": session.root_id,
                "lifecycle": session.lifecycle,
                "outcome": session.outcome,
                "runnable": session.runnable,
                "body": session.model_dump(mode="json"),
            },
            on_conflict="replace",
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
            self.records.insert(
                "configs", {"id": config_id, "body": json.loads(body)}, on_conflict="ignore"
            )
            self._save(session)
            subscription = any(
                p.name == "codex_subscription"
                for p in [
                    *config.models.values(),
                    *([config.provider] if not config.routing.default else []),
                ]
            )
            self.records.insert(
                "usage",
                {
                    "session_id": sid,
                    "body": Usage(cost=None if subscription else 0).model_dump(mode="json"),
                },
            )
            self.event(sid, "session_transition", {"from": None, "to": Lifecycle.ADMITTED})
            if mode == "goal":
                self.records.insert(
                    "goals",
                    {
                        "session_id": sid,
                        "objective": instruction,
                        "status": "active",
                        "created_at": now(),
                        "updated_at": now(),
                    },
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
        self.records.insert(
            "events",
            {
                "id": eid,
                "session_id": sid,
                "root_id": session.root_id,
                "timestamp": timestamp,
                "type": kind,
                "payload": payload,
                "parent_event_id": parent,
                "usage": usage or {},
            },
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
        return dict(row)

    def events(self, sid: str, *, limit=30, after=0, kind=None, tree=False) -> list[dict]:
        session = self.session(sid)
        match = {"root_id": session.root_id} if tree else {"session_id": sid}
        if kind:
            match["type"] = kind
        rows = self.records.select(
            "events",
            where=lambda row: row["seq"] > after,
            order=(("seq", not bool(after)),),
            limit=min(max(limit, 1), 500),
            **match,
        )
        return rows if after else list(reversed(rows))

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
        row = self.records.first("events", id=event_id)
        if row is None:
            raise KeyError(f"Unknown event: {event_id}")
        return self._event_row(row)

    def add_context(self, sid: str, event_id: str, messages: list[dict]):
        with self.transaction():
            session = self.session(sid)
            session.context.append({"event_id": event_id, "messages": messages})
            self.records.insert(
                "conversation_blocks",
                {"session_id": sid, "event_id": event_id, "messages": messages},
                on_conflict="ignore",
            )
            self.update(sid, context=session.context)

    def charge(self, sid: str, usage: Usage, *, parent=None):
        with self.transaction():
            old = self.usage(sid)
            values = {
                k: None if getattr(old, k) is None or v is None else getattr(old, k) + v
                for k, v in usage.model_dump().items()
            }
            self.records.update(
                "usage", {"body": Usage(**values).model_dump(mode="json")}, session_id=sid
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
            self.records.insert(
                "configs", {"id": identifier, "body": json.loads(body)}, on_conflict="ignore"
            )
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
            row = self.records.first("usage", session_id=sid, fields=("body",))
            if row is None:
                raise KeyError(sid)
            return Usage.model_validate(row["body"])
        sessions = self.sessions(root_id=self.session(sid).root_id)
        total = Usage().model_dump()
        for session in sessions:
            for key, value in self.usage(session.id).model_dump().items():
                total[key] = None if total[key] is None or value is None else total[key] + value
        return Usage(**total)

    def reservations(self, root_id):
        sessions = {session.id for session in self.sessions(root_id=root_id)}
        return self.records.select("reservations", where=lambda row: row["session_id"] in sessions)

    def reserved(self, root_id: str) -> tuple[int, float]:
        rows = self.reservations(root_id)
        return sum(row["input_tokens"] + row["output_tokens"] for row in rows), sum(
            row["cost"] for row in rows
        )

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
            self.records.insert(
                "messages",
                {
                    "id": mid,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "body": body,
                    "created_at": now(),
                    "received_at": None,
                    "source_event": eid,
                    "delivery": delivery,
                    "causal_request_id": causal_request_id,
                },
            )
        return mid

    def messages(self, sid: str, *, pending=False, limit=30, include_followups=True) -> list[dict]:
        self.session(sid)
        if type(limit) is not int or limit < 1:
            raise ValueError("Message limit must be a positive integer; at most 100 are returned")
        match = {"recipient_id": sid}
        if pending:
            match["received_at"] = None
        if not include_followups:
            match["delivery"] = "boundary"
        return self.records.select(
            "messages",
            order=(("created_at", not pending), ("_order", not pending)),
            limit=min(limit, 100),
            **match,
        )

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
                self.records.update(
                    "messages", {"received_at": message["received_at"]}, id=message["id"]
                )
            return messages

    def goal(self, sid: str) -> dict | None:
        row = self.records.first("goals", session_id=sid)
        if not row:
            return None
        result = dict(row)
        budget = self.records.first("goal_budgets", session_id=sid)
        if budget:
            used, reserved = self.subtree_tokens(sid)
            result.update(
                token_budget=budget["token_budget"],
                tokens_used=max(0, used - budget["starting_tokens"]),
                tokens_reserved=reserved,
            )
        return result

    def subtree_tokens(self, sid):
        sessions = self.sessions(root_id=self.session(sid).root_id)
        ids = {sid}
        while more := {s.id for s in sessions if s.parent_id in ids} - ids:
            ids.update(more)
        used = sum(self.usage(id).input_tokens + self.usage(id).output_tokens for id in ids)
        reserved = sum(
            row["input_tokens"] + row["output_tokens"]
            for row in self.records.select(
                "reservations", where=lambda row: row["session_id"] in ids
            )
        )
        return used, reserved

    def running_attempts(self, sid, *, ignore_planning=False):
        requests = {
            row["id"]
            for row in self.records.select("model_requests", session_id=sid)
            if not ignore_planning or row["purpose"] not in {"refinement", "refinement_review"}
        }
        return self.records.select(
            "model_attempts", status="running", where=lambda row: row["request_id"] in requests
        )

    def finish(self, sid: str, outcome: Outcome, result: str | None = None):
        with self.transaction():
            self.update(sid, outcome=outcome, runnable=False, result=result)
            self.records.update(
                "goals", {"status": outcome.value, "updated_at": now()}, session_id=sid
            )
            self.records.update("schedules", {"enabled": 0}, session_id=sid)
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
