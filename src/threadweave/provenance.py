"""Durable logical requests and transport attempts on the Store's transaction boundary."""

import hashlib
import json

from .models import Usage, now


class RequestHistory:
    def last_request(self, sid, *, purpose=None, trajectory=False):
        clause = " AND purpose=?" if purpose else ""
        args = (sid, purpose) if purpose else (sid,)
        if trajectory:
            clause += " AND purpose IN ('agent','compaction')"
        row = self.db.execute(
            "SELECT id FROM model_requests WHERE session_id=? AND status='completed'"
            + clause
            + " ORDER BY ended_at DESC,rowid DESC LIMIT 1",
            args,
        ).fetchone()
        return row[0] if row else None

    def queue_request_edge(self, sid, source, kind):
        if source:
            self.db.execute(
                "INSERT OR IGNORE INTO pending_request_edges VALUES(?,?,?)", (sid, source, kind)
            )

    def commit_compaction(self, sid, request_id):
        if self.last_request(sid) != request_id:
            return
        for edge in self.db.execute(
            "SELECT source,kind FROM pending_request_edges WHERE session_id=?", (sid,)
        ).fetchall():
            self.db.execute(
                "DELETE FROM request_edges WHERE source=? AND target=? AND kind='continuation'",
                (edge["source"], request_id),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO request_edges VALUES(?,?,?)",
                (edge["source"], request_id, edge["kind"]),
            )
        self.db.execute("DELETE FROM pending_request_edges WHERE session_id=?", (sid,))
        self.queue_request_edge(sid, request_id, "compaction")

    def begin_request(self, request, artifact):
        from .storage import encode

        sid, purpose = request.session_id, request.metadata.get("purpose", "agent")
        inbound = []
        if purpose == "agent":
            inbound = [
                dict(r)
                for r in self.db.execute(
                    "SELECT source,kind FROM pending_request_edges WHERE session_id=?", (sid,)
                )
            ]
            parent = self.session(sid).spawned_by_request_id
            if parent and not self.last_request(sid, purpose="agent"):
                inbound.append({"source": parent, "kind": "subagent_call"})
        previous = self.last_request(sid)
        if previous and not any(e["source"] == previous for e in inbound):
            inbound.append({"source": previous, "kind": "continuation"})
        # Hash the immutable provider-shaped body, including private continuation identity.
        from .request_context import projection

        body = {
            "config": request.config.model_dump(),
            **projection(request.messages, request.tools, request.config),
        }
        fingerprint = hashlib.sha256(encode(body).encode()).hexdigest()
        self.db.execute(
            "INSERT INTO model_requests VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request.request_id,
                sid,
                purpose,
                fingerprint,
                artifact,
                request.config.name,
                request.config.model,
                "running",
                now(),
                None,
                None,
                encode(inbound),
            ),
        )

    def finish_request_attempt(
        self, request_id, event_id, usage, *, response_event=None, failure=None, retry=False
    ):
        from .storage import encode

        status = "completed" if response_event else "failed"
        self.db.execute(
            "UPDATE model_attempts SET status=?,usage=?,failure=?,ended_at=? WHERE event_id=?",
            (
                status,
                encode(usage.model_dump()),
                encode(failure) if failure else None,
                now(),
                event_id,
            ),
        )
        self.db.execute(
            "UPDATE model_requests SET status=?,ended_at=?,response_event=? WHERE id=?",
            ("retrying" if retry else status, None if retry else now(), response_event, request_id),
        )
        if response_event:
            row = self.db.execute(
                "SELECT * FROM model_requests WHERE id=?", (request_id,)
            ).fetchone()
            for edge in json.loads(row["inbound"]):
                self.db.execute(
                    "INSERT OR IGNORE INTO request_edges VALUES(?,?,?)",
                    (edge["source"], request_id, edge["kind"]),
                )
                if row["purpose"] == "agent":
                    self.db.execute(
                        "DELETE FROM pending_request_edges WHERE session_id=? AND source=? AND kind=?",
                        (row["session_id"], edge["source"], edge["kind"]),
                    )

    def request_graph(self, sid):
        root = self.session(sid).root_id
        requests = [
            dict(r)
            for r in self.db.execute(
                "SELECT r.* FROM model_requests r JOIN sessions s ON s.id=r.session_id WHERE s.root_id=? ORDER BY r.started_at,r.rowid",
                (root,),
            )
        ]
        for request in requests:
            request["inbound"] = json.loads(request["inbound"])
            request["attempts"] = [
                dict(r)
                for r in self.db.execute(
                    "SELECT * FROM model_attempts WHERE request_id=? ORDER BY attempt",
                    (request["id"],),
                )
            ]
            for attempt in request["attempts"]:
                attempt["usage"] = json.loads(attempt["usage"])
                attempt["failure"] = json.loads(attempt["failure"]) if attempt["failure"] else None
        edges = [
            dict(r)
            for r in self.db.execute(
                "SELECT e.* FROM request_edges e JOIN model_requests r ON r.id=e.target JOIN sessions s ON s.id=r.session_id WHERE s.root_id=? ORDER BY r.started_at,e.source,e.kind",
                (root,),
            )
        ]
        return {"requests": requests, "edges": edges}

    def request_usage(self, request_id, *, delegated=False):
        row = self.db.execute(
            "SELECT session_id FROM model_requests WHERE id=?", (request_id,)
        ).fetchone()
        if not row:
            raise KeyError(request_id)
        ids = {request_id}
        if delegated:
            # Attribute entire descendant lifetimes to the exact spawning request, not siblings.
            sessions = self.sessions(root_id=self.session(row[0]).root_id)
            selected = set()
            while True:
                more = {
                    s.id
                    for s in sessions
                    if s.spawned_by_request_id in ids or s.parent_id in selected
                }
                if more <= selected:
                    break
                selected |= more
                ids |= {
                    r[0]
                    for s in selected
                    for r in self.db.execute(
                        "SELECT id FROM model_requests WHERE session_id=?", (s,)
                    )
                }
        total = Usage().model_dump()
        for rid in ids:
            for r in self.db.execute("SELECT usage FROM model_attempts WHERE request_id=?", (rid,)):
                for key, value in json.loads(r[0]).items():
                    total[key] = None if value is None or total[key] is None else total[key] + value
        return Usage(**total)
