"""Durable logical requests and transport attempts on the Store's transaction boundary."""

import hashlib

from .models import Usage, now


class RequestHistory:
    def last_request(self, sid, *, purpose=None, trajectory=False):
        match = {"session_id": sid, "status": "completed"}
        if purpose:
            match["purpose"] = purpose
        if trajectory:
            match["request_kind"] = "trajectory"
        row = self.records.first(
            "model_requests", order=(("ended_at", True), ("_order", True)), **match
        )
        return row["id"] if row else None

    def queue_request_edge(self, sid, source, kind):
        if source:
            self.records.insert(
                "pending_request_edges",
                {"session_id": sid, "source": source, "kind": kind},
                on_conflict="ignore",
            )

    def commit_compaction(self, sid, request_id):
        if self.last_request(sid, trajectory=True) != request_id:
            return
        for edge in self.records.select(
            "pending_request_edges", session_id=sid, fields=("source", "kind")
        ):
            self.records.delete(
                "request_edges", source=edge["source"], target=request_id, kind="continuation"
            )
            self.records.insert(
                "request_edges",
                {"source": edge["source"], "target": request_id, "kind": edge["kind"]},
                on_conflict="ignore",
            )
        self.records.delete("pending_request_edges", session_id=sid)
        self.queue_request_edge(sid, request_id, "compaction")

    def begin_request(self, request, artifact):
        from .storage import encode

        sid, purpose = request.session_id, request.metadata.get("purpose", "agent")
        inbound = []
        if request.request_kind == "trajectory" and purpose == "agent":
            inbound = [
                dict(r)
                for r in self.records.select(
                    "pending_request_edges", session_id=sid, fields=("source", "kind")
                )
            ]
            parent = self.session(sid).spawned_by_request_id
            if parent and not self.last_request(sid, purpose="agent"):
                inbound.append({"source": parent, "kind": "subagent_call"})
        previous = (
            self.last_request(sid, trajectory=True)
            if request.request_kind == "trajectory"
            else None
        )
        if previous and not any(e["source"] == previous for e in inbound):
            inbound.append({"source": previous, "kind": "continuation"})
        # Hash the immutable provider-shaped body, including private continuation identity.
        from .request_context import projection

        body = {
            "config": request.config.model_dump(),
            **projection(request.messages, request.tools, request.config),
        }
        fingerprint = hashlib.sha256(encode(body).encode()).hexdigest()
        self.records.insert(
            "model_requests",
            {
                "id": request.request_id,
                "session_id": sid,
                "purpose": purpose,
                "body_hash": fingerprint,
                "body_artifact": artifact,
                "provider": request.config.name,
                "model": request.config.model,
                "status": "running",
                "started_at": now(),
                "ended_at": None,
                "response_event": None,
                "inbound": inbound,
                "request_kind": request.request_kind,
            },
        )

    def finish_request_attempt(
        self, request_id, event_id, usage, *, response_event=None, failure=None, retry=False
    ):

        status = "completed" if response_event else "failed"
        self.records.update(
            "model_attempts",
            {
                "status": status,
                "usage": usage.model_dump(),
                "failure": (failure if failure else None),
                "ended_at": now(),
            },
            event_id=event_id,
        )
        self.records.update(
            "model_requests",
            {
                "status": "retrying" if retry else status,
                "ended_at": None if retry else now(),
                "response_event": response_event,
            },
            id=request_id,
        )
        if response_event:
            row = self.records.first("model_requests", id=request_id)
            for edge in row["inbound"]:
                self.records.insert(
                    "request_edges",
                    {"source": edge["source"], "target": request_id, "kind": edge["kind"]},
                    on_conflict="ignore",
                )
                if row["purpose"] == "agent":
                    self.records.delete(
                        "pending_request_edges",
                        session_id=row["session_id"],
                        source=edge["source"],
                        kind=edge["kind"],
                    )

    def request_history(self, sid, *, kind=None):
        """All durable model calls, including auxiliary inference and transport attempts."""
        sessions = {s.id for s in self.sessions(root_id=self.session(sid).root_id)}
        requests = self.records.select(
            "model_requests",
            where=lambda row: (
                row["session_id"] in sessions and (kind is None or row["request_kind"] == kind)
            ),
            order=(("started_at", False), ("_order", False)),
        )
        for request in requests:
            request["attempts"] = self.records.select(
                "model_attempts", request_id=request["id"], order=(("attempt", False),)
            )
        return requests

    def request_graph(self, sid):
        """Primary agent/compaction trajectory; auxiliary accounting is in request_history."""
        requests = self.request_history(sid, kind="trajectory")
        by_id = {row["id"]: row for row in requests}
        sources = {
            row["id"]
            for row in self.records.select(
                "model_requests", request_kind="trajectory", fields=("id",)
            )
        }
        edges = self.records.select(
            "request_edges", where=lambda row: row["source"] in sources and row["target"] in by_id
        )
        edges.sort(key=lambda row: (by_id[row["target"]]["started_at"], row["source"], row["kind"]))
        return {"requests": requests, "edges": edges}

    def request_usage(self, request_id, *, delegated=False):
        row = self.records.first("model_requests", id=request_id, fields=("session_id",))
        if not row:
            raise KeyError(request_id)
        ids = {request_id}
        if delegated:
            # Attribute entire descendant lifetimes to the exact spawning request, not siblings.
            sessions = self.sessions(root_id=self.session(row["session_id"]).root_id)
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
                    r["id"]
                    for s in selected
                    for r in self.records.select("model_requests", session_id=s, fields=("id",))
                }
        total = Usage().model_dump()
        for rid in ids:
            for r in self.records.select("model_attempts", request_id=rid, fields=("usage",)):
                for key, value in r["usage"].items():
                    total[key] = None if value is None or total[key] is None else total[key] + value
        return Usage(**total)
