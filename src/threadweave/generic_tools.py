"""Domain-independent retrieval and artifact capabilities."""

import json

from pydantic import Field

from .models import Record
from .retrieval import search
from .tools import PathArgs, Tool


class QueryArgs(Record):
    query: str
    limit: int = Field(default=20, ge=1, le=100)


class HistorySearchArgs(QueryArgs):
    kind: str | None = None
    session_id: str | None = None


def register(registry):
    async def history(c, a):
        result = search(
            c.runtime.store,
            c.session_id,
            **a.model_dump(),
            priority_kinds=c.runtime.environment.call(c.session_id, "evidence_signals", default=()),
        )
        c.runtime.store.event(
            c.session_id,
            "history_retrieval",
            {"query": a.query, "matches": [r["id"] for r in result]},
            parent=c.source_event,
        )
        return result

    async def artifacts(c, a):
        return search(c.runtime.store, c.session_id, a.query, kind="artifact", limit=a.limit)

    async def import_artifact(c, a):
        with c.path(a.path).open("rb") as stream:
            aid = c.runtime.artifacts.put_stream(
                c.session_id,
                stream,
                source_event=c.source_event,
                media_type="application/octet-stream",
            )
        return c.runtime.artifacts.metadata(c.session_id, aid)

    async def skills(c, a):
        return [
            {k: e[k] for k in ("id", "title", "version", "content")}
            for e in c.runtime.store.harness.entries(c.session_id)
            if e["kind"] == "skill" and a.query.lower() in json.dumps(e).lower()
        ][: a.limit]

    for name, args, handler, permissions, feature in (
        ("history_search", HistorySearchArgs, history, (), "history_retrieval"),
        ("artifact_search", QueryArgs, artifacts, (), "history_retrieval"),
        ("artifact_import", PathArgs, import_artifact, ("workspace.read",), None),
        ("skill_search", QueryArgs, skills, ("state",), None),
    ):
        registry.register(
            Tool(
                name,
                {
                    "history_search": "Search durable trajectory events and observations with lexical search.",
                    "artifact_search": "Search indexed artifact excerpts in this trajectory.",
                    "artifact_import": "Retain an exact workspace file or binary as an artifact.",
                    "skill_search": "Search versioned executable skills and retained procedures.",
                }[name],
                args,
                handler,
                permissions,
                feature=feature,
            )
        )
