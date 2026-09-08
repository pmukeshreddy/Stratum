"""Bounded FTS retrieval across explicit trajectory ancestry."""

import json
import re


def search(store, sid, query, *, kind=None, session_id=None, limit=20):
    roots = store.history_roots(sid)
    if session_id and store.session(session_id).root_id not in roots:
        raise PermissionError("Search session is outside trajectory ancestry")
    # Treat user query as terms, not executable FTS syntax. Avoid malformed MATCH expressions.
    terms = re.findall(r"\w+", query, re.UNICODE)
    if not terms:
        return []
    expression = " OR ".join('"' + term + '"' for term in terms[:20])
    sql = (
        "SELECT id,session_id,kind,bm25(history_fts) AS lexical_score,rowid AS ordinal,"
        "snippet(history_fts,4,'[',']','…',40) AS excerpt FROM history_fts WHERE history_fts MATCH ? AND root_id IN ("
        + ",".join("?" for _ in roots)
        + ")"
    )
    params = [expression, *roots]
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    if session_id:
        sql += " AND session_id=?"
        params.append(session_id)
    sql += " ORDER BY rank LIMIT ?"
    params.append(min(100, max(20, limit * 4)))
    candidates = [dict(row) for row in store.db.execute(sql, params)]
    newest = max((r["ordinal"] for r in candidates), default=1)
    important = {
        "verifier_result",
        "coding_command",
        "failure",
        "experiment_conclusion",
        "agent_message_received",
    }
    for row in candidates:
        row["excerpt"] = row["excerpt"][:900]
        overlap = sum(t.casefold() in row["excerpt"].casefold() for t in terms) / len(terms)
        row["score"] = (
            overlap + (0.2 if row["kind"] in important else 0) + 0.1 * row["ordinal"] / newest
        )
        row["ranking"] = (
            "FTS/BM25 candidates + term overlap + evidence type + relative recency; not semantic"
        )
    lexical = sorted(candidates, key=lambda r: (r["score"], -r["lexical_score"]), reverse=True)
    if store.config(sid).context.embedding_model:
        from .semantic_retrieval import search as semantic_search

        try:
            semantic = semantic_search(
                store, sid, query, kind=kind, session_id=session_id, limit=limit * 2
            )
        except (ImportError, ValueError, OSError) as exc:
            store.event(
                sid,
                "semantic_retrieval_unavailable",
                {"reason": str(exc)[:500], "fallback": "FTS and structural/recency ranking"},
            )
            semantic = []
        fused = {}
        for source, rows in (("lexical", lexical), ("semantic", semantic)):
            for rank, row in enumerate(rows):
                record = fused.setdefault(row["id"], {**row, "score": 0, "sources": []})
                record["score"] += 1 / (60 + rank)
                record["sources"].append(source)
                record["ranking"] = (
                    "reciprocal-rank fusion of local embeddings and FTS/type/recency"
                )
        return sorted(fused.values(), key=lambda r: -r["score"])[:limit]
    return lexical[:limit]


def coding_focus(store, sid, *, index_provider=None):
    """Small working set, not automatic recall of the whole trajectory."""
    if (
        store.config(sid).task.adapter != "coding"
        or not store.config(sid).features.enhanced_code_index
    ):
        return ""
    state = {}
    focus = store.events(sid, kind="working_focus", limit=1)
    if focus:
        state["investigation"] = focus[0]["payload"]
    verifications = store.db.execute(
        "SELECT passed,body FROM final_verifications WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    ).fetchone()
    if verifications and not verifications[0]:
        details = json.loads(verifications[1])["details"]
        state["unresolved_verifier_failures"] = details["violations"][:10]
    experiments = store.db.execute(
        "SELECT id,status,body FROM experiments WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    ).fetchone()
    if experiments:
        state["latest_experiment"] = {
            "id": experiments[0],
            "status": experiments[1],
            "hypothesis": json.loads(experiments[2])["hypothesis"][:500],
        }
    edits = store.events(sid, kind="code_edit", limit=3) + store.events(
        sid, kind="workspace_effects", limit=3
    )
    state["recent_failures"] = []
    for event in store.events(sid, kind="coding_command", limit=5):
        payload = event["payload"]
        if payload.get("passed"):
            continue
        failures = []
        for failure in payload.get("failures", [])[:2]:
            # Keep decision-bearing fields, not repeated JUnit stack/output blobs.
            # Exact full diagnostics remain retrievable by source event/artifact.
            item = {
                k: failure[k]
                for k in ("test_id", "file", "line", "failure_type", "expected", "actual")
                if failure.get(k) is not None
            }
            item["message"] = str(failure.get("message", ""))[:350]
            if not item["message"]:
                item["message"] = str(failure.get("stack", ""))[-350:]
            failures.append(item)
        state["recent_failures"].append(
            {
                "event_id": event["id"],
                "artifact_id": payload.get("structured_artifact"),
                "failures": failures,
            }
        )
    state["recent_child_evidence"] = [
        {"id": m["id"], "body": m["body"][:400]}
        for m in store.messages(sid, limit=3)
        if m.get("sender_id")
    ]
    state["diff_evidence_events"] = [e["id"] for e in edits[:3]]
    state["recently_modified_files"] = sorted(
        {p for e in edits for p in e["payload"].get("files", {})}
    )[:30]
    if focus and store.config(sid).features.history_retrieval:
        terms = " ".join(
            focus[0]["payload"].get("symbols", []) + focus[0]["payload"].get("files", [])
        )
        if terms:
            cited = (
                {e["event_id"] for e in state["recent_failures"]}
                | {e["id"] for e in edits}
                | {focus[0]["id"]}
            )
            state["related_evidence"] = [
                r for r in search(store, sid, terms, limit=8) if r["id"] not in cited
            ][:3]
    if index_provider:
        from pathlib import Path

        from .repository import LANGUAGES

        implicated = state["recently_modified_files"] + (
            focus[0]["payload"].get("files", []) if focus else []
        )
        for command in store.events(sid, kind="coding_command", limit=2):
            if not command["payload"].get("passed"):
                implicated += [d.get("file", "") for d in command["payload"].get("diagnostics", [])]
        packet = []
        for path in dict.fromkeys(p for p in implicated if p and Path(p).suffix in LANGUAGES):
            if len(packet) == 2:
                break
            try:
                index = index_provider(sid)
                outline = index.outline(path)
                packet.append(
                    {
                        "file": path,
                        "definitions": [
                            {
                                k: s[k]
                                for k in ("name", "line", "end_line", "signature", "quality")
                                if k in s
                            }
                            for s in outline["symbols"][:4]
                        ],
                        "module_dependencies": index.resolver.bindings(path)[:3],
                    }
                )
            except (ValueError, PermissionError, OSError):
                continue
        if packet:
            state["source_evidence"] = packet
    return json.dumps(state)[:6000]
