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
        overlap = sum(t.casefold() in row["excerpt"].casefold() for t in terms) / len(terms)
        row["score"] = (
            overlap + (0.2 if row["kind"] in important else 0) + 0.1 * row["ordinal"] / newest
        )
        row["ranking"] = (
            "FTS/BM25 candidates + term overlap + evidence type + relative recency; not semantic"
        )
    return sorted(candidates, key=lambda r: (r["score"], -r["lexical_score"]), reverse=True)[:limit]


def coding_focus(store, sid):
    """Small working set, not automatic recall of the whole trajectory."""
    if store.config(sid).task.adapter != "coding":
        return ""
    state = {}
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
    state["recent_failures"] = [
        {"event_id": e["id"], "failures": e["payload"].get("failures", [])[:3]}
        for e in store.events(sid, kind="coding_command", limit=5)
        if not e["payload"].get("passed")
    ]
    state["recent_child_evidence"] = [
        {"id": m["id"], "body": m["body"][:400]}
        for m in store.messages(sid, limit=3)
        if m.get("sender_id")
    ]
    state["diff_evidence_events"] = [e["id"] for e in edits[:3]]
    state["recently_modified_files"] = sorted(
        {p for e in edits for p in e["payload"].get("files", {})}
    )[:30]
    return json.dumps(state)[:2500]
