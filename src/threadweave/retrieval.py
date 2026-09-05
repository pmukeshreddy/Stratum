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
    expression = " AND ".join('"' + term + '"' for term in terms[:20])
    sql = (
        "SELECT id,session_id,kind,snippet(history_fts,4,'[',']','…',40) AS excerpt FROM history_fts WHERE history_fts MATCH ? AND root_id IN ("
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
    params.append(min(100, limit))
    return [dict(row) for row in store.db.execute(sql, params)]


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
    edits = store.events(sid, kind="code_edit", limit=3)
    try:
        from .gitops import git

        state["current_diff_summary"] = git(
            store.session(sid).workspace.path,
            "diff",
            "HEAD",
            "--stat",
            "--no-ext-diff",
            "--no-textconv",
            "--",
        )[:1000]
    except (ValueError, OSError):
        state["current_diff_summary"] = "Git diff currently unavailable"
    state["recently_modified_files"] = sorted(
        {p for e in edits for p in e["payload"].get("files", {})}
    )[:30]
    return json.dumps(state)[:2500]
