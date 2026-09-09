"""Bounded FTS retrieval across explicit trajectory ancestry."""

import re


def search(store, sid, query, *, kind=None, session_id=None, limit=20, priority_kinds=()):
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
        "failure",
        "agent_message_received",
        *priority_kinds,
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
