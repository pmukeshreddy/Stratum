"""Bounded lexical retrieval across explicit trajectory ancestry."""

import re


def search(store, sid, query, *, kind=None, session_id=None, limit=20, priority_kinds=()):
    roots = store.history_roots(sid)
    if session_id and store.session(session_id).root_id not in roots:
        raise PermissionError("Search session is outside trajectory ancestry")
    # Tokenize the query consistently with the rebuildable lexical index.
    terms = re.findall(r"\w+", query, re.UNICODE)
    if not terms:
        return []
    # Build the lexical candidates directly from authoritative history and artifact metadata.
    # The in-memory inverted index is discarded on restart and can always be rebuilt.
    candidates = store.search_index.search(
        roots, terms[:20], kind=kind, session_id=session_id, limit=min(100, max(20, limit * 4))
    )
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
            "BM25 candidates + term overlap + evidence type + relative recency; not semantic"
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
                {"reason": str(exc)[:500], "fallback": "lexical and structural/recency ranking"},
            )
            semantic = []
        fused = {}
        for source, rows in (("lexical", lexical), ("semantic", semantic)):
            for rank, row in enumerate(rows):
                record = fused.setdefault(row["id"], {**row, "score": 0, "sources": []})
                record["score"] += 1 / (60 + rank)
                record["sources"].append(source)
                record["ranking"] = (
                    "reciprocal-rank fusion of local embeddings and lexical/type/recency"
                )
        return sorted(fused.values(), key=lambda r: -r["score"])[:limit]
    return lexical[:limit]
