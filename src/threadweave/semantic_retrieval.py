"""Local embedding evidence retrieval. No network requests in the agent hot path.

Install the semantic extra and explicitly cache a FastEmbed model to enable this.
History FTS remains available when the local model cannot be loaded. Vectors are
versioned by encoder ID; a bounded recent semantic window complements full FTS.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def encoder(model, cache):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model, cache_dir=cache, local_files_only=True, threads=2)


@lru_cache(maxsize=128)
def query_vector(model, cache, query):
    return next(iter(encoder(model, cache).query_embed(query)))


def search(store, sid, query, *, kind=None, session_id=None, limit=10):
    import numpy as np

    config = store.config(sid).context
    roots = store.history_roots(sid)
    if session_id and store.session(session_id).root_id not in roots:
        raise PermissionError("Semantic retrieval outside trajectory ancestry")
    model = encoder(config.embedding_model, config.embedding_cache)
    kinds = (
        "user_input",
        "agent_message_received",
        "verifier_result",
        "coding_command",
        "failure",
        "workspace_effects",
        "code_edit",
        "experiment_conclusion",
        "working_focus",
        "context_compaction",
    )
    for root in roots:
        cursor = store.db.execute(
            "SELECT seq FROM semantic_cursors WHERE root_id=? AND model=?",
            (root, config.embedding_model),
        ).fetchone()
        rows = store.db.execute(
            "SELECT id,session_id,type,seq,substr(payload,1,2000) AS excerpt FROM events WHERE root_id=? AND seq>? AND type IN ("
            + ",".join("?" for _ in kinds)
            + ") ORDER BY seq LIMIT 64",
            (root, cursor[0] if cursor else 0, *kinds),
        ).fetchall()
        if rows:
            vectors = model.embed([r["excerpt"] for r in rows])
            with store.transaction():
                for row, vector in zip(rows, vectors, strict=True):
                    store.db.execute(
                        "INSERT OR REPLACE INTO semantic_evidence VALUES(?,?,?,?,?,?,?,?)",
                        (
                            row["id"],
                            config.embedding_model,
                            root,
                            row["session_id"],
                            row["type"],
                            row["seq"],
                            row["excerpt"],
                            np.asarray(vector, dtype=np.float32).tobytes(),
                        ),
                    )
                store.db.execute(
                    "INSERT OR REPLACE INTO semantic_cursors VALUES(?,?,?)",
                    (root, config.embedding_model, rows[-1]["seq"]),
                )
    sql = (
        "SELECT * FROM semantic_evidence WHERE model=? AND root_id IN ("
        + ",".join("?" for _ in roots)
        + ")"
    )
    args = [config.embedding_model, *roots]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    if session_id:
        sql += " AND session_id=?"
        args.append(session_id)
    rows = store.db.execute(sql + " ORDER BY seq DESC LIMIT 2048", args).fetchall()
    if not rows:
        return []
    vector = query_vector(config.embedding_model, config.embedding_cache, query)
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    scores = matrix @ vector / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector) + 1e-12)
    return [
        {
            "id": rows[i]["id"],
            "session_id": rows[i]["session_id"],
            "kind": rows[i]["kind"],
            "excerpt": rows[i]["excerpt"][:900],
            "semantic_score": float(scores[i]),
            "encoder": config.embedding_model,
            "ranking": "local embedding cosine similarity; bounded 2048-event semantic window",
        }
        for i in np.argsort(-scores)[:limit]
    ]
