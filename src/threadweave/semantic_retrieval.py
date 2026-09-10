"""Local embedding evidence retrieval. No network requests in the agent hot path.

Install the semantic extra and explicitly cache a FastEmbed model to enable this.
Lexical history search remains available when the local model cannot be loaded. Vectors are
versioned by encoder ID; a bounded recent semantic window complements full lexical search.
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
    import json

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
        cursor = store.records.first(
            "semantic_cursors", root_id=root, model=config.embedding_model, fields=("seq",)
        )
        after = cursor["seq"] if cursor else 0
        rows = [
            {**row, "excerpt": json.dumps(row["payload"], ensure_ascii=False)[:2000]}
            for row in store.records.select(
                "events",
                root_id=root,
                where=lambda row, after=after: row["seq"] > after and row["type"] in kinds,
                order=(("seq", False),),
                limit=64,
            )
        ]
        if rows:
            vectors = model.embed([r["excerpt"] for r in rows])
            with store.transaction():
                for row, vector in zip(rows, vectors, strict=True):
                    store.records.insert(
                        "semantic_evidence",
                        {
                            "id": row["id"],
                            "model": config.embedding_model,
                            "root_id": root,
                            "session_id": row["session_id"],
                            "kind": row["type"],
                            "seq": row["seq"],
                            "excerpt": row["excerpt"],
                            "vector": np.asarray(vector, dtype=np.float32).tolist(),
                        },
                        on_conflict="replace",
                    )
                store.records.insert(
                    "semantic_cursors",
                    {"root_id": root, "model": config.embedding_model, "seq": rows[-1]["seq"]},
                    on_conflict="replace",
                )
    rows = store.records.select(
        "semantic_evidence",
        model=config.embedding_model,
        where=lambda row: (
            row["root_id"] in roots
            and (kind is None or row["kind"] == kind)
            and (session_id is None or row["session_id"] == session_id)
        ),
        order=(("seq", True),),
        limit=2048,
    )
    if not rows:
        return []
    vector = query_vector(config.embedding_model, config.embedding_cache, query)
    matrix = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in rows])
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
