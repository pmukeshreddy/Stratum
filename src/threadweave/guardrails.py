"""No-progress evidence, not a planner. Recovery is allowed before the configured stop threshold."""

import hashlib
import json

from .storage import encode


def state_fingerprint(runtime, sid):
    store, session = runtime.store, runtime.store.session(sid)
    state = {}
    checkpoint = store.directory / "kernels" / session.kernel_id / "checkpoint.json"
    if checkpoint.is_file():
        data = json.loads(checkpoint.read_text())
        # Receipts change every execution even when computation made no progress.
        state["repl"] = {k: data.get(k) for k in ("values", "recipes", "missing")}
    state["messages"] = [m["id"] for m in store.messages(sid, limit=20)]
    state["entries"] = [(e["id"], e["version"]) for e in store.harness.entries(sid)]
    if store.config(sid).task.adapter == "coding":
        row = store.db.execute(
            "SELECT state_id FROM mutation_workspaces WHERE path=?", (session.workspace.path,)
        ).fetchone()
        state["files"] = row[0] if row else "not-yet-observed"
        state["edits"] = [e["id"] for e in store.events(sid, kind="code_edit", limit=1)]
    return hashlib.sha256(encode(state).encode()).hexdigest()


def observe(runtime, sid, name, arguments):
    read_action = name in {
        "workspace_read",
        "repo_search",
        "symbol_search",
        "references_search",
        "repo_definition",
        "repo_callers",
        "repo_callees",
        "repo_context_for_symbol",
        "history_search",
        "run_tests",
        "run_targeted_tests",
        "apply_patch",
    }
    if read_action:
        session = runtime.store.session(sid)
        row = runtime.store.db.execute(
            "SELECT state_id FROM mutation_workspaces WHERE path=?", (session.workspace.path,)
        ).fetchone()
        # Assigning a Python result variable does not make an unchanged query new.
        edited = runtime.store.events(sid, kind="code_edit", limit=1)
        fingerprint = encode(
            [row[0] if row else "unobserved-workspace", edited[0]["id"] if edited else None]
        )
    else:
        fingerprint = state_fingerprint(runtime, sid)
    signature = hashlib.sha256(encode([name, arguments, fingerprint]).encode()).hexdigest()
    recent = runtime.store.events(sid, kind="action_fingerprint", limit=100)
    count = 1 + sum(event["payload"]["signature"] == signature for event in recent)
    eid = runtime.store.event(
        sid,
        "action_fingerprint",
        {"signature": signature, "state": fingerprint, "action": name, "repetitions": count},
    )
    policy = runtime.store.config(sid).loop
    if count >= policy.warn_repetitions:
        evidence = {
            "action": name,
            "identical_repetitions": count,
            "state_changed": False,
            "recommendation": "Inspect retained evidence, change the approach, or explain why repetition is necessary.",
        }
        warning = runtime.store.event(sid, "no_progress", evidence, parent=eid)
        runtime.store.add_context(sid, warning, [{"role": "user", "content": encode(evidence)}])
    return count >= policy.stop_repetitions
