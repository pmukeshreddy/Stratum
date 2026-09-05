"""No-progress evidence, not a planner. Recovery is allowed before the configured stop threshold."""

import hashlib

from .storage import encode


def state_fingerprint(runtime, sid):
    if runtime.store.config(sid).task.adapter != "coding":
        return "generic"
    index = runtime.index(sid)
    return hashlib.sha256(
        encode([(r["path"], r["sha256"]) for r in index.entries()]).encode()
    ).hexdigest()


def observe(runtime, sid, name, arguments):
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
