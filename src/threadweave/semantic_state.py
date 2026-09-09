"""Host-owned live semantic state projected from the trajectory tree into compaction."""

from .context_budget import pending_ledger
from .storage import encode


def completion_evidence(store, sid):
    """A small live receipt must not disappear inside a retired verifier transcript."""
    verifiers = store.events(sid, kind="verifier_result", limit=1)
    children = [s for s in store.sessions() if s.parent_id == sid]
    if not verifiers and not children:
        return None
    return {
        "last_full_verification": (
            {
                "source_event": verifiers[0]["id"],
                "passed": verifiers[0]["payload"].get("passed"),
                "receipt": verifiers[0]["payload"].get("result"),
            }
            if verifiers
            else None
        ),
        "children": [
            {
                "id": s.id,
                "status": str(s.outcome),
                "conclusion": s.result[:600] if s.result else None,
                "completion_events": [
                    e["id"] for e in store.events(s.id, kind="completion", limit=1)
                ],
            }
            for s in children
        ],
    }


def work_items(store, sid):
    items = {}
    for event in store.iter_events(sid, kind="semantic_state_updated"):
        payload = event["payload"]
        items[payload["id"]] = {**payload, "source_event": event["id"]}
    return list(items.values())


def capture(runtime_context, sid):
    store = runtime_context.store
    session = store.session(sid)
    children = [child for child in store.sessions() if child.parent_id == sid]
    snapshots = store.events(sid, kind="kernel_snapshot", limit=1)
    refinements = [
        {k: e[k] for k in ("id", "kind", "version", "title")} for e in store.harness.entries(sid)
    ]
    evidence = store.events(sid, kind="verification_evidence", limit=3) + store.events(
        sid, kind="verifier_result", limit=1
    )
    return {
        "owner": sid,
        "root": session.root_id,
        "parent": session.parent_id,
        "active_goal": store.goal(sid),
        "original_assignment": session.instruction,
        "unresolved_work": pending_ledger(session.summary),
        "tracked_work": work_items(store, sid),
        "branches": [
            {
                "id": child.id,
                "assignment": child.instruction,
                "status": str(child.outcome),
                "kernel_id": child.kernel_id,
                "completion_events": [
                    e["id"] for e in store.events(child.id, kind="completion", limit=1)
                ],
                "result": child.result,
                "evidence": [
                    e["id"] for e in store.events(child.id, kind="python_result", limit=3)
                ],
                "pending_messages": [
                    m["id"] for m in store.messages(child.id, pending=True, limit=20)
                ],
            }
            for child in children
        ],
        "verification": [{"event_id": e["id"], "evidence": e["payload"]} for e in evidence],
        "kernel_snapshot": snapshots[-1]["payload"] if snapshots else None,
        "harness_versions": refinements,
    }


def protected_summary(tree):
    fields = {
        "requirement": "unresolved_requirements",
        "hypothesis": "active_hypotheses",
        "blocker": "blockers",
        "decision": "decisions",
        "failed_approach": "decisions",
    }
    result = {field: [] for field in fields.values()}
    for item in tree["tracked_work"]:
        if item["status"] == "open":
            result[fields[item["kind"]]].append(
                {"id": item["id"], "text": item["text"], "source_events": [item["source_event"]]}
            )
    for branch in tree["branches"]:
        if branch["status"] == "active":
            result["unresolved_requirements"].append(
                {
                    "id": "child-" + branch["id"],
                    "text": "Pending child work; retrieve branch assignment and partial evidence",
                    "child_id": branch["id"],
                }
            )
        elif branch.get("result"):
            result.setdefault("established_facts", []).append(
                {
                    "child_id": branch["id"],
                    "conclusion": branch["result"][:600],
                    "source_events": branch["evidence"],
                }
            )
    return result


def compact_reference(tree, artifact):
    return {
        "id": artifact,
        "purpose": "Live trajectory tree: unresolved work, child conclusions, verification, REPL manifest/recipes and harness versions",
        "reference": f"artifacts.load({artifact!r}); children="
        + encode([{"id": b["id"], "status": b["status"]} for b in tree["branches"]]),
    }
