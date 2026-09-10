"""Coding evidence selection for the coding adapter context."""

import json

from .retrieval import search


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
    verifications = store.records.first(
        "final_verifications",
        session_id=sid,
        order=(("created_at", True),),
        limit=1,
        fields=("passed", "body"),
    )
    if verifications and not verifications["passed"]:
        details = verifications["body"]["details"]
        state["unresolved_verifier_failures"] = details["violations"][:10]
    experiments = store.records.first(
        "experiments",
        session_id=sid,
        order=(("created_at", True),),
        limit=1,
        fields=("id", "status", "body"),
    )
    if experiments:
        state["latest_experiment"] = {
            "id": experiments["id"],
            "status": experiments["status"],
            "hypothesis": experiments["body"]["hypothesis"][:500],
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
                r
                for r in search(
                    store,
                    sid,
                    terms,
                    limit=8,
                    priority_kinds=("coding_command", "experiment_conclusion"),
                )
                if r["id"] not in cited
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
