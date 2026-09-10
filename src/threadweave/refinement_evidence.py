"""Observed task-local evidence for optional refinement."""

import hashlib
import json

from .semantic_state import work_items


def bounded_records(records, limit):
    """Keep complete early/recent records where possible; disclose every omission."""
    encoded = [json.dumps(row, ensure_ascii=False, default=str) for row in records]
    if sum(map(len, encoded)) <= limit:
        return records, {"omitted": 0, "truncated": []}
    selected, truncated = {}, []
    # Preserve before-behavior as well as recent observations, without interpreting either.
    for indices, budget in (
        (range(len(records)), limit // 3),
        (range(len(records) - 1, -1, -1), limit * 2 // 3),
    ):
        for index in indices:
            if index in selected:
                continue
            if budget < 256:
                break
            text = encoded[index]
            if len(text) <= budget:
                selected[index] = records[index]
                budget -= len(text)
            else:
                selected[index] = {
                    "id": records[index].get("id"),
                    "type": records[index].get("type"),
                    "truncated": True,
                    "excerpt": text[: budget // 2 - 80]
                    + "\n[...omitted...]\n"
                    + text[-budget // 2 + 80 :],
                }
                truncated.append(records[index].get("id"))
                break
    return [selected[i] for i in sorted(selected)], {
        "omitted": len(records) - len(selected),
        "truncated": truncated,
    }


def trajectory_evidence(store, sid):
    """Read existing trajectory evidence, including success and already attempted corrections."""
    records = []
    for event in store.iter_events(sid):
        kind, payload = event["type"], event["payload"]
        if kind == "model_response":
            if payload.get("metadata", {}).get("purpose", "agent") != "agent":
                continue
            payload = {k: payload.get(k) for k in ("text", "actions")}
        elif kind in {"python_error", "python_result"}:
            result = payload.get("result", {})
            payload = {k: result.get(k) for k in ("stdout", "stderr", "value", "error")}
        elif kind not in {
            "python_execution",
            "tool_result",
            "verification_result",
            "verifier_result",
            "semantic_state_updated",
            "observation",
            "message_received",
            "intervention",
        }:
            continue
        records.append({"id": event["id"], "seq": event["seq"], "type": kind, "payload": payload})
    records, coverage = bounded_records(records, 24_000)
    return {"records": records, "coverage": coverage}


def opportunity(store, sid):
    """Project live failures and tracked work without scheduling a review."""
    events = list(store.iter_events(sid))
    latest = {}
    for event in events:
        kind, payload = event["type"], event["payload"]
        if kind in {"python_error", "python_result"}:
            latest["python"] = event
        elif kind in {"verification_result", "verifier_result"}:
            latest[(kind, payload.get("level"))] = event
    evidence = []
    for event in latest.values():
        payload = event["payload"]
        if event["type"] != "python_error" and payload.get("passed") is not False:
            continue
        if event["type"] == "python_error":
            result = payload.get("result", {})
            payload = {
                "error": result.get("error"),
                "stdout": result.get("stdout", "")[-4000:],
                "stderr": result.get("stderr", "")[-2000:],
            }
        evidence.append({"event_id": event["id"], "type": event["type"], "evidence": payload})
    unresolved = [
        item
        for item in work_items(store, sid)
        if item["status"] == "open"
        and item["kind"] in {"hypothesis", "blocker", "decision", "failed_approach"}
    ]
    return {
        "observed_failures": evidence,
        "unresolved_work": unresolved,
    }


def opportunity_key(value):
    ids = [e["event_id"] for e in value["observed_failures"]]
    ids += [e["source_event"] for e in value["unresolved_work"]]
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest() if ids else None
