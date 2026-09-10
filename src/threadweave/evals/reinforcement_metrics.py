"""Read-only lifecycle evidence. Never schedules or changes reinforcement."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path


def object_response(text):
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, TypeError):
        return {}


def lifecycle(directory):
    directory = Path(directory)
    with sqlite3.connect(f"file:{directory / 'state/history.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        root = next(
            json.loads(r[0])
            for r in db.execute("SELECT body FROM sessions")
            if json.loads(r[0])["parent_id"] is None
        )
        sid = root["id"]
        config = json.loads(
            db.execute("SELECT body FROM configs WHERE id=?", (root["config_id"],)).fetchone()[0]
        )
        events = [
            dict(r)
            for r in db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq", (sid,))
        ]
        requests = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM model_requests WHERE session_id=? ORDER BY started_at", (sid,)
            )
        ]
        actions = [dict(r) for r in db.execute("SELECT * FROM actions WHERE session_id=?", (sid,))]
        paths = dict(db.execute("SELECT id,path FROM artifacts"))
    for e in events:
        e["payload"] = json.loads(e["payload"])
    bodies = {
        r["id"]: json.loads((directory / "state" / paths[r["body_artifact"]]).read_text())
        for r in requests
    }
    responses = {r["response_event"]: r for r in requests if r["purpose"] == "agent"}
    agent_requests = [r for r in requests if r["purpose"] == "agent"]
    reviews = [e for e in events if e["type"] == "refinement_review"]
    by_id = {e["id"]: e for e in events}
    proposals = [
        object_response(by_id[r["response_event"]]["payload"].get("text", ""))
        for r in requests
        if r["purpose"] == "refinement" and r["response_event"] in by_id
    ]
    completed = [e for e in events if e["type"] == "refine_complete"]
    edits = [x for e in completed for x in e["payload"].get("appliedEdits", []) if x.get("applied")]
    by_kind = Counter(e["type"] for e in events)
    turns_since_review, peak, boundaries = 0, 0, []
    for e in events:
        if e["id"] in responses:
            turns_since_review += 1
            peak = max(peak, turns_since_review)
            boundaries.append(
                {
                    "root_turn": bodies[responses[e["id"]]["id"]]["turn"] + 1,
                    "turns_since_last_review": turns_since_review,
                    "response_event": e["id"],
                }
            )
        if e["type"] == "refine_complete" or (
            e["type"] == "refinement_review" and not e["payload"]["shouldRefine"]
        ):
            turns_since_review = 0
    automatic = []
    for r in requests:
        if r["purpose"] != "refinement_review":
            continue
        body = bodies[r["id"]]
        # The actual reviewer input is the authoritative trigger, including failures.
        text = body["messages"][-1]["content"]
        if text.startswith("{"):
            trigger = json.loads(text)["trigger"]  # Preserved pre-port artifacts.
        else:
            match = re.search(r"<trigger>\n(compact|turn_interval); ([0-9]+) assistant turns", text)
            if not match:
                raise ValueError("Reviewer request has no Prime trigger section")
            trigger = {"reason": match[1], "turnsSinceLastReview": int(match[2])}
        automatic.append(
            {
                "request_id": r["id"],
                "root_turn": sum(
                    e["id"] in responses and e["timestamp"] <= r["started_at"] for e in events
                ),
                "status": r["status"],
                "trigger": trigger,
            }
        )
    explicit = []
    for a in actions:
        args = json.loads(a["arguments"])
        if a["name"] != "host_request" or args.get("operation") != "refine.run":
            continue
        event = next(e for e in events if e["id"] == a["source_event"])
        preceding = [r for r in agent_requests if r["started_at"] <= event["timestamp"]]
        r = preceding[-1] if preceding else None
        explicit.append(
            {
                "action_id": a["id"],
                "timestamp": event["timestamp"],
                "root_turn": bodies[r["id"]]["turn"] + 1 if r else None,
                "instructions": args.get("payload", {}).get("instructions"),
                "request": args,
                "evidence_request_id": r["id"] if r else None,
                "preceding_conversation": bodies[r["id"]]["messages"] if r else [],
                "result": json.loads(a["result"]) if a["result"] else None,
            }
        )
    outcomes = []
    from threadweave.refinement import parse_object

    failed_responses = False
    for request in requests:
        if request["purpose"] not in {"refinement", "refinement_review"}:
            continue
        if request["status"] in {"failed", "interrupted"}:
            failed_responses = True
        event = by_id.get(request["response_event"])
        if event:
            if event["payload"].get("metadata", {}).get("stop_reason") in {
                "error",
                "length",
                "aborted",
            }:
                failed_responses = True
            try:
                parse_object(event["payload"].get("text", ""))
            except ValueError:
                failed_responses = True
    if by_kind["refine_failed"] or failed_responses:
        outcomes.append("REFINEMENT_FAILED")
    if edits:
        outcomes.append("TRIGGER_APPLIED_EDIT")
    if any(
        not e["payload"].get("edits") and not e["payload"].get("appliedEdits") for e in completed
    ):
        outcomes.append("TRIGGER_APPROVED_EMPTY_PLAN")
    if any(e["payload"]["shouldRefine"] is False for e in reviews):
        outcomes.append("TRIGGER_REVIEW_DECLINED")
    if not outcomes:
        outcomes.append("TRIGGER_PENDING" if automatic or explicit else "NO_TRIGGER")
    triggers = Counter(a["trigger"]["reason"] for a in automatic)
    delivery = []
    for event in completed:
        result = event["payload"]
        future = [r for r in agent_requests if r["started_at"] > event["timestamp"]]
        next_body = bodies[future[0]["id"]] if future else None
        system = (
            "\n".join(m.get("content", "") for m in next_body["messages"] if m["role"] == "system")
            if next_body
            else ""
        )
        ordinary = [m for m in next_body["messages"] if m["role"] != "system"] if next_body else []
        path = Path(result["harnessStatePath"])
        state = json.loads(path.read_text()) if path.is_file() else None
        delivery.append(
            {
                "refinement_id": result["id"],
                "harness_json_exists": state is not None,
                "audit_event": event["id"],
                "next_request_id": future[0]["id"] if future else None,
                "next_root_turn": next_body["turn"] + 1 if next_body else None,
                "entries": [
                    {
                        "id": edit["id"],
                        "kind": edit["kind"],
                        "action": edit["action"],
                        "current_disk_matches_applied_version": state is not None
                        and state["entries"][edit["kind"]].get(edit["id"]) == edit.get("after"),
                        "applied_version": edit.get("after", {}).get("version"),
                        "current_disk_version": state["entries"][edit["kind"]]
                        .get(edit["id"], {})
                        .get("version")
                        if state
                        else None,
                        "next_request_harness_versions": next_body.get("metadata", {})
                        .get("execution_inputs", {})
                        .get("harness_state", [])
                        if next_body
                        else [],
                        "system_contains_title": edit.get("after", {}).get("title", "") in system
                        if edit.get("after")
                        else None,
                        "system_contains_id": edit["id"] in system if edit.get("after") else None,
                    }
                    for edit in result["appliedEdits"]
                    if edit["applied"]
                ],
                "ordinary_messages_contain_refinement_notice": any(
                    "refinement_notice" in str(m) or "<harness_state>" in str(m) for m in ordinary
                ),
                "ordinary_messages_contain_audit_summary": any(
                    result["summary"] in str(m) for m in ordinary
                )
                if result["summary"]
                else None,
            }
        )
    return {
        "root_assistant_turns": root["turns"],
        "explicit_refine_calls": len(explicit),
        "compactions": by_kind["context_compaction"],
        "interval_threshold_reached": peak >= config["refinement"]["turn_interval"],
        "compaction_trigger_reached": bool(
            by_kind["context_compaction"] and config["refinement"]["compact"]
        ),
        "interval_triggers": triggers["turn_interval"],
        "compaction_triggers": triggers["compact"],
        "automatic_review_requests": len(automatic),
        "review_approvals": sum(e["payload"]["shouldRefine"] is True for e in reviews),
        "review_declines": sum(e["payload"]["shouldRefine"] is False for e in reviews),
        "planner_calls": sum(r["purpose"] == "refinement" for r in requests),
        "refinements_applied": sum(
            any(x.get("applied") for x in e["payload"].get("appliedEdits", [])) for e in completed
        ),
        "non_empty_refinement_proposals": sum(bool(p.get("edits")) for p in proposals),
        "typed_edits": len(edits),
        "edits_by_kind": dict(Counter(e["kind"] for e in edits)),
        "failures": by_kind["refine_failed"],
        "outcomes": outcomes,
        "policy": config["refinement"],
        "max_turns_since_review": peak,
        "REPL": by_kind["python_execution"],
        "RLM": sum(
            a["name"] == "host_request" and json.loads(a["arguments"]).get("operation") == "rlm.run"
            for a in actions
        ),
        "automatic_reviews": automatic,
        "explicit_calls": explicit,
        "assistant_boundaries": boundaries,
        "delivery_proofs": delivery,
        "refinement_events": [
            e
            for e in events
            if e["type"]
            in {
                "refine_scheduled",
                "refinement_review",
                "refine_complete",
                "refine_failed",
                "context_compaction",
            }
        ],
    }
