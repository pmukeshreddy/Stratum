"""Evidence-based reports. Delivery, visibility, behavioral use and benefit are distinct."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .schema import accounting, save


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    content = path.read_text()
    lines = content.splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have finished the final line yet.
            if index != len(lines) - 1 or content.endswith("\n"):
                raise
    return rows


def task_result(task_id, packet, journal, failure, wall_seconds):
    graded = [a for a in journal.attempts if a.get("status") == "BENCHMARK_RESULT"]
    last = graded[-1] if graded else None
    autonomous = packet.get("autonomous", {}) if packet else {}
    stop = autonomous.get("stop_reason")
    status = failure.category if failure else "BENCHMARK_RESULT" if last else "ADAPTER_FAILURE"
    if not failure and stop in {"error", "aborted"}:
        status = "ADAPTER_FAILURE"
    if not failure and stop == "timeoutMs":
        status = "TIMEOUT"
    calls = journal.calls
    root_id = packet["after"]["root_session_id"] if packet else journal.progress.get("root_id")
    if root_id is None and calls:
        root_id = calls[0]["request"].get("root_id")
    limit_event = next(
        (
            e
            for e in reversed(journal.events)
            if e["type"] == "termination"
            and e["session_id"] == root_id
            and e["payload"].get("outcome") == "limited"
        ),
        None,
    )
    if (
        status == "ADAPTER_FAILURE"
        and limit_event
        and (failure is None or "Root stopped: limited:" in str(failure))
    ):
        status = "RESOURCE_LIMIT"
    valid = status == "BENCHMARK_RESULT"
    root_calls = [
        c
        for c in calls
        if c["request"]["session_id"] == root_id
        and c["request"].get("metadata", {}).get("purpose", "agent") == "agent"
    ]
    child_calls = [
        c
        for c in calls
        if c["request"].get("parent_id")
        and c["request"].get("metadata", {}).get("purpose", "agent") == "agent"
    ]
    auxiliaries = [
        c for c in calls if c["request"].get("metadata", {}).get("purpose", "agent") != "agent"
    ]

    def usage(rows, scope):
        value = accounting([c.get("response", {}).get("usage", {}) for c in rows], wall_seconds)
        value["noncached_input_tokens"] = max(
            0, value["input_tokens"] - value["cached_input_tokens"]
        )
        value["calls_attempted"] = len(rows)
        # Provider usage objects describe tokens; they do not necessarily count
        # invocations. A cancelled streaming request may report no usage at all.
        value["model_calls"] = len(rows)
        value["calls_completed"] = sum(bool(c.get("response")) for c in rows)
        unreported = [c for c in rows if not c.get("response", {}).get("usage_reported")]
        value["usage_complete"] = not unreported
        value["calls_without_reported_usage"] = len(unreported)
        value["unreported_call_attempts"] = [
            {"request_id": c["id"], "started": c.get("started")} for c in unreported
        ]
        value["token_accounting"] = (
            "complete_provider_report"
            if not unreported
            else "partial_provider_report; token totals exclude unreported attempts"
        )
        value["scope"] = scope
        return value

    return {
        "task_id": task_id,
        "status": status,
        "official_score": last["score"] if valid else None,
        "public_source_score": last.get("public_source_score") if valid else None,
        "verification_mode": last.get("verification_mode", "signed_controller") if last else None,
        "trusted_oracle_verified": last.get("trusted_oracle_verified", True) if valid else None,
        "official_passed": last.get("official_passed", last["passed"]) if valid else None,
        "passed": last["passed"] if valid else None,
        "last_graded_score": last["score"] if last else None,
        "score_attempt_id": last["id"] if last else None,
        "stop_reason": (
            limit_event["payload"].get("result")
            if status == "RESOURCE_LIMIT"
            else str(failure)
            if failure
            else stop
        ),
        "resource_limit_event_id": limit_event["id"] if status == "RESOURCE_LIMIT" else None,
        "verifier_attempts": len(journal.attempts),
        "public_rehearsal_attempts": len(getattr(journal, "rehearsals", [])),
        "public_rehearsal_attempt_ids": [a["id"] for a in getattr(journal, "rehearsals", [])],
        "public_rehearsal_scores": [
            a.get("public_source_score") for a in getattr(journal, "rehearsals", [])
        ],
        "verifier_suppressed": sum(not c["rerun"] for c in autonomous.get("checks", [])),
        "autonomous": autonomous,
        "usage": {
            "root": usage(root_calls, "root agent calls only"),
            "children": usage(child_calls, "descendant agent calls only"),
            "auxiliary": usage(
                auxiliaries, "refinement/review/compaction and other auxiliary calls"
            ),
        },
        "wall_seconds": wall_seconds,
        "root_id": root_id,
        "kernel_id": packet["after"]["kernel_id"] if packet else None,
        "workspace": packet["after"]["workspace"] if packet else None,
    }


def refinement_counts(events, calls):
    reviews = [e for e in events if e["type"] == "refinement_review"]
    plans = [e for e in events if e["type"] == "refinement_plan"]
    edits = [
        edit
        for e in events
        if e["type"] == "refine_complete"
        for edit in e["payload"].get("appliedEdits", [])
    ]
    return {
        "explicit": sum(
            e["type"] == "refine_scheduled" and e["payload"].get("source") == "self" for e in events
        ),
        "compactions": sum(e["type"] == "context_compaction" for e in events),
        "compact_reviews": sum(e["payload"].get("reason") == "compact" for e in reviews),
        "interval_reviews": sum(e["payload"].get("reason") == "turn_interval" for e in reviews),
        "declines": sum(not e["payload"].get("shouldRefine") for e in reviews),
        "approvals": sum(bool(e["payload"].get("shouldRefine")) for e in reviews),
        "review_calls": sum(
            c["request"].get("metadata", {}).get("purpose") == "refinement_review" for c in calls
        ),
        "planner_calls": sum(
            c["request"].get("metadata", {}).get("purpose") == "refinement" for c in calls
        ),
        "empty_plans": sum(e["payload"].get("status") == "APPROVED_EMPTY_PLAN" for e in plans),
        "candidate_edits": sum(len(e["payload"].get("assessments", [])) for e in plans),
        "rejected_redundant": sum(
            a.get("status") == "REJECTED_REDUNDANT"
            for e in plans
            for a in e["payload"].get("assessments", [])
        ),
        "rejected_unsupported": sum(
            a.get("status") == "REJECTED_UNSUPPORTED"
            for e in plans
            for a in e["payload"].get("assessments", [])
        ),
        "applied_edits": sum(bool(e.get("applied")) for e in edits),
        "rejected_edits": sum(not e.get("applied") for e in edits),
    }


def audit_task(packet, journal):
    if not packet:
        return {
            "claims": [],
            "evidence_status": "No completed worker packet; inspect raw-events and model-calls",
        }
    events = sorted(journal.events, key=lambda e: e["seq"])
    root = packet["after"]["root_session_id"]
    calls = journal.calls

    def of(kind):
        return [e for e in events if e["type"] == kind]

    claims, children = [], []
    for event in of("rlm_admitted"):
        child_id = event["payload"]["child_id"]
        registry = next((s for s in packet["after"]["children"] if s["id"] == child_id), {})
        messages = [
            e for e in of("agent_message_sent") if e["payload"].get("sender_id") == child_id
        ]
        received = [
            e
            for e in of("agent_message_received")
            if e["session_id"] == root and e["payload"].get("sender_id") == child_id
        ]
        visible = [
            e
            for e in of("execution_input_consumed")
            if e["session_id"] == root and child_id in e["payload"].get("child_evidence", {})
        ]
        completions = [e for e in of("completion") if e["session_id"] == child_id]
        observed = [e for e in of("evaluation_file_observation") if e["session_id"] == child_id]
        claim = {"claim": "RLM_CALLED", "child_id": child_id, "event_ids": [event["id"]]}
        claims.append(claim)
        for kind, evidence in (
            ("CHILD_COMPLETED", completions),
            ("ROOT_RECEIVED_RESULT", received),
        ):
            if evidence:
                claims.append(
                    {"claim": kind, "child_id": child_id, "event_ids": [e["id"] for e in evidence]}
                )
        children.append(
            {
                "child_id": child_id,
                "parent_id": event["session_id"],
                "depth": event["payload"]["depth"],
                "assignment": event["payload"]["assignment"],
                "admission_event": event["id"],
                "admission_root_turn": sum(
                    e["session_id"] == root
                    and e["seq"] < event["seq"]
                    and e["payload"].get("metadata", {}).get("purpose", "agent") == "agent"
                    for e in of("model_response")
                ),
                "completion_status": registry.get("outcome"),
                "model_request_ids": [
                    c["id"] for c in calls if c["request"]["session_id"] == child_id
                ],
                "message_event_ids": [e["id"] for e in messages],
                "root_received_event_ids": [e["id"] for e in received],
                "root_context_event_ids": [e["id"] for e in visible],
                "file_observations": [{"event_id": e["id"], **e["payload"]} for e in observed],
                "files_touched": [
                    {
                        "path": change["path"],
                        "event_id": e["id"],
                        "attribution": e["payload"]["attribution"],
                    }
                    for e in observed
                    for change in e["payload"]["changes"]
                ],
                "root_used_result": None,
                "root_used_result_with_benefit": None,
                "use_status": "REVIEW_REQUIRED; completion, delivery and file overlap do not establish use",
            }
        )
    refinements = []
    for event in of("refine_complete"):
        payload = event["payload"]
        applied = payload.get("appliedEdits", [])
        for edit in applied:
            after = edit.get("after") or {}
            visibility = [
                e
                for e in of("execution_input_consumed")
                if e["seq"] > event["seq"]
                and e["session_id"] == root
                and any(
                    item.get("id") == after.get("id")
                    and item.get("version") == after.get("version")
                    and item.get("kind") == after.get("kind")
                    and item.get("scope") == after.get("scope")
                    for item in e["payload"].get("harness_state", [])
                )
            ]
            refinements.append(
                {
                    "edit": edit,
                    "event_id": event["id"],
                    "later_visibility_events": [e["id"] for e in visibility],
                    "later_retrieval": None,
                    "later_behavioral_use": None,
                    "outcome_benefit": None,
                    "status": "APPLIED" if edit.get("applied") else "REJECTED",
                }
            )
    reviews = of("refinement_review")
    boundaries = of("autonomous_gate_boundary")
    identities = [e["payload"]["identity"] for e in boundaries]
    invariants = {}
    for key in ("root_session_id", "kernel_id", "kernel_pid", "workspace", "workspace_inode"):
        observed = [i[key] for i in identities if i[key] is not None]
        invariants[key] = len(set(observed)) == 1 if len(observed) >= 2 else None
    return {
        "claims": claims,
        "children": children,
        "refinement_edits": refinements,
        "refinement_counts": refinement_counts(events, calls),
        "refinement_review_events": [e["id"] for e in reviews],
        "refinement_plan_events": [e["id"] for e in of("refinement_plan")],
        "harness_retrieval_candidates": [
            e["id"]
            for e in events
            if e["type"] == "skill_loaded"
            or (
                e["type"] == "tool_call"
                and any(
                    s in str(e["payload"].get("arguments", {}).get("operation", ""))
                    for s in ("memory", "harness", "skill")
                )
            )
        ],
        "persistence": {
            "invariants": invariants,
            "boundaries_observed": len(boundaries),
            "boundary_event_ids": [e["id"] for e in boundaries],
        },
        "causal_use_status": "UNREVIEWED; unknown is not zero. Raw requests, messages, edits and grader attempts support later review.",
    }


def aggregate(rows, selected):
    if len({r["task_id"] for r in rows}) != len(rows) or any(
        r["task_id"] not in selected for r in rows
    ):
        raise ValueError("Duplicate or unselected task result")
    for row in rows:
        score = row.get("official_score")
        if score is not None and (
            row["status"] != "BENCHMARK_RESULT"
            or type(score) not in {int, float}
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(
                "An official score requires a graded result and a finite value in [0, 1]"
            )
    by_id = {r["task_id"]: r for r in rows}
    scores = [by_id.get(t, {}).get("official_score") for t in selected]
    public_scores = [by_id.get(t, {}).get("public_source_score") for t in selected]
    complete = all(s is not None for s in scores)
    return {
        "selected_tasks": selected,
        "tasks_finished": len(rows),
        "tasks_planned": len(selected),
        "official_mean_score": sum(scores) / len(scores) if complete else None,
        "public_source_mean_score": sum(public_scores) / len(public_scores)
        if all(s is not None for s in public_scores)
        else None,
        "aggregation": "unweighted arithmetic mean of official task rewards; null if any selected task is ungraded",
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "tasks": [
            by_id.get(t, {"task_id": t, "status": "NOT_RUN", "official_score": None})
            for t in selected
        ],
        "leaderboard_comparable": False,
    }


def reviewed_claims(packet, journal, reviews):
    """Validate human causal annotations against the immutable event/request graph.

    A reviewer must justify use. These are explicitly observational judgments;
    delivery, string overlap, and rising scores never create a claim automatically.
    """
    events = {e["id"]: e for e in journal.events}
    requests = {r["id"]: r for r in packet["requests"]}
    calls = {c["id"]: c for c in journal.calls}
    attempts = {a["id"]: a for a in [*journal.attempts, *getattr(journal, "rehearsals", [])]}
    root = packet["after"]["root_session_id"]
    accepted = []
    allowed = {
        "ROOT_USED_RESULT",
        "ROOT_USED_RESULT_WITH_BENEFIT",
        "REFINEMENT_USED",
        "REFINEMENT_USED_WITH_OUTCOME_BENEFIT",
    }
    for review in reviews:
        if (
            review.get("claim") not in allowed
            or not review.get("reviewer")
            or not review.get("evidence")
        ):
            raise ValueError(
                "Causal review requires an allowed claim, reviewer, and behavioral explanation"
            )
        source = events[review["source_event"]]
        action = events[review["root_action_event"]]
        response = events[requests[review["root_request"]]["response_event"]]
        call = calls[review["root_request"]]
        quote = review.get("response_quote", "")
        response_text = json.dumps(call["response"], ensure_ascii=False)
        # Permit a literal excerpt from either response prose or a Python action.
        quoted = (
            quote in call["response"].get("text", "")
            or json.dumps(quote, ensure_ascii=False)[1:-1] in response_text
        )
        if not quote.strip() or not quoted:
            raise ValueError("Causal review quote does not occur in the recorded root response")
        if (
            action["session_id"] != root
            or call["request"]["session_id"] != root
            or not source["seq"] < response["seq"] < action["seq"]
        ):
            raise ValueError("Causal review has invalid root/temporal attribution")
        ancestry, parent = set(), action
        while parent:
            if parent["id"] in ancestry:
                raise ValueError("Cyclic event ancestry")
            ancestry.add(parent["id"])
            parent = events.get(parent.get("parent_event_id"))
        if response["id"] not in ancestry or action["type"] not in {
            "python_execution",
            "python_result",
            "tool_result",
            "evaluation_file_observation",
        }:
            raise ValueError("Causal review action is not linked to the cited root response")
        supporting = [source["id"], response["id"], action["id"]]
        if review["claim"].startswith("ROOT_USED_RESULT"):
            received = events[review["root_received_event"]]
            if (
                source["type"] != "agent_message_sent"
                or source["payload"].get("sender_id") in {None, root}
                or received["type"] != "agent_message_received"
                or received["session_id"] != root
                or received["payload"].get("source_event") != source["id"]
                or received["seq"] >= response["seq"]
            ):
                raise ValueError(
                    "Causal review lacks a received child message preceding the root response"
                )
            supporting.append(received["id"])
        else:
            visible = events[review["visibility_event"]]
            entries = [
                edit["after"]
                for edit in source["payload"].get("appliedEdits", [])
                if edit.get("applied") and edit.get("after")
            ]
            entry = next(
                (
                    e
                    for e in entries
                    if e["id"] == review["entry_id"] and e["version"] == review["entry_version"]
                ),
                None,
            )
            if (
                source["type"] != "refine_complete"
                or entry is None
                or visible["type"] != "execution_input_consumed"
                or visible["session_id"] != root
                or not source["seq"] < visible["seq"] <= action["seq"]
                or not any(
                    all(item.get(k) == entry.get(k) for k in ("id", "version", "kind", "scope"))
                    for item in visible["payload"].get("harness_state", [])
                )
            ):
                raise ValueError(
                    "Refinement review lacks later visibility of the exact persisted entry version"
                )
            supporting.append(visible["id"])
        if "BENEFIT" in review["claim"]:
            before, after = attempts[review["before_attempt"]], attempts[review["after_attempt"]]
            measure = review.get("score_field", "score")
            if measure not in {"score", "public_source_score"} or (
                measure == "public_source_score"
                and any(a.get("verification_mode") != "public_source" for a in (before, after))
            ):
                raise ValueError("Benefit review has an invalid score field")
            positions = {
                e["payload"]["host_attempt_id"]: e["seq"]
                for e in events.values()
                if e["type"] in {"evaluation_verifier_result", "evaluation_rehearsal_result"}
                and e["payload"].get("host_attempt_id")
                and not e["payload"].get("suppressed")
            }
            if (
                before["status"] != "BENCHMARK_RESULT"
                or after["status"] != "BENCHMARK_RESULT"
                or before.get(measure) is None
                or after.get(measure) is None
                or after[measure] <= before[measure]
                or not positions[before["id"]] < action["seq"] < positions[after["id"]]
            ):
                raise ValueError(
                    "Benefit review requires official improvement bracketing the action"
                )
        accepted.append(
            {
                **review,
                "review_id": hashlib.sha256(
                    json.dumps(review, sort_keys=True).encode()
                ).hexdigest(),
                "event_ids": supporting,
                "assessment": "reviewer_assessed_observational; not an ablation or automatic causal inference",
            }
        )
    return accepted


def audit_directory(directory):
    directory = Path(directory)
    packet = json.loads((directory / "trajectory.json").read_text())
    journal = SimpleNamespace(
        events=read_jsonl(directory / "raw-events.jsonl"),
        calls=read_jsonl(directory / "model-calls.jsonl"),
        attempts=read_jsonl(directory / "verifier-attempts.jsonl"),
        rehearsals=read_jsonl(directory / "rehearsal-attempts.jsonl"),
    )
    audit = audit_task(packet, journal)
    review_path = directory / "audit-review.json"
    if review_path.exists():
        claims = reviewed_claims(packet, journal, json.loads(review_path.read_text()))
        audit["reviewed_claims"] = claims
        events = {e["id"]: e for e in journal.events}
        for claim in claims:
            for child in audit["children"]:
                if (
                    claim["claim"].startswith("ROOT_USED_RESULT")
                    and events[claim["source_event"]]["payload"].get("sender_id")
                    == child["child_id"]
                ):
                    child["root_used_result"] = True
                    child["use_status"] = "REVIEWER_ASSESSED"
                    child.setdefault("use_review_ids", []).append(claim["review_id"])
                    if "BENEFIT" in claim["claim"]:
                        child["root_used_result_with_benefit"] = True
            for entry in audit["refinement_edits"]:
                if (
                    claim["claim"].startswith("REFINEMENT_USED")
                    and entry["event_id"] == claim["source_event"]
                    and (entry["edit"].get("after") or {}).get("id") == claim["entry_id"]
                ):
                    entry["later_behavioral_use"] = True
                    entry.setdefault("use_review_ids", []).append(claim["review_id"])
                    if "BENEFIT" in claim["claim"]:
                        entry["outcome_benefit"] = True
        audit["causal_use_status"] = (
            "Reviewer-assessed claims attached; all other use/benefit fields remain unknown"
        )
    save(directory / "task-audit.json", audit)
    return audit


def report_directory(directory):
    from .emulatorbench import pins

    rows = [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*/task-result.json"))]
    result = aggregate(rows, pins()["selected_tasks"])
    save(Path(directory) / "aggregate.json", result)
    return result


def monitor_snapshot(directory):
    from .emulatorbench import pins

    selected = pins()["selected_tasks"]
    lines = ["EMULATORBENCH — 4 canonical tasks"]
    for index, task in enumerate(selected, 1):
        path = Path(directory) / task
        live = json.loads((path / "live.json").read_text()) if (path / "live.json").exists() else {}
        result = (
            json.loads((path / "task-result.json").read_text())
            if (path / "task-result.json").exists()
            else {}
        )
        events = read_jsonl(path / "raw-events.jsonl")
        calls = read_jsonl(path / "model-calls.jsonl")
        totals = accounting([c.get("response", {}).get("usage", {}) for c in calls], 0)
        identity_failures = [e for e in events if e["type"] == "evaluation_trajectory_error"]
        manifest = (
            json.loads((path / "manifest.json").read_text())
            if (path / "manifest.json").exists()
            else {}
        )
        limits = manifest.get("autonomous", {})
        audit_path = path / "task-audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        reviewed = audit.get("reviewed_claims", [])
        useful = (
            str(sum(c["claim"] == "ROOT_USED_RESULT_WITH_BENEFIT" for c in reviewed))
            if reviewed
            else "UNREVIEWED"
        )
        elapsed = 0
        if manifest.get("started"):
            start = datetime.fromisoformat(manifest["started"])
            end = (
                datetime.fromisoformat(manifest["ended"])
                if manifest.get("ended")
                else datetime.now(start.tzinfo)
            )
            elapsed = max(0, int((end - start).total_seconds()))
        admissions = [e for e in events if e["type"] == "rlm_admitted"]
        child_ids = {e["payload"]["child_id"] for e in admissions}
        states = dict.fromkeys(child_ids, "active")
        for e in events:
            if e["session_id"] in states:
                if e["type"] in {"completion", "termination"}:
                    states[e["session_id"]] = (
                        "completed" if e["type"] == "completion" else "stopped"
                    )
                elif e["type"] == "subagent_continued":
                    states[e["session_id"]] = "active"
        refinement = refinement_counts(events, calls)
        boundaries = [e for e in events if e["type"] == "autonomous_gate_boundary"]
        check = boundaries[-1]["payload"].get("check") if boundaries else None
        suppressed = sum(
            (e["payload"].get("check") or {}).get("rerun") is False for e in boundaries
        )
        root_calls = sum(
            c["request"]["session_id"] == live.get("root_id")
            and c["request"].get("metadata", {}).get("purpose", "agent") == "agent"
            for c in calls
        )
        child_calls = sum(
            c["request"].get("parent_id") is not None
            and c["request"].get("metadata", {}).get("purpose", "agent") == "agent"
            for c in calls
        )
        feedback_boundaries = {
            e["payload"]["boundary_event"] for e in events if e["type"] == "evaluation_feedback"
        }
        illegal_feedback = [
            e["id"]
            for e in boundaries
            if e["id"] in feedback_boundaries
            and (
                e["payload"]["turns"] >= limits.get("max_turns", float("inf"))
                or e["payload"]["tokens"] >= limits.get("max_tokens", float("inf"))
                or e["payload"]["continuations"] > limits.get("max_continuations", float("inf"))
            )
        ]
        lines.extend(
            [
                f"{index}/4 {task}  {result.get('status', live.get('outcome', 'NOT_RUN'))}",
                f"  root {live.get('root_id', '—')} | kernel {live.get('kernel_id', '—')} | wall {elapsed // 60}m{elapsed % 60:02}s",
                f"  turns {live.get('root_turns', 0)}/{limits.get('max_turns', '—')} (Prime completion-boundary limit) | calls root {root_calls} child {child_calls}",
                f"  RLM admissions {len(admissions)} | nested {sum(e['payload']['depth'] > 1 for e in admissions)} | active {sum(s == 'active' for s in states.values())} | completed {sum(s == 'completed' for s in states.values())} | reviewed useful results {useful}",
                f"  refinement explicit {refinement['explicit']} | compact reviews {refinement['compact_reviews']} | interval reviews {refinement['interval_reviews']} | declines {refinement['declines']} | empty plans {refinement['empty_plans']} | applied {refinement['applied_edits']}",
                f"  verifier attempts {live.get('verifier_attempts', 0)} | latest score {live.get('latest_score')} | continuations {live.get('continuations', 0)}/{limits.get('max_continuations', '—')}",
                f"  public rehearsals {live.get('rehearsal_attempts', 0)} | latest rehearsal score {live.get('latest_rehearsal_score')}",
                f"  verifier suppressed {suppressed} | last workspace check {'changed / first check' if check and check['rerun'] else 'unchanged' if check else 'pending'}",
                f"  tokens noncached input {totals['input_tokens'] - totals['cached_input_tokens']} | cached {totals['cached_input_tokens']} | output {totals['output_tokens']} | reasoning {totals['reasoning_output_tokens']}",
                f"  root budget tokens {live.get('noncached_budget_tokens', 0)}/{limits.get('max_tokens', '—')}",
            ]
        )
        if identity_failures:
            lines.append(f"  !!! INVARIANT/TRAJECTORY FAILURE: {identity_failures[-1]['payload']}")
        if illegal_feedback:
            lines.append(f"  !!! AUTONOMOUS LIMIT VIOLATION at events {illegal_feedback}")
    return "\n".join(lines)


def monitor(directory):
    try:
        while True:
            print("\033[2J\033[H" + monitor_snapshot(directory), flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        return {"monitor": "stopped; evaluation unaffected"}
