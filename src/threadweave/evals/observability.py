"""Read-only projections of committed traces; never used to steer an agent."""

from __future__ import annotations

import ast
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from .activity import activity


def operations(code):
    """Syntactic descriptions, not model judgments about intent or correctness."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["unparsed Python/IPython; see code excerpt"]
    labels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            labels.add("assertions")
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            labels.add("iteration")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            labels.add("imports")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            labels.add("definitions")
        elif isinstance(node, ast.Call):
            labels.add("call:" + ast.unparse(node.func)[:100])
    return sorted(labels)


def task_observability(directory, record):
    directory = Path(directory)
    with sqlite3.connect(f"file:{directory / 'state/history.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        events = [dict(r) for r in db.execute("SELECT * FROM events ORDER BY seq")]
        for e in events:
            e["payload"] = json.loads(e["payload"])
        sessions = [json.loads(r[0]) for r in db.execute("SELECT body FROM sessions")]
        requests = [dict(r) for r in db.execute("SELECT * FROM model_requests ORDER BY started_at")]
        attempts = [dict(r) for r in db.execute("SELECT * FROM model_attempts")]
        actions = [dict(r) for r in db.execute("SELECT * FROM actions")]
        messages = [dict(r) for r in db.execute("SELECT * FROM messages")]
        configs = {r[0]: json.loads(r[1]) for r in db.execute("SELECT id,body FROM configs")}
        paths = dict(db.execute("SELECT id,path FROM artifacts"))
    for action in actions:
        action["arguments"] = json.loads(action["arguments"])
    root = next(s for s in sessions if s["parent_id"] is None)
    sid, task_id = root["id"], str(record["task_id"])
    children = {s["id"]: s for s in sessions if s["parent_id"] is not None}
    by_event = {e["id"]: e for e in events}
    by_request = {r["id"]: r for r in requests}
    bodies = {
        r["id"]: json.loads((directory / "state" / paths[r["body_artifact"]]).read_text())
        for r in requests
    }
    response_requests = {r["response_event"]: r for r in requests if r["response_event"]}
    starts = {e["id"]: e for e in events if e["type"] == "model_invocation_started"}
    root_inputs = []
    for r in requests:
        if r["session_id"] != sid or r["purpose"] != "agent":
            continue
        body = bodies[r["id"]]
        root_inputs.append(
            {
                "request_id": r["id"],
                "root_turn": body["turn"] + 1,
                "started_at": r["started_at"],
                "status": r["status"],
                "messages": body["messages"],
                "evidence": body.get("metadata", {})
                .get("execution_inputs", {})
                .get("child_evidence", {}),
            }
        )

    def of(kind, session=None):
        return [
            e
            for e in events
            if e["type"] == kind and (session is None or e["session_id"] == session)
        ]

    def origin(event):
        seen = set()
        while event and event["id"] not in seen:
            seen.add(event["id"])
            request = response_requests.get(event["id"])
            if request is None and event["id"] in starts:
                request = by_request.get(event["payload"]["request_id"])
            if request:
                return request
            event = by_event.get(event["parent_event_id"])

    def invocation_ref(r):
        return {k: r[k] for k in ("request_id", "root_turn", "status")}

    def root_turn(event):
        request = origin(event)
        while request and request["session_id"] in children:
            request = by_request.get(children[request["session_id"]].get("spawned_by_request_id"))
        if request and request["session_id"] == sid:
            return bodies[request["id"]]["turn"] + 1
        later = [r for r in root_inputs if r["started_at"] <= event["timestamp"]]
        return later[-1]["root_turn"] if later else None

    def child_seen(cid):
        return [invocation_ref(r) for r in root_inputs if cid in r["evidence"]]

    rlm_actions = [
        a
        for a in actions
        if a["name"] == "host_request" and a["arguments"].get("operation") == "rlm.run"
    ]
    calls = []
    admissions = of("rlm_admitted")
    for a in rlm_actions:
        source = by_event[a["source_event"]]
        admitted = next((e for e in admissions if e["parent_event_id"] == a["source_event"]), None)
        cid = admitted["payload"]["child_id"] if admitted else None
        child = children.get(cid)
        options = a["arguments"].get("payload", {})
        # The actual host request is retained even on a rejected admission.
        prompt = admitted["payload"]["assignment"] if admitted else options.get("prompt")
        cfg = configs[child["config_id"]] if child else {}
        purpose = of("child_purpose", cid) if cid else []
        delivered = (
            [
                m
                for m in messages
                if m["sender_id"] == cid
                and m["recipient_id"] == sid
                and m["received_at"] is not None
            ]
            if cid
            else []
        )
        observed = (
            [e for e in of("child_observation", sid) if e["payload"].get("child_id") == cid]
            if cid
            else []
        )
        followups = (
            [
                m
                for m in messages
                if m["recipient_id"] == cid and m["sender_id"] == child["parent_id"]
            ]
            if child
            else []
        )
        terminal = (
            [
                e
                for e in events
                if e["session_id"] == cid and e["type"] in {"completion", "termination"}
            ]
            if cid
            else []
        )
        calls.append(
            {
                "task_id": task_id,
                "action_id": a["id"],
                "source_event": source["id"],
                "root_turn": root_turn(source),
                "caller_session_id": a["session_id"],
                "child_id": cid,
                "depth": child["depth"] if child else None,
                "prompt_excerpt": (prompt or "")[:1000],
                "actual_request": a["arguments"],
                "model": cfg.get("provider", {}).get("model"),
                "thinking_level": cfg.get("provider", {})
                .get("parameters", {})
                .get("reasoning_effort"),
                "workspace_profile": purpose[0]["payload"] if purpose else None,
                "spawn_timestamp": admitted["timestamp"] if admitted else None,
                "completion_timestamp": terminal[-1]["timestamp"] if terminal else None,
                "result_status": child["outcome"] if child else a["status"],
                "evidence_reached_root": bool(delivered or observed),
                "evidence_source_events": sorted(
                    {m["source_event"] for m in delivered}
                    | {v for e in observed for v in e["payload"].get("source_events", [])}
                ),
                "later_root_invocations": child_seen(cid) if cid else [],
                "explicit_consumption_events": [
                    e["id"]
                    for e in of("child_evidence_used", sid)
                    if e["payload"].get("child_id") == cid
                ],
                "followups": [
                    {
                        "message_id": m["id"],
                        "timestamp": m["created_at"],
                        "received_at": m["received_at"],
                        "body_excerpt": m["body"][:600],
                    }
                    for m in followups
                ],
                "reuse_events": [e["id"] for e in of("subagent_continued", cid)] if cid else [],
                "lifecycle": [
                    {"event_id": e["id"], "type": e["type"], "timestamp": e["timestamp"]}
                    for e in events
                    if e["session_id"] == cid
                    and e["type"] in {"completion", "termination", "subagent_continued"}
                ],
            }
        )

    repl_trace = []
    for event in of("python_execution"):
        # Trace ancestry identifies the model's outer tool action, including Python
        # invoked through an alias. A tool receipt in a later input proves visibility.
        ancestry, current = set(), event
        while current and current["id"] not in ancestry:
            ancestry.add(current["id"])
            current = by_event.get(current["parent_event_id"])
        outer = [
            a
            for a in actions
            if a["session_id"] == event["session_id"]
            and (a["source_event"] in ancestry or a["id"] == event["payload"]["execution_id"])
        ]
        action_ids = {a["id"] for a in outer}
        output_artifacts = {
            json.loads(a["result"]).get("artifact_id") for a in outer if a["result"]
        }
        output_artifacts.discard(None)
        seen = [
            invocation_ref(r)
            for r in root_inputs
            if r["started_at"] > event["timestamp"]
            and any(
                m.get("role") == "tool"
                and (
                    m.get("tool_call_id") in action_ids
                    or any(aid in str(m.get("content", "")) for aid in output_artifacts)
                )
                for m in r["messages"]
            )
        ]
        repl_trace.append(
            {
                "event_id": event["id"],
                "session_id": event["session_id"],
                "root_turn": root_turn(event),
                "timestamp": event["timestamp"],
                "execution_id": event["payload"]["execution_id"],
                "operations": operations(event["payload"]["code"]),
                "code_excerpt": event["payload"]["code"][:1200],
                "later_root_invocations_receiving_output": seen,
                "child_evidence_in_later_root_inputs": child_seen(event["session_id"])
                if event["session_id"] != sid
                else [],
            }
        )

    refinement_events = []
    for event in events:
        if event["type"] not in {
            "refine_scheduled",
            "refinement_review",
            "refine_complete",
            "refine_failed",
            "refinement_notice",
            "refinement",
        }:
            continue
        payload = event["payload"]
        visible = (
            [
                invocation_ref(r)
                for r in root_inputs
                if r["started_at"] > event["timestamp"]
                and payload.get("content")
                and any(payload["content"] in str(m.get("content", "")) for m in r["messages"])
            ]
            if event["type"] == "refinement_notice"
            else []
        )
        refinement_events.append(
            {
                "task_id": task_id,
                "root_turn": root_turn(event),
                "event_id": event["id"],
                "session_id": event["session_id"],
                "type": event["type"],
                "timestamp": event["timestamp"],
                "payload": payload,
                "later_root_invocations_receiving_notice": visible,
            }
        )

    old = activity(directory, record)
    counts = Counter(e["type"] for e in events)
    reviews = of("refinement_review")
    edits = [
        edit
        for e in of("refine_complete")
        for edit in e["payload"].get("appliedEdits", [])
        if edit.get("applied")
    ]
    edit_kinds = Counter(e["kind"] for e in edits)
    explicit = [
        a
        for a in actions
        if a["name"] == "host_request" and a["arguments"].get("operation") == "refine.run"
    ]
    restorations = [
        json.loads(line)
        for p in (directory / "state/kernels").glob("*/restore-events.jsonl")
        for line in p.read_text().splitlines()
    ]
    recovered = [
        e
        for e in of("kernel_recovery")
        if e["payload"].get("restored") or e["payload"].get("reconstructed")
    ]
    request_sessions = {r["id"]: r["session_id"] for r in requests}
    root_repl = [e for e in repl_trace if e["session_id"] == sid]
    failed_attempts = [a for a in attempts if a["status"] == "failed"]
    timeout_events = [
        e["id"]
        for e in events
        if e["type"] in {"failure", "termination", "python_error"}
        and any(
            term in json.dumps(e["payload"]).lower()
            for term in ("timeout", "timed out", "wall_seconds", "wall time")
        )
    ]
    notice_inputs = {
        r["request_id"]
        for e in refinement_events
        for r in e["later_root_invocations_receiving_notice"]
    }
    detail = {
        "task_id": task_id,
        "score": {k: bool(record[k + "_pass"]) for k in ("functional", "style", "overall")},
        "root": {
            "root_turns": root["turns"],
            "model_calls": sum(request_sessions[a["request_id"]] == sid for a in attempts),
            "agent_model_calls": sum(
                request_sessions[a["request_id"]] == sid
                and by_request[a["request_id"]]["purpose"] == "agent"
                for a in attempts
            ),
        },
        "REPL": {
            "entered_repl": bool(root_repl),
            "any_session_entered_repl": bool(repl_trace),
            "python_executions": len(repl_trace),
            "root_python_executions": len(root_repl),
            "repl_entries": len(root_repl),
            "first_root_turn": root_repl[0]["root_turn"] if root_repl else None,
            "repl_tool_actions": old["repl_calls"],
            "all_tool_actions": len(actions),
            "operations": sorted({op for e in repl_trace for op in e["operations"]}),
            "executions_with_output_seen_by_root": sum(
                bool(e["later_root_invocations_receiving_output"]) for e in repl_trace
            ),
        },
        "RLM": {
            "rlm_calls": len(rlm_actions),
            "children_spawned": len(children),
            "max_rlm_depth": old["max_recursive_depth"],
            "child_model_calls": sum(request_sessions[a["request_id"]] != sid for a in attempts),
            "child_repl_executions": len(repl_trace) - len(root_repl),
            "child_completed": sum(s["outcome"] == "completed" for s in children.values()),
            "child_failed": sum(s["outcome"] in {"failed", "limited"} for s in children.values()),
            "child_cancelled": sum(s["outcome"] == "cancelled" for s in children.values()),
            "child_followups": sum(len(c["followups"]) for c in calls),
            "child_reuses": counts["subagent_continued"],
        },
        "child_evidence": {
            "evidence_delivered_to_root": sum(c["evidence_reached_root"] for c in calls),
            "children_seen_in_later_root_inputs": sum(
                bool(c["later_root_invocations"]) for c in calls
            ),
            "later_root_inputs_containing_child_evidence": sum(
                bool(r["evidence"]) for r in root_inputs
            ),
            "explicit_evidence_consumption_events": len(of("child_evidence_used", sid)),
        },
        "continual_harness": {
            "explicit_refine_calls": len(explicit),
            "automatic_refine_reviews": sum(r["purpose"] == "refinement_review" for r in requests),
            "review_trigger": dict(
                Counter(
                    "compaction"
                    if e["payload"].get("reason") == "compact"
                    else e["payload"].get("reason", "legacy_unspecified")
                    for e in reviews
                )
            ),
            "reviews_approved": sum(e["payload"].get("shouldRefine") is True for e in reviews),
            "reviews_declined": sum(e["payload"].get("shouldRefine") is False for e in reviews),
            "refinements_applied": sum(
                any(x.get("applied") for x in e["payload"].get("appliedEdits", []))
                for e in of("refine_complete")
            ),
            "applied_edits": len(edits),
            **{k + "_edits": edit_kinds[k] for k in ("prompt", "memory", "skill", "subagent")},
            "later_root_inputs_receiving_refinement_notice": len(notice_inputs),
        },
        "compaction": {
            "L1_compactions": counts["context_compaction"],
            "L2_checkpoints": sum(
                not e["payload"].get("commit_failed", False) for e in of("kernel_snapshot")
            ),
            "state_offloads": sum(
                len(e["payload"]["names"]) for e in of("kernel_variables_offloaded")
            ),
            "state_prunes": sum(len(e["payload"]["names"]) for e in of("kernel_variables_pruned")),
            "restore_events": len(restorations) + len(recovered),
            "explicit_value_restores": len(restorations),
            "kernel_recoveries_with_state": len(recovered),
        },
        "verification": {
            "level1_checks": counts["verification_result"],
            "level2_checks": counts["verification_targeted"],
            "level3_checks": counts["verifier_started"],
            "verification_receipt_reuses": counts["verification_receipt_reused"],
            "verification_receipt_invalidations": counts["verification_receipt_invalidated"],
        },
        "execution": {
            "wall_seconds": record["wall_time_seconds"],
            "timeouts": len(timeout_events),
            "timeout_events": timeout_events,
            "retries": counts["retry"],
            "task_attempts": 1,
            "failures": counts["failure"],
            "failed_model_attempts": len(failed_attempts),
            "python_errors": counts["python_error"],
            "task_failed": record.get("stop_reason") != "completed",
            "stop_reason": record.get("stop_reason"),
        },
        "raw_trace_directory": str(directory),
    }
    from .refinement_use import refinement_use

    detail["refinement_use"] = refinement_use(
        events,
        root_inputs,
        sid,
        task_id,
        (directory / "answer.txt").read_text() if (directory / "answer.txt").exists() else "",
    )
    return (
        detail,
        {
            "task_id": task_id,
            "metrics": detail,
            "repl_trace": repl_trace,
            "tool_actions": actions,
            "root_invocations": [
                {k: v for k, v in r.items() if k != "messages"} for r in root_inputs
            ],
            "event_counts": dict(counts),
            "restore_events": restorations,
        },
        calls,
        refinement_events,
    )


def summarize(rows, previous, elapsed):
    from .refinement_use import use_summary

    totals = {}
    for section in (
        "root",
        "REPL",
        "RLM",
        "child_evidence",
        "continual_harness",
        "compaction",
        "verification",
        "execution",
    ):
        values = {}
        for k in rows[0][section]:
            v = [r[section][k] for r in rows]
            if all(type(x) in (int, float, bool) for x in v):
                values[k] = max(v) if k == "max_rlm_depth" else sum(v)
        totals[section] = values
    totals["continual_harness"]["review_trigger"] = dict(
        sum((Counter(r["continual_harness"]["review_trigger"]) for r in rows), Counter())
    )
    seconds = sorted(r["execution"]["wall_seconds"] for r in rows)
    old = {str(r["task_id"]): r for r in previous}
    cohorts = {}
    for name, predicate in {
        "REPL": lambda r: r["REPL"]["entered_repl"],
        "RLM": lambda r: r["RLM"]["children_spawned"] > 0,
        "refinement": lambda r: r["continual_harness"]["refinements_applied"] > 0,
    }.items():
        cohorts[name] = {}
        for used in (True, False):
            selected = [r for r in rows if bool(predicate(r)) == used]
            cohorts[name]["used" if used else "unused"] = {
                "tasks": len(selected),
                "new_overall_pass": sum(r["score"]["overall"] for r in selected),
                "previous_overall_pass": sum(old[r["task_id"]]["overall_pass"] for r in selected),
                "gains": [
                    r["task_id"]
                    for r in selected
                    if r["score"]["overall"] and not old[r["task_id"]]["overall_pass"]
                ],
                "regressions": [
                    r["task_id"]
                    for r in selected
                    if not r["score"]["overall"] and old[r["task_id"]]["overall_pass"]
                ],
                "task_ids": [r["task_id"] for r in selected],
            }
    return {
        "totals": totals,
        "refinement_use": use_summary(rows),
        "tasks_spawning_children": sum(r["RLM"]["children_spawned"] > 0 for r in rows),
        "runtime": {
            "median_task_seconds": statistics.median(seconds),
            "p95_task_seconds": seconds[math.ceil(0.95 * len(seconds)) - 1],
            "max_task_seconds": max(seconds),
            "total_elapsed_seconds": elapsed,
            "p95_method": "nearest rank",
        },
        "observational_cohorts": cohorts,
        "definitions": {
            "root_turn": "1-based model request turn; root_turns is committed session turns",
            "model_calls": "transport attempts, including auxiliary calls; agent_model_calls excludes auxiliaries",
            "REPL_entries": "root Python execution events, not worker startup",
            "evidence_delivered": "unique children with received messages or explicit child observations at root",
            "evidence_seen": "source references retained in actual later root request bodies; no claim of reasoning use",
            "explicit_consumption": "only existing child_evidence_used events",
            "refinements_applied": "successful refinement runs with one or more applied edits; edit counts separate",
            "restore_events": "successful explicit value loads plus worker recoveries with restored state",
            "timeouts": "recorded timeout failure/termination/Python-error events; multiple events can concern one task",
            "cohorts": "descriptive association on this fixed set, not causal attribution",
        },
    }
