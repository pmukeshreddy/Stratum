"""Descriptive activity accounting. Optional mechanism use never determines task validity."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from threadweave.file_store import FileStore


def activity(directory, record=None):
    directory, record = Path(directory), record or {}
    with closing(FileStore(directory / "state")) as records:
        events = [
            {**dict(r), "payload": r["payload"]}
            for r in records.select("events", order=(("seq", False),))
        ]
        sessions = [r["body"] for r in records.select("sessions")]
        requests = [dict(r) for r in records.select("model_requests")]
        actions = [dict(r) for r in records.select("actions")]
    root = next(s for s in sessions if s["parent_id"] is None)
    sid = root["id"]

    def of(kind, root_only=False):
        return [
            e for e in events if e["type"] == kind and (not root_only or e["session_id"] == sid)
        ]

    completed = [r for r in requests if r["status"] == "completed"]
    agent = [r for r in completed if r["purpose"] == "agent"]
    reviews = of("refinement_review", True)
    applied = of("refine_complete", True)
    inputs = of("execution_input_consumed", True)
    changes = [
        edit
        for event in of("refine_complete")
        for edit in event["payload"]["appliedEdits"]
        if edit["applied"]
    ]
    shown = {(v["id"], v["version"]) for e in inputs for v in e["payload"]["harness_state"]}
    child_ids = {s["id"] for s in sessions if s["parent_id"]}
    active, peak = set(), 0
    for event in events:
        if event["type"] == "subagent_created":
            active.add(event["payload"]["child_id"])
        elif event["session_id"] in child_ids:
            if event["type"] in {"completion", "termination"}:
                active.discard(event["session_id"])
            elif event["type"] == "subagent_continued":
                active.add(event["session_id"])
        peak = max(peak, len(active))
    failed_verifiers = {e["id"] for e in of("verifier_result") if not e["payload"].get("passed")}
    row = {
        "task_id": str(record.get("task_id", directory.name)),
        "root_model_turns": sum(r["session_id"] == sid for r in agent),
        "all_model_turns": len(completed),
        "all_agent_turns": len(agent),
        "repl_calls": sum(a["name"] == "ipython" for a in actions),
        "python_executions": len(of("python_execution")),
        "tool_calls": len(actions),
        "rlm_calls": sum(
            a["name"] in {"rlm", "agent_spawn"}
            or a["name"] == "host_request"
            and a["arguments"].get("operation") == "rlm.run"
            for a in actions
        ),
        "rlm_admissions": len(of("rlm_admitted")),
        "subagents_created": sum(bool(s["parent_id"]) for s in sessions),
        "parallel_subagents_peak": peak,
        "max_recursive_depth": max(s["depth"] for s in sessions),
        "child_model_turns": sum(r["session_id"] != sid for r in agent),
        "child_completed": sum(
            bool(s["parent_id"]) and s["outcome"] == "completed" for s in sessions
        ),
        "child_failed": sum(
            bool(s["parent_id"]) and s["outcome"] in {"failed", "limited", "cancelled"}
            for s in sessions
        ),
        "agent_messages": sum(
            e["payload"].get("sender_id") not in {None, e["payload"].get("recipient_id")}
            for e in of("agent_message_sent")
        ),
        "agent_messages_received": sum(
            e["payload"].get("sender_id") not in {None, e["payload"].get("recipient_id")}
            for e in of("agent_message_received")
        ),
        "child_observations": len(of("child_observation")),
        "child_results_consumed": len(
            {e["payload"]["child_id"] for e in of("child_evidence_used", True)}
        ),
        "child_evidence_in_root_invocations": sum(
            bool(e["payload"]["child_evidence"]) for e in inputs
        ),
        "verification_attempts": len(of("verifier_result")),
        "verification_failures": len(failed_verifiers),
        "refinement_reviews": len(reviews),
        "refinement_declines": sum(not e["payload"]["shouldRefine"] for e in reviews),
        "refinement_planner_calls": sum(r["purpose"] == "refinement" for r in completed),
        "refinement_approvals": sum(e["payload"]["shouldRefine"] for e in reviews),
        "refinements_applied": sum(
            any(edit["applied"] for edit in e["payload"]["appliedEdits"]) for e in applied
        ),
        "refinement_applied_edits": sum(
            sum(edit["applied"] for edit in e["payload"]["appliedEdits"]) for e in applied
        ),
        "harness_entries_in_root_invocations": len(shown),
        "skills_loaded": len(of("skill_loaded")),
        "compactions": len(of("context_compaction")),
        "wall_seconds": record.get("wall_time_seconds"),
        **{
            k: record.get(k)
            for k in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "functional_pass",
                "style_pass",
                "overall_pass",
            )
        },
        "stop_reason": root["outcome"],
        "child_use_provenance": [e["payload"] for e in of("child_evidence_used", True)],
    }
    for kind, prefix in (
        ("memory", "memory"),
        ("prompt", "prompts"),
        ("skill", "skills"),
        ("subagent", "subagents"),
    ):
        edits = [e for e in changes if e["kind"] == kind]
        row[prefix + "_created"] = sum(e["action"] == "create" for e in edits)
        row[prefix + "_updated"] = sum(e["action"] == "update" for e in edits)
        row[prefix + "_deleted"] = sum(e["action"] == "delete" for e in edits)
        row[prefix + "_in_system_prompt"] = sum(
            v["kind"] == kind for e in inputs for v in e["payload"]["harness_state"]
        )
    return row


def aggregate(rows):
    keys = {k for row in rows for k, v in row.items() if type(v) is int}
    keys.update({"input_tokens", "output_tokens", "total_tokens", "wall_seconds"})
    values = {k: [r.get(k, 0) for r in rows] for k in sorted(keys)}
    peaks = {"parallel_subagents_peak", "max_recursive_depth"}
    return {
        "totals": {
            k: (max(v, default=0) if k in peaks else sum(v)) if None not in v else None
            for k, v in values.items()
        },
        "tasks_using": {k: sum(n is not None and n > 0 for n in v) for k, v in values.items()},
    }
