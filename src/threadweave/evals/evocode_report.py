"""Evidence accounting, separate from Harbor/EvoCode rewards.

Availability and textual references are deliberately not causal-use claims.
"""

from __future__ import annotations

import json
from collections import Counter


def round_report(task, index, packet, attempts, model_calls):
    events = packet["events"]
    root = packet["after"]["root_session_id"]

    def kinds(kind):
        return [e for e in events if e["type"] == kind]

    responses = kinds("model_response")
    calls = [c for c in model_calls if c["round"] == packet["round"]]
    reviews = kinds("refinement_review")
    review_calls = [
        c for c in calls if c["request"].get("metadata", {}).get("purpose") == "refinement_review"
    ]
    refinements = kinds("refine_complete")
    edits = [
        {"event_id": e["id"], "event_seq": e["seq"], "refinement_id": e["payload"]["id"], **edit}
        for e in refinements
        for edit in e["payload"]["appliedEdits"]
        if edit["applied"]
    ]
    counts = Counter(e["kind"] for e in edits)
    admissions = [e["payload"] for e in kinds("subagent_created")]
    child_messages = [
        {"event_id": e["id"], **e["payload"]}
        for e in kinds("agent_message_sent")
        if e["payload"]["recipient_id"] == root and e["payload"]["sender_id"] != root
    ]
    child_calls = [c for c in calls if c["request"]["parent_id"]]
    boundaries = sorted(
        [(c["started"], 1) for c in child_calls] + [(c["ended"], -1) for c in child_calls]
    )
    active = peak = 0
    for _, delta in boundaries:
        active += delta
        peak = max(peak, active)
    response_positions = {e["id"]: e["seq"] for e in responses}
    request_positions = {
        r["id"]: response_positions.get(r["response_event"], -1) for r in packet["requests"]
    }
    visibility = [
        {
            "request_id": c["request"]["request_id"],
            "turn": c["request"]["turn"],
            "response_event_seq": request_positions.get(c["request"]["request_id"], -1),
            "inputs": c["request"].get("metadata", {}).get("execution_inputs", {}),
        }
        for c in calls
        if c["request"]["session_id"] == root
        and c["request"].get("metadata", {}).get("purpose", "agent") == "agent"
    ]
    return {
        "task": task,
        "round": index,
        "round_name": packet["round"],
        "root_session_id": root,
        "agent_turns": sum(
            e["payload"].get("metadata", {}).get("purpose", "agent") == "agent" for e in responses
        ),
        "verifier_attempts": len(attempts),
        "gate_checks": len(packet["autonomous"]["checks"]),
        "final_round_reward": attempts[-1]["rewards"] if attempts else None,
        "case_summary": attempts[-1].get("case_summary") if attempts else None,
        "repl_executions": len(kinds("python_execution")),
        "rlm": {
            "children_spawned": len(admissions),
            "admissions": admissions,
            "topology": [
                {k: s[k] for k in ("id", "name", "parent_id", "root_id", "kernel_id")}
                for s in packet["after"]["children"]
            ],
            "child_model_calls": len(child_calls),
            "messages_returned": len(child_messages),
            "messages": child_messages,
            "overlapping_child_calls_peak": peak,
            "outputs_visible": [v for v in visibility if v["inputs"].get("child_evidence")],
            "child_outputs_used": None,
            "use_evidence_status": "Requires trajectory review; visibility is not use",
            "child_created_artifacts": [
                a
                for a in packet.get("artifacts", [])
                if a["session_id"] != root and a["source_event"] in {e["id"] for e in events}
            ],
            "child_workspace_artifact_evidence": [
                e
                for e in events
                if e["session_id"] != root
                and (
                    "artifact" in e["type"] or e["type"] in {"python_execution", "execution_result"}
                )
            ],
            "registry_operations": [
                e
                for e in events
                if e["type"] in {"subagent_deleted", "subagent_continued", "kernel_recovery"}
                or (
                    e["type"] == "tool_call"
                    and e["payload"].get("arguments", {}).get("operation")
                    in {"rlm.list_subagents", "rlm.delete_subagent"}
                )
            ],
        },
        "refinement": {
            "explicit_refine_calls": sum(
                e["payload"].get("arguments", {}).get("operation") == "refine.run"
                for e in kinds("tool_call")
            ),
            "scheduled_refine_calls": len(kinds("refine_scheduled")),
            "interval_triggers": sum(
                "<trigger>\nturn_interval;"
                in json.dumps(c["request"]["messages"], ensure_ascii=False).replace("\\n", "\n")
                for c in review_calls
            ),
            "compaction_triggers": len(kinds("context_compaction")),
            "automatic_reviews": len(review_calls),
            "review_approvals": sum(e["payload"]["shouldRefine"] for e in reviews),
            "review_declines": sum(not e["payload"]["shouldRefine"] for e in reviews),
            "planner_calls": sum(
                c["request"].get("metadata", {}).get("purpose") == "refinement" for c in calls
            ),
            "refinements_applied": sum(
                any(x["applied"] for x in e["payload"]["appliedEdits"]) for e in refinements
            ),
            "typed_edits": edits,
            "kinds": {k: counts[k] for k in ("prompt", "memory", "skill", "subagent")},
            "review_events": reviews,
        },
        "learned_state_used_later": [],
        "harness_visibility": visibility,
        "before_learning_evidence": [
            {
                "event_id": e["id"],
                "seq": e["seq"],
                "type": e["type"],
                "payload": e["payload"]
                if e["type"] == "autonomous_gate_boundary"
                else {"text": e["payload"].get("text")},
            }
            for e in events
            if e["type"] in {"autonomous_gate_boundary", "model_response"}
        ],
        "persistence": {"before": packet["before"], "after": packet["after"]},
        "autonomous": packet["autonomous"],
        "cumulative_usage": packet["usage"],
    }


def task_report(
    task, declared_rounds, rounds, calls, wall_seconds, harbor_reward, *, official_steps=None
):
    learned = []
    for row in rounds:
        for edit in row["refinement"]["typed_edits"]:
            after = edit.get("after")
            evidence = {
                "learning_round": row["round"],
                "edit": edit,
                "before": {
                    "round": row["round"],
                    "events": [
                        e for e in row["before_learning_evidence"] if e["seq"] < edit["event_seq"]
                    ],
                    "earlier_round_rewards": [
                        r["final_round_reward"] for r in rounds if r["round"] < row["round"]
                    ],
                },
                "first_later_visibility": None,
                "use": None,
                "outcome": None,
                "causal_improvement_demonstrated": False,
            }
            if after:
                for later in rounds:
                    if later["round"] < row["round"]:
                        continue
                    visible = next(
                        (
                            v
                            for v in later["harness_visibility"]
                            if v["response_event_seq"] > edit["event_seq"]
                            and any(
                                e["id"] == after["id"]
                                and e["kind"] == after["kind"]
                                and e["scope"] == after["scope"]
                                and e["version"] == after["version"]
                                for e in v["inputs"].get("harness_state", [])
                            )
                        ),
                        None,
                    )
                    if visible:
                        evidence["first_later_visibility"] = {"round": later["round"], **visible}
                        evidence["outcome"] = {
                            "round": later["round"],
                            "reward": later["final_round_reward"],
                        }
                        break
                references = []
                for call in calls:
                    if call["round_index"] <= row["round"] or call["request"]["parent_id"]:
                        continue
                    response = call.get("response", {})
                    text = json.dumps(
                        {"text": response.get("text"), "actions": response.get("actions")}
                    )
                    if after["id"] in text or after["title"] in text:
                        references.append(
                            {
                                "round": call["round_index"],
                                "request_id": call["request"]["request_id"],
                                "status": "Explicit textual reference; requires behavioral review",
                            }
                        )
                evidence["reference_candidates"] = references
            learned.append(evidence)
    official_steps = (
        official_steps
        if official_steps is not None
        else [{"round_name": r["round_name"], "rewards": r["final_round_reward"]} for r in rounds]
    )
    passed = sum((r["rewards"] or {}).get("reward") == 1 for r in official_steps)
    cases = [r["case_summary"] for r in rounds if r["case_summary"]]
    return {
        "protocol": "EvoCode-Bench + Prime-style autonomous feedback",
        "leaderboard_comparable": False,
        "task": task,
        "rounds_passed": passed,
        "rounds_total": declared_rounds,
        "rounds_reached": len(official_steps),
        "rounds_with_instrumentation": len(rounds),
        "official_step_rewards": official_steps,
        "evocode_task_score": passed / declared_rounds,
        "harbor_aggregate_reward": harbor_reward,
        "case_score": sum(
            c["success_count"] / c["total_cases"] if c["total_cases"] else 0 for c in cases
        )
        / declared_rounds
        if cases
        else None,
        "total_model_turns": len(calls),
        "total_usage": rounds[-1]["cumulative_usage"] if rounds else None,
        "host_observed_tokens": {
            key: sum(c.get("response", {}).get("usage", {}).get(key, 0) for c in calls)
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
        },
        "wall_seconds": wall_seconds,
        "rounds": rounds,
        "learned_entries": learned,
        "rlm_activity": {
            "children_spawned": sum(r["rlm"]["children_spawned"] for r in rounds),
            "messages_returned": sum(r["rlm"]["messages_returned"] for r in rounds),
        },
        "refinement_activity": {
            "refinements_applied": sum(r["refinement"]["refinements_applied"] for r in rounds),
            "typed_edits": sum(len(r["refinement"]["typed_edits"]) for r in rounds),
        },
        "claims": "Applied edits and visibility are measured. Behavioral use and causal benefit require review.",
    }
