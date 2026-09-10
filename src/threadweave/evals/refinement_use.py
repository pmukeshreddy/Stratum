"""Read-only refinement-use evidence; intent is never counted as demonstrated use."""

import ast
import re


def meaningful(code):
    """Exclude cells consisting only of printing, messaging or refinement bookkeeping."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Assert, ast.FunctionDef, ast.AsyncFunctionDef, ast.For, ast.While)
        ):
            return True
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if (
                name in {"exec", "eval", "compile", "bash"}
                or name.startswith(("tests.", "verify.", "edit.", "repo."))
                or name.endswith((".write_text", ".write_bytes"))
            ):
                return True
    return False


def source_candidates(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    candidates = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.search(r"\bdef\s+\w+\(", node.value)
        ):
            value = node.value
            if "\\n" in value and "\n" not in value:
                value = value.replace("\\n", "\n").replace("\\t", "\t")
            try:
                ast.parse(value)
                candidates.append(value)
            except SyntaxError:
                pass
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in tree.body:
            candidates.append(ast.get_source_segment(code, node))
    return candidates


def refinement_use(events, root_inputs, sid, task_id, answer):
    """Export actual before/after evidence. Semantic attribution requires trace review."""
    executions = [e for e in events if e["session_id"] == sid and e["type"] == "python_execution"]
    applies = [
        e for e in events if e["type"] == "refine_complete" and e["payload"].get("appliedEdits")
    ]
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
    final = (fenced[0] if fenced else answer).strip()
    try:
        names = {
            n.name
            for n in ast.parse(final).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    except SyntaxError:
        names = set()
    rows = []
    for event in applies:
        timestamp, payload = event["timestamp"], event["payload"]
        before = [e for e in executions if e["timestamp"] < timestamp]
        after = [e for e in executions if e["timestamp"] > timestamp]
        candidate_evidence = []
        for e in before:
            for candidate in source_candidates(e["payload"]["code"]):
                if any(
                    re.search(r"\bdef\s+" + re.escape(name) + r"\(", candidate) for name in names
                ):
                    candidate_evidence.append({"event_id": e["id"], "source": candidate})
        # Inspectable candidates, not a reconstruction by executing model code.
        candidate = candidate_evidence[-1] if candidate_evidence else None
        later = [r for r in root_inputs if r["started_at"] > timestamp]
        contents = [
            e.get("after", e).get("content", e.get("content", ""))
            for e in payload["appliedEdits"]
            if e.get("action") != "delete"
        ]
        full = bool(contents) and all(
            any(c in str(m.get("content", "")) for r in later for m in r["messages"])
            for c in contents
        )
        observed = payload.get("opportunity", {})
        application = payload.get("application", {})
        meaningful_after = [e for e in after if meaningful(e["payload"]["code"])]
        row = {
            "task_id": task_id,
            "refinement_id": payload["id"],
            "source_event": event["id"],
            "refinement_applied_at": timestamp,
            "meaningful_root_actions_before_refinement": sum(
                meaningful(e["payload"]["code"]) for e in before
            ),
            "meaningful_root_actions_after_refinement": len(meaningful_after),
            "later_tool_calls_after_refinement": len(after),
            "full_or_expanded_entry_delivered": full,
            "refinement_content": contents,
            "unresolved_issue_before_refinement": {
                "observed": observed,
                "planner_proposal": application,
            },
            "whether_later_action_addressed_that_issue": None,
            "whether_final_candidate_changed_after_refinement": candidate["source"].strip() != final
            if candidate
            else None,
            "candidate_before_refinement": candidate,
            "final_candidate": final,
            "later_actions": [
                {
                    "event_id": e["id"],
                    "timestamp": e["timestamp"],
                    "code": e["payload"]["code"],
                    "meaningful": meaningful(e["payload"]["code"]),
                }
                for e in after
            ],
            "later_validation_events": [
                {"event_id": e["id"], "type": e["type"], "payload": e["payload"]}
                for e in events
                if e["timestamp"] > timestamp
                and e["session_id"] == sid
                and e["type"] in {"python_error", "python_result", "verifier_result"}
            ],
            "later_root_inputs": [
                {"request_id": r["request_id"], "root_turn": r["root_turn"]} for r in later
            ],
            "classification": "POST-CORRECTION"
            if application.get("status") == "post_correction"
            else "NO-EFFECT",
            "classification_status": "requires_trajectory_review",
            "classification_basis": "No semantic use inferred from model intent, delivery, or a subsequent tool call alone.",
        }
        rows.append(row)
    return rows


def use_summary(rows):
    uses = [u for r in rows for u in r.get("refinement_use", [])]
    return {
        "applied_refinements": len(uses),
        "tasks_with_meaningful_work_after_refinement": len(
            {u["task_id"] for u in uses if u["meaningful_root_actions_after_refinement"]}
        ),
        "tasks_with_observed_candidate_change": len(
            {u["task_id"] for u in uses if u["whether_final_candidate_changed_after_refinement"]}
        ),
        "full_entries_delivered": sum(u["full_or_expanded_entry_delivered"] for u in uses),
        "classification_pending": sum(
            u["classification_status"] == "requires_trajectory_review" for u in uses
        ),
        "note": "Candidate string differences and post-edit actions are observational. Review linked traces to establish relevant use and whether formatting/docstring differences affect outcomes.",
    }
