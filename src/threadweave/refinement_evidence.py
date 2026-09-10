"""Task-local evidence for an actionable refinement opportunity."""

import ast
import hashlib
import json
import re

from .semantic_state import work_items


def opportunity(store, sid, instruction_messages=()):
    """Project live failures, not every historical error or ordinary requirement."""
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
    # A declared conflict is a risk signal, not a finding that the root is wrong.
    # Review the first interpretation once; the review model must identify an
    # actual unaddressed mistake before it can approve a planner invocation.
    contract = json.dumps(instruction_messages, ensure_ascii=False)
    decision = None
    if re.search(r"\b(?:conflict\w*|contradict\w*|incompatible)\b", contract, re.I) and any(
        e["type"] == "python_execution" for e in events
    ):
        candidates = {}
        for event in events:
            if event["type"] != "python_execution":
                continue
            code = event["payload"]["code"]
            try:
                parsed = ast.parse(code)
            except SyntaxError:
                continue
            for node in ast.walk(parsed):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node in parsed.body
                ):
                    candidates[node.name] = ast.get_source_segment(code, node)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = re.findall(r"\bdef\s+(\w+)\s*\(", node.value)
                    for name in names:
                        candidates[name] = node.value
        decision = {
            "id": "contract-"
            + hashlib.sha256(
                (contract + json.dumps(candidates, sort_keys=True)).encode()
            ).hexdigest(),
            "status": "unreviewed interpretation, not a confirmed failure",
            "question": "Does the root's interpretation of explicitly competing requirements "
            "discard a compatible requirement or leave an actual contradiction unresolved? "
            "Inspect its decisions/candidate against the original instructions. If consistent, "
            "decline; do not request generic checklists or re-test already established facts.",
            "observed_source_candidates": candidates,
        }
    return {
        "observed_failures": evidence,
        "unresolved_work": unresolved,
        "instruction_decision": decision,
    }


def opportunity_key(value):
    # Stable evidence IDs prevent another review of the same unresolved signal.
    ids = [e["event_id"] for e in value["observed_failures"]]
    ids += [e["source_event"] for e in value["unresolved_work"]]
    if value.get("instruction_decision"):
        ids.append(value["instruction_decision"]["id"])
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest() if ids else None
