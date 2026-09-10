"""Offline paired evidence report; missing lifecycle coverage remains missing."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .reinforcement_metrics import lifecycle, object_response
from .schema import save


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def prime_lifecycle(directory):
    directory = Path(directory)
    calls = lines(directory / "provider-calls.jsonl")
    requests = lines(directory / "provider-requests.jsonl")
    events = [x for x in lines(directory / "events.jsonl") if x["depth"] == 0]
    root_id = requests[0]["request"]["session_id"] if requests else None
    root_calls = [
        x
        for x in calls
        if x["request"]["session_id"] == root_id and x["request"]["metadata"]["purpose"] == "agent"
    ]
    reviews = [x for x in calls if x["request"]["metadata"]["purpose"] == "refinement_review"]
    review_requests = [
        x for x in requests if x["request"]["metadata"]["purpose"] == "refinement_review"
    ]
    plans = [x for x in requests if x["request"]["metadata"]["purpose"] == "refinement"]
    applied = [x for x in events if x["event"]["type"] == "refine_complete"]
    failures = [x for x in events if x["event"]["type"] == "refine_failed"]
    edits = [e for x in applied for e in x["event"]["result"]["appliedEdits"] if e["applied"]]
    compactions = [
        x
        for x in events
        if x["event"]["type"] == "compaction_end"
        and x["event"].get("result")
        and not x["event"]["aborted"]
    ]
    automatic = []
    for x in review_requests:
        r = x["request"]
        trigger = re.search(
            r"<trigger>\s*(.*?); (\d+) assistant turns", r["messages"][-1]["content"]
        )
        automatic.append(
            {
                "request_id": r["request_id"],
                "root_turn": r["turn"],
                "trigger": {"reason": trigger[1], "turnsSinceLastReview": int(trigger[2])}
                if trigger
                else None,
            }
        )
    explicit = [
        x
        for x in lines(directory / "host-requests.jsonl")
        if x["depth"] == 0 and x["operation"] == "refine.run"
    ]
    for x in explicit:
        before = [c for c in root_calls if c["request"]["turn"] < x["root_turn"]]
        x["preceding_request"] = before[-1]["request"] if before else None
    approvals = sum(
        object_response(x["response"]["text"]).get("shouldRefine") is True for x in reviews
    )
    declines = sum(
        object_response(x["response"]["text"]).get("shouldRefine") is False for x in reviews
    )
    outcomes = []
    if failures:
        outcomes.append("REFINEMENT_FAILED")
    if edits:
        outcomes.append("TRIGGER_APPLIED_EDIT")
    if any(not x["event"]["result"]["appliedEdits"] for x in applied):
        outcomes.append("TRIGGER_APPROVED_EMPTY_PLAN")
    if declines:
        outcomes.append("TRIGGER_REVIEW_DECLINED")
    if not outcomes:
        outcomes.append("TRIGGER_PENDING" if review_requests or explicit else "NO_TRIGGER")
    counts = Counter(x["trigger"]["reason"] for x in automatic if x["trigger"])
    root_tools = [a for x in root_calls for a in x["response"]["actions"] if a["name"] == "ipython"]
    # Production child lifecycle events avoid double-counting printed/recovered handles.
    child_ids = {
        x["event"]["child"]["id"] for x in events if x["event"]["type"] == "rlm_child_update"
    }
    return {
        "root_assistant_turns": sum(
            x["event"]["type"] == "message_end" and x["event"]["message"]["role"] == "assistant"
            for x in events
        ),
        "explicit_refine_calls": len(explicit),
        "compactions": len(compactions),
        "interval_triggers": counts["turn_interval"],
        "compaction_triggers": counts["compact"],
        "automatic_review_requests": len(review_requests),
        "review_approvals": approvals,
        "review_declines": declines,
        "planner_calls": len(plans),
        "non_empty_refinement_proposals": sum(
            bool(object_response(x["response"]["text"]).get("edits"))
            for x in calls
            if x["request"]["metadata"]["purpose"] == "refinement"
        ),
        "refinements_applied": sum(
            any(e["applied"] for e in x["event"]["result"]["appliedEdits"]) for x in applied
        ),
        "typed_edits": len(edits),
        "edits_by_kind": dict(Counter(e["kind"] for e in edits)),
        "REPL": len(root_tools),
        "RLM": len(child_ids),
        "RLM_measurement": "unique direct child admissions in production rlm_child_update events",
        "failures": len(failures),
        "outcomes": outcomes,
        "automatic_reviews": automatic,
        "explicit_calls": explicit,
        "explicit_calls_instrumented": (directory / "host-requests.jsonl").exists(),
        "refinement_events": applied + failures,
        "compaction_events": compactions,
    }


def report(directory):
    directory = Path(directory)
    engines = {}
    if (directory / "buffalo/state/history.sqlite3").exists():
        engines["Buffalo"] = lifecycle(directory / "buffalo")
    if (directory / "prime/provider-requests.jsonl").exists():
        engines["Prime"] = prime_lifecycle(directory / "prime")
    metrics = [
        "root_assistant_turns",
        "REPL",
        "RLM",
        "explicit_refine_calls",
        "compactions",
        "interval_triggers",
        "compaction_triggers",
        "automatic_review_requests",
        "review_approvals",
        "review_declines",
        "planner_calls",
        "non_empty_refinement_proposals",
        "typed_edits",
    ]
    table = ["| Metric | Prime | Buffalo |", "|---|---:|---:|"]
    table += [
        f"| {k} | {engines.get('Prime', {}).get(k, 'not run')} | {engines.get('Buffalo', {}).get(k, 'not run')} |"
        for k in metrics
    ]
    coverage = {
        engine: {
            "interval_review_observed": row["interval_triggers"] > 0,
            "compaction_review_observed": row["compaction_triggers"] > 0,
            "natural_explicit_observed": row["explicit_refine_calls"] > 0,
            "review_decline_observed": row["review_declines"] > 0,
            "empty_plan_observed": "TRIGGER_APPROVED_EMPTY_PLAN" in row["outcomes"],
            "typed_edit_observed": row["typed_edits"] > 0,
        }
        for engine, row in engines.items()
    }
    save(
        directory / "comparison.json",
        {"engines": engines, "coverage": coverage, "canonical_score": False},
    )
    return "\n".join(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(report(parser.parse_args().directory))
