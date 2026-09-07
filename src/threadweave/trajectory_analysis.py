"""Descriptive trajectory signals. Repetition is not proof that an action was useless."""

from collections import Counter

from .storage import encode


def analyze_events(events):
    repeated, failed, searches, reads, epochs = [], [], Counter(), Counter(), 0
    seen = {}
    retrievals, compactions, child_events = [], [], []
    for event in events:
        kind, body = event["type"], event["payload"]
        if kind in {"code_edit", "workspace_effects"} and body.get("files"):
            epochs += 1
        if kind == "tool_call":
            name, arguments = body["name"], body["arguments"]
            signature = encode([name, arguments, epochs])
            if signature in seen:
                repeated.append(
                    {
                        "first_event": seen[signature],
                        "repeated_event": event["id"],
                        "tool": name,
                        "workspace_epoch": epochs,
                    }
                )
            seen[signature] = event["id"]
            if "search" in name or name.startswith("repo_"):
                searches[encode(arguments)] += 1
            if name in {"workspace_read", "file_outline"}:
                reads[encode(arguments)] += 1
        if kind in {"tool_result", "python_error"} and (
            body.get("error") or body.get("result", {}).get("error")
        ):
            failed.append(
                {"event_id": event["id"], "error": body.get("error") or body["result"]["error"]}
            )
        if kind == "history_retrieval":
            retrievals.append(
                {
                    "event_id": event["id"],
                    "matches": len(body.get("matches", [])),
                    "used_by_later_reasoning": "not directly observable",
                }
            )
        if kind == "context_compaction":
            compactions.append(
                {
                    "event_id": event["id"],
                    "source_events": body.get("source_events", []),
                    "savings_estimate": body.get("savings_estimate"),
                    "lost_useful_information": "requires trajectory review, not inferred",
                }
            )
        if kind == "child_purpose":
            child_events.append({"event_id": event["id"], **body})
    return {
        "repeated_actions_without_observed_edit": repeated,
        "repeated_searches": [{"arguments": k, "count": n} for k, n in searches.items() if n > 1],
        "repeated_reads": [{"arguments": k, "count": n} for k, n in reads.items() if n > 1],
        "failed_actions": failed,
        "retrievals": retrievals,
        "compactions": compactions,
        "children": child_events,
        "limitations": [
            "Arbitrary Python reads/searches are not syscall-traced; tool counts are lower bounds.",
            "No causal usefulness claim follows from repetition or a retrieval hit.",
        ],
    }
