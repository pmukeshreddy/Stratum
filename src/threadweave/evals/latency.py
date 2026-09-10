"""Read-only wall-time profiles of production runtime ledgers (no model calls)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def covered(intervals):
    """Union duration, so concurrent work is never added as elapsed wall time."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def profile(directory):
    directory = Path(directory)
    with sqlite3.connect(f"file:{directory / 'state/history.sqlite3'}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        sessions = {r["id"]: json.loads(r["body"]) for r in db.execute("SELECT * FROM sessions")}
        root = next(s for s in sessions.values() if not s["parent_id"])
        events = {r["id"]: dict(r) for r in db.execute("SELECT * FROM events ORDER BY seq")}
        start = min(e["timestamp"] for e in events.values())
        stop = max(e["timestamp"] for e in events.values())
        intervals, calls = defaultdict(list), []
        for r in db.execute("SELECT * FROM model_requests ORDER BY started_at"):
            end = r["ended_at"] or stop
            session = sessions[r["session_id"]]
            kind = (
                r["purpose"]
                if r["purpose"] != "agent"
                else "child_model"
                if session["parent_id"]
                else "root_model"
            )
            intervals[kind].append((r["started_at"], end))
            artifact = db.execute(
                "SELECT path FROM artifacts WHERE id=?", (r["body_artifact"],)
            ).fetchone()
            body = json.loads((directory / "state" / artifact[0]).read_text())
            event = events.get(r["response_event"])
            response = json.loads(event["payload"]) if event else {}
            calls.append(
                {
                    "request_id": r["id"],
                    "session_id": session["id"],
                    "name": session["name"],
                    "kind": kind,
                    "start_seconds": round(r["started_at"] - start, 3),
                    "end_seconds": round(end - start, 3),
                    "seconds": round(end - r["started_at"], 3),
                    "status": r["status"],
                    "input_token_bound": body.get("input_token_bound"),
                    "message_characters": len(json.dumps(body["messages"], ensure_ascii=False)),
                    "output_characters": len(response.get("text", "")),
                }
            )
        for a in db.execute("SELECT * FROM actions"):
            if a["source_event"] in events and a["result_event"] in events:
                kind = "root_actions" if a["session_id"] == root["id"] else "child_actions"
                intervals[kind].append(
                    (events[a["source_event"]]["timestamp"], events[a["result_event"]]["timestamp"])
                )
        root_model = intervals["root_model"]
        auxiliary = [
            i
            for k, values in intervals.items()
            if k not in {"root_model", "child_model", "root_actions", "child_actions"}
            for i in values
        ]
        model_intervals = [
            i for k, values in intervals.items() if not k.endswith("_actions") for i in values
        ]
        root_overlap = covered(root_model) + covered(auxiliary) - covered(root_model + auxiliary)
        counts = Counter(e["type"] for e in events.values())
        boundaries = []
        for e in events.values():
            if e["type"] in {
                "completion_attempt",
                "refine_scheduled",
                "refinement_review",
                "refine_complete",
                "refine_failed",
                "completion_deferred",
                "context_compaction",
                "child_evidence_used",
            }:
                payload = json.loads(e["payload"])
                boundaries.append(
                    {
                        "seconds": round(e["timestamp"] - start, 3),
                        "name": sessions[e["session_id"]]["name"],
                        "type": e["type"],
                        **{
                            k: payload[k]
                            for k in (
                                "trigger",
                                "shouldRefine",
                                "turn",
                                "version",
                                "budget_seconds",
                            )
                            if k in payload
                        },
                    }
                )
        result = (
            json.loads((directory / "agent-result.json").read_text())
            if (directory / "agent-result.json").exists()
            else {}
        )
        return {
            "directory": str(directory),
            "outcome": root["outcome"],
            "reported_wall_seconds": result.get("usage", {}).get("wall_seconds"),
            "ledger_seconds": round(stop - start, 3),
            "union_seconds": {k: round(covered(v), 3) for k, v in intervals.items()},
            "summed_seconds": {k: round(sum(b - a for a, b in v), 3) for k, v in intervals.items()},
            "root_model_auxiliary_overlap_seconds": round(root_overlap, 3),
            "no_model_running_seconds": round(stop - start - covered(model_intervals), 3),
            "event_counts": {
                k: counts[k]
                for k in (
                    "context_compaction",
                    "completion_attempt",
                    "completion_deferred",
                    "agent_message_received",
                    "child_evidence_used",
                    "refine_scheduled",
                )
            },
            "calls": calls,
            "boundaries": boundaries,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([profile(d) for d in args.directories], indent=2) + "\n")


if __name__ == "__main__":
    main()
