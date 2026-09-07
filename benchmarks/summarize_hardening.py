"""Recompute descriptive evidence from saved real trajectories; never rescore patches."""

import argparse
import csv
import io
import json
import shlex
from pathlib import Path

from threadweave.artifacts import atomic_write
from threadweave.trajectory_analysis import analyze_events


def command_tokens(command):
    if isinstance(command, list):
        if (
            len(command) >= 3
            and Path(command[0]).name in {"sh", "bash", "zsh"}
            and command[1] == "-c"
        ):
            return command_tokens(command[2])
        return command
    try:
        return shlex.split(command)
    except ValueError:
        return []


def summarize(directory):
    rows = []
    for line in (directory / "results.jsonl").read_text().splitlines():
        record = json.loads(line)
        events = [
            json.loads(line) for line in Path(record["trajectory_path"]).read_text().splitlines()
        ]
        evidence = analyze_events(events)
        counts = record["metrics"]
        commands = [event["payload"] for event in events if event["type"] == "execution_result"]
        recognized = [
            c
            for c in commands
            if any(
                Path(t).name in {"pytest", "py.test", "unittest", "ctest", "jest", "vitest"}
                for t in command_tokens(c["command"])
            )
        ]
        row = {
            "run_id": record["id"],
            "task": record["instance_id"],
            "profile": record["profile"],
            "outcome": record["outcome"],
            "solved": record["solved"],
            "verifier_score": record["verifier_score"],
            "input_tokens": counts["input_tokens"],
            "output_tokens": counts["output_tokens"],
            "total_tokens": counts["input_tokens"] + counts["output_tokens"],
            "wall_seconds": counts["wall_time"],
            "model_calls": counts["model_calls"],
            "turns": counts["turns"],
            "python_cells": counts["python_cells"],
            "subagents": counts["subagent_count"],
            "compactions": counts["context_compactions"],
            "retrieval_calls": counts["retrieval_calls"],
            "failed_tool_results": sum(
                e["type"] == "tool_result" and bool(e["payload"].get("result", {}).get("error"))
                for e in events
            ),
            "recognized_test_command_completions": len(recognized),
            "test_command_failed_completions": sum(not c.get("passed") for c in recognized),
            "command_classification": "lexical argv/shell classification; includes baseline and evaluator; not all Python-owned subprocesses",
            "dollar_cost": counts["cost"],
            "source_sha256": record["harness_source"]["sha256"],
            "trajectory": record["trajectory_path"],
        }
        rows.append(row)
        atomic_write(
            directory / (record["id"] + "-review.json"), json.dumps(evidence, indent=2).encode()
        )
    aggregates = {}
    for profile in {r["profile"] for r in rows}:
        selected = [r for r in rows if r["profile"] == profile]
        solved = sum(r["solved"] for r in selected)
        tokens = sum(r["total_tokens"] for r in selected)
        aggregates[profile] = {
            "tasks": len(selected),
            "solved": solved,
            "solve_rate": solved / len(selected),
            "verification_passes": sum(r["verifier_score"] == 1 for r in selected),
            "tokens": tokens,
            "tokens_per_task": tokens / len(selected),
            "tokens_per_solved": tokens / solved if solved else None,
            "wall_seconds": sum(r["wall_seconds"] for r in selected),
            "model_calls": sum(r["model_calls"] for r in selected),
            "subagents": sum(r["subagents"] for r in selected),
            "compactions": sum(r["compactions"] for r in selected),
            "failed_tool_results": sum(r["failed_tool_results"] for r in selected),
            "recognized_test_command_completions": sum(
                r["recognized_test_command_completions"] for r in selected
            ),
            "dollar_cost": None
            if any(r["dollar_cost"] is None for r in selected)
            else sum(r["dollar_cost"] for r in selected),
        }
    summary = {
        "rows": rows,
        "profiles": aggregates,
        "note": "Descriptive small sample; no causal or superiority inference. Limited != solved, even when a withheld check passes.",
    }
    atomic_write(directory / "review-summary.json", json.dumps(summary, indent=2).encode())
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(directory / "review-summary.csv", text.getvalue().encode())
    return aggregates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    for directory in parser.parse_args().directories:
        print(directory, json.dumps(summarize(directory), indent=2))
