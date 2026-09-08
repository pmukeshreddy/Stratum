"""Paired observed metrics with sampling uncertainty; no causal effectiveness claims."""

import argparse
import ast
import collections
import json
import math
import random
import statistics
from pathlib import Path

from threadweave.artifacts import atomic_write


def analysis(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = collections.Counter()
    events_used = collections.Counter()
    for row in rows:
        events = [
            json.loads(line) for line in Path(row["trajectory_path"]).read_text().splitlines()
        ]
        for event in events:
            payload = event["payload"]
            if event["type"] == "tool_call":
                counts[payload["name"]] += 1
            if event["type"] == "python_execution":
                try:
                    tree = ast.parse(payload.get("code", ""))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                        if name in {"read_text", "read_bytes", "write_text", "write_bytes"}:
                            counts["python_" + name] += 1
            if event["type"] in {
                "test_selection",
                "coverage_import",
                "working_focus",
                "history_retrieval",
                "child_context_package",
                "no_progress",
            }:
                events_used[event["type"]] += 1
    solved = sum(r["solved"] for r in rows)
    n, z = len(rows), 1.96
    probability = solved / n
    center = (probability + z * z / (2 * n)) / (1 + z * z / n)
    radius = (
        z * math.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )

    def total(key):
        return sum(r.get("metrics", {}).get(key, 0) or 0 for r in rows)

    tokens = total("input_tokens") + total("output_tokens")
    return rows, {
        "tasks": n,
        "solved": solved,
        "solve_rate": probability,
        "solve_rate_wilson_95": [center - radius, center + radius],
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "tokens_per_task": tokens / n,
        "tokens_per_solved": tokens / solved if solved else None,
        "wall_seconds_per_solved": total("wall_time") / solved if solved else None,
        "model_calls": total("model_calls"),
        "python_cells": total("python_cells"),
        "model_seconds": total("model_seconds"),
        "tool_seconds": total("tool_seconds"),
        "repo_queries": total("repo_queries"),
        "retrievals": total("retrieval_calls"),
        "subagents": total("subagent_count"),
        "compactions": total("context_compactions"),
        "verifier_failures": total("verifier_failures"),
        "failed_actions": total("failed_actions"),
        "actions": dict(counts),
        "capability_events": dict(events_used),
        "monetary_cost": None,
        "billing": "ChatGPT subscription; no API price estimate",
    }


def paired(directory):
    left, base = analysis(directory / "base.jsonl")
    right, buffalo = analysis(directory / "buffalo.jsonl")
    a, b = {r["instance_id"]: r for r in left}, {r["instance_id"]: r for r in right}
    assert a.keys() == b.keys(), "Comparison requires identical task IDs"
    differences, per_task = [], []
    for key in a:
        x, y = a[key], b[key]
        assert x["base_revision"] == y["base_revision"]
        assert x["resolved_config"]["provider"] == y["resolved_config"]["provider"]
        assert x["resolved_config"]["limits"] == y["resolved_config"]["limits"]

        def tokens(r):
            return r["metrics"]["input_tokens"] + r["metrics"]["output_tokens"]

        differences.append(tokens(y) - tokens(x))
        per_task.append(
            {
                "id": key,
                "base_tokens": tokens(x),
                "buffalo_tokens": tokens(y),
                "base_calls": x["metrics"]["model_calls"],
                "buffalo_calls": y["metrics"]["model_calls"],
                "base_solved": x["solved"],
                "buffalo_solved": y["solved"],
            }
        )
    rng = random.Random(1729)
    means = sorted(
        statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000)
    )
    return {
        "base": base,
        "buffalo": buffalo,
        "paired_token_difference_buffalo_minus_base": {
            "mean": statistics.mean(differences),
            "bootstrap_95": [means[250], means[9750]] if len(differences) > 1 else None,
            "resampling_unit": "task",
        },
        "per_task": per_task,
        "limitations": [
            f"{len(left)} supplied public tasks; this subset alone does not establish representative coding effectiveness",
            "Public programs may be present in model training data",
            "One stochastic run per profile/task; no inference seed control",
            "Base uses the same minimal persistent Python/runtime and independent verifier, without additional Buffalo intelligence; not a separate commercial harness",
            "API invocation counts do not establish causal use of evidence in the final solution",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = paired(args.directory)
    atomic_write(args.directory / "comparison.json", json.dumps(result, indent=2).encode())
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))
