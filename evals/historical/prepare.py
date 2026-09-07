"""Prepare two confirmed historical defects, not generated bug imitations.

The evaluator-only test patch is never included in model context or checkout
until independent final evaluation. This local suite is not a public benchmark.
"""

import argparse
import json
import sys
from pathlib import Path

from threadweave.artifacts import atomic_write
from threadweave.editing import make_diff
from threadweave.models import RunConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2]
    cases = {
        "followup": "Fix Runtime.message so an explicit parent follow-up to a successfully completed child resumes that SAME child. Preserve Python checkpoint/kernel identity, history, workspace and accumulated usage across restart. Do not revive cancelled, failed, paused or exhausted sessions, reset budgets, or duplicate concurrent turns. Add your own focused tests and run relevant existing tests.",
        "refine": "Fix interactive /refine: Chat.submit currently routes it to slash(), which rejects it before Runtime.message's refinement mechanism. Wire this control command and help text into the existing evidence-based refinement path. It must work when refinement is enabled but periodic automatic refinement is disabled. Report requested/applied/skipped/failed honestly; no evidence is a no-op. Add your own tests and preserve existing behavior.",
    }
    tasks = []
    for name, objective in cases.items():
        test = source / "evals" / "historical" / (name + "_test.py")
        target = "tests/test_frozen_acceptance.py"
        patch = make_diff(target, None, test.read_bytes())
        atomic_write(output / (name + ".patch"), patch.encode())
        visible = f"PYTHONPATH=src {sys.executable} -m pytest tests/test_python_control.py tests/test_chat.py -q"
        hidden = f"PYTHONPATH=src {sys.executable} -m pytest {target} -q"
        tasks.append(
            {
                "id": "historical-" + name,
                "adapter": "repository_issue",
                "repository": str(source),
                "base_commit": "e5c8c41",
                "objective": objective,
                "test_commands": [["bash", "-c", visible]],
                "verifier_commands": [["bash", "-c", hidden]],
                "test_patch": name + ".patch",
            }
        )
    atomic_write(output / "tasks.json", json.dumps(tasks, indent=2).encode())
    config = RunConfig(
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        provider={
            "name": "codex_subscription",
            "model": "gpt-6-astra",
            "parameters": {"reasoning_effort": "low"},
            "max_output_tokens": 4096,
            "timeout_seconds": 180,
        },
        task={"adapter": "coding", "capture_baseline": True, "protect_tests": True},
        refinement={"automatic": False},
        limits={
            "max_turns": 20,
            "wall_seconds": 600,
            "token_budget": 120000,
            "tool_timeout_seconds": 90,
            "python_timeout_seconds": 90,
            "max_subagents": 2,
            "concurrency": 3,
        },
        context={"max_tokens": 60000},
    )
    atomic_write(output / "config.json", config.model_dump_json(indent=2).encode())
    print(output / "tasks.json")


if __name__ == "__main__":
    main()
