"""Real-model coding behavior sample. No feature-use quotas and no scripted actions.

Run against a source checkout with PYTHONPATH to compare production runtimes.
Independent acceptance tests run outside the agent; traces retain all evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from threadweave.models import RunConfig
from threadweave.runtime import Runtime

CASES = [
    {
        "name": "identifier_repair",
        "task": "Fix normalize_identifier in identifiers.py. It must strip leading/trailing whitespace, collapse internal whitespace to one underscore, lowercase text, preserve digits and existing underscores, and reject non-string inputs with TypeError. Add regression tests for edge cases.",
        "files": {
            "identifiers.py": "def normalize_identifier(value):\n    return str(value).strip().lower().replace(' ', '_')\n",
            "test_identifiers.py": "from identifiers import normalize_identifier\ndef test_simple():\n    assert normalize_identifier(' Hello World ') == 'hello_world'\n",
        },
        "acceptance": "from identifiers import normalize_identifier as f\nassert f(' A\\t B\\nC ') == 'a_b_c'\nassert f(' X_2  Y ') == 'x_2_y'\nassert f('') == ''\ntry: f(None)\nexcept TypeError: pass\nelse: raise AssertionError('non-string accepted')\n",
    },
    {
        "name": "ledger_aggregation",
        "task": "Implement totals(records) in ledger.py. Return a dict mapping each currency to the integer sum of amount_cents over records whose status is settled. Ignore pending/cancelled records. Negative amounts are valid. Do not mutate inputs. Inspect the sample transactions.json and add tests, including empty input and mixed currencies.",
        "files": {
            "ledger.py": "def totals(records):\n    return {}\n",
            "test_ledger.py": "from ledger import totals\ndef test_empty():\n    assert totals([]) == {}\n",
            "transactions.json": json.dumps(
                [
                    {
                        "currency": ["USD", "EUR", "JPY"][i % 3],
                        "status": ["settled", "pending"][i % 2],
                        "amount_cents": i - 1000,
                    }
                    for i in range(2500)
                ]
            ),
        },
        "acceptance": "import copy\nfrom ledger import totals\nx=[dict(currency='USD',status='settled',amount_cents=7),dict(currency='USD',status='settled',amount_cents=-3),dict(currency='EUR',status='pending',amount_cents=9)]\ny=copy.deepcopy(x)\nassert totals(x)=={'USD':4}\nassert x==y\nassert totals([])=={}\n",
    },
    {
        "name": "independent_components",
        "task": "Repair two independent components against their contracts. intervals.merge must merge overlapping and touching closed intervals, accept unsorted inputs, preserve inputs, and return [] for empty input. versions.compare must compare dot-separated nonnegative integer versions numerically, treating missing trailing components as zero, and return -1/0/1; reject malformed versions with ValueError. Investigate the components and their edge cases and add focused regression tests.",
        "files": {
            "intervals.py": "def merge(items):\n    return sorted(items)\n",
            "versions.py": "def compare(a,b):\n    return (a>b)-(a<b)\n",
            "test_components.py": 'from intervals import merge\nfrom versions import compare\ndef test_smoke():\n    assert merge([])==[]\n    assert compare("1","1")==0\n',
        },
        "acceptance": "from intervals import merge\nfrom versions import compare\na=[(3,5),(1,3),(9,10)]\nassert [tuple(x) for x in merge(a)]==[(1,5),(9,10)]\nassert a==[(3,5),(1,3),(9,10)]\nassert compare('1.10','1.2')==1\nassert compare('1.0','1')==0\nfor a in ['1..2','-1','a','']:\n    try: compare(a,'1')\n    except ValueError: pass\n    else: raise AssertionError(a)\n",
    },
]


def summarize(runtime, root, elapsed):
    events = list(runtime.store.iter_events(root.id, tree=True))
    counts = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    executions = [
        e for e in events if e["type"] == "python_execution" and e["session_id"] == root.id
    ]
    return {
        "outcome": str(runtime.store.session(root.id).outcome),
        "elapsed_seconds": elapsed,
        "root_entered_repl": bool(executions),
        "root_repl_actions": len(executions),
        "children": len([s for s in runtime.store.sessions() if s.parent_id]),
        "child_evidence_consumptions": sum(
            bool(e["payload"].get("child_evidence"))
            for e in events
            if e["type"] == "execution_input_consumed"
        ),
        "events": counts,
        "usage": runtime.store.usage(root.id, tree=True).model_dump(),
        "verification_levels": {
            str(level): sum(
                e["type"] == "verifier_started"
                if level == 3
                else e["type"] == "verification_result" and e["payload"].get("level") == 1
                if level == 1
                else e["type"] == "verification_targeted"
                for e in events
            )
            for level in (1, 2, 3)
        },
    }


async def evaluate(directory, *, model="", names=None):
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in CASES:
        if names and case["name"] not in names:
            continue
        target = directory / case["name"]
        workspace = target / "workspace"
        workspace.mkdir(parents=True)
        for name, text in case["files"].items():
            (workspace / name).write_text(text)
        (workspace / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
        for args in [
            ("init", "-q"),
            ("add", "."),
            (
                "-c",
                "user.name=Evaluation",
                "-c",
                "user.email=evaluation@localhost",
                "commit",
                "-qm",
                "Task inputs",
            ),
        ]:
            await asyncio.to_thread(
                subprocess.run, ["git", *args], cwd=workspace, check=True, capture_output=True
            )
        config = RunConfig(
            provider={
                "name": "codex_subscription",
                "model": model,
                "max_output_tokens": 8192,
                "parameters": {"reasoning_effort": "medium"},
            },
            context={"max_tokens": 24000, "summary_tokens": 2500},
            refinement={"evaluation_isolation": True},
            task={
                "adapter": "coding",
                "capture_baseline": False,
                "test_commands": [[sys.executable, "-m", "pytest", "-q"]],
                "verify_each_turn": False,
            },
            permissions=[
                "workspace.read",
                "workspace.write",
                "python",
                "process",
                "agents",
                "state",
            ],
            limits={
                "wall_seconds": 240,
                "token_budget": 500000,
                "max_turns": 24,
                "concurrency": 3,
                "max_subagents": 4,
            },
        )
        runtime = Runtime(target / "state")
        start = time.monotonic()
        try:
            root = runtime.create(case["task"], workspace, config=config)
            await runtime.start()
            await runtime.wait(root.id, timeout=270)
            report = {"name": case["name"], **summarize(runtime, root, time.monotonic() - start)}
            actual_workspace = runtime.store.session(root.id).workspace.path
            check = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-c", case["acceptance"]],
                cwd=actual_workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            report.update(
                acceptance_passed=check.returncode == 0, acceptance_error=check.stderr[-3000:]
            )
            report["final_answer"] = runtime.store.session(root.id).result
        finally:
            await runtime.shutdown()
        reports.append(report)
        (directory / "report.json").write_text(json.dumps(reports, indent=2))
        print(
            json.dumps({k: v for k, v in report.items() if k not in {"events", "final_answer"}}),
            flush=True,
        )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    asyncio.run(evaluate(args.output.resolve(), model=args.model, names=args.case))


if __name__ == "__main__":
    main()
