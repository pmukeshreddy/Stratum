"""Unscored, non-benchmark tasks for inspecting adaptive production behavior."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..coding_config import update_coding_options
from ..gitops import git
from ..models import RunConfig
from .activity import activity
from .harness import run_buffalo
from .schema import file_digest, save

ROOT = Path(__file__).parents[3]


def source_hashes(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in sorted(directory.rglob("*.py"))}


def validation_config():
    config = RunConfig.model_validate_json((ROOT / "configs/coding.json").read_text())
    config.provider.model = "gpt-6-astra"
    config.provider.parameters = {"reasoning_effort": "xhigh"}
    config.provider.max_output_tokens = 32768
    config.provider.timeout_seconds = 300
    config.limits.wall_seconds = 300
    config.refinement.reasoning = "inherit"
    config.refinement.completion_followup = True
    # Read-only audits and data tasks do not require source edits or repository test suites.
    update_coding_options(
        config.task, capture_baseline=False, require_tests=False, require_change=False
    )
    return config


def prepare_workspace(workspace, task):
    workspace.mkdir(parents=True)
    (workspace / "TASK.txt").write_text(task["messages"][-1]["content"])
    (workspace / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(workspace, "init", "-q")
    git(workspace, "add", ".")
    git(
        workspace,
        "-c",
        "user.name=Validation",
        "-c",
        "user.email=validation@localhost",
        "commit",
        "-qm",
        "Task inputs",
    )


def validate_request(request):
    purpose = request.metadata.get("purpose", "agent")
    if purpose != "agent":
        return
    foundation = request.messages[0]["content"]
    assert foundation.startswith("You are Buffalo, a code-using agent")
    markers = [
        "Recursive mechanics:",
        "Independent work:",
        "Available subagent specifications:",
        "Harness state:",
        "Skills and project context:",
    ]
    assert [foundation.index(m) for m in markers] == sorted(foundation.index(m) for m in markers)
    assert "Direct reasoning and a direct answer are valid" in foundation
    assert [t["function"]["name"] for t in request.tools] == ["ipython"]


def cases():
    transactions = [
        {
            "currency": ("USD", "EUR", "JPY")[i % 3],
            "status": ("settled", "pending", "settled", "cancelled")[i % 4],
            "amount_cents": i % 997 - 500,
        }
        for i in range(6000)
    ]
    inspection = "Analyze transactions.json. Write result.json mapping each currency to its settled transaction count and the exact signed integer sum of amount_cents for those settled transactions. Pending and cancelled transactions do not contribute. Return a brief description of the result."
    repair_code = """def normalize_identifier(value):
    return str(value).strip().lower().replace(' ', '_')
"""
    repair_checks = """from identifiers import normalize_identifier
assert normalize_identifier('  Alpha  Beta ') == 'alpha_beta'
assert normalize_identifier('Alpha\\tBeta') == 'alpha_beta'
assert normalize_identifier('Straße') == 'strasse'
assert normalize_identifier('Already_valid') == 'already_valid'
for invalid in (None, 17, '', '   ', 'two-words', '2start'):
    try:
        normalize_identifier(invalid)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(repr(invalid))
print('all identifier checks passed')
"""
    parallel_files = {
        "intervals.py": """def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if start > end:
            raise ValueError('reversed interval')
        if not result or start >= result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = end
    return result

def covered_length(intervals):
    return sum(end-start for start,end in merge_intervals(intervals))
""",
        "ledger.py": """def reconcile(events):
    seen, balances = set(), {}
    for event in events:
        identity = event['id']
        if identity in seen:
            continue
        seen.add(identity)
        if event['status'] != 'settled':
            continue
        account = event['account']
        balances[account] = balances.get(account, 0) + int(event['amount'])
    return balances
""",
        "records.py": """def decode_record(text):
    return [part.strip('"') for part in text.split(',')]

def encode_record(fields):
    return ','.join(str(field) for field in fields)
""",
        "ordering.py": """def stable_topological_order(nodes, edges):
    incoming = {node: 0 for node in nodes}
    children = {node: [] for node in nodes}
    for parent, child in edges:
        children[parent].append(child)
        incoming[child] += 1
    ready = [node for node in nodes if incoming[node] == 0]
    result = []
    while ready:
        node = ready.pop()
        result.append(node)
        for child in children[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    return result
""",
        "SPEC.md": """Four independent library components need specification compliance review.
intervals: integer half-open intervals; return sorted disjoint nonempty intervals as lists. Merge overlap and adjacency. Ignore empty intervals. Reject reversed intervals. Never shrink an interval when another is nested. Do not mutate input. covered_length is the union length.
ledger: each event has account, id, status, amount (signed integer). Identity is (account,id). Pending/cancelled observations do not prevent a later settled event from applying. Repeated settled identity is applied once; conflicting settled amounts for the same identity raise ValueError. Amount must be an int, excluding bool; do not coerce strings or floats. Accounts with applied zero balance remain present.
records: encode/decode one CSV record using standard double-quote escaping. Preserve literal leading/trailing whitespace inside fields. Round-trip empty fields, commas, quotes, Unicode and embedded newlines. Return string fields. decode_record rejects malformed quoting and multiple records. No terminal newline in encoded output, except newlines belonging inside fields.
ordering: return a deterministic topological order. Each step picks the eligible node earliest in the supplied nodes order. Duplicate edges count once. Unknown edge endpoints and duplicate nodes raise ValueError. Cycles raise ValueError. Do not mutate inputs.
Implementations must use only the Python standard library. Add meaningful regression tests covering boundary failures and cross-checks against independent simple oracles where practical.
""",
    }
    return {
        "message_contract": {
            "instruction": "What is two plus three? Reply with digits only.",
            "messages": [
                {
                    "role": "system",
                    "content": "For this task, express quantities using lowercase English words only, with no punctuation or extra explanation.",
                },
                {"role": "user", "content": "What is two plus three? Reply with digits only."},
            ],
            "files": {},
        },
        "local": {"instruction": "What is 17 + 25? Reply with the number.", "files": {}},
        "inspection": {
            "instruction": inspection,
            "files": {"transactions.json": json.dumps(transactions)},
        },
        "component_audit": {
            "instruction": "Audit all four independent library components against SPEC.md. Report concrete specification violations with minimal reproducible checks and their observed outputs. Cover each component, distinguish demonstrated defects from assumptions, and recommend focused corrections. Do not modify the implementation files; this is a read-only audit. Work within the available time.",
            "files": parallel_files,
        },
        "optimality_review": {
            "instruction": "Assess the claim in CLAIM.md against allocator.py. Produce a rigorous conclusion, including concrete counterexamples with observed results if the claim is false. Assess feasibility, optimality, and the algorithm's scaling separately. Do not edit allocator.py.",
            "files": {
                "CLAIM.md": "Claim: least_loaded always assigns a list of positive integer job durations to two identical machines with the minimum possible makespan. Input order is arbitrary. Each job runs without preemption on exactly one machine. Include a precise explanation of any conditions under which the claim holds.",
                "allocator.py": "def least_loaded(jobs):\n    loads = [0, 0]\n    assignments = [[], []]\n    for duration in jobs:\n        index = loads.index(min(loads))\n        assignments[index].append(duration)\n        loads[index] += duration\n    return max(loads), assignments\n",
            },
        },
        "repeated_failure": {
            "instruction": "First run checks_amounts.py and checks_invoices.py to establish the current behavior. Repair amounts.py and invoices.py against SPEC.md, preserving the public interfaces. Verify the corrected library and both consumers on the supplied examples and additional boundary cases. Return the observed defects, corrections, and verification results.",
            "files": {
                "SPEC.md": "Money values arrive as strings with an optional leading minus, at least one ASCII digit before a decimal point if present, and zero to two fractional digits (at least one after a decimal point). No whitespace, exponent, plus sign, bool, or non-string input is allowed. cents(text) returns exact signed integer cents and rejects malformed strings with ValueError and non-strings with TypeError. total(values) sums exact cents. invoice_total(rows) sums quantity * unit_price in cents; quantity must be a nonnegative int excluding bool. Both functions must use the same validated cents contract. Round no input. Values may be arbitrarily large. The library is shared by transaction reporting and invoicing; both consumers must remain consistent.",
                "amounts.py": "def cents(text):\n    return int(float(text) * 100)\n\ndef total(values):\n    return sum(cents(value) for value in values)\n",
                "invoices.py": "def invoice_total(rows):\n    return int(sum(float(row['unit_price']) * row['quantity'] for row in rows) * 100)\n",
                "checks_amounts.py": "from amounts import cents, total\nassert cents('0.29') == 29\nassert cents('-0.29') == -29\nassert cents('9007199254740993.01') == 900719925474099301\nassert total(['0.29', '-0.29', '2']) == 200\nfor bad in [' 1', '1e2', '+2', '1.001']:\n    try: cents(bad)\n    except ValueError: pass\n    else: raise AssertionError(bad)\nprint('amount checks passed')\n",
                "checks_invoices.py": "from invoices import invoice_total\nassert invoice_total([{'quantity': 1, 'unit_price': '0.29'}]) == 29\nassert invoice_total([{'quantity': 3, 'unit_price': '0.29'}]) == 87\nfor bad in [True, -1, 1.5]:\n    try: invoice_total([{'quantity': bad, 'unit_price': '1'}])\n    except (TypeError, ValueError): pass\n    else: raise AssertionError(bad)\nprint('invoice checks passed')\n",
            },
        },
        "one_off": {
            "instruction": "Sort the distinct words pear, apple, plum, banana alphabetically. Reply only with the comma-separated words.",
            "files": {},
        },
        "failed_checks": {
            "instruction": "Run checks.py first to establish the current failures, then repair identifiers.py. normalize_identifier accepts only strings, strips leading/trailing whitespace, collapses every run of Unicode whitespace to one underscore, applies casefold, and requires a nonempty ASCII Python identifier (letters/underscore initially, then letters/digits/underscore). Non-string inputs raise TypeError; invalid normalized identifiers raise ValueError. Run the checks after correction and add useful additional cases.",
            "files": {"identifiers.py": repair_code, "checks.py": repair_checks},
        },
    }


async def execute(output, selected=None, wall_seconds=None):
    output = await asyncio.to_thread(Path(output).resolve)
    output.mkdir(parents=True, exist_ok=False)
    config = validation_config()
    if wall_seconds is not None:
        config.limits.wall_seconds = wall_seconds
    save(
        output / "manifest.json",
        {
            "kind": "unscored adaptive behavior validation",
            "source_hashes": source_hashes(ROOT / "src"),
            "config": config.model_dump(mode="json"),
        },
    )
    semaphore = asyncio.Semaphore(4)

    async def one(name, case):
        async with semaphore:
            directory = output / name
            task = {
                "messages": case.get("messages", [{"role": "user", "content": case["instruction"]}])
            }
            save(directory / "task.json", task)
            workspace = directory / "workspace"
            prepare_workspace(workspace, task)
            for filename, content in case["files"].items():
                (workspace / filename).write_text(content)
            if case["files"]:
                git(workspace, "add", ".")
                git(
                    workspace,
                    "-c",
                    "user.name=Validation",
                    "-c",
                    "user.email=validation@localhost",
                    "commit",
                    "-qm",
                    "Task inputs",
                )

            def inspect(request):
                validate_request(request)
                if request.metadata.get("purpose") in {
                    "refinement_review",
                    "refinement",
                    "compaction",
                }:
                    evidence = json.loads(request.messages[-1]["content"])
                    assert evidence["original_task"]["messages"] == task["messages"]
                first = directory / "first-provider-request.json"
                if not first.exists():
                    save(first, request.public_dump())

            print(json.dumps({"event": "started", "case": name}), flush=True)
            raw = await run_buffalo(
                config,
                task,
                directory,
                workspace=workspace,
                task_config=config.task,
                request_validator=inspect,
            )
            save(directory / "agent-result.json", raw)
            row = activity(
                directory,
                {
                    "task_id": name,
                    "wall_time_seconds": raw["usage"]["wall_seconds"],
                    **{
                        k: raw["usage"].get(k)
                        for k in ("input_tokens", "output_tokens", "total_tokens")
                    },
                },
            )
            save(directory / "activity.json", row)
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "case": name,
                        "rlm_calls": row["rlm_calls"],
                        "seconds": row["wall_seconds"],
                    }
                ),
                flush=True,
            )
            return row

    rows = await asyncio.gather(
        *(one(name, case) for name, case in cases().items() if not selected or name in selected)
    )
    save(output / "activity.json", rows)
    # Deliberately no automatic activity quota or benchmark authorization. Review actual
    # choices, task outputs and evidence before issuing a separate validation receipt.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", choices=list(cases()))
    parser.add_argument("--wall-seconds", type=float, help="Override the uniform task time budget")
    args = parser.parse_args()
    asyncio.run(execute(args.output, args.cases, args.wall_seconds))


if __name__ == "__main__":
    main()
