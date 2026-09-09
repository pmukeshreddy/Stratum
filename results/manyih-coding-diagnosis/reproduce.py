"""Post-hoc diagnostics on saved answers. No model calls or production mutations.

The hand-selected answer edits use known failures, so their grades are NOT a new
Buffalo score or a prospective evaluation. Official sources remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

from threadweave.evals.bridge import OfficialWorker
from threadweave.evals.schema import BenchmarkSetup


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def edit_answer(task_id, code):
    if task_id in {3, 24, 32, 53}:
        return code.replace(") -> ", ")->"), "Return annotation whitespace only"
    if task_id == 21:
        return code.replace(
            "(str1: str):", "(str1: str) -> str | None:"
        ), "Add missing return annotation"
    if task_id in {33, 40, 65, 82}:
        # Existing blank line is between def and the first statement; put a short
        # permitted docstring before it so it is actually inside the AST body.
        lines = code.splitlines()
        definition = next(i for i, line in enumerate(lines) if line.startswith("def "))
        next_statement = next(line for line in lines[definition + 1 :] if line.strip())
        indent = next_statement[: len(next_statement) - len(next_statement.lstrip())]
        descriptions = {
            33: "Return the first element of each input list.",
            40: "Swap the case of each letter.",
            65: "Return the largest value minus the smallest value.",
            82: "Append the list elements to the tuple.",
        }
        lines.insert(definition + 1, indent + '"""' + descriptions[task_id] + '"""')
        changed = "\n".join(lines)
        if task_id == 40:
            changed = changed.replace(") -> ", ")->")
        return changed, "Move existing blank into function body by adding a permitted docstring" + (
            "; annotation spacing" if task_id == 40 else ""
        )
    if task_id == 69:
        return code.replace(
            "Sum lengths of names with an uppercase first letter and lowercase rest.",
            "Sum lengths of names with a capital first letter.",
        ), "Shorten overlong docstring"
    if task_id == 77:
        return code.replace("element-wise", "element wise"), "Remove hyphen from docstring"
    if task_id == 83:
        return code.replace(
            "or -1 if absent", "or negative one if absent"
        ), "Spell out negative number in docstring"
    if task_id == 96:
        return code.replace("km/h", "kilometers per hour"), "Spell out units in docstring"
    raise ValueError(task_id)


def execution_tree(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
    return ast.dump(tree, include_attributes=False)


def assertion_details(source, code, tests):
    # Official helper mutates process globals through its execution guard. Keep
    # that behavior confined to a disposable process, like the official worker.
    program = """import json,sys
sys.path.insert(0,sys.argv[1])
from manyih.coding.eval_utils import check_correctness_per_assertion
x=json.load(sys.stdin)
print(json.dumps(check_correctness_per_assertion(x['code'],x['tests'],timeout=3.0)))
"""
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-c", program, str(source)],
            input=json.dumps({"code": code, "tests": tests}),
            text=True,
            timeout=20,
        )
    )


async def main(args):
    archive, source, output = args.archive.resolve(), args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((source / "manyih/data/coding.json").read_text())["data"]
    commit = (
        await asyncio.to_thread(
            subprocess.check_output,
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
        )
    ).strip()
    assert commit == "3287f82436fff86d7506b7737dbd3b5e2e524d01"
    worker = OfficialWorker(
        BenchmarkSetup(source=source, commit=commit, python=sys.executable),
        "manyih-coding",
        output / "official-diagnostics",
    )
    rows, edits, errors = [], [], []
    calls, turns, verifier_checks, outcomes = Counter(), Counter(), Counter(), Counter()
    selected = {3, 21, 24, 32, 33, 40, 53, 65, 69, 77, 82, 83, 96}
    await worker.start()
    try:
        for task_id in range(100):
            folder = archive / f"full/buffalo/{task_id}"
            old = json.loads((folder / "official-grade.json").read_text())["evaluation"]
            other = json.loads((archive / f"full/codex/{task_id}/official-grade.json").read_text())[
                "evaluation"
            ]
            row = {
                "task_id": str(task_id),
                "functional_pass": old["test_passed"],
                "style_pass": old["style_passed"],
                "overall_pass": old["overall_passed"],
                "codex_functional_pass": other["test_passed"],
                "codex_style_pass": other["style_passed"],
                "codex_overall_pass": other["overall_passed"],
                "failed_style_categories": [
                    k for k, v in old["each_style_passed"].items() if not v
                ],
                "style_diagnostics": {
                    k: old["style_messages"][k]
                    for k, v in old["each_style_passed"].items()
                    if not v
                },
            }
            rows.append(row)
            with sqlite3.connect(
                f"file:{folder / 'state/history.sqlite3'}?mode=ro", uri=True
            ) as db:
                purposes = dict(
                    db.execute("SELECT purpose,count(*) FROM model_requests GROUP BY purpose")
                )
                calls.update(purposes)
                turns[purposes.get("agent", 0)] += 1
                for (body,) in db.execute(
                    "SELECT payload FROM events WHERE type='verifier_result'"
                ):
                    event = json.loads(body)
                    body = json.loads(event["result"]["preview"])
                    verifier_checks[body["metrics"]["checks"]] += 1
                    outcomes[str(event["passed"])] += 1
            if task_id in selected:
                changed, description = edit_answer(task_id, old["extracted_code"])
                assert execution_tree(changed) == execution_tree(old["extracted_code"])
                destination = output / f"posthoc-answer-edits/{task_id}"
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "original.py").write_text(old["extracted_code"] + "\n")
                (destination / "edited.py").write_text(changed + "\n")
                new = await worker.call(
                    "grade", task_id=str(task_id), response="```python\n" + changed + "\n```"
                )
                save(destination / "official-grade.json", new)
                edits.append(
                    {
                        "task_id": str(task_id),
                        "edit": description,
                        "original_overall_pass": old["overall_passed"],
                        "posthoc_overall_pass": new["evaluation"]["overall_passed"],
                        "posthoc_functional_pass": new["evaluation"]["test_passed"],
                        "posthoc_style_pass": new["evaluation"]["style_passed"],
                        "same_execution_ast_ignoring_docstrings_and_return_annotations": True,
                    }
                )
            if not old["test_passed"]:
                row_data = dataset[task_id]
                details = assertion_details(source, old["extracted_code"], row_data["test_code"])
                ref = assertion_details(source, row_data["reference_code"], row_data["test_code"])
                item = {
                    "task_id": str(task_id),
                    "public_task": row_data["original_prompt"],
                    "saved_answer_details": details,
                    "reference_details": ref,
                }
                if task_id in {15, 48, 95, 99}:
                    item["diagnostic_only_with_math_import"] = assertion_details(
                        source, "import math\n" + old["extracted_code"], row_data["test_code"]
                    )
                errors.append(item)
    finally:
        await worker.close()
    categories = Counter(k for row in rows for k in row["failed_style_categories"])
    both_false = sum(not r["functional_pass"] and not r["style_pass"] for r in rows)
    positives = sum(e["posthoc_overall_pass"] and not e["original_overall_pass"] for e in edits)
    negative = sum(not e["posthoc_overall_pass"] and e["original_overall_pass"] for e in edits)
    paired = {
        "buffalo_only": [
            r["task_id"] for r in rows if r["overall_pass"] and not r["codex_overall_pass"]
        ],
        "codex_only": [
            r["task_id"] for r in rows if not r["overall_pass"] and r["codex_overall_pass"]
        ],
    }
    n = sum(len(v) for v in paired.values())
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(map(len, paired.values())) + 1)) / 2**n)
    findings = {
        "official_baseline_score_unchanged": 62,
        "tasks": 100,
        "style_only_failures": sum(r["functional_pass"] and not r["style_pass"] for r in rows),
        "functional_only_failures": sum(not r["functional_pass"] and r["style_pass"] for r in rows),
        "both_failed": both_false,
        "style_category_failures": dict(categories),
        "model_request_purposes": dict(calls),
        "agent_request_count_histogram": dict(turns),
        "verifier_check_count_histogram": dict(verifier_checks),
        "verifier_pass_histogram": dict(outcomes),
        "paired": paired,
        "exact_two_sided_mcnemar_p": p,
        "posthoc_diagnostics": {
            "NOT_AN_AGENT_SCORE": True,
            "uses_known_failures": True,
            "new_model_calls": 0,
            "edited_task_count": len(edits),
            "official_grader_recovered_answers": positives,
            "new_failures": negative,
            "saved_answer_counterfactual_total": 62 + positives - negative,
        },
    }
    save(output / "findings.json", findings)
    save(output / "task-diagnoses.json", rows)
    save(output / "posthoc-edit-results.json", edits)
    save(output / "functional-failure-details.json", errors)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
