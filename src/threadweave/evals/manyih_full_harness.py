"""One Buffalo-only run of the fixed, previously evaluated ManyIH Coding tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from .activity import activity, aggregate
from .bridge import OfficialWorker
from .harness import run_buffalo
from .manyih_coding import (
    MODEL,
    REASONING,
    TIMEOUT,
    audit_trace,
    git,
    measured_usage,
    prepare_workspace,
    records_summary,
    run_config,
)
from .schema import BenchmarkSetup, file_digest, save, timestamp

IDS = [str(i) for i in range(100)]
ROOT = Path(__file__).parents[3]


def experiment_config():
    config = run_config()
    config.refinement.reasoning = "inherit"
    config.refinement.automatic_budget_seconds = 120
    config.refinement.continuation_reserve_seconds = 60
    config.refinement.completion_followup = True
    return config


def source_hashes(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in sorted(directory.rglob("*.py"))}


def mechanisms(directory):
    """Count committed runtime evidence, separately from model-request attempts."""
    with sqlite3.connect(f"file:{directory / 'state/history.sqlite3'}?mode=ro", uri=True) as db:
        events = Counter(dict(db.execute("SELECT type,count(*) FROM events GROUP BY type")))
        purposes = Counter(
            dict(db.execute("SELECT purpose,count(*) FROM model_requests GROUP BY purpose"))
        )
        actions = [(n, json.loads(a)) for n, a in db.execute("SELECT name,arguments FROM actions")]
        operations = Counter(a.get("operation") for n, a in actions if n == "host_request")
        sessions = [json.loads(r[0]) for r in db.execute("SELECT body FROM sessions")]
        statuses = dict(db.execute("SELECT status,count(*) FROM refinement_runs GROUP BY status"))
        requests = [
            json.loads(r[0])
            for r in db.execute("SELECT payload FROM events WHERE type='model_request'")
        ]
        initial_states = [
            json.loads(r[0])
            for r in db.execute("SELECT payload FROM events WHERE type='session_created'")
        ]
    return {
        "repl_calls": sum(n == "ipython" for n, _ in actions),
        "python_executions": events["python_result"],
        "rlm_calls": operations["rlm.run"],
        "subagents": sum(s["parent_id"] is not None for s in sessions),
        "max_depth": max((s["depth"] for s in sessions), default=0),
        "agent_requests": purposes["agent"],
        "root_turns": sum(s["turns"] for s in sessions if s["parent_id"] is None),
        "all_agent_turns": sum(s["turns"] for s in sessions),
        "refinement_reviews": purposes["refinement_review"],
        "refinement_planner_and_reducer_requests": purposes["refinement"],
        "refinement_applied_edits": events["refinement"],
        "refinement_continuations": events["refinement_continuation"],
        "refinement_budget_exhausted": events["refinement_budget_exhausted"],
        "auxiliary_deferred": events["auxiliary_deferred"],
        "state_retrievals": events["state_retrieved"],
        "compaction_requests": purposes["compaction"],
        "refinement_statuses": statuses,
        "host_operations": dict(operations),
        "event_counts": dict(events),
        "request_purposes": dict(purposes),
        "request_events": len(requests),
        "initial_session_events": len(initial_states),
    }


def aggregate_mechanisms(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, int)]
    return {
        "totals": {k: sum(r[k] for r in rows) for k in keys},
        "tasks_using": {k: sum(r[k] > 0 for r in rows) for k in keys},
    }


def audit(output, previous):
    manifest = json.loads((output / "manifest.json").read_text())
    assert source_hashes(ROOT / "src") == manifest["source_hashes"], "Runtime changed during run"
    records, old_metrics = [], []
    for task_id in IDS:
        directory = output / "full/buffalo" / task_id
        record = json.loads((directory / "result.json").read_text())
        assert record["task_id"] == task_id
        record.update(measured_usage(record))
        record["usage_complete"] = not record["unknown_usage_attempts"]
        if not record["usage_complete"]:
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                record[k] = None
        audit_trace(record)
        old = json.loads((previous / "full/buffalo" / task_id / "result.json").read_text())
        assert record["prompt_sha256"] == old["prompt_sha256"]
        assert record["workspace_hashes"] == old["workspace_hashes"]
        grade = json.loads((directory / "official-grade.json").read_text())["evaluation"]
        for k, official_key in (
            ("functional_pass", "test_passed"),
            ("style_pass", "style_passed"),
            ("overall_pass", "overall_passed"),
        ):
            assert record[k] == bool(grade[official_key])
        config = json.loads((directory / "buffalo-config.json").read_text())
        initial = json.loads((directory / "initial-harness-state.json").read_text())
        assert initial["states"] == [] and initial["selected_state"] == []
        assert Path(initial["state_directory"]) == directory / "state"
        assert config["refinement"]["completion_followup"]
        assert config["refinement"]["automatic"] and config["refinement"]["enabled"]
        assert all(config["features"].values())
        assert config["limits"]["wall_seconds"] == TIMEOUT
        assert config["limits"]["concurrency"] == 4
        assert config["skill_paths"] == [] and config["refinement"]["selected_entries"] == []
        for line in (directory / "provider-calls.jsonl").read_text().splitlines():
            request = json.loads(line)["request"]
            assert request["config"]["model"] == MODEL
            assert request["config"]["parameters"]["reasoning_effort"] == REASONING
            if request["metadata"].get("purpose", "agent") == "agent":
                foundation = request["messages"][0]["content"]
                for term in (
                    "persistent",
                    "rlm(",
                    "agent_message",
                    "prompt_note",
                    "subagent_spec",
                    "harness.get",
                    "skills.run",
                    "refine()",
                    "compact()",
                ):
                    assert term in foundation, term
        record["mechanisms"] = mechanisms(directory)
        record["activity"] = activity(directory, record)
        save(directory / "result.json", record)
        records.append(record)
        old_metrics.append(mechanisms(previous / "full/buffalo" / task_id))
    old_records = [
        json.loads(line) for line in (previous / "buffalo-results.jsonl").read_text().splitlines()
    ]
    codex_records = [
        json.loads(line) for line in (previous / "codex-results.jsonl").read_text().splitlines()
    ]
    for name, sha in manifest["existing_results_sha256"].items():
        assert file_digest(previous / name) == sha, "Existing results were changed"
    summary = {
        "denominator": 100,
        "old_buffalo": records_summary(old_records),
        "new_buffalo": records_summary(records),
        "existing_codex": records_summary(codex_records),
        "old_mechanisms": aggregate_mechanisms(old_metrics),
        "new_mechanisms": aggregate_mechanisms([r["mechanisms"] for r in records]),
        "integrity": "PASS",
        "single_attempts": len(list((output / "full/buffalo").glob("*/attempt.json"))),
        "codex_rerun": False,
        "cross_task_learning": False,
        "source_hashes_unchanged": True,
        "natural_activity": aggregate([r["activity"] for r in records]),
        "optional_feature_usage_affects_validity": False,
    }
    old_by_id = {r["task_id"]: r for r in old_records}
    summary["paired_old_vs_new"] = dict(
        Counter(
            "both_pass"
            if old_by_id[r["task_id"]]["overall_pass"] and r["overall_pass"]
            else "new_only"
            if r["overall_pass"]
            else "old_only"
            if old_by_id[r["task_id"]]["overall_pass"]
            else "both_fail"
            for r in records
        )
    )
    assert summary["single_attempts"] == 100
    assert summary["old_buffalo"]["overall_pass"] == 62
    assert summary["existing_codex"]["overall_pass"] == 66
    save(output / "summary.json", summary)
    (output / "buffalo-results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output / "activity.jsonl").write_text(
        "".join(json.dumps(r["activity"]) + "\n" for r in records)
    )
    return summary


async def execute(args):
    validation = None
    if args.validation:
        validation = json.loads(await asyncio.to_thread(Path(args.validation).read_text))
        assert validation["adaptive_policy_demonstrated"] is True
        assert validation["source_hashes"] == source_hashes(ROOT / "src"), (
            "Production source changed after live validation"
        )
        assert validation["config"] == experiment_config().model_dump(mode="json")
    output, previous, source = map(
        lambda p: Path(p).resolve(), (args.output, args.previous, args.source)
    )
    # A new output root is mandatory. There is no task retry, smoke run, baseline or variant path.
    output.mkdir(parents=True, exist_ok=False)
    old_manifest = json.loads((previous / "manifest.json").read_text())
    codex_inputs = {
        r["task_id"]: r
        for r in map(json.loads, (previous / "codex-results.jsonl").read_text().splitlines())
    }
    assert old_manifest["task_ids"] == IDS
    assert git(source, "rev-parse", "HEAD") == old_manifest["official"]["starting_state"]["commit"]
    assert not git(source, "status", "--porcelain", "--untracked-files=no")
    official = OfficialWorker(
        BenchmarkSetup(
            source=source, commit=git(source, "rev-parse", "HEAD"), python=sys.executable
        ),
        "manyih-coding",
        output / "official",
    )
    began = time.monotonic()
    try:
        provenance = await official.start()
        assert provenance["task_ids"][:100] == IDS
        assert provenance["starting_state"] == old_manifest["official"]["starting_state"]
        config = experiment_config()
        tasks = {}
        for task_id in IDS:
            task = await official.call("task", task_id=task_id)
            directory = output / "full/buffalo" / task_id
            save(directory / "task.json", task)
            old = json.loads((previous / "full/buffalo" / task_id / "result.json").read_text())
            hashes = await asyncio.to_thread(prepare_workspace, directory / "workspace", task)
            assert file_digest(directory / "task.json") == old["prompt_sha256"]
            assert hashes == old["workspace_hashes"]
            assert file_digest(directory / "task.json") == codex_inputs[task_id]["prompt_sha256"]
            assert hashes == codex_inputs[task_id]["workspace_hashes"]
            tasks[task_id] = (task, hashes)
        save(output / "task-ids.json", IDS)
        save(output / "buffalo-config.json", config.model_dump(mode="json"))
        save(
            output / "manifest.json",
            {
                "created_at": timestamp(),
                "official": provenance,
                "task_ids": IDS,
                "model": MODEL,
                "reasoning": REASONING,
                "all_auxiliary_reasoning": REASONING,
                "per_task_timeout": TIMEOUT,
                "concurrency": 4,
                "denominator": 100,
                "attempts_per_task": 1,
                "systems_run": ["buffalo"],
                "smoke_runs": 0,
                "cross_task_learning": False,
                "adaptive_validation": validation,
                "python": sys.version,
                "previous_archive": str(previous),
                "buffalo_git_commit": git(ROOT, "rev-parse", "HEAD"),
                "source_hashes": source_hashes(ROOT / "src"),
                "existing_results_sha256": {
                    name: file_digest(previous / name)
                    for name in (
                        "buffalo-results.jsonl",
                        "codex-results.jsonl",
                        "summary.json",
                        "manifest.json",
                    )
                },
                "protocol": "Fresh workspace, Runtime and state database per task. Official prompts and starting files hash-match old run. No hidden feedback before completion.",
            },
        )
        semaphore = asyncio.Semaphore(4)

        async def one(task_id):
            async with semaphore:
                directory = output / "full/buffalo" / task_id
                task, hashes = tasks[task_id]
                with (directory / "attempt.json").open("x") as marker:
                    json.dump({"task_id": task_id, "attempt": 1, "started_at": timestamp()}, marker)
                start = time.monotonic()
                print(json.dumps({"event": "started", "task_id": task_id}), flush=True)
                try:
                    raw = await run_buffalo(
                        config,
                        task,
                        directory,
                        workspace=directory / "workspace",
                        task_config=config.task,
                    )
                except Exception as exc:
                    usage_path = directory / "usage.json"
                    raw = {
                        "response": "",
                        "usage": json.loads(usage_path.read_text()) if usage_path.exists() else {},
                        "stop_reason": f"{type(exc).__name__}: {exc}",
                    }
                save(directory / "agent-result.json", raw)
                (directory / "answer.txt").write_text(raw.get("response", ""))
                grade = await official.call(
                    "grade", task_id=task_id, response=raw.get("response", "")
                )
                save(directory / "official-grade.json", grade)
                evaluation, usage = grade["evaluation"], raw.get("usage", {})
                record = {
                    "task_id": task_id,
                    "system": "buffalo",
                    "functional_pass": bool(evaluation["test_passed"]),
                    "style_pass": bool(evaluation["style_passed"]),
                    "overall_pass": bool(evaluation["overall_passed"]),
                    "wall_time_seconds": usage.get("wall_seconds", time.monotonic() - start),
                    **{
                        k: None if usage.get("estimated_calls") else usage.get(k)
                        for k in ("input_tokens", "output_tokens", "total_tokens")
                    },
                    "estimated_cost": None,
                    "run_artifact": str(directory),
                    "official_evaluation": str(directory / "official-grade.json"),
                    "prompt_sha256": file_digest(directory / "task.json"),
                    "workspace_hashes": hashes,
                    "model": MODEL,
                    "reasoning": REASONING,
                    "stop_reason": raw.get("stop_reason"),
                    "failure_reason": None
                    if evaluation["overall_passed"]
                    else raw.get("stop_reason")
                    if raw.get("stop_reason") != "completed"
                    else evaluation.get("test_result")
                    if not evaluation["test_passed"]
                    else "instruction/style violation",
                }
                save(directory / "result.json", record)
                print(
                    json.dumps(
                        {
                            "event": "completed",
                            "task_id": task_id,
                            "overall_pass": record["overall_pass"],
                            "stop_reason": record["stop_reason"],
                            "seconds": record["wall_time_seconds"],
                        }
                    ),
                    flush=True,
                )
                return grade

        launch_directory = Path(args.validation).parent if args.validation else output
        launch = launch_directory / "manyih-launch.json"
        with launch.open("x") as stream:
            json.dump(
                {"output": str(output), "started_at": timestamp(), "attempts_per_task": 1}, stream
            )
        grades = await asyncio.gather(*(one(task_id) for task_id in IDS))
        summary = await official.call("summarize", grades=grades, profile="buffalo")
        save(output / "official-buffalo-summary.json", summary)
        save(output / "experiment-time.json", {"wall_seconds": time.monotonic() - began})
        print(json.dumps(audit(output, previous)), flush=True)
    finally:
        await official.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--validation", help="Reviewed targeted live validation receipt for this production source"
    )
    args = parser.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    if args.audit_only:
        print(json.dumps(audit(Path(args.output).resolve(), Path(args.previous).resolve())))
    else:
        asyncio.run(execute(args))


if __name__ == "__main__":
    main()
