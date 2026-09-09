"""Official ManyIH Coding: production Buffalo coding sessions versus native Codex exec."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..coding_config import update_coding_options
from ..models import RunConfig
from .bridge import OfficialWorker
from .harness import run_buffalo
from .schema import BenchmarkSetup, file_digest, save, timestamp

MODEL = "gpt-6-astra"
REASONING = "xhigh"
TIMEOUT = 300


def git(workspace, *args):
    return subprocess.check_output(["git", "-C", str(workspace), *args], text=True).strip()


def prepare_workspace(workspace, task):
    workspace.mkdir(parents=True)
    (workspace / "TASK.txt").write_text(task["messages"][-1]["content"])
    (workspace / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    git(workspace, "init", "-q")
    git(workspace, "add", ".")
    git(
        workspace,
        "-c",
        "user.name=Evaluation",
        "-c",
        "user.email=evaluation@localhost",
        "commit",
        "-qm",
        "Official task input",
    )
    return {p.name: file_digest(p) for p in workspace.iterdir() if p.is_file()}


def run_config():
    config = RunConfig.model_validate_json(
        (Path(__file__).parents[3] / "configs/coding.json").read_text()
    )
    config.provider.model = MODEL
    config.provider.parameters = {"reasoning_effort": REASONING}
    config.provider.max_output_tokens = 32768
    config.provider.timeout_seconds = TIMEOUT
    config.limits.wall_seconds = TIMEOUT
    config.limits.token_budget = 3_000_000
    config.limits.concurrency = 4
    # Official task supplies an answer-generation interface, not a repository test suite.
    # Gold assertions and winning styles remain solely in the official grading worker.
    update_coding_options(
        config.task, capture_baseline=False, require_tests=False, require_change=False
    )
    return config


async def baseline(directory, workspace, task, config):
    answer = directory / "answer.txt"
    home = Path(tempfile.mkdtemp(prefix="manyih-codex-home-"))
    await asyncio.to_thread(home.chmod, 0o700)
    source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    auth = source_home / "auth.json"
    if not auth.is_file():
        raise RuntimeError(
            "Native Codex file authentication is unavailable for an isolated baseline home"
        )
    (home / "auth.json").symlink_to(auth)
    environment = dict(os.environ)
    environment["CODEX_HOME"] = str(home)
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN", "PYTHONPATH"):
        environment.pop(key, None)
    command = [
        shutil.which("codex"),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
        "--color",
        "never",
        "-m",
        MODEL,
        "-c",
        f"model_reasoning_effort={json.dumps(REASONING)}",
        "-c",
        'approval_policy="never"',
        "--sandbox",
        "danger-full-access",
        "-c",
        "developer_instructions=" + json.dumps(task["messages"][0]["content"]),
        "-C",
        str(workspace),
        "-o",
        str(answer),
        "-",
    ]
    save(
        directory / "command.json",
        {
            "argv": command,
            "stdin": task["messages"][-1]["content"],
            "isolated_codex_home": str(home),
            "authentication": "existing ChatGPT login; credential file never copied into artifacts",
        },
    )
    start = time.monotonic()
    started = timestamp()
    process = None
    reason = None
    try:
        with (
            (directory / "codex.jsonl").open("wb") as log,
            (directory / "stderr.log").open("wb") as error,
        ):
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=log,
                stderr=error,
                env=environment,
                start_new_session=True,
            )
            try:
                async with asyncio.timeout(config.limits.wall_seconds):
                    await process.communicate(task["messages"][-1]["content"].encode())
            except TimeoutError:
                reason = "timeout"
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        if reason is None and process.returncode:
            reason = f"codex_exit_{process.returncode}"
        events = [
            json.loads(line)
            for line in (directory / "codex.jsonl").read_text().splitlines()
            if line.strip()
        ]
        usages = [
            e["usage"] for e in events if e.get("type") == "turn.completed" and e.get("usage")
        ]
        usage = {
            key: sum(u.get(key, 0) for u in usages) if usages else None
            for key in ["input_tokens", "output_tokens", "cached_input_tokens"]
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"] if usages else None
        usage["wall_seconds"] = time.monotonic() - start
        response = answer.read_text() if answer.exists() and reason is None else ""
        result = {
            "response": response,
            "usage": usage,
            "stop_reason": reason or "completed",
            "start_time": started,
            "end_time": timestamp(),
            "trajectory_reference": str(directory),
        }
        save(directory / "usage.json", usage)
        if (home / "sessions").exists():
            shutil.copytree(home / "sessions", directory / "native-sessions")
        return result
    finally:
        if process and process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        (home / "auth.json").unlink(missing_ok=True)
        shutil.rmtree(home)


def measured_usage(record):
    """Reported usage only; durable runtime reservations are not provider measurements."""
    directory = Path(record["run_artifact"])
    if record["system"] == "buffalo":
        with sqlite3.connect(directory / "state/history.sqlite3") as db:
            attempts = [
                json.loads(row[0]) for row in db.execute("SELECT usage FROM model_attempts")
            ]
        unknown = sum(bool(u.get("estimated_calls")) for u in attempts)
        known = [u for u in attempts if not u.get("estimated_calls")]
        counts = {
            key: sum(u.get(key, 0) for u in known)
            for key in ("input_tokens", "output_tokens", "cached_input_tokens")
        }
    else:
        usage = json.loads((directory / "agent-result.json").read_text())["usage"]
        events = [json.loads(line) for line in (directory / "codex.jsonl").read_text().splitlines()]
        interruptions = sum(
            event.get("type") == "error"
            and any(
                term in event.get("message", "").lower()
                for term in ("reconnecting", "stream disconnected", "connection reset")
            )
            for event in events
        )
        unknown = max(int(usage.get("total_tokens") is None), interruptions)
        counts = {
            key: usage.get(key) or 0
            for key in ("input_tokens", "output_tokens", "cached_input_tokens")
        }
    return {
        **{"known_" + key: value for key, value in counts.items()},
        "known_total_tokens": counts["input_tokens"] + counts["output_tokens"],
        "unknown_usage_attempts": unknown,
    }


def records_summary(records):
    n = len(records)
    return {
        "denominator": n,
        **{
            key: sum(bool(r[key]) for r in records)
            for key in ["functional_pass", "style_pass", "overall_pass"]
        },
        **{
            key: sum(r[key] for r in records) if all(r[key] is not None for r in records) else None
            for key in ["input_tokens", "output_tokens", "total_tokens", "estimated_cost"]
        },
        **{
            key: sum(r.get(key, r.get(key.removeprefix("known_")) or 0) for r in records)
            for key in (
                "known_input_tokens",
                "known_output_tokens",
                "known_total_tokens",
                "known_cached_input_tokens",
                "unknown_usage_attempts",
            )
        },
        "summed_task_wall_seconds": sum(r["wall_time_seconds"] for r in records),
        "median_task_wall_seconds": statistics.median(r["wall_time_seconds"] for r in records)
        if n
        else None,
    }


def audit_trace(record):
    directory = Path(record["run_artifact"])
    raw = json.loads((directory / "agent-result.json").read_text())
    usage = raw["usage"]
    incomplete = bool(
        usage.get("estimated_calls") or measured_usage(record)["unknown_usage_attempts"]
    )
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        assert record[key] == (None if incomplete else usage.get(key))
    if record["total_tokens"] is not None:
        assert record["total_tokens"] == record["input_tokens"] + record["output_tokens"]
    task = json.loads((directory / "task.json").read_text())
    assert file_digest(directory / "task.json") == record["prompt_sha256"]
    if record["system"] == "buffalo":
        config = json.loads((directory / "buffalo-config.json").read_text())
        assert config["task"]["adapter"] == "coding"
        assert config["provider"]["model"] == MODEL
        assert config["provider"]["parameters"]["reasoning_effort"] == REASONING
        with sqlite3.connect(directory / "state/history.sqlite3") as db:
            bodies = [
                row[0]
                for row in db.execute("SELECT model FROM model_requests WHERE purpose='agent'")
            ]
            assert bodies or record["stop_reason"] != "completed"
            assert all(model == MODEL for model in bodies)
        for line in (
            (directory / "provider-calls.jsonl").read_text().splitlines()
            if (directory / "provider-calls.jsonl").exists()
            else ()
        ):
            request = json.loads(line)["request"]
            if (
                request.get("request_kind") == "trajectory"
                and request.get("metadata", {}).get("purpose", "agent") == "agent"
            ):
                assert request["config"]["model"] == MODEL
                assert request["config"]["parameters"]["reasoning_effort"] == REASONING
    else:
        command = json.loads((directory / "command.json").read_text())
        assert command["stdin"] == task["messages"][-1]["content"]
        assert (
            "developer_instructions=" + json.dumps(task["messages"][0]["content"])
            in command["argv"]
        )
        assert "--ignore-user-config" in command["argv"]
        for path in (directory / "native-sessions").rglob("*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("type") == "turn_context":
                    assert row["payload"]["model"] == MODEL
                    assert row["payload"]["effort"] == REASONING
                if row.get("type") == "response_item":
                    text = json.dumps(row)
                    assert all(
                        term not in text
                        for term in (
                            "Coding APIs:",
                            "Coding decision support:",
                            "harness.refine",
                            "threadweave",
                            "agent_observe",
                        )
                    )
    return True


def audit(output, expected):
    rows = {
        system: [
            json.loads(p.read_text())
            for p in sorted((output / "full" / system).glob("*/result.json"))
        ]
        for system in ["buffalo", "codex"]
    }
    assert len(expected) == len(set(expected)) == 100
    for system, records in rows.items():
        ids = [r["task_id"] for r in records]
        assert len(ids) == len(set(ids)) == 100 and set(ids) == set(expected), (system, ids)
        for record in records:
            record.update(measured_usage(record))
            record["usage_complete"] = not record["unknown_usage_attempts"]
            if not record["usage_complete"]:
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    record[key] = None
            audit_trace(record)
            save(Path(record["run_artifact"]) / "result.json", record)
            grade = json.loads(Path(record["official_evaluation"]).read_text())["evaluation"]
            assert record["functional_pass"] == grade["test_passed"]
            assert record["style_pass"] == grade["style_passed"]
            assert record["overall_pass"] == grade["overall_passed"]
            assert record["overall_pass"] == (record["functional_pass"] and record["style_pass"])
    mapped = {s: {r["task_id"]: r for r in rows[s]} for s in rows}
    for task_id in expected:
        a, b = mapped["buffalo"][task_id], mapped["codex"][task_id]
        assert (
            a["prompt_sha256"] == b["prompt_sha256"]
            and a["workspace_hashes"] == b["workspace_hashes"]
        )
    paired = {"both_pass": [], "buffalo_only": [], "codex_only": [], "both_fail": []}
    for task_id in expected:
        a, b = mapped["buffalo"][task_id]["overall_pass"], mapped["codex"][task_id]["overall_pass"]
        paired[
            "both_pass" if a and b else "buffalo_only" if a else "codex_only" if b else "both_fail"
        ].append(task_id)
    summary = {s: records_summary(rows[s]) for s in rows}
    for system in rows:
        timing = output / f"{system}-phase-time.json"
        if timing.exists():
            summary[system]["phase_wall_seconds"] = json.loads(timing.read_text())["wall_seconds"]
    timing = output / "experiment-time.json"
    if timing.exists():
        summary["experiment_wall_seconds"] = json.loads(timing.read_text())["wall_seconds"]
    a, b = summary["buffalo"]["overall_pass"], summary["codex"]["overall_pass"]
    summary.update(
        paired=paired,
        absolute_percentage_points=a - b,
        relative_improvement_percent=100 * (a - b) / b if b else None,
        winner="BUFFALO WINS" if a > b else "CODEX WINS" if b > a else "TIE",
        integrity="PASS",
    )
    save(output / "summary.json", summary)
    for system in rows:
        (output / f"{system}-results.jsonl").write_text(
            "".join(json.dumps(mapped[system][task]) + "\n" for task in expected)
        )
    return summary


async def execute(args):
    output = await asyncio.to_thread(Path(args.output).resolve)
    output.mkdir(parents=True, exist_ok=True)
    gate = Path(args.generality_gate)
    if any(
        value != "PASS"
        for value in json.loads(await asyncio.to_thread(gate.read_text))["checks"].values()
    ):
        raise RuntimeError("Part A generality gate has not passed")
    source = await asyncio.to_thread(Path(args.source).resolve)
    commit = git(source, "rev-parse", "HEAD")
    setup = BenchmarkSetup(source=source, commit=commit, python=sys.executable)
    official = OfficialWorker(setup, "manyih-coding", output / "official")
    start = time.monotonic()
    try:
        provenance = await official.start()
        ids = provenance["task_ids"][:100]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise RuntimeError("Expected 100 unique canonical task IDs")
        save(output / "task-ids.json", ids)
        tasks = {task_id: await official.call("task", task_id=task_id) for task_id in ids}
        config = run_config()
        save(output / "buffalo-config.json", config.model_dump(mode="json"))
        save(
            output / "manifest.json",
            {
                "created_at": timestamp(),
                "official": provenance,
                "task_ids": ids,
                "model": MODEL,
                "reasoning": REASONING,
                "per_task_timeout": TIMEOUT,
                "concurrency": args.concurrency,
                "python": sys.version,
                "python_executable": sys.executable,
                "codex_version": (
                    await asyncio.to_thread(
                        subprocess.check_output, ["codex", "--version"], text=True
                    )
                ).strip(),
                "buffalo_git_commit": git(Path(__file__).parents[3], "rev-parse", "HEAD"),
                "generality_gate_sha256": file_digest(gate),
                "deviations": [
                    "Native Codex retains its own agent instructions/tools; Buffalo retains its production coding harness instructions/tools.",
                    "The official system prompt is supplied in Buffalo system instructions and native Codex developer_instructions; exact text is unchanged.",
                    "Both use native ChatGPT Responses defaults for output length and temperature; neither exposes a matching per-task provider output cap. Buffalo reserves 32768 output tokens per call and has a 3M-token runtime ceiling.",
                    "Buffalo uses three accounted transport attempts; native Codex retains native request/stream retry behavior. Neither system retries a completed failed task.",
                    "Buffalo refinement auxiliary calls use its production minimum supported reasoning setting; primary agent and child default reasoning is xhigh.",
                    "Official hidden assertions and expected styles are available only to the grading worker; both agent workspaces contain identical TASK.txt and .gitignore.",
                    "No cross-task harness state is shared. Baseline uses a fresh CODEX_HOME per task with only access to the existing login, no Buffalo memory/skills/configuration.",
                ],
            },
        )
        semaphore = asyncio.Semaphore(args.concurrency)

        async def one(phase, system, task_id):
            directory = output / phase / system / task_id
            result_path = directory / "result.json"
            if result_path.exists():
                return json.loads(result_path.read_text())
            async with semaphore:
                directory.mkdir(parents=True, exist_ok=True)
                task = tasks[task_id]
                save(directory / "task.json", task)
                workspace = directory / "workspace"
                if workspace.exists():
                    raise RuntimeError(
                        f"Incomplete attempt requires explicit recovery: {directory}"
                    )
                hashes = await asyncio.to_thread(prepare_workspace, workspace, task)
                began = time.monotonic()
                try:
                    if system == "buffalo":
                        raw = await run_buffalo(
                            config, task, directory, workspace=workspace, task_config=config.task
                        )
                    else:
                        raw = await baseline(directory, workspace, task, config)
                except Exception as exc:
                    usage_path = directory / "usage.json"
                    usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
                    raw = {
                        "response": "",
                        "usage": usage,
                        "stop_reason": f"{type(exc).__name__}: {exc}",
                    }
                save(directory / "agent-result.json", raw)
                (directory / "answer.txt").write_text(raw.get("response", ""))
                grade = await official.call(
                    "grade", task_id=task_id, response=raw.get("response", "")
                )
                save(directory / "official-grade.json", grade)
                evaluation = grade["evaluation"]
                usage = raw.get("usage", {})
                elapsed = usage.get("wall_seconds", time.monotonic() - began)
                record = {
                    "task_id": task_id,
                    "system": system,
                    "functional_pass": bool(evaluation["test_passed"]),
                    "style_pass": bool(evaluation["style_passed"]),
                    "overall_pass": bool(evaluation["overall_passed"]),
                    "wall_time_seconds": elapsed,
                    **{
                        key: None if usage.get("estimated_calls") else usage.get(key)
                        for key in ["input_tokens", "output_tokens", "total_tokens"]
                    },
                    "estimated_cost": None,
                    "failure_reason": None
                    if evaluation["overall_passed"]
                    else raw.get("stop_reason")
                    if raw.get("stop_reason") != "completed"
                    else evaluation.get("test_result")
                    if not evaluation["test_passed"]
                    else "instruction/style violation",
                    "run_artifact": str(directory),
                    "official_evaluation": str(directory / "official-grade.json"),
                    "prompt_sha256": file_digest(directory / "task.json"),
                    "workspace_hashes": hashes,
                    "model": MODEL,
                    "reasoning": REASONING,
                    "stop_reason": raw.get("stop_reason"),
                }
                save(result_path, record)
                print(json.dumps({"phase": phase, **record}), flush=True)
                return record

        # Smoke uses separate attempts and never contributes to the final denominators.
        for system in ["buffalo", "codex"]:
            smoke = [await one("smoke", system, task_id) for task_id in ids[:2]]
            answers = [
                await asyncio.to_thread((Path(r["run_artifact"]) / "answer.txt").read_text)
                for r in smoke
            ]
            if any(r["total_tokens"] is None for r in smoke) or not all(a.strip() for a in answers):
                save(output / "smoke-blocker.json", {"system": system, "records": smoke})
                raise RuntimeError(
                    f"{system} smoke did not produce auditable model answers; inspect artifacts before proceeding"
                )
            for record in smoke:
                audit_trace(record)
            save(output / f"smoke-{system}-metrics.json", records_summary(smoke))
        save(
            output / "smoke-gate.json",
            {"status": "PASS", "task_ids": ids[:2], "official_grading": True},
        )
        if args.smoke_only:
            return
        for system in ["buffalo", "codex"]:
            phase_start = time.monotonic()
            records = await asyncio.gather(*(one("full", system, task_id) for task_id in ids))
            grades = [
                json.loads(await asyncio.to_thread(Path(r["official_evaluation"]).read_text))
                for r in records
            ]
            summary = await official.call("summarize", grades=grades, profile=system)
            save(output / f"official-{system}-summary.json", summary)
            save(
                output / f"{system}-phase-time.json",
                {"wall_seconds": time.monotonic() - phase_start},
            )
        save(output / "experiment-time.json", {"wall_seconds": time.monotonic() - start})
        summary = audit(output, ids)
        print(json.dumps(summary), flush=True)
    finally:
        await official.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generality-gate", default="results/domain-generality/gate.json")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    # Shell tools in both harnesses resolve the same Python/dependency environment.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    if args.audit_only:
        output = Path(args.output).resolve()
        print(
            json.dumps(audit(output, json.loads((output / "task-ids.json").read_text())), indent=2)
        )
    else:
        asyncio.run(execute(args))


if __name__ == "__main__":
    main()
