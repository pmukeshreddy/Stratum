"""Run externally supplied workloads through the production runtime; report observed outcomes only."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from .artifacts import atomic_write
from .coding import run_command
from .editing import unified_changes
from .gitops import GitWorkspace, git, revision
from .models import BenchmarkConfig, Record, Workspace, new_id, now
from .repository import EXCLUDED, confined
from .storage import encode


class Instance(Record):
    id: str
    adapter: Literal["repository_issue", "long_context", "kernel"]
    repository: str
    objective: str
    base_commit: str | None = None
    test_commands: list[list[str]] = Field(default_factory=list)
    build_commands: list[list[str]] = Field(default_factory=list)
    lint_commands: list[list[str]] = Field(default_factory=list)
    typecheck_commands: list[list[str]] = Field(default_factory=list)
    verifier_commands: list[list[str]] = Field(default_factory=list)
    test_patch: str | None = None
    context_bundle: list[str] = Field(default_factory=list)
    benchmark: BenchmarkConfig | None = None
    required_package: str | None = None


def prepare_instance(instance, directory, config, *, base_directory):
    import importlib.util

    if instance.required_package and importlib.util.find_spec(instance.required_package) is None:
        raise ValueError(
            f"Install the externally supplied benchmark package: {instance.required_package}"
        )
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = directory / "repository"
    source = instance.repository
    if not source.startswith(("https://", "ssh://", "git@")):
        path = (base_directory / source).resolve()
        if not path.is_dir():
            raise ValueError(f"External task repository is missing: {path}")
        source = str(path)
    if (
        instance.adapter == "repository_issue"
        or Path(source, ".git").exists()
        or source.startswith(("https://", "ssh://", "git@"))
    ):
        git(directory, "clone", "--no-hardlinks", "--quiet", "--", source, str(destination))
        if instance.base_commit:
            git(destination, "checkout", "--detach", revision(instance.base_commit))
    else:
        shutil.copytree(
            source, destination, ignore=shutil.ignore_patterns(*EXCLUDED), symlinks=True
        )
        git(destination, "init", "--quiet")
        git(destination, "add", "--all")
        git(
            destination,
            "-c",
            "user.name=Threadweave",
            "-c",
            "user.email=local@localhost",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "External workload input",
        )
    instruction = instance.objective
    for i, supplied in enumerate(instance.context_bundle):
        path = (base_directory / supplied).resolve()
        if not path.is_file():
            raise ValueError(f"External context bundle file missing: {path}")
        target = destination / ".task_context" / f"{i}-{path.name}"
        atomic_write(target, path.read_bytes())
        instruction += f"\nAdditional task context: {target.relative_to(destination)}"
    resolved = config.model_copy(deep=True)
    resolved.task.adapter = "coding"
    resolved.task.repository = str(destination)
    resolved.task.base_commit = None
    # Context bundles are explicit untracked inputs, preserved in the baseline.
    resolved.task.require_clean_baseline = not bool(instance.context_bundle)
    for key in ("test_commands", "build_commands", "lint_commands", "typecheck_commands"):
        supplied = getattr(instance, key)
        if supplied:
            setattr(resolved.task, key, supplied)
    if instance.adapter == "kernel" and (
        not instance.build_commands or not instance.test_commands or not instance.benchmark
    ):
        raise ValueError(
            "Kernel instances require real build_commands, test_commands and benchmark configuration"
        )
    if instance.benchmark:
        resolved.task.benchmark = instance.benchmark
    return destination, instruction, resolved


async def external_verify(runtime, sid, instance, base_directory):
    """Apply evaluator-only tests to a private copy, never to the delivered agent patch."""
    from .tools import ToolContext

    event = runtime.store.event(sid, "external_verifier_started", {"instance": instance.id})
    context = ToolContext(runtime, sid, new_id(), event)
    workspace = GitWorkspace(context)
    checkpoint = workspace.snapshot("external-evaluation-input")
    isolated = workspace.isolate(checkpoint)
    if instance.test_patch:
        file = (base_directory / instance.test_patch).resolve()
        if not file.is_file():
            raise ValueError(f"External test patch missing: {file}")

        def read(path):
            p = confined(isolated, path)
            return p.read_bytes() if p.is_file() else None

        changes = unified_changes(file.read_text(), read)
        for relative, content in changes.items():
            path = confined(isolated, relative)
            if content is None:
                path.unlink()
            else:
                atomic_write(path, content)
    context.workspace_override = Workspace(path=str(isolated))
    commands = instance.verifier_commands or instance.test_commands
    if not commands:
        raise ValueError("External final evaluation needs verifier_commands or test_commands")
    results = [await run_command(context, c, kind="external_verifier") for c in commands]
    passed = all(r["passed"] for r in results)
    runtime.store.event(
        sid, "external_verifier_result", {"passed": passed, "results": results}, parent=event
    )
    return {"passed": passed, "results": results}


def metrics(runtime, sid):
    store = runtime.store
    events = [
        dict(r)
        for r in store.db.execute("SELECT * FROM events WHERE root_id=? ORDER BY seq", (sid,))
    ]
    counts = Counter(e["type"] for e in events)
    commands = [json.loads(e["payload"]) for e in events if e["type"] == "coding_command"]
    failed_commands = Counter(encode(c["command"]) for c in commands if not c["passed"])
    fingerprints = [json.loads(e["payload"]) for e in events if e["type"] == "action_fingerprint"]
    children = [
        json.loads(r[0])
        for r in store.db.execute(
            "SELECT c.body FROM candidates c JOIN sessions s ON c.child_id=s.id WHERE s.root_id=?",
            (sid,),
        )
    ]
    verification = store.db.execute(
        "SELECT body FROM final_verifications WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (sid,),
    ).fetchone()
    details = json.loads(verification[0]) if verification else None
    edits = [json.loads(e["payload"]) for e in events if e["type"] == "code_edit"]
    model_time = 0
    by_id = {e["id"]: e for e in events}
    for e in events:
        if e["type"] == "model_response" and e["parent_event_id"] in by_id:
            model_time += e["timestamp"] - by_id[e["parent_event_id"]]["timestamp"]
    return {
        **store.usage(sid, tree=True).model_dump(),
        "wall_time": runtime._elapsed(sid),
        "context_compactions": counts["context_compaction"],
        "retrieval_calls": counts["history_retrieval"],
        "experiments": counts["experiment_created"],
        "experiment_runs": counts["experiment_conclusion"],
        "experiment_successes": sum(
            json.loads(e["payload"]).get("passed", False)
            for e in events
            if e["type"] == "experiment_conclusion"
        ),
        "tests_runs": sum("test" in c["kind"] for c in commands),
        "build_runs": sum("build" in c["kind"] for c in commands),
        "failed_command_repetitions": sum(max(0, n - 1) for n in failed_commands.values()),
        "repeated_action_rate": sum(f["repetitions"] > 1 for f in fingerprints)
        / max(1, len(fingerprints)),
        "subagent_results_consumed": sum(c.get("consumed", False) for c in children),
        "subagent_patches_accepted": sum(c.get("accepted", False) for c in children),
        "subagent_usefulness": children,
        "verifier_failures": sum(
            not json.loads(e["payload"]).get("passed", True)
            for e in events
            if e["type"] == "verifier_result"
        ),
        "model_seconds": model_time,
        "tool_seconds": sum(
            e["timestamp"] - by_id[e["parent_event_id"]]["timestamp"]
            for e in events
            if e["type"] == "tool_result"
            and e["parent_event_id"] in by_id
            and by_id[e["parent_event_id"]]["type"] == "tool_call"
            and not json.loads(by_id[e["parent_event_id"]]["payload"]).get("from_python")
            and not json.loads(e["payload"]).get("recovered")
        ),
        "command_seconds": sum(
            json.loads(e["payload"]).get("duration", 0)
            for e in events
            if e["type"] == "execution_result"
        ),
        "code_churn_files": sum(len(e.get("files", {})) for e in edits),
        "observed_external_changed_files": sum(
            len(json.loads(e["payload"]).get("files", {}))
            for e in events
            if e["type"] == "workspace_effects"
        ),
        "reverted_edits": sum(bool(e.get("rollback_of")) for e in edits),
        "final_diff_size": details["metrics"]["diff_bytes"] if details else None,
        "benchmark_metrics": details["details"].get("benchmark") if details else None,
    }


async def evaluate(tasks, config, directory, *, repetitions=1, seed=0, output, providers=None):
    from .runtime import Runtime
    from .tools import ToolContext

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    tasks, directory, output = (
        Path(tasks).resolve(),  # noqa: ASYNC240 - local CLI configuration paths
        Path(directory).resolve(),  # noqa: ASYNC240 - local CLI configuration paths
        Path(output).resolve(),  # noqa: ASYNC240 - local CLI configuration paths
    )
    if not tasks.is_file():
        raise ValueError(f"Supply an external workload JSON/JSONL file: {tasks}")
    raw = tasks.read_text()
    instances = (
        json.loads(raw)
        if tasks.suffix == ".json"
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )
    if not isinstance(instances, list):
        instances = [instances]
    instances = [Instance.model_validate(i) for i in instances]
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for instance in instances:
        for repetition in range(repetitions):
            identifier, started = new_id(), now()
            run_dir = directory / "evaluations" / identifier
            runtime = None
            result = {
                "id": identifier,
                "instance_id": instance.id,
                "adapter": instance.adapter,
                "repetition": repetition,
                "seed": seed + repetition,
                "seed_applied_to_model": False,
                "config": config.model_dump(mode="json"),
                "solved": False,
            }
            try:
                workspace, instruction, resolved = prepare_instance(
                    instance, run_dir, config, base_directory=tasks.parent
                )
                runtime = Runtime(run_dir / "state", providers=providers)
                session = runtime.create(instruction, workspace, config=resolved)
                result.update(
                    session_id=session.id,
                    config_id=session.config_id,
                    resolved_config=runtime.store.config(session.id).model_dump(mode="json"),
                )
                await runtime.start()
                finished = await runtime.wait(session.id, timeout=resolved.limits.wall_seconds + 30)
                # Final supplied verifier is separate from the model's completion request.
                verified = await external_verify(runtime, session.id, instance, tasks.parent)
                result.update(
                    solved=finished.outcome == "completed" and verified["passed"],
                    outcome=finished.outcome,
                    verifier_score=1.0 if verified["passed"] else 0.0,
                    external_verification=verified,
                    metrics=metrics(runtime, session.id),
                )
                patch = GitWorkspace(
                    ToolContext(runtime, session.id, new_id(), "evaluation")
                ).diff()
                patch_file = run_dir / "final.patch"
                atomic_write(patch_file, patch.encode())
                result["patch_path"] = str(patch_file)
            except Exception as exc:
                result["error"] = {
                    "category": "environment" if runtime is None else "evaluation",
                    "message": str(exc),
                }
            finally:
                result["elapsed_seconds"] = now() - started
                if runtime:
                    runtime.store.db.execute(
                        "INSERT INTO eval_runs VALUES(?,?,?,?)",
                        (identifier, result.get("session_id"), now(), encode(result)),
                    )
                    await runtime.shutdown()
                fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as stream:
                    stream.write(encode(result) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            results.append(result)
    return {
        "runs": len(results),
        "solved": sum(r["solved"] for r in results),
        "output": str(output),
    }


def analyze(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    solved = sum(r["solved"] for r in rows)
    totals = {
        key: sum(r.get("metrics", {}).get(key, 0) or 0 for r in rows)
        for key in (
            "cost",
            "turns",
            "tool_calls",
            "subagent_count",
            "subagent_patches_accepted",
            "experiments",
            "experiment_successes",
            "context_compactions",
            "retrieval_calls",
            "failed_command_repetitions",
            "verifier_failures",
            "model_seconds",
            "tool_seconds",
            "code_churn_files",
            "reverted_edits",
        )
    }
    return {
        "runs": len(rows),
        "solved": solved,
        "success_rate": solved / len(rows) if rows else None,
        "cost_per_solved": totals["cost"] / solved if solved else None,
        "turns_per_solved": totals["turns"] / solved if solved else None,
        "tool_calls_per_solved": totals["tool_calls"] / solved if solved else None,
        "mean_repeated_action_rate": sum(
            r.get("metrics", {}).get("repeated_action_rate", 0) for r in rows
        )
        / len(rows)
        if rows
        else None,
        "subagent_acceptance_rate": totals["subagent_patches_accepted"] / totals["subagent_count"]
        if totals["subagent_count"]
        else None,
        "experiment_success_rate": totals["experiment_successes"]
        / sum(r.get("metrics", {}).get("experiment_runs", 0) for r in rows)
        if any(r.get("metrics", {}).get("experiment_runs") for r in rows)
        else None,
        "totals": totals,
        "note": "Descriptive evidence only; no causal claims about features.",
    }
