"""Repository preparation, command evidence, and independent completion gates."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from .diagnostics import parse_diagnostics
from .execution import executor
from .gitops import GitWorkspace
from .models import Verification, new_id, now
from .repository import confined, detect, is_test, symbols
from .storage import encode

KINDS = {
    "test": "test_commands",
    "build": "build_commands",
    "lint": "lint_commands",
    "typecheck": "typecheck_commands",
    "benchmark": "benchmark_commands",
}


async def run_command(context, command, *, kind="command", timeout_seconds=None):
    config = context.runtime.store.config(context.session_id)
    if "process" not in config.permissions:
        raise PermissionError("Coding commands require process permission")
    result = await executor(config.execution).run(
        context, command, timeout_seconds=timeout_seconds or config.limits.tool_timeout_seconds
    )
    text = "\n".join(
        context.runtime.artifacts.load(context.session_id, result[k + "_artifact"])
        for k in ("stdout", "stderr")
    )
    result.update(parse_diagnostics(text))
    result["kind"] = kind
    context.runtime.store.event(
        context.session_id, "coding_command", result, parent=context.source_event
    )
    return result


def baseline(context):
    row = context.runtime.store.db.execute(
        "SELECT body FROM coding_baselines WHERE session_id=?", (context.session_id,)
    ).fetchone()
    if not row:
        raise ValueError("Coding baseline is not prepared")
    return json.loads(row[0])


async def run_checks(context, kind, *, targets=None):
    commands = baseline(context)["commands"].get(kind, [])
    if not commands:
        raise ValueError(f"No {kind} commands configured; add task.{KINDS[kind]}")
    if targets:
        for target in targets:
            if target.startswith("-"):
                raise ValueError("Test targets cannot be command-line options")
            context.path(target.split("::")[0])
    results = [await run_command(context, [*cmd, *(targets or [])], kind=kind) for cmd in commands]
    return {
        "passed": all(r["passed"] for r in results),
        "results": results,
        "failures": [f for r in results for f in r["failures"]][:30],
        "diagnostics": [d for r in results for d in r["diagnostics"]][:30],
    }


class CodingTask:
    async def prepare(self, context, task):
        store = context.runtime.store
        row = store.db.execute(
            "SELECT body FROM coding_baselines WHERE session_id=?", (context.session_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        git = GitWorkspace(context)
        status = git.status()
        if task.repository and Path(task.repository).resolve() != git.root.resolve():  # noqa: ASYNC240 - bounded local admission metadata
            raise ValueError("task.repository must match the admitted workspace")
        if task.base_commit:
            from .gitops import git as git_command
            from .gitops import revision

            expected = git_command(
                git.root, "rev-parse", "--verify", revision(task.base_commit) + "^{commit}"
            ).strip()
            if status["head"] != expected:
                raise ValueError(
                    "Repository is not at task.base_commit; prepare the checkout explicitly"
                )
        if task.require_clean_baseline and status["status"].strip():
            raise ValueError(
                "Repository is dirty and require_clean_baseline is true; commit/stash yourself or explicitly disable this gate"
            )
        metadata = detect(git.root)
        commands = {
            kind: getattr(task, field) or metadata["suggested_commands"].get(kind, [])
            for kind, field in KINDS.items()
        }
        if task.require_tests and not commands["test"] and context.session.mode != "interactive":
            raise ValueError("Coding tasks require test commands: configure task.test_commands")
        if any(commands.values()) and "process" not in store.config(context.session_id).permissions:
            raise PermissionError("Coding baseline/verification requires process permission")
        checkpoint = git.snapshot("coding-baseline")
        tests = {}
        for relative, entry in git.checkpoint(checkpoint)["manifest"].items():
            if is_test(relative) and "symlink" not in entry:
                tests[relative] = {
                    "hash": entry["hash"],
                    "symbols": symbols(
                        git.content(entry).decode(errors="replace"),
                        "python" if relative.endswith(".py") else "text",
                    )["symbols"],
                }
        result = {
            "git": status,
            "metadata": metadata,
            "commands": commands,
            "checkpoint_id": checkpoint,
            "tests": tests,
            "results": {},
            "benchmark": None,
        }
        if task.capture_baseline:
            for kind, configured in commands.items():
                result["results"][kind] = [
                    await run_command(context, cmd, kind="baseline_" + kind) for cmd in configured
                ]
            if task.benchmark:
                from .benchmarks import run_benchmark

                correctness = result["results"].get("test", [])
                result["benchmark"] = await run_benchmark(
                    context,
                    task.benchmark,
                    label="baseline",
                    correctness_passed=all(r["passed"] for r in correctness)
                    if correctness
                    else None,
                )
        store.db.execute(
            "INSERT INTO coding_baselines VALUES(?,?)", (context.session_id, encode(result))
        )
        context.runtime.index(context.session_id).refresh()
        store.event(context.session_id, "coding_baseline", result, parent=context.source_event)
        return result

    async def verify(self, context, task):
        original = baseline(context)
        git = GitWorkspace(context)
        violations, results, regressions = [], {}, []
        if task.require_tests and not original["commands"]["test"]:
            violations.append("No test commands configured; coding completion cannot be verified")
        patch = git.diff(original["checkpoint_id"])
        if task.require_change and not patch.strip():
            violations.append("A nonempty change is required")
        if git.status()["head"] != original["git"]["head"]:
            violations.append("Repository HEAD changed; automatic commits are not permitted")
        prior = git.checkpoint(original["checkpoint_id"])["manifest"]
        for relative in set(git.files()) | set(prior):
            entry, path = prior.get(relative), git.root / relative
            if entry and "symlink" in entry:
                import os

                changed = not path.is_symlink() or os.readlink(path) != entry["symlink"]
            elif path.is_symlink():
                changed = True
            else:
                before = git.content(entry) if entry else None
                after = path.read_bytes() if path.is_file() else None
                changed = before != after
            if changed and (
                not any(fnmatch.fnmatch(relative, p) for p in task.allowed_paths)
                or any(fnmatch.fnmatch(relative, p) for p in task.forbidden_paths)
            ):
                violations.append(f"Forbidden modification: {relative}")
        for relative, recorded in original["tests"].items():
            path = confined(git.root, relative)
            if task.prohibit_test_deletion and not path.is_file():
                violations.append(f"Test file deleted: {relative}")
            elif path.is_file():
                from .repository import digest

                if task.protect_tests and digest(path.read_bytes()) != recorded["hash"]:
                    violations.append(f"Protected test modified: {relative}")
                if task.prohibit_test_deletion:
                    old = {s["name"] for s in recorded["symbols"]}
                    current = {
                        s["name"]
                        for s in symbols(
                            path.read_text(), "python" if relative.endswith(".py") else "text"
                        )["symbols"]
                    }
                    if old - current:
                        violations.append(
                            f"Test definitions deleted: {relative}: {sorted(old - current)}"
                        )
        for required in task.required_files:
            if not confined(git.root, required).is_file():
                violations.append(f"Required file missing: {required}")
        for kind, commands in original["commands"].items():
            results[kind] = []
            for i, command in enumerate(commands):
                result = await run_command(context, command, kind="verify_" + kind)
                results[kind].append(result)
                previous = original["results"].get(kind, [])
                old = previous[i] if i < len(previous) else None
                if not result["passed"]:
                    old_names = {f["name"] for f in old["failures"] if f["name"]} if old else set()
                    names = {f["name"] for f in result["failures"] if f["name"]}
                    known = bool(old and not old["passed"] and names and names <= old_names)
                    if not (task.allow_baseline_failures and known):
                        violations.append(f"{kind} command failed: {command}")
                    if old and old["passed"] or old and names - old_names:
                        regressions.append(
                            {
                                "kind": kind,
                                "command": command,
                                "new_failures": sorted(names - old_names),
                            }
                        )
        measured = None
        if task.benchmark:
            from .benchmarks import run_benchmark

            if task.benchmark.required_improvement and not (original.get("benchmark") or {}).get(
                "passed"
            ):
                violations.append(
                    "A valid baseline measurement is required to accept performance improvement"
                )
            measured = await run_benchmark(
                context,
                task.benchmark,
                reference=original["benchmark"],
                correctness_passed=all(
                    r["passed"] for k, rows in results.items() if k != "benchmark" for r in rows
                ),
            )
            if not measured["passed"]:
                violations.append("Benchmark correctness/performance threshold failed")
        aid = context.runtime.artifacts.put_bytes(context.session_id, patch.encode(), "text/x-diff")
        verification = Verification(
            passed=not violations and not regressions,
            details={
                "violations": violations,
                "regressions": regressions,
                "results": results,
                "patch_artifact": aid,
                "benchmark": measured,
            },
            metrics={
                "diff_bytes": len(patch.encode()),
                "checks": sum(map(len, results.values())),
                "violations": len(violations),
                "regressions": len(regressions),
            },
        )
        store = context.runtime.store
        store.db.execute(
            "INSERT INTO final_verifications VALUES(?,?,?,?,?)",
            (
                new_id(),
                context.session_id,
                now(),
                verification.passed,
                verification.model_dump_json(),
            ),
        )
        return verification
