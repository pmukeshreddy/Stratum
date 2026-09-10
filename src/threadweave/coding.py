"""Repository preparation, command evidence, and independent completion gates."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from .coding_config import coding_options
from .execution import executor
from .gitops import GitWorkspace
from .models import Verification, new_id, now
from .repository import confined, detect, is_test, symbols
from .test_evidence import machine_command, resolve_locations, structured

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
    report = context.runtime.store.directory / "test-reports" / (new_id() + ".xml")
    report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    framework, executed = machine_command(
        command, report, local=config.execution.backend == "local"
    )
    result = await executor(config.execution).run(
        context, executed, timeout_seconds=timeout_seconds or config.limits.tool_timeout_seconds
    )
    text = "\n".join(
        context.runtime.artifacts.load(context.session_id, result[k + "_artifact"])
        for k in ("stdout", "stderr")
    )
    evidence = resolve_locations(
        structured(text, framework, report.read_text() if report.is_file() else None),
        context.session.workspace.path,
        container=config.execution.backend != "local",
    )
    evidence_id = context.runtime.artifacts.put(
        context.session_id, evidence, source_event=context.source_event
    )
    if report.is_file():
        result["test_report_artifact"] = context.runtime.artifacts.put_bytes(
            context.session_id, report.read_bytes(), "application/xml"
        )
    result.update(
        {
            **evidence,
            "tests": evidence["tests"][:30],
            "structured_artifact": evidence_id,
            "requested_command": command,
        }
    )
    result["kind"] = kind
    context.runtime.store.event(
        context.session_id, "coding_command", result, parent=context.source_event
    )
    return result


def baseline(context):
    row = context.runtime.store.records.first(
        "coding_baselines", session_id=context.session_id, fields=("body",)
    )
    if not row:
        raise ValueError("Coding baseline is not prepared")
    return row["body"]


async def run_checks(context, kind, *, targets=None):
    config = context.runtime.store.config(context.session_id)
    # Standalone Environment actions don't require enrolling in a coding task,
    # capturing a baseline or invoking other commands first.
    commands = (
        baseline(context)["commands"].get(kind, [])
        if config.task.adapter == "coding"
        else getattr(coding_options(config.task), KINDS[kind])
        or detect(context.path("."))["suggested_commands"].get(kind, [])
    )
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
        options = coding_options(task)
        store = context.runtime.store
        row = store.records.first(
            "coding_baselines", session_id=context.session_id, fields=("body",)
        )
        if row:
            return row["body"]
        git = GitWorkspace(context)
        status = git.status()
        if options.repository and Path(options.repository).resolve() != git.root.resolve():  # noqa: ASYNC240 - bounded local admission metadata
            raise ValueError("options.repository must match the admitted workspace")
        if options.base_commit:
            from .gitops import git as git_command
            from .gitops import revision

            expected = git_command(
                git.root, "rev-parse", "--verify", revision(options.base_commit) + "^{commit}"
            ).strip()
            if status["head"] != expected:
                raise ValueError(
                    "Repository is not at options.base_commit; prepare the checkout explicitly"
                )
        if options.require_clean_baseline and status["status"].strip():
            raise ValueError(
                "Repository is dirty and require_clean_baseline is true; commit/stash yourself or explicitly disable this gate"
            )
        metadata = detect(git.root)
        commands = {
            kind: getattr(options, field) or metadata["suggested_commands"].get(kind, [])
            for kind, field in KINDS.items()
        }
        if options.require_tests and not commands["test"] and context.session.mode != "interactive":
            raise ValueError("Coding tasks require test commands: configure options.test_commands")
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
        if options.capture_baseline:
            for kind, configured in commands.items():
                result["results"][kind] = [
                    await run_command(context, cmd, kind="baseline_" + kind) for cmd in configured
                ]
            if options.benchmark:
                from .benchmarks import run_benchmark

                correctness = result["results"].get("test", [])
                result["benchmark"] = await run_benchmark(
                    context,
                    options.benchmark,
                    label="baseline",
                    correctness_passed=all(r["passed"] for r in correctness)
                    if correctness
                    else None,
                )
        store.records.insert("coding_baselines", {"session_id": context.session_id, "body": result})
        _, manifest, artifact, _, _ = context.runtime.environment.adapter(
            context.session_id
        ).mutations.scan(context)
        result["mutation_baseline"] = {
            "artifact": artifact,
            "owner": context.session_id,
            "state_id": manifest["state_id"],
        }
        store.records.update("coding_baselines", {"body": result}, session_id=context.session_id)
        context.runtime.environment.adapter(context.session_id).index(context.session_id).refresh()
        store.event(context.session_id, "coding_baseline", result, parent=context.source_event)
        return result

    async def verify(self, context, task):
        options = coding_options(task)
        original = baseline(context)
        commands_by_kind = {
            kind: getattr(options, field) or original["commands"].get(kind, [])
            for kind, field in KINDS.items()
        }
        observer = context.runtime.environment.adapter(context.session_id).mutations
        observer.reconcile(context, reason="before_verifier")
        git = GitWorkspace(context)
        _, observed = observer.previous(str(git.root.resolve()))
        origin = original.get("mutation_baseline")
        if origin:
            key = "verification:" + origin["artifact"]
            if key not in observer.baselines:
                observer.baselines[key] = observer.load(origin["owner"], origin["artifact"])
            initial = observer.baselines[key]
            cache_key = (origin["artifact"], observed["state_id"])
            cached = observer.verification_changes.get(context.session_id)
            if cached and cached[0] == cache_key:
                all_changed_paths = cached[1]
            else:
                all_changed_paths = {
                    p
                    for p in initial["files"].keys() | observed["files"].keys()
                    if initial["files"].get(p, {}).get("value")
                    != observed["files"].get(p, {}).get("value")
                }
                observer.verification_changes[context.session_id] = (cache_key, all_changed_paths)
            changed_paths = all_changed_paths.copy()
        else:
            changed_paths = (
                None  # Migrated sessions without an observation baseline reconcile conservatively.
            )
        violations, results, regressions = [], {}, []
        prior = git.checkpoint(original["checkpoint_id"])["manifest"]
        governed_paths = set(git.files()) | set(prior)
        if changed_paths is not None:
            changed_paths &= governed_paths
        if options.require_tests and not commands_by_kind["test"]:
            violations.append("No test commands configured; coding completion cannot be verified")
        patch = git.diff(original["checkpoint_id"], paths=changed_paths)
        if options.require_change and not patch.strip():
            violations.append("A nonempty change is required")
        if observed["head"] != original["git"]["head"]:
            violations.append("Repository HEAD changed; automatic commits are not permitted")
        for relative in (
            changed_paths if changed_paths is not None else set(git.files()) | set(prior)
        ):
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
            if changed_paths is not None and relative not in changed_paths:
                continue
            path = confined(git.root, relative)
            if options.prohibit_test_deletion and not path.is_file():
                violations.append(f"Test file deleted: {relative}")
            elif path.is_file():
                from .repository import digest

                if options.protect_tests and digest(path.read_bytes()) != recorded["hash"]:
                    violations.append(f"Protected test modified: {relative}")
                if options.prohibit_test_deletion:
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
        for required in options.required_files:
            if not confined(git.root, required).is_file():
                violations.append(f"Required file missing: {required}")
        for kind, commands in commands_by_kind.items():
            results[kind] = []
            for i, command in enumerate(commands):
                result = await run_command(context, command, kind="verify_" + kind)
                results[kind].append(result)
                previous = original["results"].get(kind, [])
                old = (
                    previous[i]
                    if i < len(previous)
                    and i < len(original["commands"].get(kind, []))
                    and original["commands"][kind][i] == command
                    else None
                )
                if not result["passed"]:
                    old_names = {f["name"] for f in old["failures"] if f["name"]} if old else set()
                    names = {f["name"] for f in result["failures"] if f["name"]}
                    known = bool(old and not old["passed"] and names and names <= old_names)
                    if not (options.allow_baseline_failures and known):
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
        if options.benchmark:
            from .benchmarks import run_benchmark

            if options.benchmark.required_improvement and not (original.get("benchmark") or {}).get(
                "passed"
            ):
                violations.append(
                    "A valid baseline measurement is required to accept performance improvement"
                )
            measured = await run_benchmark(
                context,
                options.benchmark,
                reference=original["benchmark"],
                correctness_passed=all(
                    r["passed"] for k, rows in results.items() if k != "benchmark" for r in rows
                ),
            )
            if not measured["passed"]:
                violations.append("Benchmark correctness/performance threshold failed")
        # The entry boundary already performed a full trust reconciliation. A
        # healthy watch stream fences command effects; uncertainty still forces
        # full reconciliation. Never repeat an unchanged full scan just for exit.
        observer.reconcile(context, reason="verification_effects")
        _, final = observer.previous(str(git.root.resolve()))
        if final["state_id"] != observed["state_id"] and any(
            final["files"].get(p, {}).get("value") != observed["files"].get(p, {}).get("value")
            for p in governed_paths | set(git.files())
        ):
            violations.append("Workspace changed during verification; retry against stable inputs")
        if origin := original.get("mutation_baseline"):
            stable = final["state_id"] == observed["state_id"]
            observed = initial
            if not stable:
                governed_paths = set(git.files()) | set(prior)
            for path in (
                all_changed_paths if stable else observed["files"].keys() | final["files"].keys()
            ):
                if observed["files"].get(path, {}).get("value") == final["files"].get(path, {}).get(
                    "value"
                ):
                    continue
                if (
                    path in governed_paths
                    and not any(fnmatch.fnmatch(path, p) for p in task.allowed_paths)
                ) or any(fnmatch.fnmatch(path, p) for p in task.forbidden_paths):
                    message = f"Forbidden modification: {path}"
                    if message not in violations:
                        violations.append(message)
        aid = context.runtime.artifacts.put_bytes(context.session_id, patch.encode(), "text/x-diff")
        verification = Verification(
            passed=not violations and not regressions,
            details={
                "violations": violations,
                "regressions": regressions,
                "results": results,
                "patch_artifact": aid,
                "benchmark": measured,
                "observed_final_state": final["state_id"],
                "diagnostic_oracle": verifier_evidence(context, original, results, patch),
            },
            metrics={
                "diff_bytes": len(patch.encode()),
                "checks": sum(map(len, results.values())),
                "violations": len(violations),
                "regressions": len(regressions),
            },
        )
        store = context.runtime.store
        store.records.insert(
            "final_verifications",
            {
                "id": new_id(),
                "session_id": context.session_id,
                "created_at": now(),
                "passed": verification.passed,
                "body": verification.model_dump(mode="json"),
            },
        )
        return verification


def verifier_evidence(context, original, results, patch):
    from .diagnostics import localize

    failures, existing, localized = [], [], []
    for kind, rows in results.items():
        prior = original["results"].get(kind, [])
        for i, result in enumerate(rows):
            known = {f["name"] for f in prior[i].get("failures", [])} if i < len(prior) else set()
            for failure in result.get("failures", []):
                entry = {
                    **failure,
                    "kind": kind,
                    "artifact": result["structured_artifact"],
                    "baseline_state": "existing" if failure["name"] in known else "new_or_unknown",
                }
                (existing if failure["name"] in known else failures).append(entry)
            if not result["passed"]:
                localized.extend(localize(context, result)["evidence"])
    return {
        "new_or_unclassified_failures": failures[:8],
        "baseline_existing_failures": existing[:5],
        "source_evidence": localized[:5],
        "diff_excerpt": patch[:3000],
        "complete_results_external": True,
        "independent": True,
    }
