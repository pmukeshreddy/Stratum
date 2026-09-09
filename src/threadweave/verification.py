"""Continuous diagnostics and targeted checks; only level 3 can accept completion."""

from __future__ import annotations

import ast
import json
from pathlib import PurePath

from .models import Verification, new_id
from .tools import ToolContext


def changed_sources(runtime, sid, effects):
    """Test-runner caches remain in provenance, but do not request another test run."""
    ignored = set(runtime.store.config(sid).verification.generated_path_parts)
    files = set()
    for effect in effects:
        body = effect["payload"]
        if body.get("truncated"):
            body = runtime.artifacts.load(sid, body["observation_artifact"])
        files.update(
            name for name in body.get("files", {}) if not ignored.intersection(PurePath(name).parts)
        )
    return files


def concise(result, policy):
    """Keep diagnostics actionable; full verifier receipts remain durable artifacts."""
    data = result.model_dump() if hasattr(result, "model_dump") else result
    if not isinstance(data, dict):
        return {"detail": str(data)[: policy.diagnostic_chars]}

    def trim(value):
        if isinstance(value, str):
            return value[: policy.diagnostic_chars]
        if isinstance(value, list):
            return [trim(v) for v in value[: policy.failure_items]]
        if isinstance(value, dict):
            return {
                k: trim(v)
                for k, v in value.items()
                if k not in {"stdout", "stderr", "tests", "traceback"}
            }
        return value

    return trim(data)


class VerificationScheduler:
    def __init__(self, runtime):
        self.runtime = runtime

    async def run(self, sid, parent, *, level=None):
        runtime, store = self.runtime, self.runtime.store
        config = store.config(sid)
        policy = config.verification
        if level == 3:
            return await runtime._verify(sid, parent)
        if not policy.continuous and level is None:
            return None
        prior = store.events(sid, kind="verification_result", limit=1)
        after = prior[-1]["seq"] if prior else 0
        effects = list(store.iter_events(sid, kind="workspace_effects", after=after))
        files = changed_sources(runtime, sid, effects)
        errors = list(store.iter_events(sid, kind="python_error", after=after))
        level = level or 1
        ctx = ToolContext(runtime, sid, new_id(), parent)
        failures = []
        for name in sorted(files):
            try:
                path = ctx.path(name)
                if not path.is_file() or path.suffix not in {".py", ".pyi", ".json"}:
                    continue
                text = path.read_text()
                if path.suffix == ".json":
                    json.loads(text)
                else:
                    ast.parse(text, filename=name)
            except (SyntaxError, ValueError, OSError, UnicodeError) as exc:
                failures.append(
                    {"path": name, "line": getattr(exc, "lineno", None), "message": str(exc)}
                )
        for error in errors:
            failures.append(
                {"source_event": error["id"], **error["payload"].get("result", {}).get("error", {})}
            )
        result = {"level": 1, "passed": not failures, "files": sorted(files), "failures": failures}
        marker = store.event(sid, "verification_result", result, parent=parent)
        if failures:
            self.publish(sid, marker, result)
        # Dirty evidence is accumulated since the last targeted run, not lost on
        # intervening syntax checks. Unchanged state never triggers another suite.
        targeted = store.events(sid, kind="verification_targeted", limit=1)
        last_seq = targeted[-1]["seq"] if targeted else 0
        last_turn = targeted[-1]["payload"]["turn"] if targeted else -policy.targeted_every_turns
        dirty = list(store.iter_events(sid, kind="workspace_effects", after=last_seq))
        changed = changed_sources(runtime, sid, dirty)
        due = store.session(sid).turns - last_turn >= policy.targeted_every_turns
        if not failures and ((changed and due) or level == 2):
            commands = list(policy.targeted_commands)
            if not commands and config.task.adapter == "coding" and changed:
                commands = self.pytest_commands(ctx, changed)
            if commands:
                from .coding import run_command

                results = [await run_command(ctx, command, kind="targeted") for command in commands]
                result = {
                    "level": 2,
                    "passed": all(r["passed"] for r in results),
                    "files": sorted(changed),
                    "results": results,
                    "failures": [f for r in results for f in r.get("failures", [])],
                }
                marker = store.event(
                    sid,
                    "verification_targeted",
                    {"turn": store.session(sid).turns, "commands": commands},
                    parent=parent,
                )
                self.publish(sid, marker, result)
        return result

    def pytest_commands(self, context, changed):
        from .coding_config import coding_options
        from .test_selection import related

        candidates = related(
            context,
            files=sorted(changed),
            limit=context.runtime.store.config(context.session_id).verification.max_target_files,
        )
        targets = [
            r["target"]
            for r in candidates["selections"]
            if r["target"].split("::")[0].endswith(".py")
        ]
        commands = []
        for command in coding_options(
            context.runtime.store.config(context.session_id).task
        ).test_commands:
            if "pytest" not in command or not targets:
                continue
            # Preserve interpreter, flags and their values; replace only existing
            # repository path selectors. Unknown command syntaxes need configuration.
            index = command.index("pytest") + 1
            prefix, args = command[:index], command[index:]
            flags, operand, supported = [], False, True
            value_options = {
                "-k",
                "-m",
                "-c",
                "-o",
                "--override-ini",
                "--ignore",
                "--ignore-glob",
                "--deselect",
                "--confcutdir",
                "--rootdir",
                "--basetemp",
                "--junitxml",
                "--junit-xml",
                "--junit-prefix",
                "--capture",
                "--tb",
                "--maxfail",
                "--import-mode",
                "--durations",
                "--color",
                "--code-highlight",
            }
            switches = {
                "-q",
                "-qq",
                "-v",
                "-vv",
                "-x",
                "-s",
                "-ra",
                "-rA",
                "--disable-warnings",
                "--strict-markers",
                "--strict-config",
                "--lf",
                "--ff",
                "--no-header",
                "--no-summary",
            }
            for arg in args:
                if operand:
                    flags.append(arg)
                    operand = False
                elif arg in value_options:
                    flags.append(arg)
                    operand = True
                elif arg.startswith("-"):
                    if arg not in switches and "=" not in arg:
                        supported = False
                        break
                    flags.append(arg)
                elif not context.path(arg.split("::")[0]).exists():
                    supported = False
                    break
            if not supported or operand:
                continue  # Plugin-specific syntax uses explicit targeted_commands.
            commands.append([*prefix, *flags, *targets])
        return commands

    def publish(self, sid, parent, result):
        runtime = self.runtime
        artifact = runtime.artifacts.put(sid, result, source_event=parent)
        evidence = {**concise(result, runtime.store.config(sid).verification), "artifact": artifact}
        eid = runtime.store.event(sid, "verification_evidence", evidence, parent=parent)
        runtime.store.add_context(
            sid,
            eid,
            [{"role": "user", "content": "Verification evidence: " + json.dumps(evidence)}],
        )
        if not result["passed"]:
            runtime.retain_failure(sid, eid, Verification(passed=False, details=evidence))
        return evidence
