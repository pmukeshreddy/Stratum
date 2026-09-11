"""Host-owned completion gates, ported from Prime core/autonomous.ts.

The gate callable owns execution and diagnostic disclosure. This controller neither
loads a grader nor triggers refinement. A None snapshot disables unchanged-worktree
suppression, as in Prime (including repositories without a HEAD commit).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .harness import js_slice

DEFAULT_CONTINUATION = (
    "No human input is available in autonomous mode. Continue working until the host "
    "evaluator, verifier, or configured autonomous limits stop the run. If you were asking "
    "the user a question, make a reasonable assumption and verify it. If you believe you "
    "are blocked, prove it with host-observable evidence, preserve that evidence, and keep "
    "looking for safe progress while budget remains. Do not end the session yourself; "
    "the verifier/evaluator decides completion when configured gates pass."
)
UNCHANGED = (
    "The autonomous gate was not rerun because the workspace has not changed since this "
    "failure. Edit source files, tests, or a blocker artifact before attempting to finish again."
)
EXCLUDED = (
    "verification",
    "target",
    ".vf-prime-agent",
    "Cargo.lock",
    "submission.tar.gz",
    "runner_args.log",
)


@dataclass(frozen=True)
class AutonomousPolicy:
    enabled: bool = True
    max_continuations: int = 3
    max_turns: int = 12
    max_tokens: int = 80_000
    timeout_seconds: float = 1800
    max_retries: int = 3
    gate_timeout_seconds: float = 300

    def __post_init__(self):
        import math

        for key, value in asdict(self).items():
            if key != "enabled" and (
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")


@dataclass
class GateResult:
    passed: bool
    exit_text: str
    output: str = ""


def bounded_output(output: str) -> str:
    prefix = js_slice(output, 0, 6000)
    return prefix + ("\n... [truncated]" if prefix != output else "")


def failure_message(command, attempt, max_retries, result, *, timestamp=None):
    stamp = (
        datetime.fromtimestamp(timestamp or time.time(), UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    output = bounded_output(result.output.strip())
    return (
        f"Autonomous quality gate failed (attempt {attempt}/{max_retries}): "
        f"`{command}` {result.exit_text}.\n"
        + (f"\nOutput:\n{output}\n" if output else "\n")
        + f"\nContinue working. Fix the failure, then produce terminal evidence. Timestamp: {stamp}."
    )


async def git_snapshot(workspace: Path) -> dict | None:
    pathspec = ["--", ".", *(f":(exclude){p}" for p in EXCLUDED)]

    async def git(*args):
        proc = await asyncio.create_subprocess_exec(
            "git",
            "--no-optional-locks",
            *args,
            *pathspec,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(10):
                # Refuse truncated evidence instead of equating incomplete snapshots.
                data = bytearray()
                while chunk := await proc.stdout.read(65536):
                    data.extend(chunk)
                    if len(data) > 1024 * 1024:
                        return None
                await proc.wait()
                return bytes(data) if proc.returncode == 0 else None
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    try:
        status = await git("status", "--porcelain=v1", "-z", "-uall", "--no-renames")
        diff = await git("diff", "--no-ext-diff", "--binary", "HEAD")
        if status is None or diff is None:
            return None
        aggregate = hashlib.sha256()
        for raw in sorted(row[3:] for row in status.split(b"\0") if row.startswith(b"?? ")):
            path = workspace / os.fsdecode(raw)
            try:
                stat = path.lstat()
                if path.is_symlink():
                    value = "symlink:" + os.readlink(path)
                elif path.is_file():

                    def hash_file(path=path):
                        with path.open("rb") as stream:
                            return "file:" + hashlib.file_digest(stream, "sha256").hexdigest()

                    value = await asyncio.to_thread(hash_file)
                else:
                    value = f"other:{stat.st_mode}:{stat.st_size}:{stat.st_mtime * 1000}"
            except OSError as exc:
                value = f"error:{exc}"
            aggregate.update(raw + b"\0" + value.encode() + b"\0")
        return {
            "status": hashlib.sha256(status).hexdigest(),
            "diff": hashlib.sha256(diff).hexdigest(),
            "untrackedHash": aggregate.hexdigest(),
        }
    except (OSError, TimeoutError):
        return None


@dataclass
class AutonomousCycle:
    policy: AutonomousPolicy
    commands: list[str]
    started: float = field(default_factory=time.monotonic)
    continuations: int = 0
    turns: int = 0
    tokens: int = 0
    attempts: dict = field(default_factory=dict)
    last_failure: dict | None = None
    last_snapshot: dict | None = None
    stop_reason: str | None = None
    checks: list = field(default_factory=list)

    def record_response(self, response):
        if not self.policy.enabled or response.metadata.get("stop_reason") == "error":
            return
        self.turns += 1
        # Buffalo input includes cache reads. Prime input does not; cache writes
        # are already included in Buffalo's input counter.
        usage = response.usage
        self.tokens += max(0, usage.input_tokens - usage.cached_input_tokens) + usage.output_tokens

    def limit_reason(self, *, now=None):
        now = time.monotonic() if now is None else now
        for exhausted, reason in (
            (self.continuations >= self.policy.max_continuations, "maxContinuations"),
            (self.turns >= self.policy.max_turns, "maxTurns"),
            (self.tokens >= self.policy.max_tokens, "maxTokens"),
            (now - self.started >= self.policy.timeout_seconds, "timeoutMs"),
        ):
            if exhausted:
                return reason
        return None

    async def next_message(
        self,
        stop_reason: str | None,
        *,
        snapshot: Callable[[], Awaitable[dict | None]],
        run_gate: Callable[[str, float], Awaitable[GateResult]],
    ) -> str | None:
        # Prime captures `now` on entering nextAutonomousContinuation, before
        # running the quality gate. A slow gate must not retroactively withdraw
        # a continuation that was within its wall budget at this boundary.
        boundary_time = time.monotonic()
        if not self.policy.enabled or stop_reason in {"error", "aborted"}:
            self.stop_reason = stop_reason or "disabled"
            return None
        for command in self.commands:
            current = await snapshot()
            skipped = bool(
                self.last_failure
                and self.last_failure["command"] == command
                and current is not None
                and current == self.last_snapshot
            )
            if skipped:
                result = GateResult(
                    False, "not rerun: workspace unchanged since previous failed gate", UNCHANGED
                )
            else:
                # Infrastructure/provider exceptions propagate; they are never code failures.
                result = await run_gate(command, self.policy.gate_timeout_seconds)
                current = await snapshot()
            self.checks.append({"command": command, "rerun": not skipped, **asdict(result)})
            if result.passed:
                self.attempts[command] = 0
                if self.last_failure and self.last_failure["command"] == command:
                    self.last_failure = self.last_snapshot = None
                continue
            attempt = self.attempts.get(command, 0) + 1
            self.attempts[command] = attempt
            self.last_failure = {"command": command, "attempt": attempt, **asdict(result)}
            self.last_snapshot = current
            self.stop_reason = (
                "retry_exhausted"
                if attempt > self.policy.max_retries
                else self.limit_reason(now=boundary_time)
            )
            if self.stop_reason:
                return None
            self.continuations += 1
            return failure_message(command, attempt, self.policy.max_retries, result)
        if self.commands:
            self.last_failure = self.last_snapshot = None
            self.stop_reason = "passed"
            return None
        self.stop_reason = self.limit_reason(now=boundary_time)
        if self.stop_reason:
            return None
        self.continuations += 1
        return DEFAULT_CONTINUATION
