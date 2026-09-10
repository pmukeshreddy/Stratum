"""Model-controlled asynchronous execution and durable coordination.

Only this manager schedules sessions. Clients submit commands through the daemon;
disconnecting never cancels an execution task.
"""

from __future__ import annotations

import asyncio
import fcntl
import importlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from croniter import croniter

from .artifacts import Artifacts
from .context import Context
from .environment import Environment
from .execution import environment
from .kernel import Kernel, fork_checkpoint
from .models import (
    Action,
    HarnessError,
    Lifecycle,
    ModelRequest,
    ModelResponse,
    Outcome,
    RunConfig,
    Usage,
    Workspace,
    new_id,
    now,
)
from .providers import default_providers
from .refinement import RefinementServices, refine_command
from .routing import route
from .skills import discover
from .storage import Store, encode
from .tools import ToolContext, builtins

log = logging.getLogger("threadweave.runtime")


class LimitReached(Exception):
    pass


class TurnAdmissionLimit(LimitReached):
    """No new turn may start; already-admitted turns may still return evidence."""


class GoalLimitReached(LimitReached):
    def __init__(self, sid):
        self.session_id = sid
        super().__init__("Persistent goal token budget exhausted")


class BudgetBusy(Exception):
    """Another invocation temporarily owns a reservation that may be released."""


class Runtime(RefinementServices):
    def __init__(
        self,
        directory: str | Path,
        *,
        providers=None,
        adapters=None,
        tools=None,
        concurrency=8,
        idle_seconds=60,
    ):
        if concurrency < 1 or idle_seconds < 0:
            raise ValueError("Concurrency must be positive and idle_seconds nonnegative")
        self.store = Store(directory)
        self._owner_lock = (self.store.directory / "runtime.lock").open("a+")
        try:
            fcntl.flock(self._owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._owner_lock.close()
            self.store.close()
            raise RuntimeError("A runtime already owns this data directory") from exc
        self.artifacts = Artifacts(self.store)
        self.context = Context(self.store, environment=lambda: self.environment)
        self.providers = providers if providers is not None else default_providers()
        self.tools = tools if tools is not None else builtins()
        self.environment = Environment(self, adapters)
        self.adapters = self.environment.adapters  # Extension registration remains compatible.
        self.concurrency, self.idle_seconds = concurrency, idle_seconds
        self.kernels: dict[str, Kernel] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._admitted_turns: set[str] = set()
        self._scheduler_task = None
        self._wake = asyncio.Event()
        self._closing = False
        self._extensions: set[str] = set()
        self._python_parent: dict[str, str] = {}
        self._transitioning = set()
        self._refinement_states = {}
        self.context.on_compact = self.refinement_compacted
        from .verification import VerificationScheduler

        self.verification = VerificationScheduler(self)

    def load_extensions(self, config):
        for reference in config.extensions:
            if reference not in self._extensions:
                module, function = reference.split(":", 1)
                getattr(importlib.import_module(module), function)(self)
                self._extensions.add(reference)

    def validate_config(self, config):
        RunConfig.model_validate(config.model_dump())
        self.load_extensions(config)
        self.environment.configure(config, defaults=False)
        providers = list(config.models.values()) + (
            [config.provider] if not config.routing.default else []
        )
        for provider in providers:
            if provider.name not in self.providers:
                raise ValueError(f"Unknown provider: {provider.name}; configure a real provider")
            if (
                not provider.model.strip() and provider.name != "codex_subscription"
            ) or provider.model.startswith("REPLACE_"):
                raise ValueError(
                    "Explicit provider.model is required; pass --config with a real model ID"
                )
            if (
                provider.name == "chat"
                and provider.api_key_env
                and not os.environ.get(provider.api_key_env)
            ):
                raise ValueError(
                    f"Set {provider.api_key_env} before starting the daemon (or use api_key_env='' for an unauthenticated local endpoint)"
                )
            if config.limits.cost_budget and (
                provider.input_cost_per_million is None or provider.output_cost_per_million is None
            ):
                raise ValueError(
                    "Every routed model needs input/output prices when using cost budgets"
                )
        if config.task.adapter not in self.adapters:
            raise ValueError(f"Unknown task adapter: {config.task.adapter}")

    def create(
        self,
        instruction: str,
        workspace: str | Path,
        *,
        config=None,
        name="root",
        mode="autonomous",
    ):
        config = config or RunConfig()
        config = config.model_copy(deep=True)
        self.load_extensions(config)
        self.environment.configure(config)
        self.validate_config(config)
        path = Path(workspace).resolve()
        if not path.is_dir():
            raise ValueError(f"Workspace must already exist: {path}")
        if not instruction.strip():
            raise ValueError("A task instruction is required")
        with self.store.transaction():
            session = self.store.create(
                instruction, Workspace(path=str(path)), config, name=name, mode=mode
            )
            self.load_adapter_context(session.id)
            session = self.store.session(session.id)
            self.store.event(
                session.id,
                "task_admitted",
                {
                    "instruction": instruction,
                    "config_id": session.config_id,
                    "workspace": str(path),
                },
            )
            if mode == "interactive":
                session = self.store.transition(session.id, Lifecycle.IDLE, runnable=False)
        self._wake.set()
        return session

    def spawn(
        self,
        parent_id: str,
        instruction: str,
        *,
        name=None,
        role="agent",
        isolate=None,
        provider=None,
        model=None,
        thinking=None,
        adapter=None,
        purpose=None,
        _defer_workspace=False,
    ):
        parent = self.store.session(parent_id)
        if parent.outcome != Outcome.ACTIVE:
            raise ValueError("Cannot spawn from a terminated session")
        self._check_limits(parent_id)
        root_config = self.store.config(parent.root_id)
        if not root_config.features.subagents:
            raise PermissionError("Subagents disabled by feature configuration")
        usage = self.store.usage(parent.root_id, tree=True)
        if usage.subagent_count >= root_config.limits.max_subagents:
            raise HarnessError("tool", "subagent_limit", "Root subagent limit reached")
        depth, ancestor = 1, parent
        while ancestor.parent_id:
            depth += 1
            ancestor = self.store.session(ancestor.parent_id)
        if depth > root_config.limits.max_depth:
            raise HarnessError("tool", "depth_limit", "Recursive depth limit reached")
        config = self.store.config(parent_id).model_copy(deep=True)
        # Snapshot active registry capabilities separately from the permission
        # allowlist; tools loaded later by another session must not leak in.
        config.active_tool_names = [
            name for name in self.tools.entries if self.tools.allowed(name, config)
        ]
        if adapter is not None and adapter != config.task.adapter:
            config.task.adapter = adapter
            config.task.verifier = "none"
            config.task.require_verifier = False
            config.active_tool_names = None
            self.environment.configure(config)
        profile, isolate = self.environment.child_policy(config, purpose, isolate)
        if profile.read_only:
            config.execution.read_only = True
            from .isolation import readonly_worker

            readonly_worker(self.store.directory)
            instruction += "\nRead-only investigation. Return findings, evidence and remaining uncertainty; do not make changes."
        if profile.instruction:
            instruction += "\n" + profile.instruction
        if profile.configure:
            profile.configure(config)
        # Snapshot the parent's effective agent route, including a role-based model.
        effective = route(self.store, parent_id, parent.role).model_copy(deep=True)
        if provider is not None:
            effective = provider.model_copy(deep=True)
        if model is not None:
            choices = {
                v.name + "/" + v.model: v for v in [config.provider, *config.models.values()]
            }
            if model in config.models:
                effective = config.models[model].model_copy(deep=True)
            elif model in choices:
                effective = choices[model].model_copy(deep=True)
            else:
                effective.model = model
        if thinking is not None:
            effective.parameters["reasoning_effort"] = thinking
        config.provider = effective
        config.routing.default = None
        config.routing.roles.pop(role, None)
        self.validate_config(config)
        if _defer_workspace:
            workspace, checkpoint = parent.workspace.model_copy(deep=True), None
            workspace.metadata["admission_pending"] = True
        else:
            workspace, checkpoint = self.environment.continuation_workspace(
                parent, config, child=True, isolate=isolate
            )
        evidence = self.store.events(parent_id, kind="verifier_result", limit=1)
        package = {
            "parent_objective": parent.instruction[:1500],
            "assignment": instruction,
            "verifier_event": evidence[0]["id"] if evidence else None,
            "verification_evidence": [
                e["payload"]
                for e in self.store.events(parent_id, kind="verification_evidence", limit=1)
            ],
            "result_contract": "Return findings, supporting evidence and remaining uncertainty.",
            **getattr(
                self.environment.adapters[config.task.adapter], "delegation", lambda *args: {}
            )(parent, config),
        }
        if config.control_plane == "python":
            instruction += "\nDelegation context: " + encode(package)
        session = self.store.create(
            instruction,
            workspace,
            config,
            parent_id=parent_id,
            spawned_by_request_id=self.store.last_request(parent_id, purpose="agent"),
            depth=depth,
            name=name or f"child-{usage.subagent_count + 1}",
            mode="autonomous" if config.control_plane == "python" else "goal",
        )
        if not isolate:
            self.environment.call(session.id, "inherit_shared_admission", parent, session)
        self.load_adapter_context(session.id)
        self.store.event(
            session.id,
            "task_admitted",
            {"instruction": instruction, "parent_id": parent_id, "depth": depth},
        )
        self.store.event(
            session.id,
            "child_purpose",
            {
                "purpose": purpose or "shared",
                "workspace_mode": "isolated" if isolate else "shared",
                "base": workspace.metadata.get("base_revision"),
                "parent": parent_id,
                "trusted_python_mutation_risk": not isolate and not config.execution.read_only,
                "kernel_write_policy": "os_read_only" if config.execution.read_only else "writable",
            },
        )
        self.environment.call(session.id, "child_admitted", parent, session, checkpoint)
        self.store.update(session.id, role=role)
        if _defer_workspace:
            self.store.update(session.id, paused=True, runnable=False)
        self.store.event(session.id, "child_context_package", package)
        self._wake.set()
        return self.store.session(session.id)

    async def spawn_async(self, parent_id, instruction, **options):
        """Reserve a child on the daemon thread; prepare Git state off its event loop.

        Each worker has a separate WAL connection. No Store or asyncio object is
        shared between threads. Cancellation waits for the owned preparation to
        settle, preventing an untracked checkout from appearing after shutdown.
        """
        parent_config = self.store.config(parent_id).model_copy(deep=True)
        if options.get("adapter"):
            parent_config.task.adapter = options["adapter"]
        _, isolate = self.environment.child_policy(
            parent_config, options.get("purpose"), options.get("isolate")
        )
        if not isolate:
            return self.spawn(parent_id, instruction, **options)
        child = self.spawn(parent_id, instruction, _defer_workspace=True, **options)
        job = asyncio.create_task(
            asyncio.to_thread(self.environment.isolated_child_workspace, parent_id, child.id)
        )
        try:
            workspace, checkpoint = await asyncio.shield(job)
            self._check_limits(parent_id)
            if self.store.session(child.id).outcome != Outcome.ACTIVE:
                raise ValueError("Child was stopped during admission")
            config = self.store.config(child.id).model_copy(deep=True)
            self.environment.call(child.id, "workspace_admitted", config, workspace, child=True)
            self.store.reconfigure(child.id, config)
            self.store.update(child.id, workspace=workspace, paused=False, runnable=True)
            self.environment.call(
                child.id,
                "child_admitted",
                self.store.session(parent_id),
                self.store.session(child.id),
                checkpoint,
            )
            self.store.event(
                child.id,
                "workspace_ready",
                {"workspace": workspace.model_dump(), "checkpoint": checkpoint},
            )
            self._wake.set()
            return self.store.session(child.id)
        except BaseException:
            # The failed reservation remains auditable; usage is never refunded.
            result = await asyncio.gather(job, return_exceptions=True)
            if self.store.session(child.id).outcome == Outcome.ACTIVE:
                self.store.update(child.id, outcome=Outcome.FAILED, paused=True, runnable=False)
            self.store.event(
                child.id,
                "workspace_admission_failed",
                {
                    "workspace_retained": str(result[0][0].path)
                    if result and isinstance(result[0], tuple)
                    else None
                },
            )
            raise

    def related(self, sid: str):
        session = self.store.session(sid)
        results = []
        for candidate in self.store.sessions(root_id=session.root_id):
            if candidate.id == sid:
                continue
            try:
                self.store.ensure_related(sid, candidate.id)
            except HarnessError:
                continue
            results.append(
                {
                    "id": candidate.id,
                    "name": candidate.name,
                    "parent_id": candidate.parent_id,
                    "lifecycle": candidate.lifecycle,
                    "outcome": candidate.outcome,
                    "turns": candidate.turns,
                    "result": (candidate.result or "")[:1000],
                }
            )
        return results

    def inspect(self, sid: str):
        session = self.store.session(sid)
        config = self.store.config(sid)
        return {
            "session": session.model_dump(mode="json"),
            "goal": self.store.goal(sid),
            "usage": self.store.usage(sid).model_dump(),
            "tree_usage": self.store.usage(sid, tree=True).model_dump(),
            "root_elapsed_seconds": self._elapsed(session.root_id),
            "config": config.model_dump(mode="json"),
            "messages": self.store.messages(sid, limit=10),
            "information": self.information(sid),
        }

    def information(self, sid: str):
        """Bounded layer metadata. Never implicitly serialize L2 variables or L3 contents."""
        session = self.store.session(sid)
        checkpoint = self.store.directory / "kernels" / session.kernel_id / "checkpoint.json"
        names, missing = [], []
        if checkpoint.is_file():
            data = json.loads(checkpoint.read_text())
            names = sorted(set(data.get("values", {})) | set(data.get("recipes", {})))
            missing = sorted(data.get("missing", {}))

        def count(table, clause=""):
            return self.store.db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?{clause}", (sid,)
            ).fetchone()[0]

        children = [s for s in self.store.sessions(root_id=session.root_id) if s.parent_id == sid]
        return {
            "L1": {
                "blocks": len(session.context),
                "summary_chars": len(session.summary),
                "context_limit": self.store.config(sid).context.max_tokens,
            },
            "L2": {
                "kernel_id": session.kernel_id,
                "worker_live": sid in self.kernels and self.kernels[sid].process is not None,
                "checkpointed_variables": len(names),
                "checkpointed_names": names[:100],
                "unrecoverable_names": missing[:100],
                "children": [
                    {"id": s.id, "name": s.name, "lifecycle": s.lifecycle, "outcome": s.outcome}
                    for s in children
                ],
                "values_included": False,
            },
            "L3": {
                "events": count("events"),
                "artifacts": count("artifacts"),
                "compactions": count("compactions"),
                "harness_entries": len(self.store.harness.entries(sid)),
                "pending_messages": self.store.db.execute(
                    "SELECT COUNT(*) FROM messages WHERE recipient_id=? AND received_at IS NULL",
                    (sid,),
                ).fetchone()[0],
                "schedules": count("schedules", " AND enabled=1"),
                "goal": self.store.goal(sid),
                "history_append_only": True,
            },
        }

    def message(
        self, sender_id, recipient_id, body, *, delivery="boundary", causal_request_id=None
    ):
        if sender_id:
            self.store.ensure_related(sender_id, recipient_id)
        with self.store.transaction():
            recipient = self.store.session(recipient_id)
            if (
                delivery == "idle"
                and recipient.wake_at is None
                and (
                    recipient.outcome == Outcome.COMPLETED
                    or not recipient.runnable
                    and not recipient.pending_turn
                    and recipient_id not in self.tasks
                )
            ):
                # The requested idle boundary has already been reached. Mark
                # ready before admission so the first new request sees this input.
                delivery = "boundary"
            mid = self.store.send(
                sender_id,
                recipient_id,
                body,
                delivery=delivery,
                causal_request_id=causal_request_id,
            )
            if sender_id is None and (command := refine_command(body)):
                self.queue_refine_command(recipient_id, command)
                # Control input is consumed by the refinement queue, not by an
                # ordinary agent turn (and is not evidence of reusable learning).
                self.store.db.execute("UPDATE messages SET received_at=? WHERE id=?", (now(), mid))
                return mid
            self._continue_completed_child(recipient_id)
            recipient = self.store.session(recipient_id)
            if recipient.outcome == Outcome.ACTIVE and not recipient.paused:
                self.store.update(recipient_id, runnable=True, wake_at=None)
        self._wake.set()
        return mid

    def _continue_completed_child(self, sid):
        """Admit queued explicit parent/human follow-ups without resetting a trajectory.

        Called inside the send/completion transaction and during recovery. Replies
        from descendants are observations, not new assignments to a finished parent.
        The scheduler's per-session task slot owns execution, including kernel cleanup.
        """
        session = self.store.session(sid)
        if not session.parent_id or session.outcome != Outcome.COMPLETED or session.paused:
            return False
        message = self.store.db.execute(
            "SELECT id FROM messages WHERE recipient_id=? AND received_at IS NULL "
            "AND (sender_id=? OR sender_id IS NULL) ORDER BY created_at LIMIT 1",
            (sid, session.parent_id),
        ).fetchone()
        if not message:
            return False
        ancestor = self.store.session(session.parent_id)
        while ancestor:
            if ancestor.outcome in {Outcome.CANCELLED, Outcome.FAILED, Outcome.LIMITED}:
                return False
            ancestor = self.store.session(ancestor.parent_id) if ancestor.parent_id else None
        try:
            self._check_limits(sid, resource="turns")
            self._check_limits(sid, resource="model_calls")
        except LimitReached:
            return False
        except BudgetBusy:
            # Reservations are temporary; normal invocation admission will wait.
            pass
        self.store.db.execute(
            "UPDATE messages SET delivery='boundary' WHERE recipient_id=? AND received_at IS NULL AND delivery='idle'",
            (sid,),
        )
        self.store.update(sid, outcome=Outcome.ACTIVE, runnable=True, wake_at=None, result=None)
        self.store.db.execute(
            "UPDATE goals SET status='active',updated_at=? WHERE session_id=?", (now(), sid)
        )
        self.store.event(
            sid,
            "subagent_continued",
            {
                "message_id": message["id"],
                "previous_outcome": "completed",
                "kernel_id": session.kernel_id,
            },
        )
        self._wake.set()
        return True

    def receive(self, sid, *, include_followups=True):
        cap = self.store.config(sid).context.result_chars

        def render(message):
            return encode(
                {
                    "message_id": message["id"],
                    "sender_id": message["sender_id"],
                    "body": message["body"][:cap],
                    "source_event": message["source_event"],
                    "truncated": len(message["body"]) > cap,
                }
            )

        return self.store.receive(sid, render, limit=20, include_followups=include_followups)

    def interact(self, sid, body):
        """Atomic human input + continuation; no client-side input/resume race."""
        if not body.strip():
            raise ValueError("Message cannot be empty")
        if refine_command(body):
            return self.message(None, sid, body)
        session = self.store.session(sid)
        if session.outcome == Outcome.LIMITED:
            raise ValueError("Session resource limit reached; use /new or fork a new run")
        with self.store.transaction():
            mid = self.message(None, sid, body)
            if session.paused or session.outcome != Outcome.ACTIVE:
                self.resume(sid)
        return mid

    def defer(self, sid, seconds):
        session = self.store.session(sid)
        if self.store.messages(sid, pending=True, limit=1) and not session.paused:
            self.store.update(sid, runnable=True, wake_at=None)
        else:
            self.store.update(sid, runnable=False, wake_at=now() + seconds)

    def resume(self, sid: str):
        self.context._harness_context_ready.discard(sid)
        session = self.store.session(sid)
        self.validate_config(self.store.config(sid))
        if self._stopped_ancestor(sid):
            raise ValueError("Resume the stopped parent before resuming its child")
        if session.outcome == Outcome.LIMITED:
            raise ValueError(
                "A resource-limited trajectory cannot reset its budget; fork a new run"
            )
        with self.store.transaction():
            self.store.update(
                sid,
                outcome=Outcome.ACTIVE,
                runnable=True,
                paused=False,
                wake_at=None,
                last_error=None,
            )
            self.store.db.execute(
                "UPDATE goals SET status='active',updated_at=? WHERE session_id=?", (now(), sid)
            )
            self.store.event(sid, "resumed", {})
        self._wake.set()

    def _stopped_ancestor(self, sid):
        parent = self.store.session(sid).parent_id
        while parent:
            session = self.store.session(parent)
            if session.outcome in {Outcome.CANCELLED, Outcome.FAILED, Outcome.LIMITED}:
                return parent
            parent = session.parent_id
        return None

    def schedule(
        self, sid: str, *, interval_seconds=None, cron=None, instruction="Scheduled continuation"
    ):
        if (interval_seconds is None) == (cron is None):
            raise ValueError("Specify exactly one of interval_seconds or cron")
        if interval_seconds is not None and interval_seconds < 0.1:
            raise ValueError("Schedule interval must be at least 0.1 seconds")
        if cron is not None and (len(cron.split()) != 5 or not croniter.is_valid(cron)):
            raise ValueError("Expected a valid five-field UTC cron expression")
        session = self.store.session(sid)
        if session.outcome != Outcome.ACTIVE:
            raise ValueError("Resume the session before scheduling it")
        next_at = self._next_schedule(interval_seconds, cron, now())
        schedule_id = new_id()
        with self.store.transaction():
            self.store.db.execute(
                "INSERT INTO schedules VALUES(?,?,?,?,?,?,?)",
                (schedule_id, sid, interval_seconds, cron, next_at, 1, instruction),
            )
            self.store.event(
                sid,
                "schedule_created",
                {
                    "schedule_id": schedule_id,
                    "interval_seconds": interval_seconds,
                    "cron": cron,
                    "next_at": next_at,
                },
            )
        self._wake.set()
        return schedule_id

    @staticmethod
    def _next_schedule(interval, cron, base):
        return (
            base + interval
            if interval
            else croniter(cron, datetime.fromtimestamp(base, UTC)).get_next(float)
        )

    def _schedules_due(self):
        timestamp = now()
        rows = self.store.db.execute(
            "SELECT * FROM schedules WHERE enabled=1 AND next_at<=?", (timestamp,)
        ).fetchall()
        for row in rows:
            with self.store.transaction():
                if self.store.session(row["session_id"]).outcome != Outcome.ACTIVE:
                    continue
                self.message(None, row["session_id"], row["instruction"])
                # Coalesce missed ticks: never unleash a backlog of overdue model calls.
                next_at = self._next_schedule(row["interval_seconds"], row["cron"], timestamp)
                self.store.db.execute(
                    "UPDATE schedules SET next_at=? WHERE id=?", (next_at, row["id"])
                )
                self.store.event(
                    row["session_id"],
                    "heartbeat",
                    {"schedule_id": row["id"], "scheduled_at": row["next_at"], "next_at": next_at},
                )
        for session in self.store.sessions():
            if (
                session.outcome == Outcome.ACTIVE
                and session.wake_at
                and session.wake_at <= timestamp
            ):
                self.store.update(session.id, runnable=True, wake_at=None)

    def _elapsed(self, root_id):
        root = self.store.session(root_id)
        if root.mode == "interactive":
            # Human think time / detached idle time is not execution. Count aggregate
            # descendant execution seconds conservatively, without resetting budgets.
            sessions = self.store.sessions(root_id=root_id)
            return self.store.usage(root_id, tree=True).wall_seconds + sum(
                max(0, now() - s.running_since) for s in sessions if s.running_since
            )
        return max(0, now() - root.started_at) if root.started_at else 0

    def _check_limits(self, sid: str, *, resource: str | None = None, input_bound=0, provider=None):
        session = self.store.session(sid)
        root = self.store.session(session.root_id)
        config, usage = self.store.config(root.id), self.store.usage(root.id, tree=True)
        limits = config.limits
        if root.outcome == Outcome.LIMITED:
            raise LimitReached("Root resource limit already reached")
        ancestor = session
        while ancestor:
            goal = self.store.goal(ancestor.id)
            # Completion does not erase a trajectory budget: completed children
            # may accept follow-ups, and completed ancestors still own their spend.
            if goal and goal["status"] in {"active", "completed"} and goal.get("token_budget"):
                bound = (
                    input_bound + (provider or self.store.config(sid).provider).max_output_tokens
                    if resource == "model_calls"
                    else 0
                )
                if goal["tokens_used"] + goal["tokens_reserved"] + bound >= goal["token_budget"]:
                    if (
                        goal["tokens_reserved"]
                        and goal["tokens_used"] + bound < goal["token_budget"]
                    ):
                        raise BudgetBusy()
                    raise GoalLimitReached(ancestor.id)
            ancestor = self.store.session(ancestor.parent_id) if ancestor.parent_id else None
        if self._elapsed(root.id) >= limits.wall_seconds:
            raise LimitReached("Root wall-clock budget exhausted")
        reserved_tokens, reserved_cost = self.store.reserved(root.id)
        tokens = usage.input_tokens + usage.output_tokens + reserved_tokens
        if usage.input_tokens + usage.output_tokens >= limits.token_budget:
            raise LimitReached("Root token budget exhausted")
        if limits.output_token_budget is not None:
            if usage.output_tokens >= limits.output_token_budget:
                raise LimitReached("Root output token budget exhausted")
            if resource == "model_calls":
                reserved_output = self.store.db.execute(
                    "SELECT COALESCE(SUM(r.output_tokens),0) FROM reservations r "
                    "JOIN sessions s ON s.id=r.session_id WHERE s.root_id=?",
                    (root.id,),
                ).fetchone()[0]
                output = (provider or self.store.config(sid).provider).max_output_tokens
                if usage.output_tokens + reserved_output + output > limits.output_token_budget:
                    if (
                        reserved_output
                        and usage.output_tokens + output <= limits.output_token_budget
                    ):
                        raise BudgetBusy()
                    raise LimitReached(
                        "Insufficient root output tokens to reserve the next invocation"
                    )
        if limits.cost_budget is not None and usage.cost >= limits.cost_budget:
            raise LimitReached("Root cost budget exhausted")
        if resource == "turns" and usage.turns >= limits.max_turns:
            raise TurnAdmissionLimit("Root turn limit exhausted")
        for key, maximum in (
            ("tool_calls", limits.max_tool_calls),
            ("python_executions", limits.max_python_executions),
            ("model_calls", limits.max_model_calls),
        ):
            if resource == key and getattr(usage, key) >= maximum:
                raise LimitReached(f"Root {key} limit exhausted")
        if resource == "model_calls":
            provider = provider or self.store.config(sid).provider
            if tokens + input_bound + provider.max_output_tokens > limits.token_budget:
                if (
                    reserved_tokens
                    and tokens - reserved_tokens + input_bound + provider.max_output_tokens
                    <= limits.token_budget
                ):
                    raise BudgetBusy()
                raise LimitReached("Insufficient root token budget to reserve the next invocation")
            cost = (
                input_bound * (provider.input_cost_per_million or 0)
                + provider.max_output_tokens * (provider.output_cost_per_million or 0)
            ) / 1_000_000
            if (
                limits.cost_budget is not None
                and usage.cost + reserved_cost + cost > limits.cost_budget
            ):
                if reserved_cost and usage.cost + cost <= limits.cost_budget:
                    raise BudgetBusy()
                raise LimitReached("Insufficient root cost budget to reserve the next invocation")

    async def start(self):
        if self._scheduler_task:
            return
        self._closing = False
        await self.recover()
        self._scheduler_task = asyncio.create_task(self._scheduler(), name="session-scheduler")

    async def recover(self):
        for session in self.store.sessions():
            config = self.store.config(session.id)
            try:
                self.validate_config(config)
                if config != self.store.config(session.id):
                    self.store.reconfigure(session.id, config)
            except Exception as exc:
                self.store.update(session.id, runnable=False)
                self.store.event(session.id, "configuration_recovery_failed", {"reason": str(exc)})
        await self.environment.recover()
        if (
            not hasattr(self, "background")
            and self.store.db.execute("SELECT 1 FROM process_jobs LIMIT 1").fetchone()
        ):
            from .background import BackgroundProcesses

            self.background = BackgroundProcesses(self)
        if hasattr(self, "background"):
            self.background.recover()
        self.store.db.execute(
            "UPDATE model_attempts SET status='interrupted',ended_at=? WHERE status='running'",
            (now(),),
        )
        self.store.db.execute(
            "UPDATE model_requests SET status='interrupted',ended_at=? WHERE status IN ('running','retrying')",
            (now(),),
        )
        # Reservations from an interrupted model request have unknown billing. Conservatively
        # charge the full reservation instead of silently resetting spend after a crash.
        for row in self.store.db.execute("SELECT * FROM reservations").fetchall():
            with self.store.transaction():
                self.store.charge(
                    row["session_id"],
                    Usage(
                        input_tokens=row["input_tokens"],
                        output_tokens=row["output_tokens"],
                        cost=row["cost"],
                        estimated_calls=1,
                    ),
                    parent=row["id"],
                )
                self.store.db.execute(
                    "UPDATE model_attempts SET usage=?,failure=? WHERE event_id=?",
                    (
                        encode(
                            Usage(
                                input_tokens=row["input_tokens"],
                                output_tokens=row["output_tokens"],
                                cost=row["cost"],
                                estimated_calls=1,
                                model_calls=1,
                            ).model_dump()
                        ),
                        encode({"code": "runtime_restart", "uncertain": True}),
                        row["id"],
                    ),
                )
                self.store.db.execute("DELETE FROM reservations WHERE id=?", (row["id"],))
                self.store.event(
                    row["session_id"],
                    "recovery",
                    {
                        "interrupted_invocation": row["id"],
                        "accounting": "charged reservation; provider usage unknown",
                    },
                )
        for session in self.store.sessions():
            if session.running_since is not None:
                elapsed = max(0, now() - session.running_since)
                with self.store.transaction():
                    self.store.charge(session.id, Usage(wall_seconds=elapsed))
                    self.store.update(session.id, running_since=None)
                    self.store.event(
                        session.id,
                        "recovery",
                        {
                            "execution_seconds_estimate": elapsed,
                            "note": "Interrupted turn duration conservatively includes downtime",
                        },
                    )
            if session.lifecycle != Lifecycle.INACTIVE:
                self.store.transition(session.id, Lifecycle.INACTIVE)
            try:
                config = self.store.config(session.id)
                self.validate_config(config)
                if config != self.store.config(session.id):
                    self.store.reconfigure(session.id, config)
            except Exception as exc:
                failure = HarnessError("environment", "extension_unavailable", str(exc)).failure
                self.store.update(session.id, last_error=failure, runnable=False)
                self.store.event(session.id, "failure", failure.model_dump())
                continue
            # A completed worker checkpoint can close the action receipt crash window.
            rows = self.store.db.execute(
                "SELECT * FROM actions WHERE session_id=? AND status='running'", (session.id,)
            ).fetchall()
            for row in rows:
                raw = (
                    self._kernel(session.id).receipt(row["id"])
                    if row["name"] in ("python", "ipython")
                    else None
                )
                if raw is not None:
                    raw = self._capture_kernel_logs(session.id, raw, row["source_event"])
                    self._action_result(
                        session.id, row["id"], raw, row["source_event"], recovered=True
                    )
                else:
                    failure = HarnessError(
                        "runtime",
                        "interrupted_action",
                        "Action interrupted; side effects are uncertain. Inspect before repeating.",
                        uncertain=True,
                    )
                    self._action_result(
                        session.id,
                        row["id"],
                        {"error": failure.failure.model_dump()},
                        row["source_event"],
                        recovered=True,
                    )
            with self.store.transaction():
                self._continue_completed_child(session.id)
            session = self.store.session(session.id)
            if session.outcome == Outcome.ACTIVE and (
                ancestor := self._stopped_ancestor(session.id)
            ):
                await self.stop(session.id)
                self.store.event(
                    session.id,
                    "orphan_child_cancelled",
                    {
                        "stopped_ancestor": ancestor,
                        "reason": "Recovered interrupted parent termination",
                    },
                )
                continue
            if session.outcome == Outcome.ACTIVE:
                eid = self.store.event(
                    session.id,
                    "recovery",
                    {
                        "kernel_id": session.kernel_id,
                        "pending_turn": session.pending_turn is not None,
                        "interrupted_actions": [r["id"] for r in rows],
                    },
                )
                if rows:
                    self.store.add_context(
                        session.id,
                        eid,
                        [
                            {
                                "role": "user",
                                "content": "Runtime recovered interrupted actions. Inspect their results and uncertainty before retrying.",
                            }
                        ],
                    )

    async def _scheduler(self):
        while not self._closing:
            self._wake.clear()
            try:
                self._schedules_due()
                self.environment.poll()
                for root in self.store.sessions(roots_only=True):
                    if root.started_at and (
                        root.outcome == Outcome.ACTIVE or self._active_descendants(root.id)
                    ):
                        if self._elapsed(root.id) >= self.store.config(root.id).limits.wall_seconds:
                            self._limit_tree(root.id, "Root wall-clock budget exhausted")
                root_counts: dict[str, int] = {}
                for sid in self.tasks:
                    root = self.store.session(sid).root_id
                    root_counts[root] = root_counts.get(root, 0) + 1
                sessions = sorted(self.store.sessions(), key=lambda s: (s.updated_at, s.created_at))
                for session in sessions:
                    if len(self.tasks) >= self.concurrency:
                        break
                    if (
                        session.id in self.tasks
                        or session.paused
                        or (not session.runnable and not self.has_pending_refinement(session.id))
                        or (
                            session.outcome != Outcome.ACTIVE
                            and not (
                                session.outcome == Outcome.COMPLETED
                                and self.has_pending_refinement(session.id)
                            )
                        )
                    ):
                        continue
                    root_limit = self.store.config(session.root_id).limits.concurrency
                    if root_counts.get(session.root_id, 0) >= root_limit:
                        continue
                    if session.pending_turn is None and session.runnable:
                        active = any(
                            self.store.session(s).root_id == session.root_id
                            for s in self._admitted_turns
                        )
                        if (
                            active
                            and self.store.usage(session.root_id, tree=True).turns
                            >= self.store.config(session.root_id).limits.max_turns
                        ):
                            continue  # Drain admitted turns without lifecycle/audit polling churn.
                    root_counts[session.root_id] = root_counts.get(session.root_id, 0) + 1
                    task = asyncio.create_task(
                        self._run_turn(session.id), name=f"session-{session.id}"
                    )
                    self.tasks[session.id] = task
                    task.add_done_callback(lambda t, sid=session.id: self._turn_done(sid, t))
                for session in self.store.sessions():
                    sid = session.id
                    if (
                        sid not in self.tasks
                        and session.lifecycle != Lifecycle.INACTIVE
                        and (
                            session.outcome != Outcome.ACTIVE
                            or now() - session.updated_at > self.idle_seconds
                        )
                    ):
                        await self.unload(sid)
            except Exception:
                log.exception("Scheduler error")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=0.1)
            except TimeoutError:
                pass

    def _turn_done(self, sid, task):
        self.tasks.pop(sid, None)
        if not task.cancelled() and task.exception():
            log.error("Unhandled session task failure: %s", task.exception())
        self._wake.set()

    def _limit_tree(self, root_id, reason):
        current = asyncio.current_task()
        for session in self.store.sessions(root_id=root_id):
            if session.outcome == Outcome.ACTIVE:
                self.store.finish(session.id, Outcome.LIMITED, reason)
                task = self.tasks.get(session.id)
                if task and task is not current:
                    task.cancel()

    def _completion_feedback(self, sid, reason, *, parent=None):
        event = self.store.event(sid, "completion_continuation", {"reason": reason}, parent=parent)
        self.store.add_context(
            sid,
            event,
            [
                {
                    "role": "user",
                    "content": "Runtime completion feedback: " + reason + ". "
                    "Your earlier candidate has not been returned as the task's final result. "
                    "Consider the new evidence, then return a complete, self-contained answer to "
                    "the original task, including its requested details and observed results. "
                    "Only the accepted final answer is returned; a short acknowledgment or a "
                    "reference to your earlier assessment would lose that assessment. If the "
                    "evidence warrants no changes, return the complete candidate again.",
                }
            ],
        )

    async def _run_turn(self, sid):
        started = now()
        try:
            await self.wait_refinement_barrier(sid)
            session = self.store.session(sid)
            if session.lifecycle == Lifecycle.INACTIVE:
                self.store.transition(sid, Lifecycle.IDLE)
            self.store.transition(sid, Lifecycle.RUNNING, running_since=started)
            if not self.store.session(session.root_id).started_at:
                self.store.update(session.root_id, started_at=now())
            config = self.store.config(sid)
            pending = session.pending_turn
            if pending is None:
                if not session.runnable or session.outcome == Outcome.COMPLETED:
                    await self.refinement_checkpoint(sid)
                    return
                self._check_limits(sid, resource="turns")
                self.receive(sid, include_followups=not session.runnable or session.turns == 0)
                await self._prepare(sid)
                self._check_limits(sid, resource="turns")
                self.store.charge(sid, Usage(turns=1))
                self._admitted_turns.add(sid)
                response, response_event = await self._invoke(sid)
                if (
                    not response.actions
                    and config.control_plane == "direct"
                    and self.store.session(sid).mode in {"autonomous", "goal"}
                ):
                    from .guardrails import observe

                    if observe(self, sid, "model_no_actions", {}):
                        raise LimitReached("Configured no-action loop limit reached")
                pending = self.store.session(sid).pending_turn
                assert pending and pending["event_id"] == response_event
            else:
                self._admitted_turns.add(sid)
                response = ModelResponse.model_validate(pending["response"])
                response_event = pending["event_id"]
            self.refinement_message_end(sid)
            while pending["index"] < len(response.actions):
                self._check_limits(sid)
                index = pending["index"]
                action = response.actions[index]
                action_id = pending["action_ids"][index]
                result = await self._execute_action(sid, action_id, action, response_event)
                # Persist cursor and complete conversation pair together. Recovered actions
                # reuse stored results, so a crash here cannot duplicate side effects.
                with self.store.transaction():
                    pending = self.store.session(sid).pending_turn
                    pending["index"] = index + 1
                    pending["results"].append(result)
                    if action.name == "finish" and not result.get("error"):
                        pending["completion"] = action.arguments.get("result", "")
                    self.store.update(sid, pending_turn=pending)
            # Commit one complete assistant/tools block, never orphan tool messages.
            with self.store.transaction():
                pending = self.store.session(sid).pending_turn
                if not pending.get("context_committed"):
                    messages = [
                        {
                            "role": "assistant",
                            "content": response.text or None,
                            "provider_response_event": response_event,
                        }
                    ]
                    if response.actions:
                        messages[0]["tool_calls"] = [
                            {
                                "id": a.id,
                                "type": "function",
                                "function": {"name": a.name, "arguments": encode(a.arguments)},
                            }
                            for a in response.actions
                        ]
                        for aid, result in zip(
                            [a.id for a in response.actions], pending["results"], strict=True
                        ):
                            messages.append(
                                {"role": "tool", "tool_call_id": aid, "content": encode(result)}
                            )
                    self.store.add_context(sid, response_event, messages)
                    pending["context_committed"] = True
                    self.store.update(sid, pending_turn=pending)
            completion = pending.get("completion")
            # Ordinary assistant text ends a request; explicit persistent goals still
            # require goal.complete(). Task-specific verifiers remain optional gates.
            if (
                completion is None
                and not response.actions
                and response.text
                and config.control_plane == "python"
                and self.store.session(sid).mode not in {"goal", "interactive", "heartbeat"}
            ):
                completion = response.text
            if completion is not None:
                self.store.event(
                    sid, "completion_attempt", {"result": completion}, parent=response_event
                )
            verification = None
            verification_error = False
            await self.verification.run(sid, response_event)
            completion_waiting = bool(
                completion is not None
                and config.task.wait_for_children
                and self._active_descendants(sid)
            )
            if config.task.verifier != "none" and (
                config.task.verify_each_turn or completion is not None and not completion_waiting
            ):
                verification, verification_error = await self._verify(sid, response_event)
            explicit_ok = completion is not None and (
                not config.task.require_verifier
                or (verification is not None and verification.passed)
            )
            verifier_ok = verification is not None and verification.passed
            if verification_error and config.task.require_verifier:
                explicit_ok = False
            delivered = self.receive(
                sid, include_followups=explicit_ok or verifier_ok or not response.actions
            )
            if delivered:
                if explicit_ok or verifier_ok:
                    self._completion_feedback(
                        sid, "Additional messages arrived before completion", parent=response_event
                    )
                if (
                    (explicit_ok or verifier_ok)
                    and session.parent_id
                    and any(m["sender_id"] in {None, session.parent_id} for m in delivered)
                ):
                    self.store.event(
                        sid,
                        "subagent_continued",
                        {
                            "message_id": delivered[0]["id"],
                            "previous_outcome": "active",
                            "boundary": "completion",
                            "kernel_id": session.kernel_id,
                        },
                    )
                explicit_ok = verifier_ok = False
                self.store.update(sid, runnable=True, wake_at=None)
            await self.refinement_checkpoint(sid, completed_turn=True)
            if (explicit_ok or verifier_ok) and not self._active_descendants(sid):
                late_delivery = self.receive(sid)
                if late_delivery:
                    if explicit_ok or verifier_ok:
                        self._completion_feedback(
                            sid,
                            "Additional messages arrived during completion review",
                            parent=response_event,
                        )
                    delivered.extend(late_delivery)
                    explicit_ok = verifier_ok = False
                    self.store.update(sid, runnable=True, wake_at=None)
            with self.store.transaction():
                children_active = self._active_descendants(sid)
                if (
                    (explicit_ok or verifier_ok or completion_waiting)
                    and config.task.wait_for_children
                    and children_active
                ):
                    eid = self.store.event(
                        sid, "completion_deferred", {"active_children": children_active}
                    )
                    self.store.add_context(
                        sid,
                        eid,
                        [
                            {
                                "role": "user",
                                "content": "Completion gate: descendants are still active: "
                                + encode(children_active)
                                + ". This candidate has not been returned. After incorporating "
                                "their evidence, return the complete self-contained task answer; "
                                "only the accepted final answer is returned.",
                            }
                        ],
                    )
                    self.defer(sid, config.verification.completion_wait_seconds)
                elif explicit_ok or verifier_ok:
                    if self.store.session(sid).mode == "interactive":
                        self.store.update(sid, result=completion or "Task verifier passed")
                        self.store.event(
                            sid,
                            "conversation_completed",
                            {
                                "result": completion or "Task verifier passed",
                                "verified": verifier_ok,
                            },
                        )
                        self.store.update(sid, runnable=False, wake_at=None)
                    else:
                        self.store.finish(
                            sid, Outcome.COMPLETED, completion or "Task verifier passed"
                        )
                    if self.store.session(sid).parent_id:
                        self.message(
                            sid,
                            self.store.session(sid).parent_id,
                            "Child completed: "
                            + sid
                            + "\n"
                            + (completion or "Task verifier passed")[:64000],
                            causal_request_id=self.store.last_request(sid, trajectory=True),
                        )
                current = self.store.session(sid)
                self.store.update(sid, pending_turn=None, turns=current.turns + 1)
                self._continue_completed_child(sid)
                self.store.event(sid, "turn_completed", {"turn": current.turns})
                if current.mode == "interactive":
                    if not response.actions and not delivered:
                        self.store.update(sid, runnable=False, wake_at=None)
                        self.store.event(sid, "conversation_reply", {"verified": False})
                    # Interventions arriving during a response or verifier must not
                    # be stranded by that response's conversational stop boundary.
                    if self.store.messages(sid, pending=True, limit=1) and not current.paused:
                        self.store.update(sid, runnable=True, wake_at=None)
                if current.mode == "heartbeat" and current.outcome == Outcome.ACTIVE:
                    pending_input = bool(
                        delivered or self.store.messages(sid, pending=True, limit=1)
                    )
                    self.store.update(sid, runnable=pending_input and not current.paused)
        except TurnAdmissionLimit as exc:
            root_id = self.store.session(sid).root_id
            active = [
                other
                for other in self._admitted_turns
                if other != sid and self.store.session(other).root_id == root_id
            ]
            if active:
                # A sibling exhausting admission must not cancel the final paid
                # invocation already admitted within the root's turn budget.
                self.defer(sid, 0.1)
            else:
                self._limit_tree(root_id, str(exc))
        except GoalLimitReached as exc:
            current = asyncio.current_task()
            for target in [exc.session_id, *self._active_descendants(exc.session_id)]:
                self.store.finish(target, Outcome.LIMITED, str(exc))
                task = self.tasks.get(target)
                if task and task is not current:
                    task.cancel()
                if hasattr(self, "background"):
                    await self.background.close_session(target)
        except LimitReached as exc:
            self._limit_tree(self.store.session(sid).root_id, str(exc))
        except asyncio.CancelledError:
            self.store.event(
                sid, "interruption", {"reason": "shutdown" if self._closing else "cancelled"}
            )
            raise
        except Exception as exc:
            failure = (
                exc.failure
                if isinstance(exc, HarnessError)
                else HarnessError("runtime", type(exc).__name__, str(exc)).failure
            )
            self.store.update(sid, last_error=failure)
            self.store.event(sid, "failure", failure.model_dump())
            self.store.finish(sid, Outcome.FAILED, failure.message)
            parent = self.store.session(sid).parent_id
            if parent:
                self.message(
                    sid,
                    parent,
                    f"Child {sid} failed ({failure.category}): {failure.message}",
                    causal_request_id=self.store.last_request(sid, trajectory=True),
                )
        finally:
            self._admitted_turns.discard(sid)
            if self._closing or self.store.session(sid).outcome in (
                Outcome.CANCELLED,
                Outcome.FAILED,
                Outcome.LIMITED,
            ):
                self.invalidate_refinement(sid)
            with self.store.transaction():
                self.store.charge(sid, Usage(wall_seconds=max(0, now() - started)))
                self.store.update(sid, running_since=None)
                if self.store.session(sid).lifecycle == Lifecycle.RUNNING:
                    self.store.transition(sid, Lifecycle.IDLE)
            if self.store.session(sid).outcome != Outcome.ACTIVE or self._closing:
                await self._close_kernel(sid)
                self.store.transition(sid, Lifecycle.INACTIVE)
                self._adapter_completed(sid)

    def _adapter_completed(self, sid):
        self.environment.call(sid, "completed", sid)

    async def _prepare(self, sid):
        await self.environment.prepare(sid)

    def load_adapter_context(self, sid):
        self.environment.call(sid, "admitted", sid)

    async def _invoke(self, sid):
        if sid not in self._admitted_turns:
            await self.wait_refinement_barrier(sid)
        self.context.ensure_harness_digest(sid)
        config = self.store.config(sid)
        if config.control_plane == "python" and config.features.persistent_repl:
            kernel = self._kernel(sid)
            fresh = kernel.process is None
            await kernel.start()
            if fresh:
                event = self.store.event(sid, "kernel_recovery", kernel.recovery)
                if kernel.recovery.get("missing"):
                    self.store.add_context(
                        sid,
                        event,
                        [
                            {
                                "role": "user",
                                "content": "Worker recovery missing values: "
                                + encode(kernel.recovery["missing"])[:2000],
                            }
                        ],
                    )
        compacted = await self.semantic_compact(sid)
        # Preparation, review and compaction can await while children finish.
        # Drain their queued evidence at the last safe boundary before assembling
        # this request, rather than needlessly withholding it until after the reply.
        self.receive(sid, include_followups=False)
        schemas = self.tools.schemas(config)
        messages, size = self.context.assemble(sid, schemas, proactive=compacted is not False)
        if await self._sync_l2_compaction(sid):
            messages, size = self.context.assemble(sid, schemas, proactive=False)
        session = self.store.session(sid)
        usage = self.store.usage(session.root_id, tree=True)
        remaining = (
            self.store.config(session.root_id).limits.token_budget
            - usage.input_tokens
            - usage.output_tokens
        )
        reserve = max(p.max_output_tokens for p in [config.provider, *config.models.values()])
        # Evidence-driven context pressure: leave room for continuation when repeated
        # replay approaches the cumulative budget. Limits/accounting are unchanged.
        if session.context and remaining < 3 * (size + reserve):
            target = max(4096, remaining // 2 - reserve)
            if target < size:
                self.store.event(
                    sid,
                    "budget_context_pressure",
                    {
                        "remaining_tokens": remaining,
                        "previous_estimate": size,
                        "target_input_tokens": target,
                    },
                )
                try:
                    messages, size = self.context.assemble(sid, schemas, input_budget=target)
                except HarnessError as exc:
                    if exc.failure.code != "context_capacity":
                        raise
                    raise LimitReached(
                        "Remaining token budget cannot fit necessary context"
                    ) from exc
        request = ModelRequest(
            session_id=sid,
            root_id=session.root_id,
            parent_id=session.parent_id,
            name=session.name,
            turn=session.turns,
            messages=messages,
            tools=schemas,
            config=route(self.store, sid, session.role, context_size=size),
            input_token_bound=size,
            metadata={"execution_inputs": self.context.execution_inputs(sid, messages)},
        )
        for recovery in range(config.retry.attempts):
            try:
                response, event = await self._model_call(sid, request)
                self.store.event(
                    sid,
                    "execution_input_consumed",
                    request.metadata["execution_inputs"],
                    parent=event,
                )
                self.store.event(
                    sid,
                    "agent_operating_path",
                    {
                        "path": "repl"
                        if any(a.name in {"ipython", "python"} for a in response.actions)
                        else "direct_response",
                        "rationale": response.text,
                        "actions": [a.name for a in response.actions],
                    },
                    parent=event,
                )
                return response, event
            except HarnessError as exc:
                if exc.failure.code not in {
                    "context_overflow",
                    "context_length_exceeded",
                    "context_capacity",
                }:
                    raise
                if recovery + 1 >= config.retry.attempts or not self.store.session(sid).context:
                    raise
                await self.semantic_compact(sid, force=True)
                messages, smaller = self.context.assemble(sid, schemas)
                if smaller >= size:
                    self.context.compact(sid)
                    messages, smaller = self.context.assemble(sid, schemas)
                if smaller >= size:
                    raise
                if await self._sync_l2_compaction(sid):
                    messages, smaller = self.context.assemble(sid, schemas, proactive=False)
                self.store.charge(sid, Usage(retries=1))
                self.store.event(
                    sid,
                    "context_overflow_recovery",
                    {
                        "turn": request.turn,
                        "before_tokens": size,
                        "after_tokens": smaller,
                        "failure": exc.failure.code,
                    },
                )
                size = smaller
                request = request.model_copy(
                    update={
                        "messages": messages,
                        "input_token_bound": size,
                        "metadata": {
                            **request.metadata,
                            "execution_inputs": self.context.execution_inputs(sid, messages),
                        },
                    }
                )
        raise AssertionError("Unreachable context recovery loop")

    async def _model_call(self, sid, request, *, persist_turn=True, max_attempts=None):
        request = request.model_copy(deep=True, update={"request_id": new_id()})
        try:
            return await self._model_call_with_retries(
                sid, request, persist_turn=persist_turn, max_attempts=max_attempts
            )
        except BaseException:
            # Admission failure or cancellation during retry backoff also closes
            # the logical request, even though no new transport attempt started.
            self.store.db.execute(
                "UPDATE model_requests SET status='failed',ended_at=? WHERE id=? AND status IN ('running','retrying')",
                (now(), request.request_id),
            )
            raise

    async def _model_call_with_retries(self, sid, request, *, persist_turn=True, max_attempts=None):
        config = self.store.config(sid)
        attempts = max_attempts if max_attempts is not None else config.retry.attempts
        session = self.store.session(sid)
        size, provider = request.input_token_bound, request.config
        if provider.name == "codex_subscription":
            previous = provider
            resolver = self.providers[provider.name]
            provider, _ = (
                await resolver.resolve(provider, reasoning_off=True)
                if request.reasoning_mode == "off"
                else await resolver.resolve(provider)
            )
            if request.metadata.get("purpose") in {"refinement", "refinement_review"}:
                from .refinement_model import refinement_output_limit

                provider = provider.model_copy(
                    update={
                        "max_output_tokens": refinement_output_limit(
                            provider, review=request.metadata["purpose"] == "refinement_review"
                        )
                    }
                )
            if request.request_kind == "trajectory":
                self.store.pin_provider(sid, previous, provider)
            request = request.model_copy(update={"config": provider})
            self.store.event(
                sid,
                "subscription_model_selected",
                {
                    "model": provider.model,
                    "parameters": provider.parameters,
                    "billing": "subscription",
                    "output_limit_enforcement": "provider model token limit; cumulative runtime token budget",
                },
            )
        # One logical identity per body; transport retries below reuse it.
        request_artifact = self.artifacts.put(sid, request.public_dump())
        registered = False
        for attempt in range(attempts):
            while True:
                try:
                    with self.store.transaction():
                        self._check_limits(
                            sid, resource="model_calls", input_bound=size, provider=provider
                        )
                        if not registered:
                            self.store.begin_request(request, request_artifact)
                            registered = True
                        eid = self.store.event(
                            sid,
                            "model_invocation_started",
                            {
                                "request_id": request.request_id,
                                "attempt": attempt + 1,
                                "request_artifact": request_artifact,
                                "turn": session.turns,
                                "provider": provider.name,
                                "model": provider.model,
                                "purpose": request.metadata.get("purpose", "agent"),
                            },
                        )
                        self.store.db.execute(
                            "INSERT INTO model_attempts VALUES(?,?,?,?,?,?,?)",
                            (
                                eid,
                                request.request_id,
                                attempt + 1,
                                "running",
                                encode(Usage(model_calls=1).model_dump()),
                                None,
                                None,
                            ),
                        )
                        self.store.db.execute(
                            "UPDATE model_requests SET status='running' WHERE id=?",
                            (request.request_id,),
                        )
                        cost = (
                            size * (provider.input_cost_per_million or 0)
                            + provider.max_output_tokens * (provider.output_cost_per_million or 0)
                        ) / 1_000_000
                        self.store.db.execute(
                            "INSERT INTO reservations VALUES(?,?,?,?,?)",
                            (eid, sid, size, provider.max_output_tokens, cost),
                        )
                        self.store.charge(sid, Usage(model_calls=1), parent=eid)
                    break
                except BudgetBusy:
                    await asyncio.sleep(0.05)
            buffer = []
            last_flush = 0.0

            async def emit(delta, buffer=buffer, eid=eid):
                nonlocal last_flush
                buffer.append(delta)
                if sum(map(len, buffer)) >= 512 or now() - last_flush >= 0.1:
                    self.store.event(
                        sid,
                        "model_stream",
                        {
                            "text": "".join(buffer),
                            "purpose": request.metadata.get("purpose", "agent"),
                        },
                        parent=eid,
                    )
                    buffer.clear()
                    last_flush = now()

            try:
                async with asyncio.timeout(
                    min(
                        provider.timeout_seconds,
                        max(
                            0.01,
                            self.store.config(session.root_id).limits.wall_seconds
                            - self._elapsed(session.root_id),
                        ),
                    )
                ):
                    response = await self.providers[provider.name].invoke(
                        request.model_copy(deep=True), emit
                    )
                if buffer:
                    self.store.event(
                        sid,
                        "model_stream",
                        {
                            "text": "".join(buffer),
                            "purpose": request.metadata.get("purpose", "agent"),
                        },
                        parent=eid,
                    )
                if not persist_turn:
                    response.metadata["purpose"] = request.metadata.get("purpose", "auxiliary")
                if not response.usage_reported:
                    response.usage.estimated_calls += 1
                with self.store.transaction():
                    self.store.charge(sid, response.usage, parent=eid)
                    self.store.db.execute("DELETE FROM reservations WHERE id=?", (eid,))
                    response_event = self.store.event(
                        sid, "model_response", response.model_dump(mode="json"), parent=eid
                    )
                    measured = response.usage.model_copy(
                        update={"model_calls": 1, "retries": int(attempt > 0)}
                    )
                    self.store.finish_request_attempt(
                        request.request_id, eid, measured, response_event=response_event
                    )
                    if response.provider_items:
                        self.store.db.execute(
                            "INSERT INTO provider_continuations VALUES(?,?,?,?)",
                            (
                                response_event,
                                provider.name,
                                provider.model,
                                encode(response.provider_items),
                            ),
                        )
                    if persist_turn:
                        from .request_context import record_usage

                        record_usage(self.store, sid, request, response, response_event)
                        self.store.update(
                            sid,
                            pending_turn={
                                "response": response.model_dump(mode="json"),
                                "event_id": response_event,
                                "action_ids": [new_id() for _ in response.actions],
                                "index": 0,
                                "results": [],
                                "completion": None,
                            },
                        )
                return response, response_event
            except asyncio.CancelledError:
                # Graceful cancellation settles now; hard process death is handled in recover().
                with self.store.transaction():
                    self.store.charge(
                        sid,
                        Usage(
                            input_tokens=size,
                            output_tokens=provider.max_output_tokens,
                            cost=cost,
                            estimated_calls=1,
                        ),
                        parent=eid,
                    )
                    self.store.db.execute("DELETE FROM reservations WHERE id=?", (eid,))
                    self.store.finish_request_attempt(
                        request.request_id,
                        eid,
                        Usage(
                            input_tokens=size,
                            output_tokens=provider.max_output_tokens,
                            cost=cost,
                            estimated_calls=1,
                            model_calls=1,
                            retries=int(attempt > 0),
                        ),
                        failure={"code": "cancelled", "uncertain": True},
                    )
                raise
            except Exception as exc:
                failure = (
                    exc.failure
                    if isinstance(exc, HarnessError)
                    else HarnessError(
                        "provider",
                        type(exc).__name__,
                        str(exc),
                        retryable=isinstance(exc, TimeoutError),
                        uncertain=True,
                    ).failure
                )
                with self.store.transaction():
                    # Conservatively charge unknown failed requests, even when a provider omits usage.
                    self.store.charge(
                        sid,
                        Usage(
                            input_tokens=size,
                            output_tokens=provider.max_output_tokens,
                            cost=cost,
                            estimated_calls=1,
                        ),
                        parent=eid,
                    )
                    self.store.db.execute("DELETE FROM reservations WHERE id=?", (eid,))
                    self.store.event(sid, "failure", failure.model_dump(), parent=eid)
                    self.store.finish_request_attempt(
                        request.request_id,
                        eid,
                        Usage(
                            input_tokens=size,
                            output_tokens=provider.max_output_tokens,
                            cost=cost,
                            estimated_calls=1,
                            model_calls=1,
                            retries=int(attempt > 0),
                        ),
                        failure=failure.model_dump(),
                        retry=failure.retryable and attempt + 1 < attempts,
                    )

                if (
                    failure.code
                    in {"context_overflow", "context_length_exceeded", "context_capacity"}
                    or not failure.retryable
                    or attempt + 1 >= attempts
                ):
                    if isinstance(exc, HarnessError):
                        raise
                    raise HarnessError(
                        failure.category,
                        failure.code,
                        failure.message,
                        retryable=failure.retryable,
                        uncertain=failure.uncertain,
                    ) from exc
                self.store.charge(sid, Usage(retries=1))
                delay = min(config.retry.max_delay, config.retry.initial_delay * 2**attempt)
                self.store.event(
                    sid,
                    "retry",
                    {"attempt": attempt + 2, "delay": delay, "category": failure.category},
                    parent=eid,
                )
                await asyncio.sleep(delay)
        raise AssertionError("Unreachable retry loop")

    async def _execute_action(self, sid, action_id, action: Action, parent, *, from_python=False):
        if self.store.session(sid).paused:
            raise asyncio.CancelledError("Session paused before action")
        old = self.store.db.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone()
        if old and old["status"] == "done":
            return json.loads(old["result"])
        if old:
            return self._action_result(
                sid,
                action_id,
                {
                    "error": HarnessError(
                        "runtime",
                        "uncertain_action",
                        "An interrupted action must be inspected before retrying",
                        uncertain=True,
                    ).failure.model_dump()
                },
                parent,
            )
        self._check_limits(sid, resource="tool_calls")
        from .guardrails import observe

        if observe(self, sid, action.name, action.arguments):
            raise LimitReached("Configured repeated-action limit reached without progress")
        with self.store.transaction():
            eid = self.store.event(
                sid,
                "tool_call",
                {
                    "action_id": action_id,
                    "name": action.name,
                    "arguments": action.arguments,
                    "from_python": from_python,
                },
                parent=parent,
            )
            self.store.db.execute(
                "INSERT INTO actions VALUES(?,?,?,?,?,?,?,?)",
                (action_id, sid, action.name, encode(action.arguments), "running", None, eid, None),
            )
            self.store.charge(sid, Usage(tool_calls=1), parent=eid)
        config = self.store.config(sid)
        context = ToolContext(self, sid, action_id, eid, from_python)
        external_checkpoint = None
        interrupted = None
        raw = None
        admitted = False
        try:
            if not from_python and config.control_plane == "python" and action.name != "ipython":
                raise HarnessError(
                    "model",
                    "tool_not_exposed",
                    "Only ipython is exposed; call capabilities from Python",
                )
            tool, validated = self.tools.resolve(context, action.name, action.arguments)
            external_checkpoint = await self.environment.before_action(context, tool.name)
            admitted = True
            self.store.event(
                sid,
                "environment_action_started",
                {"action_id": action_id, "capability": tool.name, "from_python": from_python},
                parent=eid,
            )
            timeout = (
                config.limits.python_timeout_seconds + 25
                if action.name in ("python", "ipython")
                else config.limits.tool_timeout_seconds
            )
            async with asyncio.timeout(timeout):
                raw = await self.tools.execute(context, tool, validated)
                encode(raw)  # Structured tools must produce JSON-compatible values.
        except (LimitReached, asyncio.CancelledError) as exc:
            interrupted = exc
            raw = {
                "error": HarnessError(
                    "runtime",
                    "action_cancelled",
                    "Action cancelled; inspect external effects before retrying",
                    uncertain=True,
                ).failure.model_dump()
            }
        except Exception as exc:
            failure = (
                exc.failure
                if isinstance(exc, HarnessError)
                else HarnessError(
                    "tool", type(exc).__name__, str(exc), uncertain=isinstance(exc, TimeoutError)
                ).failure
            )
            raw = {"error": failure.model_dump()}
            self.store.event(sid, "failure", failure.model_dump(), parent=eid)
        finally:
            if admitted:
                try:
                    self.environment.after_action(context, external_checkpoint)
                    self.store.event(
                        sid,
                        "environment_action_finished",
                        {"action_id": action_id, "capability": tool.name},
                        parent=eid,
                    )
                except Exception as exc:
                    failure = HarnessError(
                        "environment", "after_action_failed", str(exc), uncertain=True
                    ).failure
                    # Execution may already have changed the workspace. Preserve its
                    # result separately and never report a successful/rolled-back action.
                    execution = self.artifacts.expose(sid, raw, source_event=eid)
                    raw = {
                        "error": failure.model_dump(),
                        "execution_result": execution,
                        "rollback_performed": False,
                    }
                    self.store.event(sid, "failure", failure.model_dump(), parent=eid)
                    self.store.update(sid, paused=True, runnable=False)
        result = self._action_result(sid, action_id, raw, eid)
        if interrupted is not None:
            raise interrupted
        # Python receives full values for computation, while the action journal/context
        # always stores a bounded preview plus the durable artifact.
        if from_python:
            if isinstance(raw, dict) and raw.get("error"):
                raise HarnessError("tool", "nested_tool_failed", encode(raw["error"]))
            return raw
        return result

    def _action_result(self, sid, action_id, raw, parent, *, recovered=False):
        with self.store.transaction():
            exposed = self.artifacts.expose(sid, raw, source_event=parent)
            if isinstance(raw, dict) and raw.get("error"):
                exposed["error"] = raw["error"]
            eid = self.store.event(
                sid,
                "tool_result",
                {"action_id": action_id, "result": exposed, "recovered": recovered},
                parent=parent,
            )
            self.store.db.execute(
                "UPDATE actions SET status='done',result=?,result_event=? WHERE id=?",
                (encode(exposed), eid, action_id),
            )
            if isinstance(raw, dict) and (
                raw.get("passed") is False
                or isinstance(raw.get("exit_code"), int)
                and raw["exit_code"] != 0
                or raw.get("error")
            ):
                self.store.event(
                    sid,
                    "execution_failure_observed",
                    {"action_id": action_id, "result_event": eid, "result": exposed},
                    parent=eid,
                )
            return exposed

    def _kernel(self, sid):
        if sid not in self.kernels:
            session = self.store.session(sid)

            async def bridge(name, arguments):
                # Nested calls are journaled independently and share root accounting.
                return await self._execute_action(
                    sid,
                    new_id(),
                    Action(name=name, arguments=arguments),
                    self._python_parent[sid],
                    from_python=True,
                )

            self.kernels[sid] = Kernel(
                self.store.directory / "kernels" / session.kernel_id,
                Path(session.workspace.path),
                bridge,
                env=environment(self.store.config(sid).execution),
                bootstrap={
                    "session_id": sid,
                    "kernel_state": self.store.config(sid).kernel_state.model_dump(),
                    "root_id": session.root_id,
                    "parent_id": session.parent_id,
                    "depth": session.depth,
                    "name": session.name,
                    "task": session.instruction,
                    "original_task": self.context.original_task(sid),
                    "messages_path": str(self.context.history_file(sid)),
                    "control_plane": self.store.config(sid).control_plane,
                    "research_read_only": self.store.config(sid).execution.read_only,
                    "namespace_factories": self.environment.namespace_factories(
                        self.store.config(sid)
                    ),
                    "argument_schemas": {
                        name: tool.arguments.model_json_schema()
                        for name, tool in self.tools.entries.items()
                        if tool.python_callable and self.tools.allowed(name, self.store.config(sid))
                    },
                    "skills": [
                        e
                        for e in discover(
                            Path(session.workspace.path), self.store.config(sid).skill_paths
                        )
                        if set(e.get("required_permissions", []))
                        <= set(self.store.config(sid).permissions)
                    ],
                },
            )
        return self.kernels[sid]

    async def execute_python(self, context, code):
        observation = await self.environment.before_python(context)
        try:
            return await self._execute_python(context, code)
        finally:
            self.environment.after_python(context, observation)

    async def _execute_python(self, context, code):
        sid = context.session_id
        if self.store.config(sid).control_plane == "python":
            self.context.history_file(sid)
        if not self.store.config(sid).features.persistent_repl:
            await self._close_kernel(sid)
            checkpoint = (
                self.store.directory
                / "kernels"
                / self.store.session(sid).kernel_id
                / "checkpoint.json"
            )
            checkpoint.unlink(missing_ok=True)
        self._check_limits(sid, resource="python_executions")
        self._python_parent[sid] = context.source_event
        kernel = self._kernel(sid)
        fresh = kernel.process is None
        await kernel.start()
        if fresh:
            eid = self.store.event(
                sid, "kernel_recovery", kernel.recovery, parent=context.source_event
            )
            if kernel.recovery.get("missing"):
                self.store.add_context(
                    sid,
                    eid,
                    [
                        {
                            "role": "user",
                            "content": "Worker recovery missing values: "
                            + encode(kernel.recovery["missing"])[:2000],
                        }
                    ],
                )
        self.store.charge(sid, Usage(python_executions=1), parent=context.source_event)
        eid = self.store.event(
            sid,
            "python_execution",
            {"execution_id": context.action_id, "code": code},
            parent=context.source_event,
        )
        result = await kernel.execute(
            context.action_id, code, self.store.config(sid).limits.python_timeout_seconds
        )
        self._record_kernel_state(sid, result, eid)
        result = self._capture_kernel_logs(sid, result, eid)
        self.store.event(
            sid,
            "python_error" if result.get("error") else "python_result",
            {"result": result},
            parent=eid,
        )
        return result

    def _record_kernel_state(self, sid, result, parent):
        metrics = result.get("snapshot_metrics", {})
        artifact = self.artifacts.put(sid, result.get("kernel_state", {}), source_event=parent)
        event = self.store.event(
            sid,
            "kernel_snapshot",
            {
                **metrics,
                "manifest_artifact": artifact,
                "missing": result.get("not_checkpointed", {}),
            },
            parent=parent,
        )
        for action in ("offloaded", "pruned", "reconstructible"):
            if metrics.get(action):
                self.store.event(
                    sid,
                    "kernel_variables_" + action,
                    {
                        "names": metrics[action],
                        "manifest_artifact": artifact,
                        "reason": metrics.get("reason"),
                    },
                    parent=event,
                )
        # Full inventory lives in L3; expose a small operational receipt in L1.
        result.pop("kernel_state", None)
        result["state_manifest_artifact"] = artifact

    async def _sync_l2_compaction(self, sid):
        boundaries = self.store.events(sid, kind="context_compaction", limit=1)
        snapshots = self.store.events(sid, kind="kernel_snapshot", limit=1)
        if not boundaries or sid not in self.kernels:
            return
        if snapshots and snapshots[0]["seq"] > boundaries[0]["seq"]:
            return
        try:
            result = await self.kernels[sid].checkpoint_state()
            self._record_kernel_state(sid, result, boundaries[0]["id"])
            return True
        except Exception as exc:
            self.store.event(
                sid, "kernel_snapshot_failed", {"reason": str(exc)}, parent=boundaries[0]["id"]
            )

    def _capture_kernel_logs(self, sid, result, event):
        result = dict(result)
        kernel_dir = self._kernel(sid).directory.resolve()
        for key in ("stdout", "stderr"):
            path = Path(result.pop(key + "_path", "")).resolve()
            if path.is_file() and path.is_relative_to(kernel_dir):
                with path.open("rb") as stream:
                    result[key + "_artifact"] = self.artifacts.put_stream(
                        sid, stream, source_event=event
                    )
        return result

    async def _verify(self, sid, parent):
        # Serialize gates for one working tree. The second requester observes the
        # first committed receipt instead of launching an identical concurrent suite.
        if not hasattr(self, "_verification_locks"):
            self._verification_locks = {}
        workspace = str(
            await asyncio.to_thread(Path(self.store.session(sid).workspace.path).resolve)
        )
        lock = self._verification_locks.setdefault(workspace, asyncio.Lock())
        async with lock:
            return await self._verify_locked(sid, parent)

    async def _verify_locked(self, sid, parent):
        config = self.store.config(sid)
        for attempt in range(config.retry.attempts):
            self._check_limits(sid)
            eid = parent
            try:
                async with asyncio.timeout(config.limits.tool_timeout_seconds):
                    await self.environment.prepare(sid, force=True)
                    from .verification_receipts import VerificationReceipts

                    receipts = VerificationReceipts(ToolContext(self, sid, new_id(), parent))
                    identity = receipts.fingerprint()
                    verification = receipts.find(identity)
                    reused = verification is not None
                    if not reused:
                        self.store.charge(sid, Usage(verifier_calls=1), parent=parent)
                        eid = self.store.event(
                            sid,
                            "verifier_started",
                            {
                                "attempt": attempt + 1,
                                "level": 3,
                                "reason": "completion gate or explicit full verification",
                            },
                            parent=parent,
                        )
                        self.store.event(
                            sid,
                            "verification_executed",
                            {
                                "level": 3,
                                "key": identity["key"] if identity else None,
                            },
                            parent=eid,
                        )
                        verification = await self.adapters[config.task.adapter].verify(
                            ToolContext(self, sid, new_id(), eid), config.task
                        )
                result = verification.model_dump() if verification else {"skipped": True}
                from .verification import concise

                full_artifact = self.artifacts.put(sid, result, source_event=eid)
                exposed = self.artifacts.expose(
                    sid,
                    {
                        **concise(result, config.verification),
                        "full_verifier_artifact": full_artifact,
                    },
                    source_event=eid,
                )
                result_event = self.store.event(
                    sid,
                    "verifier_result",
                    {
                        "result": exposed,
                        "passed": verification.passed if verification else None,
                        "level": 3,
                        "receipt_reused": reused,
                    },
                    parent=eid,
                )
                self.store.add_context(
                    sid, result_event, [{"role": "user", "content": "Verifier: " + encode(exposed)}]
                )
                if not reused:
                    receipts.save(identity, verification, full_artifact, result_event)
                if verification and not verification.passed:
                    self.retain_failure(sid, result_event, verification)
                return verification, False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = HarnessError(
                    "verifier", type(exc).__name__, str(exc), retryable=True
                ).failure
                fail_event = self.store.event(sid, "failure", failure.model_dump(), parent=eid)
                if attempt + 1 < config.retry.attempts:
                    self.store.charge(sid, Usage(retries=1))
                    self.store.event(
                        sid, "retry", {"category": "verifier", "attempt": attempt + 2}, parent=eid
                    )
                    await asyncio.sleep(
                        min(config.retry.max_delay, config.retry.initial_delay * 2**attempt)
                    )
                else:
                    self.store.add_context(
                        sid,
                        fail_event,
                        [
                            {
                                "role": "user",
                                "content": "Verifier infrastructure failure (not task failure): "
                                + encode(failure.model_dump())[:2000],
                            }
                        ],
                    )
        return None, True

    def _active_descendants(self, sid):
        root = self.store.session(sid).root_id
        descendants, frontier = [], [sid]
        sessions = self.store.sessions(root_id=root)
        while frontier:
            parent = frontier.pop()
            for session in sessions:
                if session.parent_id == parent:
                    frontier.append(session.id)
                    if session.outcome == Outcome.ACTIVE:
                        descendants.append(session.id)
        return descendants

    async def stop(self, sid, *, tree=True):
        self.store.session(sid)
        ids = [sid]
        if tree:
            # Finished children may still own host-managed background processes.
            # Traverse the whole ownership tree, preserving completed outcomes.
            sessions = self.store.sessions(root_id=self.store.session(sid).root_id)
            for owner in ids:
                ids.extend(s.id for s in sessions if s.parent_id == owner)
        for target in ids:
            if target == sid or self.store.session(target).outcome == Outcome.ACTIVE:
                self.store.finish(target, Outcome.CANCELLED, "Stopped by user")
            self.invalidate_refinement(target)
            if task := self.tasks.get(target):
                task.cancel()
        await asyncio.gather(
            *(self.tasks[target] for target in ids if target in self.tasks), return_exceptions=True
        )
        for target in ids:
            if hasattr(self, "background"):
                await self.background.close_session(target)
            if hasattr(self, "mcp"):
                await self.mcp.close_session(target)
            await self.unload(target)

    async def _close_kernel(self, sid):
        if kernel := self.kernels.pop(sid, None):
            cleanup = asyncio.create_task(kernel.close())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Shutdown may cancel a turn already in its finally block. The
                # kernel has left the registry, so this owner must finish teardown.
                await cleanup
                raise

    async def unload(self, sid):
        if sid in self.tasks and not self.tasks[sid].done():
            raise ValueError("Cannot unload a running session")
        await self._close_kernel(sid)
        self.store.transition(sid, Lifecycle.INACTIVE)

    async def pause(self, sid):
        self.store.update(sid, runnable=False, paused=True, wake_at=None)
        self.invalidate_refinement(sid)
        kernel = self.kernels.get(sid)
        if kernel and kernel.active_execution:
            await kernel.interrupt(kernel.active_execution)
        if task := self.tasks.get(sid):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Keep an interrupted live namespace loaded; the daemon owns it, not UI.
        if not kernel or not kernel.process:
            await self.unload(sid)
        self.store.event(sid, "paused", {})

    async def fork(self, sid, *, name=None):
        source = self.store.session(sid)
        if sid in self.tasks:
            raise ValueError("Pause the source session before forking it")
        if command := self.refinement_state(sid).command_task:
            await asyncio.shield(command)
        await self.refinement_branch_changed(sid)
        last = self.store.events(sid, limit=1)
        config = self.store.config(sid).model_copy(deep=True)
        workspace, _ = self.environment.continuation_workspace(source, config)
        with self.store.transaction():
            branch = self.store.create(
                source.instruction,
                workspace,
                config,
                name=name or source.name + "-branch",
                mode=source.mode,
                branch_from=sid,
                branch_event=last[-1]["id"] if last else None,
            )
            turn_index = source.turns + (1 if source.pending_turn else 0)
            self.store.update(
                branch.id,
                context=source.context,
                summary=source.summary,
                summary_harness_digest=source.summary_harness_digest,
                summary_timestamp=source.summary_timestamp,
                adapter_context=source.adapter_context,
                turns=turn_index,
            )
            eid = self.store.event(
                branch.id,
                "branch",
                {
                    "source_session": sid,
                    "source_event": branch.branch_event,
                    "inherited_turn_index": turn_index,
                },
            )
            if source.pending_turn:
                self.store.add_context(
                    branch.id,
                    eid,
                    [
                        {
                            "role": "user",
                            "content": "Forked after a partial turn. The source action journal remains intact; no pending actions "
                            "were replayed in this branch. Retrieve source history and inspect external effects. "
                            "Source session: " + sid,
                        }
                    ],
                )
            from .harness import (
                save_harness_state,
            )

            save_harness_state(self.store.harness.path(branch.id), self.store.harness.load(sid))
            for record in self.store.refinement_history(sid):
                self.store.record_harness_refinement(branch.id, record)
            source_dir = self.store.directory / "kernels" / source.kernel_id
            target_dir = self.store.directory / "kernels" / branch.kernel_id
            if source_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = source_dir / "checkpoint.json"
                if checkpoint.exists():
                    fork_checkpoint(
                        checkpoint,
                        target_dir / "checkpoint.json",
                        source.workspace.path,
                        workspace.path,
                        owner=branch.id,
                    )
            self.store.event(
                sid, "branch_created", {"branch_id": branch.id}, parent=source.branch_event
            )
        self._wake.set()
        return self.store.session(branch.id)

    async def wait(self, sid, *, timeout=30):  # noqa: ASYNC109 - public bounded inspection API
        async with asyncio.timeout(timeout):
            while True:
                session = self.store.session(sid)
                if session.outcome != Outcome.ACTIVE and sid not in self.tasks:
                    return session
                await asyncio.sleep(0.02)

    async def shutdown(self):
        if not hasattr(self, "_shutdown_task"):
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self):
        # Prime drains valid refinement before disposing the context/kernel stores.
        for sid in list(self._refinement_states):
            await self.drain_refinement(sid)
        self._closing = True
        self._wake.set()
        if self._scheduler_task:
            await self._scheduler_task
            self._scheduler_task = None
        tasks = list(
            set(self.tasks.values())
            | {state.task for state in self._refinement_states.values() if state.task}
            | {state.background for state in self._refinement_states.values() if state.background}
            | {state.claim for state in self._refinement_states.values() if state.claim}
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sid in list(self.kernels):
            await self._close_kernel(sid)
        for service in ("background", "mcp"):
            if hasattr(self, service):
                await getattr(self, service).close()
        for session in self.store.sessions():
            if session.lifecycle != Lifecycle.INACTIVE:
                self.store.transition(session.id, Lifecycle.INACTIVE)
        self.environment.close()
        self.store.close()
        self._owner_lock.close()
