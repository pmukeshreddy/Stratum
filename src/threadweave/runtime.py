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
import shutil
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
from .refinement import MemoryServices
from .routing import route
from .storage import Store, encode
from .tools import ToolContext, builtins

log = logging.getLogger("threadweave.runtime")


class LimitReached(Exception):
    pass


class BudgetBusy(Exception):
    """Another invocation temporarily owns a reservation that may be released."""


class Runtime(MemoryServices):
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
        self.context = Context(self.store)
        self.providers = providers if providers is not None else default_providers()
        self.environment = Environment(self, adapters)
        self.adapters = self.environment.adapters  # Extension registration remains compatible.
        self.tools = tools if tools is not None else builtins()
        self.concurrency, self.idle_seconds = concurrency, idle_seconds
        self.kernels: dict[str, Kernel] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self._scheduler_task = None
        self._wake = asyncio.Event()
        self._closing = False
        self._extensions: set[str] = set()
        self._python_parent: dict[str, str] = {}

    def load_extensions(self, config):
        for reference in config.extensions:
            if reference not in self._extensions:
                module, function = reference.split(":", 1)
                getattr(importlib.import_module(module), function)(self)
                self._extensions.add(reference)

    def validate_config(self, config):
        RunConfig.model_validate(config.model_dump())
        self.load_extensions(config)
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

    def spawn(self, parent_id: str, instruction: str, *, name=None, role="agent"):
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
        workspace, checkpoint = self.environment.continuation_workspace(parent, config, child=True)
        config.refinement.selected_entries = [
            eid
            for eid in parent.selected_state
            if self.store.state(parent_id, eid)["owner_id"] is None
        ]
        session = self.store.create(
            instruction,
            workspace,
            config,
            parent_id=parent_id,
            name=name or f"child-{usage.subagent_count + 1}",
            mode="goal",
        )
        if checkpoint:
            self.store.db.execute(
                "INSERT INTO candidates VALUES(?,?,?,?)",
                (
                    session.id,
                    parent_id,
                    checkpoint,
                    encode(
                        {
                            "instruction": instruction,
                            "start_time": session.created_at,
                            "consumed": False,
                            "accepted": False,
                        }
                    ),
                ),
            )
        self.store.update(session.id, role=role)
        self._wake.set()
        return self.store.session(session.id)

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
                "selected_entries": len(session.selected_state),
                "selected_entry_ids": session.selected_state[:50],
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
                "state_entries": len(self.store.states(sid, include_deleted=True)),
                "pending_messages": self.store.db.execute(
                    "SELECT COUNT(*) FROM messages WHERE recipient_id=? AND received_at IS NULL",
                    (sid,),
                ).fetchone()[0],
                "schedules": count("schedules", " AND enabled=1"),
                "goal": self.store.goal(sid),
                "history_append_only": True,
            },
        }

    def message(self, sender_id, recipient_id, body):
        if sender_id:
            self.store.ensure_related(sender_id, recipient_id)
        with self.store.transaction():
            mid = self.store.send(sender_id, recipient_id, body)
            if sender_id is None and body.strip() == "/refine":
                self.store.event(
                    recipient_id, "refinement_trigger", {"reason": "human", "message_id": mid}
                )
            recipient = self.store.session(recipient_id)
            # Terminal sessions retain messages, but are resumed only by explicit human control.
            if recipient.outcome == Outcome.ACTIVE and not recipient.paused:
                self.store.update(recipient_id, runnable=True, wake_at=None)
        self._wake.set()
        return mid

    def receive(self, sid):
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

        return self.store.receive(sid, render, limit=20)

    def interact(self, sid, body):
        """Atomic human input + continuation; no client-side input/resume race."""
        if not body.strip():
            raise ValueError("Message cannot be empty")
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
        session = self.store.session(sid)
        self.validate_config(self.store.config(sid))
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
        if self._elapsed(root.id) >= limits.wall_seconds:
            raise LimitReached("Root wall-clock budget exhausted")
        reserved_tokens, reserved_cost = self.store.reserved(root.id)
        tokens = usage.input_tokens + usage.output_tokens + reserved_tokens
        if usage.input_tokens + usage.output_tokens >= limits.token_budget:
            raise LimitReached("Root token budget exhausted")
        if limits.cost_budget is not None and usage.cost >= limits.cost_budget:
            raise LimitReached("Root cost budget exhausted")
        if resource == "turns" and usage.turns >= limits.max_turns:
            raise LimitReached("Root turn limit exhausted")
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
        await self.environment.recover()
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
                self.validate_config(self.store.config(session.id))
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
                    if row["name"] in ("python", "skill_run")
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
                        or not session.runnable
                        or session.outcome != Outcome.ACTIVE
                    ):
                        continue
                    root_limit = self.store.config(session.root_id).limits.concurrency
                    if root_counts.get(session.root_id, 0) >= root_limit:
                        continue
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

    async def _run_turn(self, sid):
        started = now()
        try:
            session = self.store.session(sid)
            if session.lifecycle == Lifecycle.INACTIVE:
                self.store.transition(sid, Lifecycle.IDLE)
            self.store.transition(sid, Lifecycle.RUNNING, running_since=started)
            if not self.store.session(session.root_id).started_at:
                self.store.update(session.root_id, started_at=now())
            config = self.store.config(sid)
            pending = session.pending_turn
            if pending is None:
                self._check_limits(sid, resource="turns")
                self.store.apply_refinements(sid)
                self.receive(sid)
                await self._prepare(sid)
                await self.auto_refine(sid)
                self._check_limits(sid, resource="turns")
                self.store.charge(sid, Usage(turns=1))
                response, response_event = await self._invoke(sid)
                if not response.actions and self.store.session(sid).mode in {"autonomous", "goal"}:
                    from .guardrails import observe

                    if observe(self, sid, "model_no_actions", {}):
                        raise LimitReached("Configured no-action loop limit reached")
                pending = self.store.session(sid).pending_turn
                assert pending and pending["event_id"] == response_event
            else:
                response = ModelResponse.model_validate(pending["response"])
                response_event = pending["event_id"]
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
                    messages = [{"role": "assistant", "content": response.text or None}]
                    if response.actions:
                        messages[0]["tool_calls"] = [
                            {
                                "id": aid,
                                "type": "function",
                                "function": {"name": a.name, "arguments": encode(a.arguments)},
                            }
                            for a, aid in zip(response.actions, pending["action_ids"], strict=True)
                        ]
                        for aid, result in zip(
                            pending["action_ids"], pending["results"], strict=True
                        ):
                            messages.append(
                                {"role": "tool", "tool_call_id": aid, "content": encode(result)}
                            )
                    self.store.add_context(sid, response_event, messages)
                    pending["context_committed"] = True
                    self.store.update(sid, pending_turn=pending)
            completion = pending.get("completion")
            if completion is not None:
                self.store.event(
                    sid, "completion_attempt", {"result": completion}, parent=response_event
                )
            verification = None
            verification_error = False
            if config.task.verifier != "none" and (
                config.task.verify_each_turn or completion is not None
            ):
                verification, verification_error = await self._verify(sid, response_event)
            explicit_ok = completion is not None and (
                not config.task.require_verifier
                or (verification is not None and verification.passed)
            )
            verifier_ok = verification is not None and verification.passed
            if verification_error and config.task.require_verifier:
                explicit_ok = False
            if (explicit_ok or verifier_ok) and not self._active_descendants(sid):
                await self.auto_refine(sid, trigger="completion")
            with self.store.transaction():
                children_active = self._active_descendants(sid)
                if (
                    (explicit_ok or verifier_ok)
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
                                + encode(children_active),
                            }
                        ],
                    )
                    self.defer(sid, 0.2)
                elif explicit_ok or verifier_ok:
                    self.store.apply_refinements(sid)
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
                            "Child completed: " + (completion or "Task verifier passed"),
                        )
                current = self.store.session(sid)
                self.store.update(sid, pending_turn=None, turns=current.turns + 1)
                self.store.event(sid, "turn_completed", {"turn": current.turns})
                if current.mode == "interactive":
                    if not response.actions:
                        self.store.update(sid, runnable=False, wake_at=None)
                        self.store.event(sid, "conversation_reply", {"verified": False})
                    # Interventions arriving during a response or verifier must not
                    # be stranded by that response's conversational stop boundary.
                    if self.store.messages(sid, pending=True, limit=1) and not current.paused:
                        self.store.update(sid, runnable=True, wake_at=None)
                if current.mode == "heartbeat" and current.outcome == Outcome.ACTIVE:
                    pending_input = bool(self.store.messages(sid, pending=True, limit=1))
                    self.store.update(sid, runnable=pending_input and not current.paused)
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
                    sid, parent, f"Child {sid} failed ({failure.category}): {failure.message}"
                )
        finally:
            with self.store.transaction():
                self.store.charge(sid, Usage(wall_seconds=max(0, now() - started)))
                self.store.update(sid, running_since=None)
                if self.store.session(sid).lifecycle == Lifecycle.RUNNING:
                    self.store.transition(sid, Lifecycle.IDLE)
            if self.store.session(sid).outcome != Outcome.ACTIVE or self._closing:
                await self._close_kernel(sid)
                self.store.transition(sid, Lifecycle.INACTIVE)
                self._candidate_accounting(sid)

    def _candidate_accounting(self, sid):
        row = self.store.db.execute(
            "SELECT body FROM candidates WHERE child_id=?", (sid,)
        ).fetchone()
        if not row:
            return
        session = self.store.session(sid)
        body = json.loads(row[0])
        body.update(
            end_time=session.updated_at if session.outcome != Outcome.ACTIVE else None,
            usage=self.store.usage(sid).model_dump(),
            outcome=session.outcome,
            tools_used=sorted(
                {e["payload"]["name"] for e in self.store.events(sid, kind="tool_call", limit=500)}
            ),
            verifier=[
                e["payload"] for e in self.store.events(sid, kind="verifier_result", limit=1)
            ],
        )
        try:
            from .gitops import GitWorkspace

            patch = GitWorkspace(ToolContext(self, sid, new_id(), "candidate-accounting")).diff()
            body["patch_artifact"] = self.artifacts.put_bytes(sid, patch.encode(), "text/x-diff")
            body["patch_produced"] = bool(patch)
        except (ValueError, OSError) as exc:
            body["patch_error"] = str(exc)
        self.store.db.execute("UPDATE candidates SET body=? WHERE child_id=?", (encode(body), sid))

    async def _prepare(self, sid):
        await self.environment.prepare(sid)

    async def _invoke(self, sid):
        config = self.store.config(sid)
        await self.semantic_compact(sid)
        schemas = self.tools.schemas(config)
        messages, size = self.context.assemble(sid, schemas)
        session = self.store.session(sid)
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
        )
        return await self._model_call(sid, request)

    async def _model_call(self, sid, request, *, persist_turn=True):
        config = self.store.config(sid)
        session = self.store.session(sid)
        size, provider = request.input_token_bound, request.config
        if provider.name == "codex_subscription":
            previous = provider
            provider, _ = await self.providers[provider.name].resolve(provider)
            self.store.pin_provider(sid, previous, provider)
            request = request.model_copy(update={"config": provider})
            self.store.event(
                sid,
                "subscription_model_selected",
                {
                    "model": provider.model,
                    "parameters": provider.parameters,
                    "billing": "subscription",
                    "output_limit_enforcement": "client_observed_bytes; server token cap unavailable",
                },
            )
        for attempt in range(config.retry.attempts):
            while True:
                try:
                    with self.store.transaction():
                        self._check_limits(
                            sid, resource="model_calls", input_bound=size, provider=provider
                        )
                        request_artifact = self.artifacts.put(sid, request.model_dump(mode="json"))
                        eid = self.store.event(
                            sid,
                            "model_invocation_started",
                            {
                                "attempt": attempt + 1,
                                "request_artifact": request_artifact,
                                "turn": session.turns,
                                "provider": provider.name,
                                "model": provider.model,
                                "purpose": request.metadata.get("purpose", "agent"),
                            },
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
                    response = await self.providers[provider.name].invoke(request, emit)
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
                    if persist_turn:
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
                if not failure.retryable or attempt + 1 >= config.retry.attempts:
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
        try:
            external_checkpoint = await self.environment.before_action(context, action.name)
            timeout = (
                config.limits.python_timeout_seconds + 25
                if action.name in ("python", "skill_run")
                else config.limits.tool_timeout_seconds
            )
            async with asyncio.timeout(timeout):
                raw = await self.tools.call(context, action.name, action.arguments)
                encode(raw)  # Structured tools must produce JSON-compatible values.
        except LimitReached:
            raise
        except asyncio.CancelledError:
            self._action_result(
                sid,
                action_id,
                {
                    "error": HarnessError(
                        "runtime",
                        "action_cancelled",
                        "Action cancelled; inspect external effects before retrying",
                        uncertain=True,
                    ).failure.model_dump()
                },
                eid,
            )
            raise
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
            self.environment.after_action(context, external_checkpoint)
        result = self._action_result(sid, action_id, raw, eid)
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
            )
        return self.kernels[sid]

    async def execute_python(self, context, code):
        sid = context.session_id
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
        result = self._capture_kernel_logs(sid, result, eid)
        self.store.event(
            sid,
            "python_error" if result.get("error") else "python_result",
            {"result": result},
            parent=eid,
        )
        return result

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
        config = self.store.config(sid)
        for attempt in range(config.retry.attempts):
            self._check_limits(sid)
            self.store.charge(sid, Usage(verifier_calls=1), parent=parent)
            eid = self.store.event(sid, "verifier_started", {"attempt": attempt + 1}, parent=parent)
            try:
                async with asyncio.timeout(config.limits.tool_timeout_seconds):
                    await self.environment.prepare(sid, force=True)
                    verification = await self.adapters[config.task.adapter].verify(
                        ToolContext(self, sid, new_id(), eid), config.task
                    )
                result = verification.model_dump() if verification else {"skipped": True}
                exposed = self.artifacts.expose(sid, result, source_event=eid)
                result_event = self.store.event(
                    sid,
                    "verifier_result",
                    {"result": exposed, "passed": verification.passed if verification else None},
                    parent=eid,
                )
                self.store.add_context(
                    sid, result_event, [{"role": "user", "content": "Verifier: " + encode(exposed)}]
                )
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
        ids = [sid, *self._active_descendants(sid)] if tree else [sid]
        for target in ids:
            self.store.finish(target, Outcome.CANCELLED, "Stopped by user")
            if task := self.tasks.get(target):
                task.cancel()
        await asyncio.gather(
            *(self.tasks[target] for target in ids if target in self.tasks), return_exceptions=True
        )
        for target in ids:
            await self.unload(target)

    async def _close_kernel(self, sid):
        if kernel := self.kernels.pop(sid, None):
            await kernel.close()

    async def unload(self, sid):
        if sid in self.tasks and not self.tasks[sid].done():
            raise ValueError("Cannot unload a running session")
        await self._close_kernel(sid)
        self.store.transition(sid, Lifecycle.INACTIVE)

    async def pause(self, sid):
        self.store.update(sid, runnable=False, paused=True, wake_at=None)
        if task := self.tasks.get(sid):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.unload(sid)
        self.store.event(sid, "paused", {})

    async def fork(self, sid, *, name=None):
        source = self.store.session(sid)
        if sid in self.tasks:
            raise ValueError("Pause the source session before forking it")
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
                selected_state=[],
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
            # Local adaptive entries get new identities with explicit version provenance.
            selected = []
            from .models import StateEdit

            for entry in self.store.states(sid):
                if entry["owner_id"] is None:
                    if entry["id"] in source.selected_state:
                        selected.append(entry["id"])
                    continue
                edit = StateEdit(
                    kind=entry["kind"],
                    title=entry["title"],
                    content=entry["content"],
                    source_events=[eid],
                    intended_effect=f"Fork of {entry['id']} version {entry['version']}",
                )
                copied = self.store._apply_edit(branch.id, edit, eid)
                if entry["id"] in source.selected_state:
                    selected.append(copied)
            self.store.update(branch.id, selected_state=selected)
            source_dir = self.store.directory / "kernels" / source.kernel_id
            target_dir = self.store.directory / "kernels" / branch.kernel_id
            if source_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = source_dir / "checkpoint.json"
                if checkpoint.exists():
                    if workspace.path != source.workspace.path:
                        fork_checkpoint(
                            checkpoint,
                            target_dir / "checkpoint.json",
                            source.workspace.path,
                            workspace.path,
                        )
                    else:
                        shutil.copy2(checkpoint, target_dir / "checkpoint.json")
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
        self._closing = True
        self._wake.set()
        if self._scheduler_task:
            await self._scheduler_task
            self._scheduler_task = None
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sid in list(self.kernels):
            await self._close_kernel(sid)
        for session in self.store.sessions():
            if session.lifecycle != Lifecycle.INACTIVE:
                self.store.transition(session.id, Lifecycle.INACTIVE)
        self.store.close()
        self._owner_lock.close()
