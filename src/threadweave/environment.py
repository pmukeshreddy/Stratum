"""Environment policies and capabilities, separate from session strategy.

The default environment is an ordinary workspace. Coding admission, checkpoints,
candidate isolation and completion gates are opt-in environment policy, not stages
of the agent loop. Third-party TaskAdapters retain their prepare/verify interface.
"""

from __future__ import annotations

import asyncio

from .coding import CodingTask
from .models import HarnessError, Workspace, new_id
from .storage import encode
from .tasks import WorkspaceTask
from .tools import ToolContext


class Environment:
    def __init__(self, runtime, adapters=None):
        self.runtime = runtime
        self.adapters = (
            adapters
            if adapters is not None
            else {"workspace": WorkspaceTask(), "coding": CodingTask()}
        )

    def configure(self, config):
        if config.task.adapter == "coding":
            config.task.verifier = "coding"
            config.task.require_verifier = True
            config.task.verify_each_turn = False

    async def prepare(self, sid, *, force=False):
        runtime, store = self.runtime, self.runtime.store
        config, session = store.config(sid), store.session(sid)
        if store.events(sid, kind="environment_prepared", limit=1):
            return
        # Even an explicitly coding-configured interactive conversation may greet,
        # ask questions and inspect files without running a test/build baseline.
        if config.task.adapter == "coding" and session.mode == "interactive" and not force:
            return
        event = store.event(sid, "environment_prepare", {"adapter": config.task.adapter})
        try:
            async with asyncio.timeout(config.limits.tool_timeout_seconds):
                result = await self.adapters[config.task.adapter].prepare(
                    ToolContext(runtime, sid, new_id(), event), config.task
                )
        except Exception as exc:
            raise HarnessError("environment", "prepare_failed", str(exc)) from exc
        eid = store.event(sid, "environment_prepared", {"result": result}, parent=event)
        selected = runtime.artifacts.expose(sid, result, source_event=eid)
        store.add_context(
            sid, eid, [{"role": "user", "content": "Environment: " + encode(selected)}]
        )

    async def before_action(self, context, name):
        runtime, sid = self.runtime, context.session_id
        config = runtime.store.config(sid)
        if config.task.adapter != "coding" or not runtime.tools.allowed(name, config):
            return None
        tool = runtime.tools.entries[name]
        if name in {"finish", "rlm", "agent_spawn"} or set(tool.permissions) & {
            "workspace.write",
            "python",
            "ipython",
            "process",
        }:
            await self.prepare(sid, force=True)
        if context.from_python or name not in {
            "python",
            "skill_run",
            "process_run",
            "run_tests",
            "run_targeted_tests",
            "run_build",
            "run_lint",
            "run_typecheck",
            "run_benchmark",
            "run_profile",
            "experiment_run",
        }:
            return None
        from .gitops import GitWorkspace

        checkpoint = GitWorkspace(context).snapshot("before-external-action")
        runtime.store.event(
            sid,
            "workspace_observation_started",
            {"action_id": context.action_id, "checkpoint_id": checkpoint},
            parent=context.source_event,
        )
        return checkpoint

    def after_action(self, context, checkpoint):
        if not checkpoint:
            return
        from .gitops import GitWorkspace

        try:
            GitWorkspace(context).observe_effects(checkpoint, context.action_id)
        except (ValueError, OSError) as exc:
            self.runtime.store.event(
                context.session_id,
                "workspace_observation_failed",
                {"action_id": context.action_id, "reason": str(exc)},
                parent=context.source_event,
            )
            self.runtime.store.update(context.session_id, paused=True, runnable=False)

    def continuation_workspace(self, source, config, *, child=False):
        """Explicit coding environments isolate writable continuations; others share metadata."""
        if config.task.adapter != "coding":
            if child:
                config.task.verifier = "none"
                config.task.require_verifier = False
                config.task.verifier_options = {}
            return source.workspace, None
        from .gitops import GitWorkspace

        event = self.runtime.store.event(
            source.id, "candidate_preparing" if child else "fork_workspace", {}
        )
        git = GitWorkspace(ToolContext(self.runtime, source.id, new_id(), event))
        checkpoint = git.snapshot("candidate-source" if child else "fork-source")
        isolated = git.isolate(checkpoint)
        workspace = Workspace(
            path=str(isolated),
            metadata={
                "source_session": source.id,
                "source_checkpoint": checkpoint,
                "isolation": "private_git_copy",
            },
        )
        config.task.repository = str(isolated)
        config.task.base_commit = None
        if child:
            config.task.require_change = False
            config.task.verifier_options = {}
        return workspace, checkpoint

    async def recover(self):
        from .editing import recover_edits
        from .execution import recover_containers
        from .gitops import recover_workspace_effects

        recover_edits(self.runtime)
        await recover_containers(self.runtime)
        await recover_workspace_effects(self.runtime)
        self.runtime.store.db.execute(
            "UPDATE experiments SET status='interrupted' WHERE status='running'"
        )
