"""Environment policies and capabilities, separate from session strategy.

The default environment is an ordinary workspace. Coding admission, checkpoints,
candidate isolation and completion gates are opt-in environment policy, not stages
of the agent loop. Third-party TaskAdapters retain their prepare/verify interface.
"""

from __future__ import annotations

import asyncio

from .coding import CodingTask
from .models import HarnessError, Workspace, new_id
from .mutations import MutationObserver
from .storage import encode
from .tasks import WorkspaceTask
from .tools import ToolContext


class Environment:
    def __init__(self, runtime, adapters=None):
        self.runtime = runtime
        self.mutations = MutationObserver(runtime)
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
        if config.task.adapter != "coding":
            return None
        tool = context.capability or runtime.tools.entries[name]
        if name in {"finish", "rlm", "rlm.run", "agent_spawn"} or set(tool.permissions) & {
            "workspace.write",
            "python",
            "ipython",
            "process",
        }:
            await self.prepare(sid, force=True)
        if "process" in tool.permissions:
            # A preceding Path.write_text in the SAME cell must be visible to tests,
            # including same-size writes with timestamp-based Python bytecode.
            self.mutations.reconcile(context, reason="before_process")
        # The executable capability, not its origin/envelope, determines policy.
        # Keep explicit legacy snapshot behavior; Python-first cells use incremental
        # metadata observation rather than copying all content for every invocation.
        token = {"window": None, "checkpoint": None}
        if name not in {"python", "ipython", "skill_run"}:
            token["window"] = self.mutations.begin(context)
        if config.control_plane != "direct" or name not in {
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
            return token
        from .gitops import GitWorkspace

        checkpoint = GitWorkspace(context).snapshot("before-external-action")
        runtime.store.event(
            sid,
            "workspace_observation_started",
            {"action_id": context.action_id, "checkpoint_id": checkpoint},
            parent=context.source_event,
        )
        token["checkpoint"] = checkpoint
        return token

    def after_action(self, context, token):
        if not token:
            return
        from .gitops import GitWorkspace

        try:
            self.mutations.end(context, token["window"])
            if token["checkpoint"]:
                GitWorkspace(context).observe_effects(token["checkpoint"], context.action_id)
        except Exception as exc:
            self.mutations.failed(context, exc)
            raise HarnessError(
                "environment", "observation_failed", str(exc), uncertain=True
            ) from exc

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
        checkpoint = git.snapshot_tree("candidate-source" if child else "fork-source")
        isolated = git.isolate(checkpoint)
        workspace = Workspace(
            path=str(isolated),
            metadata={
                "source_session": source.id,
                "source_checkpoint": checkpoint,
                "isolation": "git_worktree",
                "base_revision": git.checkpoint(checkpoint)["head"],
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
        self.mutations.recover()
        self.runtime.store.db.execute(
            "UPDATE experiments SET status='interrupted' WHERE status='running'"
        )
