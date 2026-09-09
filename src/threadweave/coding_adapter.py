"""Coding specialization: capabilities, workspace policy, evidence and child profiles."""

from __future__ import annotations

import json
from pathlib import Path

from .capabilities import CapabilityProvider, ChildProfile
from .coding import CodingTask
from .coding_config import coding_options, update_coding_options
from .models import HarnessError, Outcome, Workspace, new_id, now
from .mutations import MutationObserver
from .repository import RepositoryIndex
from .storage import encode
from .tools import ToolContext


def investigation(config):
    config.task.verifier = "none"
    config.task.require_verifier = False
    update_coding_options(
        config.task,
        capture_baseline=False,
        require_tests=False,
        require_clean_baseline=False,
        require_change=False,
    )


class CodingAdapter(CodingTask):
    capabilities = ("coding",)
    profiles = {
        "shared": ChildProfile(isolate=False),
        "research": ChildProfile(isolate=False, read_only=True, configure=investigation),
        "review": ChildProfile(isolate=False, read_only=True, configure=investigation),
        **{
            name: ChildProfile(isolate=True, require_isolation=True)
            for name in ("candidate", "test", "performance")
        },
    }

    def bind(self, runtime):
        self.runtime = runtime
        self.mutations = MutationObserver(runtime)
        self._repository_indexes = {}

    def providers(self):
        from .coding_tools import register

        return [
            CapabilityProvider(
                "coding", register, "threadweave.coding_api:bindings", coding_instructions, self
            )
        ]

    def validate(self, config):
        coding_options(config.task)

    def configure(self, config):
        config.task.verifier = "coding"
        config.task.require_verifier = True
        config.task.verify_each_turn = False

    def evidence_signals(self):
        return ("code_edit", "coding_command", "experiment_conclusion")

    def defer_prepare(self, session):
        return session.mode == "interactive"

    def default_isolation(self, config, *, child=False):
        return not child or config.control_plane == "direct"

    def delegation(self, parent, config):
        store = self.runtime.store
        focus = store.events(parent.id, kind="working_focus", limit=1)
        return {
            "focus": focus[0]["payload"] if focus else None,
            "test_commands": coding_options(config.task).test_commands,
            "result_contract": "Report source locations, findings, executed test evidence, uncertainty; candidate patches require explicit parent acceptance.",
        }

    def write(self, context, path, content):
        from .editing import Editor

        return Editor(context).apply({path: content})

    async def before_python(self, context):
        await self.runtime.environment.prepare(context.session_id, force=True)
        return self.mutations.begin(context)

    def after_python(self, context, token):
        if token:
            try:
                self.mutations.end(context, token)
            except Exception as exc:
                self.mutations.failed(context, exc)
                raise HarnessError(
                    "environment", "observation_failed", str(exc), uncertain=True
                ) from exc

    def validate_path(self, root, raw, resolved):
        if ".git" in raw.relative_to(root).parts or ".git" in resolved.relative_to(root).parts:
            raise PermissionError("Direct .git access is prohibited; use Git tools")

    def process_completed(self, context):
        try:
            self.mutations.reconcile(context, reason="background_process_exit")
        except Exception as exc:
            self.mutations.failed(context, exc)

    def poll(self):
        self.mutations.poll()

    def runtime_close(self):
        self.mutations.close()
        for index in self._repository_indexes.values():
            index.close()

    def workspace_admitted(self, config, workspace, *, child=False):
        update_coding_options(config.task, repository=workspace.path, base_commit=None)
        if child:
            update_coding_options(config.task, require_change=False)

    def child_admitted(self, parent, session, checkpoint):
        if checkpoint:
            self.runtime.store.event(
                session.id,
                "candidate_ready",
                {"workspace": session.workspace.model_dump(), "checkpoint": checkpoint},
            )
            self.runtime.store.db.execute(
                "INSERT INTO candidates VALUES(?,?,?,?)",
                (
                    session.id,
                    parent.id,
                    checkpoint,
                    encode(
                        {
                            "instruction": session.instruction,
                            "start_time": session.created_at,
                            "consumed": False,
                            "accepted": False,
                        }
                    ),
                ),
            )

    def context_messages(self, sid):
        from .coding_retrieval import coding_focus

        messages = []
        for instruction in (
            self.runtime.store.session(sid)
            .adapter_context.get("coding", {})
            .get("repository_instructions", [])
        ):
            messages.append(
                {
                    "role": "user",
                    "content": f"Repository instructions from {instruction['path']} (scope {instruction['scope']}):\n"
                    + instruction["content"],
                }
            )
        focus = coding_focus(self.runtime.store, sid, index_provider=self.index)
        if focus:
            messages.append({"role": "user", "content": "Current adapter evidence: " + focus})
        return messages

    def inherit_shared_admission(self, parent, child):
        store = self.runtime.store
        if (
            parent.workspace.path != child.workspace.path
            or store.config(child.id).task.adapter != "coding"
        ):
            return
        row = store.db.execute(
            "SELECT body FROM coding_baselines WHERE session_id=?", (parent.id,)
        ).fetchone()
        if row:
            # A shared child observes the SAME workspace changes against the
            # original verified baseline; it must not establish a new dirty baseline.
            with store.transaction():
                store.db.execute("INSERT INTO coding_baselines VALUES(?,?)", (child.id, row[0]))
                store.event(
                    child.id,
                    "coding_baseline_inherited",
                    {"parent_id": parent.id, "workspace": child.workspace.path},
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
            await self.runtime.environment.prepare(sid, force=True)
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

    def continuation_workspace(self, source, config, *, child=False, isolate=None):
        """Explicit coding environments isolate writable continuations; others share metadata."""
        if isolate is None:
            isolate = config.task.adapter == "coding"
        if not isolate:
            return source.workspace.model_copy(deep=True), None
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
        update_coding_options(config.task, repository=str(isolated))
        update_coding_options(config.task, base_commit=None)
        if child:
            update_coding_options(config.task, require_change=False)
        return workspace, checkpoint

    def isolated_child_workspace(self, parent_id, child_id):
        """Thread-local connection around immutable Git capture and worktree admission."""
        from types import SimpleNamespace

        from .artifacts import Artifacts
        from .gitops import GitWorkspace
        from .storage import Store

        store = Store(self.runtime.store.directory)
        try:
            runtime = SimpleNamespace(store=store, artifacts=Artifacts(store))
            event = store.event(child_id, "candidate_workspace_started", {"parent": parent_id})
            workspace = GitWorkspace(ToolContext(runtime, parent_id, new_id(), event))
            checkpoint = workspace.snapshot_tree("candidate-source")
            isolated = workspace.isolate(checkpoint)
            result = Workspace(
                path=str(isolated),
                metadata={
                    "source_session": parent_id,
                    "source_checkpoint": checkpoint,
                    "isolation": "git_worktree",
                    "base_revision": workspace.checkpoint(checkpoint)["head"],
                },
            )
            # Persist the lease even if the daemon exits before admission completes.
            store.event(
                child_id,
                "candidate_workspace_lease",
                {"workspace": result.model_dump(), "checkpoint": checkpoint},
            )
            return result, checkpoint
        finally:
            store.close()

    def completed(self, sid):
        row = self.runtime.store.db.execute(
            "SELECT body FROM candidates WHERE child_id=?", (sid,)
        ).fetchone()
        if not row:
            return
        session = self.runtime.store.session(sid)
        body = json.loads(row[0])
        body.update(
            end_time=session.updated_at if session.outcome != Outcome.ACTIVE else None,
            usage=self.runtime.store.usage(sid).model_dump(),
            outcome=session.outcome,
            tools_used=sorted(
                {
                    e["payload"]["name"]
                    for e in self.runtime.store.events(sid, kind="tool_call", limit=500)
                }
            ),
            verifier=[
                e["payload"]
                for e in self.runtime.store.events(sid, kind="verifier_result", limit=1)
            ],
        )
        try:
            from .gitops import GitWorkspace

            patch = GitWorkspace(
                ToolContext(self.runtime, sid, new_id(), "candidate-accounting")
            ).diff()
            body["patch_artifact"] = self.runtime.artifacts.put_bytes(
                sid, patch.encode(), "text/x-diff"
            )
            body["patch_produced"] = bool(patch)
        except (ValueError, OSError) as exc:
            body["patch_error"] = str(exc)
        self.runtime.store.db.execute(
            "UPDATE candidates SET body=? WHERE child_id=?", (encode(body), sid)
        )

    def admitted(self, sid):
        from .repository_instructions import discover_instructions

        files = discover_instructions(Path(self.runtime.store.session(sid).workspace.path))
        self.runtime.store.update(
            sid,
            adapter_context={
                **self.runtime.store.session(sid).adapter_context,
                "coding": {"repository_instructions": files},
            },
        )
        if files:
            self.runtime.store.event(
                sid,
                "repository_instructions_loaded",
                {"files": [{k: v for k, v in item.items() if k != "content"} for item in files]},
            )

    def index(self, sid):
        config = self.runtime.store.config(sid)
        if not hasattr(self, "_repository_indexes"):
            self._repository_indexes = {}
        key = (sid, self.runtime.store.session(sid).workspace.path)
        if key not in self._repository_indexes:
            self._repository_indexes[key] = RepositoryIndex(
                self.runtime.store,
                self.runtime.store.session(sid).workspace.path,
                enhanced=config.features.enhanced_code_index,
                allowed=config.task.allowed_paths,
                forbidden=config.task.forbidden_paths,
            )
        return self._repository_indexes[key]

    def retain_failure(self, sid, event, verification):
        if self.runtime.store.config(sid).task.adapter != "coding":
            return
        details = verification.details or {}
        edits = self.runtime.store.events(sid, kind="code_edit", limit=5)
        body = {
            "task_pattern": self.runtime.store.session(sid).instruction[:1000],
            "attempted_strategy": "Recent actions: "
            + encode(
                [
                    e["payload"].get("name")
                    for e in self.runtime.store.events(sid, kind="tool_call", limit=5)
                ]
            ),
            "evidence": [event, *[e["id"] for e in edits]],
            "failure_reason": details.get("violations", ["Independent verifier failed"]),
            "affected_files": sorted({p for e in edits for p in e["payload"].get("files", {})}),
            "verifier_output": details,
            "recommendation": "Retrieve this evidence before repeating the same approach.",
        }
        identifier = new_id()
        self.runtime.store.db.execute(
            "INSERT INTO failure_memories VALUES(?,?,?,?)", (identifier, sid, now(), encode(body))
        )
        self.runtime.store.event(
            sid, "failure_memory", {"memory_id": identifier, **body}, parent=event
        )

    async def recover(self):
        from .editing import recover_edits
        from .gitops import recover_workspace_effects

        recover_edits(self.runtime)
        await recover_workspace_effects(self.runtime)
        self.mutations.recover()
        for session in self.runtime.store.sessions():
            if (
                session.workspace.metadata.get("admission_pending")
                and self.runtime.store.config(session.id).task.adapter == "coding"
            ):
                from .models import Outcome

                self.runtime.store.update(
                    session.id, outcome=Outcome.FAILED, paused=True, runnable=False
                )
                self.runtime.store.event(
                    session.id,
                    "candidate_recovery",
                    {
                        "status": "admission_interrupted",
                        "workspace_leases_retained": True,
                        "replayed": False,
                    },
                )
        self.runtime.store.db.execute(
            "UPDATE experiments SET status='interrupted' WHERE status='running'"
        )


def coding_instructions(config):
    text = ""
    if config.features.enhanced_code_index:
        text += """Coding APIs: repo, tests, context, verify, git, edit. Their help() gives short signatures.
Prefer repo.context_for_symbol(name) for precise evidence; tests.related_to(files=[...]) explains
test selection. context.focus(files=[...],hypothesis='...') retains your investigation.
await edit(path,old_str,new_str) edits a unique match. Raw Python remains available.
"""
    text += "Coding decision support: reuse observed failures and exact source evidence. State a hypothesis when debugging; prefer focused definitions/callers and related failing tests over repeatedly reading full files. Use repo.help()/tests.help() to discover APIs. Make small evidence-supported changes; escalate tests when warranted. Delegation must have a distinct purpose and a bounded evidence request. Repeated unchanged reads/searches/tests are a signal to revise the hypothesis, not proof of progress. These are cost-aware guidelines, not a mandatory workflow; the independent final verifier still decides correctness.\n"
    if config.features.subagents:
        text += "Coding child profiles: use purpose='research' or 'review' for independent investigation; these are read-only and finish on findings without waiting for the parent repository's full test suite to pass. The default shared profile inherits the parent's completion gates and suits implementation work. candidate/test/performance use isolated worktrees. agents.candidate(handle, accept=False) inspects; accept=True explicitly applies.\n"
    if config.features.experiments:
        text += "experiment.help() lists durable experiment/measurement procedures.\n"
    return text
