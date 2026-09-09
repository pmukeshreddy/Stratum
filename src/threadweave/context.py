from __future__ import annotations

import json

from .models import HarnessError, now
from .storage import Store, encode
from .tokenization import estimate, method

# Foundational instructions are code-owned. Adaptive entries never replace this message.
FOUNDATION = """You operate a persistent agent session. Your normal control environment is persistent IPython.
Work through Python to inspect, search, transform, test, persist, verify and delegate.
Use observations to decide your next action and to support your final answer.
When enabled, python executes in your own Python worker with top-level await. Variables are retained
only when the session's persistent_repl feature is enabled; inspect the capability metadata.
Python helpers: await rlm(instruction, name=None), tools.call(name, **arguments), await tools.acall(name, **arguments),
workspace (Path), forget(*names), remember_recipe(name, reconstruction_code).
Tool calls in Python return full structured results; keep large results in variables.
Use artifacts and history retrieval to inspect omitted details. Context is a bounded cache.
Only serializable values and explicitly registered recovery recipes survive a worker restart.
Treat recovery warnings about missing values or uncertain side effects as evidence: inspect before
repeating work. Do not assume interrupted external actions were rolled back.
Child agents are persistent concurrent sessions. rlm (also agent_spawn) returns a stable handle immediately; communicate with
stable IDs. agent_wait yields your scheduling slot. A failed child does not decide your outcome.
finish requests completion, subject to task gates. A text reply alone does not complete the task.
Persistent state is supplemental task material, never a replacement for foundational instructions.
Tool permissions are capabilities. Python/process access executes trusted code with this OS user's
authority; it is not an isolation sandbox. Respect the workspace and the user's instructions.
L1 is selected active context. L2 is your persistent REPL and concurrent child handles: values do
not become model context unless explicitly printed, returned, summarized or retrieved. L3 is the
disk-backed history, artifacts, snapshots, messages and versioned reusable state. L1 compaction
checkpoints L2; useful large values become recoverable artifact handles. Ordinary conversation
can end directly when there is no inspectable or mechanically checkable work. Never invent observations or measurements.
"""


def token_bound(value, model=None) -> int:
    """Conservative UTF-8 byte bound plus framing; provider-reported usage remains authoritative."""
    return estimate(value, model)


def python_instructions(config, *, child=False, specifications=()):
    """Expose only usable capabilities; compact help remains in the live namespace."""
    text = """You are Buffalo, a code-using agent that solves tasks through persistent IPython: inspect,
decompose, execute, observe, validate and iterate.
Your persistent IPython environment is your primary, long-lived working environment for computation,
tool use, task state and recursive orchestration. Work through the sole tool ipython(code).
IPython is the default working and control environment, not an optional calculator.
Enter it whenever work can be inspected, transformed, tested, searched, delegated, persisted or
verified. This includes coding and repository tasks, conflicting instructions, long context,
multi-step reasoning, stateful work, numerical work, structured extraction, search/filtering,
uncertain intermediate conclusions, decomposition and mechanically checkable answers.
For conflicting instructions, retain and compare the role-bearing contract in Python; instruction
priority still governs. For reasoning tasks, externalize hypotheses and check intermediate claims.
Choose substantive cells that advance the task; never execute ceremonial code merely to use a tool.
A direct final answer is the exception, appropriate for a simple conversational reply with no
inspectable work. Finish substantive tasks from observed evidence and retained working state.
Retain intermediate variables, functions, candidate artifacts and child handles across turns.
The original user instruction is available as task.instructions. task.messages contains the complete
original message contract in order; respect its system, developer and user roles. task.assignment
is your own assignment, task.context contains task metadata, and task.workspace is a Path.
All execution surfaces are preloaded: tools, harness, skills, rlm, agent_message, agent_observe,
workspace, bash, compact and refine. Top-level await works. rlm() is recursive computation inside Python.
Delegate useful independent work through await rlm(...); admission immediately returns a stable child
handle. The persistent child runs asynchronously in its own IPython environment and active model
context. Continue complementary computation while it runs; receive evidence through messaging and
observation, evaluate it and synthesize. A child uses the same execution model and inherited capabilities.
Compaction reduces model context without destroying persistent computational state, task data,
child handles, versioned harness state or retrievable history. Print or retrieve relevant evidence
for the next model turn; retaining a Python value alone does not expose it to the model.
"""
    if not config.features.subagents:
        begin = text.index("Delegate useful independent work")
        end = text.index("Compaction reduces", begin)
        text = text[:begin] + "Recursive delegation is unavailable in this session.\n" + text[end:]
    text += """\nPython API guide: Path, pathlib, os, asyncio, json, workspace (Path), session (metadata),
task (original task and current assignment), context['task'] (current complete assignment).
Use Path.read_text()/write_text(), ordinary Python, or await bash(command) for execution.
await bash returns exit_code, stdout, stderr, duration and artifact IDs. bash(command) without
await returns a background handle with poll(), tail(), kill(). Use the workspace environment.
Never claim execution results you did not observe. A normal final text reply requests completion;
the independently configured verifier remains authoritative and may return failure evidence.
L1 is selected context, L2 is live Python state, L3 is durable history/artifacts/snapshots.
Compaction preserves the kernel while offloading large values and retiring stale values with
recovery metadata. Inspect repl_state.describe(); repl_state.rehydrate(name) restores a value.
Use repl_state.retain(name) for important live state and remember_recipe(name, code, dependencies=[...])
for reconstructible state. Offloaded handles have load(); assign the returned value before using it. On recovery inspect warnings; only checkpointed values/recipes restore.
Interrupted actions may have partial effects; inspect them before retrying. Python and shell run
as trusted local code, not a general sandbox. Stay within the supplied workspace and user scope.
Respect all resource limits. Do not fetch hidden/reference solutions. Never modify foundational
instructions through supplemental state. Disabled capabilities must not be invoked.
"""
    if "process" not in config.permissions or (
        config.execution.read_only and config.execution.backend == "local"
    ):
        text = text.replace(
            "Use Path.read_text()/write_text(), ordinary Python, or await bash(command) for execution.\nawait bash returns exit_code, stdout, stderr, duration and artifact IDs. bash(command) without\nawait returns a background handle with poll(), tail(), kill(). Use the workspace environment.",
            "Use Path.read_text() and ordinary Python for inspection. Host shell execution is not\navailable with this session's permissions/backend. Do not invoke bash.start to bypass that policy.",
        )
    if config.execution.read_only:
        text += "This session is OS read-only outside its private kernel state. Do not attempt workspace writes.\n"
    if config.features.subagents:
        text += """\nRecursive mechanics:\nRLM is recursive orchestration through this persistent REPL. When delegation is useful,
give independent, self-contained work to children and continue your own useful work in parallel.
handle = await rlm('assignment', name='worker') returns at admission,
not completion; it returns a stable session_id, name, session_dir and model, never the child's answer.
Put the work description in the prompt. Optional purpose selects a registered child workspace/capability
profile, not a prose description; omit it for the default shared profile. Inspect agents.help() for profiles.
Children inherit the effective model, reasoning, capabilities and limits; they can recursively delegate
within the shared task budget and depth limit. Each child has its own persistent Python namespace.
Completion normally waits for admitted children so their evidence can be incorporated.
Cancel children whose work is no longer relevant. Keep handles in variables. await agents.followup(handle, assignment) reuses a child and its kernel
after completion. await agents.delete_subagent(handle) cancels it. Receive results through messages or inspect committed child trajectories:
await agent_message.send('findings', receiver_role='parent') in a child;
await agent_message.send('follow-up', receiver_role='child', receiver_name='worker') in the parent.
await agent_message.receive(); await agent_observe.recent_messages(handle.session_id).
await agents.wait(seconds=10) requests a pause before the next model turn and returns a receipt
immediately. End the cell afterward; the scheduler wakes you on a message or timeout.
agents.help(), agent_message.help() and agent_observe.help() describe the live APIs.
Choose complementary, substantive child assignments; avoid duplicating work without a verification purpose.
"""
        text += """\nIndependent work:
For a candidate solution with distinct specification risks, a child can investigate edge cases while
you develop and test the candidate. Evaluate its evidence before incorporating it.
Delegate independent investigation, repository exploration, long-context analysis, isolated
implementation experiments, alternative hypotheses, review, testing, performance work or research
branches when their evidence can advance your task. For separable investigations, start independent
children without waiting for each to finish:
left = await rlm('Investigate the first component against its specification.', name='left')
right = await rlm('Investigate the second component against its specification.', name='right')
Then continue complementary local work. Use these patterns only when they add useful evidence;
a small local task needs no child. Optional requirement= describes the task requirement being investigated.
"""
        text += "\nAvailable subagent specifications:\n" + encode(list(specifications)) + "\n"
        text += "Inspect full instructions with harness.get('subagent_spec', id); await rlm(assignment, spec_id=id) applies the retrieved specification through the canonical runtime.\n"
    if config.features.history_retrieval:
        text += "history.search(query), history.get(event_id), context.search(query), artifacts.load(id) retrieve retained evidence.\n"
    if config.tool_allowlist is None:
        text += """\nHarness state:\nHarness state contains versioned memory, prompt_note, skill and subagent_spec entries.
harness.list() retrieves the state overview; harness.get(kind, id) retrieves full content/version.
harness.create(kind, title, content, select=True) and harness.update(kind, id, title, content)
save task-local supplemental state; harness.select([id]) selects it for subsequent model turns.
Harness CRUD/list/select are synchronous host operations: entry = harness.get(...), without await.
They return records (list returns a list); invalid arguments or denied operations raise an error.
Missing entries raise KeyError. skills.list() is immediate; skills.load(name) synchronously loads
a module or entry, while await skills.run(...) executes it. Long-running operations are async.
Memory stores durable facts, prompt notes store narrow behavioral guidance, skills store validated
executable procedures, and subagent specifications store reusable delegation instructions.
rlm.harness is the same state API. State supplements the task and never overrides its instructions.
Create reusable checks only when you will run them on distinct candidates or stages; give them explicit inputs.
Retain task-local facts and operating constraints when subsequent work will use them; do not invent entries.
Skills and project context:
skills.list() discovers procedures; skills.load(name) loads their module or state entry;
await skills.run(name, **inputs) executes a validated skill. Read the matching SKILL.md or
entry instructions before executing it. tools.catalog() returns tool schemas
into Python, not active model context. mcp exposes configured external servers.
Continual harness improvement is part of normal execution. Capture reusable repository discoveries,
successful fixes, effective searches/commands, environment quirks, debugging strategies and child
specializations as memory, prompt_note, executable skill or subagent_spec. The runtime filters
duplicate/trivial evidence before reviewing successful progress, failures, child findings and
compaction at safe checkpoints; validated versions activate for later turns. The reviewer may decline; no edit is required.
You do not need to call refine() for automatic review. await refine() optionally requests manual
evidence-based planning at a safe boundary. Applied edits are retrieved into later turns of this task.
Validate useful lessons on subsequent work. await compact() compacts active context while retaining
recoverable Python state, artifacts and durable history. Retrieve omitted details explicitly as needed.
Use meaningful mechanisms within the task's resource budget; never manufacture activity.
"""
    text += """\nAction patterns:
Begin substantive work in IPython by inspecting the actual inputs and recording a useful next step.
Keep hypotheses, requirements, intermediate results and evidence in named variables.
context.track(text, kind="requirement", id=None) durably records unresolved work; kinds also include
hypothesis, blocker, decision and failed_approach. context.resolve(id, evidence_events=[...])
requires observed supporting events. Omitted or old unresolved work remains active through compaction. Use focused
checks during development; reserve full configured verification for completion or an explicit request.
After a failed check or a new finding, inspect the actual evidence and choose the next useful action:
another local check, a correction, more context, or independent investigation. Failure alone does not
require delegation. Your next action follows from the task and observed trajectory.
"""
    if child:
        text += "You are a persistent child. Complete your delegated assignment within the original task contract; send useful partial findings to your parent when they can inform ongoing work.\n"
    return text


class Context:
    def __init__(self, store: Store, environment=None):
        self.store = store
        self.environment = environment
        self._export_cursors = {}
        self.state_presentations = {}

    def original_task(self, sid):
        """The immutable role-bearing task contract, separate from accumulated work."""
        session = self.store.session(sid)
        root = self.store.session(session.root_id)
        task = self.store.config(root.id).task
        messages = task.original_messages or [
            *task.instruction_messages,
            {"role": "user", "content": root.instruction},
        ]
        return {
            "instructions": next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                root.instruction,
            ),
            "messages": messages,
            "current_assignment": session.instruction,
            "context": {
                "root_id": root.id,
                "adapter": task.adapter,
                "specification": task.specification,
            },
        }

    def execution_inputs(self, sid, messages):
        """Record source references actually retained in this invocation's context."""
        children = {s.id for s in self.store.sessions() if s.parent_id == sid}
        visible = encode(messages)
        evidence = {}
        for row in self.store.db.execute(
            "SELECT sender_id,source_event FROM messages WHERE recipient_id=? AND received_at IS NOT NULL",
            (sid,),
        ):
            if row["sender_id"] in children and row["source_event"] in visible:
                evidence.setdefault(row["sender_id"], []).append(row["source_event"])
        for event in self.store.events(sid, kind="child_observation", limit=100):
            data = event["payload"]
            if data["child_id"] in children:
                sources = [source for source in data["source_events"] if source in visible]
                if sources:
                    evidence.setdefault(data["child_id"], []).extend(sources)
        return {
            "child_evidence": {k: list(dict.fromkeys(v)) for k, v in evidence.items()},
            "harness_state": self.state_presentations.get(sid, []),
        }

    def history_file(self, sid):
        """Materialized, readable conversation/event log; SQLite remains authoritative.

        Append only newly committed events between turns. A runtime restart rebuilds
        this disposable projection, never the original trajectory.
        """
        from .artifacts import atomic_write

        directory = self.store.directory / "session_logs" / sid
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / "messages.jsonl"
        if sid not in self._export_cursors:
            atomic_write(path, b"")
            self._export_cursors[sid] = 0
        rows = self.store.db.execute(
            "SELECT seq,payload,type,id,timestamp FROM events WHERE session_id=? AND seq>? ORDER BY seq",
            (sid, self._export_cursors[sid]),
        )
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(encode({**dict(row), "payload": json.loads(row["payload"])}) + "\n")
                self._export_cursors[sid] = row["seq"]
        return path

    def supplemental(self, sid: str) -> str:
        session, config = self.store.session(sid), self.store.config(sid)
        from .context_budget import policy_tokens, state_excerpt
        from .state_retrieval import relevant_state

        model = config.provider.model
        selected, automatic = [], []
        for entry_id in dict.fromkeys(session.selected_state):
            try:
                entry = self.store.state(sid, entry_id)
            except KeyError:
                continue
            if not entry["deleted"]:
                selected.append(entry)
        selected_ids = {e["id"] for e in selected}
        automatic = [e for e in relevant_state(self.store, sid) if e["id"] not in selected_ids]
        budget = policy_tokens(config.context, "supplemental", model)
        # Explicit selections receive equal minimum semantic allocations before incidental state.
        minimum = 320 if model.startswith(("gpt-", "o1", "o3", "o4", "codex")) else 1100
        budget = max(budget, len(selected) * minimum)
        ceiling = config.context.max_tokens - config.provider.max_output_tokens
        if budget > ceiling and selected:
            raise HarnessError(
                "runtime",
                "selected_state_capacity",
                "Selected state needs more context; no selected body was silently omitted",
            )
        records, used, remaining = [], [], min(budget, ceiling)
        for index, entry in enumerate([*selected, *automatic]):
            explicit = index < len(selected)
            allocation = remaining // (len(selected) - index) if explicit else remaining
            record = state_excerpt(entry, allocation, model, explicit=explicit)
            if record is None:
                if explicit:
                    raise HarnessError(
                        "runtime",
                        "selected_state_capacity",
                        "Selected state metadata leaves no room for useful content",
                    )
                continue
            records.append(record)
            remaining -= token_bound(record, model)
            used.append({"id": entry["id"], "version": entry["version"]})
        entries = [encode(record) for record in records]
        self.state_presentations[sid] = [
            {**entry, "kind": self.store.state(sid, entry["id"], entry["version"])["kind"]}
            for entry in used
        ]
        previous = self.store.events(sid, kind="state_retrieved", limit=1)
        if used and (not previous or previous[0]["payload"]["entries"] != used):
            self.store.event(
                sid, "state_retrieved", {"entries": used, "selection": "explicit or task relevance"}
            )
        return "\n".join(entries)

    def messages(self, sid: str) -> list[dict]:
        session = self.store.session(sid)
        cap = min(6000, self.store.config(sid).context.max_tokens // 4)
        task = session.instruction[:cap]
        if len(session.instruction) > cap:
            task += "\n[Task excerpt. Full instruction: context['task'] in Python; or session_inspect in direct mode.]"
        metadata = {
            "id": sid,
            "root_id": session.root_id,
            "parent_id": session.parent_id,
            "name": session.name,
            "depth": session.depth,
            "max_depth": self.store.config(session.root_id).limits.max_depth,
            "max_subagents": self.store.config(session.root_id).limits.max_subagents,
            "role": session.role,
            "workspace": session.workspace.path,
            "features": self.store.config(sid).features.model_dump(),
            "execution_backend": self.store.config(sid).execution.backend,
            "permissions": self.store.config(sid).permissions,
            "read_only": self.store.config(sid).execution.read_only,
            "permitted_host_capabilities": self.store.config(sid).tool_allowlist,
        }
        usage = self.store.usage(session.root_id, tree=True)
        limits = self.store.config(session.root_id).limits
        status = {"turn": session.turns, "goal": self.store.goal(sid)}
        status["resources_remaining"] = {
            "tokens": max(0, limits.token_budget - usage.input_tokens - usage.output_tokens),
            "root_turns": max(0, limits.max_turns - usage.turns),
            "session_turns": max(0, limits.max_turns - session.turns),
        }
        root = self.store.session(session.root_id)
        if root.mode != "interactive":
            elapsed = max(0, now() - root.started_at) if root.started_at else 0
            status["resources_remaining"]["wall_seconds"] = max(0, limits.wall_seconds - elapsed)
        if session.parent_id:
            metadata["delegation_budget_note"] = (
                "All siblings share root_turns and tokens. Send useful partial evidence promptly; do not consume the shared budget polishing a report."
            )
        if self.store.config(sid).control_plane == "python":
            metadata["conversation_log"] = str(self.history_file(sid))
        # A long goal is still available in L3, and must not defeat context bounds.
        if status["goal"]:
            status["goal"]["objective"] = status["goal"]["objective"][:cap]
        messages = [
            {
                "role": "system",
                "content": python_instructions(
                    self.store.config(sid),
                    child=bool(session.parent_id),
                    specifications=[
                        {k: e[k] for k in ("id", "title", "version")}
                        for e in self.store.states(sid)
                        if e["kind"] == "subagent_spec"
                    ][:20],
                )
                if self.store.config(sid).control_plane == "python"
                else FOUNDATION,
            },
            {"role": "user", "content": "Session: " + encode(metadata)},
        ]
        if self.environment:
            instructions = self.environment().instructions(self.store.config(sid))
            if instructions:
                messages.append({"role": "system", "content": instructions})
        if session.mode == "interactive" and self.store.config(sid).control_plane == "direct":
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": (
                        "This is an interactive conversation in the SAME persistent session. Follow the "
                        "latest human messages. For questions, inspection, or a progress report, a text "
                        "reply without tool calls yields to the human; it does not claim verified task "
                        "completion. After tools, continue until you can answer or need user input. "
                        "Call finish when the requested work is done; any configured completion gate "
                        "still applies. Do not claim independent verification unless a verifier passed. "
                        "A successful finish returns to the human without destroying the conversation."
                    ),
                },
            )
        messages.extend(self.store.config(sid).task.instruction_messages)
        if session.parent_id:
            messages.append(
                {
                    "role": "user",
                    "content": "Original root task (complete contract: task.messages):\n"
                    + self.context_task_excerpt(root.instruction, cap),
                }
            )
        messages.append({"role": "user", "content": task})
        supplemental = self.supplemental(sid)
        if self.store.config(sid).control_plane == "python":
            # Bounded state/skill menus mirror available capabilities, not a ranking
            # of coding evidence. Full procedures/content stay outside L1.
            from pathlib import Path

            from .skills import discover

            config = self.store.config(sid)
            menu = {
                "state": [
                    {k: e[k] for k in ("id", "kind", "title", "version")}
                    for e in self.store.states(sid)[:20]
                ],
                "skills": [
                    {k: e[k] for k in ("name", "path", "description", "import_name") if k in e}
                    for e in discover(Path(session.workspace.path), config.skill_paths)[:20]
                ],
            }
            if menu["state"] or menu["skills"]:
                messages.append(
                    {
                        "role": "user",
                        "content": "Available supplemental state/skills (inspect from Python): "
                        + encode(menu)[: config.context.supplemental_chars],
                    }
                )
        if supplemental:
            messages.append(
                {"role": "user", "content": "Selected supplemental state:\n" + supplemental}
            )
        from .semantic_state import completion_evidence, work_items

        completion_state = completion_evidence(self.store, sid)
        if completion_state:
            # Expose status and source handles, not the potentially huge receipt.
            verification = completion_state["last_full_verification"]
            if verification:
                verification.pop("receipt", None)
            messages.append(
                {
                    "role": "user",
                    "content": "Live completion evidence (survives compaction): "
                    + encode(completion_state)
                    + "\nThis is the last observed result, not a claim about unobserved changes. "
                    "When requirements are satisfied and relevant children have finished, synthesize "
                    "your final answer. Do not repeatedly reread unchanged files or rerun successful "
                    "checks without a new change, unresolved requirement or concrete uncertainty. "
                    "The runtime still performs the configured completion gate on your final answer.",
                }
            )

        active_work = [i for i in work_items(self.store, sid) if i["status"] == "open"]
        if active_work:
            messages.append(
                {
                    "role": "user",
                    "content": "Live unresolved work (context.track/resolve): "
                    + encode(active_work),
                }
            )
        snapshots = self.store.events(sid, kind="kernel_snapshot", limit=1)
        if snapshots:
            snapshot = snapshots[-1]["payload"]
            messages.append(
                {
                    "role": "user",
                    "content": "L2 state: "
                    + encode(
                        {
                            k: snapshot.get(k)
                            for k in (
                                "manifest_artifact",
                                "offloaded",
                                "pruned",
                                "reconstructible",
                                "missing",
                            )
                        }
                    ),
                }
            )
        if session.summary:
            messages.append(
                {
                    "role": "user",
                    "content": "Earlier trajectory digest (details remain in history):\n"
                    + session.summary,
                }
            )
        for block in session.context:
            for source in block["messages"]:
                message = dict(source)
                event = message.get("provider_response_event")
                if event:
                    row = self.store.db.execute(
                        "SELECT * FROM provider_continuations WHERE event_id=?", (event,)
                    ).fetchone()
                    config = self.store.config(sid)
                    if row and any(
                        p.name == row["provider"] and p.model == row["model"]
                        for p in [config.provider, *config.models.values()]
                    ):
                        message["provider_items"] = json.loads(row["items"])
                        message["provider_identity"] = [row["provider"], row["model"]]
                messages.append(message)
        if self.environment:
            messages.extend(self.environment().call(sid, "context_messages", sid, default=[]))
        messages.append(
            {
                "role": "developer",
                "content": "Current runtime status (accounting metadata): " + encode(status),
                "context_status": True,
            }
        )
        return messages

    @staticmethod
    def context_task_excerpt(instruction, cap):
        return instruction[:cap] + (
            "\n[Excerpt; full task in task.messages.]" if len(instruction) > cap else ""
        )

    def request_estimate(self, sid, messages, tools, provider=None):
        from .request_context import request_estimate

        if provider is None:
            config, session = self.store.config(sid), self.store.session(sid)
            alias = config.routing.default
            if config.routing.policy == "role_based" and session.role in config.routing.roles:
                alias = config.routing.roles[session.role]
            provider = config.models[alias] if alias else config.provider
        return request_estimate(self.store, sid, messages, tools, provider)

    def compact(
        self,
        sid: str,
        *,
        count: int | None = None,
        summary=None,
        provenance=None,
        review_checkpoint=True,
    ):
        with self.store.transaction():
            session, policy = self.store.session(sid), self.store.config(sid).context
            if not session.context:
                return None
            count = count or max(1, len(session.context) - policy.recent_blocks)
            removed = session.context[:count]
            from .artifacts import Artifacts
            from .semantic_state import capture, compact_reference, protected_summary

            live_tree = capture(self, sid)
            tree_artifact = Artifacts(self.store).put(sid, live_tree)
            archive = Artifacts(self.store).put(
                sid,
                {
                    "previous_summary": session.summary,
                    "blocks": removed,
                    "selected_state": session.selected_state,
                    "proposed_summary": summary,
                    "semantic_tree_artifact": tree_artifact,
                },
            )
            from .context_budget import (
                extractive_summary,
                merge_summary,
                policy_tokens,
                summary_budget,
            )

            model = self.store.config(sid).provider.model
            try:
                parsed = json.loads(summary) if isinstance(summary, str) else summary
            except ValueError:
                parsed = None
            if not isinstance(parsed, dict):
                parsed = extractive_summary(session.summary, removed)
                if summary:
                    parsed["established_facts"].append(summary)
            resolutions = [
                {
                    "id": "child-" + b["id"],
                    "reason": "Child completed; conclusion and provenance retained",
                    "source_events": b.get("completion_events", []),
                }
                for b in live_tree["branches"]
                if b["status"] == "completed" and b.get("completion_events")
            ]
            parsed.setdefault("resolved_items", []).extend(resolutions)
            parsed = merge_summary(
                session.summary,
                parsed,
                source_events=[b["event_id"] for b in session.context]
                + [e for r in resolutions for e in r["source_events"]],
            )
            for field, items in protected_summary(live_tree).items():
                parsed.setdefault(field, []).extend(
                    i for i in items if i not in parsed.get(field, [])
                )
            parsed.setdefault("important_references", []).append(
                compact_reference(live_tree, tree_artifact)
            )
            summary = summary_budget(
                parsed,
                budget=policy_tokens(policy, "summary", model),
                model=model,
                reference=f"Complete compacted history and prior digest: artifacts.load({archive!r})",
            )
            source_events = [b["event_id"] for b in removed]
            eid = self.store.event(
                sid,
                "context_compaction",
                {
                    "source_events": source_events,
                    "archive_artifact": archive,
                    "semantic_tree_artifact": tree_artifact,
                    "pending_children": [
                        b["id"] for b in live_tree["branches"] if b["status"] == "active"
                    ],
                    "summary": summary,
                    "method": "model_structured" if provenance else "bounded_extractive",
                    "model_response_event": provenance,
                    "removed_tokens_estimate": estimate(
                        removed, self.store.config(sid).provider.model
                    ),
                    "summary_tokens_estimate": estimate(
                        summary, self.store.config(sid).provider.model
                    ),
                    "savings_estimate": estimate(removed, self.store.config(sid).provider.model)
                    - estimate(summary, self.store.config(sid).provider.model),
                },
            )
            self.store.db.execute(
                "INSERT INTO compactions VALUES(?,?,?,?,?)",
                (
                    eid,
                    sid,
                    encode(source_events),
                    summary,
                    self.store.event_by_id(eid)["timestamp"],
                ),
            )
            self.store.update(sid, context=session.context[count:], summary=summary)
            if provenance:
                request = self.store.db.execute(
                    "SELECT id FROM model_requests WHERE response_event=?", (provenance,)
                ).fetchone()
                if request:
                    self.store.commit_compaction(sid, request[0])
            config = self.store.config(sid)
            if (
                review_checkpoint
                and config.refinement.enabled
                and config.refinement.automatic
                and config.features.automatic_refinement
            ):
                self.store.enqueue_refinement_request(
                    sid,
                    source="compaction",
                    source_event=eid,
                    request_id=f"compaction-{eid}",
                    trigger="compaction",
                )
            return eid

    def assemble(
        self, sid: str, tools: list[dict], *, input_budget=None, proactive=True
    ) -> tuple[list[dict], int]:
        config = self.store.config(sid)
        available = config.context.max_tokens - max(
            p.max_output_tokens for p in [config.provider, *config.models.values()]
        )
        if input_budget is not None:
            available = min(available, input_budget)
        messages = self.messages(sid)
        size, estimation = self.request_estimate(sid, messages, tools)
        threshold = int(available * config.context.compact_at) if proactive else available
        while size > threshold and self.store.session(sid).context:
            if (
                len(self.store.session(sid).context) <= config.context.recent_blocks
                and size <= available
            ):
                break  # Keep the recent region verbatim when it fits the actual input budget.
            self.compact(sid)
            messages = self.messages(sid)
            size, estimation = self.request_estimate(sid, messages, tools)
        if size > available:
            raise HarnessError(
                "runtime",
                "context_capacity",
                f"Instructions and tool schemas need {size} token-bound units; "
                f"only {available} available. Increase context.max_tokens or reduce tools.",
            )
        self.store.event(
            sid,
            "context_estimate",
            {
                "estimated_pre_call_tokens": size,
                **method(config.provider.model),
                **estimation,
                "summary_tokens": estimate(self.store.session(sid).summary, config.provider.model),
                "evidence_tokens": sum(
                    estimate(m, config.provider.model)
                    for m in messages
                    if (m.get("content") or "").startswith("Current adapter evidence:")
                ),
            },
        )
        return messages, size
