from __future__ import annotations

import json

from .models import HarnessError
from .storage import Store, encode
from .tokenization import estimate, method

# Foundational instructions are code-owned. Adaptive entries never replace this message.
FOUNDATION = """You operate a persistent agent session. Decide your own strategy and next actions.
Use the provided tools to compute, inspect evidence, delegate, communicate, and finish.
When enabled, python executes in your own Python worker with top-level await. Variables are retained
only when the session's persistent_repl feature is enabled; inspect the capability metadata.
Python helpers: rlm(instruction, name=None), tools.call(name, **arguments), await tools.acall(name, **arguments),
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
disk-backed history, artifacts, messages and versioned reusable state. Compaction affects only L1.
The Environment exposes capabilities, not a required sequence of actions. Ordinary conversation
needs no repository, test baseline or code change. Never invent observations or measurements.
"""


def token_bound(value, model=None) -> int:
    """Conservative UTF-8 byte bound plus framing; provider-reported usage remains authoritative."""
    return estimate(value, model)


def python_instructions(config):
    """Expose only usable capabilities; compact help remains in the live namespace."""
    text = """You control a persistent agent session through the sole tool ipython(code).
Choose your strategy. Python supports top-level await and retains variables; keep large data in
variables and print only relevant evidence. Preloaded: Path, pathlib, os, asyncio, json,
workspace (Path), session (metadata), context['task'] (complete instruction).
Use Path.read_text()/write_text(), ordinary Python, or await bash(command) for execution.
await bash returns exit_code, stdout, stderr, duration and artifact IDs. bash(command) without
await returns a background handle with poll(), tail(), kill(). Use the repository's environment.
Never claim execution results you did not observe. A normal final text reply requests completion;
the independently configured verifier remains authoritative and may return failure evidence.
L1 is selected context, L2 is Python state, L3 is durable history/artifacts. Compaction does not
delete Python values. On recovery inspect warnings; only checkpointed values/recipes restore.
Interrupted actions may have partial effects; inspect them before retrying. Python and shell run
as trusted local code, not a general sandbox. Stay within the supplied workspace and user scope.
Respect all resource limits. Do not fetch hidden/reference solutions. Never modify foundational
instructions through supplemental state. Disabled capabilities must not be invoked.
"""
    if "process" not in config.permissions or (
        config.execution.read_only and config.execution.backend == "local"
    ):
        text = text.replace(
            "Use Path.read_text()/write_text(), ordinary Python, or await bash(command) for execution.\nawait bash returns exit_code, stdout, stderr, duration and artifact IDs. bash(command) without\nawait returns a background handle with poll(), tail(), kill(). Use the repository's environment.",
            "Use Path.read_text() and ordinary Python for inspection. Host shell execution is not\navailable with this session's permissions/backend. Do not invoke bash.start to bypass that policy.",
        )
    if config.execution.read_only:
        text += "This session is OS read-only outside its private kernel state. Do not attempt repository writes.\n"
    if config.features.enhanced_code_index:
        text += """Coding APIs: repo, tests, context, verify, git, edit. Their help() gives short signatures.
Prefer repo.context_for_symbol(name) for precise evidence; tests.related_to(files=[...]) explains
test selection. context.focus(files=[...],hypothesis='...') retains your investigation.
await edit(path,old_str,new_str) edits a unique match. Raw Python remains available.
"""
    if config.features.subagents:
        text += """await rlm('assignment',name='...',purpose='research'|'candidate'|'shared') returns a
persistent HANDLE asynchronously, not an answer. Research is OS read-only; candidates have isolated
worktrees. agents.help() explains messaging and explicit patch acceptance. Delegate with purpose.
"""
    if config.features.history_retrieval:
        text += "history.search(query), history.get(event_id), context.search(query), artifacts.load(id) retrieve retained evidence.\n"
    if config.features.experiments:
        text += "experiment.help() lists durable experiment/measurement procedures.\n"
    if config.tool_allowlist is None:
        text += "harness, skills, mcp expose versioned state, executable skills and configured servers; inspect their methods/help as needed. tools.catalog() returns schemas into Python, not L1. await refine() requests evidence-based refinement; await compact() compacts context.\n"
    return text


class Context:
    def __init__(self, store: Store, index_provider=None):
        self.store = store
        self.index_provider = index_provider
        self._export_cursors = {}

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
                "content": python_instructions(self.store.config(sid))
                if self.store.config(sid).control_plane == "python"
                else FOUNDATION,
            },
            {"role": "user", "content": "Session: " + encode(metadata) + "\nTask: " + task},
        ]
        if (
            self.store.config(sid).task.adapter == "coding"
            and self.store.config(sid).features.enhanced_code_index
        ):
            messages.append(
                {
                    "role": "system",
                    "content": "Coding decision support: reuse observed failures and exact source evidence. State a hypothesis when debugging; prefer focused definitions/callers and related failing tests over repeatedly reading full files. Use repo.help()/tests.help() to discover APIs. Make small evidence-supported changes; escalate tests when warranted. Delegation must have a distinct purpose and a bounded evidence request. Repeated unchanged reads/searches/tests are a signal to revise the hypothesis, not proof of progress. These are cost-aware guidelines, not a mandatory workflow; the independent final verifier still decides correctness.",
                }
            )
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
        for instruction in session.repository_instructions:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Repository instructions from {instruction['path']} (scope {instruction['scope']}):\n"
                        + instruction["content"]
                    ),
                }
            )
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
        from .retrieval import coding_focus

        focus = coding_focus(self.store, sid, index_provider=self.index_provider)
        if focus:
            messages.append({"role": "user", "content": "Current coding evidence: " + focus})
        messages.append(
            {
                "role": "developer",
                "content": "Current runtime status (accounting metadata): " + encode(status),
                "context_status": True,
            }
        )
        return messages

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

            archive = Artifacts(self.store).put(
                sid,
                {
                    "previous_summary": session.summary,
                    "blocks": removed,
                    "selected_state": session.selected_state,
                    "proposed_summary": summary,
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
            parsed = merge_summary(
                session.summary,
                parsed,
                source_events=[b["event_id"] for b in session.context],
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

    def assemble(self, sid: str, tools: list[dict], *, input_budget=None) -> tuple[list[dict], int]:
        config = self.store.config(sid)
        available = config.context.max_tokens - max(
            p.max_output_tokens for p in [config.provider, *config.models.values()]
        )
        if input_budget is not None:
            available = min(available, input_budget)
        messages = self.messages(sid)
        size, estimation = self.request_estimate(sid, messages, tools)
        threshold = int(available * config.context.compact_at)
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
                    if (m.get("content") or "").startswith("Current coding evidence:")
                ),
            },
        )
        return messages, size
