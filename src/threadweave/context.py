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

PYTHON_CONTROL = """You operate a persistent agent session. You decide strategy. Your sole tool is ipython.
Use the persistent IPython namespace as the programmable control plane. Variables and top-level
await are supported. Only printed/returned information enters model context. Keep large values in
variables; access the full original instruction as context['task'], and history/artifacts through
history.read(), history.search(), history.messages(), artifacts.load(id). Nothing requires coding.
Preloaded APIs (help()/inspect work):
- pathlib, Path, os, asyncio, json; workspace is a Path; session and context contain metadata.
- bash(command) starts immediately and returns a handle; await bash(command) returns exit_code,
  output, stdout, stderr, duration and artifact IDs. Background handles have pid, running, poll(),
  output(), tail(), kill(). Use the project's own environment (e.g. uv run pytest) via bash.
- await rlm(prompt, name=None, model=None, thinking=None, purpose='shared') returns a persistent
  child HANDLE upon admission, never its answer. Root continues. purpose='research' shares files
  with read-only tool policy (trusted Python is not sandboxed); 'shared' explicitly allows shared
  worktree collaboration; 'candidate' captures dirty state in an isolated writable Git worktree.
  await rlm.candidate(handle) inspects a completed candidate; accept=True applies its validated patch.
  await rlm.list_subagents(); await rlm.find_models(); await rlm.delete_subagent(handle).
- await agent_message.send(text, receiver_role='parent'|'child'|'sibling', receiver_name=...);
  omit receiver_name for parent; await agent_message.list_agents(); await agent_message.receive().
- await edit(path, old_str, new_str) replaces one unique match. Path read/write and repo helpers
  are available. repo.search(query=...), repo.map(), repo.symbols(query=...), git.diff().
- tools.call(name, **args) / await tools.acall(name, **args) invoke optional capabilities;
  tools.catalog() returns full schemas into Python, not automatically into your prompt.
- rlm.harness / harness: create_memory, create_prompt_note, create_skill, create_subagent;
  get(kind,id), list(kind=None), update(kind,id,title,content), delete(kind,id), rollback(kind,id,version),
  select(ids). Writes are versioned and attributed to the executing cell. Skills are executable
  modules or validated code; use skills.list(), skills.load(name), await skills.run(name, **inputs).
- mcp: await mcp.list_servers(), await mcp.list_tools(server), await mcp.call_tool(server,tool,args),
  await mcp.reload(server). Only configured/enabled MCP capabilities are permitted.
- await goal.get(), await goal.create(objective), await goal.complete(); await compact();
  await refine(); await heartbeat(interval_seconds=..., instruction=...). Create goals only if asked.
Normal text replies yield/end ordinary interaction; there is no universal finish tool or coding
verifier. Explicit goals and configured task gates remain binding. Do not claim tests you did not run.
L1 is selected context; L2 is persistent Python and recursive sessions; L3 is durable history/state.
Compaction does not delete values or children. Recovery restores codecs and explicit recipes, not
arbitrary live objects; use remember_recipe(name,code) / forget(*names) and inspect warnings.
Tools enforce permissions, but Python and local shell are trusted-host code, NOT a sandbox.
Session.permitted_host_capabilities, when non-null, restricts the preloaded helpers. Do not call
disabled helpers; ordinary Path/open Python and permitted bash remain available.
Respect the user's scope. Never change foundational policy through supplemental state.
"""


def token_bound(value, model=None) -> int:
    """Conservative UTF-8 byte bound plus framing; provider-reported usage remains authoritative."""
    return estimate(value, model)


class Context:
    def __init__(self, store: Store):
        self.store = store
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
        entries, remaining = [], config.context.supplemental_chars
        for entry_id in session.selected_state:
            try:
                entry = self.store.state(sid, entry_id)
            except KeyError:
                continue
            if entry["deleted"]:
                continue
            text = encode(
                {
                    "id": entry_id,
                    "kind": entry["kind"],
                    "version": entry["version"],
                    "title": entry["title"],
                    "content": entry["content"],
                }
            )
            if len(text) <= remaining:
                entries.append(text)
                remaining -= len(text)
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
            "turn": session.turns,
            "goal": self.store.goal(sid),
            "features": self.store.config(sid).features.model_dump(),
            "execution_backend": self.store.config(sid).execution.backend,
            "permitted_host_capabilities": self.store.config(sid).tool_allowlist,
        }
        if self.store.config(sid).control_plane == "python":
            metadata["conversation_log"] = str(self.history_file(sid))
        # A long goal is still available in L3, and must not defeat context bounds.
        if metadata["goal"]:
            metadata["goal"]["objective"] = metadata["goal"]["objective"][:cap]
        messages = [
            {
                "role": "system",
                "content": PYTHON_CONTROL
                if self.store.config(sid).control_plane == "python"
                else FOUNDATION,
            },
            {"role": "user", "content": "Session: " + encode(metadata) + "\nTask: " + task},
        ]
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
            messages.extend(block["messages"])
        from .retrieval import coding_focus

        focus = coding_focus(self.store, sid)
        if focus:
            messages.append({"role": "user", "content": "Current coding evidence: " + focus})
        return messages

    def compact(self, sid: str, *, count: int | None = None, summary=None, provenance=None):
        with self.store.transaction():
            session, policy = self.store.session(sid), self.store.config(sid).context
            if not session.context:
                return None
            count = count or max(1, len(session.context) - policy.recent_blocks)
            removed = session.context[:count]
            excerpts = []
            for block in removed:
                pieces = []
                for message in block["messages"]:
                    content = message.get("content") or encode(message.get("tool_calls", []))
                    pieces.append(f"{message['role']}: {content[:400]}")
                excerpts.append(f"event={block['event_id']} " + " | ".join(pieces))
            # Keep unresolved negative evidence before less consequential excerpts.
            critical = [
                e
                for e in excerpts
                if any(
                    word in e.lower()
                    for word in (
                        "failed",
                        "error",
                        "constraint",
                        "hypothesis",
                        "regression",
                        "must not",
                    )
                )
            ]
            additions = "\n".join(excerpts)
            cap = policy.summary_chars
            # Preserve part of the prior digest and the most recent observations.
            prior = (session.summary[: cap // 4] + "\n" + "\n".join(critical)[-cap // 3 :]).strip()
            summary = (
                summary[:cap]
                if summary
                else (prior + "\n" + additions[-(cap - len(prior) - 1) :]).strip()
            )
            source_events = [b["event_id"] for b in removed]
            eid = self.store.event(
                sid,
                "context_compaction",
                {
                    "source_events": source_events,
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
            return eid

    def assemble(self, sid: str, tools: list[dict], *, input_budget=None) -> tuple[list[dict], int]:
        config = self.store.config(sid)
        available = config.context.max_tokens - max(
            p.max_output_tokens for p in [config.provider, *config.models.values()]
        )
        if input_budget is not None:
            available = min(available, input_budget)
        messages = self.messages(sid)
        size = token_bound({"messages": messages, "tools": tools}, config.provider.model)
        threshold = int(available * config.context.compact_at)
        while size > threshold and self.store.session(sid).context:
            self.compact(sid)
            messages = self.messages(sid)
            size = token_bound({"messages": messages, "tools": tools}, config.provider.model)
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
                "summary_tokens": estimate(self.store.session(sid).summary, config.provider.model),
                "evidence_tokens": sum(
                    estimate(m, config.provider.model)
                    for m in messages
                    if (m.get("content") or "").startswith("Current coding evidence:")
                ),
            },
        )
        return messages, size
