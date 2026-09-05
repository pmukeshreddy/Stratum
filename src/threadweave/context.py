from __future__ import annotations

from .models import HarnessError
from .storage import Store, encode

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


def token_bound(value) -> int:
    """Conservative UTF-8 byte bound plus framing; provider-reported usage remains authoritative."""
    return len(encode(value).encode("utf-8")) + 64


class Context:
    def __init__(self, store: Store):
        self.store = store

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
            task += "\n[Task excerpt. Retrieve the full instruction with session_inspect.]"
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
        }
        # A long goal is still available in L3, and must not defeat context bounds.
        if metadata["goal"]:
            metadata["goal"]["objective"] = metadata["goal"]["objective"][:cap]
        messages = [
            {"role": "system", "content": FOUNDATION},
            {"role": "user", "content": "Session: " + encode(metadata) + "\nTask: " + task},
        ]
        if session.mode == "interactive":
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
            additions = "\n".join(excerpts)
            cap = policy.summary_chars
            # Preserve part of the prior digest and the most recent observations.
            prior = session.summary[: cap // 3] if session.summary else ""
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

    def assemble(self, sid: str, tools: list[dict]) -> tuple[list[dict], int]:
        config = self.store.config(sid)
        available = config.context.max_tokens - max(
            p.max_output_tokens for p in [config.provider, *config.models.values()]
        )
        messages = self.messages(sid)
        size = token_bound({"messages": messages, "tools": tools})
        threshold = int(available * config.context.compact_at)
        while size > threshold and self.store.session(sid).context:
            self.compact(sid)
            messages = self.messages(sid)
            size = token_bound({"messages": messages, "tools": tools})
        if size > available:
            raise HarnessError(
                "runtime",
                "context_capacity",
                f"Instructions and tool schemas need {size} token-bound units; "
                f"only {available} available. Increase context.max_tokens or reduce tools.",
            )
        return messages, size
