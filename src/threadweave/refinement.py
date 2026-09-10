"""Serialized Prime-style refinement checkpoints, independent of trajectory persistence."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .auxiliary import AuxiliaryServices
from .harness import (
    HarnessAuditError,
    history_for_refinement,
    infer_scope,
    js_slice,
    load_harness_state,
    normalize_proposal,
    overview_for_refinement,
    rollback_proposal,
    timestamp,
)
from .models import Outcome
from .refinement_context import custom_message, refinement_messages, serialize_conversation

_UNSET = object()


class RefineSkippedError(Exception):
    pass


def refine_command(text):
    if not text.startswith("/refine") or any(c in text for c in "\n\r\u2028\u2029"):
        return None
    rest = text[len("/refine") :]
    if rest and rest[0] != "\t" and unicodedata.category(rest[0]) != "Zs":
        return None
    return {"name": "refine", "args": rest.strip(), "text": text}


def refine_command_options(args):
    rest, global_ = args.strip(), False
    if re.match(r"^--global(?=\s|$)", rest):
        global_, rest = True, rest[len("--global") :].strip()
    if rest == "rollback":
        raise ValueError("Usage: /refine rollback <refinement-id>")
    if (
        rest.startswith("rollback")
        and len(rest) > 8
        and (rest[8] == "\t" or unicodedata.category(rest[8]) == "Zs")
    ):
        identifier = rest[9:].strip()
        if identifier == "--global":
            raise ValueError("Usage: /refine rollback <refinement-id>")
        if re.search(r"\s--global$", identifier):
            global_, identifier = True, re.sub(r"\s--global$", "", identifier).strip()
        if not identifier:
            raise ValueError("Usage: /refine rollback <refinement-id>")
        return {"rollback_id": identifier, "global_": global_}
    return {"global_": global_, **({"instructions": rest} if rest else {})}


# Behavioral prompts ported from Prime's core/refinement/refinement.ts.
PLANNER_PROMPT = """You are Buffalo's /refine continual harness subsystem.

Your job is to improve the editable continual harness state from the current trajectory.
This is similar in spirit to context compaction, but instead of summarizing the
conversation you emit precise Create, Update, or Delete edits to reusable state.
The continual harness is the persistent, editable set of prompt notes, memories,
skills, and subagent specs that lets Buffalo improve reusable behavior
outside the token history.
Use "continual harness" for that persistent artifact layer; keep "RLM" for the
runtime, Python REPL kernel, and native call interface that executes those artifacts.

Continual harness components:
- prompt: supplemental prompt notes only. The base system prompt is immutable and MUST NOT be rewritten.
- memory: durable facts, decisions, failures, preferences, and outcomes.
- skill: installed Python REPL skill. Skill create/update edits MUST include a `reference` object with `{"type":"python"}`, a Python import, and a callable or call pattern; they also MUST include an `arguments` object describing accepted inputs, required fields, defaults, and constraints. Use `{}` for `arguments` only when the Python callable truly needs no external inputs. Include the RLM-native call form `await <skill_import>(...)`.
- subagent: reusable delegation specs, including purpose, instructions, and when to invoke. Include the RLM-native call form: compose a concise task prompt and spawn with `handle = await rlm("sub-task")`; admission returns immediately with `rlm_child_id`, `name`, `session_dir`, and `model`, never the child's answer. Results arrive only through explicit `agent_message` replies or files; children reply with `await agent_message.send(message, receiver_role="parent")`. Use `await rlm.list_subagents()` to recover direct child handles and `await agent_message.send(..., receiver_role="child", receiver_name=handle.name)` for follow-ups. Do not invent wrappers like `run_subagent(...)`.

Scope and persistence policy:
- The default editable continual harness store is local to the current Buffalo session. Use it for session-specific progress, active task state, current-run coordination notes, temporary blockers, and project facts that should not affect other sessions.
- A caller may explicitly request global refinement. Global edits must be stable cross-session lessons, durable user preferences, reusable skills/subagents, or tool/environment facts that should affect future sessions.
- Entry ids in the harness overview may carry a display-only `local:` or `global:` prefix. Always use the bare id (no prefix) in edits.
- All edits in one refinement apply only to the requested scope's store. During a local refinement, global entries are read-only context: never propose update or delete edits for them; create a local entry instead when a session-specific override is genuinely needed.
- Project/workspace-specific lessons may be persisted globally only when the title, path, or content explicitly names the project/workspace and the lesson is likely to be reused in future sessions for that project. Prefer local edits when the lesson only belongs in the current conversation.
- Use memory for declarative facts and preferences, skill for repeatable procedures exposed as Python calls, prompt for narrow behavioral policy addendums, and subagent for reusable delegation roles.
- Create or update the smallest relevant component: repeated delegation roles should become subagent specs, repeated procedures should become skills, durable facts/preferences should become memories, and narrow behavioral policies should become prompt addendums.
- When an edit is persisted, include metadata such as `{"scope":"local"}` or `{"scope":"global"}` when that helps future review understand the intended blast radius.

Use the trajectory, current continual harness state, and prior refinement history. Prefer
small evidence-backed edits. If prior refinements caused issues, rollback or
replace the faulty editable entries. Never edit source files directly. Output
JSON only with this exact shape:

{
  "summary": "one sentence",
  "rationale": "why these edits are justified by trajectory evidence",
  "expectedOutcome": "what should improve and how to validate it",
  "edits": [
    {
      "action": "create|update|delete",
      "kind": "prompt|memory|skill|subagent",
      "id": "stable id for update/delete, optional for create",
      "title": "required for create/update except delete",
      "content": "required for create/update except delete",
      "path": "optional grouping path",
      "reference": {"type": "python", "import": "package.module", "callable": "function_name", "call_pattern": "await function_name(...)"},
      "arguments": {"name": {"type": "string", "required": true, "description": "accepted input"}},
      "metadata": {},
      "reason": "why this edit is useful"
    }
  ]
}"""

REVIEW_PROMPT = """You are Buffalo's automatic /refine review gate.

Decide whether this checkpoint should run /refine. Auto /refine writes local continual harness state by default, so approve when the trajectory contains evidence useful to this session's future turns.
Reject one-off noise, unsupported hypotheses, and transient tool outputs. Ask for global refinement only for durable cross-session lessons or explicitly project-qualified lessons likely to be reused in future sessions.

Return JSON only:
{
  "shouldRefine": true|false,
  "rationale": "short reason",
  "instructions": "optional concise instructions for /refine if shouldRefine is true"
}"""


def auto_refine_instructions(reason, review):
    detail = (
        f"\nReviewer instructions: {review['instructions']}" if review.get("instructions") else ""
    )
    return (
        f"Automatic refine review triggered by {reason}. Only create/update/delete local "
        "harness entries if there is clear evidence that should help this session continue. "
        "Prefer an empty edits array over speculative or one-off memories. Do not promote "
        f"anything global unless explicitly requested. Reviewer rationale: {review['rationale']}{detail}"
    )


TRUNCATED_JSON_ERROR = (
    "the model stopped before completing its JSON object. This usually means the output "
    "budget was exhausted; retry with a smaller request."
)


def incomplete_json(candidate):
    depth, in_string, escaped = 0, False, False
    for char in candidate:
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return in_string or depth > 0


def parse_object(text):
    def parse(candidate):
        def invalid_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        return json.loads(candidate, parse_constant=invalid_constant)

    text = text.strip()
    candidate = text
    if not (text.startswith("{") and text.endswith("}")):
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            candidate = fenced[1].strip()
        else:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    parse(candidate)
                except ValueError:
                    candidate = text[start:]
            elif incomplete_json(text):
                raise ValueError(TRUNCATED_JSON_ERROR)
            else:
                raise ValueError("Refiner did not return a JSON object")
    try:
        result = parse(candidate)
    except ValueError as exc:
        if incomplete_json(candidate):
            raise ValueError(TRUNCATED_JSON_ERROR) from exc
        raise ValueError(f"the model did not return valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Refinement model JSON must be an object")
    return result


@dataclass
class Checkpoint:
    turns_since_review: int = 0
    last_review_at: float = 0
    in_progress: bool = False
    pending_request: dict | None = None
    pending_review: tuple | None = None
    pending_compact: bool = False
    pending_interval: bool = False
    branch_version: int = 0
    task: asyncio.Task | None = None
    background: asyncio.Task | None = None
    background_options: dict | None = None
    claim: asyncio.Task | None = None
    counted_response: str | None = None
    reviewing: bool = False
    plan_task: asyncio.Task | None = None
    apply_task: asyncio.Task | None = None
    auto_task: asyncio.Task | None = None
    draining: bool = False
    command_task: asyncio.Task | None = None


class RefinementServices(AuxiliaryServices):
    def serialized_refinement(self, sid):
        configured = self.store.config(sid).serialized_refine
        return (
            configured if configured is not None else self.store.session(sid).mode != "interactive"
        )

    def refinement_state(self, sid):
        if sid not in self._refinement_states:
            self._refinement_states[sid] = Checkpoint()
        return self._refinement_states[sid]

    def refinement_status(self, sid):
        state = self.refinement_state(sid)
        return {
            "pending": state.pending_request is not None,
            "in_flight": (state.in_progress and not state.reviewing)
            or state.background is not None
            or state.plan_task is not None
            or state.apply_task is not None,
        }

    def request_refinement(self, sid, *, instructions=_UNSET, global_=_UNSET):
        if instructions is not _UNSET and not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        if global_ is not _UNSET and not isinstance(global_, bool):
            raise TypeError("global_ must be bool")
        session = self.store.session(sid)
        if session.depth != 0:
            return {
                "scheduled": False,
                "reason": "Refinement is available only to the root session",
            }
        if not session.pending_turn:
            return {
                "scheduled": False,
                "reason": "no active turn; refine can only be requested while a turn is running",
            }
        if session.outcome not in (Outcome.ACTIVE, Outcome.COMPLETED):
            return {"scheduled": False, "reason": "Session is stopped"}
        state = self.refinement_state(sid)
        previous = state.pending_request or state.background_options or {}
        request = {**previous, "source": "self"}
        if instructions is not _UNSET:
            request["instructions"] = instructions
        if global_ is not _UNSET:
            request["global_"] = global_
        if self.serialized_refinement(sid) and state.background:
            state.branch_version += 1
            if state.background:
                state.background.cancel()
        state.pending_request = request
        self.store.event(sid, "refine_scheduled", request)
        self.maybe_start_refinement_plan(sid)
        self._wake.set()
        return {
            "scheduled": True,
            "note": "Refinement runs when the current turn ends; applied edits are appended to your context as a refinement notice and you resume automatically. Continue working normally.",
        }

    def has_pending_refinement(self, sid):
        state, policy = self.refinement_state(sid), self.store.config(sid).refinement
        if state.pending_request is not None or state.background is not None:
            return True
        return (
            policy.enabled
            and (
                not state.last_review_at
                or time.time() - state.last_review_at >= policy.cooldown_seconds
            )
            and (
                state.pending_review is not None
                or state.pending_interval
                or (policy.compact and state.pending_compact)
            )
        )

    def invalidate_refinement(self, sid, *, branch_change=False):
        self.context._pending_harness_digest.pop(sid, None)
        state = self.refinement_state(sid)
        state.branch_version += 1
        if branch_change:
            state.turns_since_review = 0
            state.counted_response = None
            state.pending_review = None
            state.pending_compact = state.pending_interval = False
        else:
            # requestAbort clears explicit work and aborts the current operations;
            # it does not reset the automatic turn counter or cooldown.
            state.pending_request = None
        for task in (
            state.task,
            state.background,
            state.claim,
            state.plan_task,
            state.apply_task,
            state.auto_task,
            state.command_task,
        ):
            if task and task is not asyncio.current_task():
                task.cancel()

    async def refinement_branch_changed(self, sid):
        command = self.refinement_state(sid).command_task
        if command and command is not asyncio.current_task():
            await asyncio.gather(asyncio.shield(command), return_exceptions=True)
        self.invalidate_refinement(sid, branch_change=True)
        state = self.refinement_state(sid)
        tasks = {state.background, state.plan_task, state.apply_task, state.auto_task, state.claim}
        await asyncio.gather(
            *(task for task in tasks if task and task is not asyncio.current_task()),
            return_exceptions=True,
        )
        state.background = state.background_options = None

    def start_queued_refine_command(self, sid):
        """Commands share durable FIFO input ownership with ordinary user input."""
        state = self.refinement_state(sid)
        session = self.store.session(sid)
        if state.command_task or sid in self.tasks or session.pending_turn or session.paused:
            return None
        queued = self.store.messages(sid, pending=True, limit=1)
        if not queued or queued[0]["sender_id"] is not None:
            return None
        message = queued[0]
        command = refine_command(message["body"])
        if command is None:
            return None
        message_id = message["id"]

        def record(content, *, result=False, error=False):
            message = custom_message(
                "session_slash_command_result" if result else "session_slash_command",
                content,
                details={
                    "command": command,
                    **(
                        {"success": not error, "severity": "error" if error else "info"}
                        if result
                        else {}
                    ),
                },
                display=not result or error,
            )
            try:
                event = self.store.event(sid, message["customType"], message["details"])
                self.store.add_context(sid, event, [message])
            except Exception:
                self.context.unpersisted_refinement_messages.setdefault(sid, []).append(message)

        async def run():
            try:
                await self._wait_refinement_quiescence(sid)
                with self.store.transaction():
                    record(command["text"])
                    self.store.db.execute(
                        "UPDATE messages SET received_at=? WHERE id=?",
                        (time.time(), message_id),
                    )
                try:
                    result = await self.refine(sid, **refine_command_options(command["args"]))
                except Exception as exc:
                    self.store.event(sid, "refine_failed", {"error": str(exc)})
                    record(f"Command failed: {exc}", result=True, error=True)
                    return
                applied = sum(edit["applied"] for edit in result["appliedEdits"])
                record(
                    f"Refined continual harness state: {applied} edit{'s' if applied != 1 else ''} applied.",
                    result=True,
                )
            finally:
                if state.command_task is asyncio.current_task():
                    state.command_task = None
                    if not self._closing:
                        self.start_queued_refine_command(sid)
                self._wake.set()

        state.command_task = asyncio.create_task(run(), name=f"refine-command-{sid}")
        return state.command_task

    def refinement_compacted(self, sid):
        if self.store.session(sid).depth == 0:
            self.refinement_state(sid).pending_compact = True
            self._wake.set()
            if not self.serialized_refinement(sid):
                self._schedule_deferred_auto_refine(sid)

    def refinement_boundary(self, sid, *, ignore_planning=False):
        session = self.store.session(sid)
        if self.refinement_state(sid).draining:
            return not self._closing
        if (
            self._closing
            or session.paused
            or session.outcome not in (Outcome.ACTIVE, Outcome.COMPLETED)
        ):
            return False
        if session.pending_turn and not session.pending_turn.get("context_committed"):
            return False
        if session.pending_turn is None and (
            sid in self.tasks or self.store.messages(sid, pending=True, limit=1)
        ):
            return False  # Prime gives already-owned/preparing user input its turn.
        if sid in self._transitioning:
            return False
        if self.store.db.execute(
            "SELECT 1 FROM actions WHERE session_id=? AND status='running' LIMIT 1", (sid,)
        ).fetchone():
            return False
        return not self.store.db.execute(
            "SELECT 1 FROM model_attempts a JOIN model_requests r ON r.id=a.request_id WHERE r.session_id=? AND a.status='running' AND (?=0 OR r.purpose NOT IN ('refinement','refinement_review')) LIMIT 1",
            (sid, int(ignore_planning)),
        ).fetchone()

    def refinement_input(self, sid, *, review=False, reason=None, instructions=None, global_=False):
        session = self.store.session(sid)
        pending = session.pending_turn or {}
        response = pending.get("response", {})
        conversation = [m for block in session.context for m in block["messages"]]
        # Prime serializes ordinary agent conversation, not the root system prompt.
        initial_messages = [
            m
            for m in self.store.config(sid).task.instruction_messages
            if m.get("role") not in {"system", "developer"}
        ]
        if session.summary:
            conversation.insert(0, self.context.compaction_message(session))
        else:
            index = int(
                bool(conversation) and conversation[0].get("customType") == "harness_digest"
            )
            conversation[index:index] = [
                *initial_messages,
                {"role": "user", "content": session.instruction},
            ]
        conversation.extend(self.context.unpersisted_refinement_messages.get(sid, []))
        for index, message in enumerate(conversation):
            event = message.get("provider_response_event")
            if message.get("role") == "assistant" and event:
                payload = self.store.event_by_id(event)["payload"]
                thinking = payload.get("metadata", {}).get("reasoning_summary")
                if thinking:
                    conversation[index] = {
                        **message,
                        "content": [
                            {"type": "thinking", "thinking": thinking},
                            {"type": "text", "text": message.get("content") or ""},
                        ],
                    }
        if response and not pending.get("context_committed"):
            conversation.append(
                {
                    "role": "assistant",
                    "content": response.get("text", ""),
                    "tool_calls": response.get("actions", []),
                }
            )
        text = serialize_conversation(conversation)
        inputs = {
            "current_harness_state": overview_for_refinement(
                self.store.harness.load() if global_ else self.store.harness.merged(sid)
            ),
            "refinement_history": history_for_refinement(self.store.harness.history(sid)),
            "conversation": js_slice(text, -(40_000 if review else 80_000)),
        }
        if review:
            inputs["trigger"] = {
                "reason": reason,
                "turnsSinceLastReview": self.refinement_state(sid).turns_since_review,
            }
        else:
            inputs["scope_policy"] = (
                "Requested refinement scope: global. Only propose stable cross-session continual "
                "harness edits, durable user preferences, reusable skills/subagents, or explicitly "
                "project-qualified facts that should affect future Buffalo sessions. Do not persist "
                "session-only progress, temporary blockers, or current-run coordination globally."
                if global_
                else "Requested refinement scope: local. Prefer local continual harness edits for current "
                "task progress, temporary blockers, current-run coordination, and project facts "
                "that are not clearly reusable across Buffalo sessions. Global entries in the overview "
                "are read-only context: do not propose update or delete edits for them; create "
                "a local entry instead if an override is needed."
            )
            if instructions:
                inputs["user_refine_instructions"] = instructions
        return inputs

    async def review_refinement(self, sid, reason):
        response, _ = await self.auxiliary(
            sid,
            "refinement_review",
            REVIEW_PROMPT,
            self.refinement_input(sid, review=True, reason=reason),
        )
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        self.check_refinement_response(response, "Auto-refine review")
        value = parse_object(response.text)
        review = {
            "shouldRefine": value.get("shouldRefine") is True,
            "rationale": value["rationale"]
            if isinstance(value.get("rationale"), str)
            else "No rationale provided.",
        }
        if isinstance(value.get("instructions"), str):
            review["instructions"] = value["instructions"]
        self.store.event(sid, "refinement_review", {"reason": reason, **review})
        return review

    async def plan_refinement(self, sid, options):
        global_ = options.get("global_", False)
        rollback_id = options.get("rollback_id")
        target = None
        if rollback_id:
            target = next(
                (r for r in self.store.harness.history(sid) if r["id"] == rollback_id), None
            )
            if target is None:
                raise ValueError(f"Refinement {rollback_id} not found")
            global_ = infer_scope(target) == "global"
        directory = self.store.harness.path(None if global_ else sid)
        if target and target.get("harnessStatePath"):
            path = Path(target["harnessStatePath"])
            directory = path.parent
            global_ = directory.resolve() == self.store.harness.path().resolve()
        baseline = copy.deepcopy(load_harness_state(directory, "global" if global_ else "local"))
        if target:
            proposal = rollback_proposal(target)
            identifier = "refine_" + re.sub(r"\D", "", timestamp())
        else:
            inputs = self.refinement_input(
                sid, instructions=options.get("instructions"), global_=global_
            )
            preparation = {
                "trigger": "auto" if options.get("source") == "auto" else "manual",
                "instructions": options.get("instructions"),
                "scope": "global" if global_ else "local",
                "planningState": self.store.harness.load()
                if global_
                else self.store.harness.merged(sid),
                "history": self.store.harness.history(sid),
                "conversationText": inputs["conversation"],
            }
            extension = self.environment.call(sid, "session_before_refine", preparation)
            if inspect.isawaitable(extension):
                extension = await extension
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError()
            if extension and extension.get("skip"):
                raise RefineSkippedError("Refinement skipped by extension")
            identifier = "refine_" + re.sub(r"\D", "", timestamp())
            if extension and "proposal" in extension:
                proposal = normalize_proposal(extension["proposal"])
            else:
                response, _ = await self.auxiliary(
                    sid,
                    "refinement",
                    PLANNER_PROMPT,
                    inputs,
                )
                self.check_refinement_response(response, "Refinement")
                proposal = normalize_proposal(parse_object(response.text))
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        return {
            "id": identifier,
            "proposal": proposal,
            "baseline_state": baseline,
            "global_": global_,
            "rollback_of": rollback_id,
            "target_directory": directory,
        }

    def count_refinement_turn(self, sid):
        state = self.refinement_state(sid)
        pending = self.store.session(sid).pending_turn or {}
        response_id = pending.get("event_id")
        if pending.get("response", {}).get("metadata", {}).get("stop_reason") in {
            "error",
            "aborted",
        }:
            return
        if response_id and response_id == state.counted_response:
            return
        state.counted_response = response_id
        state.turns_since_review += 1
        state.pending_interval |= (
            state.turns_since_review >= self.store.config(sid).refinement.turn_interval
        )

    def refinement_message_end(self, sid):
        if self.store.session(sid).depth == 0:
            pending = self.store.session(sid).pending_turn or {}
            reason = pending.get("response", {}).get("metadata", {}).get("stop_reason")
            if reason == "aborted":
                # Prime's aborted agent_end path clears scheduled explicit
                # work even when no separate host requestAbort was received.
                self.invalidate_refinement(sid)
                return
            if reason == "error":
                return
            self.count_refinement_turn(sid)
            self.maybe_start_refinement_plan(sid)

    def maybe_start_refinement_plan(self, sid):
        state, session = self.refinement_state(sid), self.store.session(sid)
        if not self.serialized_refinement(sid):
            return
        if self._closing or session.depth or session.paused or session.outcome != Outcome.ACTIVE:
            return
        if state.background or state.claim or state.in_progress:
            return
        # The primary response must have finished. Planning may overlap its tools only.
        if (
            not session.pending_turn
            or self.store.db.execute(
                "SELECT 1 FROM model_attempts a JOIN model_requests r ON r.id=a.request_id "
                "WHERE r.session_id=? AND a.status='running' LIMIT 1",
                (sid,),
            ).fetchone()
        ):
            return
        options = state.pending_request
        if options is None:
            policy = self.store.config(sid).refinement
            if not policy.enabled or state.turns_since_review < policy.turn_interval:
                return
            if (
                state.last_review_at
                and time.time() - state.last_review_at < policy.cooldown_seconds
            ):
                return
        state.pending_request = None
        state.background_options = options
        state.background = asyncio.create_task(
            self.background_refinement_plan(sid, options, state.branch_version),
            name=f"refinement-plan-{sid}",
        )

    async def background_refinement_plan(self, sid, options, branch):
        explicit = options is not None
        reason = None if explicit else "turn_interval"
        try:
            if not explicit:
                review = await self.review_refinement(sid, reason)
                if self._closing or self.refinement_state(sid).branch_version != branch:
                    return {"status": "invalidated", "branch": branch}
                if not review["shouldRefine"]:
                    return {"status": "skip", "branch": branch}
                options = {
                    "instructions": auto_refine_instructions(reason, review),
                    "source": "auto",
                }
            plan = await self.plan_refinement(sid, options)
            if self._closing or self.refinement_state(sid).branch_version != branch:
                return {"status": "invalidated", "branch": branch}
            return {
                "status": "plan",
                "branch": branch,
                "plan": plan,
                "options": options,
                "reason": reason,
            }
        except asyncio.CancelledError:
            return {"status": "invalidated", "branch": branch}
        except Exception as exc:
            # Prime classifies a late rejection against its originating branch
            # before considering extension skips or recoverable planning failures.
            if self._closing or self.refinement_state(sid).branch_version != branch:
                return {"status": "invalidated", "branch": branch}
            if isinstance(exc, RefineSkippedError):
                return {"status": "skip", "branch": branch, "explicit": explicit}
            return {
                "status": "failure",
                "branch": branch,
                "explicit": explicit,
                "options": options,
                "error": str(exc),
            }

    async def refinement_checkpoint(self, sid, *, completed_turn=False):
        state = self.refinement_state(sid)
        if self.store.session(sid).depth != 0:
            return False
        if completed_turn:
            self.count_refinement_turn(sid)
        if not self.serialized_refinement(sid) and not state.draining:
            if state.pending_request:
                options, state.pending_request = state.pending_request, None
                self._start_interactive_refine(sid, options)
            elif not state.auto_task and not state.plan_task and not state.apply_task:
                state.auto_task = asyncio.create_task(self.maybe_auto_refine(sid))
                state.auto_task.add_done_callback(
                    lambda task: self._auto_refine_finished(sid, task)
                )
            return False
        if state.claim:
            await asyncio.shield(state.claim)
            return False  # The owning drain already processed this boundary.
        if not self.refinement_boundary(sid, ignore_planning=state.background is not None):
            return False
        state.claim = asyncio.current_task()
        claim_branch = state.branch_version
        try:
            applied = False
            if background := state.background:
                try:
                    result = await asyncio.shield(background)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    result = {"status": "invalidated", "branch": state.branch_version}
                if state.background is background:
                    state.background = state.background_options = None
                current = result["branch"] == state.branch_version and not self._closing
                if result["status"] == "plan":
                    if current:
                        state.apply_task = asyncio.current_task()
                        try:
                            outcome = await self.apply_refinement_plan(
                                sid, result["plan"], result["options"]
                            )
                            applied = any(edit["applied"] for edit in outcome["appliedEdits"])
                        except Exception as exc:
                            self.store.event(sid, "refine_failed", {"error": str(exc)})
                        finally:
                            state.apply_task = None
                    if current or (not state.pending_request and not state.draining):
                        state.last_review_at, state.turns_since_review = time.time(), 0
                        state.pending_interval = False
                elif result["status"] == "failure":
                    if claim_branch == state.branch_version:
                        state.last_review_at = time.time()
                    if state.draining:
                        state.last_review_at = time.time()
                        state.turns_since_review = 0
                        state.pending_interval = False
                    # Explicit work gets one boundary retry; an interval failure does not.
                    if current and result["explicit"] and not state.pending_request:
                        state.pending_request = result["options"]
                elif result["status"] == "skip" or (
                    result["status"] == "invalidated" and not state.pending_request
                ):
                    state.last_review_at, state.turns_since_review = time.time(), 0
                    state.pending_interval = False
                    if result.get("explicit") and result["status"] == "skip":
                        self.store.event(
                            sid, "refine_failed", {"error": "Refinement skipped by extension"}
                        )
                if state.pending_request is None:
                    return applied
            followup = await self._refinement_checkpoint_after_background(sid)
            return applied or followup
        except asyncio.CancelledError:
            if state.branch_version == claim_branch:
                self.invalidate_refinement(sid)
            raise
        finally:
            state.claim = None

    async def _refinement_checkpoint_after_background(self, sid):
        state = self.refinement_state(sid)
        policy = self.store.config(sid).refinement
        if state.in_progress or not self.refinement_boundary(sid):
            return False
        while state.plan_task or state.apply_task:
            other = state.apply_task or state.plan_task
            if other is asyncio.current_task():
                break
            try:
                await asyncio.shield(other)
            except (Exception, asyncio.CancelledError):
                if asyncio.current_task().cancelling():
                    raise
        options = state.pending_request
        explicit = options is not None
        reason = None
        if options is None:
            if not policy.enabled:
                state.pending_compact = state.pending_interval = False
                state.pending_review = None
                return False
            if not policy.compact:
                state.pending_compact = False
            if (
                state.last_review_at
                and time.time() - state.last_review_at < policy.cooldown_seconds
            ):
                return False
            reason = (
                "compact"
                if state.pending_compact
                else "turn_interval"
                if state.turns_since_review >= policy.turn_interval
                else None
            )
            if reason is None:
                return False
        state.pending_request = None
        branch = state.branch_version
        state.in_progress, state.task = True, asyncio.current_task()
        try:
            if options is None:
                if reason == "compact":
                    state.pending_compact = False
                state.reviewing = True
                try:
                    review = await self.review_refinement(sid, reason)
                finally:
                    state.reviewing = False
                if state.branch_version != branch:
                    return False
                if not review["shouldRefine"]:
                    state.last_review_at, state.turns_since_review = time.time(), 0
                    state.pending_review = None
                    state.pending_interval = False
                    if reason == "compact":
                        state.pending_compact = False
                    return False
                options = {
                    "instructions": auto_refine_instructions(reason, review),
                    "source": "auto",
                }
            state.plan_task = asyncio.current_task()
            try:
                plan = await self.plan_refinement(sid, options)
            finally:
                state.plan_task = None
            if state.branch_version != branch or self._closing:
                return False
            state.apply_task = asyncio.current_task()
            try:
                result = await self.apply_refinement_plan(sid, plan, options)
            finally:
                if state.apply_task is asyncio.current_task():
                    state.apply_task = None
            # The extension completion hook may yield while the branch changes.
            # Automatic review must not stamp the new branch's cooldown/counter.
            if reason and (state.branch_version != branch or self._closing):
                return False
            state.last_review_at, state.turns_since_review = time.time(), 0
            state.pending_review = None
            state.pending_interval = False
            if reason == "compact":
                state.pending_compact = False
            return any(edit["applied"] for edit in result["appliedEdits"])
        except asyncio.CancelledError:
            if state.branch_version == branch:
                self.invalidate_refinement(sid)
            if explicit and not state.draining:
                self.store.event(
                    sid,
                    "refine_failed",
                    {"error": "Refinement cancelled because the session was disposed."},
                )
            raise
        except RefineSkippedError as exc:
            if state.branch_version == branch:
                state.last_review_at, state.turns_since_review = time.time(), 0
                state.pending_interval = False
                if not reason and not state.draining:
                    self.store.event(sid, "refine_failed", {"error": str(exc)})
            return False
        except Exception as exc:
            if state.branch_version == branch:
                state.last_review_at = time.time()
                if options and options.get("source") == "self":
                    state.turns_since_review = 0
                    state.pending_interval = False
                if reason or not state.draining:
                    self.store.event(sid, "refine_failed", {"error": str(exc)})
            return False
        finally:
            if explicit:
                # Prime consumes an explicit boundary round even when an abort
                # or navigation races its completion; automatic rounds differ.
                state.last_review_at, state.turns_since_review = time.time(), 0
                state.pending_interval = False
            state.in_progress, state.task = False, None

    @staticmethod
    def check_refinement_response(response, label):
        reason = response.metadata.get("stop_reason")
        if reason == "length":
            raise ValueError(f"{label} failed: {TRUNCATED_JSON_ERROR}")
        if reason == "error":
            raise ValueError(
                f"{label} failed: {response.metadata.get('error_message') or 'Unknown error'}"
            )

    async def apply_refinement_plan(self, sid, plan, options):
        audit_error = None
        try:
            result = self.store.harness.apply(sid, **plan)
        except HarnessAuditError as exc:
            result, audit_error = exc.result, exc.cause
        source = options.get("source", "user")
        for message in refinement_messages(result, source):
            if audit_error and message["customType"] == "refinement_notice":
                break
            try:
                event = self.store.event(sid, message["customType"], message["details"])
                self.store.add_context(sid, event, [message])
            except Exception:
                self.context.unpersisted_refinement_messages.setdefault(sid, []).append(message)
        if audit_error:
            raise audit_error
        try:
            self.store.event(sid, "refine_complete", result)
            complete = self.environment.call(
                sid,
                "refine_complete",
                {
                    "id": result["id"],
                    "summary": result["summary"],
                    "appliedEdits": sum(edit["applied"] for edit in result["appliedEdits"]),
                    "scope": result["scope"],
                },
            )
            if inspect.isawaitable(complete):
                await complete
        except Exception:
            pass  # A listener cannot turn an already persisted edit into a failure.
        return result

    async def wait_refinement_barrier(self, sid):
        state = self.refinement_state(sid)
        while (
            barrier := state.command_task or state.apply_task
        ) and barrier is not asyncio.current_task():
            try:
                await asyncio.shield(barrier)
            except (Exception, asyncio.CancelledError):
                if asyncio.current_task().cancelling():
                    raise

    async def _wait_refinement_quiescence(self, sid):
        while not self._closing:
            session = self.store.session(sid)
            pending = session.pending_turn if not session.paused else None
            if not pending and sid not in self._transitioning and sid not in self._admitted_turns:
                return
            active = self.tasks.get(sid)
            if active and active is not asyncio.current_task() and sid in self._admitted_turns:
                await asyncio.shield(active)
            else:
                await asyncio.sleep(0.005)
        raise asyncio.CancelledError()

    async def refine(
        self, sid, *, instructions=None, global_=False, rollback_id=None, source="user"
    ):
        """Prime's public refinement: overlapping planning, quiescent serialized apply."""
        if self._closing:
            raise RuntimeError("Cannot refine a disposed session")
        state = self.refinement_state(sid)
        current = asyncio.current_task()
        while state.plan_task or state.apply_task or state.background or state.claim:
            other = state.apply_task or state.plan_task or state.background or state.claim
            if other is current:
                break
            try:
                await asyncio.shield(other)
            except (Exception, asyncio.CancelledError):
                if current.cancelling():
                    raise
            if other is state.background:
                await self._wait_refinement_quiescence(sid)
                if state.background is other:
                    state.background = state.background_options = None
        branch = state.branch_version
        options = {
            "instructions": instructions,
            "global_": global_,
            "rollback_id": rollback_id,
            "source": source,
        }
        state.plan_task = current
        try:
            plan = await self.plan_refinement(sid, options)
        finally:
            if state.plan_task is current:
                state.plan_task = None
            self._wake.set()
        state.apply_task = current
        try:
            await self._wait_refinement_quiescence(sid)
            if self._closing or branch != state.branch_version:
                raise asyncio.CancelledError()
            return await self.apply_refinement_plan(sid, plan, options)
        finally:
            if state.apply_task is current:
                state.apply_task = None
            self._wake.set()

    def _start_interactive_refine(self, sid, options):
        async def run():
            try:
                await self.refine(sid, **options)
            except Exception as exc:
                self.store.event(sid, "refine_failed", {"error": str(exc)})

        # Keep the task owned even before the first coroutine scheduling point.
        state = self.refinement_state(sid)
        state.task = asyncio.create_task(run())

    def _auto_refine_finished(self, sid, task):
        state = self.refinement_state(sid)
        if state.auto_task is task:
            state.auto_task = None
        if not task.cancelled():
            task.exception()
        self._wake.set()
        if not self._closing and not state.draining:
            self._schedule_deferred_auto_refine(sid)

    def _schedule_deferred_auto_refine(self, sid):
        state = self.refinement_state(sid)
        if self._closing or state.draining or state.auto_task or state.reviewing:
            return
        if self.has_pending_refinement(sid) and self.refinement_boundary(sid):
            if not state.pending_compact:
                # Prime consumes the deferred interval flag before dispatch;
                # a below-threshold check must not reschedule itself forever.
                state.pending_interval = False
            state.auto_task = asyncio.create_task(self.maybe_auto_refine(sid))
            state.auto_task.add_done_callback(lambda task: self._auto_refine_finished(sid, task))

    async def maybe_auto_refine(self, sid, reason=None):
        state, session, policy = (
            self.refinement_state(sid),
            self.store.session(sid),
            self.store.config(sid).refinement,
        )
        if self._closing or session.depth or not policy.enabled:
            state.pending_review = None
            state.pending_compact = state.pending_interval = False
            return
        reason = reason or (
            "compact" if state.pending_compact and policy.compact else "turn_interval"
        )
        if (
            state.reviewing
            or state.plan_task
            or state.apply_task
            or not self.refinement_boundary(sid)
        ):
            if reason == "compact":
                state.pending_compact = True
            else:
                state.pending_interval = True
            return
        if not policy.compact:
            state.pending_compact = False
            reason = "turn_interval"
        if (
            reason == "turn_interval"
            and state.turns_since_review < policy.turn_interval
            and not state.pending_review
        ):
            return
        if state.last_review_at and time.time() - state.last_review_at < policy.cooldown_seconds:
            if reason == "compact":
                state.pending_compact = True
            else:
                state.pending_interval = True
            return
        branch, started = state.branch_version, time.time()
        pending = state.pending_review
        if reason == "turn_interval":
            state.pending_interval = False
        state.reviewing = True
        try:
            if pending:
                reason, review = pending
            else:
                review = await self.review_refinement(sid, reason)
            if self._closing or branch != state.branch_version:
                return
            if not review["shouldRefine"]:
                state.pending_compact = False if reason == "compact" else state.pending_compact
                if reason == "compact" and state.turns_since_review >= policy.turn_interval:
                    state.pending_interval = True
                    return
                state.last_review_at, state.turns_since_review = started, 0
                state.pending_interval = False
                return
            if not self.refinement_boundary(sid):
                state.pending_review = (reason, review)
                return
            try:
                await self.refine(
                    sid, instructions=auto_refine_instructions(reason, review), source="auto"
                )
            except RefineSkippedError:
                pass
            except (Exception, asyncio.CancelledError):
                # Prime's interactive _runApprovedRefine consumes its failure
                # and stamps cooldown even if an abort raced the planner.
                state.last_review_at = time.time()
                return
            state.pending_review = None
            state.pending_interval = False
            if reason == "compact":
                state.pending_compact = False
            state.last_review_at, state.turns_since_review = time.time(), 0
        except Exception:
            if branch == state.branch_version:
                state.last_review_at = time.time()
        finally:
            state.reviewing = False
            if not state.auto_task and not self._closing and not state.draining:
                self._schedule_deferred_auto_refine(sid)

    async def drain_refinement(self, sid):
        state = self.refinement_state(sid)
        # Re-read ownership after every await: a public plan can hand off to
        # apply, or a queued request can start as its predecessor settles.
        while tasks := {
            task
            for task in (
                state.command_task,
                state.auto_task,
                state.plan_task,
                state.apply_task,
                state.task,
            )
            if task and task is not asyncio.current_task() and not task.done()
        }:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
        state.draining = True
        try:
            if state.pending_request or state.background or state.claim:
                await self.refinement_checkpoint(sid)
            # Compaction is checked separately, before re-reading live settings
            # for the interval drain. No speculative checkpoint comes first.
            if self.serialized_refinement(sid) and state.pending_compact:
                policy = self.store.config(sid).refinement
                due = (
                    policy.enabled
                    and policy.compact
                    and (
                        not state.last_review_at
                        or time.time() - state.last_review_at >= policy.cooldown_seconds
                    )
                )
                if due:
                    try:
                        await self.refinement_checkpoint(sid)
                    except Exception:
                        pass  # Prime's compaction disposal drain is best effort.
                    finally:
                        state.pending_compact = False
                    return
                state.pending_compact = False
            policy = self.store.config(sid).refinement
            if (
                policy.enabled
                and state.turns_since_review >= policy.turn_interval
                and (
                    not state.last_review_at
                    or time.time() - state.last_review_at >= policy.cooldown_seconds
                )
            ):
                if self.serialized_refinement(sid):
                    await self.refinement_checkpoint(sid)
                else:
                    await self.maybe_auto_refine(sid, "turn_interval")
        finally:
            state.draining = False
