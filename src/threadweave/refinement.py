"""Serialized Prime-style refinement checkpoints, independent of trajectory persistence."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .auxiliary import AuxiliaryServices, refinement_provider_config
from .harness import (
    format_harness_state,
    infer_scope,
    load_harness_state,
    normalize_proposal,
    refinement_notice,
    rollback_proposal,
)
from .models import Outcome, new_id
from .refinement_evidence import opportunity, opportunity_key

# Re-exported for provider integrations that require structured inference settings.
__all__ = ["RefinementServices", "refinement_provider_config"]

PLANNER_PROMPT = """You are Buffalo's continual harness refinement subsystem. Improve reusable behavior from the recent trajectory, current harness and refinement history. Return precise create/update/delete edits. The base system prompt is immutable.
Kinds: prompt (narrow supplemental behavioral instructions), memory (durable facts/preferences/decisions/failures), skill (an existing reusable Python callable), subagent (purpose, instructions and when to delegate).
Skill create/update MUST include reference={"type":"python","import":"package.module","callable":"function","call_pattern":"await function(...)"} and arguments describing accepted inputs, required fields, defaults and constraints. {} is valid only for no external inputs. Never embed executable code; create a normal source/module artifact before learning its reference.
Subagents execute through the native runtime: compose a task from the spec and use handle = await rlm('task'). Admission returns a handle, never the answer. Children send evidence with await agent_message.send(message, receiver_role='parent'). Use await rlm.list_subagents() and messaging for follow-ups. Do not invent a child engine or named registry.
Default scope is local to this persisted session. Global refinement is explicit and only for durable cross-session lessons, preferences, reusable skills/subagents or explicitly project-qualified reusable facts. Global entries are read-only during local refinement: never update/delete them; create a local override instead. All edits apply to the requested store. Strip display-only local:/global: prefixes from IDs.
Choose the smallest useful component, including correcting or deleting wrong learned behavior. Do not edit source files directly. No useful lesson is a valid result: return edits=[].
Prioritize a correction to CURRENT UNFINISHED WORK. Read in_task_opportunity and the actual observed results, not just the root's claim of success. Identify the unresolved mistake/decision and explain a concrete next action and validation that would resolve it. Check the original instructions independently: do not merely repeat the root's interpretation or its successful workflow. For a failed source/format/type/behavior check, address that specific failure and distinguish a defective candidate from a defective test. Never invent evaluator feedback or task requirements. No official post-completion grading is available here.
If the root already fixed the issue and there is no remaining use for the lesson, return edits=[] for local refinement. Generic restatements of existing instructions, finished-task summaries and already-followed checklists are redundant. Global/human-requested durable learning remains valid when explicitly requested. For an actionable edit, keep its content focused and include application={"status":"actionable|post_correction","issue":"unresolved issue","evidence_events":["observed event ids when available"],"next_action":"specific remaining work","validation":"observable check"}. Application is a proposed use, not proof that the root used the edit.
Return JSON only: {"summary":"one sentence","rationale":"trajectory evidence","expectedOutcome":"improvement and validation","application":{"status":"actionable|post_correction","issue":"specific unresolved issue","evidence_events":[],"next_action":"concrete remaining action","validation":"observable check"},"edits":[{"action":"create|update|delete","kind":"prompt|memory|skill|subagent","id":"stable bare id, optional for create","title":"required for create/update","content":"required for create/update","path":"optional group","reference":{},"arguments":{},"metadata":{},"reason":"why"}]}.
"""
REVIEW_PROMPT = """You are Buffalo's automatic continual harness review gate. Decide whether this checkpoint should run refinement. Approve fresh observed failure or an unresolved decision when a reusable correction can improve remaining work. Name that issue and a concrete next action. Independently inspect the actual contract and evidence rather than echoing the root's conclusions. Reject already-corrected failures with no further use, completed-work summaries, generic checklists, one-off noise and unsupported hypotheses. A completion attempt does not justify a review by itself. Default refinement is local; global is for durable cross-session or explicitly project-qualified lessons.
Return JSON only: {"shouldRefine":true|false,"rationale":"short reason","instructions":"optional concise instructions when approved"}."""


def parse_object(text):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        result = json.loads(text[start : end + 1])
    except ValueError as exc:
        raise ValueError("Refinement model did not return a complete valid JSON object") from exc
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
    pending_plan: tuple | None = None
    pending_compact: bool = False
    pending_interval: bool = False
    branch_version: int = 0
    task: asyncio.Task | None = None
    background: asyncio.Task | None = None
    background_options: dict | None = None
    claim: asyncio.Task | None = None
    counted_response: str | None = None


class RefinementServices(AuxiliaryServices):
    def current_refinement_opportunity(self, sid):
        return opportunity(self.store, sid, self.context.original_task(sid)["messages"])

    def refinement_state(self, sid):
        if sid not in self._refinement_states:
            self._refinement_states[sid] = Checkpoint()
        return self._refinement_states[sid]

    def refinement_status(self, sid):
        state = self.refinement_state(sid)
        return {
            "pending": state.pending_request is not None,
            "in_flight": state.in_progress
            or state.background is not None
            or state.claim is not None,
        }

    def request_refinement(
        self, sid, *, instructions=None, global_=False, rollback_id=None, source="self"
    ):
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be str or None")
        if not isinstance(global_, bool):
            raise TypeError("global_ must be bool")
        session = self.store.session(sid)
        if session.depth != 0:
            return {
                "scheduled": False,
                "reason": "Refinement is available only to the root session",
            }
        if source == "self" and not session.pending_turn:
            return {
                "scheduled": False,
                "reason": "no active turn; refine can only be requested while a turn is running",
            }
        if session.outcome not in (Outcome.ACTIVE, Outcome.COMPLETED):
            return {"scheduled": False, "reason": "Session is stopped"}
        state = self.refinement_state(sid)
        previous = state.pending_request or state.background_options or {}
        request = {
            "instructions": instructions
            if instructions is not None
            else previous.get("instructions"),
            "global_": global_ or previous.get("global_", False),
            "rollback_id": rollback_id,
            "source": "self",
        }
        if state.in_progress or state.pending_plan or state.background:
            state.branch_version += 1
            state.pending_plan = None
            if state.background:
                state.background.cancel()
        state.pending_request = request
        self.store.event(sid, "refine_scheduled", request)
        if source == "self":
            self.maybe_start_refinement_plan(sid)
        self._wake.set()
        return {
            "scheduled": True,
            "note": "Refinement runs when the current turn ends; applied edits enter your context as a refinement notice and you resume automatically. Continue working normally.",
        }

    def has_pending_refinement(self, sid):
        state, policy = self.refinement_state(sid), self.store.config(sid).refinement
        if (
            state.pending_request is not None
            or state.pending_plan is not None
            or state.background is not None
        ):
            return True
        return (
            policy.enabled
            and (
                (state.pending_review and state.pending_review[0] == "actionable_evidence")
                or not state.last_review_at
                or time.time() - state.last_review_at >= policy.cooldown_seconds
            )
            and (
                state.pending_review is not None
                or state.pending_interval
                or (policy.compact and state.pending_compact)
            )
        )

    def invalidate_refinement(self, sid):
        state = self.refinement_state(sid)
        state.branch_version += 1
        state.turns_since_review = 0
        state.counted_response = None
        state.pending_request = state.pending_review = state.pending_plan = None
        state.pending_compact = state.pending_interval = False
        for task in (state.task, state.background, state.claim):
            if task and task is not asyncio.current_task():
                task.cancel()

    def refinement_compacted(self, sid):
        if self.store.session(sid).depth == 0:
            self.refinement_state(sid).pending_compact = True
            self._wake.set()

    def refinement_boundary(self, sid, *, ignore_planning=False):
        session = self.store.session(sid)
        if (
            self._closing
            or session.paused
            or session.outcome not in (Outcome.ACTIVE, Outcome.COMPLETED)
        ):
            return False
        if session.pending_turn and not session.pending_turn.get("context_committed"):
            return False
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
        # Current context already includes tool results, child findings and compaction.
        pending = session.pending_turn or {}
        response = pending.get("response", {})
        active_message = (
            [
                {
                    "role": "assistant",
                    "content": response.get("text", ""),
                    "tool_calls": response.get("actions", []),
                }
            ]
            if response and not pending.get("context_committed")
            else []
        )
        trajectory = json.dumps(
            [
                *self.context.original_task(sid)["messages"],
                *([{"role": "user", "content": session.summary}] if session.summary else []),
                *[m for block in session.context for m in block["messages"]],
                *active_message,
            ],
            ensure_ascii=False,
            default=str,
        )
        harness = (
            self.store.harness.load() if global_ and not review else self.store.harness.merged(sid)
        )
        history = self.store.harness.history(sid)[-20:]
        evidence = {
            "in_task_opportunity": self.current_refinement_opportunity(sid),
            "conversation": trajectory[-(40_000 if review else 80_000) :],
            "current_harness_state": format_harness_state(
                harness, entry_limit=40, content_limit=240
            ),
            "refinement_history": [
                {k: r.get(k) for k in ("id", "summary", "expectedOutcome", "rollbackOf")}
                | {
                    "edits": [
                        f"{'applied' if e['applied'] else 'failed'} {e['action']} {e['kind']}:{e['id']}"
                        for e in r["appliedEdits"]
                    ]
                }
                for r in history
            ],
            "scope_policy": "Requested refinement scope: global. Do not persist session-only progress or temporary blockers globally."
            if global_
            else "Requested refinement scope: local. Global entries are read-only context; create a local override when needed.",
        }
        if review:
            evidence["trigger"] = {
                "reason": reason,
                "turnsSinceLastReview": self.refinement_state(sid).turns_since_review,
            }
        if instructions:
            evidence["user_refine_instructions"] = instructions
        return evidence

    def prepare_refinement_opportunity(self, sid, *, completing=False):
        """Allow one review of fresh evidence while there is time to act on it."""
        session, state = self.store.session(sid), self.refinement_state(sid)
        policy = self.store.config(sid)
        if session.depth or not policy.refinement.enabled or self.has_pending_refinement(sid):
            return
        evidence = self.current_refinement_opportunity(sid)
        key = opportunity_key(evidence)
        if not key or any(
            e["payload"].get("key") == key
            for e in self.store.iter_events(sid, kind="refinement_opportunity")
        ):
            return
        durations = self.store.db.execute(
            "SELECT ended_at-started_at FROM model_requests WHERE session_id=? "
            "AND purpose='agent' AND status='completed' ORDER BY started_at DESC LIMIT 8",
            (sid,),
        ).fetchall()
        turn_time = max((r[0] for r in durations), default=1)
        remaining = policy.limits.wall_seconds - self._elapsed(session.root_id)
        if policy.limits.max_turns - session.turns < 2 or remaining < 3 * turn_time:
            self.store.event(
                sid,
                "refinement_opportunity_deferred",
                {"key": key, "reason": "Preserve time for work, validation and final response"},
            )
            return
        self.store.event(
            sid,
            "refinement_opportunity",
            {"key": key, "boundary": "pre_completion" if completing else "work", **evidence},
        )
        # Fresh evidence is independent of the periodic-review cooldown. A declined
        # review is consumed, so an unchanged issue cannot add ceremonial turns.
        state.pending_review = ("actionable_evidence", None)

    async def review_refinement(self, sid, reason):
        response, _ = await self.auxiliary(
            sid,
            "refinement_review",
            REVIEW_PROMPT,
            self.refinement_input(sid, review=True, reason=reason),
        )
        value = parse_object(response.text)
        review = {
            "shouldRefine": value.get("shouldRefine") is True,
            "rationale": value.get("rationale", "No rationale provided."),
            "instructions": value.get("instructions", ""),
        }
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
            if not path.exists():  # noqa: ASYNC240 - bounded local metadata check before atomic apply
                raise ValueError(f"Refinement state file not found: {path}")
            directory = path.parent
            global_ = directory.resolve() == self.store.harness.path().resolve()
        baseline = copy.deepcopy(load_harness_state(directory, "global" if global_ else "local"))
        live_opportunity = self.current_refinement_opportunity(sid)
        key = opportunity_key(live_opportunity)
        if key and not any(
            e["payload"].get("key") == key
            for e in self.store.iter_events(sid, kind="refinement_opportunity")
        ):
            self.store.event(
                sid,
                "refinement_opportunity",
                {"key": key, "boundary": "requested", **live_opportunity},
            )
        application = {}
        if target:
            proposal = rollback_proposal(target)
        else:
            response, _ = await self.auxiliary(
                sid,
                "refinement",
                PLANNER_PROMPT,
                self.refinement_input(
                    sid, instructions=options.get("instructions"), global_=global_
                ),
            )
            value = parse_object(response.text)
            proposal = normalize_proposal(value)
            application = value.get("application", {})
            if not isinstance(application, dict):
                application = {}
        return {
            "id": "refine_" + new_id(),
            "proposal": proposal,
            "baseline_state": baseline,
            "global_": global_,
            "rollback_of": rollback_id,
            "target_directory": directory,
            "application": application,
            "opportunity": live_opportunity,
        }

    def count_refinement_turn(self, sid):
        state = self.refinement_state(sid)
        pending = self.store.session(sid).pending_turn or {}
        response_id = pending.get("event_id")
        if response_id and response_id == state.counted_response:
            return
        state.counted_response = response_id
        state.turns_since_review += 1
        state.pending_interval |= (
            state.turns_since_review >= self.store.config(sid).refinement.turn_interval
        )

    def refinement_message_end(self, sid):
        if self.store.session(sid).depth == 0:
            self.count_refinement_turn(sid)
            self.maybe_start_refinement_plan(sid)

    def maybe_start_refinement_plan(self, sid):
        state, session = self.refinement_state(sid), self.store.session(sid)
        if self._closing or session.depth or session.paused or session.outcome != Outcome.ACTIVE:
            return
        if state.background or state.claim or state.in_progress or state.pending_plan:
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
                    "instructions": f"Automatic refinement checkpoint: {reason}.\n{review['rationale']}\n{review.get('instructions', '')}",
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
        if state.claim:
            await asyncio.shield(state.claim)
            return False  # The owning drain already processed this boundary.
        if not self.refinement_boundary(sid, ignore_planning=state.background is not None):
            return False
        state.claim = asyncio.current_task()
        try:
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
                if current and result["status"] == "plan":
                    state.pending_plan = (result["plan"], result["options"], result["reason"])
                elif current and result["status"] == "failure":
                    state.last_review_at = time.time()
                    self.store.event(
                        sid, "refine_failed", {"error": result["error"], "phase": "background"}
                    )
                    # Explicit work gets one boundary retry; an interval failure does not.
                    if result["explicit"] and not state.pending_request:
                        state.pending_request = result["options"]
                elif current:
                    state.last_review_at, state.turns_since_review = time.time(), 0
                    state.pending_interval = False
                if state.pending_request is None and state.pending_plan is None:
                    return False
            return await self._refinement_checkpoint_after_background(sid)
        except asyncio.CancelledError:
            self.invalidate_refinement(sid)
            raise
        finally:
            state.claim = None

    async def _refinement_checkpoint_after_background(self, sid):
        state = self.refinement_state(sid)
        policy = self.store.config(sid).refinement
        if state.in_progress or not self.refinement_boundary(sid):
            return False
        ready = state.pending_plan
        options = state.pending_request or (ready[1] if ready else None)
        reason = ready[2] if ready else None
        if options is None:
            if not policy.enabled:
                state.pending_compact = state.pending_interval = False
                state.pending_review = None
                return False
            if not policy.compact:
                state.pending_compact = False
            if (
                not (state.pending_review and state.pending_review[0] == "actionable_evidence")
                and state.last_review_at
                and time.time() - state.last_review_at < policy.cooldown_seconds
            ):
                return False
            reason = (
                state.pending_review[0]
                if state.pending_review
                else "compact"
                if state.pending_compact
                else "turn_interval"
                if state.pending_interval
                else None
            )
            if reason is None:
                return False
        state.pending_request = None
        branch = state.branch_version
        state.in_progress, state.task = True, asyncio.current_task()
        try:
            if options is None:
                review = (
                    state.pending_review[1]
                    if state.pending_review and state.pending_review[1] is not None
                    else await self.review_refinement(sid, reason)
                )
                if state.branch_version != branch:
                    return False
                if not review["shouldRefine"]:
                    state.last_review_at, state.turns_since_review = time.time(), 0
                    state.pending_review = None
                    state.pending_interval = False
                    if reason == "compact":
                        state.pending_compact = False
                    return False
                state.pending_review = (reason, review)
                if not self.refinement_boundary(sid):
                    return False
                options = {
                    "instructions": f"Automatic refinement checkpoint: {reason}.\n{review['rationale']}\n{review.get('instructions', '')}",
                    "source": "auto",
                }
            plan = ready[0] if ready else await self.plan_refinement(sid, options)
            if state.branch_version != branch or self._closing:
                return False
            if not self.refinement_boundary(sid):
                # Preserve the exact plan; apply-time baseline checks detect disk changes.
                state.pending_plan = (plan, options, reason)
                return False
            state.pending_plan = None
            application = plan.pop("application", {})
            evidence = plan.pop("opportunity", {})
            result = self.store.harness.apply(sid, **plan)
            result.update(application=application, opportunity=evidence)
            notice = refinement_notice(result, options.get("source", "self"), expand=True)
            self.store.event(sid, "refine_complete", result)
            if notice:
                event = self.store.event(
                    sid,
                    "refinement_notice",
                    {
                        "content": notice,
                        "refinement_id": result["id"],
                        "expanded": True,
                        "application": application,
                    },
                )
                self.store.add_context(sid, event, [{"role": "user", "content": notice}])
            state.last_review_at, state.turns_since_review = time.time(), 0
            state.pending_review = None
            state.pending_interval = False
            if reason == "compact":
                state.pending_compact = False
            return bool(notice)
        except asyncio.CancelledError:
            if state.branch_version == branch:
                self.invalidate_refinement(sid)
            raise
        except Exception as exc:
            if state.branch_version == branch:
                state.last_review_at = time.time()
                if options and options.get("source") == "self":
                    state.turns_since_review = 0
                    state.pending_interval = False
                self.store.event(sid, "refine_failed", {"error": str(exc)})
            return False
        finally:
            state.in_progress, state.task = False, None
