"""Committed conversation and runtime history, independent of the live context window."""

import json


class TrajectoryHistory:
    def trajectory(
        self, sid, *, limit=1000, max_chars=None, char_budget=80000, include_bookkeeping=True
    ):
        from .storage import encode

        session = self.session(sid)
        # A model response and its tools become visible together. Streaming and
        # action receipts remain private until the owning turn has committed.
        cutoff = None
        pending = session.pending_turn
        if pending and not pending.get("context_committed"):
            cutoff = self.event_by_id(pending["event_id"])["seq"]
        running = self.db.execute(
            "SELECT MIN(e.seq) FROM model_attempts a JOIN events e ON e.id=a.event_id "
            "JOIN model_requests r ON r.id=a.request_id WHERE r.session_id=? AND r.purpose='agent' AND a.status='running'",
            (sid,),
        ).fetchone()[0]
        if running:
            cutoff = min(cutoff, running) if cutoff else running
        ignored = {
            "model_stream",
            "model_response",
            "model_invocation_started",
            "resource_usage",
            "context_usage_anchor",
            "context_estimate",
            "model_routing",
            "subscription_model_selected",
            "agent_message_sent",
            "user_intervention",
            "refine_scheduled",
            "refinement_review",
            "refine_complete",
            "refine_failed",
            "harness_refinement",
            "harness_digest",
        }
        if not include_bookkeeping:
            # These receipts duplicate the committed work below. Keep them in the ledger,
            # but do not spend reviewer context on scheduling and accounting machinery.
            ignored.update(
                {
                    "kernel_snapshot",
                    "agent_operating_path",
                    "verification_result",
                    "action_fingerprint",
                    "environment_action_started",
                    "environment_action_finished",
                    "session_transition",
                    "mutation_observation_started",
                    "mutation_observation_finished",
                    "execution_input_consumed",
                    "turn_completed",
                    "environment_prepare",
                    "environment_prepared",
                }
            )
        records = []
        rows = self.db.execute(
            "SELECT e.*,b.messages AS committed_messages FROM events e LEFT JOIN conversation_blocks b "
            "ON b.event_id=e.id AND b.session_id=e.session_id WHERE e.session_id=? "
            "AND (? IS NULL OR e.seq<? OR b.messages IS NOT NULL) ORDER BY e.seq DESC LIMIT 4000",
            (sid, cutoff, cutoff),
        )
        # Build complete records first, then select a chronological bounded suffix.
        for row in rows:
            event = self._event_row(row)
            base = {k: event[k] for k in ("id", "seq", "timestamp", "session_id", "type")}
            block = []
            if row["committed_messages"] is not None:
                from .refinement_context import convert_to_llm

                for message in convert_to_llm(json.loads(row["committed_messages"])):
                    role = message["role"]
                    if message.get("content"):
                        block.append(
                            {
                                **base,
                                "role": "tool-result" if role == "tool" else role,
                                "body": message["content"],
                                "tool_call_id": message.get("tool_call_id"),
                            }
                        )
                    for call in message.get("tool_calls", []):
                        block.append(
                            {
                                **base,
                                "role": "tool",
                                "body": encode(call["function"]),
                                "tool_call_id": call["id"],
                            }
                        )
                if event["type"] == "model_response":
                    reasoning = event["payload"].get("metadata", {}).get("reasoning_summary")
                    if reasoning:
                        block.insert(
                            0,
                            {
                                **base,
                                "role": "assistant",
                                "type": "reasoning_summary",
                                "body": reasoning,
                            },
                        )
            elif event["type"] == "task_admitted":
                block.append({**base, "role": "user", "body": event["payload"]["instruction"]})
            elif event["type"] == "context_compaction":
                block.append({**base, "role": "summary", "body": event["payload"]["summary"]})
            elif event["type"] not in ignored:
                # Include commands, repository reads, verifier outputs and state
                # transitions, even when they are not explicit model messages.
                role = (
                    "tool"
                    if event["type"] == "tool_call"
                    else "tool-result"
                    if event["type"]
                    in {"tool_result", "python_result", "coding_command", "execution_result"}
                    else "system"
                )
                block.append({**base, "role": role, "body": encode(event["payload"])})
            records.extend(reversed(block))
        if not any(r["type"] == "task_admitted" for r in records):
            records.append(
                {
                    "id": self.events(sid, after=0, limit=500)[0]["id"],
                    "seq": 0,
                    "timestamp": session.created_at,
                    "session_id": sid,
                    "type": "task",
                    "role": "user",
                    "body": session.instruction,
                }
            )
        result, remaining = [], char_budget
        for record in records:
            if len(result) >= limit or remaining <= 0:
                break
            body = record["body"] if isinstance(record["body"], str) else encode(record["body"])
            cap = min(max_chars or remaining, remaining)
            clipped = body[:cap] if max_chars else body[-cap:]
            result.append({**record, "body": clipped, "truncated": len(clipped) < len(body)})
            remaining -= len(clipped)
        return list(reversed(result))
