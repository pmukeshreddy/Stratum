"""Prime's refinement message conversion and bounded conversation serializer."""

import json

from .harness import compact_text, js_json, js_length, js_slice
from .models import now

HARNESS_PREFIX = "The persistent memories produced across this session so far:\n\n<harness_state>\n"
HARNESS_SUFFIX = "\n</harness_state>"
SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
SUMMARY_SUFFIX = "\n</summary>"


def custom_message(kind, content, details, *, display=False):
    return {
        "role": "custom",
        "customType": kind,
        "content": content,
        "details": details,
        "display": display,
        "timestamp": now(),
    }


def harness_digest_message(digest):
    return custom_message(
        "harness_digest", HARNESS_PREFIX + digest + HARNESS_SUFFIX, {"digest": digest}
    )


def refinement_messages(result, source):
    details = {
        "refinementId": result["id"],
        "summary": result["summary"],
        "scope": result.get("scope") or "local",
        "edits": result["appliedEdits"],
    }
    if result.get("rollbackOf"):
        details["rollbackOf"] = result["rollbackOf"]
    messages = [
        custom_message(
            "refinement_outcome", "Refinement complete: " + result["summary"], details, display=True
        )
    ]
    applied = [edit for edit in result["appliedEdits"] if edit["applied"]]
    if applied:
        lines = [compact_text(result["summary"])]
        for edit in applied:
            entry = edit.get("after") or edit.get("before") or {}
            scope = entry.get("scope") or result.get("scope") or "local"
            lines.append(
                f"- {edit['action']} {edit['kind']} [{scope}:{edit['id']}] "
                f"{entry.get('title', edit['id'])}: {compact_text(entry.get('content', ''))}"
            )
        messages.append(
            custom_message(
                "refinement_notice",
                f"[{source}-refinement]\n\n" + "\n".join(lines),
                {**details, "source": source},
            )
        )
    return messages


def convert_to_llm(messages):
    result = []
    for message in messages:
        role = message.get("role")
        if role == "custom":
            if message.get("customType") in {
                "session_slash_command",
                "session_slash_command_result",
                "compaction_outcome",
                "refinement_outcome",
            }:
                continue
            result.append({"role": "user", "content": message["content"]})
        elif role == "compactionSummary":
            digest = message.get("harnessDigest")
            text = HARNESS_PREFIX + digest + HARNESS_SUFFIX + "\n\n" if digest else ""
            result.append(
                {
                    "role": "user",
                    "content": text + SUMMARY_PREFIX + message["summary"] + SUMMARY_SUFFIX,
                }
            )
        elif role in {"user", "assistant", "tool", "toolResult"}:
            result.append(message)
    return result


def serialize_conversation(messages):
    parts = []
    for message in convert_to_llm(messages):
        content = message.get("content") or ""
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        text = [b["text"] for b in blocks if b.get("type") == "text"]
        role = message["role"]
        if role == "assistant":
            thinking = [b["thinking"] for b in blocks if b.get("type") == "thinking"]
            if thinking:
                parts.append("[Assistant thinking]: " + "\n".join(thinking))
            if text:
                parts.append("[Assistant]: " + "\n".join(text))
            calls = [b for b in blocks if b.get("type") == "toolCall"]
            for call in message.get("tool_calls", []):
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({"name": function["name"], "arguments": arguments})
            if calls:
                rendered = []
                for call in calls:
                    args = ", ".join(f"{k}={js_json(v)}" for k, v in call["arguments"].items())
                    rendered.append(f"{call['name']}({args})")
                parts.append("[Assistant tool calls]: " + "; ".join(rendered))
        elif text:
            joined = "".join(text)
            if role == "user":
                parts.append("[User]: " + joined)
            else:
                if js_length(joined) > 2000:
                    joined = (
                        js_slice(joined, 0, 2000)
                        + f"\n\n[... {js_length(joined) - 2000} more characters truncated]"
                    )
                parts.append("[Tool result]: " + joined)
    return "\n\n".join(parts)


def refinement_user_prompt(inputs, *, review=False):
    fields = list(inputs)
    if review:
        fields = ["trigger", "current_harness_state", "refinement_history", "conversation"]
    parts = []
    for field in fields:
        value = inputs[field]
        if field == "trigger":
            value = f"{value['reason']}; {value['turnsSinceLastReview']} assistant turns since last auto-refine review"
        parts.append(f"<{field}>\n{value}\n</{field}>")
    parts.append(
        "Return shouldRefine=true when the trajectory contains evidence useful to this session's future turns. Prefer local harness edits for current task progress, temporary blockers, and current-run coordination. Ask for global refinement only for durable cross-session lessons or explicitly project-qualified facts likely to be reused in future sessions."
        if review
        else "Return only JSON edits. If no useful edit is justified, return an empty edits array with a rationale."
    )
    return "\n\n".join(parts)
