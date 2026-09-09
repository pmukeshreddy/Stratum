"""Token-aware structured context selection; never clip an unresolved requirement."""

import hashlib
import json
import re

from .models import HarnessError
from .storage import encode
from .tokenization import encoder, estimate

FIELDS = (
    "objective",
    "established_facts",
    "decisions",
    "completed_work",
    "unresolved_requirements",
    "active_hypotheses",
    "blockers",
    "next_actions",
    "important_references",
)
PROTECTED = (
    "objective",
    "unresolved_requirements",
    "blockers",
    "next_actions",
    "active_hypotheses",
)
ALIASES = {
    "task": "objective",
    "facts": "established_facts",
    "state": "established_facts",
    "unresolved_work": "unresolved_requirements",
    "hypotheses": "active_hypotheses",
    "blockers_errors": "blockers",
    "failed_approaches_and_reasons": "decisions",
    "evidence_ids": "important_references",
    "retained_repl_names": "important_references",
    "child_handles": "important_references",
}


def policy_tokens(policy, field, model):
    explicit = getattr(policy, field + "_tokens")
    if explicit is not None:
        return explicit
    # Legacy configuration remains readable, but all selection measures actual tokens.
    return getattr(policy, field + "_chars") // (4 if encoder(model) else 1)


def structured(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = {"established_facts": value.splitlines()}
    if not isinstance(value, dict):
        value = {"established_facts": [value]} if value else {}
    result = {key: [] for key in FIELDS}
    for key, content in (value or {}).items():
        key = ALIASES.get(key, key)
        key = key if key in result else "established_facts"
        items = content if isinstance(content, list) else [content]
        for item in items:
            if item and item not in result[key]:
                result[key].append(item)
    return result


def pending_ledger(value):
    """Stable identifiers let the host preserve work omitted by a later summary."""
    result = []
    for field, items in structured(value).items():
        if field not in PROTECTED or field == "objective":
            continue
        for item in items:
            identifier = item.get("id") if isinstance(item, dict) else None
            result.append(
                {
                    "id": identifier
                    or hashlib.sha256(encode([field, item]).encode()).hexdigest()[:20],
                    "field": field,
                    "content": item,
                }
            )
    return result


def merge_summary(previous, update, *, source_events=()):
    """An omitted pending item is unchanged, never implicitly resolved."""
    if isinstance(update, str):
        update = json.loads(update)
    controls = {"resolved_items", "pending_updates"}
    incoming = structured({k: v for k, v in update.items() if k not in controls})
    old = structured(previous)
    allowed = set(source_events)
    ledger = pending_ledger(previous)

    def supported(change):
        return (
            isinstance(change, dict)
            and isinstance(change.get("source_events"), list)
            and bool(change["source_events"])
            and all(isinstance(e, str) and e in allowed for e in change["source_events"])
            and isinstance(change.get("reason"), str)
            and bool(change["reason"].strip())
        )

    resolved = {c.get("id") for c in update.get("resolved_items", []) if supported(c)}
    # Extractive/model summaries can repeat prior pending items. An evidenced
    # resolution applies to both copies, otherwise merging resurrects finished work.
    for item in pending_ledger(incoming):
        if item["id"] in resolved:
            incoming[item["field"]].remove(item["content"])
    changes = {
        c.get("id"): c
        for c in update.get("pending_updates", [])
        if supported(c) and isinstance(c.get("text"), str) and c["text"].strip()
    }
    for item in ledger:
        if item["id"] in resolved:
            old[item["field"]].remove(item["content"])
        elif item["id"] in changes:
            index = old[item["field"]].index(item["content"])
            old[item["field"]][index] = {"id": item["id"], "text": changes[item["id"]]["text"]}
    for field in PROTECTED:
        if field == "objective":
            incoming[field] = incoming[field] or old[field]
        else:
            incoming[field] = old[field] + [
                item for item in incoming[field] if item not in old[field]
            ]
    return incoming


def concise_reference(item, model):
    if isinstance(item, dict):
        if set(item) - {"id", "reference", "purpose", "retrieve", "path"}:
            return False
        if not all(isinstance(v, str) for v in item.values()):
            return False
    elif not isinstance(item, str):
        return False
    # Never clip arbitrary strings; oversized detail is covered by the full archive.
    return estimate(item, model) <= (192 if encoder(model) else 768)


def summary_budget(value, *, budget, model, reference):
    source = structured(value)
    result = {key: source[key] if key in PROTECTED else [] for key in FIELDS}
    result["important_references"] = [reference]

    def render():
        return encode({key: items for key, items in result.items() if items})

    if estimate(render(), model) > budget:
        raise HarnessError(
            "runtime",
            "compaction_capacity",
            "Unresolved requirements, hypotheses and next actions exceed the summary token "
            "budget. Source context retained; increase context.summary_tokens.",
        )
    # Reserved fields are already allocated. Admit whole lower-priority facts in stable order.
    for key in ("decisions", "established_facts", "important_references", "completed_work"):
        for item in source[key]:
            if key == "important_references" and not concise_reference(item, model):
                continue
            if item in result[key]:
                continue
            result[key].append(item)
            if estimate(render(), model) > budget:
                result[key].pop()
    return render()


def extractive_summary(previous, blocks):
    """Fallback scans the COMPLETE region. Suspected pending work stays verbatim."""

    def lines(value):
        if isinstance(value, dict):
            # Raw outputs and source code are evidence, not unresolved task instructions.
            # Keep their durable references; explicit structured pending/error fields
            # and authored user/assistant observations still enter the protected ledger.
            for key, content in value.items():
                if key in {
                    "stdout",
                    "stderr",
                    "traceback",
                    "code",
                    "arguments",
                    "value",
                    "preview",
                    "tests",
                }:
                    continue
                yield from lines(content)
        elif isinstance(value, list):
            for content in value:
                yield from lines(content)
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                # Tool observations often carry a label before a JSON envelope. Do not
                # mistake keys such as error:null for an unresolved error spanning it all.
                match = re.match(r"^[^\n{}\[\]]*?:\s*([\[{].*)$", value, re.S)
                if match:
                    try:
                        parsed = json.loads(match[1])
                    except ValueError:
                        pass
                    else:
                        yield from lines(parsed)
                        return
                # Sentence/line boundaries retain semantic units, including late requirements.
                yield from re.split(r"\n|(?<=[.!?])\s+", value)
            else:
                if parsed != value:
                    yield from lines(parsed)

    result = structured(previous)
    for block in blocks:
        for message in block["messages"]:
            text = message.get("content") or encode(message.get("tool_calls", []))
            for line in lines(text):
                key = "established_facts"
                if re.search(
                    r"\b(unresolved|pending|still must|must not|require[sd]?|constraint|todo)\b",
                    line,
                    re.I,
                ):
                    key = "unresolved_requirements"
                elif re.search(r"\b(blocker|error|failed|failure)\b", line, re.I):
                    key = "blockers"
                elif re.search(r"\b(hypothes[ie]s|hypotheses)\b", line, re.I):
                    key = "active_hypotheses"
                elif re.search(r"\b(next action|next step)\b", line, re.I):
                    key = "next_actions"
                if line and line not in result[key]:
                    result[key].append(line)
    return result


def prefix_end(text, fits):
    """Find a token-fitting chunk boundary, preferring complete lines/words."""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if fits(text[:mid]):
            low = mid
        else:
            high = mid - 1
    if low == len(text):
        return low
    boundary = max(text.rfind("\n", 0, low), text.rfind(" ", 0, low))
    return boundary + 1 if boundary >= low // 2 else low
