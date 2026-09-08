"""Token-aware structured context selection; never clip an unresolved requirement."""

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
            for content in value.values():
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


def state_excerpt(entry, budget, model, *, explicit):
    content = entry["content"]
    body = content.get("text") or content.get("code") or encode(content)
    if not isinstance(body, str):
        body = encode(body)
    record = {
        "id": entry["id"],
        "kind": entry["kind"],
        "version": entry["version"],
        "title": entry["title"],
        "selected_because": "explicitly selected for this invocation"
        if explicit
        else "task/context relevance",
        "excerpt": "",
        "additional_content": True,
        "reference": f"{entry['kind']}://{entry['id']}",
        "retrieve": f"harness.get({entry['kind']!r}, {entry['id']!r}, global_={entry.get('owner_id') is None})",
    }
    end = prefix_end(body, lambda part: estimate({**record, "excerpt": part}, model) <= budget)
    if not end and body:
        return None
    record.update(excerpt=body[:end], additional_content=end < len(body))
    return record
