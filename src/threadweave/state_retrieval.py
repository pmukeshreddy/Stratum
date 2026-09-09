"""Bounded relevance selection of durable lessons; no unconditional memory dump."""

import re

from .storage import encode

STOP = set("the a an to of in for and or is are be with this that it on from use task fix".split())


def terms(text):
    return {
        word
        for word in re.findall(r"[a-z0-9_]+", text.lower())
        if len(word) > 2 and word not in STOP
    }


def relevant_state(store, sid, *, limit=8):
    session = store.session(sid)
    query = terms(session.instruction + " " + encode(session.context[-3:]))
    selected = set(session.selected_state)
    ranked = []
    for entry in store.states(sid):
        title = terms(entry["title"])
        content = terms(encode(entry["content"]))
        score = 3 * len(query & title) + len(query & content)
        if entry["id"] in selected:
            score += 100
        if score:
            ranked.append((score, entry.get("created_at", 0), entry))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in ranked[:limit]]


def state_overview(entries, *, model=None, token_budget=20000, per_kind=40):
    """Stable, kind-balanced previews alongside (not instead of) relevance retrieval."""
    from .context import token_bound

    kinds = ("memory", "prompt_note", "skill", "subagent_spec")
    groups = {
        kind: sorted(
            (entry for entry in entries if entry["kind"] == kind and not entry.get("deleted")),
            key=lambda entry: (entry.get("created_at", 0), entry["id"]),
        )
        for kind in kinds
    }
    overview = {
        "entries": [],
        "totals": {kind: len(group) for kind, group in groups.items()},
        "omitted": {kind: len(group) for kind, group in groups.items()},
    }

    def preview(value, cap=240):
        return re.sub(r"\s+", " ", value if isinstance(value, str) else encode(value))[:cap]

    for index in range(per_kind):
        for kind, group in groups.items():
            if index >= len(group):
                continue
            entry = group[index]
            content = entry["content"]
            item = {
                "id": entry["id"],
                "kind": kind,
                "version": entry["version"],
                "title": preview(entry["title"]),
                "scope": "session" if entry["owner_id"] else "global",
                "owner_id": entry["owner_id"],
                "content_preview": preview(
                    content.get("text")
                    or content.get("instruction")
                    or content.get("description")
                    or content.get("code")
                    or content
                ),
                "metadata": {
                    key: preview(content[key])
                    for key in (
                        "name",
                        "description",
                        "path",
                        "arguments",
                        "reference",
                        "inputs",
                        "required_permissions",
                        "validation_status",
                        "metadata",
                    )
                    if key in content
                },
                "intended_effect": preview(entry.get("intended_effect", "")),
            }
            overview["entries"].append(item)
            overview["omitted"][kind] -= 1
            if token_bound(overview, model) > token_budget:
                overview["entries"].pop()
                overview["omitted"][kind] += 1
    if token_bound(overview, model) > token_budget:
        raise ValueError("Harness overview budget cannot fit kind counts")
    return overview
