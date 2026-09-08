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
