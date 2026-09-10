"""Rebuildable in-memory BM25 index over JSONL events and artifact metadata."""

import json
import math
import re
from collections import Counter, defaultdict


class HistorySearch:
    def __init__(self, store):
        self.store = store
        self.documents = {}
        self.postings = defaultdict(dict)
        self.cursor = 0
        self.total_length = 0
        self.rollback_generation = store.records.rollback_generation

    def _add(self, identifier, sid, root, kind, text, ordinal):
        if identifier in self.documents:
            return
        terms = Counter(re.findall(r"\w+", text.casefold()))
        length = sum(terms.values())
        self.documents[identifier] = {
            "id": identifier,
            "session_id": sid,
            "root_id": root,
            "kind": kind,
            "text": text,
            "length": length,
            "ordinal": ordinal,
        }
        self.total_length += length
        for term, frequency in terms.items():
            self.postings[term][identifier] = frequency

    def search(self, roots, terms, *, kind=None, session_id=None, limit=100):
        with self.store.transaction():
            if self.rollback_generation != self.store.records.rollback_generation:
                self.documents.clear()
                self.postings.clear()
                self.cursor = self.total_length = 0
                self.rollback_generation = self.store.records.rollback_generation
            events = self.store.records.select(
                "events", where=lambda row: row["seq"] > self.cursor, order=(("seq", False),)
            )
            for row in events:
                self._add(
                    row["id"],
                    row["session_id"],
                    row["root_id"],
                    row["type"],
                    json.dumps(row["payload"], ensure_ascii=False),
                    row["seq"],
                )
                self.cursor = row["seq"]
            for row in self.store.records.select("artifacts"):
                if row.get("search_text") and row["id"] not in self.documents:
                    root = self.store.session(row["session_id"]).root_id
                    self._add(
                        row["id"],
                        row["session_id"],
                        root,
                        "artifact",
                        row["search_text"],
                        self.cursor,
                    )
        count = len(self.documents)
        average = self.total_length / max(1, count)
        scores = defaultdict(float)
        for term in set(t.casefold() for t in terms):
            posting = self.postings.get(term, {})
            inverse = math.log(1 + (count - len(posting) + 0.5) / (len(posting) + 0.5))
            for identifier, frequency in posting.items():
                doc = self.documents[identifier]
                if (
                    doc["root_id"] not in roots
                    or kind
                    and doc["kind"] != kind
                    or session_id
                    and doc["session_id"] != session_id
                ):
                    continue
                scores[identifier] += (
                    inverse
                    * frequency
                    * 2.2
                    / (frequency + 1.2 * (0.25 + 0.75 * doc["length"] / max(1, average)))
                )
        result = []
        for identifier in sorted(scores, key=lambda item: -scores[item])[:limit]:
            doc = self.documents[identifier]
            positions = [doc["text"].casefold().find(term.casefold()) for term in terms]
            start = max(0, min((p for p in positions if p >= 0), default=0) - 120)
            result.append(
                {key: doc[key] for key in ("id", "session_id", "kind", "ordinal")}
                | {
                    "excerpt": doc["text"][start : start + 900],
                    "lexical_score": -scores[identifier],
                }
            )
        return result
