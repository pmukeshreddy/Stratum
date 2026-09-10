"""One-time conversion of existing stores. SQLite is never used after import.

The old database is read in a consistent snapshot and left untouched. A completion
marker is written last, so an interrupted import is safely retried at next startup.
"""

import array
import json
import sqlite3
from pathlib import Path

from .file_store import FileStore, atomic_write, directory_lock, json_bytes

JSON_FIELDS = {
    "configs": ("body",),
    "sessions": ("body",),
    "usage": ("body",),
    "events": ("payload", "usage"),
    "actions": ("arguments", "result"),
    "repository_files": ("body",),
    "coding_baselines": ("body",),
    "checkpoints": ("manifest",),
    "edits": ("body",),
    "experiments": ("body",),
    "experiment_runs": ("body",),
    "benchmark_measurements": ("body",),
    "final_verifications": ("body",),
    "routing_decisions": ("body",),
    "failure_memories": ("body",),
    "candidates": ("body",),
    "process_jobs": ("body",),
    "code_evidence": ("body",),
    "compactions": ("source_events",),
    "provider_continuations": ("items",),
    "model_requests": ("inbound",),
    "model_attempts": ("usage", "failure"),
    "conversation_blocks": ("messages",),
}
COLLECTIONS = set(JSON_FIELDS) | {
    "reservations",
    "messages",
    "artifacts",
    "goals",
    "schedules",
    "goal_budgets",
    "mutation_workspaces",
    "mutation_windows",
    "module_bindings",
    "test_coverage",
    "semantic_evidence",
    "semantic_cursors",
    "request_edges",
    "pending_request_edges",
}


def import_if_needed(directory):
    directory = Path(directory)
    database = directory / "history.sqlite3"
    if not database.exists():
        return
    with directory_lock(directory):
        marker = directory / ".legacy-imported.json"
        if marker.exists():
            return
        records = FileStore(directory)
        try:
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("BEGIN")
                tables = {
                    r[0]
                    for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                with records.transaction():
                    for collection in sorted(tables & COLLECTIONS):
                        ordering = "seq" if collection == "events" else "rowid"
                        for source in connection.execute(
                            f'SELECT rowid AS _legacy_id,* FROM "{collection}" ORDER BY {ordering}'
                        ):
                            row = dict(source)
                            legacy_id = row.pop("_legacy_id")
                            for field in JSON_FIELDS.get(collection, ()):
                                if row.get(field) is not None:
                                    row[field] = json.loads(row[field])
                            if collection == "code_evidence":
                                row["id"] = f"imported_{legacy_id}"
                            if collection == "semantic_evidence":
                                values = array.array("f")
                                values.frombytes(row["vector"])
                                row["vector"] = values.tolist()
                            if collection == "artifacts":
                                path = directory / row["path"]
                                if path.is_file():
                                    with path.open("rb") as stream:
                                        data = stream.read(16000)
                                    if b"\0" not in data:
                                        row["search_text"] = data.decode(errors="replace")
                            records.insert(collection, row, on_conflict="ignore")
                # Move local refinement receipts into the same format used by new runs.
                from .harness import append_refinement_history, load_refinement_history

                seen = {}
                for event in records.select(
                    "events", type="harness_refinement", order=(("seq", False),)
                ):
                    path = directory / "sessions" / event["session_id"] / "harness"
                    if path not in seen:
                        seen[path] = {row["id"] for row in load_refinement_history(path, "local")}
                    result = event["payload"]
                    if result["id"] not in seen[path]:
                        append_refinement_history(path, result)
                        seen[path].add(result["id"])
                atomic_write(marker, json_bytes({"source": database.name, "format": 1}))
        finally:
            records.close()
