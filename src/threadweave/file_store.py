"""JSON documents and authoritative per-session JSONL event logs.

A process lock serializes writers. A durable write intent makes multi-file commits
recoverable; it is removed once every document and event has reached disk. No
second copy of conversation history or query language is involved.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

KEYS = {
    "usage": ("session_id",),
    "goals": ("session_id",),
    "goal_budgets": ("session_id",),
    "coding_baselines": ("session_id",),
    "provider_continuations": ("event_id",),
    "model_attempts": ("event_id",),
    "candidates": ("child_id",),
    "mutation_workspaces": ("path",),
    "repository_files": ("workspace", "path"),
    "module_bindings": ("workspace", "path", "module", "alias", "symbol"),
    "test_coverage": ("workspace", "test_id", "path", "line"),
    "semantic_evidence": ("id", "model"),
    "semantic_cursors": ("root_id", "model"),
    "request_edges": ("source", "target", "kind"),
    "pending_request_edges": ("session_id", "source", "kind"),
    "conversation_blocks": ("session_id", "event_id"),
}


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path):
    path = Path(path)
    if path.is_dir():
        return
    ensure_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        return
    path.chmod(0o700)
    sync_directory(path.parent)


def atomic_write(path, data):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path, *, repair_tail=False):
    """Only an incomplete final append is recoverable; interior corruption is an error."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("rb") as stream:
        while raw := stream.readline():
            if not raw.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Incomplete final record: {path}")
                offset = stream.tell() - len(raw)
                with path.open("r+b") as target:
                    target.truncate(offset)
                    target.flush()
                    os.fsync(target.fileno())
                break
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"Invalid JSONL record in {path} at byte {stream.tell() - len(raw)}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}")
            records.append(record)
    return records


class FileStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        ensure_directory(self.directory)
        self._mutex = threading.RLock()
        self._lock = (self.directory / ".store.lock").open("a+b")
        os.chmod(self._lock.name, 0o600)
        self._cache = {}
        self._indexes = {}
        self._orders = {}
        self._frames = []
        self._revision = None
        self._sequence = 0
        self.rollback_generation = 0
        with self.transaction():
            if not (self.directory / "store.json").exists():
                atomic_write(self.directory / "store.json", json_bytes({"format": 1}))
            manifest = json.loads((self.directory / "store.json").read_text())
            if manifest.get("format") != 1:
                raise ValueError("Unsupported file-store format")

    def close(self):
        self._lock.close()

    def _key(self, collection, record):
        fields = KEYS.get(collection, ("id",))
        return json.dumps(
            [record.get(field) for field in fields], ensure_ascii=False, separators=(",", ":")
        )

    def _path(self, collection, key):
        if not re.fullmatch(r"[a-z_]+", collection):
            raise ValueError(f"Invalid collection: {collection}")
        return (
            self.directory
            / "records"
            / collection
            / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        )

    def history_path(self, session_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", session_id):
            raise ValueError("Invalid session ID")
        return self.directory / "sessions" / session_id / "messages.jsonl"

    def _load(self, collection):
        if collection not in self._cache:
            if collection == "events":
                rows = [
                    row
                    for path in sorted((self.directory / "sessions").glob("*/messages.jsonl"))
                    for row in read_jsonl(path, repair_tail=True)
                ]
                rows.sort(key=lambda row: row["seq"])
                self._sequence = max((row["seq"] for row in rows), default=0)
            else:
                rows = [
                    json.loads(path.read_text())
                    for path in sorted((self.directory / "records" / collection).glob("*.json"))
                ]
                rows.sort(key=lambda row: row.get("_order", 0))
            self._cache[collection] = {self._key(collection, row): row for row in rows}
            self._orders[collection] = max((row.get("_order", 0) for row in rows), default=0)
        return self._cache[collection]

    def _recover(self):
        pending = self.directory / ".commit.json"
        if pending.exists():
            self._apply(json.loads(pending.read_text()), recovering=True)
            pending.unlink()
            sync_directory(self.directory)

    def _apply(self, commit, *, recovering=False):
        event_groups = {}
        for change in commit["changes"]:
            collection, key, record = change["collection"], change["key"], change["record"]
            if collection == "events":
                event_groups.setdefault(record["session_id"], []).append(record)
                continue
            path = self._path(collection, key)
            if record is None:
                if path.exists():
                    path.unlink()
                    sync_directory(path.parent)
            else:
                atomic_write(path, json_bytes(record))
        for sid, events in event_groups.items():
            path = self.history_path(sid)
            ensure_directory(path.parent)
            # An interrupted commit may already have appended some of these IDs.
            existing = (
                {row["id"] for row in read_jsonl(path, repair_tail=True)} if recovering else set()
            )
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "ab") as stream:
                for event in events:
                    if event["id"] not in existing:
                        stream.write(json_bytes(event))
                        existing.add(event["id"])
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(path.parent)
        atomic_write(self.directory / ".revision", commit["id"].encode())

    @contextmanager
    def transaction(self):
        """Synchronous, nestable transactions. Never hold this boundary across await."""
        with self._mutex:
            outer = not self._frames
            if outer:
                fcntl.flock(self._lock, fcntl.LOCK_EX)
                try:
                    self._recover()
                    revision = self.directory / ".revision"
                    current = revision.read_text() if revision.exists() else ""
                    if current != self._revision:
                        self._cache.clear()
                        self._indexes.clear()
                        self._orders.clear()
                        self._revision = current
                except BaseException:
                    fcntl.flock(self._lock, fcntl.LOCK_UN)
                    raise
            frame = {}
            self._frames.append(frame)
            try:
                yield
                if outer and frame:
                    changes = [
                        {
                            "collection": collection,
                            "key": key,
                            "record": self._cache[collection].get(key),
                        }
                        for collection, key in frame
                    ]
                    commit = {"id": uuid4().hex, "changes": changes}
                    atomic_write(self.directory / ".commit.json", json_bytes(commit))
                    try:
                        self._apply(commit)
                        (self.directory / ".commit.json").unlink()
                        sync_directory(self.directory)
                    except BaseException:
                        self._cache.clear()
                        self._indexes.clear()
                        self._orders.clear()
                        self._revision = None
                        raise
                    self._revision = commit["id"]
                elif not outer:
                    for key, original in frame.items():
                        self._frames[-2].setdefault(key, original)
            except BaseException:
                self._indexes.clear()
                if frame:
                    self.rollback_generation += 1
                for (collection, key), original in frame.items():
                    if collection in self._cache:
                        if original is None:
                            self._cache[collection].pop(key, None)
                        else:
                            self._cache[collection][key] = original
                raise
            finally:
                self._frames.pop()
                if outer:
                    fcntl.flock(self._lock, fcntl.LOCK_UN)

    def _change(self, collection, key, record):
        rows = self._load(collection)
        self._frames[-1].setdefault((collection, key), rows.get(key))
        old = rows.get(key)
        for (name, field), index in self._indexes.items():
            if name == collection:
                if old is not None:
                    index.get(old.get(field), set()).discard(key)
                if record is not None:
                    index.setdefault(record.get(field), set()).add(key)

        if record is None:
            rows.pop(key, None)
        else:
            rows[key] = record

    def select(self, collection, *, where=None, order=(), limit=None, fields=None, **match):
        with self.transaction():
            documents = self._load(collection)
            key_fields = KEYS.get(collection, ("id",))
            if all(field in match for field in key_fields):
                keys = [self._key(collection, match)]
            elif match:
                field = "session_id" if "session_id" in match else next(iter(match))
                token = (collection, field)
                if token not in self._indexes:
                    index = {}
                    for key, record in documents.items():
                        index.setdefault(record.get(field), set()).add(key)
                    self._indexes[token] = index
                keys = self._indexes[token].get(match[field], ())
                keys = sorted(
                    keys, key=lambda key: documents[key].get("_order", documents[key].get("seq", 0))
                )
            else:
                keys = documents.keys()
            rows = [
                documents[key]
                for key in keys
                if key in documents
                and all(documents[key].get(field) == value for field, value in match.items())
                and (where is None or where(documents[key]))
            ]
            for field, reverse in reversed(order):
                rows.sort(
                    key=lambda row: (row.get(field) is not None, row.get(field)), reverse=reverse
                )
            if limit is not None:
                rows = rows[:limit]
            if fields is not None:
                rows = [{field: row.get(field) for field in fields} for row in rows]
            else:
                rows = [
                    {key: value for key, value in row.items() if key != "_order"} for row in rows
                ]
            return copy.deepcopy(rows)

    def first(self, collection, **options):
        options["limit"] = 1
        return next(iter(self.select(collection, **options)), None)

    def count(self, collection, **options):
        options.setdefault("fields", ())
        return len(self.select(collection, **options))

    def insert(self, collection, record, *, on_conflict="error"):
        with self.transaction():
            record = copy.deepcopy(record)
            if collection == "code_evidence":
                record.setdefault("id", uuid4().hex)
            rows = self._load(collection)
            key = self._key(collection, record)
            old = rows.get(key)
            if old is not None:
                if on_conflict == "ignore":
                    return
                if collection == "events" or on_conflict != "replace":
                    raise ValueError(f"Duplicate {collection} record: {key}")
            if collection == "events":
                sequence = record.get("seq", self._sequence + 1)
                if sequence <= self._sequence:
                    raise ValueError("Event sequence must increase")
                self._sequence = record["seq"] = sequence
            else:
                if old:
                    record["_order"] = old["_order"]
                else:
                    self._orders[collection] += 1
                    record["_order"] = self._orders[collection]
            # Validate before changing the in-memory view or creating a commit intent.
            record = json.loads(json_bytes(record))
            self._change(collection, key, record)

    def insert_many(self, collection, records, **options):
        with self.transaction():
            for record in records:
                self.insert(collection, record, **options)

    def update(self, collection, changes, *, where=None, **match):
        if collection == "events":
            raise ValueError("Events are append-only")
        count = 0
        with self.transaction():
            rows = self._load(collection)
            fields = KEYS.get(collection, ("id",))
            keys = (
                [self._key(collection, match)]
                if all(field in match for field in fields)
                else list(rows)
            )
            for key in keys:
                row = rows.get(key)
                if row is None:
                    continue
                if all(row.get(k) == v for k, v in match.items()) and (where is None or where(row)):
                    updated = {**row, **(changes(row) if callable(changes) else changes)}
                    if self._key(collection, updated) != key:
                        raise ValueError("Record identity cannot be changed")
                    updated = json.loads(json_bytes(updated))
                    self._change(collection, key, updated)
                    count += 1

        return count

    def delete(self, collection, *, where=None, **match):
        if collection == "events":
            raise ValueError("Events are append-only")
        with self.transaction():
            rows = self._load(collection)
            fields = KEYS.get(collection, ("id",))
            keys = (
                [self._key(collection, match)]
                if all(field in match for field in fields)
                else list(rows)
            )
            for key in keys:
                row = rows.get(key)
                if row is None:
                    continue
                if all(row.get(k) == v for k, v in match.items()) and (where is None or where(row)):
                    self._change(collection, key, None)


# Harness state is also accessed by kernel processes that do not own a Store.
_directory_locks = WeakValueDictionary()
_directory_guard = threading.Lock()
_held_directories = threading.local()


@contextmanager
def directory_lock(directory):
    directory = Path(directory).resolve()
    with _directory_guard:
        mutex = _directory_locks.setdefault(directory, threading.RLock())
    with mutex:
        held = getattr(_held_directories, "paths", None)
        if held is None:
            held = _held_directories.paths = set()
        if directory in held:
            yield
            return
        ensure_directory(directory)
        fd = os.open(directory / ".harness.lock", os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held.add(directory)
            try:
                yield
            finally:
                held.remove(directory)
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
