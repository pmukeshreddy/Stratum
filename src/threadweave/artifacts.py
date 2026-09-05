from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import new_id, now
from .storage import Store, encode


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Artifacts:
    def __init__(self, store: Store):
        self.store = store
        self.directory = store.directory / "artifacts"
        self.directory.mkdir(exist_ok=True, mode=0o700)

    def put(self, sid: str, value: Any, *, source_event=None) -> str:
        return self.put_bytes(sid, encode(value).encode(), "application/json", source_event)

    def put_bytes(self, sid: str, data: bytes, media_type="text/plain", source_event=None) -> str:
        aid = new_id()
        path = self.directory / aid
        atomic_write(path, data)
        self.store.db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
            (
                aid,
                sid,
                str(path.relative_to(self.store.directory)),
                media_type,
                len(data),
                hashlib.sha256(data).hexdigest(),
                now(),
                source_event,
            ),
        )
        return aid

    def put_stream(self, sid: str, stream, *, source_event=None) -> str:
        aid = new_id()
        path = self.directory / aid
        digest, size = hashlib.sha256(), 0
        with path.open("xb") as target:
            while chunk := stream.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        self.store.db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
            (
                aid,
                sid,
                str(path.relative_to(self.store.directory)),
                "text/plain",
                size,
                digest.hexdigest(),
                now(),
                source_event,
            ),
        )
        return aid

    def metadata(self, sid: str, aid: str) -> dict:
        row = self.store.db.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        if not row:
            raise KeyError(f"Unknown artifact: {aid}")
        # Tree members share artifacts. A fork can read its explicit ancestry.
        roots = self.store.history_roots(sid)
        if self.store.session(row["session_id"]).root_id not in roots:
            raise PermissionError("Artifact belongs to another session tree")
        return dict(row)

    def read(self, sid: str, aid: str, *, offset=0, limit=16000) -> dict:
        meta = self.metadata(sid, aid)
        path = self.store.directory / meta["path"]
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(min(limit, 64000))
        return {
            "artifact_id": aid,
            "offset": offset,
            "next_offset": offset + len(data),
            "total_bytes": meta["size"],
            "text": data.decode("utf-8", errors="replace"),
        }

    def load(self, sid: str, aid: str) -> Any:
        meta = self.metadata(sid, aid)
        raw = (self.store.directory / meta["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError("Artifact checksum mismatch")
        return json.loads(raw) if meta["media_type"] == "application/json" else raw.decode()

    def expose(self, sid: str, value: Any, *, source_event=None) -> dict:
        aid = self.put(sid, value, source_event=source_event)
        serialized = encode(value)
        cap = self.store.config(sid).context.result_chars
        return {
            "artifact_id": aid,
            "preview": serialized[:cap],
            "truncated": len(serialized) > cap,
            "characters": len(serialized),
        }
