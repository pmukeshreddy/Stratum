from __future__ import annotations

import base64
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
        checksum = hashlib.sha256(data).hexdigest()
        existing = self.store.db.execute(
            "SELECT id FROM artifacts WHERE session_id=? AND sha256=? AND media_type=? AND size=? LIMIT 1",
            (sid, checksum, media_type, len(data)),
        ).fetchone()
        if existing:
            return existing[0]
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
                checksum,
                now(),
                source_event,
            ),
        )
        self.index(sid, aid, data[:16000])
        return aid

    def put_stream(self, sid: str, stream, *, source_event=None, media_type="text/plain") -> str:
        aid = new_id()
        path = self.directory / aid
        digest, size = hashlib.sha256(), 0
        with path.open("xb") as target:
            os.chmod(path, 0o600)
            while chunk := stream.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        existing = self.store.db.execute(
            "SELECT id FROM artifacts WHERE session_id=? AND sha256=? AND media_type=? AND size=? LIMIT 1",
            (sid, digest.hexdigest(), media_type, size),
        ).fetchone()
        if existing:
            path.unlink()  # Only this just-created duplicate blob; existing history is unchanged.
            return existing[0]
        self.store.db.execute(
            "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
            (
                aid,
                sid,
                str(path.relative_to(self.store.directory)),
                media_type,
                size,
                digest.hexdigest(),
                now(),
                source_event,
            ),
        )
        with path.open("rb") as stream:
            self.index(sid, aid, stream.read(16000))
        return aid

    def index(self, sid, aid, data):
        if b"\0" not in data:
            self.store.db.execute(
                "INSERT INTO history_fts(id,session_id,root_id,kind,text) VALUES(?,?,?,?,?)",
                (
                    aid,
                    sid,
                    self.store.session(sid).root_id,
                    "artifact",
                    data.decode(errors="replace"),
                ),
            )

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
        if meta["media_type"] == "application/octet-stream":
            return {"encoding": "base64", "data": base64.b64encode(raw).decode()}
        return (
            json.loads(raw)
            if meta["media_type"] == "application/json"
            else raw.decode(errors="replace")
        )

    def expose(self, sid: str, value: Any, *, source_event=None) -> dict:
        aid = self.put(sid, value, source_event=source_event)
        serialized = encode(value)
        cap = self.store.config(sid).context.result_chars
        result = {
            "artifact_id": aid,
            "preview": serialized[:cap],
            "truncated": len(serialized) > cap,
            "characters": len(serialized),
        }
        if result["truncated"]:
            result["inspection"] = (
                f"Full result: artifacts.load({aid!r}). Prefer selecting fields from your retained Python variable. Do not repeat an unchanged read to recover truncated output."
            )
        return result
