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
        existing = self.store.records.first(
            "artifacts",
            session_id=sid,
            sha256=checksum,
            media_type=media_type,
            size=len(data),
            limit=1,
            fields=("id",),
        )
        if existing:
            return existing["id"]
        aid = new_id()
        path = self.directory / aid
        atomic_write(path, data)
        self.store.records.insert(
            "artifacts",
            {
                "id": aid,
                "session_id": sid,
                "path": str(path.relative_to(self.store.directory)),
                "media_type": media_type,
                "size": len(data),
                "sha256": checksum,
                "created_at": now(),
                "source_event": source_event,
            },
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
        existing = self.store.records.first(
            "artifacts",
            session_id=sid,
            sha256=digest.hexdigest(),
            media_type=media_type,
            size=size,
            limit=1,
            fields=("id",),
        )
        if existing:
            path.unlink()  # Only this just-created duplicate blob; existing history is unchanged.
            return existing["id"]
        self.store.records.insert(
            "artifacts",
            {
                "id": aid,
                "session_id": sid,
                "path": str(path.relative_to(self.store.directory)),
                "media_type": media_type,
                "size": size,
                "sha256": digest.hexdigest(),
                "created_at": now(),
                "source_event": source_event,
            },
        )
        with path.open("rb") as stream:
            self.index(sid, aid, stream.read(16000))
        return aid

    def index(self, sid, aid, data):
        if b"\0" not in data:
            self.store.records.update(
                "artifacts", {"search_text": data.decode(errors="replace")}, id=aid
            )

    def metadata(self, sid: str, aid: str) -> dict:
        row = self.store.records.first("artifacts", id=aid)
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
            data = stream.read(min(limit, 65536))
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
        from .tokenization import estimate

        config = self.store.config(sid)
        available = max(
            128, int((config.context.max_tokens - config.provider.max_output_tokens) * 0.65)
        )
        low, high = 0, min(cap, len(serialized))
        while low < high:
            middle = (low + high + 1) // 2
            if estimate(serialized[:middle], config.provider.model) <= available:
                low = middle
            else:
                high = middle - 1
        cap = low
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
            result["retrieval"] = {
                "artifact_id": aid,
                "offset": 0,
                "limit": 65536,
                "path": str(self.store.directory / self.metadata(sid, aid)["path"]),
            }
        return result
