"""Bounded snapshot streams. No whole-value JSON tree or serialized bytes buffer."""

import base64
import hashlib
import json
import os
import tempfile
import types
from pathlib import Path


class SnapshotLimit(ValueError):
    pass


class CappedFile:
    def __init__(self, directory, limit, chunk_bytes, budget):
        fd, name = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=directory)
        self.path = Path(name)
        self.stream = os.fdopen(fd, "wb")
        self.limit, self.chunk_bytes, self.budget = limit, chunk_bytes, budget
        self.size, self.max_write = 0, 0
        self.has_paths = False
        self.digest = hashlib.sha256()

    def write(self, data):
        view = memoryview(data).cast("B")
        length = len(view)
        if self.size + length > self.limit or self.budget[0] + length > self.limit:
            raise SnapshotLimit("Snapshot serialized-size limit exceeded; use a recovery recipe")
        self.size += length
        self.budget[0] += length
        for offset in range(0, length, self.chunk_bytes):
            chunk = view[offset : offset + self.chunk_bytes]
            self.stream.write(chunk)
            self.digest.update(chunk)
            self.max_write = max(self.max_write, len(chunk))
        return length

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()

    def discard(self):
        self.stream.close()
        self.path.unlink(missing_ok=True)


def json_text(sink, value, chunk_bytes):
    sink.write(b'"')
    # ensure_ascii has at most twelve output bytes per input Unicode character.
    for offset in range(0, len(value), max(1, chunk_bytes // 12)):
        piece = value[offset : offset + max(1, chunk_bytes // 12)]
        sink.write(json.dumps(piece, ensure_ascii=True)[1:-1].encode("ascii"))
    sink.write(b'"')


def write_packed(sink, value, chunk_bytes, seen=None):
    """Stream the existing typed JSON codec without first constructing pack(value)."""
    from .kernel_api import Record, snapshot_handle

    seen = set() if seen is None else seen
    if value is None or type(value) in (str, int, float, bool):
        sink.write(b'["scalar",')
        if isinstance(value, str):
            json_text(sink, value, chunk_bytes)
        else:
            sink.write(json.dumps(value, allow_nan=False).encode())
        sink.write(b"]")
        return
    if id(value) in seen:
        raise ValueError("Cyclic value requires a recovery recipe")
    if isinstance(value, Record):
        sink.write(b'["record",')
        # Record is dict-like: do not copy the entire mapping.
        kind = "dict"
    else:
        kind = type(value).__name__
    if type(value) in (dict, list, tuple, set, frozenset) or isinstance(value, Record):
        seen.add(id(value))
        sink.write((f'["{kind}",[').encode())
        try:
            for index, item in enumerate(value.items() if kind == "dict" else value):
                if index:
                    sink.write(b",")
                if kind == "dict":
                    sink.write(b"[")
                    write_packed(sink, item[0], chunk_bytes, seen)
                    sink.write(b",")
                    write_packed(sink, item[1], chunk_bytes, seen)
                    sink.write(b"]")
                else:
                    write_packed(sink, item, chunk_bytes, seen)
            sink.write(b"]]")
        finally:
            seen.remove(id(value))
        if isinstance(value, Record):
            sink.write(b"]")
    elif type(value) is bytes:
        sink.write(b'["bytes","')
        step = max(3, chunk_bytes // 4 * 3)
        for offset in range(0, len(value), step):
            sink.write(base64.b64encode(memoryview(value)[offset : offset + step]))
        sink.write(b'"]')
    elif isinstance(value, Path):
        sink.has_paths = True
        sink.write(b'["path",')
        json_text(sink, str(value), chunk_bytes)
        sink.write(b"]")
    elif isinstance(value, types.ModuleType):
        sink.write(b'["module",')
        json_text(sink, value.__name__, chunk_bytes)
        sink.write(b"]")
    elif handle := snapshot_handle(value):
        # Built-in handles are small records; their identity stays owner-bound.
        for piece in json.JSONEncoder(allow_nan=False).iterencode(handle):
            sink.write(piece.encode())
    else:
        raise TypeError(f"{type(value).__module__}.{type(value).__name__} needs a recovery recipe")


def rebind_paths(source, sink, old, new, chunk_bytes):
    """Rewrite typed Path prefixes in canonical snapshot JSON, without loading its tree."""
    before = json.dumps(str(old), ensure_ascii=True)[1:-1].encode()
    after = json.dumps(str(new), ensure_ascii=True)[1:-1].encode()
    buffer = b""
    quoted = False

    def peek(count):
        nonlocal buffer
        if len(buffer) < count:
            buffer += source.read(max(chunk_bytes, count - len(buffer)))
        return buffer[:count]

    def take(count):
        nonlocal buffer
        part, buffer = buffer[:count], buffer[count:]
        return part

    while block := peek(chunk_bytes):
        if quoted:
            positions = [i for token in (b'"', b"\\") if (i := block.find(token)) >= 0]
            position = min(positions, default=len(block))
            if position:
                sink.write(take(position))
            elif block[0] == ord('"'):
                sink.write(take(1))
                quoted = False
            else:
                peek(2)
                sink.write(take(2))
        else:
            peek(max(16, len(before) + 16))
            prefix = next((p for p in (b'["path","', b'["path", "') if buffer.startswith(p)), None)
            if prefix:
                sink.write(take(len(prefix)))
                path_prefix = peek(len(before) + 1)
                if path_prefix.startswith(before) and path_prefix[
                    len(before) : len(before) + 1
                ] in (b"/", b'"'):
                    take(len(before))
                    sink.write(after)
                quoted = True
            elif buffer.startswith(b'"'):
                sink.write(take(1))
                quoted = True
            else:
                # Stop at the next possible type tag/string, including across chunks.
                block = peek(chunk_bytes)
                positions = [i for token in (b"[", b'"') if (i := block.find(token, 1)) >= 0]
                sink.write(take(min(positions, default=len(block))))
