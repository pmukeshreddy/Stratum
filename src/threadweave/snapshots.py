"""Trusted local snapshot blobs. Never load snapshots from an untrusted source.

Explicit codecs remain preferred. Cloudpickle is restricted to user procedures,
classes and instances; opaque OS resources are rejected. Mutable objects still
need serialization to detect in-place changes; immutable values can be cached.
"""

import hashlib
import inspect
import io
import json
import signal
import socket
import subprocess
import time
import types
from contextlib import contextmanager

import cloudpickle

from .artifacts import atomic_write


class SnapshotTimeout(ValueError):
    pass


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise SnapshotTimeout("Snapshot time bound exceeded; use an artifact/recovery recipe")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class SafeProcedurePickler(cloudpickle.CloudPickler):
    def reducer_override(self, value):
        if isinstance(value, (io.IOBase, socket.socket, subprocess.Popen)) or (
            inspect.isgenerator(value)
            or inspect.iscoroutine(value)
            or inspect.isframe(value)
            or type(value).__module__ == "_thread"
        ):
            raise ValueError("Opaque OS/execution resource requires a recovery recipe")
        return super().reducer_override(value)


class SnapshotBlobs:
    def __init__(self, directory):
        self.directory = directory / "values"
        self.directory.mkdir(exist_ok=True, mode=0o700)
        self.cache = {}
        self.stats = {}

    def encode(self, name, value, pack):
        immutable = type(value) in {str, bytes, int, float, bool, type(None)}
        cached = self.cache.get(name)
        if immutable and cached and cached[0] is value:
            self.stats["cache_hits"] += 1
            return cached[1], cached[2]
        try:
            encoded = pack(value)
            data = json.dumps(encoded, allow_nan=False).encode()
            codec = "json"
        except TypeError:
            if (
                not isinstance(value, (types.FunctionType, type))
                and type(value).__module__ != "__session__"
                and (type(value).__module__, type(value).__name__)
                not in {("array", "array"), ("numpy", "ndarray")}
            ):
                raise
            buffer = io.BytesIO()
            SafeProcedurePickler(buffer, protocol=5).dump(value)
            data, codec = buffer.getvalue(), "cloudpickle"
            encoded = None
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Variable exceeds 16 MiB snapshot bound; use artifacts/recipes")
        self.stats["serialized_bytes"] += len(data)
        if len(data) > 65536 or codec == "cloudpickle":
            digest = hashlib.sha256(data).hexdigest()
            path = self.directory / digest
            if not path.exists():
                atomic_write(path, data)
                self.stats["written_bytes"] += len(data)
            encoded = ["blob", {"sha256": digest, "codec": codec, "bytes": len(data)}]
        if immutable:
            self.cache[name] = (value, encoded, len(data))
        return encoded, len(data)

    def decode(self, record, unpack):
        if record[0] != "blob":
            return unpack(record)
        metadata = record[1]
        digest = metadata["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid snapshot blob identity")
        data = (self.directory / digest).read_bytes()
        if len(data) > 16 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Snapshot blob checksum/size mismatch")
        if metadata["codec"] == "json":
            return unpack(json.loads(data))
        if metadata["codec"] == "cloudpickle":
            return cloudpickle.loads(data)
        raise ValueError("Unknown snapshot blob codec")

    def begin(self):
        self.stats = {"serialized_bytes": 0, "written_bytes": 0, "cache_hits": 0}
        return time.monotonic()
