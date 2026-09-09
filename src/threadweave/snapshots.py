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
    def __init__(self, directory, policy=None):
        from .models import KernelStatePolicy

        self.policy = policy or KernelStatePolicy()
        self.directory = directory / "values"
        self.directory.mkdir(exist_ok=True, mode=0o700)
        self.cache = {}
        self.shadows = {}
        self.array_shadows = {}
        self.stats = {}

    def encode(self, name, value, pack):
        immutable = type(value) in {str, bytes, int, float, bool, type(None)}
        scalars = {str, bytes, int, float, bool, type(None)}
        flat = type(value) in {list, dict} and all(
            type(item) in scalars
            for item in (value if type(value) is list else (*value.keys(), *value.values()))
        )
        shadow = self.shadows.get(name)
        array = (
            type(value).__module__ == "numpy"
            and type(value).__name__ == "ndarray"
            and not value.dtype.hasobject
            and value.flags.c_contiguous
            and value.nbytes <= self.policy.mutable_cache_bytes
        )
        saved_array = self.array_shadows.get(name)
        if array and saved_array:
            import numpy as np

            prior = saved_array[0]
            if (
                value.dtype == prior.dtype
                and value.shape == prior.shape
                and np.array_equal(value.view(np.uint8), prior.view(np.uint8))
            ):
                self.stats["mutable_cache_hits"] += 1
                return saved_array[1], saved_array[2]

        def identical(a, b):
            # Python equality deliberately conflates True/1/1.0. Check types
            # too: recovery must reproduce the actual scalar values/types.
            if len(a) != len(b) or a != b:
                return False
            if type(a) is list:
                return all(type(x) is type(y) for x, y in zip(a, b, strict=True))
            return all(
                type(key) is type(other_key) and type(value) is type(other_value)
                for (key, value), (other_key, other_value) in zip(a.items(), b.items(), strict=True)
            )

        if flat and shadow and type(value) is type(shadow[0]) and identical(value, shadow[0]):
            self.stats["mutable_cache_hits"] += 1
            return shadow[1], shadow[2]
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
                not in {
                    ("array", "array"),
                    ("numpy", "ndarray"),
                    ("pandas.core.frame", "DataFrame"),
                    ("pandas.core.series", "Series"),
                }
            ):
                raise
            buffer = io.BytesIO()
            SafeProcedurePickler(buffer, protocol=5).dump(value)
            data, codec = buffer.getvalue(), "cloudpickle"
            encoded = None
        if len(data) > self.policy.artifact_bytes:
            raise ValueError(
                "Variable exceeds artifact size budget; register a reconstruction recipe"
            )
        self.stats["serialized_bytes"] += len(data)
        if len(data) > self.policy.inline_bytes or codec == "cloudpickle":
            digest = hashlib.sha256(data).hexdigest()
            path = self.directory / digest
            if not path.exists():
                atomic_write(path, data)
                self.stats["written_bytes"] += len(data)
            encoded = ["blob", {"sha256": digest, "codec": codec, "bytes": len(data)}]
        if immutable:
            self.cache[name] = (value, encoded, len(data))
        if flat and len(data) <= self.policy.mutable_cache_bytes:
            # Exact scalar-container equality, not mutable identity or a hash.
            # Nested/opaque mutations deliberately take the full codec path.
            self.shadows[name] = (value.copy(), encoded, len(data))
        else:
            self.shadows.pop(name, None)
        if array:
            self.array_shadows[name] = (value.copy(), encoded, len(data))
        else:
            self.array_shadows.pop(name, None)
        return encoded, len(data)

    def offload(self, record):
        if record[0] == "blob":
            return record
        data = json.dumps(record, allow_nan=False).encode()
        digest = hashlib.sha256(data).hexdigest()
        path = self.directory / digest
        if not path.exists():
            atomic_write(path, data)
            self.stats["written_bytes"] += len(data)
        return ["blob", {"sha256": digest, "codec": "json", "bytes": len(data)}]

    def decode(self, record, unpack):
        if record[0] != "blob":
            return unpack(record)
        metadata = record[1]
        digest = metadata["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid snapshot blob identity")
        data = (self.directory / digest).read_bytes()
        if len(data) > self.policy.artifact_bytes or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Snapshot blob checksum/size mismatch")
        if metadata["codec"] == "json":
            return unpack(json.loads(data))
        if metadata["codec"] == "cloudpickle":
            return cloudpickle.loads(data)
        raise ValueError("Unknown snapshot blob codec")

    def begin(self):
        self.stats = {
            "serialized_bytes": 0,
            "written_bytes": 0,
            "cache_hits": 0,
            "mutable_cache_hits": 0,
        }
        return time.monotonic()
