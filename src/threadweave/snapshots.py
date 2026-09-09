"""Trusted local snapshot blobs. Never load snapshots from an untrusted source.

Explicit codecs remain preferred. Cloudpickle is restricted to user procedures,
classes and instances; opaque OS resources are rejected. Mutable objects still
need serialization to detect in-place changes; immutable values can be cached.
"""

import hashlib
import inspect
import io
import json
import os
import pickle
import signal
import socket
import subprocess
import time
import types
from contextlib import contextmanager
from itertools import chain

import cloudpickle

from .snapshot_io import CappedFile, write_packed


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
            for item in (value if type(value) is list else chain(value.keys(), value.values()))
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
        budget, created = [0], []
        try:
            encoded = self._encode_stream(value, budget, created)
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        size = budget[0]
        self.stats["serialized_bytes"] += size
        if immutable:
            self.cache[name] = (value, encoded, size)
        if flat and size <= self.policy.mutable_cache_bytes:
            # Exact scalar-container equality, not mutable identity or a hash.
            # Nested/opaque mutations deliberately take the full codec path.
            self.shadows[name] = (value.copy(), encoded, size)
        else:
            self.shadows.pop(name, None)
        if array:
            self.array_shadows[name] = (value.copy(), encoded, size)
        else:
            self.array_shadows.pop(name, None)
        return encoded, size

    def _encode_stream(self, value, budget, created, *, force_blob=False):
        policy = self.policy
        sink = CappedFile(self.directory, policy.artifact_bytes, policy.stream_chunk_bytes, budget)
        codec, dependencies, buffers = "json", [], []
        typecode = None
        try:
            if type(value) in (bytes, memoryview) and (
                force_blob or len(value) > policy.inline_bytes
            ):
                codec = "bytes"
                sink.write(value)
            elif type(value) is str and (force_blob or len(value) > policy.inline_bytes):
                codec = "utf8"
                step = max(1, policy.stream_chunk_bytes // 4)
                for offset in range(0, len(value), step):
                    sink.write(value[offset : offset + step].encode("utf8", errors="surrogatepass"))
            elif (type(value).__module__, type(value).__name__) == ("array", "array"):
                codec, typecode = "array", value.typecode
                sink.write(memoryview(value))
            else:
                try:
                    write_packed(sink, value, policy.stream_chunk_bytes)
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
                    budget[0] -= sink.size
                    sink.discard()
                    sink = CappedFile(
                        self.directory, policy.artifact_bytes, policy.stream_chunk_bytes, budget
                    )
                    codec = "cloudpickle"
                    owner = self

                    class StreamingPickler(SafeProcedurePickler):
                        def persistent_id(self, item):
                            if type(item) in (str, bytes) and len(item) > policy.inline_bytes:
                                record = owner._encode_stream(
                                    item, budget, created, force_blob=True
                                )
                                dependencies.append(record)
                                return ("snapshot_blob", record)
                            return None

                    def buffer_callback(buffer):
                        # Protocol 5 exports array storage without copying it into pickle bytes.
                        buffers.append(
                            self._encode_stream(buffer.raw(), budget, created, force_blob=True)
                        )

                    StreamingPickler(sink, protocol=5, buffer_callback=buffer_callback).dump(value)
            sink.finish()
            self.stats["max_write_bytes"] = max(self.stats["max_write_bytes"], sink.max_write)
            if codec == "json" and not force_blob and sink.size <= policy.inline_bytes:
                with sink.path.open() as stream:
                    return json.load(stream)
            digest = sink.digest.hexdigest()
            target = self.directory / digest
            if not target.exists():
                os.replace(sink.path, target)
                created.append(target)
                self.stats["written_bytes"] += sink.size
            metadata = {"sha256": digest, "codec": codec, "bytes": sink.size}
            if typecode:
                import sys

                metadata.update(typecode=typecode, byteorder=sys.byteorder)
            if codec == "json":
                metadata["has_paths"] = sink.has_paths
            if dependencies:
                metadata["dependencies"] = dependencies
            if buffers:
                metadata["buffers"] = buffers
            return ["blob", metadata]
        finally:
            sink.discard()

    def offload(self, record):
        if record[0] == "blob":
            return record
        sink = CappedFile(
            self.directory, self.policy.artifact_bytes, self.policy.stream_chunk_bytes, [0]
        )
        try:
            # Inline records are capped already; iterencode avoids a second bytes buffer.
            for part in json.JSONEncoder(allow_nan=False).iterencode(record):
                sink.write(part.encode())
            sink.finish()
            digest = sink.digest.hexdigest()
            path = self.directory / digest
            if not path.exists():
                os.replace(sink.path, path)
                self.stats["written_bytes"] += sink.size
            return ["blob", {"sha256": digest, "codec": "json", "bytes": sink.size}]
        finally:
            sink.discard()

    def checked_path(self, metadata):
        digest = metadata["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid snapshot blob identity")
        path = self.directory / digest
        if (
            path.stat().st_size != metadata["bytes"]
            or path.stat().st_size > self.policy.artifact_bytes
        ):
            raise ValueError("Snapshot blob size mismatch")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(self.policy.stream_chunk_bytes):
                checksum.update(block)
        if checksum.hexdigest() != digest:
            raise ValueError("Snapshot blob checksum mismatch")
        return path

    def decode(self, record, unpack):
        if record[0] != "blob":
            return unpack(record)
        metadata = record[1]
        path = self.checked_path(metadata)
        if metadata["codec"] == "json":
            with path.open() as stream:
                return unpack(json.load(stream))
        if metadata["codec"] == "bytes":
            return path.read_bytes()
        if metadata["codec"] == "array":
            import array
            import sys

            value = array.array(metadata["typecode"])
            with path.open("rb") as stream:
                value.fromfile(stream, metadata["bytes"] // value.itemsize)
            if metadata.get("byteorder", sys.byteorder) != sys.byteorder:
                value.byteswap()
            return value
        if metadata["codec"] == "utf8":
            with path.open(encoding="utf8", errors="surrogatepass", newline="") as stream:
                return stream.read()
        if metadata["codec"] == "cloudpickle":
            owner = self

            class SnapshotUnpickler(pickle.Unpickler):
                def persistent_load(self, identity):
                    kind, leaf = identity
                    if kind != "snapshot_blob":
                        raise ValueError("Unknown persistent snapshot reference")
                    return owner.decode(leaf, unpack)

            def buffers():
                for record in metadata.get("buffers", []):
                    buffer_path = self.checked_path(record[1])
                    value = bytearray(record[1]["bytes"])
                    with buffer_path.open("rb") as source:
                        source.readinto(value)
                    yield value

            with path.open("rb") as stream:
                return SnapshotUnpickler(stream, buffers=buffers()).load()
        raise ValueError("Unknown snapshot blob codec")

    def begin(self):
        self.stats = {
            "serialized_bytes": 0,
            "written_bytes": 0,
            "cache_hits": 0,
            "mutable_cache_hits": 0,
            "max_write_bytes": 0,
        }
        return time.monotonic()
