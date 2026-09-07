"""Native event-assisted candidates, with explicit reconciliation on uncertainty.

A fence waits for a filesystem marker in the same watch stream. Events are hints,
not a journal: periodic and verification-time reconciliation remain mandatory.
"""

import os
import threading
import time
import uuid
from pathlib import Path

from watchfiles import watch


class ChangeTracker:
    def __init__(self, root, *, reconcile_seconds=60, max_candidates=100_000):
        self.root = Path(root).resolve()
        gitdir = self.root / ".git"
        if gitdir.is_file():
            gitdir = (self.root / gitdir.read_text().strip().removeprefix("gitdir: ")).resolve()
        marker_dir = gitdir if gitdir.is_dir() else self.root
        self.marker = marker_dir / (".threadweave-watch-" + uuid.uuid4().hex)
        self.watch_roots = [self.root]
        if not marker_dir.is_relative_to(self.root):
            self.watch_roots.append(marker_dir)
        self.interval, self.maximum = reconcile_seconds, max_candidates
        self.pending, self.uncertain = set(), True
        self.lock, self.ready, self.seen = threading.Lock(), threading.Event(), threading.Event()
        self.stop = threading.Event()
        self.last_full = 0.0
        self.error = None
        self.generation = self.scanned_generation = 0
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        self.ready.wait(1)

    def _watch(self):
        try:
            for batch in watch(
                *self.watch_roots,
                watch_filter=None,
                stop_event=self.stop,
                debounce=1,
                step=1,
                rust_timeout=10,
                yield_on_timeout=True,
            ):
                with self.lock:
                    for _, path in batch:
                        if path == str(self.marker):
                            self.seen.set()
                        elif not Path(path).name.startswith(".threadweave-watch-"):
                            self.pending.add(path)
                    if len(self.pending) > self.maximum:
                        self.pending.clear()
                        self.uncertain = True
                        self.generation += 1
                self.ready.set()
        except Exception as exc:
            self.error = str(exc)
            self.uncertain = True
            self.generation += 1
            self.ready.set()

    def candidates(self, *, fence=True):
        if fence and self.thread.is_alive() and self.ready.is_set():
            self.marker.unlink(missing_ok=True)
            self.marker = self.marker.with_name(".threadweave-watch-" + uuid.uuid4().hex)
            self.seen.clear()
            self.marker.write_bytes(os.urandom(8))
            if not self.seen.wait(0.25):
                self.uncertain = True
        with self.lock:
            full = self.uncertain or not self.thread.is_alive()
            full |= time.monotonic() - self.last_full >= self.interval
            paths, self.pending = self.pending, set()
            self.scanned_generation = self.generation
            return full, paths

    def reconciled(self):
        self.uncertain = self.generation != self.scanned_generation
        self.last_full = time.monotonic()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.marker.unlink(missing_ok=True)


def expand_paths(root, candidates, previous):
    """Expand only changed directories (including deleted directory descendants)."""
    result = set()
    root = Path(root)
    for raw in candidates:
        path = Path(raw)
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        if ".git" in path.relative_to(root).parts or path.name.startswith(".threadweave-watch-"):
            continue
        if path.is_dir() and not path.is_symlink():
            if relative in previous:
                result.add(relative)
            for directory, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = [d for d in dirs if d != ".git"]
                result.update((Path(directory) / f).relative_to(root).as_posix() for f in files)
        else:
            result.add(relative)
            if relative not in previous:
                result.update(p for p in previous if p.startswith(relative + "/"))
    return result
