"""Incremental, metadata-cached observations of a trusted Git working tree.

Observations are not transactions or process attribution. Content is hashed only
on initial discovery or stat-signature changes. Immutable metadata artifacts make
open observation windows recoverable without copying every source file per cell.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from .change_tracking import ChangeTracker, expand_paths
from .editing import invalidate_bytecode
from .gitops import git
from .models import new_id
from .storage import encode
from .tools import ToolContext


def signature(info):
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def state_id(head, files):
    return hashlib.sha256(
        encode([head, {p: v["value"] for p, v in sorted(files.items())}]).encode()
    ).hexdigest()


class MutationObserver:
    def __init__(self, runtime):
        self.runtime, self.store = runtime, runtime.store
        self.cache = {}
        self.active = {}  # session -> outer window; no lock held across Python/RPC awaits
        self.baselines = {}
        self.last_poll = 0.0
        self.stats = {}
        self.trackers = {}
        self.force_full = False
        self.verification_changes = {}

    def close(self):
        for tracker in self.trackers.values():
            tracker.close()

    def load(self, owner, artifact):
        return self.runtime.artifacts.load(owner, artifact)

    def previous(self, root):
        row = self.store.db.execute(
            "SELECT * FROM mutation_workspaces WHERE path=?", (root,)
        ).fetchone()
        if not row:
            return None, None
        if root not in self.cache:
            self.cache[root] = self.load(row["session_id"], row["manifest_artifact"])
        return row, self.cache[root]

    def scan(self, context):
        root_path = Path(context.session.workspace.path).resolve()
        root = str(root_path)
        row, previous = self.previous(root)
        start = time.perf_counter()
        if root not in self.trackers:
            self.trackers[root] = ChangeTracker(root)
        tracker = self.trackers[root]
        full, candidates = tracker.candidates()
        full = full or previous is None or self.force_full
        if previous is not None and not full and not candidates:
            self.stats = {
                "seconds": time.perf_counter() - start,
                "files": 0,
                "hashed_files": 0,
                "hashed_bytes": 0,
                "full": False,
            }
            return root, previous, row["manifest_artifact"], previous, row
        # Include ignored paths as well: generated outputs can affect correctness.
        # Git excludes .git and does not follow directory symlinks. No index refresh,
        # staging, external diff driver, or repository content command is run here.
        private = self.store.directory.resolve()
        private_workspace = root_path.is_relative_to(private / "workspaces")
        excludes = (
            ["--exclude=" + private.relative_to(root_path).as_posix() + "/"]
            if private.is_relative_to(root_path)
            else []
        )
        paths = (
            set(
                filter(
                    None,
                    git(
                        root, "ls-files", "--cached", "--others", "-z", *excludes, errors="strict"
                    ).split("\0"),
                )
            )
            if full
            else expand_paths(root_path, candidates, previous["files"])
        )
        head = (
            git(root, "rev-parse", "HEAD").strip()
            if full or any(".git" in Path(p).parts for p in candidates)
            else previous["head"]
        )
        files, hashed_bytes, hashed_files = ({} if full else previous["files"].copy()), 0, 0
        old = previous["files"] if previous else {}
        parents = {}
        enumerated = {}
        if full:
            # Directory-entry stat avoids constructing/resolving a Path per
            # metadata syscall. Trust reconciliation still examines every file.
            for parent in {os.path.dirname(p) for p in paths}:
                current, safe = root, True
                for part in parent.split("/") if parent else []:
                    current = os.path.join(current, part)
                    if os.path.islink(current):
                        safe = False
                        break
                parents[parent] = safe
                if not safe:
                    continue
                try:
                    with os.scandir(current) as entries:
                        for entry in entries:
                            relative = parent + "/" + entry.name if parent else entry.name
                            if relative in paths:
                                try:
                                    enumerated[relative] = entry.stat(follow_symlinks=False)
                                except FileNotFoundError:
                                    pass
                except (FileNotFoundError, NotADirectoryError):
                    pass
        for relative in sorted(paths):
            files.pop(relative, None)
            if Path(relative).name.startswith(".threadweave-watch-"):
                continue
            path = root_path / relative
            if (not private_workspace and path.is_relative_to(private)) or ".git" in Path(
                relative
            ).parts:
                continue
            # A tracked directory can have been replaced with a symlink. Never
            # hash an outside target, even though Python itself is unrestricted.
            safe = True
            parent = os.path.dirname(relative)
            if parent not in parents:
                parts = parent.split("/") if parent else []
                current = root
                parents[parent] = True
                for part in parts:
                    current = os.path.join(current, part)
                    if os.path.islink(current):
                        parents[parent] = False
                        break
            safe = parents[parent]
            if not safe:
                continue
            try:
                info = enumerated.get(relative) if full else path.lstat()
                if info is None:
                    continue
            except FileNotFoundError:
                continue
            sig = signature(info)
            if relative in old and old[relative]["stat"] == sig:
                files[relative] = old[relative]
                continue
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                raw = os.fsencode(os.readlink(path))
                value = {
                    "kind": "symlink",
                    "hash": hashlib.sha256(raw).hexdigest(),
                    "mode": mode,
                    "size": len(raw),
                }
            elif stat.S_ISREG(info.st_mode):
                stable = False
                for _ in range(3):
                    digest = hashlib.sha256()
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode):
                            raise ValueError(f"File type changed while observing: {relative}")
                        while block := stream.read(1024 * 1024):
                            digest.update(block)
                            hashed_bytes += len(block)
                        after = os.fstat(stream.fileno())
                    if signature(before) == signature(after) == signature(path.lstat()):
                        sig, mode, stable = signature(after), stat.S_IMODE(after.st_mode), True
                        break
                if not stable:
                    raise ValueError(f"File kept changing during observation: {relative}")
                hashed_files += 1
                value = {
                    "kind": "file",
                    "hash": digest.hexdigest(),
                    "mode": mode,
                    "size": after.st_size,
                }
            else:
                # Submodule directories and special nodes: do not open devices/FIFOs.
                value = {
                    "kind": "directory" if stat.S_ISDIR(info.st_mode) else "special",
                    "mode": mode,
                }
            files[relative] = {"stat": sig, "value": value}
        if previous and head == previous["head"] and files == old:
            identity, manifest = previous["state_id"], previous
        else:
            identity = state_id(head, files)
            manifest = {"head": head, "state_id": identity, "files": files}
        if previous != manifest:
            artifact = self.runtime.artifacts.put(
                context.session_id, manifest, source_event=context.source_event
            )
            self.store.db.execute(
                "INSERT OR REPLACE INTO mutation_workspaces VALUES(?,?,?,?)",
                (root, context.session_id, artifact, identity),
            )
            self.cache[root] = manifest
        else:
            artifact = row["manifest_artifact"]
        self.stats = {
            "seconds": time.perf_counter() - start,
            "files": len(files),
            "hashed_files": hashed_files,
            "hashed_bytes": hashed_bytes,
            "full": full,
            "candidates": len(paths),
        }
        if full:
            tracker.reconciled()
        return root, manifest, artifact, previous, row

    def report(self, context, before, after, *, reason, window_id=None, recovered=False):
        if before is None or before["state_id"] == after["state_id"]:
            return
        a, b = before["files"], after["files"]
        changed = {
            p: {"before": a.get(p, {}).get("value"), "after": b.get(p, {}).get("value")}
            for p in sorted(a.keys() | b.keys())
            if a.get(p, {}).get("value") != b.get(p, {}).get("value")
        }
        if not changed and before["head"] == after["head"]:
            return
        added = [p for p, v in changed.items() if v["before"] is None]
        deleted = [p for p, v in changed.items() if v["after"] is None]
        renames, additions, deletions = [], {}, {}
        for path in added:
            additions.setdefault(encode(changed[path]["after"]), []).append(path)
        for path in deleted:
            deletions.setdefault(encode(changed[path]["before"]), []).append(path)
        for old in deleted:
            key = encode(changed[old]["before"])
            matches = additions.get(key, [])
            if len(matches) == 1 and len(deletions[key]) == 1:
                renames.append(
                    {"from": old, "to": matches[0], "basis": "identical_content_and_mode"}
                )
        body = {
            "action_id": context.action_id,
            "execution_id": context.action_id,
            "window_id": window_id,
            "kind": "externally_observed",
            "transactional": False,
            "rollback_performed": False,
            "reason": reason,
            "recovered": recovered,
            "attribution": "workspace_interval_not_exclusive_process",
            "before_state": before["state_id"],
            "after_state": after["state_id"],
            "before_head": before["head"],
            "after_head": after["head"],
            "files": changed,
            "added": added,
            "deleted": deleted,
            "modified": sorted(changed.keys() - set(added) - set(deleted)),
            "renames": renames,
            "file_count": len(changed),
        }
        artifact = self.runtime.artifacts.put(
            context.session_id, body, source_event=context.source_event
        )
        bounded = {
            **body,
            "files": dict(list(changed.items())[:100]),
            "observation_artifact": artifact,
            "truncated": len(changed) > 100,
        }
        for key in ("added", "deleted", "modified", "renames"):
            bounded[key] = bounded[key][:100]
        self.store.event(
            context.session_id, "workspace_effects", bounded, parent=context.source_event
        )
        # Use the existing invalidation path, not new repository-intelligence policy.
        safe = []
        for relative in changed:
            try:
                path = context.path(relative)
                if path.is_symlink() or any(
                    p.is_symlink()
                    for p in (Path(context.session.workspace.path) / relative).parents
                ):
                    continue
                safe.append(relative)
            except (OSError, ValueError, PermissionError):
                continue  # Record forbidden/outside mutations without reading/indexing them.
        invalidate_bytecode(Path(context.session.workspace.path) / p for p in safe)
        self.runtime.index(context.session_id).refresh(safe)

    def begin(self, context):
        if context.session_id in self.active:
            return None  # nested RPC shares the cell's interval; hooks/audit still run
        with self.atomic(context):
            return self._begin(context)

    @contextmanager
    def atomic(self, context):
        try:
            with self.store.transaction():
                yield
        except BaseException:
            self.cache.pop(str(Path(context.session.workspace.path).resolve()), None)
            raise

    def _begin(self, context):
        root, manifest, artifact, previous, row = self.scan(context)
        self.report(context, previous, manifest, reason="between_actions")
        identifier = new_id()
        owner = context.session_id if previous != manifest else row["session_id"]
        self.store.db.execute(
            "INSERT INTO mutation_windows VALUES(?,?,?,?,?,?,?,?)",
            (
                identifier,
                root,
                context.session_id,
                context.action_id,
                context.source_event,
                artifact,
                owner,
                "open",
            ),
        )
        self.active[context.session_id] = identifier
        self.baselines[identifier] = manifest
        self.store.event(
            context.session_id,
            "mutation_observation_started",
            {
                "window_id": identifier,
                "action_id": context.action_id,
                "before_state": manifest["state_id"],
            },
            parent=context.source_event,
        )
        return identifier

    def end(self, context, identifier, *, recovered=False):
        if not identifier:
            return
        try:
            with self.atomic(context):
                self._end(context, identifier, recovered=recovered)
        finally:
            self.baselines.pop(identifier, None)
            self.active.pop(context.session_id, None)

    def _end(self, context, identifier, *, recovered=False):
        row = self.store.db.execute(
            "SELECT * FROM mutation_windows WHERE id=?", (identifier,)
        ).fetchone()
        try:
            _, after, _, _, _ = self.scan(context)
            before = self.baselines.get(identifier)
            if before is None:
                before = self.load(row["before_owner"], row["before_artifact"])
            self.report(
                context,
                before,
                after,
                reason="recovery" if recovered else "action",
                window_id=identifier,
                recovered=recovered,
            )
            self.store.db.execute(
                "UPDATE mutation_windows SET status='observed' WHERE id=?", (identifier,)
            )
            self.store.event(
                context.session_id,
                "mutation_observation_finished",
                {
                    "window_id": identifier,
                    "action_id": context.action_id,
                    "after_state": after["state_id"],
                    "metrics": self.stats,
                },
                parent=context.source_event,
            )
        finally:
            self.active.pop(context.session_id, None)

    def reconcile(self, context, *, reason):
        self.force_full = reason in {"recovery", "before_verifier", "after_verifier", "checkpoint"}
        try:
            with self.atomic(context):
                _, after, _, before, _ = self.scan(context)
                self.report(context, before, after, reason=reason)
        finally:
            self.force_full = False

    def poll(self, *, force=False, reason="background_reconciliation"):
        if not force and time.monotonic() - self.last_poll < 1:
            return
        self.last_poll = time.monotonic()
        busy = {self.store.session(sid).workspace.path for sid in self.active}
        for row in self.store.db.execute("SELECT * FROM mutation_workspaces").fetchall():
            if row["path"] in busy:
                continue
            context = ToolContext(
                self.runtime,
                row["session_id"],
                new_id(),
                self.store.events(row["session_id"], limit=1)[0]["id"],
            )
            try:
                self.reconcile(context, reason=reason)
            except Exception as exc:
                self.failed(context, exc)

    def failed(self, context, exc):
        self.store.event(
            context.session_id,
            "workspace_observation_failed",
            {"action_id": context.action_id, "reason": str(exc), "rollback_performed": False},
            parent=context.source_event,
        )
        self.store.update(context.session_id, paused=True, runnable=False)

    def recover(self):
        for row in self.store.db.execute(
            "SELECT * FROM mutation_windows WHERE status='open'"
        ).fetchall():
            context = ToolContext(
                self.runtime, row["session_id"], row["action_id"], row["source_event"]
            )
            try:
                self.end(context, row["id"], recovered=True)
            except Exception as exc:
                self.failed(context, exc)
        self.poll(force=True, reason="recovery")
