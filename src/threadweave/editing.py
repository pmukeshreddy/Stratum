"""Strict unified diffs and recoverable multi-file edits; no fuzzy matching or shell patching."""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from .artifacts import atomic_write
from .models import new_id, now
from .repository import digest
from .storage import encode


def unified_changes(patch, read):
    lines = patch.splitlines(keepends=True)
    changes, i = {}, 0
    while i < len(lines):
        if lines[i].startswith(("diff --git ", "index ")) or not lines[i].strip():
            i += 1
            continue
        if (
            not lines[i].startswith("--- ")
            or i + 1 >= len(lines)
            or not lines[i + 1].startswith("+++ ")
        ):
            raise ValueError(
                f"Malformed unified diff at patch line {i + 1}; expected ---/+++ headers"
            )
        old = lines[i][4:].rstrip("\r\n").split("\t")[0]
        new = lines[i + 1][4:].rstrip("\r\n").split("\t")[0]
        old = old[2:] if old.startswith("a/") else old
        new = new[2:] if new.startswith("b/") else new
        path = new if new != "/dev/null" else old
        if not path or path == "/dev/null" or (old != new and "/dev/null" not in (old, new)):
            raise ValueError(
                "Use move_file for renames; patch must modify, create or delete one path per section"
            )
        if path in changes:
            raise ValueError(f"Duplicate patch section: {path}")
        original = read(path)
        if old == "/dev/null":
            if original is not None:
                raise ValueError(f"Creation would overwrite {path}")
            original = b""
        elif original is None:
            raise ValueError(f"Patch source missing: {path}")
        before = original.decode("utf-8").splitlines(keepends=True)
        output, cursor, hunks = [], 0, 0
        i += 2
        while i < len(lines) and lines[i].startswith("@@"):
            match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?", lines[i])
            if not match:
                raise ValueError(f"Malformed hunk header at patch line {i + 1}")
            start, count = int(match[1]), int(match[2] or 1)
            new_start, new_count = int(match[3]), int(match[4] or 1)
            position = start - 1 if count else start
            if position < cursor or position > len(before):
                raise ValueError(f"Hunk outside source or overlapping in {path}:{start}")
            output.extend(before[cursor:position])
            if len(output) != (new_start - 1 if new_count else new_start):
                raise ValueError(f"Inconsistent new hunk offset in {path}")
            i += 1
            source_lines, target_lines = [], []
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff --git ")):
                line = lines[i]
                if not line or line[0] not in " +-\\":
                    break
                if line.startswith("\\ No newline at end of file"):
                    raise ValueError("Newline marker must immediately follow a hunk line")
                sign, content = line[0], line[1:]
                i += 1
                if i < len(lines) and lines[i].startswith("\\ No newline at end of file"):
                    content = content.removesuffix("\n")
                    i += 1
                if sign in " -":
                    source_lines.append(content)
                if sign in " +":
                    target_lines.append(content)
            if len(source_lines) != count or len(target_lines) != new_count:
                raise ValueError(f"Hunk counts do not match header in {path}:{start}")
            if before[position : position + count] != source_lines:
                raise ValueError(
                    f"Context mismatch in {path}:{start}; reread the file before retrying"
                )
            output.extend(target_lines)
            cursor, hunks = position + count, hunks + 1
        if not hunks:
            raise ValueError(f"No hunks in patch for {path}")
        output.extend(before[cursor:])
        if new == "/dev/null" and output:
            raise ValueError(f"Deletion patch did not remove all contents of {path}")
        changes[path] = None if new == "/dev/null" else "".join(output).encode()
    if not changes:
        raise ValueError("Patch contains no file changes")
    return changes


def make_diff(path, before, after):
    lines = list(
        difflib.unified_diff(
            (before or b"").decode(errors="replace").splitlines(True),
            (after or b"").decode(errors="replace").splitlines(True),
            fromfile="a/" + path if before is not None else "/dev/null",
            tofile="b/" + path if after is not None else "/dev/null",
        )
    )
    if before is None and after == b"":
        return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +0,0 @@\n"
    if before == b"" and after is None:
        return f"--- a/{path}\n+++ /dev/null\n@@ -0,0 +0,0 @@\n"
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines
    )


def invalidate_bytecode(paths):
    for path in paths:
        cache = path.parent / "__pycache__"
        if path.suffix == ".py" and cache.is_dir() and not cache.is_symlink():
            for compiled in cache.iterdir():
                if (
                    compiled.name.startswith(path.stem + ".")
                    and compiled.suffix == ".pyc"
                    and not compiled.is_symlink()
                ):
                    compiled.unlink()


class Editor:
    def __init__(self, context):
        self.context = context
        self.store, self.artifacts = context.runtime.store, context.runtime.artifacts

    def read(self, path):
        root = Path(self.context.session.workspace.path)
        raw = root / path
        for component in [raw, *raw.parents]:
            if component == root:
                break
            if component.is_symlink():
                raise PermissionError("Editing through symlinks is not supported")
        resolved = self.context.path(path)
        if resolved.is_symlink():
            raise PermissionError("Editing symlinks is not supported")
        return resolved.read_bytes() if resolved.exists() else None

    def apply_patch(self, patch):
        return self.apply(unified_changes(patch, self.read), patch=patch)

    def apply(self, changes, *, patch=None, rollback_of=None, modes=None):
        config = self.store.config(self.context.session_id)
        if config.execution.read_only:
            raise PermissionError("Workspace is configured read-only")
        record, paths = {}, {}
        for relative, after in changes.items():
            path = self.context.path(relative)
            if path == path.parent or path.is_dir():
                raise ValueError("Edits must target files")
            before = self.read(relative)
            paths[relative] = path
            record[relative] = {
                "before_hash": digest(before) if before is not None else None,
                "after_hash": digest(after) if after is not None else None,
                "before_artifact": self.artifacts.put_bytes(
                    self.context.session_id, before, "application/octet-stream"
                )
                if before is not None
                else None,
                "after_artifact": self.artifacts.put_bytes(
                    self.context.session_id, after, "application/octet-stream"
                )
                if after is not None
                else None,
                "mode": path.stat().st_mode & 0o777 if path.exists() else 0o644,
            }
            record[relative]["after_mode"] = (modes or {}).get(relative, record[relative]["mode"])
        patch = (
            patch
            if patch is not None
            else "".join(make_diff(p, self.read(p), a) for p, a in changes.items())
        )
        patch_id = self.artifacts.put_bytes(self.context.session_id, patch.encode(), "text/x-diff")
        eid = new_id()
        body = {
            "status": "prepared",
            "files": record,
            "patch_artifact": patch_id,
            "rollback_of": rollback_of,
        }
        self.store.db.execute(
            "INSERT INTO edits VALUES(?,?,?,?)", (eid, self.context.session_id, now(), encode(body))
        )
        try:
            for relative, after in changes.items():
                path = paths[relative]
                # Optimistic conflict detection preserves unrelated human edits.
                current = self.read(relative)
                if (digest(current) if current is not None else None) != record[relative][
                    "before_hash"
                ]:
                    raise ValueError(f"Concurrent edit detected: {relative}")
                if after is None:
                    path.unlink()
                else:
                    atomic_write(path, after)
                    os.chmod(path, record[relative]["after_mode"])
            body["status"] = "committed"
            # Timestamp-based .pyc files can otherwise serve pre-edit code when a same-size
            # patch lands within one second. Remove only the affected generated bytecode.
            invalidate_bytecode(paths.values())
        except BaseException:
            self._restore(record, expected="after_hash")
            body["status"] = "rolled_back"
            self.store.db.execute("UPDATE edits SET body=? WHERE id=?", (encode(body), eid))
            raise
        with self.store.transaction():
            self.store.db.execute("UPDATE edits SET body=? WHERE id=?", (encode(body), eid))
            self.store.event(
                self.context.session_id,
                "code_edit",
                {"edit_id": eid, **body},
                parent=self.context.source_event,
            )
        self.context.runtime.index(self.context.session_id).refresh(changes)
        return {"edit_id": eid, **body}

    def _restore(self, files, *, expected):
        for relative, metadata in files.items():
            current = self.read(relative)
            current_hash = digest(current) if current is not None else None
            if current_hash == metadata["before_hash"]:
                continue
            if current_hash != metadata[expected]:
                raise ValueError(f"Recovery conflict in {relative}; external changes preserved")
            path = self.context.path(relative)
            if metadata["before_artifact"] is None:
                path.unlink(missing_ok=True)
            else:
                meta = self.artifacts.metadata(self.context.session_id, metadata["before_artifact"])
                atomic_write(path, (self.store.directory / meta["path"]).read_bytes())
                os.chmod(path, metadata["mode"])

    def rollback(self, edit_id):
        row = self.store.db.execute(
            "SELECT * FROM edits WHERE id=? AND session_id=?", (edit_id, self.context.session_id)
        ).fetchone()
        if not row:
            raise KeyError("Unknown edit in this session")
        body = json.loads(row["body"])
        changes = {}
        for path, metadata in body["files"].items():
            current = self.read(path)
            if (digest(current) if current is not None else None) != metadata["after_hash"]:
                raise ValueError(f"Rollback conflict: {path} changed since edit")
            aid = metadata["before_artifact"]
            changes[path] = (
                (
                    self.store.directory
                    / self.artifacts.metadata(self.context.session_id, aid)["path"]
                ).read_bytes()
                if aid
                else None
            )
        return self.apply(
            changes, rollback_of=edit_id, modes={p: m["mode"] for p, m in body["files"].items()}
        )


def recover_edits(runtime):
    from .tools import ToolContext

    for row in runtime.store.db.execute("SELECT * FROM edits").fetchall():
        body = json.loads(row["body"])
        if body["status"] != "prepared":
            continue
        eid = runtime.store.event(row["session_id"], "edit_recovery", {"edit_id": row["id"]})
        editor = Editor(ToolContext(runtime, row["session_id"], new_id(), eid))
        try:
            editor._restore(body["files"], expected="after_hash")
            body["status"] = "rolled_back"
        except (ValueError, OSError) as exc:
            body.update(status="conflict", recovery_error=str(exc))
            runtime.store.update(row["session_id"], paused=True, runnable=False)
        runtime.store.db.execute("UPDATE edits SET body=? WHERE id=?", (encode(body), row["id"]))
