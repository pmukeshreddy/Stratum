"""Git inspection and harness-owned file snapshots, without modifying user commits or index."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .artifacts import atomic_write
from .editing import Editor, invalidate_bytecode, make_diff
from .models import new_id, now
from .repository import confined, digest
from .storage import encode


def git(root, *arguments):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "--no-pager", *arguments],
        cwd=root,
        capture_output=True,
        timeout=30,
        check=False,
        env={
            **{k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "TMPDIR")},
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    if result.returncode:
        raise ValueError("Git failed: " + result.stderr.decode(errors="replace")[:2000])
    return result.stdout.decode(errors="replace")


def revision(value):
    if not value or value.startswith("-") or any(c in value for c in "\n\0"):
        raise ValueError("Invalid Git revision")
    return value


class GitWorkspace:
    def __init__(self, context):
        self.context, self.root = context, Path(context.session.workspace.path)
        self.store, self.artifacts = context.runtime.store, context.runtime.artifacts

    def status(self):
        top = git(self.root, "rev-parse", "--show-toplevel").strip()
        if Path(top).resolve() != self.root.resolve():
            raise ValueError("workspace must be the Git repository root")
        return {
            "head": git(self.root, "rev-parse", "HEAD").strip(),
            "status": git(self.root, "status", "--porcelain=v1", "--untracked-files=all"),
            "branch": git(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        }

    def files(self):
        return sorted(
            set(
                p
                for p in git(
                    self.root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
                ).split("\0")
                if p
            )
        )

    def snapshot(self, label="checkpoint"):
        manifest = {}
        for relative in self.files():
            # Never traverse symlinks, including tracked links into host directories.
            raw = self.root / relative
            if raw.is_symlink():
                manifest[relative] = {"symlink": os.readlink(raw)}
                continue
            path = confined(self.root, relative)
            if not path.is_file():
                continue
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError(
                    f"Checkpoint file exceeds 64 MB: {relative}; exclude generated artifacts"
                )
            with path.open("rb") as stream:
                aid = self.artifacts.put_stream(
                    self.context.session_id, stream, source_event=self.context.source_event
                )
            metadata = self.artifacts.metadata(self.context.session_id, aid)
            manifest[relative] = {
                "artifact": aid,
                "hash": metadata["sha256"],
                "mode": path.stat().st_mode & 0o777,
            }
        cid, state = new_id(), self.status()
        self.store.db.execute(
            "INSERT INTO checkpoints VALUES(?,?,?,?,?,?)",
            (cid, self.context.session_id, now(), label, encode(manifest), state["head"]),
        )
        self.store.event(
            self.context.session_id,
            "git_checkpoint",
            {"checkpoint_id": cid, "label": label, "head": state["head"]},
            parent=self.context.source_event,
        )
        return cid

    def observe_effects(self, checkpoint_id, action_id, *, recovered=False):
        before = self.checkpoint(checkpoint_id)["manifest"]
        after_id = self.snapshot("after-external-action")
        after = self.checkpoint(after_id)["manifest"]
        changed = {
            path: {"before": before.get(path), "after": after.get(path)}
            for path in sorted(set(before) | set(after))
            if {k: v for k, v in before.get(path, {}).items() if k != "artifact"}
            != {k: v for k, v in after.get(path, {}).items() if k != "artifact"}
        }
        body = {
            "action_id": action_id,
            "before_checkpoint": checkpoint_id,
            "after_checkpoint": after_id,
            "files": changed,
            "recovered": recovered,
        }
        if changed:
            invalidate_bytecode(self.root / p for p in changed)
            patch = self.diff(checkpoint_id)
            body["patch_artifact"] = self.artifacts.put_bytes(
                self.context.session_id, patch.encode(), "text/x-diff"
            )
            self.context.runtime.index(self.context.session_id).refresh(changed)
        self.store.event(
            self.context.session_id, "workspace_effects", body, parent=self.context.source_event
        )
        return body

    def checkpoint(self, checkpoint_id):
        row = self.store.db.execute(
            "SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)
        ).fetchone()
        if not row or self.store.session(row["session_id"]).root_id not in self.store.history_roots(
            self.context.session_id
        ):
            raise KeyError("Unknown checkpoint in this trajectory")
        return {**dict(row), "manifest": json.loads(row["manifest"])}

    def content(self, entry):
        meta = self.artifacts.metadata(self.context.session_id, entry["artifact"])
        raw = (self.store.directory / meta["path"]).read_bytes()
        if digest(raw) != entry["hash"]:
            raise ValueError("Checkpoint checksum mismatch")
        return raw

    def diff(self, checkpoint_id=None):
        if checkpoint_id is None:
            tracked = git(self.root, "diff", "HEAD", "--no-ext-diff", "--no-textconv", "--")
            untracked = git(self.root, "ls-files", "--others", "--exclude-standard", "-z").split(
                "\0"
            )
            for relative in filter(None, untracked):
                path = confined(self.root, relative)
                if path.is_file() and path.stat().st_size <= 2_000_000:
                    tracked += make_diff(relative, None, path.read_bytes())
            return tracked
        checkpoint = self.checkpoint(checkpoint_id)
        patch = ""
        for relative in sorted(set(self.files()) | set(checkpoint["manifest"])):
            entry = checkpoint["manifest"].get(relative)
            if entry and "symlink" in entry:
                continue
            path = confined(self.root, relative)
            before = self.content(entry) if entry else None
            after = path.read_bytes() if path.is_file() else None
            if before != after:
                patch += make_diff(relative, before, after)
        return patch

    def restore(self, checkpoint_id, paths=None):
        checkpoint = self.checkpoint(checkpoint_id)
        if checkpoint["session_id"] != self.context.session_id:
            raise PermissionError("Restore requires a checkpoint owned by this session")
        backup = self.snapshot("before-restore")
        selected = set(paths) if paths else set(checkpoint["manifest"]) | set(self.files())
        changes = {}
        for relative in selected:
            entry = checkpoint["manifest"].get(relative)
            self.context.path(relative)
            if entry and "symlink" in entry:
                if (
                    not (self.root / relative).is_symlink()
                    or os.readlink(self.root / relative) != entry["symlink"]
                ):
                    raise ValueError("Symlink restoration requires human inspection")
                continue
            if (self.root / relative).is_symlink():
                raise PermissionError("Cannot overwrite symlinks")
            if entry:
                changes[relative] = self.content(entry)
            elif (self.root / relative).exists():
                changes[relative] = None
        result = Editor(self.context).apply(
            changes,
            rollback_of=checkpoint_id,
            modes={p: e["mode"] for p, e in checkpoint["manifest"].items() if "mode" in e},
        )
        result["safety_checkpoint"] = backup
        return result

    def isolate(self, checkpoint_id):
        """Copy a captured working state into a private repository (including uncommitted inputs)."""
        checkpoint = self.checkpoint(checkpoint_id)
        destination = self.store.directory / "workspaces" / new_id()
        destination.mkdir(parents=True, mode=0o700)
        for relative, entry in checkpoint["manifest"].items():
            path = confined(destination, relative)
            if "symlink" in entry:
                target = (path.parent / entry["symlink"]).resolve()
                if not target.is_relative_to(destination):
                    raise PermissionError(f"Candidate contains an escaping symlink: {relative}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(entry["symlink"])
            else:
                atomic_write(path, self.content(entry))
                os.chmod(path, entry["mode"])
        git(destination, "init", "--quiet")
        git(destination, "add", "--all")
        # This is a harness-owned private repo, never the user's history/index.
        git(
            destination,
            "-c",
            "user.name=Threadweave",
            "-c",
            "user.email=local@localhost",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Captured candidate input",
        )
        return destination


def candidate_result(context, child_id, *, accept=False):
    runtime = context.runtime
    row = runtime.store.db.execute(
        "SELECT * FROM candidates WHERE child_id=? AND parent_id=?", (child_id, context.session_id)
    ).fetchone()
    if not row:
        raise PermissionError("Candidate must be a direct isolated child")
    child = runtime.store.session(child_id)
    if child_id in runtime.tasks or child.outcome == "active" and not child.paused:
        raise ValueError("Pause or finish the candidate before consuming its patch")
    from .tools import ToolContext

    child_context = ToolContext(runtime, child_id, new_id(), context.source_event)
    patch = GitWorkspace(child_context).diff()
    aid = runtime.artifacts.put_bytes(child_id, patch.encode(), "text/x-diff")
    body = json.loads(row["body"])
    body.update(
        consumed=True,
        patch_artifact=aid,
        end_time=child.updated_at,
        usage=runtime.store.usage(child_id).model_dump(),
        outcome=child.outcome,
        verifier=[
            e["payload"] for e in runtime.store.events(child_id, kind="verifier_result", limit=1)
        ],
    )
    result = {
        "child_id": child_id,
        "findings": child.result,
        "patch_artifact": aid,
        "patch": patch,
        "usage": body["usage"],
        "outcome": child.outcome,
    }
    if accept:
        GitWorkspace(context).snapshot("before-candidate-acceptance")
        result["edit"] = Editor(context).apply_patch(patch)
        body.update(accepted=True, useful=True)
    runtime.store.db.execute(
        "UPDATE candidates SET body=? WHERE child_id=?", (encode(body), child_id)
    )
    runtime.store.event(
        context.session_id,
        "candidate_consumed",
        {"child_id": child_id, "accepted": accept, "patch_artifact": aid},
        parent=context.source_event,
    )
    return result


async def recover_workspace_effects(runtime):
    import asyncio

    from .tools import ToolContext

    records = runtime.store.db.execute(
        "SELECT session_id,id,type,payload FROM events WHERE type IN ('workspace_observation_started','workspace_effects') ORDER BY seq"
    ).fetchall()
    pending = {}
    for row in records:
        body = json.loads(row["payload"])
        if row["type"] == "workspace_observation_started":
            pending[body["action_id"]] = (row["session_id"], row["id"], body["checkpoint_id"])
        else:
            pending.pop(body["action_id"], None)
    if pending:
        # Let local supervisors reap orphan groups before observing interrupted effects.
        await asyncio.sleep(0.35)
    for action_id, (sid, event, checkpoint) in pending.items():
        try:
            GitWorkspace(ToolContext(runtime, sid, action_id, event)).observe_effects(
                checkpoint, action_id, recovered=True
            )
        except (ValueError, OSError) as exc:
            runtime.store.event(
                sid, "workspace_observation_failed", {"action_id": action_id, "reason": str(exc)}
            )
            runtime.store.update(sid, paused=True, runnable=False)
