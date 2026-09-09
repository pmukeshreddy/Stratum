"""Prime continual harness: canonical JSON state and append-only JSONL refinements."""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

KINDS = ("prompt", "memory", "skill", "subagent")
log = logging.getLogger(__name__)


def timestamp():
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def empty_harness_state():
    return {"schema": 1, "entries": {kind: {} for kind in KINDS}, "refinements": []}


def atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".harness-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o600)
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
    return str(path)


def load_harness_state(directory, scope="global"):
    path = Path(directory) / "harness_state.json"
    state = empty_harness_state()
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("harness state must be an object")
    except FileNotFoundError:
        return state
    except (OSError, ValueError) as exc:
        log.warning("Cannot load continual harness %s: %s; using empty state", path, exc)
        return state
    state["schema"] = raw.get("schema", 1) if isinstance(raw.get("schema", 1), int) else 1
    entries = raw.get("entries")
    for kind in KINDS:
        records = entries.get(kind) if isinstance(entries, dict) else None
        if not isinstance(records, dict):
            continue
        for id, entry in records.items():
            if not isinstance(entry, dict):
                continue
            state["entries"][kind][id] = {
                **entry,
                "scope": entry.get("scope") if entry.get("scope") in ("local", "global") else scope,
                **{
                    key: entry.get(key) if isinstance(entry.get(key), dict) else {}
                    for key in ("reference", "arguments", "metadata")
                },
            }
    if isinstance(raw.get("refinements"), list):
        state["refinements"] = raw["refinements"]
    return state


def save_harness_state(directory, state):
    return atomic_json(Path(directory) / "harness_state.json", state)


def merge_harness_states(global_state, local_state=None):
    merged = empty_harness_state()
    local_state = local_state or empty_harness_state()
    merged["schema"] = max(global_state["schema"], local_state["schema"])
    for kind in KINDS:
        for scope, state in (("global", global_state), ("local", local_state)):
            for id, entry in state["entries"][kind].items():
                entry = copy.deepcopy(entry)
                entry.setdefault("scope", scope)
                key = f"{entry['scope']}:{id}" if id in merged["entries"][kind] else id
                merged["entries"][kind][key] = entry
    merged["refinements"] = copy.deepcopy(global_state["refinements"] + local_state["refinements"])
    return merged


def infer_scope(result, default="local"):
    scopes = {
        (e.get("after") or e.get("before") or {}).get("scope")
        for e in result.get("appliedEdits", [])
    }
    scopes.discard(None)
    return result.get("scope") or (next(iter(scopes)) if len(scopes) == 1 else default)


def load_refinement_history(directory, scope="global"):
    path = Path(directory) / "refinements.jsonl"
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("Cannot read refinement history %s: %s", path, exc)
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
            if (
                isinstance(record, dict)
                and "id" in record
                and isinstance(record.get("appliedEdits"), list)
            ):
                record["scope"] = infer_scope(record, scope)
                records.append(record)
        except ValueError:
            continue
    return records


def append_refinement_history(directory, result):
    path = Path(directory) / "refinements.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return str(path)


def merge_refinement_history(global_history, local_history):
    by_id = {r["id"]: r for r in global_history}
    for record in local_history:
        previous = by_id.get(record["id"], {})
        by_id[record["id"]] = {
            **record,
            "scope": record.get("scope") or previous.get("scope", "local"),
        }
    return list(by_id.values())


def compact_text(value, limit=180):
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def format_harness_state(state, *, entry_limit=6, content_limit=180, refinement_limit=5):
    lines = [
        "# Continual Harness State",
        "",
        "You have persistent learned harness state. Local entries belong to this session; global entries survive sessions.",
        "These compact summaries are routing hints. Inspect full entries with harness.get(kind, id, global_=True/False) when detail matters.",
        "Local entries may override global guidance for this session. Prompt entries supplement the immutable base system prompt.",
        "",
    ]
    for kind in KINDS:
        entries = sorted(
            state["entries"][kind].values(),
            key=lambda e: tuple(str(e.get(k, "")) for k in ("path", "title", "id")),
        )
        lines.append(f"{kind}: {len(entries)}")
        for entry in entries[:entry_limit]:
            extra = ""
            if kind == "skill":
                extra = " ref=" + compact_text(
                    json.dumps(entry.get("reference", {})), content_limit
                )
                extra += " args=" + compact_text(
                    json.dumps(entry.get("arguments", {})), content_limit
                )
            lines.append(
                f"- [{entry.get('scope', 'global')}:{entry['id']}] {entry.get('title', '')} ({entry.get('path', 'general')}, v{entry.get('version', 1)}){extra}: {compact_text(entry.get('content', ''), content_limit)}"
            )
        if len(entries) > entry_limit:
            lines.append(f"- +{len(entries) - entry_limit} more {kind} entries")
        lines.append("")
    if not any(state["entries"].values()):
        lines.extend(["No saved harness entries yet.", ""])
    events = state["refinements"]
    lines.append(f"recent refinements: {len(events)}")
    for event in events[-refinement_limit:]:
        lines.append(
            f"- [{event.get('id')}] {compact_text(event.get('trigger', ''))}: {', '.join(event.get('changes', [])) or 'no applied edits'}; outcome: {compact_text(event.get('outcome', ''))}"
        )
    if len(events) > refinement_limit:
        lines.append(f"- +{len(events) - refinement_limit} older refinement events")
    return "\n".join(lines).strip()


def normalize_proposal(value):
    if not isinstance(value, dict):
        raise ValueError("Refiner JSON must be an object")
    return {
        "summary": value.get("summary", "Refined continual harness state"),
        "rationale": value.get("rationale", ""),
        "expectedOutcome": value.get("expectedOutcome", ""),
        "edits": [e for e in value.get("edits", []) if isinstance(e, dict)]
        if isinstance(value.get("edits", []), list)
        else [],
    }


def validate_edit(edit, id):
    action, kind = edit.get("action"), edit.get("kind")
    if action not in ("create", "update", "delete"):
        return f"unsupported action {action}"
    if kind not in KINDS:
        return f"unsupported kind {kind}"
    if kind == "prompt" and id == "base_system_prompt":
        return "base system prompt is not editable"
    if action != "create" and not id:
        return f"{action} requires id"
    if action != "delete" and not all(
        isinstance(edit.get(k), str) and edit[k] for k in ("title", "content")
    ):
        return f"{action} requires title and content"
    if action != "delete" and kind == "skill":
        if not isinstance(edit.get("arguments"), dict):
            return f"{action} skill requires arguments"
        ref = edit.get("reference")
        if not isinstance(ref, dict):
            return f"{action} skill requires python reference"
        if ref.get("type") != "python":
            return f"{action} skill reference.type must be python"
        if not any(isinstance(ref.get(k), str) and ref[k] for k in ("import", "python_import")):
            return f"{action} skill requires python import"
        if not any(isinstance(ref.get(k), str) and ref[k] for k in ("callable", "call_pattern")):
            return f"{action} skill requires callable or call_pattern"
    return None


def apply_refinement_proposal(
    state, proposal, *, id, scope="local", baseline_state=None, rollback_of=None
):
    proposal = normalize_proposal(proposal)
    applied, modified = [], set()
    for raw in proposal["edits"]:
        edit = copy.deepcopy(raw)
        identifier = edit.get("id")
        if isinstance(identifier, str):
            identifier = re.sub(r"^(local|global):", "", identifier)
        elif edit.get("action") == "create":
            identifier = (
                re.sub(
                    r"[^a-z0-9]+", "_", str(edit.get("title") or edit.get("kind", "entry")).lower()
                ).strip("_")[:80]
                or "entry"
            )
        else:
            identifier = ""
        edit["id"] = identifier
        error = validate_edit(edit, identifier)
        kind, action = edit.get("kind"), edit.get("action")
        before = (
            copy.deepcopy(state["entries"].get(kind, {}).get(identifier))
            if isinstance(kind, str)
            else None
        )
        key = (kind, identifier)
        if (
            not error
            and baseline_state is not None
            and key not in modified
            and before != baseline_state["entries"][kind].get(identifier)
        ):
            error = "entry changed during refinement planning"
        if not error and action == "create" and before:
            error = "entry already exists"
        if not error and action != "create" and not before:
            error = "entry not found"
        result = {**edit, "applied": not bool(error)}
        if before:
            result["before"] = before
        if error:
            result["error"] = error
        elif action == "delete":
            del state["entries"][kind][identifier]
            modified.add(key)
        else:
            previous = before or {}
            after = {
                "id": identifier,
                "kind": kind,
                "title": edit["title"],
                "content": edit["content"],
                "path": edit.get("path", previous.get("path", "general")),
                "scope": previous.get("scope", scope),
                **{
                    k: edit.get(k, previous.get(k, {}))
                    for k in ("reference", "arguments", "metadata")
                },
                "source": "refine",
                "version": previous.get("version", 0) + 1,
                "created_at": previous.get("created_at", timestamp()),
                "updated_at": timestamp(),
            }
            state["entries"][kind][identifier] = after
            result["after"] = copy.deepcopy(after)
            modified.add(key)
        applied.append(result)
    state["refinements"].append(
        {
            "id": id,
            "trigger": proposal["summary"],
            "changes": [f"{e['action']} {e['kind']}:{e['id']}" for e in applied if e["applied"]],
            "evidence": proposal["rationale"],
            "outcome": proposal["expectedOutcome"],
            "created_at": timestamp(),
        }
    )
    return {
        "id": id,
        "scope": scope,
        **{k: proposal[k] for k in ("summary", "rationale", "expectedOutcome")},
        "appliedEdits": applied,
        "rollbackOf": rollback_of,
        "harnessStatePath": "",
        "timestamp": timestamp(),
    }


def rollback_proposal(target):
    edits = []
    for edit in reversed(target["appliedEdits"]):
        if not edit["applied"]:
            continue
        before, after = edit.get("before"), edit.get("after")
        inverse = {"kind": edit["kind"], "id": edit["id"], "reason": f"Rollback {target['id']}"}
        if before:
            inverse.update(
                action="update" if after else "create",
                **{
                    k: before[k]
                    for k in ("title", "content", "path", "reference", "arguments", "metadata")
                },
            )
        elif after:
            inverse["action"] = "delete"
        edits.append(inverse)
    return {
        "summary": f"Rollback refinement {target['id']}",
        "rationale": f"Restores continual harness state snapshots from refinement {target['id']}.",
        "expectedOutcome": "Faulty refinement edits are reverted.",
        "edits": edits,
    }


def refinement_notice(result, source):
    edits = [e for e in result["appliedEdits"] if e["applied"]]
    if not edits:
        return None
    lines = [f"[{source}-refinement]", compact_text(result["summary"])]
    for edit in edits:
        entry = edit.get("after") or edit.get("before")
        lines.append(
            f"- {edit['action']} {edit['kind']} [{entry['scope']}:{edit['id']}] {entry['title']}: {compact_text(entry['content'])}"
        )
    return "\n".join(lines)


class HarnessStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, sid=None):
        return (
            self.directory / "harness"
            if sid is None
            else self.directory / "sessions" / sid / "harness"
        )

    def load(self, sid=None):
        return load_harness_state(self.path(sid), "local" if sid else "global")

    def merged(self, sid):
        return merge_harness_states(self.load(), self.load(sid))

    def entries(self, sid):
        return [e for entries in self.merged(sid)["entries"].values() for e in entries.values()]

    def get(self, sid, kind, id, *, global_=False):
        if kind not in KINDS:
            raise ValueError(f"Unknown harness kind: {kind}")
        return self.load(None if global_ else sid)["entries"][kind].get(id)

    def history(self, sid):
        return merge_refinement_history(
            load_refinement_history(self.path()), load_refinement_history(self.path(sid), "local")
        )

    @contextmanager
    def lock(self, sid=None, *, directory=None):
        directory = Path(directory) if directory else self.path(sid)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def apply(
        self,
        sid,
        proposal,
        *,
        id,
        global_=False,
        baseline_state=None,
        rollback_of=None,
        target_directory=None,
    ):
        target = None if global_ else sid
        directory = Path(target_directory) if target_directory else self.path(target)
        with self.lock(target, directory=directory):
            state = load_harness_state(directory, "global" if global_ else "local")
            result = apply_refinement_proposal(
                state,
                proposal,
                id=id,
                scope="global" if global_ else "local",
                baseline_state=baseline_state,
                rollback_of=rollback_of,
            )
            result["harnessStatePath"] = save_harness_state(directory, state)
            append_refinement_history(self.path(target), result)
        return result
