"""Prime continual harness: canonical JSON state and scope-aware refinement history."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

KINDS = ("prompt", "memory", "skill", "subagent")


class HarnessAuditError(Exception):
    def __init__(self, result, cause):
        super().__init__(str(cause))
        self.result, self.cause = result, cause


def timestamp():
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def empty_harness_state():
    return {"schema": 1, "entries": {kind: {} for kind in KINDS}, "refinements": []}


def atomic_json(path, value, *, python=False):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not python:
        value = json.loads(js_json(value))
    data = json.dumps(value, ensure_ascii=False, indent=2) + ("" if python else "\n")
    if not python:
        data = data.encode("utf-8", errors="backslashreplace").decode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else None
    mode = existing_mode if existing_mode is not None else 0o600
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
        if not python or existing_mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def load_harness_state(directory, scope="global", *, python=False):
    path = Path(directory) / "harness_state.json"
    state = empty_harness_state()
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("harness state must be an object")
    except FileNotFoundError:
        return state
    except (OSError, ValueError):
        return state
    state["schema"] = raw.get("schema", 1) if type(raw.get("schema", 1)) in (int, float) else 1
    entries = raw.get("entries")
    for kind in KINDS:
        records = entries.get(kind) if isinstance(entries, dict) else None
        if isinstance(records, list) and not python:
            records = {str(index): entry for index, entry in enumerate(records)}
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


def save_harness_state(directory, state, *, python=False):
    path = Path(directory) / "harness_state.json"
    atomic_json(path, state, python=python)
    return str(path)


def python_harness_state(directory, scope="local"):
    """Prime's Python HarnessEntry/RefinementEvent loader (not the host planner loader)."""
    state = load_harness_state(directory, scope, python=True)
    state["schema"] = 1
    for kind, records in state["entries"].items():
        clean = {}
        for id, entry in records.items():
            if not isinstance(entry.get("title"), str) or not isinstance(entry.get("content"), str):
                continue
            version = entry.get("version", 1)
            if isinstance(version, str):
                try:
                    version = int(version)
                except ValueError:
                    version = 1
            clean[id] = {
                "id": str(id),
                "kind": kind,
                "title": entry["title"],
                "content": entry["content"],
                "path": entry.get("path") if isinstance(entry.get("path"), str) else "general",
                "scope": entry["scope"],
                **{key: entry[key] for key in ("reference", "arguments", "metadata")},
                "source": entry.get("source") if isinstance(entry.get("source"), str) else "agent",
                "created_at": entry.get("created_at", datetime.now(UTC).isoformat()),
                "updated_at": entry.get("updated_at", datetime.now(UTC).isoformat()),
                "version": version if isinstance(version, int) else 1,
            }
        state["entries"][kind] = clean
    events = []
    for event in state["refinements"]:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("id"), str)
            or not isinstance(event.get("trigger"), str)
        ):
            continue
        changes = event.get("changes")
        if not isinstance(changes, (str, list)):
            continue
        events.append(
            {
                "id": event["id"],
                "trigger": event["trigger"],
                "changes": [changes]
                if isinstance(changes, str)
                else [str(change) for change in changes],
                "evidence": event.get("evidence", ""),
                "outcome": event.get("outcome", ""),
                "created_at": event.get("created_at", datetime.now(UTC).isoformat()),
            }
        )
    state["refinements"] = events
    return state


def merge_harness_states(global_state, local_state=None):
    merged = empty_harness_state()
    local_state = local_state or empty_harness_state()
    merged["schema"] = max(global_state["schema"], local_state["schema"])
    for kind in KINDS:
        for scope, state in (("global", global_state), ("local", local_state)):
            for id, entry in state["entries"][kind].items():
                entry = copy.deepcopy(entry)
                entry["scope"] = (
                    entry.get("scope") if entry.get("scope") in ("local", "global") else scope
                )
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
    return text if js_length(text) <= limit else js_slice(text, 0, max(0, limit - 3)) + "..."


def js_length(text):
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def js_slice(text, start=0, end=None):
    units = text.encode("utf-16-le", errors="surrogatepass")
    length = len(units) // 2
    start = max(0, length + start) if start < 0 else min(start, length)
    end = length if end is None else max(0, length + end) if end < 0 else min(end, length)
    return units[2 * start : 2 * max(start, end)].decode("utf-16-le", errors="surrogatepass")


def js_json(value):
    """JSON.stringify ordering and numeric normalization for touched-entry equality."""

    def normalize(item):
        if isinstance(item, float):
            return None if not math.isfinite(item) else int(item) if item.is_integer() else item
        if isinstance(item, list):
            return [normalize(v) for v in item]
        if isinstance(item, dict):
            indexed = sorted(
                (
                    key
                    for key in item
                    if str(key).isdigit() and str(int(key)) == key and int(key) < 4294967295
                ),
                key=int,
            )
            keys = [*indexed, *(key for key in item if key not in indexed)]
            return {key: normalize(item[key]) for key in keys}
        return item

    return json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"))


def format_harness_state(
    state,
    *,
    entry_limit=6,
    content_limit=180,
    refinement_limit=5,
    include_ipython_examples=True,
    include_shell_examples=False,
    include_refine_examples=None,
):
    if include_refine_examples is None:
        include_refine_examples = include_ipython_examples
    lines = [
        "# Continual Harness State",
        "",
        "Local continual harness entries belong to this Buffalo session. Global continual harness entries persist across Buffalo sessions.",
        "The continual harness entries below are compact summaries, not full descriptions. Use them as routing/context hints; inspect or refine the underlying continual harness entry only when detail matters.",
        "Default to local continual harness refinement for current task progress, temporary blockers, and session coordination. Use global continual harness refinement only for stable cross-session lessons, durable user preferences, reusable skills/subagents, or explicitly project-qualified facts.",
        "Use these continual harness prompt notes, memories, skills, and subagent specs when they are relevant. The base system prompt is immutable; prompt entries below are supplemental notes only.",
        "",
    ]
    call = (
        "call `await refine.run()`" if include_refine_examples else "refine the continual harness"
    )
    keep = (
        "`await refine.run()` continual harness edits"
        if include_refine_examples
        else "continual harness edits"
    )
    lines.extend(
        [
            f"When to {call}: after a repeated failure, a reusable tactic emerges, a repeated delegation role should become a subagent spec, a repeated procedure should become a skill, a durable fact/preference should become a memory, a narrow behavioral policy should become a prompt addendum, a user corrects behavior that should persist locally or globally, validation shows a continual harness entry is wrong, or a skill/subagent/memory/prompt note should be created, updated, deleted, or rolled back. Keep {keep} small and evidence-backed.",
            "",
            "Call contract: read each installed Python skill's SKILL.md and call its documented module function in the Python REPL; do not assume a `.run` entrypoint. Use `<skill_import> ...` in shell when a CLI exists. Continual harness skill entries are Python REPL skills with an explicit Python `reference` and `arguments` contract. Spawn a continual harness subagent spec by composing a concise task prompt and calling `handle = await rlm('sub-task')`; admission returns immediately with `rlm_child_id`, `name`, `session_dir`, and `model`, never the child's answer. Results arrive only through explicit `agent_message` replies or files; children reply with `await agent_message.send(message, receiver_role='parent')`. Use `await rlm.list_subagents()` to recover direct child handles and `await agent_message.send(..., receiver_role='child', receiver_name=handle.name)` for follow-ups. Do not invent wrappers such as `call_skill(...)`, `run_subagent(...)`, or named subagent registries."
            if include_ipython_examples
            else "Call contract: use installed skills as shell commands when available (for example `<skill_import> ...`). Continual harness entries are routing/context hints only in sessions without the Python REPL; do not use Python `await`, `asyncio`, or `rlm` examples unless the prompt also documents a Python kernel."
            if include_shell_examples
            else "Call contract: continual harness entries are routing/context hints only in sessions without the Python REPL or shell access; do not use Python `await`, `asyncio`, `rlm`, or shell skill commands unless the prompt also documents those interfaces.",
            "",
        ]
    )
    for kind in KINDS:
        entries = sorted(
            state["entries"][kind].values(),
            key=lambda e: tuple(str(e.get(k, "")) for k in ("path", "title", "id")),
        )
        lines.append(
            f"{kind}: {len(entries)} (invoke a spec by turning it into a concise task prompt and spawning with `await rlm('<task>')`; admission returns a child handle, never the answer)"
            if kind == "subagent" and entries and include_ipython_examples
            else f"{kind}: {len(entries)}"
        )
        for entry in entries[:entry_limit]:
            extra = ""
            if kind == "skill":
                for field, label in (("reference", "ref"), ("arguments", "args")):
                    if entry.get(field):
                        extra += f" {label}=" + compact_text(js_json(entry[field]), content_limit)
            lines.append(
                f"- [{entry.get('scope', 'global')}:{entry['id']}] {entry.get('title', '')} ({entry.get('path', 'general')}, v{entry.get('version', 1)}){extra}: {compact_text(entry.get('content', ''), content_limit)}"
            )
        if len(entries) > entry_limit:
            lines.append(f"- +{len(entries) - entry_limit} more {kind} entries")
        lines.append("")
    if not any(state["entries"].values()):
        lines.extend(["No saved harness entries yet.", ""])
    refinements = state["refinements"]
    lines.append(f"recent refinements: {len(refinements)}")
    for event in refinements[-refinement_limit:] if refinement_limit else []:
        changes = ", ".join(event["changes"]) or "no applied edits"
        outcome = (
            f"; outcome: {compact_text(event['outcome'], content_limit)}"
            if event.get("outcome")
            else ""
        )
        lines.append(
            f"- [{event['id']}] {compact_text(event['trigger'], content_limit)}: {changes}{outcome}"
        )
    if len(refinements) > refinement_limit:
        lines.append(f"- +{len(refinements) - refinement_limit} older refinement events")
    return "\n".join(lines).strip()


def overview_for_refinement(state):
    """Prime's bounded entry overview: up to 40 entries per kind, 240 content characters."""
    lines = []
    for kind in KINDS:
        entries = list(state["entries"][kind].values())
        lines.append(f"{kind}: {len(entries)}")
        for entry in entries[:40]:
            extra = ""
            if kind == "skill":
                for field, label in (("reference", "ref"), ("arguments", "args")):
                    if entry.get(field):
                        extra += f" {label}=" + js_slice(js_json(entry[field]), 0, 240)
            content = js_slice(re.sub(r"\s+", " ", entry["content"]), 0, 240)
            lines.append(
                f"- [{entry['scope']}:{entry['id']}] {entry['title']} "
                f"({entry['path']}, v{entry['version']}){extra}: {content}"
            )
        if len(entries) > 40:
            lines.append(f"- +{len(entries) - 40} more {kind} entries")
    return "\n".join(lines)


def history_for_refinement(history):
    if not history:
        return "No prior refinement history."
    lines = []
    for item in history[-20:]:
        edits = ", ".join(
            f"{'applied' if edit['applied'] else 'failed'} {edit['action']} {edit['kind']}:{edit['id']}"
            for edit in item["appliedEdits"]
        )
        rollback = f" rollbackOf={item['rollbackOf']}" if item.get("rollbackOf") else ""
        lines.append(
            f"[{item['id']}]{rollback} {item['summary']}\n{edits}\n"
            f"Expected outcome: {item['expectedOutcome']}"
        )
    return "\n\n".join(lines)


def normalize_proposal(value):
    if not isinstance(value, dict):
        value = {}
    edits = value.get("edits", [])
    normalized = []
    for edit in edits if isinstance(edits, list) else []:
        if isinstance(edit, list):
            edit = {}
        elif not isinstance(edit, dict):
            continue
        normalized.append(
            {
                **{key: edit.get(key) for key in ("action", "kind")},
                **{
                    key: edit[key]
                    for key in ("id", "title", "content", "path", "reason")
                    if isinstance(edit.get(key), str)
                },
                **{
                    key: edit[key]
                    for key in ("reference", "arguments", "metadata")
                    if isinstance(edit.get(key), dict)
                },
            }
        )
    return {
        "summary": value["summary"]
        if isinstance(value.get("summary"), str)
        else "Refined continual harness state",
        "rationale": value["rationale"] if isinstance(value.get("rationale"), str) else "",
        "expectedOutcome": value["expectedOutcome"]
        if isinstance(value.get("expectedOutcome"), str)
        else "",
        "edits": normalized,
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


def updated_entry(kind, id, edit, before=None, *, scope="local", source="agent", python=False):
    """Shared version/timestamp machinery for planner edits and direct kernel writes."""
    previous = before or {}
    return {
        "id": id,
        "kind": kind,
        "title": edit["title"],
        "content": edit["content"],
        "path": edit.get("path", previous.get("path", "general")),
        "scope": previous.get("scope", scope),
        **{k: edit.get(k, previous.get(k, {})) for k in ("reference", "arguments", "metadata")},
        "source": source,
        "version": previous.get("version", 0) + 1,
        "created_at": previous.get(
            "created_at", datetime.now(UTC).isoformat() if python else timestamp()
        ),
        "updated_at": datetime.now(UTC).isoformat() if python else timestamp(),
    }


def scope_target(id, global_=False):
    if isinstance(id, str):
        scope, sep, bare = id.partition(":")
        if sep and bare and scope in ("local", "global"):
            return bare, global_ or scope == "global"
    return id, global_


def apply_refinement_proposal(
    state, proposal, *, id, scope="local", baseline_state=None, rollback_of=None
):
    proposal = normalize_proposal(proposal)
    applied, modified = [], set()
    for raw in proposal["edits"]:
        edit = copy.deepcopy(raw)
        identifier = edit.get("id")
        if identifier is None and edit.get("action") == "create":
            identifier = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(edit.get("title", edit.get("kind", "entry"))).strip().lower(),
            ).strip("_")[:80] or str(edit.get("kind", "entry"))
        identifier = identifier if identifier is not None else ""
        edit["id"] = identifier
        # Validation checks the supplied id; computed ids are only for creates.
        error = validate_edit(edit, identifier)
        if raw.get("action") != "create" and not raw.get("id"):
            error = error or f"{raw.get('action')} requires id"
        if error:
            applied.append({**edit, "applied": False, "error": error})
            continue
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
            and js_json(before) != js_json(baseline_state["entries"][kind].get(identifier))
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
            after = updated_entry(kind, identifier, edit, before, scope=scope, source="refine")
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
        **({"rollbackOf": rollback_of} if rollback_of is not None else {}),
        "harnessStatePath": "",
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


class HarnessStore:
    def __init__(self, directory, *, session_history):
        self.directory = Path(directory)
        self.session_history = session_history

    def path(self, sid=None):
        return (
            self.directory / "harness"
            if sid is None
            else self.directory / "sessions" / sid / "harness"
        )

    def load(self, sid=None):
        return load_harness_state(self.path(sid), "local" if sid else "global")

    def python_state(self, sid=None):
        return python_harness_state(self.path(sid), "local" if sid else "global")

    def merged(self, sid):
        return merge_harness_states(self.load(), self.load(sid))

    def entries(self, sid):
        return [e for entries in self.merged(sid)["entries"].values() for e in entries.values()]

    def get(self, sid, kind, id, *, global_=False):
        if kind not in KINDS:
            raise ValueError(f"Unknown harness kind: {kind}")
        id, global_ = scope_target(id, global_)
        return self.python_state(None if global_ else sid)["entries"][kind].get(id)

    def mutate(self, sid, action, kind=None, *, id=None, global_=False, source="agent", **fields):
        id, global_ = scope_target(id, global_)
        target = None if global_ else sid
        state = self.python_state(target)  # Reload host/kernel writes before mutating.
        if action == "record_refinement":
            changes = fields["changes"]
            result = {
                "id": id or f"refine_{len(state['refinements']) + 1:04d}",
                "trigger": fields["trigger"],
                "changes": [changes] if isinstance(changes, str) else list(changes),
                "evidence": fields.get("evidence", ""),
                "outcome": fields.get("outcome", ""),
                "created_at": datetime.now(UTC).isoformat(),
            }
            state["refinements"].append(result)
        else:
            if kind not in KINDS:
                raise ValueError(f"Unknown harness kind: {kind}")
            if action not in ("create", "update", "delete", "upsert"):
                raise ValueError(f"Unknown harness operation: {action}")
            identifier = id or "_".join(
                "".join(
                    c.lower() if c.isalnum() else "_" for c in fields.get("title", "").strip()
                ).split()
            )
            identifier = id or ("_".join(p for p in identifier.split("_") if p) or kind)[:80]
            before = state["entries"][kind].get(identifier)
            if action == "delete":
                if before is None:
                    return False
                del state["entries"][kind][identifier]
                result = True
            else:
                if action == "create" and before is not None:
                    raise ValueError(f"{kind} entry {identifier!r} already exists")
                if action == "update" and before is None:
                    raise ValueError(f"{kind} entry {identifier!r} does not exist")
                fields = {k: v for k, v in fields.items() if v is not None}
                result = updated_entry(
                    kind,
                    identifier,
                    fields,
                    before,
                    scope="global" if global_ else "local",
                    source=source,
                    python=True,
                )
                state["entries"][kind][identifier] = result
        save_harness_state(self.path(target), state, python=True)
        return copy.deepcopy(result)

    def overview(self, sid, *, global_=False, max_entries_per_kind=20):
        state = self.python_state(None if global_ else sid)
        lines = [
            f"Harness state ({'global' if global_ else 'local'}): {self.path(None if global_ else sid) / 'harness_state.json'}",
            "Call contract: installed Python skills use await <skill_import>(...) or a matching shell CLI; "
            "harness skill entries are Python REPL skills and must include a Python reference plus arguments. "
            "Spawn a subagent spec by composing a concise task prompt and calling "
            "handle = await rlm('sub-task'); admission returns immediately with rlm_child_id, name, session_dir, "
            "and model, never the child's answer. Results arrive only through explicit agent_message replies or "
            "files; children reply with await agent_message.send(message, receiver_role='parent'). Use "
            "await rlm.list_subagents() to recover direct child handles and await agent_message.send(..., "
            "receiver_role='child', receiver_name=handle.name) for follow-ups.",
        ]
        for kind in KINDS:
            entries = sorted(
                state["entries"][kind].values(), key=lambda e: (e["path"], e["title"], e["id"])
            )
            lines.append(f"{kind}: {len(entries)}")
            for entry in entries[:max_entries_per_kind]:
                extra = ""
                if kind == "skill":
                    for field, label in (("reference", "ref"), ("arguments", "args")):
                        if entry.get(field):
                            extra += f" {label}=" + compact_text(
                                json.dumps(entry[field], ensure_ascii=False, sort_keys=True), 120
                            )
                summary = entry["content"].strip().replace("\n", " ")
                if len(summary) > 120:
                    summary = summary[:117] + "..."
                lines.append(
                    f"  - [{entry['scope']}:{entry['id']}] {entry['title']} ({entry['path']}, v{entry['version']}){extra}: {summary}"
                )
            if len(entries) > max_entries_per_kind:
                lines.append(f"  - +{len(entries) - max_entries_per_kind} more")
        lines.append(f"refinements: {len(state['refinements'])}")
        for event in state["refinements"][-5:]:
            lines.append(f"  - [{event['id']}] {event['trigger']}: {', '.join(event['changes'])}")
        return "\n".join(lines)

    def history(self, sid):
        return merge_refinement_history(
            load_refinement_history(self.path()), self.session_history.refinement_history(sid)
        )

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
        if rollback_of and not global_ and not (directory / "harness_state.json").exists():
            raise ValueError(f"Refinement state file not found: {directory / 'harness_state.json'}")
        proposal = copy.deepcopy(proposal)
        for edit in proposal.get("edits", []):
            if isinstance(edit.get("id"), str):
                edit["id"] = re.sub(r"^(local|global):", "", edit["id"])
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
        if global_:
            append_refinement_history(self.path(), result)
        try:
            self.session_history.record_harness_refinement(sid, result)
        except Exception as exc:
            raise HarnessAuditError(result, exc) from exc
        return result
