"""Daemon dispatch for the preloaded Python namespace. No strategy or model calls here."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from .models import Record, StateEdit, now
from .storage import encode


class Request(Record):
    operation: str
    payload: dict = Field(default_factory=dict)


def permission(context, name):
    if name not in context.runtime.store.config(context.session_id).permissions:
        raise PermissionError(f"Capability requires {name} permission")


def kind_name(kind):
    kinds = {
        "memory": "memory",
        "prompt": "prompt_note",
        "prompt_note": "prompt_note",
        "skill": "skill",
        "subagent": "subagent_spec",
        "subagent_spec": "subagent_spec",
    }
    if kind not in kinds:
        raise ValueError(f"Unknown harness category: {kind}")
    return kinds[kind]


def state_entry(context, kind, id, global_=False, version=None):
    store = context.runtime.store
    entries = store.states(context.session_id, include_deleted=True)
    for entry in entries:
        if (
            entry["kind"] == kind_name(kind)
            and (entry["owner_id"] is None) == global_
            and (entry["id"] == id or entry["content"].get("harness_id") == id)
        ):
            return store.state(context.session_id, entry["id"], version)
    return None


def harness(context, operation, payload):
    permission(context, "state")
    store, sid = context.runtime.store, context.session_id
    p = dict(payload)
    kind, global_ = p.pop("kind", None), p.pop("global_", False)
    if operation == "list":
        return [
            e
            for e in store.states(sid)
            if (kind is None or e["kind"] == kind_name(kind)) and (e["owner_id"] is None) == global_
        ]
    kind = kind_name(kind)
    identifier = p.pop("id", None)
    current = state_entry(context, kind, identifier, global_) if identifier else None
    if operation == "get":
        return state_entry(context, kind, identifier, global_, p.get("version"))
    if operation == "create" and current:
        raise ValueError("Harness entry already exists")
    if operation in {"update", "delete", "rollback"} and not current:
        raise KeyError(f"Harness entry not found: {identifier}")
    content = p.pop("content", None)
    title = p.pop("title", current["title"] if current else "")
    if content is None:
        content = current["content"] if current else {}
    if isinstance(content, str):
        if kind == "skill":
            reference = p.pop(
                "reference", current["content"].get("reference", {}) if current else {}
            )
            module = reference.get(
                "import", reference.get("module", reference.get("python_import"))
            )
            function = reference.get("callable", "run")
            if not module or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
                raise ValueError(
                    "Skill requires executable content or reference={'module': 'import_name'}"
                )
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", function):
                raise ValueError("Skill callable must be a Python attribute path")
            target = function if function.startswith(module + ".") else module + "." + function
            arguments = p.pop(
                "arguments", current["content"].get("arguments", {}) if current else {}
            )
            content = {
                "name": title,
                "description": content,
                "inputs": {"type": "object", "additionalProperties": True},
                "required_permissions": ["python"],
                "reference": reference,
                "arguments": arguments,
                "code": f"import {module}, inspect\nskill_result = {target}(**({arguments!r} | skill_inputs))\nif inspect.isawaitable(skill_result):\n    skill_result = await skill_result",
            }
        else:
            content = {"instruction" if kind == "subagent_spec" else "text": content}
    content = dict(content)
    # Skills have a validated executable schema; other entries retain reference-like grouping.
    content["harness_id"] = identifier or (
        current["content"].get("harness_id") if current else None
    )
    if "path" in p:
        content["path"] = p.pop("path")
    metadata = p.pop("metadata", {})
    edit = StateEdit(
        entry_id=current["id"] if current else None,
        kind=kind,
        scope="global" if global_ else "session",
        title=title,
        content=content,
        operation={"create": "upsert", "update": "upsert"}.get(operation, operation),
        rollback_version=p.pop("version", None),
        expected_version=current["version"] if current else None,
        source_events=p.pop("source_events", metadata.get("source_events", [context.source_event])),
        intended_effect=p.pop(
            "intended_effect", metadata.get("intended_effect", title or operation)
        ),
        select=p.pop("select", False),
    )
    p.pop("source", None)
    if p:
        raise ValueError(f"Unknown harness options: {sorted(p)}")
    # Explicit API CRUD is immediately visible in this cell; L1 selection happens only
    # at the next model boundary. Autonomous refinement remains queued and validated.
    with store.transaction():
        rid = store.queue_refinement(sid, edit)
        entry_id = store._apply_edit(sid, edit, context.source_event)
        store.db.execute("UPDATE refinements SET status='applied' WHERE id=?", (rid,))
    return store.state(sid, entry_id)


async def dispatch(context, request):
    runtime, sid = context.runtime, context.session_id
    store, op, p = runtime.store, request.operation, dict(request.payload)
    config, session = store.config(sid), store.session(sid)
    if op == "catalog":
        return [
            t.schema()
            for n, t in runtime.tools.entries.items()
            if t.python_callable and runtime.tools.allowed(n, config)
        ]
    if op.startswith("rlm."):
        permission(context, "agents")
        if not config.features.subagents:
            raise PermissionError("Subagents disabled")
        if op == "rlm.list_subagents":
            return [
                {**s, "session_id": s["id"], "rlm_child_id": s["id"], "status": s["outcome"]}
                for s in runtime.related(sid)
                if s["parent_id"] == sid
            ]
        if op == "rlm.delete_subagent":
            child = store.session(p["target"])
            if child.parent_id != sid:
                raise PermissionError("Only direct children may be cancelled")
            await runtime.stop(child.id)
            return runtime.inspect(child.id)
        if op == "rlm.find_models":
            available = [v for v in [config.provider, *config.models.values()]]
            provider = runtime.providers.get(config.provider.name)
            if config.provider.name == "codex_subscription":
                async with provider.control_factory() as control:
                    catalog = await control.models()
                rows = [
                    {
                        "provider": "codex_subscription",
                        "id": m["model"],
                        "selector": "codex_subscription/" + m["model"],
                    }
                    for m in catalog
                ]
            else:
                rows = [
                    {"provider": m.name, "id": m.model, "selector": m.name + "/" + m.model}
                    for m in available
                ]
            return [m for m in rows if p.get("query", "").lower() in m["selector"].lower()][
                : max(1, min(p.get("limit", 8), 100))
            ]
        if op == "rlm.run":
            name = p.get("name")
            if name and any(
                s["name"] == name and s["parent_id"] == sid for s in runtime.related(sid)
            ):
                raise ValueError("Child name already exists")
            provider = config.provider.model_copy(deep=True)
            if p.get("model"):
                selected = p["model"]
                choices = {
                    v.name + "/" + v.model: v for v in [config.provider, *config.models.values()]
                }
                if selected in choices:
                    provider = choices[selected].model_copy(deep=True)
                elif (
                    selected.startswith("codex_subscription/")
                    and "codex_subscription" in runtime.providers
                ):
                    provider = provider.model_copy(
                        update={"name": "codex_subscription", "model": selected.split("/", 1)[1]}
                    )
                else:
                    raise ValueError("Use an authenticated provider/model selector")
            if p.get("thinking"):
                provider.parameters["reasoning_effort"] = p["thinking"]
            if provider.name == "codex_subscription" and (p.get("model") or p.get("thinking")):
                provider, _ = await runtime.providers[provider.name].resolve(provider)
            child = runtime.spawn(sid, p["prompt"], name=name, isolate=False, provider=provider)
            return {
                "session_id": child.id,
                "name": child.name,
                "session_dir": str(store.directory / "kernels" / child.kernel_id),
                "model": provider.model,
            }
    if op.startswith("agent_message."):
        permission(context, "agents")
        related = runtime.related(sid)
        if op == "agent_message.list_agents":
            return {
                "self": sid,
                "parent": [s for s in related if s["id"] == session.parent_id],
                "children": [s for s in related if s["parent_id"] == sid],
                "siblings": [
                    s for s in related if session.parent_id and s["parent_id"] == session.parent_id
                ],
            }
        if op == "agent_message.send":
            role, name = p.get("receiver_role"), p.get("receiver_name")
            if p.get("broadcast_message") is not None:
                if p["message"] != "all" or role or name:
                    raise ValueError("Broadcast syntax is send('all', message)")
                targets, body = related, p["broadcast_message"]
            else:
                if role not in {"parent", "child", "sibling"}:
                    raise ValueError("receiver_role must be parent, child or sibling")
                if role == "parent" and name is not None or role != "parent" and not name:
                    raise ValueError("Names are required only for child/sibling recipients")
                targets = [
                    s
                    for s in related
                    if (
                        s["id"] == session.parent_id
                        if role == "parent"
                        else s["name"] == name
                        and (
                            s["parent_id"] == sid
                            if role == "child"
                            else session.parent_id and s["parent_id"] == session.parent_id
                        )
                    )
                ]
                body = p["message"]
                if len(targets) != 1:
                    raise ValueError("Recipient missing or ambiguous")
            return {
                "receipts": [
                    {
                        "message_id": runtime.message(sid, s["id"], body),
                        "recipient_id": s["id"],
                        "deliveryStatus": "queued",
                    }
                    for s in targets
                ]
            }
    if op.startswith("harness."):
        if op == "harness.refine":
            permission(context, "state")
            return runtime.request_refinement(
                sid,
                source="python",
                request_id=context.action_id,
                source_event=context.source_event,
            )
        return harness(context, op.split(".")[1], p)
    if op.startswith("agent_observe."):
        permission(context, "agents")
        visible = [session.model_dump(mode="json"), *runtime.related(sid)]
        if op == "agent_observe.list":
            return {"agents": visible}
        target = p["target"]
        if not isinstance(target, str) or not target:
            raise ValueError("A session ID/name or unambiguous suffix is required")
        matches = [
            s
            for s in visible
            if s["id"] == target or s["name"] == target or s["id"].endswith(target)
        ]
        if len(matches) != 1:
            raise ValueError("Session is not visible or target is ambiguous")
        observed = matches[0]["id"]
        if op == "agent_observe.get":
            return runtime.inspect(observed)
        if op == "agent_observe.recent":
            limit, chars = p.get("limit", 8), p.get("max_chars", 800)
            if (
                type(limit) is not int
                or type(chars) is not int
                or not 1 <= limit <= 50
                or not 80 <= chars <= 2000
            ):
                raise ValueError("limit must be 1..50 and max_chars 80..2000")
            return {
                "session_id": observed,
                "messages": [
                    {**m, "body": m["body"][:chars]} for m in store.messages(observed, limit=limit)
                ],
            }
    if op == "edit":
        permission(context, "workspace.write")
        from .editing import Editor

        path = context.path(p["path"])
        relative = str(path.relative_to(Path(session.workspace.path)))
        content = path.read_text()
        if not p["old_str"] or content.count(p["old_str"]) != 1:
            raise ValueError("old_str must match exactly once; include more surrounding text")
        return Editor(context).apply(
            {relative: content.replace(p["old_str"], p["new_str"], 1).encode()}
        )
    if op.startswith("bash."):
        permission(context, "process")
        from .background import BackgroundProcesses

        if not hasattr(runtime, "background"):
            runtime.background = BackgroundProcesses(runtime)
        return await runtime.background.call(context, op.split(".")[1], p)
    if op.startswith("mcp."):
        permission(context, "mcp")
        from .mcp_client import McpManager

        if not hasattr(runtime, "mcp"):
            runtime.mcp = McpManager(runtime)
        return await runtime.mcp.call(context, op.split(".")[1], p)
    if op.startswith("skills."):
        return skill_operation(context, op.split(".")[1], p)
    if op == "context.compact":
        return {"event_id": runtime.context.compact(sid)}
    if op == "goal.get":
        return {"goal": store.goal(sid), "usage": store.usage(sid).model_dump()}
    if op == "goal.create":
        if store.goal(sid) and store.goal(sid)["status"] == "active":
            raise ValueError("An active goal already exists")
        budget = p.get("token_budget")
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError("Goal token budget must be a positive integer")
        objective = p["objective"]
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("Goal objective must be nonempty")
        with store.transaction():
            store.db.execute("DELETE FROM goal_budgets WHERE session_id=?", (sid,))
            store.db.execute(
                "INSERT OR REPLACE INTO goals VALUES(?,?,?,?,?)",
                (sid, objective, "active", now(), now()),
            )
            if budget is not None:
                used, _ = store.subtree_tokens(sid)
                store.db.execute("INSERT INTO goal_budgets VALUES(?,?,?)", (sid, budget, used))
        store.update(sid, mode="goal")
        store.event(sid, "goal_created", {"objective": objective}, parent=context.source_event)
        return store.goal(sid)
    if op == "goal.complete":
        if not store.goal(sid):
            raise ValueError("No persistent goal exists")
        pending = store.session(sid).pending_turn
        if pending is None:
            raise ValueError("Goal completion must be requested during a model turn")
        pending["completion"] = "Persistent goal completed"
        store.update(sid, pending_turn=pending)
        return {"completion_requested": True}
    raise ValueError(f"Unknown host operation: {op}")


def skill_operation(context, operation, p):
    from .refinement import validate_inputs, validate_skill
    from .skills import discover

    permission(context, "state")
    store, sid = context.runtime.store, context.session_id
    config = store.config(sid)
    files = discover(Path(context.session.workspace.path), config.skill_paths)
    entries = [e for e in store.states(sid) if e["kind"] == "skill"]
    if operation == "list":
        return files + [
            {"name": e["content"]["name"], "id": e["id"], "kind": "stored", "version": e["version"]}
            for e in entries
        ]
    if operation == "outcome":
        ticket = store.event_by_id(p["ticket"])
        if ticket["session_id"] != sid or ticket["type"] != "skill_started":
            raise PermissionError("Skill ticket belongs to another execution")
        data = ticket["payload"]
        with store.transaction():
            if store.db.execute(
                "SELECT 1 FROM skill_outcomes WHERE id=?", (ticket["id"],)
            ).fetchone():
                raise ValueError("Skill outcome already recorded")
            store.db.execute(
                "INSERT INTO skill_outcomes VALUES(?,?,?,?,?,?)",
                (
                    ticket["id"],
                    sid,
                    data["entry_id"],
                    data["version"],
                    bool(p["passed"]),
                    encode({"source_event": context.source_event}),
                ),
            )
            store.event(
                sid,
                "skill_outcome",
                {**data, "passed": bool(p["passed"])},
                parent=context.source_event,
            )
        return {"recorded": True}
    entry = next(
        (e for e in entries if e["id"] == p["name"] or e["content"]["name"] == p["name"]), None
    )
    file = next((f for f in files if f["name"] == p["name"]), None)
    if not file and not entry:
        raise KeyError(f"Skill not found: {p['name']}")
    if operation == "load":
        if file:
            if not set(file.get("required_permissions", ["python"])) <= set(config.permissions):
                raise PermissionError("Skill requires unavailable permissions")
            return {**file, "text": Path(file["path"]).read_text()}
        return entry
    if operation != "prepare":
        raise ValueError("Unknown skill operation")
    permission(context, "python")
    if file:
        if file.get("error"):
            raise ValueError(file["error"])
        if not file.get("import_name"):
            raise ValueError("Markdown skills are instructions, not executable modules")
        if not set(file.get("required_permissions", ["python"])) <= set(config.permissions):
            raise PermissionError("Skill requires unavailable permissions")
        result, entry_id, version = dict(file), file["path"], file["sha256"]
    else:
        result = validate_skill(entry["content"], config.permissions)
        validate_inputs(result["inputs"], p["inputs"])
        entry_id, version = entry["id"], entry["version"]
    outcomes = store.db.execute(
        "SELECT passed FROM skill_outcomes WHERE entry_id=? AND version=? ORDER BY rowid DESC LIMIT ?",
        (entry_id, version, config.refinement.skill_failure_limit),
    ).fetchall()
    if len(outcomes) >= config.refinement.skill_failure_limit and not any(r[0] for r in outcomes):
        raise ValueError("Skill quarantined after repeated failures; update the procedure")
    result["ticket"] = store.event(
        sid,
        "skill_started",
        {"entry_id": entry_id, "version": version},
        parent=context.source_event,
    )
    return result


def register(registry):
    from .tools import Empty, Tool

    async def messages(context, args):
        return context.runtime.store.messages(context.session_id, limit=100)

    registry.register(
        Tool("host_request", "Internal Python-to-daemon capability bridge", Request, dispatch)
    )
    registry.register(
        Tool("message_history", "Read durable session messages", Empty, messages, ("agents",))
    )
