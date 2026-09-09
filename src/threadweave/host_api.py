"""Daemon dispatch for the preloaded Python namespace. No strategy or model calls here."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from .models import Record, StateEdit, new_id, now
from .storage import encode


class Request(Record):
    operation: str
    payload: dict = Field(default_factory=dict)


def resolve_capability(request, registry=None):
    """Expose the operation's policy identity to the common action pipeline.

    These are internal RPC capabilities, never additional model-facing tools.
    Operation-specific ownership, schema and server checks remain in dispatch.
    """
    from .tools import Tool

    op = request.operation
    if registry and op in registry.entries and registry.entries[op].host_rpc:
        return registry.entries[op]
    if op.startswith("bash."):
        permissions = ("process",)
    elif op.startswith("mcp."):
        permissions = ("mcp",)
    elif op.startswith(("rlm.", "agent_message.", "agent_observe.")):
        permissions = ("agents",)
    elif op.startswith(("harness.", "skills.")):
        permissions = ("state",)
    elif (
        op == "catalog"
        or op.startswith("goal.")
        or op in {"context.compact", "context.track", "context.resolve", "context.state"}
        or op in {"verification.run", "verification.latest"}
    ):
        permissions = ()
    else:
        raise ValueError(f"Unknown host operation: {op}")
    if op == "skills.prepare":
        permissions = ("state", "python")
    return Tool(
        op,
        "Internal Python capability",
        Request,
        dispatch,
        permissions,
        feature="subagents" if op.startswith("rlm.") else None,
    )


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
        entries = [
            e
            for e in store.states(sid)
            if (kind is None or e["kind"] == kind_name(kind)) and (e["owner_id"] is None) == global_
        ]
        for entry in entries:
            store.event(
                sid,
                "harness_state_retrieved",
                {"entry_id": entry["id"], "kind": entry["kind"], "version": entry["version"]},
                parent=context.source_event,
            )
        return entries
    kind = kind_name(kind)
    identifier = p.pop("id", None)
    current = state_entry(context, kind, identifier, global_) if identifier else None
    if operation == "get":
        entry = state_entry(context, kind, identifier, global_, p.get("version"))
        if entry:
            store.event(
                sid,
                "harness_state_retrieved",
                {"entry_id": entry["id"], "kind": entry["kind"], "version": entry["version"]},
                parent=context.source_event,
            )
        return entry
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
    extension = runtime.tools.entries.get(op)
    if extension and extension.host_rpc:
        if not runtime.tools.permitted(extension, config):
            raise PermissionError(f"Capability not permitted: {op}")
        return await extension.execute(context, request)
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
        if op == "rlm.followup":
            child = store.session(p["target"])
            if child.parent_id != sid:
                raise PermissionError("Follow-up target must be a direct child")
            return {
                "message_id": runtime.message(sid, child.id, p["instruction"]),
                "session_id": child.id,
                "kernel_id": child.kernel_id,
            }
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
            prompt = p["prompt"]
            requirement = p.get("requirement") or session.instruction
            spec = None
            if p.get("spec_id"):
                spec = store.state(sid, p["spec_id"])
                if spec["kind"] != "subagent_spec" or spec["deleted"]:
                    raise ValueError("A live subagent_spec entry is required")
                prompt = spec["content"]["instruction"] + "\n" + prompt
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("A nonempty assignment is required")
            if name and any(
                s["name"] == name and s["parent_id"] == sid for s in runtime.related(sid)
            ):
                raise ValueError("Child name already exists")
            from .routing import route

            provider = route(store, sid, session.role).model_copy(deep=True)
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
            child = await runtime.spawn_async(
                sid,
                prompt,
                name=name,
                role=p.get("role", "agent"),
                isolate=p.get("isolate"),
                provider=provider if p.get("model") or p.get("thinking") else None,
                adapter=p.get("adapter"),
                purpose=p.get("purpose", "shared"),
            )
            store.event(
                sid,
                "rlm_admitted",
                {
                    "child_id": child.id,
                    "assignment": prompt,
                    "depth": child.depth,
                    "originating_task_requirement": requirement,
                },
                parent=context.source_event,
            )
            if spec:
                store.event(
                    sid,
                    "subagent_spec_used",
                    {"entry_id": spec["id"], "version": spec["version"], "child_id": child.id},
                    parent=context.source_event,
                )
            return {
                "session_id": child.id,
                "name": child.name,
                "session_dir": str(store.directory / "kernels" / child.kernel_id),
                "model": store.config(child.id).provider.model,
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
        if op == "agent_observe.requests":
            graph = store.request_graph(observed)
            permitted = {s["id"] for s in visible}
            graph["requests"] = [r for r in graph["requests"] if r["session_id"] in permitted]
            ids = {r["id"] for r in graph["requests"]}
            graph["edges"] = [
                e for e in graph["edges"] if e["source"] in ids and e["target"] in ids
            ]
            return graph
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
            messages = store.trajectory(observed, limit=limit, max_chars=chars)
            store.event(
                sid,
                "child_observation",
                {"child_id": observed, "source_events": [m["id"] for m in messages]},
                parent=context.source_event,
            )
            return {
                "session_id": observed,
                "messages": messages,
            }
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
    if op in {"context.track", "context.resolve", "context.state"}:
        from .semantic_state import capture, work_items

        if op == "context.state":
            return capture(runtime.context, sid)
        identifier = p.get("id") or new_id()
        if op == "context.resolve":
            from .context_budget import pending_ledger

            item = next((i for i in work_items(store, sid) if i["id"] == identifier), None)
            if item is None:
                pending = next(
                    (
                        i
                        for i in pending_ledger(store.session(sid).summary)
                        if i["id"] == identifier
                    ),
                    None,
                )
                if pending is None:
                    raise KeyError(f"Unknown pending work item: {identifier}")
                item = {
                    "id": identifier,
                    "kind": {
                        "active_hypotheses": "hypothesis",
                        "blockers": "blocker",
                    }.get(pending["field"], "requirement"),
                    "text": encode(pending["content"]),
                }
            sources = p.get("evidence_events", [])
            if not sources or any(
                store.event_by_id(e)["root_id"] != context.session.root_id for e in sources
            ):
                raise ValueError("Resolution requires supporting events from this task tree")
            item.update(status="resolved", evidence_events=sources)
        else:
            kind = p.get("kind", "requirement")
            if kind not in {"requirement", "hypothesis", "blocker", "decision", "failed_approach"}:
                raise ValueError("Unsupported semantic work kind")
            if not isinstance(p.get("text"), str) or not p["text"].strip():
                raise ValueError("Work item needs nonempty text")
            item = {"id": identifier, "kind": kind, "text": p["text"], "status": "open"}
        event = store.event(sid, "semantic_state_updated", item, parent=context.source_event)
        if op == "context.resolve" and store.session(sid).summary:
            from .context_budget import merge_summary

            summary = merge_summary(
                store.session(sid).summary,
                {
                    "resolved_items": [
                        {
                            "id": identifier,
                            "reason": "Explicit resolution backed by task evidence",
                            "source_events": sources,
                        }
                    ]
                },
                source_events=sources,
            )
            store.update(sid, summary=encode(summary))
        return {**item, "source_event": event}
    if op == "context.compact":
        return {"event_id": runtime.context.compact(sid)}
    if op == "verification.latest":
        return store.events(sid, kind="verification_evidence", limit=5) + store.events(
            sid, kind="verifier_result", limit=1
        )
    if op == "verification.run":
        level = p.get("level", 2)
        if level not in (1, 2, 3):
            raise ValueError("Verification level must be 1, 2 or 3")
        if level != 3:
            return await runtime.verification.run(sid, context.source_event, level=level)
        result, error = await runtime._verify(sid, context.source_event)
        return {"result": result.model_dump(mode="json") if result else None, "error": error}
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
        if file and not set(file.get("required_permissions", ["python"])) <= set(
            config.permissions
        ):
            raise PermissionError("Skill requires unavailable permissions")
        store.event(
            sid,
            "skill_loaded",
            {
                "entry_id": entry["id"] if entry else file["path"],
                "version": entry["version"] if entry else file["sha256"],
            },
            parent=context.source_event,
        )
        if file:
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
    store.event(
        sid,
        "skill_loaded",
        {"entry_id": entry_id, "version": version, "operation": "prepare"},
        parent=context.source_event,
    )
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
