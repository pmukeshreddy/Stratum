"""Coding host operations admitted by the coding capability provider."""

from pathlib import Path

from .host_api import Request, permission
from .tools import Tool


async def dispatch(context, request):
    runtime, sid = context.runtime, context.session_id
    store, op, p = runtime.store, request.operation, dict(request.payload)
    session = store.session(sid)
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
    if op == "context.focus":
        fields = {"files", "symbols", "hypothesis", "constraints"}
        if set(p) - fields:
            raise ValueError("context.focus accepts files, symbols, hypothesis, constraints")
        for key in ("files", "symbols", "constraints"):
            values = p.get(key, [])
            if (
                not isinstance(values, list)
                or len(values) > 30
                or any(not isinstance(v, str) or len(v) > 1000 for v in values)
            ):
                raise ValueError(f"{key} must be at most 30 short strings")
        if not isinstance(p.get("hypothesis", ""), str) or len(p.get("hypothesis", "")) > 3000:
            raise ValueError("hypothesis must be a string of at most 3000 characters")
        event = store.event(sid, "working_focus", p, parent=context.source_event)
        return {"event_id": event, "focus": p}
    raise ValueError(f"Unknown coding operation: {op}")


def register(registry):
    for name, permissions in (("edit", ("workspace.write",)), ("context.focus", ())):
        registry.register(
            Tool(
                name,
                "Internal coding operation",
                Request,
                dispatch,
                permissions,
                model_callable=False,
                capability="coding",
                host_rpc=True,
            )
        )
