from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .artifacts import atomic_write
from .models import HarnessError, Record, StateEdit


@dataclass
class ToolContext:
    runtime: Any
    session_id: str
    action_id: str
    source_event: str
    from_python: bool = False
    workspace_override: Any = None

    @property
    def session(self):
        session = self.runtime.store.session(self.session_id)
        return (
            session.model_copy(update={"workspace": self.workspace_override})
            if self.workspace_override
            else session
        )

    def path(self, path: str) -> Path:
        from .repository import confined

        config = self.runtime.store.config(self.session_id)
        resolved = confined(
            Path(self.session.workspace.path),
            path,
            allowed=config.task.allowed_paths,
            forbidden=config.task.forbidden_paths,
        )
        if resolved.is_relative_to(self.runtime.store.directory) and not Path(
            self.session.workspace.path
        ).is_relative_to(self.runtime.store.directory / "workspaces"):
            raise PermissionError(
                "Runtime storage is private; retrieve evidence through artifact/history tools"
            )
        return resolved


@dataclass
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    execute: Callable[[ToolContext, Any], Awaitable[Any]]
    permissions: tuple[str, ...] = ()
    python_callable: bool = True
    coding_only: bool = False
    feature: str | None = None

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self):
        self.entries: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self.entries:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.entries[tool.name] = tool

    def allowed(self, name, config):
        tool = self.entries.get(name)
        return (
            tool
            and set(tool.permissions) <= set(config.permissions)
            and (not config.execution.read_only or "workspace.write" not in tool.permissions)
            and (config.tool_allowlist is None or name in config.tool_allowlist)
            and (not tool.coding_only or config.task.adapter == "coding")
            and (not tool.feature or getattr(config.features, tool.feature))
            and (
                name != "run_profile"
                or config.task.profiler_command
                or shutil.which("ncu")
                or shutil.which("nsys")
            )
            and (name != "agent_spawn" or config.features.subagents)
            and (
                name not in {"history_read", "history_get", "history_search", "artifact_search"}
                or config.features.history_retrieval
            )
        )

    def schemas(self, config):
        return [tool.schema() for name, tool in self.entries.items() if self.allowed(name, config)]

    async def call(self, context: ToolContext, name: str, arguments: dict):
        config = context.runtime.store.config(context.session_id)
        if name not in self.entries:
            raise HarnessError("tool", "unknown_tool", f"Unknown tool: {name}")
        if not self.allowed(name, config):
            raise HarnessError("tool", "permission_denied", f"Tool is not permitted: {name}")
        tool = self.entries[name]
        if context.from_python and not tool.python_callable:
            raise HarnessError(
                "tool", "reentrant_python", f"{name} cannot be called from inside Python"
            )
        try:
            validated = tool.arguments.model_validate(arguments)
        except ValidationError as exc:
            raise HarnessError("tool", "invalid_arguments", str(exc)[:3000]) from exc
        try:
            return await tool.execute(context, validated)
        except (HarnessError, asyncio.CancelledError):
            raise
        except (PermissionError, ValueError, KeyError, OSError) as exc:
            raise HarnessError("tool", type(exc).__name__, str(exc)) from exc


class Empty(Record):
    pass


class PythonArgs(Record):
    code: str = Field(max_length=200000)


class PathArgs(Record):
    path: str = "."


class ReadArgs(Record):
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16000, ge=1, le=1000000)


class WriteArgs(Record):
    path: str
    content: str = Field(max_length=4000000)


class ProcessArgs(Record):
    command: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=30, gt=0, le=3600)


class ArtifactArgs(Record):
    artifact_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=16000, ge=1, le=64000)


class ArtifactLoadArgs(Record):
    artifact_id: str


class SpawnArgs(Record):
    instruction: str = Field(default="", max_length=100000)
    name: str | None = Field(default=None, max_length=100)
    spec_id: str | None = None
    role: str = Field(default="agent", max_length=100)


class MessageArgs(Record):
    recipient_id: str
    body: str = Field(max_length=256000)


class InspectArgs(Record):
    session_id: str | None = None


class WaitArgs(Record):
    seconds: float = Field(default=1, ge=0.05, le=86400)


class FinishArgs(Record):
    result: str = Field(max_length=100000)


class HistoryArgs(Record):
    event_id: str | None = None
    session_id: str | None = None
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    kind: str | None = None


class StateReadArgs(Record):
    entry_id: str
    version: int | None = Field(default=None, ge=1)


class StateSelectArgs(Record):
    entry_ids: list[str] = Field(max_length=50)


class RefineArgs(Record):
    edit: StateEdit


class SkillArgs(Record):
    entry_id: str
    inputs: dict = Field(default_factory=dict)


class ScheduleArgs(Record):
    interval_seconds: float | None = Field(default=None, ge=0.1)
    cron: str | None = None
    instruction: str = Field(default="Scheduled continuation", max_length=10000)


async def run_process(context: ToolContext, args: ProcessArgs):
    from .execution import executor

    config = context.runtime.store.config(context.session_id)
    result = await executor(config.execution).run(
        context,
        args.command,
        cwd=args.cwd,
        timeout_seconds=min(args.timeout_seconds, config.limits.tool_timeout_seconds),
    )
    if result["timed_out"]:
        raise HarnessError(
            "tool",
            "command_timeout",
            "Command timed out; captured output is retained in execution_capture",
            uncertain=True,
        )
    return result


def builtins() -> ToolRegistry:
    registry = ToolRegistry()

    def add(name, description, args, handler, permissions=(), python_callable=True):
        registry.register(Tool(name, description, args, handler, permissions, python_callable))

    async def python(c, a):
        return await c.runtime.execute_python(c, a.code)

    async def read(c, a):
        with c.path(a.path).open("rb") as stream:
            stream.seek(a.offset)
            data = stream.read(a.limit)
        return {
            "text": data.decode(errors="replace"),
            "next_offset": a.offset + len(data),
            "bytes": c.path(a.path).stat().st_size,
            "sha256": hashlib.sha256(c.path(a.path).read_bytes()).hexdigest(),
        }

    async def write(c, a):
        if c.runtime.store.config(c.session_id).task.adapter == "coding":
            from .editing import Editor

            return Editor(c).apply({a.path: a.content.encode()})
        path = c.path(a.path)
        atomic_write(path, a.content.encode())
        return {"path": str(path), "bytes": len(a.content.encode())}

    async def listing(c, a):
        return [
            {"name": p.name, "directory": p.is_dir()} for p in sorted(c.path(a.path).iterdir())
        ][:1000]

    async def artifact_read(c, a):
        return c.runtime.artifacts.read(c.session_id, **a.model_dump())

    async def artifact_load(c, a):
        return c.runtime.artifacts.load(c.session_id, a.artifact_id)

    async def spawn(c, a):
        instruction = a.instruction
        if a.spec_id:
            entry = c.runtime.store.state(c.session_id, a.spec_id)
            if entry["kind"] != "subagent_spec" or entry["deleted"]:
                raise ValueError("A live subagent_spec entry is required")
            instruction = entry["content"]["instruction"] + "\n" + instruction
        if not instruction.strip():
            raise ValueError("An instruction or subagent specification is required")
        session = c.runtime.spawn(c.session_id, instruction, name=a.name, role=a.role)
        return {"session_id": session.id, "name": session.name, "parent_id": session.parent_id}

    async def message(c, a):
        return {"message_id": c.runtime.message(c.session_id, a.recipient_id, a.body)}

    async def related(c, a):
        return c.runtime.related(c.session_id)

    async def receive(c, a):
        return c.runtime.receive(c.session_id)

    async def inspect_session(c, a):
        sid = a.session_id or c.session_id
        if sid != c.session_id:
            c.runtime.store.ensure_related(c.session_id, sid)
        return c.runtime.inspect(sid)

    async def waiting(c, a):
        c.runtime.defer(c.session_id, a.seconds)
        return {"waiting_seconds": a.seconds, "wake_on_message": True}

    async def finish(c, a):
        return {"completion_requested": True, "result": a.result}

    async def history(c, a):
        sid = a.session_id or c.session_id
        roots = c.runtime.store.history_roots(c.session_id)
        if c.runtime.store.session(sid).root_id not in roots:
            raise PermissionError("History is outside this tree or branch ancestry")
        if a.event_id:
            event = c.runtime.store.event_by_id(a.event_id)
            if event["root_id"] not in roots:
                raise PermissionError("History is outside this tree or branch ancestry")
            return event
        return c.runtime.store.events(sid, after=a.after, limit=a.limit, kind=a.kind)

    async def states(c, a):
        return [
            {k: e[k] for k in ("id", "kind", "title", "version", "owner_id")}
            for e in c.runtime.store.states(c.session_id)
        ]

    async def state_read(c, a):
        entry = c.runtime.store.state(c.session_id, a.entry_id, a.version)
        if entry["kind"] == "skill":
            rows = c.runtime.store.db.execute(
                "SELECT passed FROM skill_outcomes WHERE entry_id=? AND version=? ORDER BY rowid DESC",
                (a.entry_id, entry["version"]),
            ).fetchall()
            cap = c.runtime.store.config(c.session_id).refinement.skill_failure_limit
            entry["statistics"] = {
                "successes": sum(row[0] for row in rows),
                "failures": sum(not row[0] for row in rows),
                "quarantined": len(rows) >= cap and not any(row[0] for row in rows[:cap]),
            }
        return entry

    async def select(c, a):
        for eid in a.entry_ids:
            c.runtime.store.state(c.session_id, eid)
        c.runtime.store.update(c.session_id, selected_state=a.entry_ids)
        c.runtime.store.event(c.session_id, "state_selected", a.model_dump(), parent=c.source_event)
        return {"selected": a.entry_ids}

    async def refine(c, a):
        return {
            "refinement_id": c.runtime.store.queue_refinement(c.session_id, a.edit),
            "applies": "next turn boundary",
        }

    async def skill(c, a):
        from .refinement import run_skill

        return await run_skill(c, a.entry_id, a.inputs)

    async def schedule(c, a):
        return {"schedule_id": c.runtime.schedule(c.session_id, **a.model_dump())}

    add(
        "python",
        "Execute Python in this session's persistent worker; last value is retained as _.",
        PythonArgs,
        python,
        ("python",),
        False,
    )
    add(
        "workspace_read",
        "Read a bounded file range in the workspace.",
        ReadArgs,
        read,
        ("workspace.read",),
    )
    add(
        "workspace_write",
        "Atomically write a UTF-8 workspace file.",
        WriteArgs,
        write,
        ("workspace.write",),
    )
    add(
        "workspace_list",
        "List a workspace directory (up to 1000 entries).",
        PathArgs,
        listing,
        ("workspace.read",),
    )
    add(
        "process_run",
        "Run a command as an argv list; retain output artifacts.",
        ProcessArgs,
        run_process,
        ("process",),
    )
    add("artifact_read", "Read an artifact byte range.", ArtifactArgs, artifact_read)
    add(
        "artifact_load",
        "Load a full artifact, especially into a Python variable for filtering.",
        ArtifactLoadArgs,
        artifact_load,
    )
    add(
        "agent_spawn",
        "Create a persistent child and immediately return its stable ID.",
        SpawnArgs,
        spawn,
        ("agents",),
    )
    add(
        "agent_message",
        "Queue a message to a parent, child, or permitted sibling.",
        MessageArgs,
        message,
        ("agents",),
    )
    add("agent_sessions", "List related sessions with bounded status.", Empty, related, ("agents",))
    add(
        "agent_receive",
        "Receive pending messages; also added to the next context.",
        Empty,
        receive,
        ("agents",),
    )
    add(
        "session_inspect",
        "Inspect self or a related session, full instruction, goal, and usage.",
        InspectArgs,
        inspect_session,
    )
    add(
        "agent_wait",
        "Yield until a message arrives or the delay expires; release the turn slot.",
        WaitArgs,
        waiting,
    )
    add(
        "finish",
        "Request task completion; verifier and child gates still apply.",
        FinishArgs,
        finish,
        (),
        False,
    )
    add(
        "history_read",
        "Retrieve durable events, including those omitted by compaction.",
        HistoryArgs,
        history,
    )
    add(
        "state_list",
        "List available persistent state metadata without injecting all content.",
        Empty,
        states,
        ("state",),
    )
    add(
        "state_read",
        "Read a persistent entry or historical version.",
        StateReadArgs,
        state_read,
        ("state",),
    )
    add(
        "state_select",
        "Select entries for bounded supplemental context at the next turn.",
        StateSelectArgs,
        select,
        ("state",),
    )
    add(
        "refine",
        "Queue a versioned state edit with trajectory evidence for the next turn boundary.",
        RefineArgs,
        refine,
        ("state",),
    )
    add(
        "skill_run",
        "Execute a stored skill in the session's Python worker.",
        SkillArgs,
        skill,
        ("state", "python"),
        False,
    )
    add(
        "schedule_turn",
        "Create a durable interval or five-field UTC cron schedule.",
        ScheduleArgs,
        schedule,
    )
    from .coding_tools import register

    register(registry)
    add("history_get", "Retrieve a durable event by ID.", HistoryArgs, history)
    add(
        "skill_inspect",
        "Read an executable skill and its version/provenance.",
        StateReadArgs,
        state_read,
        ("state",),
    )
    return registry
