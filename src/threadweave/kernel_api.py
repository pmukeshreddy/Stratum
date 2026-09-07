"""Preloaded Python control plane. All host operations retain daemon identity/permissions.

These objects never invoke a model themselves. rlm admits persistent runtime sessions;
MCP, shell and state requests cross the same audited bridge as other environment tools.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path


class Record(dict):
    """Structured data with attribute access, still serializable as an ordinary mapping."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class AgentHandle:
    session_id: str
    name: str
    session_dir: str
    model: str

    @property
    def rlm_child_id(self):
        return self.session_id


class Host:
    def __init__(self, bridge):
        self.bridge = bridge

    def call(self, operation, **payload):
        return self.bridge.call("host_request", operation=operation, payload=payload)

    async def acall(self, operation, **payload):
        return await self.bridge.acall("host_request", operation=operation, payload=payload)


class Harness:
    def __init__(self, host):
        self.host = host

    def list(self, kind=None, *, global_=False):
        return [Record(e) for e in self.host.call("harness.list", kind=kind, global_=global_)]

    def get(self, kind, id, *, global_=False, version=None):
        value = self.host.call("harness.get", kind=kind, id=id, global_=global_, version=version)
        return Record(value) if value else None

    def create(self, kind, title, content, **options):
        return Record(
            self.host.call("harness.create", kind=kind, title=title, content=content, **options)
        )

    def update(self, kind, id, title, content, **options):
        return Record(
            self.host.call(
                "harness.update", kind=kind, id=id, title=title, content=content, **options
            )
        )

    def delete(self, kind, id, **options):
        return self.host.call("harness.delete", kind=kind, id=id, **options)

    def rollback(self, kind, id, version, **options):
        return Record(
            self.host.call("harness.rollback", kind=kind, id=id, version=version, **options)
        )

    def select(self, ids):
        return self.host.bridge.call("state_select", entry_ids=ids)

    def __getattr__(self, name):
        # Explicit category CRUD shares validation/provenance, not arbitrary host dispatch.
        operation, _, category = name.partition("_")
        kinds = {
            "memory": "memory",
            "prompt_note": "prompt",
            "skill": "skill",
            "subagent": "subagent",
        }
        if operation in {"create", "update", "delete"} and category in kinds:

            def method(*args, **kwargs):
                return getattr(self, operation)(kinds[category], *args, **kwargs)

            return method
        raise AttributeError(name)


class Recursive:
    def __init__(self, host):
        self.host, self.harness = host, Harness(host)

    async def __call__(self, prompt, *, name=None, model=None, thinking=None, purpose="shared"):
        return await self.run(prompt, name=name, model=model, thinking=thinking, purpose=purpose)

    async def run(self, prompt, *, name=None, model=None, thinking=None, purpose="shared"):
        result = await self.host.acall(
            "rlm.run", prompt=prompt, name=name, model=model, thinking=thinking, purpose=purpose
        )
        return AgentHandle(**result)

    async def candidate(self, handle, *, accept=False):
        return await self.host.bridge.acall(
            "candidate_apply" if accept else "candidate_inspect", child_id=handle.session_id
        )

    async def list_subagents(self):
        return [Record(s) for s in await self.host.acall("rlm.list_subagents")]

    async def find_models(self, query="", limit=8):
        return await self.host.acall("rlm.find_models", query=query, limit=limit)

    async def delete_subagent(self, target):
        return await self.host.acall(
            "rlm.delete_subagent",
            target=target.session_id if isinstance(target, AgentHandle) else target,
        )


class Messaging:
    def __init__(self, host):
        self.host = host

    async def list_agents(self):
        return await self.host.acall("agent_message.list_agents")

    async def send(
        self, message, broadcast_message=None, *, receiver_role=None, receiver_name=None
    ):
        return await self.host.acall(
            "agent_message.send",
            message=message,
            broadcast_message=broadcast_message,
            receiver_role=receiver_role,
            receiver_name=receiver_name,
        )

    async def receive(self):
        return await self.host.bridge.acall("agent_receive")


class Mcp:
    def __init__(self, host):
        self.host = host

    async def list_servers(self):
        return await self.host.acall("mcp.servers")

    async def list_tools(self, server):
        return await self.host.acall("mcp.tools", server=server)

    async def call_tool(self, server, tool, arguments=None):
        result = await self.host.acall(
            "mcp.call", server=server, tool=tool, arguments=arguments or {}
        )
        texts = [b["text"] for b in result.get("content", []) if "text" in b]
        if result.get("isError"):
            raise RuntimeError("\n".join(texts) or "MCP tool returned an error")
        structured = result.get("structuredContent", result.get("structured_content"))
        if structured is not None:
            return structured
        if texts:
            return "\n".join(texts)
        return result.get("content") or result

    async def reload(self, server=None):
        return await self.host.acall("mcp.reload", server=server)

    async def close(self):
        return await self.host.acall("mcp.close")


class Observation:
    def __init__(self, host):
        self.host = host

    async def list_agents(self):
        return await self.host.acall("agent_observe.list")

    async def get_agent(self, target):
        return await self.host.acall("agent_observe.get", target=target)

    async def recent_messages(self, target, limit=8, max_chars=800):
        return await self.host.acall(
            "agent_observe.recent", target=target, limit=limit, max_chars=max_chars
        )


class BashHandle:
    def __init__(self, host, id):
        self.host, self.id, self.released = host, id, False

    def _status(self):
        self.released = True
        return Record(self.host.call("bash.status", id=self.id))

    @property
    def pid(self):
        return self._status().get("pid")

    @property
    def running(self):
        return self._status()["running"]

    def output(self):
        return self._status()["output"]

    def tail(self, n=50):
        return "\n".join(self.output().splitlines()[-n:])

    def poll(self):
        result = self._status()
        return None if result["running"] else result

    def kill(self, sig=15, grace=0.5):
        self.released = True
        return self.host.call("bash.kill", id=self.id, sig=sig, grace=grace)

    async def _wait(self, owned):
        return Record(await self.host.acall("bash.wait", id=self.id, owned=owned))

    def __await__(self):
        owned, self.released = not self.released, True
        return self._wait(owned).__await__()

    def __repr__(self):
        return f"<BashHandle id={self.id}>"


class Bash:
    def __init__(self, host):
        self.host = host

    def __call__(self, command, *, cwd=None, timeout=None):
        import os

        job = self.host.call("bash.start", command=command, cwd=cwd or os.getcwd(), timeout=timeout)
        return BashHandle(self.host, job["id"])


class Capability:
    def __init__(self, bridge, methods):
        self.bridge, self.methods = bridge, methods

    def __getattr__(self, name):
        if name not in self.methods:
            raise AttributeError(name)
        tool, positional = self.methods[name]

        def call(*args, **kwargs):
            if len(args) > len(positional):
                raise TypeError(f"{name} accepts positional arguments {positional}")
            for key, value in zip(positional, args, strict=False):
                if key in kwargs:
                    raise TypeError(f"Duplicate argument {key}")
                kwargs[key] = value
            return self.bridge.call(tool, **kwargs)

        return call


class Edit(Capability):
    def __init__(self, host):
        super().__init__(
            host.bridge,
            {"apply_patch": ("apply_patch", ["patch"]), "rollback": ("edit_rollback", ["edit_id"])},
        )
        self.host = host

    async def __call__(self, path, old_str, new_str):
        return await self.run(path, old_str, new_str)

    async def run(self, path, old_str, new_str):
        # Match ordinary Python file semantics after os.chdir(), not the daemon cwd.
        path = Path(path).expanduser().absolute()  # noqa: ASYNC240 - kernel-local cwd metadata
        return await self.host.acall("edit", path=str(path), old_str=old_str, new_str=new_str)


class Goal:
    def __init__(self, host):
        self.host = host

    async def get(self):
        return await self.host.acall("goal.get")

    async def create(self, objective, token_budget=None):
        return await self.host.acall("goal.create", objective=objective, token_budget=token_budget)

    async def complete(self):
        return await self.host.acall("goal.complete")


class Skills:
    def __init__(self, host, values, records):
        self.host, self.values = host, values
        self.records = {s["name"]: s for s in records}

    def list(self):
        return self.host.call("skills.list")

    def load(self, name):
        from .skills import load_module

        entry = self.host.call("skills.load", name=name)
        if entry.get("import_name"):
            module = load_module(entry)
            self.values[entry["import_name"]] = module
            return module
        return Record(entry)

    async def run(self, name, **inputs):
        # Preparation checks permissions/schema/quarantine and returns code, never
        # recursively enters the worker while this cell already owns it.
        ticket = await self.host.acall("skills.prepare", name=name, inputs=inputs)
        passed = False
        try:
            if ticket.get("import_name"):
                module = self.load(name)
                result = module.run(**inputs)
                if inspect.isawaitable(result):
                    result = await result
            else:
                import ast

                self.values["skill_inputs"] = inputs
                result = eval(
                    compile(
                        ticket["code"], "<skill>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
                    ),
                    self.values,
                )
                if inspect.isawaitable(result):
                    await result
                result = self.values.get("skill_result")
            passed = True
            return result
        finally:
            await self.host.acall("skills.outcome", ticket=ticket["ticket"], passed=passed)


def bootstrap(bridge, values, metadata):
    """Construct namespaces once per kernel; restore user values only afterwards."""
    import json
    import os
    import pathlib

    host = Host(bridge)
    recursive = Recursive(host)
    values.update(
        asyncio=asyncio,
        os=os,
        pathlib=pathlib,
        Path=Path,
        json=json,
        context={"task": metadata.get("task", ""), "messages_path": metadata.get("messages_path")},
        session=Record({k: v for k, v in metadata.items() if k not in {"task", "skills"}}),
        rlm=recursive,
        harness=recursive.harness,
        bash=Bash(host),
        agent_message=Messaging(host),
        agent_observe=Observation(host),
        mcp=Mcp(host),
        edit=Edit(host),
        goal=Goal(host),
    )
    values["repo"] = Capability(
        bridge,
        {
            "map": ("repo_map", []),
            "search": ("repo_search", ["query"]),
            "symbols": ("symbol_search", ["query"]),
            "references": ("references_search", ["query"]),
            "outline": ("file_outline", ["path"]),
            "dependencies": ("dependency_context", ["path"]),
            "dependents": ("repo_dependents", ["path"]),
            "definition": ("repo_definition", ["query"]),
            "callers": ("repo_callers", ["query"]),
            "callees": ("repo_callees", ["query"]),
            "context_for_symbol": ("repo_context_for_symbol", ["query"]),
            "changed_symbols": ("repo_changed_symbols", []),
        },
    )
    values["git"] = Capability(
        bridge,
        {
            "diff": ("git_diff", []),
            "status": ("git_status", []),
            "checkpoint": ("git_checkpoint", ["label"]),
            "restore": ("git_restore", ["checkpoint_id"]),
        },
    )
    values["history"] = Capability(
        bridge,
        {
            "read": ("history_read", []),
            "search": ("history_search", ["query"]),
            "messages": ("message_history", []),
            "get": ("history_get", ["event_id"]),
        },
    )
    values["artifacts"] = Capability(
        bridge,
        {
            "load": ("artifact_load", ["artifact_id"]),
            "read": ("artifact_read", ["artifact_id"]),
            "search": ("artifact_search", ["query"]),
        },
    )
    for namespace, command in (
        ("tests", "run_tests"),
        ("build", "run_build"),
        ("bench", "run_benchmark"),
    ):
        methods = {"run": (command, [])}
        if namespace == "tests":
            methods.update(
                related_to=("related_tests", ["files"]),
                targeted=("run_targeted_tests", ["targets"]),
            )
        values[namespace] = Capability(bridge, methods)
    values["skills"] = Skills(host, values, metadata.get("skills", []))

    async def compact():
        return await host.acall("context.compact")

    async def refine():
        return await host.acall("harness.refine")

    async def heartbeat(**options):
        return await bridge.acall("schedule_turn", **options)

    values.update(compact=compact, refine=refine, heartbeat=heartbeat)
    compact.run, refine.run, heartbeat.run = compact, refine, heartbeat
    # Python skill packages are imported and callable in every session. Failures
    # become inspectable placeholders, never silently disappear.
    from .skills import UnavailableSkill, load_module

    for entry in metadata.get("skills", []):
        if entry.get("import_name"):
            try:
                values[entry["import_name"]] = load_module(entry)
            except Exception as exc:
                values[entry["import_name"]] = UnavailableSkill(entry["name"], str(exc))
    return host


def snapshot_handle(value):
    if isinstance(value, AgentHandle):
        return ["agent_handle", asdict(value)]
    if isinstance(value, BashHandle):
        return ["bash_handle", value.id]
    return None
