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


class CapabilityResult(Record):
    """Already-completed mapping; optional await never repeats the host action."""

    def __await__(self):
        async def completed():
            return self

        return completed().__await__()


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

    async def __call__(
        self,
        prompt=None,
        *args,
        name=None,
        model=None,
        thinking=None,
        purpose="shared",
        isolate=None,
        adapter=None,
    ):
        if not isinstance(prompt, str) or not prompt.strip() or args:
            raise ValueError(self.help())
        return await self.run(
            prompt,
            name=name,
            model=model,
            thinking=thinking,
            purpose=purpose,
            isolate=isolate,
            adapter=adapter,
        )

    def help(self):
        return 'await rlm("assignment", name="child", purpose="shared", isolate=False) returns a persistent handle. await agents.wait(seconds=30) waits for a message or timeout. agent_message sends/receives messages; agent_observe reads child trajectories.'

    def __repr__(self):
        return self.help()

    async def run(
        self,
        prompt,
        *,
        name=None,
        model=None,
        thinking=None,
        purpose="shared",
        isolate=None,
        adapter=None,
    ):
        result = await self.host.acall(
            "rlm.run",
            prompt=prompt,
            name=name,
            model=model,
            thinking=thinking,
            purpose=purpose,
            isolate=isolate,
            adapter=adapter,
        )
        return AgentHandle(**result)

    async def list_subagents(self):
        return [Record(s) for s in await self.host.acall("rlm.list_subagents")]

    async def wait(self, seconds=30):
        return await self.host.bridge.acall("agent_wait", seconds=seconds)

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

    def help(self):
        return 'await agent_message.send("findings", receiver_role="parent")\nFor child/sibling: receiver_role="child"|"sibling", receiver_name="name".\nawait agent_message.receive(); await agent_message.list_agents()'

    def __repr__(self):
        return self.help()

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

    get = get_agent

    def help(self):
        return "await agent_observe.get(session_id); await agent_observe.list_agents(); await agent_observe.recent_messages(session_id, limit=8)"

    async def requests(self, target):
        return await self.host.acall("agent_observe.requests", target=target)

    async def recent(self, target, limit=8, max_chars=800):
        return await self.recent_messages(target, limit=limit, max_chars=max_chars)

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
        self.bridge = bridge
        schemas = getattr(bridge, "argument_schemas", None)
        self.methods = {
            name: spec for name, spec in methods.items() if schemas is None or spec[0] in schemas
        }

    def __repr__(self):
        return self.help()

    def __dir__(self):
        return sorted({*self.methods, "help"})

    def help(self, name=None):
        if name:
            return getattr(self, name).__doc__
        if not self.methods:
            return "No admitted methods"
        primary = next(iter(self.methods))
        if "search" in self.methods:
            primary = "search"
        return (
            f"{primary}{inspect.signature(getattr(self, primary))}\n"
            + "Methods: "
            + ", ".join(self.methods)
            + '\nhelp("method") shows exact arguments. Keep results in Python; print selected evidence.'
        )

    def __getattr__(self, name):
        if name not in self.methods:
            raise AttributeError(name)
        tool, positional = self.methods[name]
        schema = getattr(self.bridge, "argument_schemas", {}).get(tool)
        properties = schema.get("properties", {}) if schema else {}

        def call(*args, **kwargs):
            if len(args) > len(positional):
                raise TypeError(f"{name} accepts positional arguments {positional}")
            for key, value in zip(positional, args, strict=False):
                if key in kwargs:
                    raise TypeError(f"Duplicate argument {key}")
                kwargs[key] = value
            if schema and (unknown := kwargs.keys() - properties.keys()):
                raise TypeError(
                    f"Unknown arguments {sorted(unknown)} for {name}{call.__signature__}. {('Use path= for a file glob.' if 'glob' in unknown and 'path' in properties else '')}"
                )
            result = self.bridge.call(tool, **kwargs)
            return CapabilityResult(result) if isinstance(result, dict) else result

        call.__name__ = name
        call.__signature__ = inspect.Signature(
            [
                *[
                    inspect.Parameter(
                        p,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=properties.get(p, {}).get("default", inspect.Parameter.empty),
                    )
                    for p in positional
                ],
                *(
                    [
                        inspect.Parameter(
                            p,
                            inspect.Parameter.KEYWORD_ONLY,
                            default=detail.get(
                                "default",
                                inspect.Parameter.empty
                                if p in schema.get("required", [])
                                else None,
                            ),
                        )
                        for p, detail in properties.items()
                        if p not in positional
                    ]
                    if schema
                    else [inspect.Parameter("options", inspect.Parameter.VAR_KEYWORD)]
                ),
            ]
        )
        call.__doc__ = f"{name}{call.__signature__}. Internal capability: {tool}. Retain the result and print a bounded selection."
        bounds = {
            p: {
                k: v
                for k, v in detail.items()
                if k in {"minimum", "maximum", "enum", "minLength", "maxLength"}
            }
            for p, detail in properties.items()
        }
        bounds = {p: b for p, b in bounds.items() if b}
        if bounds:
            call.__doc__ += f" Bounds: {bounds}."
        return call


class ContextView(Record):
    def __init__(self, host, **values):
        super().__init__(values)
        self.host = host

    def help(self):
        return 'context.search("query") retrieves history; context["task"] holds the complete objective.'

    def search(self, query, limit=5):
        return self.host.bridge.call("history_search", query=query, limit=limit)


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
    bridge.argument_schemas = metadata.get("argument_schemas", {})
    recursive = Recursive(host)
    values.update(
        asyncio=asyncio,
        os=os,
        pathlib=pathlib,
        Path=Path,
        json=json,
        context=ContextView(
            host, task=metadata.get("task", ""), messages_path=metadata.get("messages_path")
        ),
        session=Record(
            {k: v for k, v in metadata.items() if k not in {"task", "skills", "argument_schemas"}}
        ),
        rlm=recursive,
        agents=recursive,
        harness=recursive.harness,
        bash=Bash(host),
        agent_message=Messaging(host),
        agent_observe=Observation(host),
        mcp=Mcp(host),
        goal=Goal(host),
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
    values["skills"] = Skills(host, values, metadata.get("skills", []))
    values["shell"] = values["bash"]

    class Verification:
        @staticmethod
        def help():
            return "await verify.run(): run the independent configured verifier; does not override its gates."

        async def run(self):
            return await host.acall("verification.run")

        def __repr__(self):
            return self.help()

    values["verify"] = Verification()

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
    import importlib

    for reference in metadata.get("namespace_factories", []):
        module, name = reference.split(":", 1)
        getattr(importlib.import_module(module), name)(bridge, values, metadata, host)
    return host


def snapshot_handle(value):
    if isinstance(value, AgentHandle):
        return ["agent_handle", asdict(value)]
    if isinstance(value, BashHandle):
        return ["bash_handle", value.id]
    return None
