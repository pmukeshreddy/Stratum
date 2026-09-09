"""Adapter composition and generic capability, preparation and isolation lifecycle."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from .artifacts import atomic_write
from .capabilities import ChildProfile, default_adapters
from .models import HarnessError, Outcome, Workspace, new_id
from .storage import encode
from .tools import ToolContext


class Environment:
    def __init__(self, runtime, adapters=None):
        self.runtime = runtime
        self.adapters = adapters if adapters is not None else default_adapters()
        self.providers = {}
        self._registered = set()
        self._bound = set()
        self.refresh()

    def refresh(self):
        for adapter in self.adapters.values():
            if id(adapter) in self._bound:
                continue
            self._bound.add(id(adapter))
            if hasattr(adapter, "bind"):
                adapter.bind(self.runtime)
            for provider in getattr(adapter, "providers", lambda: ())():
                self.register_capability(provider)

    def register_capability(self, provider):
        if provider.name in self.providers and self.providers[provider.name] != provider:
            raise ValueError(f"Capability already registered: {provider.name}")
        self.providers[provider.name] = provider

    def capability(self, sid, name):
        if name not in self.runtime.store.config(sid).effective_capabilities:
            raise PermissionError(f"Capability not admitted: {name}")
        return self.providers[name].service

    def adapter(self, sid):
        return self.adapters[self.runtime.store.config(sid).task.adapter]

    def call(self, sid, hook, *args, default=None, **kwargs):
        function = getattr(self.adapter(sid), hook, None)
        return function(*args, **kwargs) if function else default

    def configure(self, config, *, defaults=True):
        self.refresh()
        adapter = self.adapters[config.task.adapter]
        if hasattr(adapter, "validate"):
            adapter.validate(config)
        if defaults and hasattr(adapter, "configure"):
            adapter.configure(config)
        effective = set(getattr(adapter, "capabilities", ())) | set(config.capabilities)
        effective -= set(config.disabled_capabilities)
        missing = effective - self.providers.keys()
        if missing:
            raise ValueError(f"Unknown capabilities: {sorted(missing)}")
        config.effective_capabilities = sorted(effective)
        for name in config.effective_capabilities:
            if name not in self._registered:
                before = set(self.runtime.tools.entries)
                self.providers[name].register_tools(self.runtime.tools)
                for tool_name in self.runtime.tools.entries.keys() - before:
                    tool = self.runtime.tools.entries[tool_name]
                    if tool.capability not in (None, name):
                        raise ValueError(
                            f"Capability provider {name} registered foreign capability {tool.capability}"
                        )
                    tool.capability = name
                self._registered.add(name)

    def namespace_factories(self, config):
        return [
            self.providers[n].namespace_factory
            for n in config.effective_capabilities
            if self.providers[n].namespace_factory
        ]

    def instructions(self, config):
        return "\n".join(
            self.providers[n].instructions(config)
            for n in config.effective_capabilities
            if self.providers[n].instructions
        )

    def profiles(self, config):
        return {
            "shared": ChildProfile(isolate=False),
            **getattr(self.adapters[config.task.adapter], "profiles", {}),
        }

    def child_policy(self, config, purpose, isolate):
        if purpose is not None:
            profiles = self.profiles(config)
            if purpose not in profiles:
                raise ValueError(
                    f"Unknown child profile {purpose!r} for {config.task.adapter}; available: {sorted(profiles)}"
                )
            profile = profiles[purpose]
            if profile.require_isolation and isolate is False:
                raise ValueError(f"Child profile {purpose} requires an isolated workspace")
            if isolate is None:
                isolate = profile.isolate
        else:
            profile = ChildProfile()
        if isolate is None:
            adapter = self.adapters[config.task.adapter]
            isolate = getattr(adapter, "default_isolation", lambda c, **kw: False)(
                config, child=True
            )
        return profile, isolate

    async def prepare(self, sid, *, force=False):
        runtime, store = self.runtime, self.runtime.store
        config, session = store.config(sid), store.session(sid)
        if session.workspace.metadata.get("admission_pending"):
            raise HarnessError(
                "environment", "workspace_not_ready", "Workspace admission has not completed"
            )
        if store.events(sid, kind="environment_prepared", limit=1):
            return
        if not force and self.call(sid, "defer_prepare", session, default=False):
            return
        event = store.event(sid, "environment_prepare", {"adapter": config.task.adapter})
        try:
            async with asyncio.timeout(config.limits.tool_timeout_seconds):
                result = await self.adapter(sid).prepare(
                    ToolContext(runtime, sid, new_id(), event), config.task
                )
        except Exception as exc:
            raise HarnessError("environment", "prepare_failed", str(exc)) from exc
        eid = store.event(sid, "environment_prepared", {"result": result}, parent=event)
        selected = runtime.artifacts.expose(sid, result, source_event=eid)
        store.add_context(
            sid, eid, [{"role": "user", "content": "Environment: " + encode(selected)}]
        )

    async def before_action(self, context, name):
        function = getattr(self.adapter(context.session_id), "before_action", None)
        return await function(context, name) if function else None

    def after_action(self, context, token):
        return self.call(context.session_id, "after_action", context, token)

    async def before_python(self, context):
        function = getattr(self.adapter(context.session_id), "before_python", None)
        return await function(context) if function else None

    def after_python(self, context, token):
        return self.call(context.session_id, "after_python", context, token)

    def write(self, context, path, content):
        function = getattr(self.adapter(context.session_id), "write", None)
        if function:
            return function(context, path, content)
        target = context.path(path)
        atomic_write(target, content)
        return {"path": str(target), "bytes": len(content)}

    def continuation_workspace(self, source, config, *, child=False, isolate=None):
        adapter = self.adapters[config.task.adapter]
        if isolate is None:
            isolate = getattr(adapter, "default_isolation", lambda c, **kw: False)(
                config, child=child
            )
        if not isolate:
            return source.workspace.model_copy(deep=True), None
        if hasattr(adapter, "continuation_workspace"):
            return adapter.continuation_workspace(source, config, child=child, isolate=True)
        return self.copy_workspace(source)

    def copy_workspace(self, source):
        destination = self.runtime.store.directory / "workspaces" / new_id()
        private = self.runtime.store.directory.resolve()

        def ignore(directory, names):
            return [
                name for name in names if (Path(directory) / name).resolve().is_relative_to(private)
            ]

        shutil.copytree(source.workspace.path, destination, symlinks=True, ignore=ignore)
        return Workspace(
            path=str(destination),
            metadata={"source_session": source.id, "isolation": "workspace_copy"},
        ), None

    def isolated_child_workspace(self, parent_id, child_id):
        from .storage import Store

        store = Store(self.runtime.store.directory)
        try:
            adapter = self.adapters[store.config(child_id).task.adapter]
            if hasattr(adapter, "isolated_child_workspace"):
                return adapter.isolated_child_workspace(parent_id, child_id)
            workspace, checkpoint = self.copy_workspace(store.session(parent_id))
            store.event(
                child_id,
                "workspace_lease",
                {"workspace": workspace.model_dump(), "checkpoint": checkpoint},
            )
            return workspace, checkpoint
        finally:
            store.close()

    def poll(self):
        for adapter in self.adapters.values():
            if hasattr(adapter, "poll"):
                adapter.poll()

    def close(self):
        for adapter in self.adapters.values():
            if hasattr(adapter, "runtime_close"):
                adapter.runtime_close()

    async def recover(self):
        from .execution import recover_containers
        from .process_family import cleanup_registry

        for session in self.runtime.store.sessions():
            kernel = self.runtime.kernels.get(session.id)
            if not kernel or not kernel.process or kernel.process.returncode is not None:
                cleanup_registry(self.runtime.store.directory / "kernels" / session.kernel_id)
            if session.workspace.metadata.get("admission_pending"):
                self.runtime.store.update(
                    session.id, outcome=Outcome.FAILED, paused=True, runnable=False
                )
                self.runtime.store.event(
                    session.id,
                    "workspace_recovery",
                    {"status": "admission_interrupted", "replayed": False},
                )
        await recover_containers(self.runtime)
        for adapter in self.adapters.values():
            if hasattr(adapter, "recover"):
                await adapter.recover()
