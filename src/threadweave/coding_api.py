"""Kernel bindings supplied only by the admitted coding capability."""

from pathlib import Path

from .kernel_api import Capability, ContextView, Recursive


class TestCapabilities(Capability):
    def for_file(self, path, **options):
        return self.related_to(files=[path], **options)

    def for_symbol(self, symbol, **options):
        return self.related_to(symbols=[symbol], **options)

    def failed_recently(self):
        return self.related_to(tier="failing")

    def help(self, name=None):
        return (
            super().help(name)
            + "\nfor_file(path), for_symbol(symbol), failed_recently(): ranked selection with provenance, not a replacement for final verification."
        )


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


class CodingContext(ContextView):
    def focus(self, *, files=(), symbols=(), hypothesis="", constraints=()):
        return self.host.call(
            "context.focus",
            files=list(files),
            symbols=list(symbols),
            hypothesis=hypothesis,
            constraints=list(constraints),
        )

    def help(self):
        return 'context.focus(files=[...], symbols=[...], hypothesis="...")\ncontext.search("current failure")\ncontext["task"] holds the complete objective.'


class CodingAgents(Recursive):
    async def candidate(self, handle, *, accept=False):
        return await self.host.bridge.acall(
            "candidate_apply" if accept else "candidate_inspect", child_id=handle.session_id
        )

    def help(self):
        return 'await rlm("Trace cause", name="reviewer") returns a persistent handle at admission. Put descriptions in the prompt or requirement. Optional purpose selects a registered profile: shared (default), research, review, candidate, test, performance. Research/review are read-only; candidate/test/performance use isolated worktrees.\nawait agents.wait(seconds=30) returns a pause receipt immediately; end the cell to defer the next model turn until a message or timeout.\nawait agent_message.send("findings", receiver_role="parent"); await agent_message.receive().\nawait agent_observe.get(handle.session_id); await agents.candidate(handle, accept=False) inspects, accept=True applies.'


def bindings(bridge, values, metadata, host):
    values["context"] = CodingContext(host, **dict(values["context"]))
    values["rlm"] = values["agents"] = CodingAgents(host)
    values["rlm"].harness = values["harness"]
    if "edit" in bridge.argument_schemas:
        values["edit"] = Edit(host)
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
            "declaration": ("repo_declaration", ["query"]),
            "implementations": ("repo_implementations", ["query"]),
            "related_symbols": ("repo_related_symbols", ["query"]),
            "resolve": ("repo_resolve", ["path", "line", "column"]),
            "semantic": ("repo_semantic", ["path", "line", "column"]),
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
    for namespace, command in (
        ("tests", "run_tests"),
        ("build", "run_build"),
        ("lint", "run_lint"),
        ("typecheck", "run_typecheck"),
        ("bench", "run_benchmark"),
    ):
        methods = {"run": (command, [])}
        if namespace == "tests":
            methods.update(
                related_to=("related_tests", ["files"]),
                targeted=("run_targeted_tests", ["targets"]),
                import_coverage=("test_coverage_import", ["path"]),
                selection_reason=("test_selection_reason", ["query"]),
            )
        values[namespace] = (TestCapabilities if namespace == "tests" else Capability)(
            bridge, methods
        )
    values["experiment"] = Capability(
        bridge,
        {
            "create": ("experiment_create", ["hypothesis", "changes"]),
            "run": ("experiment_run", ["experiment_id"]),
            "list": ("experiment_list", []),
        },
    )

    for name in list(values):
        value = values[name]
        if isinstance(value, Capability) and not value.methods:
            del values[name]
