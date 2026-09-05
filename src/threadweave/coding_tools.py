"""Typed coding primitives. Selection and sequencing remain model-controlled."""

from __future__ import annotations

import json
import shutil

from pydantic import Field

from .coding import baseline, run_checks, run_command
from .diagnostics import localize
from .editing import Editor
from .experiments import Experiments
from .gitops import GitWorkspace, candidate_result, git, revision
from .models import BenchmarkConfig, Record
from .retrieval import search
from .tools import Empty, PathArgs, Tool


class SearchArgs(Record):
    query: str = Field(min_length=1, max_length=1000)
    regex: bool = False
    path: str = "*"
    language: str | None = None
    context: int = Field(default=2, ge=0, le=10)
    limit: int = Field(default=50, ge=1, le=200)


class QueryArgs(Record):
    query: str
    limit: int = Field(default=20, ge=1, le=100)


class HistorySearchArgs(QueryArgs):
    kind: str | None = None
    session_id: str | None = None


class PatchArgs(Record):
    patch: str = Field(min_length=1, max_length=4_000_000)


class RangeArgs(Record):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
    expected_hash: str


class CreateArgs(Record):
    path: str
    content: str


class DeleteArgs(Record):
    path: str
    expected_hash: str


class MoveArgs(Record):
    source: str
    destination: str


class EditArgs(Record):
    edit_id: str


class CheckpointArgs(Record):
    label: str = "agent-checkpoint"


class RestoreArgs(Record):
    checkpoint_id: str
    paths: list[str] | None = None


class DiffArgs(Record):
    checkpoint_id: str | None = None


class RevisionArgs(Record):
    revision: str = "HEAD"
    path: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class TargetArgs(Record):
    targets: list[str] = Field(min_length=1, max_length=100)


class ResultArgs(Record):
    artifact_id: str


class ExperimentCreateArgs(Record):
    hypothesis: str = Field(min_length=1, max_length=5000)
    changes: str = Field(max_length=10000)
    metric: BenchmarkConfig | None = None
    verifier: list[list[str]] = Field(min_length=1)


class ExperimentArgs(Record):
    experiment_id: str


class ExperimentRunArgs(ExperimentArgs):
    conclusion: str | None = None


class CompareArgs(Record):
    first: str
    second: str


class CandidateArgs(Record):
    child_id: str


class ProfileArgs(Record):
    command: list[str] = Field(min_length=1)
    profiler: str = "configured"


def register(registry):
    def add(
        name,
        description,
        args,
        handler,
        permissions=("workspace.read",),
        *,
        feature=None,
        coding=True,
    ):
        registry.register(
            Tool(name, description, args, handler, permissions, coding_only=coding, feature=feature)
        )

    async def repo_map(c, a):
        return c.runtime.index(c.session_id).repo_map()

    async def repo_search(c, a):
        result = c.runtime.index(c.session_id).search(**a.model_dump())
        if (full := result.pop("all_matches", None)) is not None:
            result["full_results_artifact"] = c.runtime.artifacts.put(c.session_id, full)
        return result

    async def symbols(c, a):
        return c.runtime.index(c.session_id).symbol_search(**a.model_dump())

    async def references(c, a):
        import re

        return c.runtime.index(c.session_id).search(
            r"\b" + re.escape(a.query) + r"\b", regex=True, limit=a.limit
        )

    async def outline(c, a):
        return c.runtime.index(c.session_id).outline(a.path)

    async def dependencies(c, a):
        return c.runtime.index(c.session_id).dependencies(a.path)

    async def patch(c, a):
        return Editor(c).apply_patch(a.patch)

    async def replace(c, a):
        from .repository import digest

        before = Editor(c).read(a.path)
        if before is None or digest(before) != a.expected_hash:
            raise ValueError("File hash changed; inspect the current file before replacing")
        lines = before.decode().splitlines(True)
        if not 1 <= a.start_line <= a.end_line <= len(lines):
            raise ValueError("Invalid 1-based inclusive line range")
        return Editor(c).apply(
            {
                a.path: (
                    "".join(lines[: a.start_line - 1]) + a.content + "".join(lines[a.end_line :])
                ).encode()
            }
        )

    async def create(c, a):
        if c.path(a.path).exists():
            raise ValueError("File already exists; use a validated edit")
        return Editor(c).apply({a.path: a.content.encode()})

    async def delete(c, a):
        from .repository import digest

        before = Editor(c).read(a.path)
        if before is None or digest(before) != a.expected_hash:
            raise ValueError("Deletion hash mismatch")
        return Editor(c).apply({a.path: None})

    async def move(c, a):
        before = Editor(c).read(a.source)
        if before is None or c.path(a.destination).exists():
            raise ValueError("Move needs an existing source and absent destination")
        return Editor(c).apply(
            {a.source: None, a.destination: before},
            modes={a.destination: c.path(a.source).stat().st_mode & 0o777},
        )

    async def rollback(c, a):
        return Editor(c).rollback(a.edit_id)

    async def diff(c, a):
        text = GitWorkspace(c).diff(a.checkpoint_id)
        return {
            "patch": text[:12000],
            "bytes": len(text.encode()),
            "artifact_id": c.runtime.artifacts.put_bytes(
                c.session_id, text.encode(), "text/x-diff"
            ),
        }

    async def status(c, a):
        return GitWorkspace(c).status()

    async def show(c, a):
        if a.path:
            c.path(a.path)
        return {
            "text": git(
                c.session.workspace.path,
                "show",
                "--no-ext-diff",
                "--no-textconv",
                revision(a.revision),
                "--",
                *([a.path] if a.path else []),
            )
        }

    async def log(c, a):
        return {
            "text": git(
                c.session.workspace.path,
                "log",
                f"-{a.limit}",
                "--format=%h %ad %s",
                "--date=iso",
                revision(a.revision),
                "--",
            )
        }

    async def checkpoint(c, a):
        return {"checkpoint_id": GitWorkspace(c).snapshot(a.label)}

    async def restore(c, a):
        return GitWorkspace(c).restore(a.checkpoint_id, a.paths)

    async def targeted(c, a):
        return await run_checks(c, "test", targets=a.targets)

    async def failure(c, a):
        return localize(c, c.runtime.artifacts.load(c.session_id, a.artifact_id))

    async def history(c, a):
        result = search(c.runtime.store, c.session_id, **a.model_dump())
        c.runtime.store.event(
            c.session_id,
            "history_retrieval",
            {"query": a.query, "matches": [r["id"] for r in result]},
            parent=c.source_event,
        )
        return result

    async def artifacts(c, a):
        return search(c.runtime.store, c.session_id, a.query, kind="artifact", limit=a.limit)

    async def benchmark(c, a):
        from .benchmarks import run_benchmark

        config = c.runtime.store.config(c.session_id).task.benchmark
        if not config:
            raise ValueError("Configure task.benchmark to run measured benchmarks")
        return await run_benchmark(c, config, reference=baseline(c)["benchmark"])

    async def experiment_create(c, a):
        return Experiments(c).create(**a.model_dump())

    async def experiment_run(c, a):
        return await Experiments(c).run(a.experiment_id, a.conclusion)

    async def experiment_result(c, a):
        return Experiments(c).get(a.experiment_id)

    async def experiment_list(c, a):
        return Experiments(c).list()

    async def experiment_compare(c, a):
        return Experiments(c).compare(a.first, a.second)

    async def candidate(c, a):
        return candidate_result(c, a.child_id)

    async def accept(c, a):
        return candidate_result(c, a.child_id, accept=True)

    async def skill_search(c, a):
        return [
            {k: e[k] for k in ("id", "title", "version", "content")}
            for e in c.runtime.store.states(c.session_id)
            if e["kind"] == "skill" and a.query.lower() in json.dumps(e).lower()
        ][: a.limit]

    async def capabilities(c, a):
        return {
            name: shutil.which(name)
            for name in ("nvcc", "ncu", "nsys", "gcc", "clang", "cmake", "cargo", "go", "node")
        }

    async def profile(c, a):
        task = c.runtime.store.config(c.session_id).task
        if a.profiler == "configured":
            if not task.profiler_command:
                raise ValueError("Configure task.profiler_command or select an installed ncu/nsys")
            prefix = task.profiler_command
        elif a.profiler == "ncu" and shutil.which("ncu"):
            prefix = ["ncu", "--csv"]
        elif a.profiler == "nsys" and shutil.which("nsys"):
            prefix = ["nsys", "profile", "--stats=true"]
        else:
            raise ValueError("Requested profiler is unavailable")
        from .diagnostics import profiler_metrics

        result = await run_command(c, [*prefix, *a.command], kind="profile")
        result["metrics"] = profiler_metrics(
            c.runtime.artifacts.load(c.session_id, result["stdout_artifact"])
        )
        c.runtime.store.event(c.session_id, "profile_result", result, parent=c.source_event)
        return result

    async def import_artifact(c, a):
        path = c.path(a.path)
        with path.open("rb") as stream:
            aid = c.runtime.artifacts.put_stream(
                c.session_id,
                stream,
                source_event=c.source_event,
                media_type="application/octet-stream",
            )
        return c.runtime.artifacts.metadata(c.session_id, aid)

    for name, desc, args, handler in [
        (
            "repo_map",
            "Map source files, languages, configs and test entry points.",
            Empty,
            repo_map,
        ),
        (
            "repo_search",
            "Search code with path/language filters; large matches retained.",
            SearchArgs,
            repo_search,
        ),
        (
            "symbol_search",
            "Find Python AST or lexical multi-language declarations.",
            QueryArgs,
            symbols,
        ),
        (
            "references_search",
            "Find likely lexical usages, not a compiler call graph.",
            QueryArgs,
            references,
        ),
        ("file_outline", "Inspect declarations/imports with line numbers.", PathArgs, outline),
        (
            "dependency_context",
            "Retrieve imports and likely local modules.",
            PathArgs,
            dependencies,
        ),
        ("git_status", "Inspect HEAD and tracked/untracked status.", Empty, status),
        ("git_diff", "Inspect the current patch; full patch artifact retained.", DiffArgs, diff),
        ("inspect_diff", "Inspect current changes or changes since a checkpoint.", DiffArgs, diff),
        (
            "git_show",
            "Read a Git revision or file without invoking external diff programs.",
            RevisionArgs,
            show,
        ),
        ("git_log", "Inspect bounded commit history.", RevisionArgs, log),
        (
            "failure_localize",
            "Connect diagnostic artifacts to source definitions and recent edits.",
            ResultArgs,
            failure,
        ),
        (
            "execution_capabilities",
            "Detect installed compilers and optional GPU profilers.",
            Empty,
            capabilities,
        ),
    ]:
        add(name, desc, args, handler)
    for name, args, handler in [
        ("apply_patch", PatchArgs, patch),
        ("replace_range", RangeArgs, replace),
        ("create_file", CreateArgs, create),
        ("delete_file", DeleteArgs, delete),
        ("move_file", MoveArgs, move),
        ("edit_rollback", EditArgs, rollback),
        ("git_checkpoint", CheckpointArgs, checkpoint),
        ("git_restore", RestoreArgs, restore),
    ]:
        add(
            name,
            "Validated, journaled repository edit/checkpoint with recoverable prior content.",
            args,
            handler,
            ("workspace.write",),
        )
    for kind in ("test", "build", "lint", "typecheck"):

        async def check(c, a, kind=kind):
            return await run_checks(c, kind)

        add(
            "run_tests" if kind == "test" else "run_" + kind,
            "Execute configured " + kind + " commands and retain structured diagnostics.",
            Empty,
            check,
            ("process",),
        )
    add(
        "run_targeted_tests",
        "Append validated file/test targets to configured test commands.",
        TargetArgs,
        targeted,
        ("process",),
    )
    add(
        "run_benchmark",
        "Run correctness-gated repeated measurements against the baseline.",
        Empty,
        benchmark,
        ("process",),
    )
    add(
        "run_profile",
        "Run an installed/configured profiler, retaining its outputs.",
        ProfileArgs,
        profile,
        ("process",),
    )
    add(
        "artifact_import",
        "Retain an exact workspace file or binary profiler report as an artifact.",
        PathArgs,
        import_artifact,
    )
    add(
        "history_search",
        "Search durable trajectory events and observations with FTS.",
        HistorySearchArgs,
        history,
        (),
        feature="history_retrieval",
        coding=False,
    )
    add(
        "artifact_search",
        "Search indexed artifact excerpts in this trajectory.",
        QueryArgs,
        artifacts,
        (),
        feature="history_retrieval",
        coding=False,
    )
    for name, args, handler in [
        ("experiment_create", ExperimentCreateArgs, experiment_create),
        ("experiment_run", ExperimentRunArgs, experiment_run),
        ("experiment_result", ExperimentArgs, experiment_result),
        ("experiment_list", Empty, experiment_list),
        ("experiment_compare", CompareArgs, experiment_compare),
    ]:
        add(
            name,
            "Create, execute or inspect durable experiments and measured evidence.",
            args,
            handler,
            ("process",),
            feature="experiments",
        )
    add(
        "candidate_inspect",
        "Consume findings and patch from a paused/completed isolated child.",
        CandidateArgs,
        candidate,
        ("agents",),
    )
    add(
        "candidate_apply",
        "Validate and apply an isolated child's patch; record acceptance.",
        CandidateArgs,
        accept,
        ("agents", "workspace.write"),
    )
    add(
        "skill_search",
        "Search validated reusable procedures.",
        QueryArgs,
        skill_search,
        ("state",),
        coding=False,
    )
