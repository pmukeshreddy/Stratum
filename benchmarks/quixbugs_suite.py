"""Freeze unmodified public QuixBugs inputs, never reference implementations.

uv run python benchmarks/quixbugs_suite.py prepare --source /path/to/QuixBugs --output results/current-quality/quixbugs
uv run python benchmarks/quixbugs_suite.py run --output results/current-quality/quixbugs --parallel 4
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from threadweave.artifacts import atomic_write
from threadweave.evaluation import evaluate, source_identity
from threadweave.gitops import git
from threadweave.models import RunConfig


def freeze(source, output):
    """Pin inputs and production source for a fresh paired run, without changing tasks."""
    if output.exists():
        raise ValueError("Choose a new run directory; existing trajectories are immutable")
    output.mkdir(parents=True)
    for name in ("tasks.json", "config.json", "provenance.json"):
        shutil.copyfile(source / name, output / name)
    import threadweave

    package = Path(threadweave.__file__).resolve().parent
    shutil.copytree(
        package, output / "frozen/threadweave", ignore=shutil.ignore_patterns("__pycache__")
    )
    repository = Path(__file__).resolve().parent.parent
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(repository / name, output / "frozen" / name)
    shutil.copyfile(__file__, output / "frozen/run.py")
    atomic_write(output / "frozen/source.json", json.dumps(source_identity(), indent=2).encode())
    print(
        f"PYTHONPATH={output.resolve() / 'frozen'} uv run python {output.resolve() / 'frozen/run.py'} run --output {output.resolve()} --parallel 4",
        flush=True,
    )


def prepare(source, output, count):
    output.mkdir(parents=True, exist_ok=True)
    root = output / "input"
    if root.exists():
        raise ValueError("Frozen input already exists; choose a new output directory")
    root.mkdir()
    for name in ("python_programs", "python_testcases", "json_testcases"):
        shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("conftest.py", "LICENSE"):
        shutil.copyfile(source / name, root / name)
    atomic_write(root / ".gitignore", b"__pycache__/\n.pytest_cache/\n")
    provenance = {
        "source": "https://github.com/jkoppel/QuixBugs",
        "commit": git(source, "rev-parse", "HEAD").strip(),
        "license": "MIT",
        "selection": f"first {count} test modules in sorted path order, fixed before inference",
        "excluded": ["correct_python_programs", "correct_java_programs", "upstream Git history"],
        "metric": "unmodified official pytest cases pass; default upstream slow-test exclusions retained; pytest-timeout 5 seconds per case",
        "file_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        },
    }
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Evaluation",
        "-c",
        "user.email=local@localhost",
        "commit",
        "-qm",
        "Frozen public defective programs and supplied tests",
    )
    provenance["input_revision"] = git(root, "rev-parse", "HEAD").strip()
    cases = sorted((root / "python_testcases").glob("test_*.py"))[:count]
    instances = []
    evaluator = {
        str(p.relative_to(root)): str(p.resolve())
        for folder in (root / "python_testcases", root / "json_testcases")
        for p in folder.rglob("*")
        if p.is_file()
    }
    evaluator["conftest.py"] = str((root / "conftest.py").resolve())
    for case in cases:
        name = case.stem.removeprefix("test_")
        command = [sys.executable, "-m", "pytest", "-q", "--timeout=5", str(case.relative_to(root))]
        instances.append(
            {
                "id": "quixbugs-" + name,
                "adapter": "repository_issue",
                "repository": str(root.resolve()),
                "base_commit": provenance["input_revision"],
                "objective": f"Repair python_programs/{name}.py so that its documented behavior and the supplied official tests are satisfied. Inspect evidence, make the necessary code change, and validate it. Do not modify tests, test data, conftest.py, or support infrastructure. The evaluator uses frozen copies of those files. Use the supplied test command: {' '.join(command)}. Do not fetch reference solutions or inspect other filesystem workspaces. Work only in this supplied repository.",
                "test_commands": [command],
                "verifier_files": evaluator,
            }
        )
    config = RunConfig(
        provider={
            "name": "codex_subscription",
            "model": "gpt-6-astra",
            "parameters": {"reasoning_effort": "medium"},
            "max_output_tokens": 4096,
            "timeout_seconds": 150,
        },
        task={
            "adapter": "coding",
            "capture_baseline": True,
            "verify_each_turn": False,
            "require_tests": True,
            "require_change": True,
        },
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        limits={
            "max_turns": 12,
            "token_budget": 50000,
            "wall_seconds": 300,
            "tool_timeout_seconds": 60,
            "concurrency": 4,
            "max_subagents": 3,
        },
        refinement={"automatic": False},
        execution={"environment_allowlist": ["PATH", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV"]},
    )
    atomic_write(output / "provenance.json", json.dumps(provenance, indent=2).encode())
    atomic_write(output / "tasks.json", json.dumps(instances, indent=2).encode())
    atomic_write(output / "config.json", config.model_dump_json(indent=2).encode())
    print(
        json.dumps({"tasks": len(instances), "input": str(root), "upstream": provenance["commit"]}),
        flush=True,
    )


async def run(output, parallel, count):
    # No API-key provider/fallback. Auth is owned by the official subscription client.
    os.environ.pop("OPENAI_API_KEY", None)
    config = RunConfig.model_validate_json((output / "config.json").read_text())
    if not {"python", "process", "workspace.read", "workspace.write"}.issubset(config.permissions):
        raise ValueError(
            "The coding evaluation requires Python, process, workspace.read and workspace.write permissions"
        )
    tasks = json.loads((output / "tasks.json").read_text())[:count]
    atomic_write(output / "harness-source.json", json.dumps(source_identity(), indent=2).encode())
    gate = asyncio.Semaphore(parallel)

    async def one(instance, profile):
        async with gate:
            task = output / "requests" / f"{instance['id']}-{profile}.json"
            atomic_write(task, json.dumps([instance]).encode())
            print(
                json.dumps(
                    {
                        "started": instance["id"],
                        "profile": profile,
                        "provider": config.provider.model_dump(),
                        "limits": config.limits.model_dump(),
                    }
                ),
                flush=True,
            )
            result = await evaluate(
                task, config, output / profile, output=output / f"{profile}.jsonl", profile=profile
            )
            print(
                json.dumps({"finished": instance["id"], "profile": profile, **result}), flush=True
            )

    # Interleave profiles to reduce effects from changing service load.
    await asyncio.gather(
        *(
            one(task, profile)
            for i, task in enumerate(tasks)
            for profile in (("base", "buffalo") if i % 2 == 0 else ("buffalo", "base"))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.output.resolve(), args.count)
    elif args.action == "freeze":
        freeze(args.source, args.output.resolve())
    else:
        asyncio.run(run(args.output.resolve(), args.parallel, args.count))
