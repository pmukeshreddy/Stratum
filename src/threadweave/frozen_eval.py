"""Frozen task/config comparison runner. python -m threadweave.frozen_eval --help."""

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from .artifacts import atomic_write
from .evaluation import Instance, analyze, evaluate, source_identity
from .gitops import git
from .machine import metadata
from .models import RunConfig
from .storage import encode


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(tasks, config, destination):
    if "process" not in config.permissions:
        raise ValueError(
            "Coding comparison requires explicit process permission for independent verification"
        )
    tasks, destination = Path(tasks).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Frozen bundle already exists; choose a new destination")
    destination.mkdir(parents=True, mode=0o700)
    shutil.copytree(
        Path(__file__).parent,
        destination / "runtime" / "threadweave",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    entries = [Instance.model_validate(item) for item in json.loads(tasks.read_text())]
    hashes = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = Path(__file__).parent.parent.parent / name
        if path.is_file():
            contents = path.read_bytes()
            atomic_write(destination / name, contents)
            hashes[name] = sha(contents)
    for i, instance in enumerate(entries):
        source = (tasks.parent / instance.repository).resolve()
        if not source.is_dir():
            raise ValueError("Freeze requires a locally prepared Git checkout")
        instance.repository = str(source)
        instance.base_commit = git(
            source, "rev-parse", (instance.base_commit or "HEAD") + "^{commit}"
        ).strip()
        for field in ("test_patch", "context_bundle"):
            supplied = getattr(instance, field)
            paths = supplied if isinstance(supplied, list) else [supplied] if supplied else []
            copied = []
            for j, path in enumerate(paths):
                contents = (tasks.parent / path).read_bytes()
                name = f"{i}-{field}-{j}" + Path(path).suffix
                atomic_write(destination / name, contents)
                hashes[name] = sha(contents)
                copied.append(name)
            setattr(
                instance,
                field,
                copied if isinstance(supplied, list) else copied[0] if copied else None,
            )
    files = {
        "tasks.json": encode([i.model_dump(mode="json") for i in entries]).encode(),
        "config.json": config.model_dump_json(indent=2).encode(),
    }
    for name, contents in files.items():
        atomic_write(destination / name, contents)
        hashes[name] = sha(contents)
    manifest = {
        "version": 1,
        "files": hashes,
        "task_count": len(entries),
        "environment": metadata(),
        "harness_source_at_freeze": source_identity(),
        "comparison": {
            "shared": [
                "model",
                "model_parameters",
                "task",
                "repository_revision",
                "resource_limits",
                "verifier",
            ],
            "base": "same Python runtime, basic filesystem/shell only; not an external raw-agent implementation",
            "buffalo": "full configured programmatic capabilities",
        },
    }
    manifest["freeze_id"] = sha(encode(manifest).encode())
    atomic_write(destination / "manifest.json", encode(manifest).encode())
    return manifest


def validate(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    identity = manifest.pop("freeze_id")
    if sha(encode(manifest).encode()) != identity:
        raise ValueError("Frozen manifest identity changed")
    for name, digest in manifest["files"].items():
        if sha((bundle / name).read_bytes()) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return identity


async def compare(bundle, directory, *, repetitions=1, providers=None):
    bundle, directory = Path(bundle), Path(directory)
    identity = validate(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if source_identity() != manifest["harness_source_at_freeze"]:
        raise ValueError(
            "Harness source differs from freeze. Run with PYTHONPATH=<bundle>/runtime using the frozen module."
        )
    config = RunConfig.model_validate_json((bundle / "config.json").read_text())
    output = directory / "results.jsonl"
    for repetition in range(repetitions):
        order = ("base", "buffalo") if repetition % 2 == 0 else ("buffalo", "base")
        for profile in order:
            print(
                encode(
                    {
                        "freeze_id": identity,
                        "profile": profile,
                        "provider": config.provider.model_dump(),
                        "context": config.context.model_dump(),
                        "limits": config.limits.model_dump(),
                    }
                ),
                flush=True,
            )
            await evaluate(
                bundle / "tasks.json",
                config,
                directory,
                seed=repetition,
                output=output,
                profile=profile,
                providers=providers,
            )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    summaries = {}
    for profile in ("base", "buffalo"):
        path = directory / (profile + ".jsonl")
        selected = [r for r in rows if r["profile"] == profile]
        atomic_write(path, ("\n".join(encode(r) for r in selected) + "\n").encode())
        summaries[profile] = analyze(path)
    result = {
        "freeze_id": identity,
        "profiles": summaries,
        "note": "Small paired descriptive comparison, no superiority or causal claim",
    }
    atomic_write(directory / "comparison.json", encode(result).encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("tasks", type=Path)
    freeze_parser.add_argument("--config", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("bundle", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(
            args.tasks, RunConfig.model_validate_json(args.config.read_text()), args.output
        )
    else:
        result = asyncio.run(compare(args.bundle, args.output, repetitions=args.repetitions))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
