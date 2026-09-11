from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from ..models import RunConfig
from . import BENCHMARKS
from .schema import EvaluationConfig, NotRun, digest, file_digest


def arguments(parser):
    parser.add_argument("benchmark", choices=["arc-agi-3"], nargs="?", default="arc-agi-3")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=positive, help="Explicit ARC validation subset; omit for all 25 games"
    )
    parser.add_argument("--games-concurrency", type=positive, default=16)
    parser.add_argument("--inference-concurrency", type=positive, default=16)
    parser.add_argument(
        "--protocol-policy",
        "--protocol-validation",
        dest="protocol_validation",
        type=Path,
        help="Fixed-game policy for a matched subset or all 25 games; no inferred scaling curve",
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only setup validation; no inference"
    )


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def load(path):
    if path is None:
        return EvaluationConfig(
            run=RunConfig()
        ), "Missing evaluation manifest: pass --config (see configs/evaluation.example.json)"
    raw = json.loads(path.read_text())
    provider = raw.get("run", {}).get("provider", {})
    model = provider.get("model", "")
    if model.startswith("${") and model.endswith("}"):
        name = model[2:-1]
        if not os.environ.get(name):
            raise NotRun(f"Set {name} to the model ID for this evaluation")
        provider["model"] = os.environ[name]
    config = EvaluationConfig.model_validate(raw)
    unknown = config.benchmarks.keys() - BENCHMARKS.keys()
    if unknown:
        raise NotRun(f"Unsupported benchmark configuration: {sorted(unknown)}")
    for setup in config.benchmarks.values():
        if setup.source and not setup.source.is_absolute():
            setup.source = (path.resolve().parent / setup.source).resolve()
        if "/" in setup.python and not Path(setup.python).is_absolute():
            setup.python = str((path.resolve().parent / setup.python).absolute())
        environments = setup.options.get("environments_dir")
        if environments and not Path(environments).is_absolute():
            setup.options["environments_dir"] = str(
                (path.resolve().parent / environments).resolve()
            )
    return config, None


async def resolve(config):
    from ..cli import resolved_config
    from ..providers import default_providers

    if config.run.models or config.run.routing.default or config.run.routing.roles:
        raise NotRun(
            "Matched evaluations require a single provider/model without routing overrides"
        )
    if config.run.extensions or config.run.mcp_servers:
        raise NotRun(
            "Matched evaluations require explicit built-in tools; external extensions/MCP are not part of the comparison harness"
        )
    if not config.run.features.persistent_repl:
        raise NotRun("Buffalo evaluation requires features.persistent_repl=true")
    config.run = await resolved_config(config.run)
    provider = config.run.provider
    if provider.name not in default_providers():
        raise NotRun(f"Unsupported model provider: {provider.name}")
    if not provider.model:
        raise NotRun("Missing provider.model")
    if provider.api_key_env and not os.environ.get(provider.api_key_env):
        raise NotRun(f"Missing API credentials: {provider.api_key_env} is unset")
    return config


def source_identity():
    source = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(source)): file_digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".rs"}
    }
    root = source.parents[1]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "commit": result.stdout.strip() or None,
        "source_sha256": digest(files),
        "source_files": files,
    }


def contract(config, provenance, task_ids, seed):
    return {
        "provider": config.run.provider.model_dump(mode="json"),
        "reasoning_level": config.run.provider.parameters.get(
            "reasoning_effort", config.run.provider.parameters.get("reasoning", "provider default")
        ),
        "budget": config.run.limits.model_dump(mode="json"),
        "output_budget_policy": "Reserve provider.max_output_tokens for each root/descendant/auxiliary request; stop admission when remaining output cannot cover it. Include reasoning tokens in provider-reported output. Invalidate measured overshoot; do not truncate or hide usage.",
        "context_max_tokens": config.run.context.max_tokens,
        "benchmark_version": provenance["benchmark_version"],
        "dataset_environment_version": provenance["dataset_environment_version"],
        "starting_state": provenance["starting_state"],
        "task_ids": task_ids,
        "seed": seed,
    }


async def execute(args):
    from .arc_scheduler import execute_arc

    return await execute_arc(args)
