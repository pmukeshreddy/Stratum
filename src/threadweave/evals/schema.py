from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from ..models import Record, RunConfig, Usage


class BenchmarkSetup(Record):
    source: Path | None = None
    commit: str | None = None
    python: str = sys.executable
    dataset: Path | None = None
    dataset_revision: str | None = None
    dataset_sha256: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class EvaluationConfig(Record):
    run: RunConfig
    benchmarks: dict[str, BenchmarkSetup] = Field(default_factory=dict)


class NotRun(Exception):
    """A concrete setup/provider/evaluator blocker, never an alternate workload."""


def timestamp():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def accounting(usages, wall_seconds):
    rows = [u.model_dump() if isinstance(u, Usage) else u for u in usages]
    result = {
        key: sum(u.get(key, 0) for u in rows)
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_output_tokens",
            "estimated_calls",
            "model_calls",
            "tool_calls",
            "python_executions",
            "subagent_count",
        )
    }
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    costs = [u.get("api_cost", u.get("cost")) for u in rows]
    result["api_cost"] = None if any(c is None for c in costs) else sum(costs)
    result["wall_seconds"] = wall_seconds
    result["scope"] = "root and all descendants; evaluator usage included separately when present"
    return result
