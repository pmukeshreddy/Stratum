"""Coding adapter options and legacy coding configuration loading."""

from typing import Literal

from pydantic import Field

from .models import Record


class BenchmarkConfig(Record):
    command: list[str] = Field(min_length=1)
    correctness_commands: list[list[str]] = Field(default_factory=list)
    repetitions: int = Field(default=10, ge=1, le=1000)
    warmups: int = Field(default=3, ge=0, le=100)
    metric_regex: str
    direction: Literal["lower_is_better", "higher_is_better"] = "lower_is_better"
    required_improvement: float = Field(default=0, ge=0)
    noise_tolerance: float = Field(default=0.01, ge=0)
    timeout_seconds: float = Field(default=60, gt=0)
    outlier_policy: Literal["retain", "iqr"] = "retain"


class CodingOptions(Record):
    repository: str | None = None
    base_commit: str | None = None
    test_commands: list[list[str]] = Field(default_factory=list)
    build_commands: list[list[str]] = Field(default_factory=list)
    lint_commands: list[list[str]] = Field(default_factory=list)
    typecheck_commands: list[list[str]] = Field(default_factory=list)
    benchmark_commands: list[list[str]] = Field(default_factory=list)
    benchmark: BenchmarkConfig | None = None
    profiler_command: list[str] | None = None
    require_clean_baseline: bool = True
    capture_baseline: bool = True
    require_tests: bool = True
    prohibit_test_deletion: bool = True
    protect_tests: bool = False
    require_change: bool = True
    required_files: list[str] = Field(default_factory=list)
    allow_baseline_failures: bool = False


def coding_options(task):
    return CodingOptions.model_validate(task.options.get("coding", {}))


def update_coding_options(task, **values):
    options = {**coding_options(task).model_dump(), **values}
    task.options = {**task.options, "coding": CodingOptions.model_validate(options).model_dump()}


def migrate_legacy(value, fields):
    legacy = {
        key: value[key] for key in CodingOptions.model_fields if key in value and key not in fields
    }
    if not legacy:
        return value
    result = {key: item for key, item in value.items() if key not in legacy}
    options = dict(result.get("options", {}))
    existing = options.get("coding", {})
    if any(key in existing and existing[key] != item for key, item in legacy.items()):
        raise ValueError("Conflicting legacy and adapter-owned coding options")
    options["coding"] = {**legacy, **existing}
    result["options"] = options
    return result


def migrate_session(value):
    result = dict(value)
    workspace = result.get("workspace")
    if isinstance(workspace, dict) and "candidate_pending" in workspace.get("metadata", {}):
        metadata = dict(workspace["metadata"])
        metadata["admission_pending"] = metadata.pop("candidate_pending")
        result["workspace"] = {**workspace, "metadata": metadata}
    instructions = result.pop("repository_instructions", [])
    context = dict(result.get("adapter_context", {}))
    if instructions:
        context["coding"] = {"repository_instructions": instructions, **context.get("coding", {})}
    result["adapter_context"] = context
    return result
