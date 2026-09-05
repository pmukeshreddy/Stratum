from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Lifecycle(StrEnum):
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    INACTIVE = "INACTIVE"


class Outcome(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    LIMITED = "limited"
    FAILED = "failed"


class Failure(Record):
    category: Literal["model", "provider", "tool", "verifier", "environment", "runtime"]
    code: str
    message: str
    retryable: bool = False
    uncertain: bool = False


class HarnessError(Exception):
    def __init__(self, category: str, code: str, message: str, *, retryable=False, uncertain=False):
        super().__init__(message)
        self.failure = Failure(
            category=category, code=code, message=message, retryable=retryable, uncertain=uncertain
        )


class ProviderConfig(Record):
    name: str = "demo"
    model: str = "offline-demo"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    parameters: dict[str, Any] = Field(default_factory=dict)
    streaming: bool = True
    timeout_seconds: float = Field(default=120, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)


class ContextPolicy(Record):
    max_tokens: int = Field(default=24000, ge=2048)
    compact_at: float = Field(default=0.8, gt=0.1, lt=1)
    recent_blocks: int = Field(default=6, ge=1)
    summary_chars: int = Field(default=3000, ge=256)
    result_chars: int = Field(default=1800, ge=128)
    supplemental_chars: int = Field(default=3000, ge=0)


class RetryPolicy(Record):
    attempts: int = Field(default=3, ge=1, le=10)
    initial_delay: float = Field(default=0.5, ge=0)
    max_delay: float = Field(default=10, ge=0)


class RefinementPolicy(Record):
    enabled: bool = True
    allow_global_writes: bool = False
    selected_entries: list[str] = Field(default_factory=list)


class ResourceLimits(Record):
    max_turns: int = Field(default=100, ge=1)
    token_budget: int = Field(default=500000, ge=1)
    wall_seconds: float = Field(default=3600, gt=0)
    cost_budget: float | None = Field(default=None, gt=0)
    max_tool_calls: int = Field(default=1000, ge=1)
    max_python_executions: int = Field(default=500, ge=1)
    max_model_calls: int = Field(default=300, ge=1)
    max_subagents: int = Field(default=20, ge=0)
    max_depth: int = Field(default=5, ge=0)
    concurrency: int = Field(default=4, ge=1, le=128)
    tool_timeout_seconds: float = Field(default=60, gt=0)
    python_timeout_seconds: float = Field(default=60, gt=0)


class TaskConfig(Record):
    adapter: str = "workspace"
    specification: dict[str, Any] = Field(default_factory=dict)
    verifier: str = Field(default="none", min_length=1)
    verifier_options: dict[str, Any] = Field(default_factory=dict)
    verify_each_turn: bool = True
    require_verifier: bool = False
    wait_for_children: bool = True
    success_metrics: dict[str, Any] = Field(default_factory=dict)


class RunConfig(Record):
    schema_version: Literal[1] = 1
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    context: ContextPolicy = Field(default_factory=ContextPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    refinement: RefinementPolicy = Field(default_factory=RefinementPolicy)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    task: TaskConfig = Field(default_factory=TaskConfig)
    permissions: list[str] = Field(
        default_factory=lambda: ["workspace.read", "workspace.write", "python", "agents", "state"]
    )
    tool_allowlist: list[str] | None = None
    allow_sibling_messages: bool = True
    extensions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if self.provider.max_output_tokens >= self.context.max_tokens:
            raise ValueError("max_output_tokens must be smaller than context.max_tokens")
        if self.task.require_verifier and self.task.verifier == "none":
            raise ValueError("require_verifier needs a configured verifier")
        if self.limits.cost_budget is not None and (
            self.provider.input_cost_per_million is None
            or self.provider.output_cost_per_million is None
        ):
            raise ValueError("cost budgets require configured input/output prices for reservations")
        if self.task.verifier == "command" and "process" not in self.permissions:
            raise ValueError("command verifiers require process permission")
        return self


class Workspace(Record):
    path: str
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(Record):
    id: str
    parent_id: str | None = None
    root_id: str
    branch_from: str | None = None
    branch_event: str | None = None
    name: str
    role: str = "agent"
    instruction: str
    config_id: str
    workspace: Workspace
    kernel_id: str
    lifecycle: Lifecycle = Lifecycle.ADMITTED
    outcome: Outcome = Outcome.ACTIVE
    mode: Literal["autonomous", "goal", "heartbeat"] = "autonomous"
    runnable: bool = True
    paused: bool = False
    turns: int = 0
    context: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    selected_state: list[str] = Field(default_factory=list)
    pending_turn: dict[str, Any] | None = None
    wake_at: float | None = None
    created_at: float
    updated_at: float
    started_at: float | None = None
    running_since: float | None = None
    last_error: Failure | None = None
    result: str | None = None


class Usage(Record):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    python_executions: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    verifier_calls: int = Field(default=0, ge=0)
    subagent_count: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)
    cost: float = Field(default=0, ge=0)
    estimated_calls: int = Field(default=0, ge=0)


class Action(Record):
    id: str = Field(default_factory=new_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(Record):
    session_id: str
    root_id: str
    parent_id: str | None
    name: str
    turn: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    config: ProviderConfig
    input_token_bound: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(Record):
    text: str = ""
    actions: list[Action] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    usage_reported: bool = True
    provider_id: str | None = None


class Verification(Record):
    passed: bool
    details: Any = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class StateEdit(Record):
    entry_id: str | None = None
    kind: Literal["prompt_note", "memory", "skill", "subagent_spec"] = "memory"
    scope: Literal["session", "global"] = "session"
    title: str = Field(default="", max_length=200)
    content: dict[str, Any] = Field(default_factory=dict)
    operation: Literal["upsert", "delete", "rollback"] = "upsert"
    rollback_version: int | None = Field(default=None, ge=1)
    expected_version: int | None = Field(default=None, ge=1)
    source_events: list[str] = Field(min_length=1)
    intended_effect: str = Field(min_length=1, max_length=2000)
    select: bool = False
