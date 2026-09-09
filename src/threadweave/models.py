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
    name: str = "codex_subscription"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    streaming: bool = True
    timeout_seconds: float = Field(default=120, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def optional_api_defaults(cls, value):
        if isinstance(value, dict) and value.get("name") == "chat":
            value = dict(value)
            value.setdefault("api_key_env", "OPENAI_API_KEY")
            value.setdefault("base_url", "https://api.openai.com/v1")
        return value

    @model_validator(mode="after")
    def subscription_auth_only(self):
        if self.name == "codex_subscription" and (
            self.api_key_env
            or self.base_url
            or self.input_cost_per_million is not None
            or self.output_cost_per_million is not None
        ):
            raise ValueError(
                "codex_subscription uses managed ChatGPT auth, a fixed subscription endpoint, and no API prices"
            )
        return self


class ContextPolicy(Record):
    semantic_first: bool = True
    max_tokens: int = Field(default=96000, ge=2048)
    compact_at: float = Field(default=0.8, gt=0.1, lt=1)
    recent_blocks: int = Field(default=6, ge=1)
    summary_chars: int = Field(default=3000, ge=256)
    summary_tokens: int | None = Field(default=None, ge=256)
    result_chars: int = Field(default=131072, ge=128)
    supplemental_chars: int = Field(default=3000, ge=0)
    supplemental_tokens: int | None = Field(default=None, ge=0)
    embedding_model: str | None = None
    embedding_cache: str | None = None


class RetryPolicy(Record):
    attempts: int = Field(default=3, ge=1, le=10)
    initial_delay: float = Field(default=0.5, ge=0)
    max_delay: float = Field(default=10, ge=0)


class RefinementPolicy(Record):
    enabled: bool = True
    allow_global_writes: bool = False
    selected_entries: list[str] = Field(default_factory=list)
    automatic: bool = True
    evaluation_isolation: bool = False
    every_turns: int = Field(default=10, ge=1)
    progress_every_turns: int = Field(default=3, ge=1)
    on_completion: bool = True
    verifier_failures: int = Field(default=3, ge=1)
    max_proposals: int = Field(default=3, ge=1, le=10)
    skill_failure_limit: int = Field(default=3, ge=1)
    automatic_budget_seconds: float = Field(default=120, gt=0)
    continuation_reserve_seconds: float = Field(default=60, ge=0)
    reasoning: Literal["off", "inherit"] = "off"
    completion_followup: bool = False
    root_only: bool = True


class KernelStatePolicy(Record):
    memory_bytes: int = Field(default=128 * 1024 * 1024, ge=1024)
    variable_bytes: int = Field(default=16 * 1024 * 1024, ge=256)
    mutable_cache_bytes: int = Field(default=8 * 1024 * 1024, ge=0)
    inline_bytes: int = Field(default=64 * 1024, ge=128)
    snapshot_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    artifact_bytes: int = Field(default=512 * 1024 * 1024, ge=1024)
    stale_cells: int = Field(default=20, ge=1)
    checkpoint_cells: int = Field(default=5, ge=1)
    snapshot_seconds: float = Field(default=5, gt=0)
    variable_seconds: float = Field(default=1, gt=0)
    size_scan_nodes: int = Field(default=10000, ge=100)
    stream_chunk_bytes: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)


class VerificationPolicy(Record):
    continuous: bool = True
    completion_wait_seconds: float = Field(default=30, gt=0)
    generated_path_parts: list[str] = Field(
        default_factory=lambda: ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]
    )
    targeted_commands: list[list[str]] = Field(default_factory=list)
    targeted_every_turns: int = Field(default=3, ge=1)
    max_target_files: int = Field(default=8, ge=1)
    failure_items: int = Field(default=12, ge=1)
    diagnostic_chars: int = Field(default=1200, ge=128)


class Features(Record):
    persistent_repl: bool = True
    subagents: bool = True
    history_retrieval: bool = True
    automatic_refinement: bool = True
    experiments: bool = True
    enhanced_code_index: bool = True
    model_compaction: bool = True


class ExecutionConfig(Record):
    backend: Literal["local", "container"] = "local"
    engine: Literal["docker", "podman"] = "docker"
    image: str | None = None
    network: bool = False
    read_only: bool = False
    environment_allowlist: list[str] = Field(
        default_factory=lambda: [
            "PATH",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "VIRTUAL_ENV",
            "CUDA_VISIBLE_DEVICES",
        ]
    )
    environment: dict[str, str] = Field(default_factory=dict)
    command_allowlist: list[str] | None = None
    output_chars: int = Field(default=65536, ge=256, le=65536)
    memory: str = "2g"
    cpus: float = Field(default=2, gt=0)


class RoutingConfig(Record):
    policy: Literal["fixed", "role_based"] = "fixed"
    default: str | None = None
    roles: dict[str, str] = Field(default_factory=dict)


class LoopPolicy(Record):
    warn_repetitions: int = Field(default=3, ge=2)
    stop_repetitions: int = Field(default=12, ge=3)


class ResourceLimits(Record):
    max_turns: int = Field(default=100, ge=1)
    token_budget: int = Field(default=500000, ge=1)
    output_token_budget: int | None = Field(default=None, ge=1)
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
    # Verbatim role-bearing task instructions survive trajectory compaction.
    instruction_messages: list[dict[str, Any]] = Field(default_factory=list)
    # Optional immutable source contract when an adapter wraps the current assignment.
    original_messages: list[dict[str, Any]] = Field(default_factory=list)
    specification: dict[str, Any] = Field(default_factory=dict)
    verifier: str = Field(default="none", min_length=1)
    verifier_options: dict[str, Any] = Field(default_factory=dict)
    verify_each_turn: bool = False
    require_verifier: bool = False
    wait_for_children: bool = True
    success_metrics: dict[str, Any] = Field(default_factory=dict)
    allowed_paths: list[str] = Field(default_factory=lambda: ["**"])
    forbidden_paths: list[str] = Field(default_factory=list)
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_adapter_options(cls, value):
        from .task_options import migrate_options

        return migrate_options(value, cls.model_fields)


class McpServerConfig(Record):
    type: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str = "."
    url: str | None = None
    enabled: bool = True
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] = Field(default_factory=list)
    # Values are ENVIRONMENT VARIABLE NAMES, never credentials in persisted configs.
    env_from: dict[str, str] = Field(default_factory=dict)
    headers_from: dict[str, str] = Field(default_factory=dict)
    startup_timeout_seconds: float = Field(default=30, gt=0)
    call_timeout_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def transport_config(self):
        if self.type == "stdio" and not self.command:
            raise ValueError("stdio MCP requires command")
        if self.type == "http" and not self.url:
            raise ValueError("http MCP requires url")
        if self.url:
            from urllib.parse import urlsplit

            parsed = urlsplit(self.url)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.username
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("MCP URL must not contain credentials, query secrets or fragments")
        return self


class RunConfig(Record):
    schema_version: Literal[1] = 1
    control_plane: Literal["python", "direct"] = "python"
    skill_paths: list[str] = Field(default_factory=list)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    context: ContextPolicy = Field(default_factory=ContextPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    refinement: RefinementPolicy = Field(default_factory=RefinementPolicy)
    kernel_state: KernelStatePolicy = Field(default_factory=KernelStatePolicy)
    verification: VerificationPolicy = Field(default_factory=VerificationPolicy)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    task: TaskConfig = Field(default_factory=TaskConfig)
    permissions: list[str] = Field(
        default_factory=lambda: ["workspace.read", "workspace.write", "python", "agents", "state"]
    )
    tool_allowlist: list[str] | None = None
    active_tool_names: list[str] | None = None
    capabilities: list[str] = Field(default_factory=list)
    disabled_capabilities: list[str] = Field(default_factory=list)
    effective_capabilities: list[str] = Field(default_factory=list)
    allow_sibling_messages: bool = True
    extensions: list[str] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    features: Features = Field(default_factory=Features)
    models: dict[str, ProviderConfig] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    loop: LoopPolicy = Field(default_factory=LoopPolicy)

    @model_validator(mode="after")
    def coherent(self):
        if (
            max(p.max_output_tokens for p in [self.provider, *self.models.values()])
            >= self.context.max_tokens
        ):
            raise ValueError("max_output_tokens must be smaller than context.max_tokens")
        if (
            self.task.require_verifier
            and self.task.verifier == "none"
            and self.task.adapter == "workspace"
        ):
            raise ValueError("require_verifier needs a configured verifier")
        priced = list(self.models.values()) + ([self.provider] if not self.routing.default else [])
        if self.limits.cost_budget is not None and any(
            p.input_cost_per_million is None or p.output_cost_per_million is None for p in priced
        ):
            raise ValueError("cost budgets require configured input/output prices for reservations")
        if self.task.verifier == "command" and "process" not in self.permissions:
            raise ValueError("command verifiers require process permission")
        if self.execution.backend == "container" and not self.execution.image:
            raise ValueError("container execution requires execution.image")
        if self.execution.backend == "container" and "python" in self.permissions:
            raise ValueError(
                "Python workers are host processes: remove python permission for container-only runs"
            )
        for alias in [self.routing.default, *self.routing.roles.values()]:
            if alias and alias not in self.models:
                raise ValueError(f"Unknown routing model alias: {alias}")
        return self


class Workspace(Record):
    path: str
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(Record):
    id: str
    parent_id: str | None = None
    root_id: str
    spawned_by_request_id: str | None = None
    depth: int = 0
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
    mode: Literal["autonomous", "goal", "heartbeat", "interactive"] = "autonomous"
    runnable: bool = True
    paused: bool = False
    turns: int = 0
    context: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    selected_state: list[str] = Field(default_factory=list)
    adapter_context: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_adapter_context(cls, value):
        from .task_options import migrate_session

        return migrate_session(value)

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
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_output_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    python_executions: int = Field(default=0, ge=0)
    wall_seconds: float = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    verifier_calls: int = Field(default=0, ge=0)
    subagent_count: int = Field(default=0, ge=0)
    turns: int = Field(default=0, ge=0)
    cost: float | None = Field(default=0, ge=0)
    estimated_calls: int = Field(default=0, ge=0)


class Action(Record):
    id: str = Field(default_factory=new_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelRequest(Record):
    request_id: str = Field(default_factory=new_id)
    request_kind: Literal["trajectory", "auxiliary"] = "trajectory"
    reasoning_mode: Literal["inherit", "off"] = "inherit"
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

    def public_dump(self):
        return self.model_dump(mode="json", exclude={"messages": {"__all__": {"provider_items"}}})


class ModelResponse(Record):
    text: str = ""
    actions: list[Action] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    usage_reported: bool = True
    provider_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Provider-native continuation is private persistence, never a user-facing event.
    provider_items: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)


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
