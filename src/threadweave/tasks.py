from __future__ import annotations

from typing import Protocol

from .models import HarnessError, TaskConfig, Verification
from .tools import ProcessArgs, ToolContext, run_process


class TaskAdapter(Protocol):
    async def prepare(self, context: ToolContext, config: TaskConfig) -> dict:
        """Describe/validate the environment without choosing agent strategy."""
        ...

    async def verify(self, context: ToolContext, config: TaskConfig) -> Verification | None: ...


class WorkspaceTask:
    async def prepare(self, context, config):
        path = context.path(".")
        if not path.is_dir():
            raise HarnessError("environment", "workspace_missing", f"Workspace missing: {path}")
        if config.verifier not in {"none", "file", "command"}:
            raise HarnessError(
                "environment",
                "unknown_verifier",
                f"Unsupported workspace verifier: {config.verifier}",
            )
        if config.verifier == "file":
            if not isinstance(config.verifier_options.get("path"), str):
                raise ValueError("File verifier requires a path")
            context.path(config.verifier_options["path"])
        if config.verifier == "command":
            ProcessArgs.model_validate(config.verifier_options)
        return {
            "workspace": str(path),
            "specification": config.specification,
            "success_metrics": config.success_metrics,
        }

    async def verify(self, context, config):
        options = config.verifier_options
        if config.verifier == "none":
            return None
        if config.verifier == "file":
            path = context.path(options["path"])
            if not path.is_file():
                return Verification(passed=False, details={"missing": str(path)})
            if path.stat().st_size > options.get("max_bytes", 1000000):
                return Verification(
                    passed=False, details="Verifier file exceeds configured size limit"
                )
            contents = path.read_text()
            expected = options.get("equals")
            passed = expected is None or contents.strip() == str(expected).strip()
            return Verification(
                passed=passed,
                details={"path": str(path), "contents": contents},
                metrics={"bytes": path.stat().st_size},
            )
        if config.verifier == "command":
            if "process" not in context.runtime.store.config(context.session_id).permissions:
                raise PermissionError("Command verifier requires process permission")
            result = await run_process(context, ProcessArgs.model_validate(options))
            return Verification(
                passed=result["returncode"] == 0,
                details=result,
                metrics={"exit_code": result["returncode"]},
            )
        raise ValueError(f"Unknown verifier: {config.verifier}")
