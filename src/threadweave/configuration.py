"""Explicit configuration loading and host capability diagnostics."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from .models import RunConfig


def load_config(path, model_override=None):
    raw = json.loads(Path(path).read_text())
    if model_override:
        alias = raw.get("routing", {}).get("default")
        if alias:
            raw["models"][alias]["model"] = model_override
        else:
            raw.setdefault("provider", {})["model"] = model_override
    for provider in [raw.get("provider", {}), *raw.get("models", {}).values()]:
        value = provider.get("model", "")
        if value.startswith("${") and value.endswith("}"):
            name = value[2:-1]
            if not os.environ.get(name):
                raise ValueError(f"Set {name} to an explicit model ID")
            provider["model"] = os.environ[name]
    config = RunConfig.model_validate(raw)
    selected = config.models[config.routing.default] if config.routing.default else config.provider
    if (not selected.model and selected.name != "codex_subscription") or selected.model.startswith(
        "REPLACE_"
    ):
        raise ValueError("Configure provider.model or an explicit routing.default model")
    return config


def doctor(directory, config=None, *, subscription_status=None):
    directory = Path(directory).resolve()
    capabilities = {
        name: shutil.which(name)
        for name in (
            "git",
            "codex",
            "rg",
            "docker",
            "podman",
            "python",
            "pytest",
            "cargo",
            "rustc",
            "go",
            "node",
            "npm",
            "gcc",
            "clang",
            "cmake",
            "make",
            "nvcc",
            "ncu",
            "nsys",
        )
    }
    issues, providers = [], []
    if config:
        for provider in [config.provider, *config.models.values()]:
            if not provider.model and config.routing.default:
                continue
            if provider.name == "codex_subscription":
                status = subscription_status or {
                    "usable": False,
                    "issue": "Run the CLI doctor for live Codex account checks",
                }
                providers.append({"provider": provider.name, **status})
                if not status.get("usable"):
                    issues.append(
                        status.get(
                            "issue",
                            "Codex subscription provider is unavailable; run threadweave auth status",
                        )
                    )
                continue
            credential = not provider.api_key_env or bool(os.environ.get(provider.api_key_env))
            providers.append(
                {
                    "provider": provider.name,
                    "model": provider.model,
                    "base_url": provider.base_url,
                    "credential_present": credential,
                    "credential_variable": provider.api_key_env,
                }
            )
            if not provider.model or not credential:
                issues.append("Missing explicit model ID or provider credential")
        if config.execution.backend == "container":
            if not capabilities[config.execution.engine]:
                issues.append(f"Container engine missing: {config.execution.engine}")
            else:
                result = subprocess.run(
                    [config.execution.engine, "info"], capture_output=True, timeout=15, check=False
                )
                if result.returncode:
                    issues.append("Container engine is installed but unavailable")
    else:
        issues.append("No provider configuration supplied; use doctor --config PATH")
    health = "not_created"
    manifest = directory / "store.json"
    if manifest.exists():
        try:
            from .file_store import FileStore

            store = FileStore(directory)
            try:
                for collection in (directory / "records").iterdir():
                    if collection.is_dir():
                        store.select(collection.name)
                store.select("events")
                health = "ok"
            finally:
                store.close()
        except (ValueError, OSError) as exc:
            health = "invalid"
            issues.append(f"File-store validation failed: {exc}")
    if not capabilities["git"] and config and config.task.adapter == "coding":
        issues.append("Git is required for coding tasks")
    if directory.exists() and not os.access(directory, os.W_OK):
        issues.append("Data directory is not writable")
    return {
        "ok": not issues,
        "providers": providers,
        "capabilities": capabilities,
        "data": str(directory),
        "storage_health": health,
        "issues": issues,
        "local_execution": "trusted-host execution, not a sandbox",
    }
