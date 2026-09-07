"""Discover SKILL.md resources and executable src-layout Python skill packages."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
import tomllib
import types
from pathlib import Path

import yaml


def discover(workspace: Path, configured=()):
    roots = [
        workspace / ".agents/skills",
        workspace / ".threadweave/skills",
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "threadweave/skills",
    ]
    roots += [Path(p).expanduser() if Path(p).is_absolute() else workspace / p for p in configured]
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for file in sorted(root.rglob("SKILL.md")):
            if not file.resolve().is_relative_to(root.resolve()) or file.stat().st_size > 1_000_000:
                continue
            text = file.read_text()
            try:
                front = yaml.safe_load(text.split("---", 2)[1]) if text.startswith("---\n") else {}
                front = front if isinstance(front, dict) else {}
                name = str(front.get("name", file.parent.name))
                permissions = front.get("permissions", ["python"])
                if not isinstance(permissions, list) or not all(
                    isinstance(p, str) for p in permissions
                ):
                    raise ValueError("Skill permissions must be a list of permission names")
                entry = {
                    "name": name,
                    "description": str(front.get("description", "")),
                    "path": str(file.resolve()),
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "kind": "markdown",
                    "required_permissions": permissions,
                }
                module = name.replace("-", "_")
                project, package = (
                    file.parent / "pyproject.toml",
                    file.parent / "src" / module / "__init__.py",
                )
                if project.is_file() and package.is_file() and module.isidentifier():
                    if not package.resolve().is_relative_to(
                        file.parent.resolve()
                    ) or not project.resolve().is_relative_to(file.parent.resolve()):
                        raise ValueError("Skill package must remain inside its directory")
                    tomllib.loads(project.read_text())
                    fingerprint = hashlib.sha256(text.encode())
                    for source in [project, *sorted(package.parent.rglob("*.py"))]:
                        if not source.resolve().is_relative_to(file.parent.resolve()):
                            raise ValueError("Skill source symlink escapes its directory")
                        fingerprint.update(str(source.relative_to(file.parent)).encode())
                        fingerprint.update(source.read_bytes())
                    entry.update(kind="python", import_name=module, package=str(package.resolve()))
                    entry["sha256"] = fingerprint.hexdigest()
                found.setdefault(name, entry)
            except (ValueError, yaml.YAMLError, IndexError) as exc:
                found.setdefault(
                    file.parent.name,
                    {
                        "name": file.parent.name,
                        "path": str(file),
                        "kind": "unavailable",
                        "error": str(exc),
                    },
                )
    return list(found.values())


class CallableSkill(types.ModuleType):
    async def __call__(self, *args, **kwargs):
        result = self.run(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


class UnavailableSkill:
    def __init__(self, name, error):
        self.name, self.error = name, error

    async def run(self, *args, **kwargs):
        raise RuntimeError(
            f"Skill {self.name} unavailable: {self.error}. Install its declared dependencies in the kernel environment."
        )

    __call__ = run


def load_module(entry):
    name, path = entry["import_name"], Path(entry["package"])
    existing = sys.modules.get(name)
    if existing and Path(getattr(existing, "__file__", "") or ".").resolve() != path:
        raise ValueError(f"Skill import collides with an existing module: {name}")
    if existing and getattr(existing, "__skill_version__", None) == entry["sha256"]:
        return existing
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.__skill_version__ = entry["sha256"]
        if callable(getattr(module, "run", None)):
            module.__class__ = CallableSkill
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise
