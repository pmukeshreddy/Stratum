"""Repository-scoped instruction snapshots, ordered from root to working directory."""

import hashlib
from pathlib import Path

NAMES = ("AGENTS.md", "CLAUDE.md", "BUFFALO.md", "THREADWEAVE.md")


def discover_instructions(workspace: Path):
    workspace = workspace.resolve()
    chain = [workspace, *workspace.parents]
    root = next((p for p in chain if (p / ".git").exists()), workspace)
    directories = list(reversed(chain[: chain.index(root) + 1]))
    files = []
    for directory in directories:
        for name in NAMES:
            path = directory / name
            if path.is_file() and path.resolve().is_relative_to(root):
                raw = path.read_bytes()
                files.append(
                    {
                        "path": str(path),
                        "scope": str(directory),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "content": raw.decode("utf-8"),
                    }
                )
    return files
