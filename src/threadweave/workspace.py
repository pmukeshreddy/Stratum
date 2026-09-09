"""Domain-independent workspace path policy."""

import fnmatch
from pathlib import Path


def confined(root: Path, path: str, *, allowed=None, forbidden=()) -> Path:
    root = root.resolve()
    raw = root / path
    resolved = raw.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError("Path must resolve within the session workspace")
    relative = resolved.relative_to(root).as_posix()
    if any(fnmatch.fnmatch(relative, p) for p in forbidden):
        raise PermissionError(f"Forbidden path: {relative}")
    if (
        allowed is not None
        and relative != "."
        and not any(fnmatch.fnmatch(relative, p) for p in allowed)
    ):
        raise PermissionError(f"Path is outside allowed_paths: {relative}")
    return resolved
