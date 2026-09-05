"""Incremental, disk-backed repository navigation. References are lexical evidence, not a call graph."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .storage import encode

EXCLUDED = {
    ".git",
    ".threadweave",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "generated",
}
LANGUAGES = {
    ".py": "python",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}
CONFIGS = {
    "pyproject.toml",
    "Cargo.toml",
    "package.json",
    "CMakeLists.txt",
    "Makefile",
    "go.mod",
    "pytest.ini",
    "tox.ini",
    "tsconfig.json",
    "setup.cfg",
    "requirements.txt",
}


def confined(root: Path, path: str, *, allowed=None, forbidden=()) -> Path:
    root = root.resolve()
    raw = root / path
    resolved = raw.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError("Path must resolve within the session workspace")
    relative = resolved.relative_to(root).as_posix()
    if ".git" in raw.parts or ".git" in resolved.parts[len(root.parts) :]:
        raise PermissionError("Direct .git access is prohibited; use Git tools")
    if any(fnmatch.fnmatch(relative, p) for p in forbidden):
        raise PermissionError(f"Forbidden path: {relative}")
    if (
        allowed is not None
        and relative != "."
        and not any(fnmatch.fnmatch(relative, p) for p in allowed)
    ):
        raise PermissionError(f"Path is outside allowed_paths: {relative}")
    return resolved


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def symbols(text: str, language: str) -> dict:
    found, imports = [], []
    if language == "python":
        try:
            tree = ast.parse(text)

            def visit(node, parent=""):
                for child in ast.iter_child_nodes(node):
                    prefix = parent
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = f"{parent}.{child.name}" if parent else child.name
                        found.append(
                            {
                                "name": name,
                                "kind": "class" if isinstance(child, ast.ClassDef) else "function",
                                "line": child.lineno,
                                "end_line": child.end_lineno,
                            }
                        )
                        prefix = name
                    elif isinstance(child, (ast.Import, ast.ImportFrom)):
                        imports.append(
                            {
                                "module": getattr(child, "module", None),
                                "names": [a.name for a in child.names],
                                "line": child.lineno,
                            }
                        )
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                found.append(
                                    {"name": target.id, "kind": "constant", "line": child.lineno}
                                )
                    visit(child, prefix)

            visit(tree)
            return {"symbols": found, "imports": imports, "parser": "python_ast"}
        except SyntaxError:
            pass  # Broken intermediate edits must remain navigable.
    declaration = re.compile(
        r"\b(class|struct|enum|trait|interface|mod|namespace|fn|func|function|def)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)"
    )
    c_function = re.compile(r"^\s*(?:[\w:*<>]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{")
    for line, content in enumerate(text.splitlines(), 1):
        match = declaration.search(content)
        if match:
            found.append({"name": match[2], "kind": match[1], "line": line})
        elif match := c_function.search(content):
            found.append({"name": match[1], "kind": "function", "line": line})
        if re.match(r"\s*(import\b|from\b|use\b|#include\b|require\()", content):
            imports.append({"text": content.strip(), "line": line})
    return {"symbols": found, "imports": imports, "parser": "lexical"}


class RepositoryIndex:
    def __init__(self, store, root, *, enhanced=True, allowed=None, forbidden=()):
        self.store, self.root = store, Path(root).resolve()
        self.enhanced, self.allowed, self.forbidden = enhanced, allowed, forbidden

    def paths(self):
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink()
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                rel = path.relative_to(self.root).as_posix()
                try:
                    confined(self.root, rel, allowed=self.allowed, forbidden=self.forbidden)
                except PermissionError:
                    continue
                if path.stat().st_size <= 2_000_000:
                    yield rel

    def refresh(self, paths=None):
        full = paths is None
        paths = list(self.paths()) if full else list(paths)
        changed = []
        for relative in paths:
            path = confined(self.root, relative, allowed=self.allowed, forbidden=self.forbidden)
            old = self.store.db.execute(
                "SELECT * FROM repository_files WHERE workspace=? AND path=?",
                (str(self.root), relative),
            ).fetchone()
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
                self.store.db.execute(
                    "DELETE FROM repository_files WHERE workspace=? AND path=?",
                    (str(self.root), relative),
                )
                continue
            stat = path.stat()
            if old and old["mtime_ns"] == stat.st_mtime_ns and old["size"] == stat.st_size:
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            text = raw.decode(errors="replace")
            language = LANGUAGES.get(path.suffix, "text")
            body = (
                symbols(text, language)
                if self.enhanced
                else {"symbols": [], "imports": [], "parser": "disabled"}
            )
            self.store.db.execute(
                "INSERT OR REPLACE INTO repository_files VALUES(?,?,?,?,?,?,?)",
                (
                    str(self.root),
                    relative,
                    stat.st_mtime_ns,
                    stat.st_size,
                    digest(raw),
                    language,
                    encode(body),
                ),
            )
            changed.append(relative)
        if full:
            keep = set(paths)
            for row in self.store.db.execute(
                "SELECT path FROM repository_files WHERE workspace=?", (str(self.root),)
            ).fetchall():
                if row[0] not in keep:
                    self.store.db.execute(
                        "DELETE FROM repository_files WHERE workspace=? AND path=?",
                        (str(self.root), row[0]),
                    )
        return changed

    def entries(self):
        self.refresh()
        return [
            dict(row)
            for row in self.store.db.execute(
                "SELECT * FROM repository_files WHERE workspace=? ORDER BY path", (str(self.root),)
            )
        ]

    def repo_map(self):
        entries = self.entries()
        return {
            "root": str(self.root),
            "files": [{k: r[k] for k in ("path", "language", "size")} for r in entries],
            "directories": sorted({str(Path(r["path"]).parent) for r in entries}),
            "languages": sorted({r["language"] for r in entries if r["language"] != "text"}),
            "configs": [r["path"] for r in entries if Path(r["path"]).name in CONFIGS],
            "test_entry_points": [r["path"] for r in entries if is_test(r["path"])],
            "excluded_directories": sorted(EXCLUDED),
        }

    def outline(self, path):
        path = (
            confined(self.root, path, allowed=self.allowed, forbidden=self.forbidden)
            .relative_to(self.root)
            .as_posix()
        )
        self.refresh([path])
        row = self.store.db.execute(
            "SELECT body FROM repository_files WHERE workspace=? AND path=?", (str(self.root), path)
        ).fetchone()
        if not row:
            raise ValueError("File is absent, binary, or exceeds the 2 MB indexing limit")
        return {"path": path, **json.loads(row[0])}

    def symbol_search(self, query, *, limit=100):
        matches = []
        for entry in self.entries():
            for symbol in json.loads(entry["body"])["symbols"]:
                if query.casefold() in symbol["name"].casefold():
                    matches.append({"path": entry["path"], **symbol})
        return {
            "matches": matches[:limit],
            "total": len(matches),
            "precision": "AST for Python; lexical otherwise",
        }

    def search(self, query, *, regex=False, path="*", language=None, context=2, limit=100):
        pattern = re.compile(query if regex else re.escape(query))
        entries = [
            r
            for r in self.entries()
            if fnmatch.fnmatch(r["path"], path) and (not language or r["language"] == language)
        ]
        candidates = [r["path"] for r in entries]
        engine = "python"
        # rg narrows candidate files; Python adds consistent bounded context/diagnostics.
        if shutil.which("rg") and candidates:
            args = ["rg", "--files-with-matches", "--null", "--hidden", "--no-ignore"]
            if not regex:
                args.append("--fixed-strings")
            try:
                matched = set()
                for offset in range(0, len(candidates), 100):
                    result = subprocess.run(
                        [*args, "--", query, *candidates[offset : offset + 100]],
                        cwd=self.root,
                        capture_output=True,
                        timeout=10,
                        check=False,
                    )
                    if result.returncode not in (0, 1):
                        raise ValueError(result.stderr.decode(errors="replace")[:500])
                    matched.update(result.stdout.decode().split("\0"))
                candidates = [p for p in candidates if p in matched]
                engine = "ripgrep"
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        matches = []
        for relative in candidates:
            lines = (self.root / relative).read_text(errors="replace").splitlines()
            for i, line in enumerate(lines):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": relative,
                            "line": i + 1,
                            "text": line[:2000],
                            "context": lines[max(0, i - context) : i + context + 1],
                        }
                    )
        return {
            "matches": matches[:limit],
            "total": len(matches),
            "engine": engine,
            "all_matches": matches if len(matches) > limit else None,
        }

    def dependencies(self, path):
        outline = self.outline(path)
        modules = {}
        for row in self.entries():
            name = str(Path(row["path"]).with_suffix("")).replace("/", ".")
            modules[name] = row["path"]
        related = []
        for item in outline["imports"]:
            for name in [item.get("module"), *item.get("names", [])]:
                if name:
                    related.extend(
                        p
                        for module, p in modules.items()
                        if module == name or module.endswith("." + name)
                    )
        return {
            "path": path,
            "imports": outline["imports"],
            "likely_local_modules": sorted(set(related)),
        }


def is_test(path):
    name = Path(path).name
    return bool(
        re.search(r"(^|/)(tests?|__tests__)/", path)
        or name.startswith("test_")
        or re.search(r"(_test\.|\.(test|spec)\.)", name)
    )


def detect(root):
    root = Path(root)
    result = {"languages": [], "build_systems": [], "suggested_commands": {}}
    for file, language, system in [
        ("pyproject.toml", "python", "pyproject"),
        ("Cargo.toml", "rust", "cargo"),
        ("package.json", "javascript/typescript", "npm"),
        ("go.mod", "go", "go"),
        ("CMakeLists.txt", "c/c++/cuda", "cmake"),
        ("Makefile", "c/c++/cuda", "make"),
    ]:
        if (root / file).is_file():
            result["languages"].append(language)
            result["build_systems"].append(system)
    if (root / "Cargo.toml").exists():
        result["suggested_commands"].update(test=[["cargo", "test"]], build=[["cargo", "build"]])
    elif (root / "go.mod").exists():
        result["suggested_commands"].update(
            test=[["go", "test", "./..."]], build=[["go", "build", "./..."]]
        )
    elif (root / "package.json").exists():
        scripts = json.loads((root / "package.json").read_text()).get("scripts", {})
        result["suggested_commands"] = {
            k: [["npm", "run", k]] for k in ("test", "build", "lint", "typecheck") if k in scripts
        }
    elif (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        result["suggested_commands"]["test"] = [["python", "-m", "pytest", "-q"]]
    return result
