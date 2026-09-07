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

from .change_tracking import ChangeTracker, expand_paths
from .storage import encode
from .syntax_index import parse as parse_syntax

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
    syntax = parse_syntax(text, language)
    if language != "python" and syntax is not None:
        return syntax
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
            return {
                **(syntax or {}),
                "symbols": (syntax or {}).get("symbols", found),
                "imports": imports,
                "parser": "python_ast",
            }
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
        self.tracker = None
        self.initialized = False
        self.last_changed = []

    def close(self):
        if self.tracker:
            self.tracker.close()

    def ensure_current(self):
        if self.tracker is None:
            self.tracker = ChangeTracker(self.root)
        full, candidates = self.tracker.candidates()
        if full or not self.initialized:
            self.refresh()
            self.tracker.reconciled()
        elif candidates:
            previous = {
                r[0]
                for r in self.store.db.execute(
                    "SELECT path FROM repository_files WHERE workspace=?", (str(self.root),)
                )
            }
            paths = expand_paths(self.root, candidates, previous)
            self.refresh(p for p in paths if not set(Path(p).parts) & EXCLUDED)

    def paths(self):
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(
                d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink()
            )
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                if name.startswith(".threadweave-watch-"):
                    continue
                rel = path.relative_to(self.root).as_posix()
                try:
                    confined(self.root, rel, allowed=self.allowed, forbidden=self.forbidden)
                except PermissionError:
                    continue
                if path.stat().st_size <= 2_000_000:
                    yield rel

    def refresh(self, paths=None):
        with self.store.transaction():
            return self._refresh(paths)

    def _refresh(self, paths=None):
        full = paths is None
        paths = list(self.paths()) if full else list(paths)
        changed = []
        for relative in paths:
            if set(Path(relative).parts) & EXCLUDED:
                continue
            path = confined(self.root, relative, allowed=self.allowed, forbidden=self.forbidden)
            old = self.store.db.execute(
                "SELECT * FROM repository_files WHERE workspace=? AND path=?",
                (str(self.root), relative),
            ).fetchone()
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
                self.store.db.execute(
                    "DELETE FROM code_evidence WHERE workspace=? AND path=?",
                    (str(self.root), relative),
                )
                self.store.db.execute(
                    "DELETE FROM repository_files WHERE workspace=? AND path=?",
                    (str(self.root), relative),
                )
                continue
            stat = path.stat()
            try:
                old_body = json.loads(old["body"]) if old else {}
                indexed = old and "quality" in old_body
            except (ValueError, TypeError):
                indexed = False  # Disposable corrupt index row is rebuilt, not trajectory data.
            if (
                full
                and old
                and indexed
                and old["mtime_ns"] == stat.st_mtime_ns
                and old["size"] == stat.st_size
                and old_body.get("stat_identity") == [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
            ):
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                self.store.db.execute(
                    "DELETE FROM repository_files WHERE workspace=? AND path=?",
                    (str(self.root), relative),
                )
                self.store.db.execute(
                    "DELETE FROM code_evidence WHERE workspace=? AND path=?",
                    (str(self.root), relative),
                )
                continue
            text = raw.decode(errors="replace")
            language = LANGUAGES.get(path.suffix, "text")
            if old and indexed and old["sha256"] == digest(raw) and old["language"] == language:
                old_body["stat_identity"] = [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
                self.store.db.execute(
                    "UPDATE repository_files SET mtime_ns=?,size=?,body=? WHERE workspace=? AND path=?",
                    (stat.st_mtime_ns, stat.st_size, encode(old_body), str(self.root), relative),
                )
                continue
            body = (
                symbols(text, language)
                if self.enhanced
                else {"symbols": [], "imports": [], "parser": "disabled"}
            )
            body.setdefault("quality", "lexical" if self.enhanced else "disabled")
            body["stat_identity"] = [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
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
            self.store.db.execute(
                "DELETE FROM code_evidence WHERE workspace=? AND path=?", (str(self.root), relative)
            )
            for kind in ("symbols", "references", "calls", "imports", "inheritance"):
                self.store.db.executemany(
                    "INSERT INTO code_evidence VALUES(?,?,?,?,?,?,?)",
                    [
                        (
                            str(self.root),
                            relative,
                            kind,
                            item.get("name") or item.get("module") or item.get("text", ""),
                            item.get("enclosing"),
                            encode(item),
                            (item.get("short_name") or item.get("name") or item.get("module") or "")
                            .split(".")[-1]
                            .split("::")[-1],
                        )
                        for item in body.get(kind, [])
                    ],
                )
        if full:
            keep = set(paths)
            for row in self.store.db.execute(
                "SELECT path FROM repository_files WHERE workspace=?", (str(self.root),)
            ).fetchall():
                if row[0] not in keep:
                    self.store.db.execute(
                        "DELETE FROM code_evidence WHERE workspace=? AND path=?",
                        (str(self.root), row[0]),
                    )
                    self.store.db.execute(
                        "DELETE FROM repository_files WHERE workspace=? AND path=?",
                        (str(self.root), row[0]),
                    )
        self.initialized = True
        self.last_changed = changed
        return changed

    def entries(self):
        self.ensure_current()
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
        return self.evidence("symbols", query, limit=limit)

    def evidence(self, kind, query="", *, limit=20, owner=False):
        self.ensure_current()
        field = "enclosing" if owner else "name"
        rows = self.store.db.execute(
            f"SELECT path,body,name FROM code_evidence WHERE workspace=? AND kind=? AND {field}=? ORDER BY path LIMIT ?",
            (str(self.root), kind, query, min(200, limit)),
        ).fetchall()
        if not rows and not owner:
            rows = self.store.db.execute(
                "SELECT path,body,name FROM code_evidence WHERE workspace=? AND kind=? AND short_name=? ORDER BY path LIMIT ?",
                (str(self.root), kind, query, min(200, limit)),
            ).fetchall()
        if not rows:
            rows = self.store.db.execute(
                f"SELECT path,body,name FROM code_evidence WHERE workspace=? AND kind=? AND {field} LIKE ? LIMIT ?",
                (str(self.root), kind, "%" + query + "%", min(200, limit)),
            ).fetchall()
        return {
            "matches": [
                {
                    "path": r["path"],
                    **json.loads(r["body"]),
                    "rank_reason": "exact" if r["name"] == query else "name overlap",
                }
                for r in rows
            ],
            "precision": "syntax-derived; unresolved names may be ambiguous",
        }

    def definition(self, query, *, limit=20):
        return self.evidence("symbols", query, limit=limit)

    def references(self, query, *, limit=20):
        return self.evidence("references", query, limit=limit)

    def callers(self, query, *, limit=20):
        return self.evidence("calls", query, limit=limit)

    def callees(self, query, *, limit=20):
        return self.evidence("calls", query, limit=limit, owner=True)

    def context_for_symbol(self, query, *, limit=10):
        return {
            "definitions": self.definition(query, limit=limit),
            "callers": self.callers(query, limit=limit),
            "callees": self.callees(query, limit=limit),
        }

    def changed_symbols(self):
        self.ensure_current()
        return [{"path": p, **s} for p in self.last_changed for s in self.outline(p)["symbols"]]

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
        self.ensure_current()
        outline = self.outline(path)
        related = []
        for item in outline["imports"]:
            for name in [item.get("module"), *item.get("names", [])]:
                if name:
                    normalized = name.strip("\"'<>").replace("::", "/").replace(".", "/")
                    stem = normalized.split("/")[-1]
                    related.extend(
                        r[0]
                        for r in self.store.db.execute(
                            "SELECT path FROM repository_files WHERE workspace=? AND "
                            "(path LIKE ? OR path LIKE ? OR path=?) LIMIT 30",
                            (str(self.root), "%/" + stem + ".%", stem + ".%", name),
                        )
                    )
        return {
            "path": path,
            "imports": outline["imports"],
            "likely_local_modules": sorted(set(related)),
            "quality": "heuristic module resolution over syntax imports",
        }

    def dependents(self, path, *, current=True):
        if current:
            self.ensure_current()
        stem = Path(path).stem
        return {
            "path": path,
            "matches": [
                dict(r)
                for r in self.store.db.execute(
                    "SELECT DISTINCT path FROM code_evidence WHERE workspace=? AND kind='imports' "
                    "AND body LIKE ? LIMIT 100",
                    (str(self.root), "%" + stem + "%"),
                )
            ],
            "quality": "heuristic import overlap",
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
