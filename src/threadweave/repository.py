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
        from .resolution import Resolver

        self.resolver = Resolver(self)

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
            previous = self.resolver.paths
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
            old = self.store.records.first(
                "repository_files", workspace=str(self.root), path=relative
            )
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
                if old:
                    changed.append(relative)
                self.store.records.delete("code_evidence", workspace=str(self.root), path=relative)
                self.store.records.delete(
                    "repository_files", workspace=str(self.root), path=relative
                )
                continue
            stat = path.stat()
            try:
                old_body = old["body"] if old else {}
                indexed = old and "quality" in old_body
            except (ValueError, TypeError):
                indexed = False  # Disposable corrupt index row is rebuilt, not trajectory data.
            if (
                old
                and indexed
                and old["mtime_ns"] == stat.st_mtime_ns
                and old["size"] == stat.st_size
                and old_body.get("stat_identity") == [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
            ):
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                self.store.records.delete(
                    "repository_files", workspace=str(self.root), path=relative
                )
                self.store.records.delete("code_evidence", workspace=str(self.root), path=relative)
                continue
            text = raw.decode(errors="replace")
            language = LANGUAGES.get(path.suffix, "text")
            if old and indexed and old["sha256"] == digest(raw) and old["language"] == language:
                old_body["stat_identity"] = [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
                self.store.records.update(
                    "repository_files",
                    {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "body": old_body},
                    workspace=str(self.root),
                    path=relative,
                )
                continue
            body = (
                symbols(text, language)
                if self.enhanced
                else {"symbols": [], "imports": [], "parser": "disabled"}
            )
            body.setdefault("quality", "lexical" if self.enhanced else "disabled")
            body["stat_identity"] = [stat.st_dev, stat.st_ino, stat.st_ctime_ns]
            self.store.records.insert(
                "repository_files",
                {
                    "workspace": str(self.root),
                    "path": relative,
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "sha256": digest(raw),
                    "language": language,
                    "body": body,
                },
                on_conflict="replace",
            )
            changed.append(relative)
            self.store.records.delete("code_evidence", workspace=str(self.root), path=relative)
            for kind in ("symbols", "references", "calls", "imports", "inheritance"):
                self.store.records.insert_many(
                    "code_evidence",
                    [
                        dict(
                            zip(
                                (
                                    "workspace",
                                    "path",
                                    "kind",
                                    "name",
                                    "enclosing",
                                    "body",
                                    "short_name",
                                ),
                                values,
                                strict=True,
                            )
                        )
                        for values in [
                            (
                                str(self.root),
                                relative,
                                kind,
                                item.get("name") or item.get("module") or item.get("text", ""),
                                item.get("enclosing"),
                                item,
                                (
                                    item.get("short_name")
                                    or item.get("name")
                                    or item.get("module")
                                    or ""
                                )
                                .split(".")[-1]
                                .split("::")[-1],
                            )
                            for item in body.get(kind, [])
                        ]
                    ],
                )
        if full:
            keep = set(paths)
            for row in self.store.records.select(
                "repository_files", workspace=str(self.root), fields=("path",)
            ):
                if row["path"] not in keep:
                    changed.append(row["path"])
                    self.store.records.delete(
                        "code_evidence", workspace=str(self.root), path=row["path"]
                    )
                    self.store.records.delete(
                        "repository_files", workspace=str(self.root), path=row["path"]
                    )
        if full and not self.initialized:
            self.resolver.configure()
            # Migration/recovery rebuilds this disposable relationship projection.
            self.resolver.update(paths)
        else:
            self.resolver.update(changed)
        self.initialized = True
        self.last_changed = changed
        return changed

    def entries(self):
        self.ensure_current()
        return [
            dict(row)
            for row in self.store.records.select(
                "repository_files", workspace=str(self.root), order=(("path", False),)
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
        row = self.store.records.first(
            "repository_files", workspace=str(self.root), path=path, fields=("body",)
        )
        if not row:
            raise ValueError("File is absent, binary, or exceeds the 2 MB indexing limit")
        return {"path": path, **row["body"]}

    def symbol_search(self, query, *, limit=100):
        return self.evidence("symbols", query, limit=limit)

    def evidence(self, kind, query="", *, limit=20, owner=False):
        self.ensure_current()
        field = "enclosing" if owner else "name"
        rows = self.store.records.select(
            "code_evidence",
            workspace=str(self.root),
            kind=kind,
            where=lambda row: row[field] == query,
            order=(("path", False),),
            limit=min(200, limit),
            fields=("path", "body", "name"),
        )
        if not rows and not owner:
            rows = self.store.records.select(
                "code_evidence",
                workspace=str(self.root),
                kind=kind,
                short_name=query,
                order=(("path", False),),
                limit=min(200, limit),
                fields=("path", "body", "name"),
            )
        if not rows:
            rows = self.store.records.select(
                "code_evidence",
                workspace=str(self.root),
                kind=kind,
                where=lambda row: query.casefold() in (row[field] or "").casefold(),
                limit=min(200, limit),
                fields=("path", "body", "name"),
            )
        return {
            "matches": [
                {
                    "path": r["path"],
                    **r["body"],
                    "rank_reason": "exact" if r["name"] == query else "name overlap",
                }
                for r in rows
            ],
            "precision": "syntax-derived; unresolved names may be ambiguous",
        }

    def definition(self, query, *, limit=20):
        path, separator, name = query.partition("::")
        result = self.evidence("symbols", name if separator else query, limit=limit)
        if separator:
            result["matches"] = [m for m in result["matches"] if m["path"] == path]
        for item in result["matches"]:
            item.update(symbol_id=item["path"] + "::" + item["name"], relationship="definition")
        return result

    def declaration(self, query, *, limit=20):
        result = self.definition(query, limit=limit)
        for item in result["matches"]:
            item["relationship"] = "declaration/definition"
        return result

    def resolve(self, path, line, column):
        self.ensure_current()
        return self.resolver.infer(path, line, column)

    def _incoming(self, query, kind, limit):
        definitions = self.definition(query, limit=limit)["matches"]
        resolved = []
        for definition in definitions:
            short = definition.get("short_name", definition["name"].split(".")[-1])
            for binding in self.store.records.select(
                "module_bindings", workspace=str(self.root), target=definition["path"]
            ):
                if binding["symbol"] not in {"", "*", short}:
                    continue
                name = (
                    binding["alias"]
                    if binding["symbol"] == short
                    else (binding["alias"] + "." + short if binding["alias"] else short)
                )
                names = (name, name.replace(".", "::"))
                for row in self.store.records.select(
                    "code_evidence",
                    workspace=str(self.root),
                    path=binding["path"],
                    kind=kind,
                    where=lambda row, names=names: row["name"] in names,
                    limit=limit,
                    fields=("body",),
                ):
                    resolved.append(
                        {
                            **row["body"],
                            "path": binding["path"],
                            "symbol_id": definition["symbol_id"],
                            "relationship": "calls" if kind == "calls" else "references",
                            "quality": "resolved structural",
                            "rank_reason": "explicit import binding to target module and exported name; dynamic shadowing is not type-checked",
                        }
                    )
        fallback = self.evidence(kind, query.split("::")[-1], limit=limit)["matches"]
        seen = {(m["path"], m["line"], m["name"]) for m in resolved}
        for item in fallback:
            if (item["path"], item["line"], item["name"]) not in seen:
                item.update(
                    relationship=kind,
                    rank_reason="unresolved syntax occurrence; may refer to a different symbol",
                )
                resolved.append(item)
        return {
            "matches": resolved[:limit],
            "precision": "per-item evidence quality; static import binding is not dynamic dispatch proof",
        }

    def references(self, query, *, limit=20):
        return self._incoming(query, "references", limit)

    def callers(self, query, *, limit=20):
        return self._incoming(query, "calls", limit)

    def callees(self, query, *, limit=20):
        return self.evidence("calls", query, limit=limit, owner=True)

    def context_for_symbol(self, query, *, limit=10):
        import itertools

        definitions = self.definition(query, limit=limit)
        sources = []
        for definition in definitions["matches"][:2]:
            path = confined(
                self.root, definition["path"], allowed=self.allowed, forbidden=self.forbidden
            )
            start = max(1, definition["line"])
            end = min(definition.get("end_line", start + 30), start + 79)
            with path.open(errors="replace") as stream:
                lines = list(itertools.islice(stream, start - 1, end))
            source = "".join(f"{start + i}: {line}" for i, line in enumerate(lines))
            sources.append(
                {
                    "symbol_id": definition["symbol_id"],
                    "file": definition["path"],
                    "start_line": start,
                    "end_line": end,
                    "source": source[:4000],
                    "truncated": len(source) > 4000 or end < definition.get("end_line", end),
                    "quality": "exact source excerpt; identity syntax-derived",
                }
            )
        return {
            "sources": sources,
            "definitions": definitions,
            "callers": self.callers(query, limit=limit),
            "callees": self.callees(query, limit=limit),
        }

    def implementations(self, query, *, limit=20):
        return self.evidence("inheritance", query, limit=limit)

    def related_symbols(self, query, *, limit=10):
        return self.context_for_symbol(query, limit=limit)

    def changed_symbols(self):
        self.ensure_current()
        return [
            {"path": p, **s}
            for p in list(self.last_changed)
            if (self.root / p).is_file()
            for s in self.outline(p)["symbols"]
        ]

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
        bindings = self.resolver.bindings(path)
        return {
            "path": path,
            "imports": bindings,
            "likely_local_modules": sorted({b["target"] for b in bindings if b["target"]}),
            "quality": "per-import resolved structural or unresolved evidence",
        }

    def _lexical_dependencies(self, path):
        outline = self.outline(path)
        related = []
        for item in outline["imports"]:
            for name in [item.get("module"), *item.get("names", [])]:
                if name:
                    normalized = name.strip("\"'<>").replace("::", "/").replace(".", "/")
                    stem = normalized.split("/")[-1]
                    related.extend(
                        r["path"]
                        for r in self.store.records.select(
                            "repository_files",
                            workspace=str(self.root),
                            where=lambda row, stem=stem, name=name: (
                                Path(row["path"]).stem.casefold() == stem.casefold()
                                or row["path"].casefold() == name.casefold()
                            ),
                            limit=30,
                            fields=("path",),
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
        parent = str(Path(path).parent)
        targets = {path, "package:" + ("" if parent == "." else parent)}
        matches = self.store.records.select(
            "module_bindings",
            workspace=str(self.root),
            where=lambda row: row["target"] in targets,
            fields=("path", "quality"),
        )
        unique = {(row["path"], row["quality"]): row for row in matches}
        return {
            "path": path,
            "matches": list(unique.values())[:100],
            "quality": "resolved structural module binding",
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
