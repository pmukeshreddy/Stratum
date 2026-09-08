"""Import/package binding and optional Python inference. Evidence is not a type proof.

Binding rows persist source names even when unresolved, so adding/moving a module
can rebind importers without reparsing them. Dynamic dispatch stays unresolved.
"""

import ast
import json
import re
from pathlib import Path


def imports(text, language, path):
    if language == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        result = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.extend((a.name, a.asname or a.name.split(".")[0], "") for a in node.names)
            if isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                result.extend((module, a.asname or a.name, a.name) for a in node.names)
        return result
    from .module_imports import imports as syntax_imports

    return syntax_imports(text, language)


class Resolver:
    def __init__(self, index):
        self.index, self.root, self.db = index, index.root, index.store.db
        self.paths = set()
        self.settings = {}

    def configure(self):
        self.paths = {
            r[0]
            for r in self.db.execute(
                "SELECT path FROM repository_files WHERE workspace=?", (str(self.root),)
            )
        }
        for file in ("tsconfig.json", "go.mod", "compile_commands.json"):
            p = self.root / file
            self.settings[file] = (
                p.read_text() if p.is_file() and p.stat().st_size < 2_000_000 else ""
            )

    def target(self, path, language, module, symbol=""):
        parent = Path(path).parent
        candidates = []
        if language == "python":
            if module.startswith("."):
                level = len(module) - len(module.lstrip("."))
                base = parent
                for _ in range(level - 1):
                    base = base.parent
                stems = [base / module[level:].replace(".", "/")]
            else:
                stems = [Path(module.replace(".", "/")), Path("src") / module.replace(".", "/")]
            for stem in stems:
                candidates.extend([str(stem) + ".py", str(stem / "__init__.py")])
                if symbol:
                    candidates.extend(
                        [str(stem / symbol) + ".py", str(stem / symbol / "__init__.py")]
                    )
        elif language in {"javascript", "typescript"}:
            stems = [parent / module] if module.startswith(".") else []
            try:
                options = json.loads(self.settings.get("tsconfig.json") or "{}").get(
                    "compilerOptions", {}
                )
                base = Path(options.get("baseUrl", "."))
                for key, values in options.get("paths", {}).items():
                    prefix, _, suffix = key.partition("*")
                    if module.startswith(prefix) and (not suffix or module.endswith(suffix)):
                        inner = module[len(prefix) : len(module) - len(suffix) if suffix else None]
                        stems.extend(base / v.replace("*", inner) for v in values)
                if options.get("baseUrl"):
                    stems.append(base / module)
            except (ValueError, TypeError):
                pass
            for stem in stems:
                candidates.extend(
                    str(stem) + ext
                    for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js")
                )
        elif language == "rust":
            parts = module.replace("::", "/")
            if parts.startswith("crate/"):
                base = Path("src") / parts[6:]
            elif parts.startswith("self/"):
                base = parent / parts[5:]
            elif parts.startswith("super/"):
                base = parent.parent / parts[6:]
            else:
                base = parent / parts
            candidates = [str(base) + ".rs", str(base / "mod.rs")]
        elif language == "go":
            match = re.search(r"^module\s+(\S+)", self.settings.get("go.mod", ""), re.M)
            if match and (module == match[1] or module.startswith(match[1] + "/")):
                directory = module[len(match[1]) :].lstrip("/")
                # A Go import binds a package, not an arbitrary single source file.
                return "package:" + directory
        elif language in {"c", "cpp", "cuda"}:
            candidates = [str(parent / module), module, "include/" + module]
            try:
                for entry in json.loads(self.settings.get("compile_commands.json") or "[]"):
                    if Path(entry.get("file", "")).name != Path(path).name:
                        continue
                    command = entry.get("command", " ".join(entry.get("arguments", [])))
                    for include in re.findall(r"-I\s*(\S+)", command):
                        candidate = Path(entry.get("directory", str(self.root))) / include / module
                        if candidate.is_relative_to(self.root):
                            candidates.append(str(candidate.relative_to(self.root)))
            except (ValueError, TypeError):
                pass
        for candidate in candidates:
            normalized = str(Path(__import__("os").path.normpath(candidate)))
            if normalized in self.paths:
                return normalized
        return None

    def update(self, changed):
        changed = list(changed)
        settings_changed = any(
            Path(p).name in {"tsconfig.json", "go.mod", "compile_commands.json"} for p in changed
        )
        if not self.paths:
            self.configure()
        if settings_changed:
            self.configure()
        topology = settings_changed
        for path in changed:
            exists = self.db.execute(
                "SELECT language,body FROM repository_files WHERE workspace=? AND path=?",
                (str(self.root), path),
            ).fetchone()
            topology |= bool(exists) != (path in self.paths)
            self.paths.discard(path)
            self.db.execute(
                "DELETE FROM module_bindings WHERE workspace=? AND path=?", (str(self.root), path)
            )
            if exists:
                self.paths.add(path)
                for module, alias, symbol in imports(
                    (self.root / path).read_text(errors="replace"), exists[0], path
                ):
                    target = self.target(path, exists[0], module, symbol)
                    self.db.execute(
                        "INSERT OR REPLACE INTO module_bindings VALUES(?,?,?,?,?,?,?)",
                        (
                            str(self.root),
                            path,
                            module,
                            target,
                            alias,
                            symbol,
                            "resolved structural" if target else "syntax-derived",
                        ),
                    )
        if topology:
            # No reparsing on module addition/deletion; only existing import rows.
            for r in self.db.execute(
                "SELECT b.*,f.language FROM module_bindings b JOIN repository_files f ON b.workspace=f.workspace AND b.path=f.path WHERE b.workspace=?",
                (str(self.root),),
            ).fetchall():
                target = self.target(r["path"], r["language"], r["module"], r["symbol"])
                self.db.execute(
                    "UPDATE module_bindings SET target=?,quality=? WHERE workspace=? AND path=? AND module=? AND alias=? AND symbol=?",
                    (
                        target,
                        "resolved structural" if target else "syntax-derived",
                        str(self.root),
                        r["path"],
                        r["module"],
                        r["alias"],
                        r["symbol"],
                    ),
                )

    def bindings(self, path):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM module_bindings WHERE workspace=? AND path=?", (str(self.root), path)
            )
        ]

    def infer(self, path, line, column):
        """Jedi static inference. No code execution, no claim of compiler proof."""
        import jedi

        from .repository import confined

        source = confined(
            self.root, path, allowed=self.index.allowed, forbidden=self.index.forbidden
        )
        if source.suffix != ".py":
            raise ValueError(
                "repo.resolve currently offers Jedi static inference for Python; use resolved structural queries for other languages"
            )
        script = jedi.Script(
            path=source,
            project=jedi.Project(self.root, sys_path=[str(self.root), str(self.root / "src")]),
        )
        return {
            "matches": [
                {
                    "symbol_id": n.full_name,
                    "name": n.name,
                    "path": str(n.module_path),
                    "line": n.line,
                    "column": n.column + 1 if n.column is not None else None,
                    "kind": n.type,
                    "quality": "Jedi static inference",
                    "relationship": "definition",
                    "rank_reason": "resolved at supplied source position",
                }
                for n in script.goto(
                    line, column - 1, follow_imports=True, follow_builtin_imports=False
                )
            ]
        }
