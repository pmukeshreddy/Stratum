"""Read-only AST inventory of refinement-related tests in a supplied Prime checkout.

Prints JSON; it neither executes Prime nor starts providers or evaluations.
Each row denotes a test definition (parameterized definitions retain their case table).
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import tree_sitter_typescript
from tree_sitter import Language, Parser


def inventory(root):
    parser = Parser(Language(tree_sitter_typescript.language_typescript()))
    paths = subprocess.run(
        ["rg", "--files", str(root / "test"), "-g", "*.test.ts"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    records = []
    for filename in sorted(paths):
        path = Path(filename)
        source = path.read_bytes()
        tree = parser.parse(source)

        def visit(node, groups=(), source=source, path=path):
            if node.type == "call_expression":
                function, arguments = (
                    node.child_by_field_name("function"),
                    node.child_by_field_name("arguments"),
                )
                if function is not None and arguments is not None and arguments.named_children:
                    callee = source[function.start_byte : function.end_byte].decode()
                    first = arguments.named_children[0]
                    if first.type == "string":
                        name = source[first.start_byte + 1 : first.end_byte - 1].decode()
                        if callee in {"describe", "describe.sequential"}:
                            groups = (*groups, name)
                        if callee in {"it", "test"} or callee.startswith(
                            ("it.each", "test.each", "it.skip", "test.skip")
                        ):
                            body = source[node.start_byte : node.end_byte].decode()
                            related = (
                                any(
                                    term in body.lower()
                                    for term in (
                                        "refin",
                                        "harnessdigest",
                                        "harness_digest",
                                        "harnessstate",
                                        "serializeconversation",
                                        "converttollm",
                                    )
                                )
                                or "digest" in name.lower()
                            )
                            if (
                                related
                                or "refin" in path.name
                                or path.name
                                in {
                                    "provider-retry.test.ts",
                                    "compaction-serialization.test.ts",
                                    "session-command-messages.test.ts",
                                }
                            ):
                                records.append(
                                    {
                                        "file": str(path.relative_to(root)),
                                        "line": node.start_point.row + 1,
                                        "end_line": node.end_point.row + 1,
                                        "name": name,
                                        "groups": groups,
                                        "body": body,
                                    }
                                )
            for child in node.named_children:
                visit(child, groups)

        visit(tree.root_node)
    path = root.parent.parent / "prime-agent-runtime/test/test_harness.py"
    source = path.read_text()
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            records.append(
                {
                    "file": "../../prime-agent-runtime/test/test_harness.py",
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "name": node.name,
                    "groups": ["Python harness"],
                    "body": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                }
            )
    return records


if __name__ == "__main__":
    rows = inventory(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        rows = rows[int(sys.argv[2]) : int(sys.argv[3])]
    else:
        rows = [{k: v for k, v in row.items() if k != "body"} for row in rows]
    print(json.dumps(rows, ensure_ascii=False))
