"""Parse common tool output and retrieve implicated source; never decide the fix."""

import csv
import io
import re

LOCATION = re.compile(
    r'(?P<file>[\w./\\-]+\.(?:py|rs|go|c|cc|cpp|h|hpp|cu|cuh|js|jsx|ts|tsx))(?::|", line )(?P<line>\d+)(?::(?P<column>\d+))?'
)


def parse_diagnostics(text, *, limit=30):
    lines, diagnostics, failures = text.splitlines(), [], []
    for i, line in enumerate(lines):
        if match := LOCATION.search(line):
            diagnostics.append(
                {
                    "file": match["file"],
                    "line": int(match["line"]),
                    "column": int(match["column"]) if match["column"] else None,
                    "message": line[:1000],
                    "excerpt": "\n".join(lines[max(0, i - 2) : i + 4])[:2000],
                }
            )
        if match := re.search(r"(?:FAILED\s+|FAIL:\s+|--- FAIL: |FAIL\s+)([^\s]+)(.*)", line):
            failures.append({"name": match[1], "message": match[2].strip()[:1000]})
        elif re.search(r"^(?:E\s+|error(?:\[|:)|AssertionError|FAILURES)", line):
            failures.append({"name": None, "message": line[:1000]})
    return {
        "diagnostics": diagnostics[:limit],
        "failures": failures[:limit],
        "diagnostic_count": len(diagnostics),
        "failure_count": len(failures),
    }


def localize(context, result):
    index = context.runtime.index(context.session_id)
    evidence = []
    edits = context.runtime.store.events(
        context.session_id, kind="code_edit", limit=20
    ) + context.runtime.store.events(context.session_id, kind="workspace_effects", limit=20)
    for diagnostic in result.get("diagnostics", [])[:15]:
        path = diagnostic["file"]
        try:
            file = context.path(path)
            relative = file.relative_to(index.root).as_posix()
            outline = index.outline(relative)
            line = diagnostic["line"]
            nearby = [
                s
                for s in outline["symbols"]
                if s["line"] <= line <= s.get("end_line", s["line"] + 30)
            ]
            evidence.append(
                {
                    "diagnostic": diagnostic,
                    "definitions": nearby,
                    "recent_edits": [
                        e["id"] for e in edits if relative in e["payload"].get("files", {})
                    ],
                    "likely_references": [
                        index.search(s["name"].split(".")[-1], limit=5)["matches"]
                        for s in nearby[:2]
                    ],
                }
            )
        except (ValueError, PermissionError, OSError):
            evidence.append({"diagnostic": diagnostic, "source_available": False})
    return {
        "evidence": evidence,
        "note": "Lexical references and edit proximity are evidence, not causal attribution.",
    }


def profiler_metrics(text):
    rows = []
    lines = text.splitlines()
    header = next(
        (i for i, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line),
        None,
    )
    if header is not None:
        for record in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
            if record.get("Metric Name") and record.get("Metric Value"):
                raw = record["Metric Value"]
                try:
                    value = float(raw.replace(",", ""))
                except ValueError:
                    value = raw
                rows.append(
                    {
                        "metric": record["Metric Name"],
                        "value": value,
                        "unit": record.get("Metric Unit"),
                        "source": "profiler_csv",
                    }
                )
    for match in re.finditer(r"^([\w.]+)\s*=\s*([-+0-9.eE]+)\s*(\S*)$", text, re.M):
        try:
            value = float(match[2])
        except ValueError:
            continue
        rows.append(
            {"metric": match[1], "value": value, "unit": match[3], "source": "output_assignment"}
        )
    return rows[:100]
