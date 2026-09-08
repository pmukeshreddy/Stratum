"""Framework-produced evidence, retaining raw output and explicit parse provenance."""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def framework(command):
    names = {Path(x).name for x in command}
    for name in ("pytest", "unittest", "jest", "vitest", "ctest"):
        if name in names:
            return name
    if "cargo" in names and "test" in names:
        return "cargo"
    if "go" in names and "test" in names:
        return "go"
    return "compiler"


def machine_command(command, report, *, local=True):
    name, cmd = framework(command), list(command)
    if name == "pytest" and local and not any(x.startswith("--junit") for x in cmd):
        cmd.append("--junitxml=" + str(report))
    elif name == "ctest" and local and "--output-junit" not in cmd:
        cmd += ["--output-junit", str(report)]
    elif name == "go" and "-json" not in cmd:
        cmd.insert(cmd.index("test") + 1, "-json")
    elif name == "cargo" and not any(x.startswith("--message-format") for x in cmd):
        position = cmd.index("--") if "--" in cmd else len(cmd)
        cmd.insert(position, "--message-format=json")
    elif name in {"jest", "vitest"} and "--json" not in cmd:
        cmd.append("--json" if name == "jest" else "--reporter=json")
    return name, cmd


def junit(text, name):
    root = ET.fromstring(text)
    records = []
    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        skipped = case.find("skipped") is not None
        identifier = case.get("classname", "").replace(".", "/") + "::" + case.get("name", "")
        if name == "pytest":
            parts = [p for p in case.get("classname", "").split(".") if p]
            boundary = next((i for i, p in enumerate(parts) if p[:1].isupper()), len(parts))
            module = "/".join(parts[:boundary])
            # Collection errors can have no classname and a module as name.
            # Never invent the bogus path '.py' when metadata is absent.
            collection = problem is not None and problem.get("message") == "collection failure"
            if not module and collection:
                module = case.get("name", "").replace(".", "/")
            filename = case.get("file") or (module + ".py" if module else None)
            identifier = "::".join(
                [
                    p
                    for p in [
                        filename,
                        *parts[boundary:],
                        None if collection else case.get("name", ""),
                    ]
                    if p
                ]
            ) or case.get("name", "unknown")
        records.append(
            {
                "framework": name,
                "test_id": identifier,
                "name": identifier,
                "file": case.get("file") or (filename if name == "pytest" else None),
                "line": int(case.get("line", "0")) or None,
                "status": "failed" if problem is not None else "skipped" if skipped else "passed",
                "duration": float(case.get("time", "0")),
                "failure_type": problem.get("type") if problem is not None else None,
                "message": problem.get("message", "")[:1500] if problem is not None else "",
                "stack": (problem.text or "")[-4000:] if problem is not None else "",
                "captured_output": (
                    case.findtext("system-out", "") + case.findtext("system-err", "")
                )[-2000:],
                "provenance": "framework_junit",
            }
        )
    return records


def resolve_locations(evidence, workspace, *, container=False):
    """Resolve framework paths without changing its rootdir/configuration semantics.

    Pytest may choose an ancestor project as rootdir for a nested checkout. Keep
    the original identity for provenance, but expose runnable workspace-relative
    test IDs. Never guess by basename/suffix; require an actual in-workspace file.
    Resolution is cached per distinct reported path, not per test case.
    """
    root = Path(workspace).resolve()
    cache = {}

    def resolve(raw):
        if raw in cache:
            return cache[raw]
        source = Path(raw)
        candidates = (
            [source] if source.is_absolute() else [p / source for p in (root, *root.parents)]
        )
        if container and source.is_absolute() and source.is_relative_to("/workspace"):
            candidates.insert(0, root / source.relative_to("/workspace"))
        result = None
        for candidate in candidates:
            try:
                path = candidate.resolve()
                if path.is_relative_to(root) and path.is_file():
                    result = str(path.relative_to(root))
                    break
            except (OSError, RuntimeError):
                continue
        cache[raw] = result
        return result

    seen = set()
    for record in [
        *evidence.get("tests", []),
        *evidence.get("failures", []),
        *evidence.get("diagnostics", []),
    ]:
        if id(record) in seen:
            continue
        seen.add(id(record))
        for item in [record, *record.get("stack_frames", [])]:
            raw = item.get("file")
            if not raw:
                continue
            relative = resolve(raw)
            item["path_resolution"] = "workspace file" if relative else "unresolved"
            if relative is None or relative == raw:
                continue
            item["reported_file"], item["file"] = raw, relative
            if item.get("test_id", "").startswith(raw + "::"):
                item["reported_test_id"] = item["test_id"]
                item["test_id"] = relative + item["test_id"][len(raw) :]
                item["name"] = item["test_id"]
    return evidence


def structured(text, name, xml=None):
    from .diagnostics import parse_diagnostics

    fallback = parse_diagnostics(text)
    tests, diagnostics, errors = [], [], []
    if xml:
        try:
            tests = junit(xml, name)
        except (ET.ParseError, ValueError) as exc:
            errors.append(str(exc))
    packets = []
    try:
        document, offset = json.JSONDecoder().raw_decode(text.lstrip())
        if (
            isinstance(document, dict)
            and ("Action" in document or "reason" in document)
            and text.lstrip()[offset:].strip()
        ):
            raise ValueError("JSONL stream: parse every event")
        packets = (
            [document]
            if isinstance(document, dict)
            else document
            if isinstance(document, list)
            else []
        )
    except ValueError:
        for line in text.splitlines():
            try:
                packet = json.loads(line)
                if isinstance(packet, dict):
                    packets.append(packet)
            except ValueError:
                continue
    go_output = {}
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        for entry in packet.get("generalDiagnostics", []):
            start = entry.get("range", {}).get("start", {})
            diagnostics.append(
                {
                    "file": entry.get("file"),
                    "line": start.get("line", 0) + 1,
                    "column": start.get("character", 0) + 1,
                    "severity": entry.get("severity"),
                    "diagnostic_code": entry.get("rule"),
                    "message": entry.get("message"),
                    "provenance": "pyright_json",
                }
            )
        if packet.get("kind") in {"error", "warning", "note"} and packet.get("locations"):
            for location in packet["locations"]:
                point = location.get("caret", {})
                diagnostics.append(
                    {
                        "file": point.get("file"),
                        "line": point.get("line"),
                        "column": point.get("column"),
                        "severity": packet["kind"],
                        "message": packet.get("message"),
                        "diagnostic_code": packet.get("option"),
                        "provenance": "gcc_json",
                    }
                )
        if packet.get("Action") == "output":
            key = (packet.get("Package", ""), packet.get("Test", ""))
            go_output[key] = (go_output.get(key, "") + packet.get("Output", ""))[-4000:]
        if packet.get("Action") in {"pass", "fail", "skip"} and packet.get("Test"):
            identifier = packet.get("Package", "") + "/" + packet["Test"]
            tests.append(
                {
                    "framework": "go",
                    "test_id": identifier,
                    "name": identifier,
                    "status": {"pass": "passed", "fail": "failed", "skip": "skipped"}[
                        packet["Action"]
                    ],
                    "duration": packet.get("Elapsed"),
                    "message": go_output.get((packet.get("Package", ""), packet["Test"]), ""),
                    "provenance": "go_test_json",
                }
            )
        for file in packet.get("testResults", []):
            for test in file.get("assertionResults", []):
                identifier = test.get("fullName") or test.get("title", "")
                tests.append(
                    {
                        "framework": name,
                        "test_id": identifier,
                        "name": identifier,
                        "file": file.get("name"),
                        "status": test.get("status"),
                        "duration": (test.get("duration") or 0) / 1000,
                        "message": "\n".join(test.get("failureMessages", []))[-4000:],
                        "provenance": "framework_json",
                    }
                )
        if packet.get("reason") == "compiler-message":
            message = packet["message"]
            for span in message.get("spans", []):
                if span.get("is_primary"):
                    diagnostics.append(
                        {
                            "file": span["file_name"],
                            "line": span["line_start"],
                            "column": span["column_start"],
                            "severity": message["level"],
                            "diagnostic_code": (message.get("code") or {}).get("code"),
                            "message": message["message"],
                            "notes": message.get("children", [])[:5],
                            "provenance": "rustc_json",
                        }
                    )
    for d in fallback["diagnostics"]:
        match = re.search(r"\b(error|warning|note)(?:\[([^]]+)\]|\s+(TS\d+))?[: ]", d["message"])
        d.update(
            severity=match[1] if match else "unknown",
            diagnostic_code=(match[2] or match[3]) if match else None,
            provenance="text_location",
        )
    if not tests and name in {"unittest", "cargo", "ctest"}:
        for line in text.splitlines():
            match = re.search(
                r"^(?:test )?(.+?)\s+\.\.\.\s+(ok|FAILED|FAIL|ERROR|ignored|skipped.*)$", line
            )
            if match:
                identifier, status = match.groups()
                tests.append(
                    {
                        "framework": name,
                        "test_id": identifier,
                        "name": identifier,
                        "status": "passed"
                        if status == "ok"
                        else "failed"
                        if status in {"FAILED", "FAIL", "ERROR"}
                        else "skipped",
                        "message": "",
                        "provenance": "framework_text_status",
                    }
                )
    for match in re.finditer(
        r"([\w./-]+\.tsx?)\((\d+),(\d+)\):\s*(error|warning)\s+(TS\d+):\s*([^\n]+)", text
    ):
        diagnostics.append(
            {
                "file": match[1],
                "line": int(match[2]),
                "column": int(match[3]),
                "severity": match[4],
                "diagnostic_code": match[5],
                "message": match[6],
                "provenance": "tsc_text_location",
            }
        )
    for match in re.finditer(
        r"([\w./-]+\.\w+):(\d+)(?::(\d+))?:\s*(error|warning|note):\s*([^\n]+?)(?:\s+\[([\w-]+)\])?$",
        text,
        re.M,
    ):
        diagnostics.append(
            {
                "file": match[1],
                "line": int(match[2]),
                "column": int(match[3]) if match[3] else None,
                "severity": match[4],
                "message": match[5],
                "diagnostic_code": match[6],
                "provenance": "compiler_text_location",
            }
        )
    for match in re.finditer(
        r"([\w./-]+\.cu(?:h)?)\((\d+)\):\s*(error|warning)\s*#?(\d+)?:?\s*([^\n]+)", text
    ):
        diagnostics.append(
            {
                "file": match[1],
                "line": int(match[2]),
                "severity": match[3],
                "diagnostic_code": match[4],
                "message": match[5],
                "provenance": "nvcc_text_location",
            }
        )
    for record in tests:
        excerpt = record.get("message", "") + "\n" + record.get("stack", "")
        match = re.search(r"(?:assert |AssertionError: )([^\n]+?)\s*==\s*([^\n]+)", excerpt)
        if match:
            record["actual_expression"], record["expected_expression"] = (
                match[1][:500],
                match[2][:500],
            )
            record["comparison_provenance"] = "assertion text, not evaluated values"
        record["stack_frames"] = [
            {"file": m[1], "line": int(m[2]), "symbol": m[3]}
            for m in re.finditer(r'File "([^"]+)", line (\d+), in (\w+)', excerpt)
        ][:12]
    failures = [t for t in tests if t["status"] == "failed"] if tests else fallback["failures"]
    return {
        **fallback,
        "framework": name,
        "tests": tests,
        "failures": failures[:30],
        "failure_count": len(failures),
        "diagnostics": (diagnostics + fallback["diagnostics"])[:30],
        "parse_errors": errors,
        "structured_test_count": len(tests),
    }
