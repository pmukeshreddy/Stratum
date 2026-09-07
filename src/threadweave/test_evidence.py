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
            parts = case.get("classname", "").split(".")
            boundary = next((i for i, p in enumerate(parts) if p[:1].isupper()), len(parts))
            filename = case.get("file") or "/".join(parts[:boundary]) + ".py"
            identifier = "::".join([filename, *parts[boundary:], case.get("name", "")])
        records.append(
            {
                "framework": name,
                "test_id": identifier,
                "name": identifier,
                "file": case.get("file"),
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
