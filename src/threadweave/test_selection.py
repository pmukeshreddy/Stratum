"""Rank related tests using attributed heuristics and persisted failure evidence."""

import json
from pathlib import Path

from .repository import is_test


def related(context, files=(), symbols=(), tier="related", limit=20):
    if tier not in {"failing", "related", "module", "full"}:
        raise ValueError("tier must be failing, related, module, or full")
    index = context.runtime.index(context.session_id)
    index.ensure_current()
    symbol_definitions = [m for symbol in symbols for m in index.definition(symbol)["matches"]]
    files = sorted(set(files) | {m["path"] for m in symbol_definitions})
    entries = [
        dict(r)
        for r in context.runtime.store.db.execute(
            "SELECT path,language FROM repository_files WHERE workspace=? AND (path LIKE '%test%' OR path LIKE '%spec%')",
            (str(index.root),),
        )
    ]
    names = {Path(p).stem for p in files} | {s.split(".")[-1] for s in symbols}
    importers = {r["path"] for p in files for r in index.dependents(p, current=False)["matches"]}
    previous = [
        f
        for e in context.runtime.store.events(context.session_id, kind="coding_command", limit=10)
        for f in e["payload"].get("failures", [])
        if f.get("name")
    ]
    ranked = {}
    for path in files:
        ranges = [
            (m["line"], m.get("end_line", m["line"]))
            for m in symbol_definitions
            if m["path"] == path
        ]
        for row in context.runtime.store.db.execute(
            "SELECT test_id,line,source FROM test_coverage WHERE workspace=? AND path=?",
            (str(index.root), path),
        ):
            if ranges and not any(start <= row["line"] <= end for start, end in ranges):
                continue
            entry = ranked.setdefault(
                row["test_id"],
                {
                    "target": row["test_id"],
                    "score": 15,
                    "quality": "runtime coverage",
                    "reasons": [],
                },
            )
            reason = f"Recorded execution of {path}:{row['line']} (source {row['source']})"
            if len(entry["reasons"]) < 5:
                entry["reasons"].append(reason)
    for failure in previous:
        identifier = failure.get("test_id") or failure["name"]
        if identifier in ranked:
            ranked[identifier]["score"] += 5
            ranked[identifier]["reasons"].append("previous failure")
            continue
        ranked[identifier] = {
            "target": identifier,
            "score": 10,
            "reasons": ["previous failure"],
            "framework": failure.get("framework", "unknown"),
        }
    if tier != "failing":
        for row in entries:
            path = row["path"]
            if not is_test(path):
                continue
            if (
                row["language"] == "python"
                and not context.runtime.store.db.execute(
                    "SELECT 1 FROM code_evidence WHERE workspace=? AND path=? AND kind='symbols' AND short_name GLOB 'test*' LIMIT 1",
                    (str(index.root), path),
                ).fetchone()
            ):
                # Fixtures/helpers are dependency evidence, not executable test
                # selectors. Explicit prior failures/coverage remain eligible.
                continue
            reasons = []
            if path in files:
                reasons.append("test itself changed")
            if path in importers:
                reasons.append("resolved import dependency on changed module")
            if any(name in path for name in names):
                reasons.append("file/symbol naming relationship")
            if tier == "module" and any(Path(path).parent == Path(p).parent for p in files):
                reasons.append("same directory/package")
            if tier == "full":
                reasons.append("full suite requested")
            if reasons:
                ranked[path] = {
                    "target": path,
                    "score": len(reasons) * 2,
                    "reasons": reasons,
                    "quality": "heuristic",
                    "language": row["language"],
                }
    result = {
        "tier": tier,
        "selections": sorted(ranked.values(), key=lambda r: -r["score"])[:limit],
        "total": len(ranked),
        "final_verifier_unchanged": True,
        "note": "Selection advice; use framework-specific selectors. Full verification remains independent.",
    }
    context.runtime.store.event(
        context.session_id, "test_selection", result, parent=context.source_event
    )
    return result


def import_coverage(context, path):
    """Import coverage.py JSON with per-test dynamic contexts, without executing code."""
    source = context.path(path)
    if source.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Coverage input exceeds 32 MiB; export a bounded package report")
    data = json.loads(source.read_text())
    records = []
    for filename, detail in data.get("files", {}).items():
        try:
            relative = (
                context.path(filename).relative_to(Path(context.session.workspace.path)).as_posix()
            )
        except (PermissionError, ValueError):
            continue
        for line, labels in detail.get("contexts", {}).items():
            for label in labels:
                test_id = label.split("|", 1)[0]
                if not test_id or len(test_id) > 1000:
                    continue
                records.append((relative, int(line), test_id))
    if not records:
        return {
            "imported": 0,
            "skipped": "No per-test dynamic contexts; aggregate coverage cannot identify related tests",
        }
    artifact = context.runtime.artifacts.put(
        context.session_id, data, source_event=context.source_event
    )
    store = context.runtime.store
    with store.transaction():
        for filename in {r[0] for r in records}:
            store.db.execute(
                "DELETE FROM test_coverage WHERE workspace=? AND path=?",
                (context.session.workspace.path, filename),
            )
        store.db.executemany(
            "INSERT OR REPLACE INTO test_coverage VALUES(?,?,?,?,?)",
            [
                (context.session.workspace.path, test, file, line, artifact)
                for file, line, test in records
            ],
        )
        store.event(
            context.session_id,
            "coverage_import",
            {"records": len(records), "artifact": artifact},
            parent=context.source_event,
        )
    return {"imported": len(records), "source_artifact": artifact}


def selection_reason(context, target):
    for event in context.runtime.store.events(context.session_id, kind="test_selection", limit=20):
        for selection in event["payload"].get("selections", []):
            if selection["target"] == target:
                return {**selection, "event_id": event["id"]}
    return {"target": target, "reason": "No retained selection for this target"}
