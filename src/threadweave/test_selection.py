"""Rank related tests using attributed heuristics and persisted failure evidence."""

from pathlib import Path

from .repository import is_test


def related(context, files=(), symbols=(), tier="related", limit=20):
    if tier not in {"failing", "related", "module", "full"}:
        raise ValueError("tier must be failing, related, module, or full")
    index = context.runtime.index(context.session_id)
    index.ensure_current()
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
    for failure in previous:
        identifier = failure.get("test_id") or failure["name"]
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
            reasons = []
            if path in files:
                reasons.append("test itself changed")
            if path in importers:
                reasons.append("syntax import overlaps changed module")
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
