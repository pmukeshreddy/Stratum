"""One-time import of retired SQLite harness tables. Never used for runtime reads."""

import json
import logging
from datetime import UTC, datetime

from .harness import atomic_json, save_harness_state, validate_edit

log = logging.getLogger(__name__)


def migrate_sqlite_harness(db, harness):
    marker = harness.directory / "harness" / ".sqlite-migrated.json"
    if marker.exists():
        return
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    migrated, skipped, stores = [], [], {}
    if {"state_entries", "state_versions"} <= tables:
        rows = db.execute(
            "SELECT e.*,v.body FROM state_entries e JOIN state_versions v ON v.entry_id=e.id AND v.version=e.current_version WHERE e.deleted=0"
        ).fetchall()
        for row in rows:
            try:
                body = json.loads(row["body"])
                content = body.get("content", {})
                kind = {"prompt_note": "prompt", "subagent_spec": "subagent"}.get(
                    row["kind"], row["kind"]
                )
                owner = row["owner_id"]
                scope = "local" if owner else "global"
                state = stores.setdefault(owner, harness.load(owner))
                created = body.get("created_at", 0)
                if isinstance(created, (float, int)):
                    created = datetime.fromtimestamp(created, UTC).isoformat()
                entry = {
                    "id": row["id"],
                    "kind": kind,
                    "title": body.get("title", row["id"]),
                    "content": content
                    if isinstance(content, str)
                    else content.get(
                        "text", content.get("instruction", content.get("description", ""))
                    ),
                    "path": content.get("path", "general")
                    if isinstance(content, dict)
                    else "general",
                    "scope": scope,
                    "reference": content.get("reference", {}) if isinstance(content, dict) else {},
                    "arguments": content.get("arguments", {}) if isinstance(content, dict) else {},
                    "metadata": {"migration": "sqlite", "provenance": body.get("provenance", {})},
                    "source": "migration",
                    "version": row["current_version"],
                    "created_at": created,
                    "updated_at": created,
                }
                error = validate_edit({"action": "create", **entry}, entry["id"])
                if error:
                    # Arbitrary executable bodies cannot become callable references safely.
                    skipped.append({"id": row["id"], "reason": error})
                    continue
                state["entries"][kind].setdefault(entry["id"], entry)
                migrated.append(entry["id"])
            except (ValueError, TypeError, KeyError) as exc:
                skipped.append({"id": row["id"], "reason": str(exc)})
    for owner, state in stores.items():
        save_harness_state(harness.path(owner), state)
    # Remove retired policy/selection fields once so persisted sessions resume under
    # the new schema without a runtime compatibility adapter.
    for row in db.execute("SELECT id,body FROM configs").fetchall():
        config = json.loads(row["body"])
        policy = config.get("refinement", {})
        config["refinement"] = {
            k: v
            for k, v in policy.items()
            if k in {"enabled", "turn_interval", "compact", "cooldown_seconds"}
        }
        if policy.get("automatic") is False:
            config["refinement"]["enabled"] = False
        features = config.get("features", {})
        if features.pop("automatic_refinement", True) is False:
            config["refinement"]["enabled"] = False
        db.execute("UPDATE configs SET body=? WHERE id=?", (json.dumps(config), row["id"]))
    for row in db.execute("SELECT id,body FROM sessions").fetchall():
        session = json.loads(row["body"])
        session.pop("selected_state", None)
        db.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(session), row["id"]))
    atomic_json(marker, {"migrated": migrated, "skipped": skipped})
    if skipped:
        log.warning(
            "Harness migration skipped %d untranslatable entries; details: %s", len(skipped), marker
        )
