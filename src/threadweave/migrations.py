"""Forward-only product schema migrations; the v1 trajectory is never rewritten."""

VERSION = 12

LEGACY_HARNESS_SCHEMA = "CREATE TABLE IF NOT EXISTS state_entries(\n id TEXT PRIMARY KEY, owner_id TEXT REFERENCES sessions(id), kind TEXT NOT NULL,\n current_version INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0\n);\nCREATE TABLE IF NOT EXISTS state_versions(\n entry_id TEXT NOT NULL REFERENCES state_entries(id), version INTEGER NOT NULL,\n body TEXT NOT NULL, PRIMARY KEY(entry_id, version)\n);\nCREATE TRIGGER IF NOT EXISTS versions_immutable_update BEFORE UPDATE ON state_versions\nBEGIN SELECT RAISE(ABORT, 'state versions are immutable'); END;\nCREATE TRIGGER IF NOT EXISTS versions_immutable_delete BEFORE DELETE ON state_versions\nBEGIN SELECT RAISE(ABORT, 'state versions are immutable'); END;\nCREATE TABLE IF NOT EXISTS refinements(\n id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),\n edit TEXT NOT NULL, status TEXT NOT NULL, source_event TEXT NOT NULL, error TEXT\n);\n"

CODING_SCHEMA = """
CREATE TABLE repository_files(
 workspace TEXT NOT NULL, path TEXT NOT NULL, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL,
 sha256 TEXT NOT NULL, language TEXT NOT NULL, body TEXT NOT NULL,
 PRIMARY KEY(workspace,path));
CREATE TABLE coding_baselines(session_id TEXT PRIMARY KEY REFERENCES sessions(id), body TEXT NOT NULL);
CREATE TABLE checkpoints(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, label TEXT NOT NULL, manifest TEXT NOT NULL, head TEXT NOT NULL);
CREATE TABLE edits(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE experiments(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE experiment_runs(id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE benchmark_measurements(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE final_verifications(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, passed INTEGER NOT NULL, body TEXT NOT NULL);
CREATE TABLE routing_decisions(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE failure_memories(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE skill_outcomes(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
 entry_id TEXT NOT NULL, version INTEGER NOT NULL, passed INTEGER NOT NULL, body TEXT NOT NULL);
CREATE TABLE candidates(child_id TEXT PRIMARY KEY REFERENCES sessions(id), parent_id TEXT NOT NULL,
 checkpoint_id TEXT NOT NULL, body TEXT NOT NULL);
CREATE VIRTUAL TABLE history_fts USING fts5(id UNINDEXED, session_id UNINDEXED, root_id UNINDEXED,
 kind UNINDEXED, text, tokenize='unicode61');
INSERT INTO history_fts(id,session_id,root_id,kind,text)
 SELECT id,session_id,root_id,type,payload FROM events;
CREATE TRIGGER events_search AFTER INSERT ON events BEGIN
 INSERT INTO history_fts(id,session_id,root_id,kind,text)
 VALUES(new.id,new.session_id,new.root_id,new.type,new.payload);
END;
"""


def migrate(db, previous, timestamp):
    if previous < 2:
        # executescript commits first; explicit transaction keeps schema+version atomic.
        db.executescript(
            "BEGIN IMMEDIATE;"
            + CODING_SCHEMA
            + f"INSERT INTO schema_migrations VALUES(2,{timestamp}); PRAGMA user_version=2; COMMIT;"
        )
    if previous < 3:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE process_jobs(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), body TEXT NOT NULL);"
            f"INSERT INTO schema_migrations VALUES(3,{timestamp}); PRAGMA user_version=3; COMMIT;"
        )
    if previous < 4:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE goal_budgets(session_id TEXT PRIMARY KEY REFERENCES goals(session_id), "
            "token_budget INTEGER NOT NULL, starting_tokens INTEGER NOT NULL);"
            f"INSERT INTO schema_migrations VALUES(4,{timestamp}); PRAGMA user_version=4; COMMIT;"
        )
    if previous < 5:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE refinement_requests(id TEXT PRIMARY KEY, "
            "session_id TEXT NOT NULL REFERENCES sessions(id), trigger_event TEXT NOT NULL REFERENCES events(id), "
            "status TEXT NOT NULL, result TEXT NOT NULL);"
            "CREATE INDEX refinement_requests_pending ON refinement_requests(session_id,status);"
            f"INSERT INTO schema_migrations VALUES(5,{timestamp}); PRAGMA user_version=5; COMMIT;"
        )
    if previous < 6:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE mutation_workspaces(path TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "manifest_artifact TEXT NOT NULL, state_id TEXT NOT NULL);"
            "CREATE TABLE mutation_windows(id TEXT PRIMARY KEY, workspace TEXT NOT NULL, "
            "session_id TEXT NOT NULL, action_id TEXT NOT NULL, source_event TEXT NOT NULL, "
            "before_artifact TEXT NOT NULL, before_owner TEXT NOT NULL, status TEXT NOT NULL);"
            f"INSERT INTO schema_migrations VALUES(6,{timestamp}); PRAGMA user_version=6; COMMIT;"
        )
    if previous < 7:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE code_evidence(workspace TEXT NOT NULL,path TEXT NOT NULL,kind TEXT NOT NULL,"
            "name TEXT NOT NULL,enclosing TEXT,body TEXT NOT NULL);"
            "CREATE INDEX code_lookup ON code_evidence(workspace,kind,name);"
            "CREATE INDEX code_file ON code_evidence(workspace,path);"
            "CREATE INDEX code_owner ON code_evidence(workspace,kind,enclosing);"
            f"INSERT INTO schema_migrations VALUES(7,{timestamp}); PRAGMA user_version=7; COMMIT;"
        )
    if previous < 8:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "ALTER TABLE code_evidence ADD COLUMN short_name TEXT;"
            "UPDATE code_evidence SET short_name=CASE WHEN json_valid(body) THEN COALESCE(json_extract(body,'$.short_name'),name) ELSE name END;"
            "CREATE INDEX code_short_lookup ON code_evidence(workspace,kind,short_name,path);"
            "CREATE INDEX code_exact_lookup ON code_evidence(workspace,kind,name,path);"
            f"INSERT INTO schema_migrations VALUES(8,{timestamp}); PRAGMA user_version=8; COMMIT;"
        )
    if previous < 9:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE module_bindings(workspace TEXT,path TEXT,module TEXT,target TEXT,alias TEXT,symbol TEXT,quality TEXT, PRIMARY KEY(workspace,path,module,alias,symbol));"
            "CREATE INDEX module_target ON module_bindings(workspace,target,path);"
            "CREATE INDEX module_source ON module_bindings(workspace,path);"
            "CREATE TABLE test_coverage(workspace TEXT,test_id TEXT,path TEXT,line INTEGER,source TEXT,PRIMARY KEY(workspace,test_id,path,line));"
            "CREATE INDEX coverage_file ON test_coverage(workspace,path,line);"
            f"INSERT INTO schema_migrations VALUES(9,{timestamp}); PRAGMA user_version=9; COMMIT;"
        )
    if previous < 10:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE semantic_evidence(id TEXT,model TEXT,root_id TEXT,session_id TEXT,kind TEXT,seq INTEGER,excerpt TEXT,vector BLOB, PRIMARY KEY(id,model));"
            "CREATE INDEX semantic_root ON semantic_evidence(model,root_id,seq);"
            "CREATE TABLE semantic_cursors(root_id TEXT,model TEXT,seq INTEGER,PRIMARY KEY(root_id,model));"
            f"INSERT INTO schema_migrations VALUES(10,{timestamp}); PRAGMA user_version=10; COMMIT;"
        )

    if previous < 11:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "CREATE TABLE model_requests(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), "
            "purpose TEXT NOT NULL, body_hash TEXT NOT NULL, body_artifact TEXT NOT NULL, provider TEXT NOT NULL, "
            "model TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, response_event TEXT, inbound TEXT NOT NULL);"
            "CREATE INDEX model_request_session ON model_requests(session_id,started_at);"
            "CREATE TABLE model_attempts(event_id TEXT PRIMARY KEY REFERENCES events(id), request_id TEXT NOT NULL REFERENCES model_requests(id), "
            "attempt INTEGER NOT NULL, status TEXT NOT NULL, usage TEXT NOT NULL, failure TEXT, ended_at REAL);"
            "CREATE TABLE request_edges(source TEXT NOT NULL REFERENCES model_requests(id), target TEXT NOT NULL REFERENCES model_requests(id), "
            "kind TEXT NOT NULL, PRIMARY KEY(source,target,kind));"
            "CREATE TABLE pending_request_edges(session_id TEXT NOT NULL REFERENCES sessions(id), source TEXT NOT NULL REFERENCES model_requests(id), "
            "kind TEXT NOT NULL, PRIMARY KEY(session_id,source,kind));"
            "CREATE TABLE conversation_blocks(session_id TEXT NOT NULL REFERENCES sessions(id), event_id TEXT NOT NULL REFERENCES events(id), "
            "messages TEXT NOT NULL, PRIMARY KEY(session_id,event_id));"
            "INSERT OR IGNORE INTO conversation_blocks SELECT s.id,json_extract(b.value,'$.event_id'),json_extract(b.value,'$.messages') "
            "FROM sessions s,json_each(s.body,'$.context') b JOIN events e ON e.id=json_extract(b.value,'$.event_id');"
            "ALTER TABLE messages ADD COLUMN delivery TEXT NOT NULL DEFAULT 'boundary';"
            "ALTER TABLE messages ADD COLUMN causal_request_id TEXT;"
            "ALTER TABLE refinements ADD COLUMN baseline TEXT;"
            "CREATE TABLE refinement_runs(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), "
            "status TEXT NOT NULL, body TEXT NOT NULL);"
            "CREATE INDEX refinement_run_session ON refinement_runs(session_id,status);"
            f"INSERT INTO schema_migrations VALUES(11,{timestamp}); PRAGMA user_version=11; COMMIT;"
        )
    if previous < 12:
        db.executescript(
            "BEGIN IMMEDIATE;"
            "ALTER TABLE model_requests ADD COLUMN request_kind TEXT NOT NULL DEFAULT 'trajectory';"
            "UPDATE model_requests SET request_kind='auxiliary' WHERE purpose NOT IN ('agent','compaction');"
            "ALTER TABLE refinement_requests ADD COLUMN trigger TEXT NOT NULL DEFAULT 'manual';"
            "UPDATE refinement_requests SET trigger=COALESCE((SELECT json_extract(e.payload,'$.trigger') "
            "FROM refinement_runs r JOIN events e ON e.id=json_extract(r.body,'$.marker') "
            "WHERE r.id=refinement_requests.id),'manual');"
        )
        try:
            _separate_auxiliary_edges(db)
            db.execute("INSERT INTO schema_migrations VALUES(12,?)", (timestamp,))
            db.execute("PRAGMA user_version=12")
            db.commit()
        except BaseException:
            db.rollback()
            raise


def _separate_auxiliary_edges(db):
    """Repair v11 continuation chains without discarding call/usage history."""
    import json

    requests = {
        row[0]: (row[1], json.loads(row[2]))
        for row in db.execute("SELECT id,request_kind,inbound FROM model_requests")
    }

    def sources(source, seen=frozenset()):
        if source in seen or source not in requests:
            return []
        kind, inbound = requests[source]
        if kind == "trajectory":
            return [source]
        return [
            ancestor for edge in inbound for ancestor in sources(edge["source"], seen | {source})
        ]

    edges = db.execute("SELECT source,target,kind FROM request_edges").fetchall()
    for source, target, kind in edges:
        if requests[source][0] == "auxiliary" or requests[target][0] == "auxiliary":
            db.execute(
                "DELETE FROM request_edges WHERE source=? AND target=? AND kind=?",
                (source, target, kind),
            )
            if requests[target][0] == "trajectory":
                for ancestor in sources(source):
                    # A committed compaction may already supply a more specific edge.
                    if not db.execute(
                        "SELECT 1 FROM request_edges WHERE source=? AND target=?",
                        (ancestor, target),
                    ).fetchone():
                        db.execute(
                            "INSERT OR IGNORE INTO request_edges VALUES(?,?,?)",
                            (ancestor, target, kind),
                        )
    for rid, (kind, inbound) in requests.items():
        repaired = (
            []
            if kind == "auxiliary"
            else [
                {"source": ancestor, "kind": edge["kind"]}
                for edge in inbound
                for ancestor in sources(edge["source"])
            ]
        )
        db.execute("UPDATE model_requests SET inbound=? WHERE id=?", (json.dumps(repaired), rid))
    for sid, source, kind in db.execute("SELECT * FROM pending_request_edges").fetchall():
        if requests[source][0] == "auxiliary":
            db.execute(
                "DELETE FROM pending_request_edges WHERE session_id=? AND source=? AND kind=?",
                (sid, source, kind),
            )
            for ancestor in sources(source):
                db.execute(
                    "INSERT OR IGNORE INTO pending_request_edges VALUES(?,?,?)",
                    (sid, ancestor, kind),
                )
