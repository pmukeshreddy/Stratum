"""Forward-only product schema migrations; the v1 trajectory is never rewritten."""

VERSION = 8

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
CREATE TABLE eval_runs(id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id),
 created_at REAL NOT NULL, body TEXT NOT NULL);
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
