"""Forward-only product schema migrations; the v1 trajectory is never rewritten."""

VERSION = 2

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
