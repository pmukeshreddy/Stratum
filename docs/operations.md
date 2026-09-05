# Configuration and operations

## Reproducible runs

`RunConfig` is a strict, versioned Pydantic model. Unknown fields, nonpositive
limits, impossible context reservations, and incomplete cost-budget configuration
are rejected. Configurations are serialized canonically, hashed, and stored once;
every session references its exact configuration. Model credentials are referenced
by environment-variable name and are not copied into the configuration.

Configuration groups:

| Group | Main fields |
| --- | --- |
| `provider` | name, model, base_url, api_key_env, parameters, streaming, timeout_seconds, max_output_tokens, optional input/output prices |
| `task` | adapter, specification, verifier, verifier_options, verify_each_turn, require_verifier, wait_for_children, success_metrics |
| `context` | max_tokens, compact_at, recent_blocks, summary_chars, result_chars, supplemental_chars |
| `refinement` | enabled, allow_global_writes, selected_entries |
| `retry` | attempts, initial_delay, max_delay |
| `limits` | max_turns, token_budget, wall_seconds, cost_budget, max_tool_calls, max_python_executions, max_model_calls, max_subagents, max_depth, concurrency, tool_timeout_seconds, python_timeout_seconds |
| top level | schema_version, permissions, tool_allowlist, allow_sibling_messages, extensions |

Omitted fields use explicit defaults in `models.py`. Run
`threadweave config SESSION_ID` to export a fully resolved configuration. Pin your
model ID, task revision, extension source version, and dependencies (`uv.lock`) for
reproducibility; the runtime cannot make a remote stochastic model deterministic.

Tree budgets are cumulative across resumes. A new fork creates a new root budget
and explicitly records ancestry, so analyses can include branch costs separately.
The daemon's `--concurrency` caps simultaneous turns across all roots; each run's
`limits.concurrency` adds a per-root cap. Both count turns, not sleeping agents.

The default capabilities are `workspace.read`, `workspace.write`, `python`, `agents`,
and `state`. `process` is opt-in. A `tool_allowlist` further narrows the registry.
Tool metadata is checked for both native model calls and Python callbacks. A
subagent inherits permissions and cannot request broader ones through `agent_spawn`.

## Local control protocol

The CLI communicates over a mode-0600 Unix socket. Each connection sends one JSON
line and receives one JSON line. Requests are bounded to 8 MiB; history/status
previews are bounded. The local same-user socket is the authentication boundary.
There is no unauthenticated network listener.

```json
{"method":"status","arguments":{"session_id":"STABLE_ID"}}
```

Responses are either `{"result": ...}` or `{"error": "type: message"}`. Available
methods include `ping`, `create`, `list`, `tree`, `status`, `config`, `history`,
`input`, `pause`, `resume`, `stop`, `fork`, `schedule`, `schedules`, `unschedule`,
`states`, `refine`, `compact`, `artifact`, and `shutdown`.

The Python `threadweave.daemon.request` client can be used by another UI. A richer
frontend can poll durable event sequence cursors without assuming an uninterrupted
socket connection. History's initial query returns a bounded tail; subsequent
positive `after` sequence cursors page forward. `history_read(event_id=...)` and
the artifact tools provide complete referenced details to agents. SQL remains an
inspectable source for complete trajectory exports.

For direct embedding, construct `Runtime`, register integrations, create a session,
then await `start()` and `wait(session_id)`. Always await `shutdown()` in `finally`.
Use `Runtime` with one event loop and one owner per data directory. Use the daemon
for detachment: shutting down an embedded Python interpreter also shuts down its
runtime ownership.

## State-edit example

```json
{
  "kind": "memory",
  "scope": "session",
  "title": "Measured constraint",
  "content": {"text": "The target environment requires Python 3.12."},
  "source_events": ["REAL_EVENT_ID_FROM_THIS_TREE"],
  "intended_effect": "Use the measured interpreter version in future commands.",
  "select": true
}
```

For update, include `entry_id` and optionally `expected_version`. For delete, set
`operation="delete"`. For rollback, set `operation="rollback"` and
`rollback_version=N`; rollback creates a new current version with the previous
content. The source-event and intended-effect requirements also apply to deletion
and rollback. The CLI queues edits; they are applied at the next execution boundary.
Resume an idle/paused session to make pending edits active.

Typed content fields are `text` for memories/prompt notes, `code` for skills, and
`instruction` for subagent specifications. Selection IDs are available from
`state_list`/`states`. `state_select` explicitly chooses the L1 supplement.

## On-disk layout and backup

```text
.threadweave/
  history.sqlite3          sessions, events, messages, versions, configs, usage, goals
  history.sqlite3-wal      SQLite WAL while open
  history.sqlite3-shm      SQLite shared-memory index while open
  artifacts/<id>          immutable result values and logs
  kernels/<kernel_id>/
    checkpoint.json       supported values, reconstruction recipes, last receipt
    worker.lock           prevents concurrent writers to one checkpoint
    <action_id>.stdout    raw execution output
    <action_id>.stderr    raw execution errors
    recovery.log          recipe output
    worker.log            worker diagnostics
  daemon.log              structured runtime events and diagnostic failures
  daemon.json             last daemon PID and socket reference (informational)
  daemon.lock             daemon ownership
  runtime.lock            embedded/runtime ownership
```

For a consistent simple backup, stop the daemon and copy the entire data directory
plus the associated workspace. For online database backups, use SQLite's backup
API and separately coordinate artifact/workspace snapshots. Copying only the main
SQLite file while WAL writes are active is not a valid full backup. Restoring to a
different machine requires updating/restoring the actual workspace location and
installing the configured extension modules; paths are intentionally stable.

Events are append-only and state versions immutable under database triggers.
Artifacts include size and SHA-256 metadata. Full artifact loads verify checksums;
bounded range reads do not rehash the entire file. Large logs go to disk, so plan
disk capacity for long runs. Automatic retention pruning, encryption, remote
replication, and disk/memory quotas are not implemented. Keep the data directory
private and avoid storing secrets in model-visible task material.

## Debugging and operational limits

Inspect `status`, `tree`, `usage`, and bounded `history` first. Events link model
invocations, tool calls, Python runs, messages, retries, verifiers, refinement,
compaction, and completion using event IDs and causal references. Model request
artifacts capture the exact L1 request and tool schemas. JSON log lines contain
session/root IDs, event IDs, timestamps, event type, and causal references; detailed
payloads live in SQLite to keep logs bounded.

Failure categories are `model`, `provider`, `tool`, `verifier`, `environment`, and
`runtime`. Transient provider failures and verifier infrastructure failures follow
the configured retry policy. Tool/Python failures are returned as evidence so the
model can choose its next action. Infrastructure failures are not task verifier
failures. A context-capacity error usually means tool schemas plus instructions
cannot fit the configured conservative bound.

This is a local, single-host runtime. It does not reconnect to an old Python process
after application death; it recreates the worker from supported values and recipes.
It does not automatically serialize arbitrary objects, roll back external side
effects, or replay uncertain actions. Forks share workspaces, so concurrent writers
must coordinate through messages or use explicit isolated workspaces at admission.
Custom tools/adapters are responsible for their own external idempotency and resource
cleanup. Deploy OS isolation and service supervision as required by your environment.
