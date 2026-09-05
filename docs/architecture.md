# Architecture and recovery

```text
 CLI / future UI
       | local JSON requests, bounded event cursor reads
       v
 Session daemon ----- async scheduler ----- task adapter / verifier
       |                     |
       |               root session <---- persistent queues ----> child sessions
       |                     |                                      |
       |                     +------ model-selected actions --------+
       |                                      |
       |                             provider / tool registry
       |                                      |
       |                           independent Python workers
       |                                      |
       +---------------------- SQLite + artifacts + checkpoints

 L1: selected instructions + conversation blocks + digest + selected state
 L2: worker variables + retained values + live sessions + stable child handles
 L3: append-only events + queues + configs + state versions + artifact files
```

The architecture was developed from the supplied report's conceptual description.
The implementation, protocol, package, and abstractions are original. The source
PDF is retained unchanged. No external reference implementation is incorporated.

## Modules and ownership

| Module | Responsibility |
| --- | --- |
| `models` | Validated configuration, sessions, actions, usage, failures, state edits |
| `storage` | SQLite schema, transactions, immutable events and versions, queues |
| `runtime` | Turn scheduling, recursion, recovery, gates, root-wide accounting |
| `context` | L1 assembly, explicit selected state, loss-preserving compaction |
| `kernel`, `kernel_worker` | Per-session process, tool bridge, snapshots, receipts |
| `artifacts` | Durable values, bounded reads/previews, ancestry-aware access |
| `tools` | Typed registry, permissions, workspace and orchestration primitives |
| `tasks` | Domain-independent adapter contract and workspace implementation |
| `providers` | Provider protocol, streaming chat transport, deterministic providers |
| `daemon`, `cli` | Session-independent client connections and human controls |

One daemon/runtime owns a data directory, enforced by OS file locks. The daemon
uses a private Unix socket, with a short path derived from its data directory.
Clients cannot acquire scheduling ownership by attaching. Individual workers also
hold a lock so a replacement cannot race an orphan's last checkpoint write.

SQLite uses WAL, foreign keys, full synchronization, and a versioned schema.
Database state transitions and their events are committed together. The schema is
declared in `storage.py`, migration version 1 is recorded in `schema_migrations`,
and a database from a newer schema is rejected. Future changes must add explicit
migrations before advancing `PRAGMA user_version`.

## Session and turn model

Root and child sessions have the same representation. Every session records its
ID, parent/root IDs, optional fork ancestry, name/role, task instruction, immutable
configuration reference, workspace, kernel ID, context, selected state, pending
turn cursor, lifecycle, outcome, mode, pause/wake state, timestamps, and errors.
Queues, goals, schedules, artifacts, and usage are normalized associated records.

Lifecycle tracks loading/execution separately from the task outcome:

```text
ADMITTED -> RUNNING -> IDLE -> INACTIVE
                ^       |        |
                +-------+--------+  (INACTIVE first becomes IDLE)

outcome: active | completed | cancelled | limited | failed
```

`RUNNING` covers a model invocation, actions, and verification. `IDLE` means the
session is loaded between turns or awaiting activation. Idle workers are unloaded
after the configured inactivity interval. `INACTIVE` remains fully inspectable
and recoverable; it does not mean the objective was completed or discarded.

Each model turn is:

1. Load the session, apply queued refinements, and atomically receive messages.
2. Prepare the adapter if needed; assemble bounded L1 context and tool schemas.
3. Reserve root resources and invoke the provider with timeout/cancellation.
4. Persist the response and its ordered action list before executing actions.
5. Journal each action before its effects; persist its result and advance the
   durable cursor. Python callbacks enter the same registry and journal.
6. Commit a complete assistant/tool conversation block. Run the configured verifier
   after each turn or completion attempt; expose a bounded result and retain details.
7. Apply completion gates, atomically commit completion/messages/turn advancement,
   or schedule the next model invocation.

No planner or mandatory agent graph appears in this loop. Actions may execute
locally, delegate, message, wait, retrieve evidence, refine state, or finish.
Plain assistant text is retained but is not an implicit completion signal.

## Working memory and context

Python runs in a separate process per loaded session, with its working directory
set to the workspace. Top-level await is supported. The last expression is retained
as `_`; the model gets a bounded representation. Tool results called from Python
are returned as full structured data so code can filter them without inflating L1.

Worker stdout and stderr are captured separately, including ordinary native file
descriptor writes. Logs and full results have artifact IDs. Protocol traffic uses
a separate duplicated output descriptor. A timeout kills and reaps the worker's
process group and leaves the last completed checkpoint available for recovery.

Checkpoint codecs cover scalar JSON values, dictionaries, lists, tuples, sets,
frozensets, bytes, paths, and modules. They do not preserve alias identity or cycles.
An individual encoded value is capped at 16 MiB and the full checkpoint at 64 MiB.
Larger/non-serializable state should be stored as files with explicit recipes.
Unsupported values, failed imports, and failed recipes are reported by name.
No pickle deserialization and no automatic execution-history replay is performed.

Context uses a conservative UTF-8 byte bound plus framing overhead, not an exact
provider tokenizer. Output space is reserved. Tool schemas and foundational
instructions are never silently truncated to fit. An undersized configuration
fails explicitly; increase the context limit or reduce the tool allowlist.

Compaction replaces complete older conversation blocks with an extractive digest
and event references. It is deterministic and offline; it is not a semantic model
summary. Digests may lose detail, so event history remains authoritative. Compaction
records retain their source event IDs, and history/artifact tools retrieve details.
Current task text, supplemental state, and result previews are separately bounded.
No automatic dump of all variables, skills, memories, or history enters L1.

## Concurrency and communication

The scheduler limits total simultaneous turns and simultaneous turns per root.
Children have independent model contexts, workers, and histories, and can spawn
descendants. `agent_spawn` commits the child identity immediately. Parent execution
can continue while descendants run. `agent_wait` returns the scheduling slot instead
of waiting for a child inside a model turn.

Messages are durable, timestamped rows with sender/recipient IDs and source events.
The sender's event and queued message commit together. Delivery, receive event, and
L1 insertion also commit together, preventing lost or duplicate consumption after a
restart. Direct parent/child messages are allowed; sibling communication is optional.
Messages to unrelated trees are denied. Terminal recipients retain messages for a
later explicit resume. A failed child notifies its parent but does not fail the root.

The default completion gate waits for active descendants. This can be disabled for
an independently continuing child. A successful verifier can finish an autonomous
task without an explicit model finish; `require_verifier` prevents an explicit
finish from bypassing a failed/unavailable verifier.

## Adaptive state

Prompt notes, memories, skills, and subagent specifications are typed entries with
immutable versions. Updates, tombstone deletion, and rollback all append versions.
Each version contains author session, source event IDs, triggering refinement event,
timestamp, operation, and intended effect. Optimistic `expected_version` detects
conflicting edits. Evidence must belong to the requesting session's tree.

Entries are session-local or explicitly global. Global writes require a configured
capability. Content selection is independent of storage: listing a skill does not
execute or inject it. `skill_run` explicitly executes its code; `agent_spawn` may
use a subagent specification. Direct agent-requested refinement is implemented;
automatic background refinement model calls are not enabled by this implementation.
No model weights or foundational instruction files are modified.

## Resource semantics

Root accounting sums every descendant, including failed/stopped children. Counters
cover input/output tokens, model calls, tools, Python executions, retries, verifiers,
turns, child admissions, execution time, estimated calls, and optional cost. There
are no per-child allowances that bypass root admission limits.

Before a call, the runtime reserves its conservative input bound and maximum output
tokens under a transaction. Optional cost reservations use explicit configured prices.
Other calls wait if a temporary reservation is the only obstacle. Actual reported
usage replaces the reservation. Unknown usage on failures, cancellation, or crash
is charged conservatively at the reservation, with `estimated_calls` incremented.
Provider over-reporting can exceed a reservation; the next operation enforces the
actual total. The harness cannot guarantee upstream billing equals a token estimate.

`max_turns` is a tree-wide count of admitted model turns, not a per-agent allowance.
Provider retries count as additional model calls/retries; resumed partial actions
do not invoke the model again. Root wall-clock budget starts on first execution
and includes downtime and waiting. Per-session `wall_seconds` sums execution
durations and can overlap across concurrent children. A killed turn is charged a
conservative recovery-time estimate that includes its downtime and is marked in
the recovery event. Root elapsed time is exposed
separately. Cost values are absent in meaning (reported as zero) unless the provider
reports a cost or prices are configured; no prices are guessed.

## Recovery guarantees and boundaries

After restart the manager retains IDs/tree, configuration, history, queues, context,
digests, goals, schedules, versions, usage, and artifact references. Runnable sessions
continue automatically. Paused and terminal sessions remain inspectable and require
explicit resume. Missing extension modules or a broken checkpoint produce structured
environment failures rather than a new session identity.

The pending-turn cursor and action table distinguish:

| Durable state at interruption | Recovery behavior |
| --- | --- |
| Model call started, no response | Charge its reservation as estimated usage; invoke again under remaining limits |
| Response persisted, action not started | Execute that pending action |
| Python finished and checkpoint receipt exists | Recover the recorded result without executing code again |
| Action started without a committed receipt | Return an explicit uncertain-effects error; do not replay it |
| Result persisted but cursor not advanced | Reuse the result and advance the cursor |
| Context committed, verification interrupted | Re-run the adapter's read-only/idempotent verifier |

External side effects cannot be atomically committed with SQLite. This system
does not claim exactly-once execution of arbitrary external operations. It avoids
blind retries and exposes uncertainty so the model/human can inspect actual effects.
Custom tools that need stronger guarantees should use their `action_id` as an
idempotency key at the external service. Reconnection of external services belongs
in explicit recovery recipes or task adapters.

Python workers and command supervisors watch their daemon parent and terminate
their process group if it disappears. Clean shutdown cancels active operations,
reaps workers, and retains runnable state. Trusted code can deliberately escape a
process group; enforce OS isolation externally for untrusted code.

Forking creates a new root and kernel identity while retaining source event ancestry.
It copies the latest recoverable state and local adaptive entries with provenance,
keeps readable access to ancestral artifacts, and starts new accounting. It shares
the workspace and does not snapshot arbitrary external systems or copy descendants.
