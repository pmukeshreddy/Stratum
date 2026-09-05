# Architecture

```text
CLI / future UI -> private Unix socket -> daemon + async scheduler
                                           |
                       persistent root / isolated recursive children
                                           |
                        context -> routed model -> chosen actions
                                           |
               repository index / editor / Git / executor / experiments
                                           |
                         evidence <- independent coding verifier
                                           |
                             per-session Python worker
                                           |
               SQLite WAL + private artifacts + worker checkpoints

L1: objective, selected/recent evidence, bounded summary, selected durable state
L2: Python values, retained tool outputs, active sessions and stable handles
L3: immutable history, artifacts, versions, indexes, experiments and relationships
```

The existing scheduler, worker protocol, session/action journal, messaging and
root-wide accounting remain the foundation. Coding adds an adapter and primitives,
not a mandatory planning graph.

## Sessions and turns

Stable IDs preserve roots, parents, children and branch ancestry. Lifecycle
ADMITTED -> RUNNING -> IDLE -> INACTIVE describes loading independently of outcome.
One runtime owns a data directory under an OS lock. Clients own no sessions.

Each turn applies queued state edits, receives messages, prepares the task once,
optionally refines/compacts selected evidence, assembles context and routes a model
call. Resources are reserved before invocation. The response and ordered actions
are persisted before execution; action receipts/cursors prevent blind replay.
Completion requests go through independent task verification before acceptance.

Auxiliary compaction/refinement uses the same retry, reservation and accounting path
without replacing the main pending turn. Routing is fixed or explicitly role-based;
there is no learned routing claim.

The scheduler enforces global/per-root concurrency, depth/subagent limits,
cancellation and tree-wide resource budgets. Children have independent histories,
contexts and workers. Failed children notify parents without deciding root outcome.

## Coding components

repository.py stats files and reparses changed entries with Python AST or lexical
multi-language extraction. It excludes common generated/vendor directories, binary
files and files over 2 MB. It is not a language server or an OS file watcher.

editing.py validates all hunks, journals exact prior contents and hashes, then writes
files atomically. On application failure it restores changes. On restart a prepared
edit is rolled back or paused if external edits conflict. Visibility is serialized
within this runtime, not a filesystem-wide transaction against unrelated writers.
Direct Python/process/build/test actions receive before/after file checkpoints and
workspace-effect events. Recovery observes interrupted effects rather than replaying
commands. Identical artifact contents are reused within a session.

gitops.py checkpoints tracked/nonignored files without changing the user's index or
commits. Children and coding forks get private Git copies with captured inputs as
private baseline commits. Copies include uncommitted inputs. Checkpoint files over
64 MiB fail explicitly. Parents select candidate patches and apply them through the
same editor. Symlink escapes are rejected.

coding.py captures the baseline once and reruns configured commands for completion.
An explicit option can tolerate identified unchanged baseline failing tests; new
failures remain regressions. protect_tests can prohibit all test edits beyond default
deletion checks. Verifiers are trusted configuration, not model completion claims.

execution.py owns command process groups, capture, timeout and cleanup. Full outputs
stay in artifacts. Local execution has host authority. The container executor
restricts the command environment, mount, resources and network; it does not
sandbox the host Python worker.

## Information and adaptive state

FTS5 indexes events (including messages, result summaries, refinements and experiment
conclusions) plus the first 16 KB of text artifacts. Retrieval is scoped to a tree
and explicit branch ancestry. Complete artifacts remain readable by ID. Context
assembly selects bounded current evidence, never the entire index/history.

Model compaction records structured facts and its response/source IDs. Extractive
fallback records failure provenance. Summaries may omit details; original events
remain authoritative.

State entries are prompt notes, memories, executable skills or subagent specs.
Updates/deletion/rollback append immutable versions. Automatic proposals must cite
selected evidence and an intended effect. Skill schemas/syntax/permissions are
validated; per-version outcomes support quarantine. Failed verifiers create
searchable failure memories. Foundational policy and model weights are immutable.

Experiments and experiment runs are separate durable entities. Runs retain source
checkpoints, patches, correctness evidence, measurements and conclusion provenance.
Measurements do not establish causality or statistical significance.

## Persistence and recovery

Schema v2 migrates v1 forward without rewriting historical sessions/configs/events.
SQLite uses WAL, foreign keys and full synchronization. Events and state versions
have immutable-table triggers.

Restart retains identities, tree, queues, contexts, goals, versions, artifacts,
indexes, checkpoints, experiments and usage. Workers are recreated using explicit
codecs and opted-in recipes. Unsupported objects are reported; action history is
never replayed as reconstruction.
Coding forks rebind explicit Path codecs into the new workspace and do not copy
potentially source-bound reconstruction recipes. Missing recipes are reported for
review. Arbitrary strings are not assumed to be filesystem handles.

A persisted response resumes its cursor. Finished Python actions can recover
checkpoint receipts. Actions started without a durable receipt become uncertain and
are not automatically repeated. Interrupted model reservations are conservatively
charged. Interrupted experiments remain explicit interrupted entities.

No distributed scheduling, arbitrary object serialization, automatic dependency
installation, disk quotas, external-system snapshots or exactly-once external
effects are claimed.
