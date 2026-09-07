# Python control plane and Environment observation

## Execution and policy

The default provider schema is still only `ipython(code)`. The provider response
enters `Runtime._execute_action`, which journals the request, resolves and checks
the registered capability, calls `Environment.before_action`, executes it, calls
`Environment.after_action`, and journals its bounded result plus an artifact.
The persistent worker sends both synchronous `tools.call` and asynchronous
`tools.acall` requests back through that same executor. `repo.search` and
`tests.run` are synchronous Python convenience methods; `edit`, `rlm` and MCP
use asynchronous host RPC. Their origin does not exempt them from Environment.

Previously, `context.from_python` bypassed external-effect observation. The
watched-action list also omitted `ipython`. Finally, the generic `host_request`
envelope hid the effective capability's permission/policy identity from hooks.

The registry now resolves a host operation (such as `edit`, `rlm.run`, or
`mcp.call`) before admission. Standard permission, read-only, feature and explicit
tool-allowlist checks apply, followed by operation-specific ownership, path,
command and server checks. With an explicit allowlist, include both `host_request`
and the intended operation names; the envelope alone does not authorize all
operations. Catalog discovery is permitted by the envelope. These internal
capabilities are **not** added to the model schema.

`environment_action_started` and `environment_action_finished` complement the
existing `tool_call`, `tool_result` and failure events. A failed before hook prevents
execution. A failed after hook preserves execution evidence, returns an uncertain
Environment error and pauses the session instead of returning a successful action.
Worker reentrancy checks remain: RPC cannot recursively enter its occupied kernel.
No observation lock is held across a Python/RPC await. Nested RPCs share their
session's cell observation interval while still executing their own policy hooks.

## Coding workspace observation

`MutationObserver` enumerates Git's tracked and other paths, including ignored
outputs, using NUL-separated names. It excludes `.git` and private runtime storage
(but not an isolated candidate's own workspace). It does not stage files, refresh
the user's index, invoke diff drivers, or change commits.

Each file has a cached device/inode/type/mode/size/mtime/ctime signature and content
SHA-256. Initial discovery hashes content once. A native `watchfiles` tracker now
collects changed-path candidates. A unique marker fence in the watched stream
synchronizes ordinary cell boundaries; an empty warm candidate set reuses the
cached manifest without enumerating the repository. Candidate scans stat files and
stream-hash only new/signature-changed regular files. Symlinks are hashed as link
text without following targets; special files are not opened. There is no per-cell
copy or rehash of all source contents. Changed-state manifest construction remains
O(number of paths); warm no-change cells do not perform it. Startup, recovery,
watcher uncertainty/loss/overflow, verification, explicit reconciliation and a
60-second reconciliation interval still perform full metadata enumeration. A
failed/unavailable watcher falls back to these full scans, not silent omission.

Immutable metadata artifacts retain manifests; schema migration 6 adds workspace
cache pointers and durable open observation windows. A no-change scan reuses the
manifest. Cache updates, observations and window completion commit together.
`workspace_effects` identifies the cell/action, session, before/after content-and-
mode state identities and HEAD, additions, modifications, deletions and exact-content
rename candidates. Event rows contain at most 100 paths per collection; the full
metadata delta is retained as an artifact. They never contain entire file contents.

Observed mutations are explicitly `externally_observed`, `transactional=false`,
`rollback_performed=false`. The existing editor's `code_edit` journal, pre/post
hashes, optimistic conflicts, atomic file writes and rollback/recovery are unchanged.
Explicit legacy direct mode retains its existing external-action checkpoint policy;
the default Python-first path uses incremental observation, not implicit full-file
snapshots on every cell. Initial coding baselines and explicit Git checkpoints
retain their content limits. Candidate admission uses Git-tree checkpoints and
worktrees without copying Git objects or changing the user's branch/index.

Reconciliation occurs before/after Python execution, before process actions,
before/after coding verification, at daemon-managed background process exit, on
recovery, and approximately once a second for initialized coding workspaces not
currently inside a cell. That last path detects delayed unrestricted subprocess
writes while the session is idle. It does not claim that a particular process
caused a change. Shared root/child windows may observe overlapping changes.

Existing bytecode invalidation and affected-file index refresh are reused. A
Python write followed by a test command in the same cell invalidates stale bytecode
before the command. The coding verifier reads the actual tree regardless of editing
API and records the observed final state. Ignored generated files are audited;
explicit forbidden-path rules apply to them. Existing allowed-path checks continue
to govern tracked/nonignored files, so verifier-generated ignored caches are not
misclassified as agent source edits.

## Failure and security boundaries

- Python is trusted-host execution, not a sandbox. `Path`, `open`, `os`, `shutil`
  and `subprocess` remain usable. Tool permissions do not prevent raw Python from
  writing forbidden paths or `.git`; the observer does not enforce OS isolation.
- Exceptions, cancellation and kernel exit still close observation windows and
  retain partial-write evidence. A daemon crash leaves an open window, reconciled
  on recovery without replaying the cell. Observation failure pauses execution,
  retains the open window, and never claims rollback. After correcting the cause,
  daemon recovery retries that observation.
- This is net-state observation, not a syscall trace. A file created and removed,
  or modified and restored entirely between observation boundaries, leaves no net
  mutation record. Concurrent writers are attributed to an interval, not exclusively
  to its initiating cell. Continually changing/unreadable files can fail observation.
- Observation is eventual, not instantaneous, and reconciliation runs synchronously
  in the daemon. Full scans and changed manifests have repository-size costs. Background processes
  may mutate between verifier checks; quiesce the environment for a stable final
  evaluation. Arbitrary detached processes are not automatically rolled back/killed.
- A malicious privileged writer can defeat stat-cache assumptions. This is not a
  tamper-proof monitor. Submodule internals need their own coding workspace/session;
  `.git` internals and Git index-only changes are not part of the worktree identity.
  Non-UTF-8 Git path output fails observation rather than being silently decoded
  to a different path. Missing repositories or lost access are reported failures.

## Validation and measured overhead

`tests/test_python_environment.py` exercises real Git repositories, persistent
workers, a local MCP server, subprocesses, kernel termination and abrupt daemon
process exit. It uses test-only providers and makes no live model calls.

Run the diagnostic measurement (not an agent or coding-task benchmark):

```sh
uv run python -m tests.perf_mutation_observation --output results/hardening-step1/mutation-overhead.json
```

Measured on macOS 26.3 arm64, Python 3.12.12; seven warm repetitions per case,
4 KiB files. Cold kernel startup and initial coding preparation/hash capture are
excluded. Observation time includes its begin/end boundaries, persistence and
invalidation. Total cell time also includes existing action/guardrail/index work
and Python checkpointing; those unrelated systems were not optimized in this step.

The original Step 1 output remains in `results/hardening-step1/mutation-overhead.json`.
The same seven-sample workload was rerun after hardening in
`results/hardening/mutation-comparable-final.json`:

| Files | Edit | Before observation | After observation | After whole cell | Content hashed per cell |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | No-op | 32.47 ms | 22.62 ms | 27.33 ms | 0 B |
| 100 | One file | 36.62 ms | 19.69 ms | 27.71 ms | 4,096 B |
| 100 | Ten files | 36.54 ms | 21.99 ms | 28.74 ms | 40,960 B |
| 10,000 | No-op | 457.36 ms | 22.24 ms | 27.66 ms | 0 B |
| 10,000 | One file | 470.59 ms | 36.63 ms | 40.74 ms | 4,096 B |

Raw samples and maximum times are in the output JSON. These are local measurements,
not portable latency guarantees or model-performance claims.
