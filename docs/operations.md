# Configuration, control and security

## Configuration and budgets

configs/session.json uses the existing ChatGPT/Codex subscription. An omitted model
uses Codex's configured/default model, resolved from the live catalog. Run auth status
and auth install-client before first use. See [subscription details](subscription.md).
The optional chat provider accepts explicit API model/base URL/credential-variable
configuration. Unknown providers, unavailable models and missing authentication fail
without a fake-provider or API-billing fallback.

The default task.adapter is workspace, with no verifier or coding preparation.
Explicit configs/coding.json opts into repository baseline/completion policy.
That task configuration includes repository/base_commit; allowed_paths/forbidden_paths;
argv test/build/lint/typecheck commands; capture_baseline/require_clean_baseline;
require_tests/prohibit_test_deletion/protect_tests; required_files/require_change;
and optional benchmark/profiler configuration. CMake/Make projects should provide
explicit commands because target names and build directories vary.

models maps aliases to full provider configurations. routing supports fixed or
role_based policies, a default alias, and mappings such as agent, reviewer,
compaction and refinement. The optional agent_spawn capability can specify a role;
Python rlm selects an optional provider/model and thinking level. Every decision/reason is
recorded. Cost budgets require prices for every routed API model; subscription
usage has null monetary cost and cannot use dollar budgets.

limits apply cumulatively across the recursive tree and resumes. Conservative
UTF-8 byte bounds reserve model input/output capacity; actual provider usage replaces
reservations. Subscription output reservations are estimates: that backend has no
server token cap. Unknown/interrupted usage is marked estimated. Root elapsed
time includes waiting/downtime for autonomous/goal runs. Interactive roots use
cumulative aggregate execution seconds, including descendants, instead of spending
the budget while the human thinks or the conversation is detached and idle. Child
execution times can overlap. An interrupted turn's recovery estimate remains conservative.

## Control

`threadweave` and `threadweave chat` open a persistent interactive root, initially
idle until the first message. `--workspace` defaults to cwd; `--config` overrides
configs/session.json discovery. `--continue` selects the most recently updated root for
that workspace; `--resume ID` opens a specific root and its original workspace and
config. No IDs are needed for subsequent messages. A plain model reply yields to
input; accepted completion also yields without destroying
the session. Only an actual verifier pass is labelled verified. Tool-only responses
continue the existing runtime loop. Attaching to an autonomous/goal/heartbeat root
does not change its existing execution mode.

Chat uses user data storage outside the repository by default (XDG_DATA_HOME or
~/.local/share/threadweave); existing workspace-local .threadweave stores are reused.
Explicit --data always wins. Administrative commands retain their legacy default
of cwd/.threadweave, so specify the chat's data directory when scripting controls.
Input history is stored there with mode 0600, alongside private runtime data.

Live model text appears incrementally in a bounded prompt toolbar; final Markdown
and bounded tool results enter scrollback. prompt_toolkit handles redraw, editing,
history and multiline paste. Alt-Enter/Ctrl-J adds a newline; Enter submits.
Ctrl-C pauses the current session's turn, not its children; use /tree to inspect them.
At an idle prompt Ctrl-C clears input. /exit and Ctrl-D on empty input detach.
Typing while busy queues a durable intervention for the next safe turn boundary.
Partial external actions are not assumed rolled back after cancellation.

Slash controls: /help, /status, /state, /states, /usage, /tree, /diff, /history, /experiments,
/compact, /refine, /pause, /resume, /new, /exit. Diff and manual compaction require the
session to be idle/paused. /new retains prior sessions and does not stop their work.
--json gives debug events; --verbose adds low-level event/artifact details.
Clients reconnect to an already restarted daemon without changing session identity.
On a daemon upgrade, stop the old daemon explicitly and reopen chat; no automatic
daemon shutdown is performed while another client may be using it.

Commands: run, attach, list, status, tree, history, usage, diff, experiments, verify,
input, pause, resume, stop, fork, doctor. Additional controls: config, states, refine,
compact, schedule, schedules, unschedule, artifact and daemon start/status/stop.

Use the same global --data directory for every command, preferably outside the
repository. Pause before manual diff/verification for a stable workspace.
Ctrl-C during attach detaches; it does not cancel. A completed child accepts explicit
parent/human follow-ups without an additional resume command. Its existing ID,
kernel association, workspace, history, reusable state and accounting remain intact;
serializable Python values recover from the existing checkpoint if unloaded.
Completion notifications from descendants do not revive a finished parent. Sibling
messages still queue, but do not automatically reopen completed work. Paused,
failed, cancelled and exhausted sessions require explicit operator control; a
follow-up cannot reset a goal budget or bypass recursive root limits.
Resource-limited sessions require a new fork for a fresh budget; original spend
remains recorded.

Schedules accept intervals or five-field UTC cron, coalescing missed ticks.
Reboot requires daemon startup, manually or through a service manager.

`buffalo refine SESSION_ID [instructions]` schedules refinement. Use `--global`
for cross-session state or `--rollback REFINEMENT_ID` to apply the inverse of a
recorded refinement. `/refine` in chat schedules the same planner. Paused sessions
wait for resume. Pending/in-flight work is in-memory and is cancelled on branch
invalidation; applied state and audit history survive restart. No SQL request
queue or version table participates in refinement. Existing model accounting and
resource limits still apply.

The continual harness uses `harness_state.json` as its only active learned state,
with `prompt`, `memory`, `skill`, and `subagent` entries. Global files live under
`DATA/harness/`; session-local files live under `DATA/sessions/SESSION_ID/harness/`.
Global refinement history appends to `DATA/harness/refinements.jsonl`. Local
refinement history is stored as session audit events, separate from model conversation.
Retired SQLite reinforcement tables and import adapters have been removed.

`await refine.run()` schedules local refinement; optional instructions focus the
planner, and `global_=True` explicitly requests global changes. `await refine.status()`
returns `pending` and `in_flight`. Planning can overlap tools after the model response
finishes; the exact plan is applied only at a safe turn boundary.
Automatic review runs at 25 assistant turns and compaction, with a 20 minute cooldown,
only in root sessions. The reviewer alone decides whether to run the planner.
Explicit refinement bypasses review. An empty edits array is valid.

Each legitimate model request rebuilds its SYSTEM prompt from the merged local/global
harness JSON. Refinement results stay in audit history; no synthetic refinement messages
enter conversation, and applied edits do not resume completed or idle tasks.
Colliding global/local IDs remain visible with
scope labels; local guidance can override global guidance within the session.
Skills reference existing Python callables and their argument contracts; reusable
subagent specifications execute through native `rlm` delegation.

See [the source, deletion, and test mapping](prime-refinement-specification.md).

## Security boundaries

Local Python and commands have the daemon user's OS authority. This is trusted-host
execution, NOT a sandbox. Tool checks cannot contain arbitrary Python, imports or
host subprocesses. Use a dedicated OS account, container or VM for untrusted tasks.

File tools resolve paths and reject workspace escapes and direct .git/private-state
access. Editors reject symlink traversal. Command allowlists match exact argv[0].
Permitting an interpreter permits its programs; this is not a shell-language sandbox.

Child environments contain only allowlisted variables and permitted overrides.
Credential-like variables are excluded; provider credentials stay in the daemon.
Event payloads redact known credential values and sensitive fields, but cannot
discover every secret in arbitrary files. Exact artifacts/checkpoints/patches may
contain secrets and must remain private.

For container commands configure, adapting the image to an installed toolchain:

```json
{
  "control_plane": "direct",
  "execution": {
    "backend": "container", "engine": "docker",
    "image": "your-prebuilt-toolchain-image:immutable-tag",
    "network": false, "read_only": false, "memory": "4g", "cpus": 2
  },
  "permissions": ["workspace.read", "workspace.write", "process", "agents", "state"]
}
```

Docker/Podman must be installed and available. The container gets a read-only root,
temporary storage, dropped capabilities, a PID limit and only its task workspace
bind mount. network=false uses the engine's no-network mode. Writable commands can
still modify the mounted repository; final verification checks configured paths.
Local network restrictions are reported as unenforced, and local read-only command
execution is rejected.

Host Python workers are unavailable in container-only configurations. GPU runtime/
device passthrough is not implemented; use a configured trusted GPU host or an
externally isolated environment. Container execution is optional for ordinary coding.

Cancellation kills/reaps process groups. Supervisors also stop groups if the daemon
disappears. Deliberately detached host processes can escape that boundary. Ordinary
container cancellation removes the named container. Durable container leases are
cleaned during daemon recovery after hard death. If the engine is unavailable,
recovery pauses the affected session and records the cleanup failure; inspect the
engine before resuming. Containers may keep running while the daemon is down.
This is not a distributed container lease service.

## Storage and backups

```text
DATA/
  history.sqlite3    durable sessions, history, queues, state versions, indexes,
                     checkpoints, experiments, routing and verification
  artifacts/         exact private values, patches, files, logs and measurements
  kernels/           per-session snapshots, receipts and logs
  workspaces/        isolated candidates and verification copies
  daemon.log         structured event IDs/types and startup diagnostics
  daemon.lock / runtime.lock
```

Stop the daemon and back up the whole data directory plus associated workspaces.
Online backups need SQLite backup APIs coordinated with file snapshots; copying
only the main database during WAL activity is insufficient.

Storage is not encrypted and has no pruning/quota service. Monitor disk use for
large logs, snapshots and candidate copies. Checkpoint files above 64 MiB fail
explicitly. The source reference PDF is not installed with the package.

doctor reports provider/model and credential presence (not values), Git/ripgrep,
requested container health, available compilers and optional CUDA/profilers,
data-directory writability and SQLite quick_check. It does not make a paid model call.
