# Architecture

```text
                         Human
                           ↕
                      Agents View
                           ↕
                      Root Session
                     ↙     ↕      ↘
           rlm() / messages ↕       actions / observations
                   ↙       ↕                ↘
      Recursive Subagents  ↕             Environment
              ↕            ↕                 ↕
              +--------- Daemon -------------+
                           ↕
                    Continual Harness

L1  Selected active model context
L2  Persistent Python REPLs, retained values, recursive sessions/handles
L3  Disk-backed history, artifacts, versions, sessions, messages

Long-horizon controls: autonomous turns · persistent goals · heartbeats
                      bounded across root + descendants
```

## Agents View ↔ Root Session

chat.py and terminal.py implement the Agents View; daemon.py exposes its private
Unix-socket API. Clients submit durable input, read bounded streaming events,
inspect history/tree/usage/layers, pause, resume and detach. They do not invoke
models or execute tools. Attachment does not change existing execution mode;
disconnection does not cancel work. The daemon owns lifecycle and scheduling.

The default configs/session.json uses an ordinary workspace, not a coding task.
A greeting requires neither Git, tests nor edits. Explicit coding interactive
inspection defers baseline preparation until an action requires it.

## Root Session ↔ Environment

models.Session and runtime.Runtime implement the persistent session loop:

1. At a safe boundary apply queued refinements and receive messages.
2. Assemble selected L1 context and invoke the configured model.
3. Persist the response/action cursor before executing chosen tools.
4. Retain observations externally, select bounded results for L1, continue.

No mandatory planning/action graph exists. finish requests completion; only a
configured verifier/end-condition supplies independent verification. Interactive
text replies yield to the human. Neither text nor unverified finish is reported as
independent verification.

environment.Environment owns adapter admission, preparation, coding checkpoints,
candidate isolation and external-effect recovery. TaskAdapters retain their
prepare/verify interface. tools.ToolRegistry exposes typed permission-controlled
primitives. Files/processes/repository/editor/Git/tests/builds/experiments/profilers
are capabilities, not stages. Tools are exposed independently of the task adapter;
missing prerequisites return structured errors. Standalone build/test actions run
configured/detected commands without requiring a baseline.

## Root → rlm() → Recursive Subagents

The model tool rlm and Python helper rlm(instruction, name=None, **options) use
Runtime.spawn and its existing scheduler. agent_spawn is a compatible spelling.
They return JSON-safe metadata containing the stable session ID, not a child
answer. Creation does not await child model execution.

Children use the same Session, model loop, history, permissions and worker
implementation as roots, with independent context/kernel identity. They can call
rlm recursively. Root values are not implicitly copied. Generic environments share
workspace metadata. Explicit coding environments isolate writable candidates
before admission: a large copy may delay admission, but never waits for the
child's reasoning/result. Candidate patches require explicit acceptance.

agent_message, agent_receive, agent_sessions, session_inspect and agent_wait expose
persistent communication/inspection. Parent/child and permitted sibling messages
are queued in SQLite, timestamped, referenced by events and delivered at boundaries.
Paused or terminated recipients retain messages; terminated sessions need explicit
resumption. Completed children preserve identity/history/recoverable state.
Failure reports reach parents without automatically terminating them.

## Sessions ↔ Daemon

daemon.Daemon owns Runtime under an exclusive data-directory lock. Runtime owns
live workers, concurrent turns and model calls. The scheduler enforces global and
per-root concurrency, depth/subagent limits and cancellation.

Lifecycle ADMITTED → RUNNING → IDLE → INACTIVE describes loading, separately from
outcome and execution mode. INACTIVE means recoverable, not deleted. Pause interrupts
one session without clearing state; stop explicitly cancels a tree. Detach affects
neither. Shutdown checkpoints workers. Restart restores registry/tree and runnable
work; a service manager or opening the CLI restarts the daemon after reboot.

## Daemon ↔ Continual Harness

storage.Store and refinement.MemoryServices implement the existing Continual
Harness: append-only history and typed, versioned memories, executable skills,
prompt notes and reusable subagent specifications.

refine(edit=StateEdit(...)) queues create/update/delete/rollback operations with
source events, intended effect and optional expected_version. Runtime applies them
at turn boundaries. Deletion and rollback append versions, never overwrite history.
Entries have session-local or explicitly permitted global scope. state_list,
state_read, state_select, skill_search, skill_inspect and skill_run provide access.

Only selected entries enter supplemental L1. Skills validate syntax/input schemas/
declared permissions, record outcomes and quarantine repeatedly failing versions.
This is not a sandbox. Automatic refinement optionally proposes evidence-backed
state at configured boundaries. Auxiliary calls use the same provider retry,
reservation and accounting path. Foundational policy/model weights are unchanged.

## L1 / L2 / L3

context.Context assembles L1 from foundational instructions, objective, metadata,
selected entries, bounded summaries, recent complete model/tool blocks and messages.
It does not automatically dump arbitrary REPL values or disk history.

kernel.Kernel and kernel_worker.Worker implement L2, one persistent process per
loaded session. Variables survive incremental executions. Top-level await,
tools.call, tools.acall, rlm, workspace, forget and remember_recipe are available.
Full programmatic results can stay in Python. Printing/returning them is explicit
selection, with bounded capture and full retained artifacts.

SQLite WAL/full synchronization, private artifacts and kernel checkpoints implement
L3: events, messages, metadata/tree, contexts, compactions, versions, goals,
schedules, action receipts and usage. FTS/history/artifact retrieval is scoped to
the tree and explicit branch ancestry.

Model compaction works in every environment, with recorded extractive fallback.
It changes L1 only: events, workers and children remain intact. /state and
information_inspect return bounded metadata, not variable contents. Checkpointed
names describe recoverable values, not arbitrary live-object serialization.

Recovery restores explicit codecs and opted-in reconstruction recipes; unsupported
objects are reported. Python trajectory history is never blindly replayed.
Receipts/cursors prevent blind action repetition; unreceipted external effects are
marked uncertain. Branching creates new IDs/explicit ancestry and preserves the
original trajectory. Isolated forks rebind Path codecs, not arbitrary strings or
potentially source-bound reconstruction recipes.

## Long-horizon execution controls

run --mode autonomous continues chosen turns until explicit completion, an
end-condition, failure or limits. --mode goal additionally persists objective/
status across continuations. --mode heartbeat yields after each triggered turn.
schedule_turn and daemon schedule controls store intervals or five-field UTC cron
triggers for existing sessions. Missed ticks coalesce.

Root accounting includes all descendants: model/input/output/cached/reasoning tokens,
tools, Python, retries, verifier calls, subagents, turns and execution time.
Reservations prevent concurrent delegation from hiding spend. Autonomous/goal wall
budgets include elapsed run time; interactive budgets exclude human idle time and
conservatively sum execution time. Subscription cost is null, not fabricated from
API prices. Resume/detach never resets limits.

## Connection tests

| Connection/control | Tests |
| --- | --- |
| Agents View ↔ Root, input/intervention/detach | test_chat.py, test_chat_terminal.py, test_architecture.py |
| Root ↔ Environment, no coding prerequisite | test_architecture.py, test_tools_and_controls.py |
| Root → rlm → children ↔ Environment → messages | test_architecture.py, test_parallel_environment.py |
| Recursive descendants/sibling permissions/failure isolation | test_runtime.py, test_storage.py |
| Sessions ↔ Daemon, hard restart, same IDs | test_daemon.py, test_chat_terminal.py |
| Daemon ↔ Continual Harness, versions/provenance/rollback | test_architecture.py, test_storage.py, test_intelligence.py |
| L1/L2/L3, compaction, persistent Python/recovery | test_architecture.py, test_context.py, test_kernel.py |
| Autonomous/goal/heartbeat, gates, budgets/accounting | test_runtime.py, test_tools_and_controls.py, test_architecture.py |
| Real subscription + CLI + children + compaction + hard restart | opt-in test_architecture_live.py |

No distributed daemon, arbitrary-object serialization, exactly-once external
effects or host-execution sandbox is claimed.
