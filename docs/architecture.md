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

No mandatory planning/action graph exists. In default Python mode final text ends
an autonomous request or yields an interactive one. Persistent goals explicitly
request `await goal.complete()`. Only a configured verifier/end-condition supplies independent verification. Interactive
text replies yield to the human. Neither text nor unverified finish is reported as
independent verification.

environment.Environment owns adapter admission, preparation, coding checkpoints,
candidate isolation and external-effect recovery. TaskAdapters retain their
prepare/verify interface. tools.ToolRegistry exposes typed permission-controlled
primitives. Files/processes/repository/editor/Git/tests/builds/experiments/profilers
are capabilities, not stages. The model receives only the `ipython` schema. All
other registered capabilities are accessed from Python, independently of the adapter;
missing prerequisites return structured errors. Standalone build/test actions run
configured/detected commands without requiring a baseline.

## Root → rlm() → Recursive Subagents

`await rlm(instruction, name=None, model=None, thinking=None)` uses a correlated
worker-to-daemon request, Runtime.spawn and its existing scheduler. It returns an
AgentHandle with stable session identity, not a child answer. Creation does not
await child model execution. Handles have explicit recovery codecs.

Children use the same Session, model loop, history, permissions and worker
implementation as roots, with independent context/kernel identity. They can call
rlm recursively. Root values are not implicitly copied. Generic environments share
workspace metadata. The default Python rlm path shares the workspace even for a
coding-configured parent. Isolated candidates require explicit selection through
the optional coding capability; candidate patches require explicit acceptance.

`agent_message.send/receive/list_agents` and `agent_observe.get_agent/recent_messages`
expose programmatic persistent communication/inspection. Parent/child and permitted sibling messages
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

The continual harness uses `harness_state.json` as its only active learned state,
with `prompt`, `memory`, `skill`, and `subagent` entries. Global files live under
`DATA/harness/`; session-local files live under `DATA/sessions/SESSION_ID/harness/`.
Each scope appends refinement records to `refinements.jsonl`. Existing SQLite
learned state is imported once; old tables have no runtime readers or writers.

`await refine.run()` schedules local refinement; optional instructions focus the
planner, and `global_=True` explicitly requests global changes. `await refine.status()`
returns `pending` and `in_flight`. Application runs at a completed-turn boundary,
then a durable `[self-refinement]` or `[auto-refinement]` notice informs the root
before it continues. Zero-edit proposals produce no update notice. Automatic review
is enabled at 25 turns and compaction, with a 20 minute cooldown. Failures and child
findings are ordinary trajectory evidence, not separate refinement triggers.

A compact merged digest enters context at session start, resume, compaction, and
stale-state detection. Unchanged digests are deduplicated. The base system prompt
stays unchanged after learning. Colliding global/local IDs remain visible with
scope labels; local guidance can override global guidance within the session.
Skills reference existing Python callables and their argument contracts; reusable
subagent specifications execute through native `rlm` delegation.

See [the source/test parity matrix](continual-harness-parity.md) and
[the deterministic session trace](continual-harness-trace.json).

## L1 / L2 / L3

context.Context assembles L1 from foundational instructions, objective, metadata,
selected entries, bounded summaries, recent complete model/tool blocks and messages.
It does not automatically dump arbitrary REPL values or disk history.

kernel.Kernel and kernel_worker.Worker implement L2, one persistent process per
loaded session. IPython transformation and a persistent asyncio loop support magic
syntax, top-level await and background tasks across cells. Tools, rlm, bash, messaging,
MCP, skills, durable state, workspace, forget and remember_recipe are preloaded.
Full programmatic results can stay in Python. Printing/returning them is explicit
selection, with bounded capture and full retained artifacts.

SQLite WAL/full synchronization, private artifacts and kernel checkpoints implement
L3: events, messages, metadata/tree, contexts, compactions, versions, goals,
schedules, process handles, action receipts and usage. FTS/history/artifact retrieval is scoped to
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
| Python-only provider schema, async rlm, shell, MCP, executable skill packages, history export | test_python_control.py, test_python_capabilities.py |
| Real subscription + Python-only CLI + children + compaction + hard restart | opt-in test_python_control_live.py |

No distributed daemon, arbitrary-object serialization, exactly-once external
effects or host-execution sandbox is claimed.
