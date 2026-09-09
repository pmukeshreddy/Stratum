# Threadweave

Threadweave is a persistent recursive agent harness. An interactive Agents View
attaches to a daemon-owned Root Session. The default model-facing tool is **ipython**.
The model writes Python to select computation, environment actions, persistent
recursive children and messages. Each session owns its context and persistent
IPython kernel; history and reusable state live on disk.
Ordinary conversation does not require a repository or a coding workflow.

## Open the Agents View

```sh
uv sync --extra dev
uv run threadweave auth status
# Only if the shared Codex ChatGPT login is absent:
uv run threadweave auth login
uv run threadweave doctor --config configs/session.json
uv run threadweave
```

The default uses the current directory and configs/session.json when available,
otherwise an equivalent subscription-backed workspace configuration. It does not
select configs/coding.json implicitly. --workspace and --config override discovery;
threadweave chat is the same interface.

```text
> hi
> Use Python to create x = 123 and remember it.
> Read x from your existing REPL.
> Create two independent children with rlm(), then continue working locally.
> /tree
> /compact
> Read x again.
> /state
> /exit
```

Use `uv run threadweave --continue` for the most recent root in this workspace,
or --resume SESSION_ID to choose one. Attachment does not change execution mode or
reset accounting. Enter sends; Alt-Enter/Ctrl-J inserts a newline; arrow keys recall
input. Input stays available during work; interventions are queued at a safe turn
boundary. Ctrl-C pauses the current turn or clears idle input. Ctrl-D and /exit
detach without killing work.

/help lists /status, /state, /states, /usage, /tree, /history, /compact, /pause,
/resume, /new, /exit, and optional Environment inspections /diff and /experiments.
Model text streams. Large tool outputs remain in artifacts; --json and --verbose
expose debugging detail.

## Authentication

The normal provider is codex_subscription, reusing the official shared Codex
ChatGPT login. No OPENAI_API_KEY is required. auth models lists supported models.
Omit provider.model to resolve the account default, or select a supported model.
provider.parameters.reasoning_effort overrides account settings.

Authentication/refresh remain owned by Codex. The official-client inference bridge
makes model requests only: Threadweave supplies context/tool schemas and executes
returned calls. It does not invoke another agent to solve tasks. There is no API
billing fallback. Subscription dollar cost is null. The backend does not expose a
server-enforced output-token cap; a client-observed byte guard and time budgets
apply. [Provider details and optional API mode](docs/subscription.md).
auth logout signs out of the shared Codex login.

## Architecture

The root's ordinary action loop chooses how to solve each task. See the
[execution architecture and Python API](docs/adaptive-orchestration.md),
[exact root foundation](docs/root-foundation.txt), and
[targeted validation](docs/adaptive-validation.md).

```text
Human ↔ Agents View ↔ Root Session ↔ Environment
                          ↕ rlm / messages    ↕
                     Recursive Subagents ────+
                          ↕       ↕
                           Daemon
                              ↕
                       Continual Harness

L1: selected active context
L2: persistent REPLs, retained values, recursive sessions/handles
L3: disk-backed history, artifacts, messages, reusable state, session metadata

Long-horizon controls: autonomous mode, persistent goals, heartbeats, budgets
```

`await rlm(instruction, name=None)` returns a stable child handle after admission without
waiting for a child answer. Children have independent contexts, REPLs and histories;
they can create descendants. Related sessions communicate through durable queues.
A parent (or human) can send follow-up work to a completed child to continue that
same session, including after its kernel has been unloaded. History, recoverable
Python values and accumulated budgets are retained. Paused, cancelled, failed and
resource-exhausted children are not automatically restarted. A failed child does
not destroy the root.

The Continual Harness retains append-only history and versioned memories,
executable skills, prompt notes and reusable subagent specifications. `rlm.harness`
provides immediate, audited CRUD from Python. Automatic refinement proposals are
validated and applied at turn boundaries. Explicit selection, deletion, rollback
and optional global scope are supported. Foundational policy and weights are unchanged.
Interactive `/refine` requests an evidence-based model pass at a safe boundary,
even when periodic automatic refinement is off. It reports requested, applied,
skipped or failed; a request is not a claim that learning has occurred. Refinement
must be enabled, and a pass with no new usable evidence makes no changes.

Compaction only changes L1. History, REPL values and children remain intact.
Recovery restores stable IDs, topology, queues, contexts, versions, goals, schedules,
accounting and supported Python checkpoint values. Non-serializable objects require
explicit reconstruction recipes; missing/uncertain state is reported, not invented.
[Component implementation and connection tests](docs/architecture.md).

## Programmable control plane

These are Python cells chosen by the model, not direct model tool calls:

```python
x = 123
py_files = list(workspace.rglob("*.py"))  # retained outside the model prompt
matches = repo.search("class Session")
review = await rlm("Inspect persistence; message me your findings.", name="reviewer")
# Admission returns a handle; the parent can keep computing here.
status = await agent_observe.get_agent(review.session_id)
await agent_message.send("Focus on recovery.", receiver_role="child", receiver_name="reviewer")
result = await bash("uv run pytest -q")
print(result.exit_code, result.output[-1000:])
note = harness.create_memory("Observation", "The test command above completed.")
```

Files/Path, shell handles, editing, repository retrieval, MCP, skills and durable
state are preloaded in roots and children. `tools.catalog()` retrieves optional
capability schemas into Python. The full instruction is in `context['task']`;
`context['messages_path']` points to a readable full-history JSONL projection.
Only explicitly returned/printed selections enter L1. Normal final text ends/yields
ordinary interaction; there is no universal coding verifier or finish tool.

[Exact API, source trace, configuration and boundaries](docs/python-control-plane.md).
Legacy direct tools require explicit `"control_plane": "direct"`; they are not the default.

## Environment capabilities

Files, processes, repository search/indexing, validated patches, Git checkpoints,
tests/builds, benchmarks/profilers and durable experiments are Environment tools.
The model chooses when to use them. Missing prerequisites return errors; a tool's
existence does not imply a successful run.

For a task explicitly needing coding baseline and independent completion gates:

```sh
uv run threadweave run "Fix the failing tests without weakening them." \
  --workspace /path/to/repository --config configs/coding.json --attach
```

This optional environment captures baseline evidence and verifies completion against
configured commands and repository constraints. Interactive greetings/inspection
do not run its baseline. Python `rlm()` defaults to a shared workspace. Its explicit
`purpose="research"` sets read-only tool policy (not an OS sandbox);
`purpose="candidate"` creates a coding environment in a Git worktree based on the
parent's captured dirty state. Inspect with `await rlm.candidate(handle)` and
explicitly accept with `await rlm.candidate(handle, accept=True)`. Worktrees share
Git objects, not writable source files. No user branch is committed or stashed;
harness checkpoints use private Git refs. configs/kernel.json selects compile/correctness/
performance commands through the same architecture.

Repository definitions/calls/references use persistent Tree-sitter syntax evidence
for Python, Rust, Go, C/C++/CUDA and JavaScript/TypeScript. Results identify evidence
quality; unresolved names are not compiler-semantic references. Native filesystem
events drive warm mutation/index updates, with full reconciliation after recovery
and at verification boundaries. See the [architecture documentation](docs/architecture.md)
for runtime APIs and remaining limits.

## Evaluation

The primary comparison holds **Astra XHigh** (`gpt-6-astra`, `xhigh`) fixed:

| Harness | Model | Reported score |
| --- | --- | ---: |
| ARC Standard harness | Astra XHigh | ≈ 59% |
| Buffalo harness | Astra XHigh | ≈ 81% |

**59 → 81: approximately +22 percentage points.** Buffalo improves the same
underlying model by replacing the ARC Standard harness.

These are reported, rounded results; their underlying score artifacts and the ARC
Standard runner are not included in this checkout.

Run `buffalo eval arc-agi-3 --config /absolute/evaluation.json` to evaluate Buffalo
on the official 25 ARC-AGI-3 environments. Its reports retain the measured scores
and usage. The other four official benchmark families are paused.

See [evaluation setup and reproducibility](docs/evaluation.md).

## Long-horizon sessions

run --mode autonomous continues until completion, a configured end-condition or
limits. --mode goal persists the objective across continuations. --mode heartbeat
executes scheduled turns. schedule_turn supports intervals and UTC cron.
Turn/token/time/tool/Python/depth/concurrency limits and accounting include
descendants; detach/resume does not reset budgets.

New chats store state in the user's XDG data directory (normally
~/.local/share/threadweave), reuse an existing workspace .threadweave/history.sqlite3,
or use --data. Automation retains list, status, tree, history, usage, states, input,
attach, pause, resume, stop, fork and schedule. Explicitly restart an old daemon
after upgrading: live processes do not reload source changes.

## Tests and boundaries

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv build

# Real subscription, terminal, root/children, compaction and hard restart:
env -u OPENAI_API_KEY THREADWEAVE_LIVE_PARITY=1 \
  THREADWEAVE_PARITY_OUTPUT=/absolute/path/to/acceptance-results \
  uv run pytest -s tests/test_python_control_live.py
```

Deterministic providers exist only in tests. The opt-in test uses the real provider
and stores raw terminal output, event trajectories and a result summary. Use a new
output directory for each acceptance run.

Local Python/processes execute trusted code with the host user's authority: not a
sandbox. Container command execution does not sandbox host REPLs. Artifacts may
contain sensitive workspace content and are private by default. This is a
single-host daemon; arbitrary-object recovery and exactly-once external effects
are not guaranteed. [Operations/security](docs/operations.md),
[extensions](docs/extensions.md), [Environment evaluation](docs/evaluation.md).
