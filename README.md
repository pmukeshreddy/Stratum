# Threadweave

A working Python harness for persistent, recursive agent sessions. A local daemon
owns the execution loop; each session has an independent Python worker, durable
history, message queue, context cache, and lifecycle. Models choose actions and
delegation. The runtime supplies persistence, concurrency, verification, and limits.

Requires Python 3.11+ on macOS or Linux. The offline demonstration and tests need
no model account, credentials, network access, or paid API calls.

## Run the complete offline example

```sh
uv sync --extra dev
mkdir -p demo-workspace
uv run threadweave demo --workspace demo-workspace
```

This starts a detached daemon, creates a root session, computes in a persistent
Python worker, spawns a concurrent child with its own worker, exchanges a message,
writes `demo-workspace/answer.txt`, and completes through a file verifier. The
answer is `499500`. The demonstration provider is a deterministic test program;
real model providers choose their own sequence of actions.

Session data defaults to `.threadweave/` in your current directory. Choose another
location with `threadweave --data /absolute/path ...`; use the same location when
resuming. Stopping an attached CLI with Ctrl-C **detaches without cancelling**.

```sh
uv run threadweave list
uv run threadweave tree SESSION_ID
uv run threadweave status SESSION_ID
uv run threadweave history SESSION_ID --limit 20
uv run threadweave usage SESSION_ID
uv run threadweave input SESSION_ID "Check the result against the original objective."
uv run threadweave resume SESSION_ID
uv run threadweave attach SESSION_ID
```

Use `pause SESSION_ID` to unload a session while retaining queued input;
`resume SESSION_ID` continues it under the same identity and remaining budgets.
Use `stop SESSION_ID` to cancel it and its active descendants. `stop --only`
cancels that session while allowing descendants to continue within the root's
remaining budget. Completion, cancellation, and failure do not delete history.

```sh
uv run threadweave daemon stop
uv run threadweave daemon start
uv run threadweave attach SESSION_ID
```

Restart recovers runnable sessions automatically. Paused sessions stay paused.
The daemon itself must be restarted after an OS reboot; it can be run under your
service manager with `python -m threadweave.daemon --data /absolute/path`.

## Use a model provider

Copy [examples/chat.json](examples/chat.json), set `provider.model` to a model ID
available at your endpoint, and adjust the budgets. Supply credentials in the
environment **before starting the daemon**:

```sh
export OPENAI_API_KEY='your-key'
uv run threadweave daemon start
uv run threadweave run "Inspect this repository and implement the requested change" \
  --workspace /absolute/path/to/repository --config examples/chat.json --attach
```

The `chat` provider implements streamed and non-streamed chat completions with
tool calls, usage, timeouts, and cancellation. Set `base_url` for a compatible
local service; set `api_key_env` to `""` when the service requires no key. There is
no default paid model ID. Provider parameters such as temperature are passed
through; the harness owns the output-token maximum and protocol fields. Its wire
format follows the [official API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Changing credentials in a CLI does not change an already running daemon's
environment. Stop and start the daemon after updating its credentials.

## Python, tools, and retained information

Every session can incrementally execute Python, including top-level `await`:

```python
rows = tools.call("artifact_load", artifact_id="ARTIFACT_ID")
selected = [row for row in rows if row["score"] > 0.9]
len(selected)  # Only this result's bounded representation enters model context.

child = await tools.acall("agent_spawn", instruction="Check the calculation", name="checker")
tools.call("agent_message", recipient_id=child["session_id"], body="Please inspect the edge cases.")
forget("rows")
```

Values persist across turns and can be inspected later. JSON-compatible values,
collections, bytes, paths, and imported modules have explicit checkpoint codecs.
Functions, clients, iterators, GPU objects, and other live handles need a saved
artifact plus an explicit reconstruction recipe when feasible:

```python
from pathlib import Path
report = Path("report.json")
client = SomeClient.from_config(report)
remember_recipe("client", "client = SomeClient.from_config(report)")
```

The recipe must recreate its dependencies too if they are not checkpointed.
Recipes are opt-in executable code. Ordinary action history is never replayed on
recovery. Missing values and failed recipes are reported, not silently restored.

Python and process permissions run trusted code with the daemon user's OS
authority. This is **not an OS sandbox**. For untrusted tasks, run the daemon in
an appropriately isolated container or account. Built-in file tools enforce
workspace boundaries, and tool capabilities/allowlists apply to Python bridge
calls as well as direct model actions.

## Goals, schedules, refinement, and branching

```sh
uv run threadweave run "Maintain this objective" --mode goal --config examples/chat.json
uv run threadweave schedule SESSION_ID --interval 300
uv run threadweave schedule SESSION_ID --cron '0 */6 * * *'
uv run threadweave schedules SESSION_ID
uv run threadweave unschedule SCHEDULE_ID

uv run threadweave states SESSION_ID
uv run threadweave refine SESSION_ID path/to/state-edit.json
uv run threadweave pause SESSION_ID
uv run threadweave fork SESSION_ID --name experiment
```

Schedules use UTC and coalesce overdue ticks. A `heartbeat` session runs one turn
per activation; autonomous and goal sessions continue until an explicit finish,
successful verifier, wait action, or limit. Messages wake active, unpaused agents.
Terminal sessions require explicit resume. A resource-limited run must be forked
to start a new accounting budget; the original spend stays recorded.

Refinement produces typed, versioned memories, prompt notes, executable skills,
and subagent specifications. Edits require source event IDs and an intended
effect, apply at a turn boundary, and support deletion and rollback as new
versions. Global writes are disabled by default. Supplemental entries enter
context only when explicitly selected. Foundational instructions are separate.

A fork has a new session/root identity, explicit ancestry, copied context and
recoverable Python state, and fresh accounting. It does not clone descendants or
the workspace. Pause a running source first. Forking a partial turn preserves its
source journal and warns the new continuation to inspect uncertain effects.

## Verify and extend

```sh
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

The tests cover lifecycle, events, compaction, kernels, versioning/rollback,
messages, recursive concurrency, resource limits, provider streaming/retries,
permissions, schedules, branching, and daemon process death/restart. API tests
use an in-memory HTTP transport. The integration suite kills a real daemon and
recovers the same session tree, and kills it during tools to check uncertain
effects are not duplicated.

See [architecture and recovery](docs/architecture.md),
[providers, tools, and task adapters](docs/extensions.md), and
[configuration and operations](docs/operations.md).
