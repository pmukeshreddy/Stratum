# Python execution semantics

## Request boundary and implementation

The following table maps Python execution behavior to its production implementation.
External reference investigation evidence is kept outside the repository.

| Executed behavior | Threadweave implementation |
| --- | --- |
| Default active tool is ipython only | `ToolRegistry.schemas`, `_execute_action` provider-origin enforcement |
| Session bootstrap with async recursion, shell, MCP, Python skill modules | `kernel_api.bootstrap`, `kernel_worker.Worker` |
| Async child admission returns stable handle before detached execution | `Recursive.run`, `host_api.dispatch`, `Runtime.spawn` |
| Child inherits cwd/environment, not parent Python variables | Shared workspace + independent kernel/session, optional isolation separate |
| Role/name-addressed persistent messages; explicit child answers | `Messaging`, daemon-mediated durable queue |
| Bounded session and message inspection | `Observation`, scoped host dispatch |
| Immediate process handle, await/poll/output/kill, background lifetime | `Bash`, `BashHandle`, `BackgroundProcesses` |
| Unique exact text replacement relative to kernel cwd | `Edit.run`, host path validation and existing Editor journal |
| SKILL.md discovery and src-layout Python modules, callable if run exists | `skills.discover/load_module`, bootstrap and explicit skills API |
| Immediate local/global durable CRUD, compact supplemental menu | `Harness`, validated versioned state, explicit content selection |
| Lazy stdio/streamable HTTP discovery/calls/lifecycle; structured content preferred | Official Python MCP SDK in `mcp_client`, kernel normalization |
| Normal final answer ends work; conversation log accessible by path | Normal-text completion/yield, readable history projection |

The production `_invoke` interface exposes only `ipython`, with one required string
argument `code`. The [runtime schema fixture](../tests/fixtures/ipython-schema.json)
and provider integration tests verify the control-plane contract. A model attempting
a direct `workspace_write` action is rejected as `tool_not_exposed`.

```text
MODEL → ipython(code) → persistent IPython kernel
                     → Environment / daemon / persistent state
```


The old direct tools remain behind explicit `control_plane: direct`. They are
also available from `tools.call` under Python when their permissions allow it.
There is no second direct tool surface in the default provider request.

## Session bootstrap and information

Every root/child receives `asyncio`, `os`, `pathlib`, `Path`, `json`, `workspace`,
`session`, `context`, `rlm`, `agent_message`, `agent_observe`, `bash`, `edit`, `mcp`,
`harness`, `skills`, `goal`, `history`, `artifacts`, `repo`, `git`, `tests`, `build`,
`bench`, `compact`, `refine`, `heartbeat`, `tools`, `forget`, `remember_recipe`.
IPython magic transformations and top-level await execute on a persistent event
loop; async tasks can remain alive between cells. Idle/background/import output
goes to the private `session.console_log`, never the worker RPC channel.

`context['task']` contains the complete original instruction. The model prompt
contains a bounded excerpt, not the entire input. `context['messages_path']`
points to JSONL events materialized from authoritative SQLite at invocation/cell
boundaries. Read, filter and aggregate it in Python. `history.read(after=-1,
limit=100)` starts forward pagination; pass the last `seq` for the next page.
`history.search`, `history.get`, `history.messages`, `artifacts.load/read/search`
provide scoped retrieval. `Store.iter_events` exports complete trajectories in
pages; the bounded recent-events endpoint is not a full export API.

Automatic coding-focus injection is disabled in Python mode. Supplemental state
is a bounded menu; full contents require inspection/selection. Compaction changes
L1 only. Full history stays in L3 and retained computations stay in L2. Supported
codecs include containers, Paths, bytes, modules, records, agent handles and process
handles. Arbitrary objects require explicit reconstruction recipes; interrupted
processes are never silently replayed.

## Environment and orchestration APIs

```python
files = list(workspace.rglob("*.py"))
matches = repo.search("Lifecycle")
outline = repo.outline("src/threadweave/runtime.py")
await edit("notes.txt", "one unique old passage", "replacement")
diff = git.diff()
job = bash("uv run pytest -q")
print(job.pid, job.running)
result = await job
print(result.exit_code, result.output[-1200:])

review = await rlm("Inspect persistence; explicitly message your findings.", name="review")
local_value = 123  # executes without waiting for review to finish
status = await agent_observe.get_agent(review.session_id)
await agent_message.send("Check restart handling.", receiver_role="child", receiver_name="review")
# Child: await agent_message.send('Evidence...', receiver_role='parent')
messages = history.messages()
```

The daemon owns process groups, streams, timeout/cancellation and external
artifacts. A returned BashHandle starts work immediately; an owned one-shot await
is cancelled with its cell, while separately retained handles can continue. Stop
and daemon shutdown clean up owned processes and MCP transports. Full stdout,
stderr and combined output are retained; preview output is bounded.

`repo.map/search/symbols/references/outline/dependencies`, `git.status/diff/checkpoint/restore`,
`edit.apply_patch/rollback`, `tests.run`, `build.run`, `bench.run` reuse the existing
capabilities. `tools.catalog()` returns their schemas into Python. For example,
`tools.call('experiment_create', ...)` remains optional; experiments do not control
the loop. Raw Python variables and bash are sufficient to run experiments.

Default `rlm` children share the parent's workspace and permissions, have their own
model context/REPL, and can recurse. They do not inherit coding baseline/verifier
policy. Explicit `tools.call('agent_spawn', instruction=..., isolate=True)` retains
the optional coding candidate isolation mechanism. Child completion generates a
lifecycle notice, not an implicit answer; useful results require explicit messages.

## Continual state and skills

```python
await refine.status()
await refine.run()
await refine.run("Correct the project validation lesson")
await refine.run(global_=True)

harness.list()
harness.get("memory", "lesson")
rlm.get_harness_state()
```

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

Installed skills remain ordinary Python modules. Read SKILL.md and invoke the
documented callable directly; `skills.list()` and `skills.load(name)` provide
inspection and importing. A learned skill's JSON holds a Python `reference` and
an `arguments` contract, never executable implementation text. Harness inspection
is read-only. `refine.run` schedules all model-driven create/update/delete planning
and application at a safe turn boundary.

## MCP configuration

MCP is an actual SDK transport, not an in-process mock. Add `mcp` to permissions.
Servers start lazily, so a greeting does not start MCP servers.

```json
{
  "permissions": ["python", "agents", "state", "workspace.read", "workspace.write", "process", "mcp"],
  "mcp_servers": {
    "local": {
      "type": "stdio", "command": "your-installed-server", "args": [],
      "cwd": ".", "env_from": {"SERVER_TOKEN": "MY_SERVER_TOKEN"},
      "enabled_tools": ["lookup"]
    },
    "remote": {
      "type": "http", "url": "https://your-server.example/mcp",
      "headers_from": {"Authorization": "MY_SERVER_AUTH_HEADER"},
      "startup_timeout_seconds": 30, "call_timeout_seconds": 60
    }
  }
}
```

```python
servers = await mcp.list_servers()
schemas = await mcp.list_tools("local")
value = await mcp.call_tool("local", "lookup", {"query": "evidence"})
await mcp.reload("local")
```

Credential values are resolved by the daemon, not stored in config or passed to
the kernel. Known values are redacted from returned strings. SDK results preserve
structured content (including falsy values), otherwise text or non-text blocks.
Tool errors raise; transport failure/timeout reports uncertain external effects.
Connections have per-session ownership, reload/close and startup/call timeouts.

## Default versus explicit policy

| Behavior | Before | Default now |
| --- | --- | --- |
| Model tools | All permitted registered tools | ipython only |
| Python rlm | Synchronous helper returning a dict | Await admission; retained typed handle; no wait for answer |
| Baseline on greeting | Already absent in default workspace chat | Still absent; coding preparation only when explicitly configured |
| Completion | Autonomous/child used finish; interactive text yielded | Normal text ends autonomous request; text yields interactive/heartbeat; explicit goal completion |
| Coding verifier | Explicit coding adapter only | Still opt-in; not universal |
| Child cwd | Coding adapter isolated by default | rlm shares cwd; isolation explicitly requested |
| Experiments | Optional registered direct tools | Optional Python capabilities, never a mandatory sequence |
| Context | Coding-focus selection plus recent context | No coding-focus injection; full task/history programmatically readable |

`await goal.create(objective, token_budget=...)` persists an optional additional
goal budget including descendants; it cannot increase root limits. Ordinary final
text is not enough to complete an explicit goal. `await goal.complete()` requests
completion subject to configured task gates. Heartbeat text yields without
destroying the recurring session. Waiting for all children is an explicit
`task.wait_for_children` gate, not the default.

## Deliberate boundaries and remaining differences

The completed-child follow-up and interactive `/refine` fixes are behavioral bugs,
not reasons to change other policies. Model-controlled SQLite FTS5 history search,
ripgrep/Python repository search and arbitrary Python processing are retained.
Refinement defaults, global-state write permissions, context sizing, checkpoint
limits and run budgets are Threadweave implementation choices and are unchanged.
An unspecified paper detail is not evidence that these choices are defects.

This is a source-traced implementation of the default Python control plane, not
byte-for-byte API equivalence to every reference extension. Important limits:

- Python executes trusted host code. Tool path/permission checks are not a Python
  sandbox. Shell processes use the configured environment allowlist, not arbitrary
  kernel environment mutations. Bash waits for its owned process group; the
  reference's foreground-shell fence/detached-descendant distinctions are not reproduced.
- MCP supports stdio and streamable HTTP with environment-backed credentials;
  interactive MCP OAuth provisioning is not implemented. Codex subscription OAuth
  remains handled by the existing official Codex authentication integration.
- SKILL.md package dependencies are explicitly installed by the operator/model,
  not automatically installed during bootstrap. Documented callable invocations
  are ordinary Python execution.
- `rlm.delete_subagent` cancels/unloads a child and preserves durable history and
  topology instead of deleting its registry entry. Observation is limited to
  related sessions, not every unrelated session owned by the daemon.
- Container-only configurations cannot run a host Python control plane. Choose a
  separately isolated host/VM for untrusted Python; do not call local execution sandboxed.

The default exposes one model control plane. Runtime correctness is verified by
unit and integration tests; capability evaluation is documented in [evaluation](evaluation.md).
