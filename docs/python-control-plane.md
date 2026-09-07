# Python execution semantics

## Source trace and changed request boundary

The reference was the attached source archive `prime-agent-main.zip`, archive
commit `d74a75fea3411136fdd2ba95c7f723ddefdadf05`. The following source paths were
traced, not inferred from its README. Paths in this table are relative to that
archive, not files shipped in Threadweave. No reference code/package was vendored.

| Reference source | Executed behavior | Threadweave implementation |
| --- | --- | --- |
| `packages/coding-agent/src/core/tools/index.ts:47`, `sdk.ts:238`, `agent-session.ts:9292` | Default active tool is ipython only | `ToolRegistry.schemas`, `_execute_action` provider-origin enforcement |
| `packages/coding-agent/src/core/tools/ipython.ts:35` | Session bootstrap with async recursion, shell, MCP, Python skill modules | `kernel_api.bootstrap`, `kernel_worker.Worker` |
| `prime-agent-runtime/src/rlm/__init__.py:92`, `agent-session.ts:10565` | Async child admission returns stable handle before detached execution | `Recursive.run`, `host_api.dispatch`, `Runtime.spawn` |
| `agent-session.ts:9590` | Child inherits cwd/environment, not parent Python variables | Shared workspace + independent kernel/session, optional isolation separate |
| `packages/coding-agent/skills/agent-message/src/agent_message/__init__.py` | Role/name-addressed persistent messages; explicit child answers | `Messaging`, daemon-mediated durable queue |
| `packages/coding-agent/skills/agent-observe/src/agent_observe/__init__.py` | Bounded session and message inspection | `Observation`, scoped host dispatch |
| `prime-agent-runtime/src/rlm/bash.py` | Immediate process handle, await/poll/output/kill, background lifetime | `Bash`, `BashHandle`, `BackgroundProcesses` |
| `packages/coding-agent/skills/edit/src/edit/__init__.py:8` | Unique exact text replacement relative to kernel cwd | `Edit.run`, host path validation and existing Editor journal |
| `packages/coding-agent/src/core/skills.ts`, `tools/ipython.ts` | SKILL.md discovery and src-layout Python modules, callable if run exists | `skills.discover/load_module`, bootstrap and explicit skills API |
| `prime-agent-runtime/src/rlm/harness.py`, `core/refinement/refinement.ts:429` | Immediate local/global durable CRUD, compact supplemental menu | `Harness`, validated versioned state, explicit content selection |
| `prime-agent-runtime/src/rlm/mcp.py`, `mcp_base.py:307` | Lazy stdio/streamable HTTP discovery/calls/lifecycle; structured content preferred | Official Python MCP SDK in `mcp_client`, kernel normalization |
| `packages/coding-agent/src/core/prompts/rlm.ts:79` | Normal final answer ends work; conversation log accessible by path | Normal-text completion/yield, readable history projection |

Before this change, at Threadweave commit
`62ba2524a82b7b2a8228cf23484a6b45564c077a`, `_invoke` supplied all 63 permitted
tool schemas from `builtins().schemas(configs/session.json)`. The exact capture is
[tool-schema-before.json](../results/parity/tool-schema-before.json).

After: `_invoke` still uses the same production provider interface, but supplies
only [ipython](../results/parity/tool-schema-after.json), with one required string
argument `code`. The live test captures the actual stored provider request artifact
for **every** invocation and checks this list, not just a registry mock. A model
attempting a direct `workspace_write` action is rejected as `tool_not_exposed`.

```text
BEFORE                              AFTER (default)
MODEL                               MODEL
 ├─ coding tools                     ↓ ipython(code)
 ├─ orchestration/state tools        persistent IPython kernel
 ├─ finish                           ↓ programmatic capabilities
 └─ python                          Environment / daemon / continual state
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
files = list(workspace.rglob('*.py'))
matches = repo.search('Lifecycle')
outline = repo.outline('src/threadweave/runtime.py')
await edit('notes.txt', 'one unique old passage', 'replacement')
diff = git.diff()
job = bash('uv run pytest -q')
print(job.pid, job.running)
result = await job
print(result.exit_code, result.output[-1200:])

review = await rlm('Inspect persistence; explicitly message your findings.', name='review')
local_value = 123  # executes without waiting for review to finish
status = await agent_observe.get_agent(review.session_id)
await agent_message.send('Check restart handling.', receiver_role='child', receiver_name='review')
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
entry = harness.create_memory('Observation', 'Retained fact', id='observation')
harness.update_memory('observation', 'Observation', 'Corrected fact')
harness.rollback('memory', 'observation', 1)
harness.select([entry.id])

procedure = harness.create_skill('Increment', {
    'name': 'increment', 'description': 'Increment the retained counter',
    'code': 'counter += 1\nskill_result = counter',
    'required_permissions': ['python'],
})
counter = 0
answer = await skills.run('increment')
```

Memory/prompt/subagent entries use text/instruction contents. Explicit CRUD commits
immediately, including within the same Python cell. Versions carry source event,
author, intended effect, deletion and rollback provenance. Optional global writes
require `refinement.allow_global_writes`; foundational policy is not editable.
Automatic refinement remains optional and applies validated proposals at safe
boundaries. `await refine()` requests that pass; it does not fabricate new state.

Discover SKILL.md under workspace `.agents/skills`, `.threadweave/skills`, user
`$XDG_CONFIG_HOME/threadweave/skills`, and configured `skill_paths`. Python skill
packages contain `pyproject.toml` and `src/<name_underscored>/__init__.py`. Bootstrap
imports permitted packages; modules defining `run` are also callable. Markdown
skills are readable instructions, not executable blobs. `skills.list/load/run`
expose discovery, reading/import and execution. Install declared dependencies in
the kernel environment explicitly; import failures remain visible. Skill module
and source hashes detect updates. Stored Python references accept import/callable
and default arguments; executable code/input schemas/permissions are validated.
`skills.run` records outcomes and quarantines repeatedly failing versions.

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
schemas = await mcp.list_tools('local')
value = await mcp.call_tool('local', 'lookup', {'query': 'evidence'})
await mcp.reload('local')
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
  not automatically installed during bootstrap. `skills.run` tracks outcomes;
  direct imported-module calls are ordinary Python execution.
- `rlm.delete_subagent` cancels/unloads a child and preserves durable history and
  topology instead of deleting its registry entry. Observation is limited to
  related sessions, not every unrelated session owned by the daemon.
- Container-only configurations cannot run a host Python control plane. Choose a
  separately isolated host/VM for untrusted Python; do not call local execution sandboxed.

These differences remain explicit; the default no longer has two parallel model
control planes. [Acceptance evidence](../results/parity/REPORT.md) separates actual
live results from deterministic/component tests.
