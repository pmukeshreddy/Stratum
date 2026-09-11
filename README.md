# Threadweave / Buffalo

Buffalo is a persistent recursive agent harness. Its interactive Agents View
attaches to a daemon-owned root session. The model uses IPython for computation,
workspace tools, recursive children and messaging. Each session keeps its own
context, Python kernel, history and reusable learned state.

## Get started

Requires Python 3.11+, uv and Node.js 22.8+. Node runs the harness formatters and
JSON serialization. EmulatorBench and EvoCode setup use Python 3.12.

```sh
uv sync --extra dev --locked
uv run buffalo auth status
# If the shared Codex ChatGPT login is absent:
uv run buffalo auth login
uv run buffalo doctor --config configs/session.json
uv run buffalo
```

The default uses the current workspace and `configs/session.json` when available.
Use `--workspace` or `--config` to override them; coding-specific tools use
`--config configs/coding.json`. The normal provider reuses the shared Codex login.
`auth models` lists available models; provider and reasoning settings are described
in [subscription configuration](docs/subscription.md).

```text
> Inspect this project and fix the failing tests.
> /tree
> /usage
> /compact
> /exit
```

`/help` lists commands. Enter sends; Alt-Enter/Ctrl-J inserts a newline. Input stays
available during work. Ctrl-C pauses the turn; Ctrl-D and `/exit` detach.
`buffalo --continue` reattaches to the most recent root in this workspace;
`--resume SESSION_ID` chooses a session. Attachment preserves its mode and budgets.

## Runtime

```text
Human ↔ Agents View ↔ Root session ↔ Environment
                          ↕ rlm / messages
                     Recursive children
                          ↕
                 Daemon / continual harness

L1: selected model context
L2: persistent Python state and recursive sessions
L3: disk-backed history, artifacts and learned state
```

The model chooses actions through the same persistent Python environment:

```python
files = list(workspace.rglob("*.py"))
child = await rlm("Inspect persistence and report concrete defects.", name="reviewer")
result = await bash("uv run pytest -q")
await agent_message.send("Include recovery edge cases.", receiver_role="child", receiver_name="reviewer")
```

`rlm()` returns a stable handle after admission; the parent can keep working while
the child runs. Children have independent contexts and kernels and can delegate
within shared limits. Follow-up work can resume the same child session.

Compaction reduces active model context while retaining history and recoverable
Python state. Unsupported Python objects need reconstruction recipes. The
continual harness stores prompt notes, memories, skills and subagent specifications.
Explicit `await refine.run()` and automatic review use the same safe application
boundaries; see [refinement](docs/refinement.md).

`run --mode autonomous` continues within configured limits; `--mode goal` retains
an objective and `--mode heartbeat` schedules turns. Turn, token, time, tool, depth
and concurrency limits account for descendants. Detach/resume does not reset them.

New chats use the XDG data directory (normally `~/.local/share/threadweave`), an
existing workspace `.threadweave` store, or explicit `--data`. Runtime records use
JSON/JSONL. Existing SQLite stores are imported once with the original preserved.
Restart an old daemon after upgrading its code.

## Evaluation

| Benchmark | Buffalo reported score |
| --- | ---: |
| ARC-AGI-3 | 81 |
| EmulatorBench | 25 |

These scores were supplied by the project owner. They are not recomputed by this
checkout; run reports retain the measured metric, task scope and provenance.
The EmulatorBench public-source score is separate from its official reward.

- [ARC-AGI-3 setup and scoring](docs/evaluation.md)
- [EmulatorBench setup, public-source verification and reports](docs/emulatorbench-evaluation.md)
- [EvoCode integration](docs/evocode-evaluation.md), whose resident worker is also used by EmulatorBench

Local results stay under `results/` and `.emulatorbench/runs/`, both ignored by Git.

## Development

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv build
```

Live provider, Docker and Modal tests are opt-in. The default suite uses test
providers and does not launch a capability evaluation.

Local Python and processes execute with the host user's authority. Container
command execution does not sandbox the host REPL. Back up durable state and treat
artifacts as private workspace data; arbitrary-object recovery and exactly-once
external effects are not guaranteed.

## Documentation

- [Architecture](docs/architecture.md) and [recursive execution API](docs/adaptive-orchestration.md)
- [Python control plane](docs/python-control-plane.md) and [workspace environment](docs/python-environment.md)
- [Configuration, storage and security](docs/operations.md)
- [Extensions](docs/extensions.md)
