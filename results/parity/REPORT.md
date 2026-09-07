# Execution-semantics change and acceptance evidence

Reference source archive commit: `d74a75fea3411136fdd2ba95c7f723ddefdadf05`.
Threadweave starting commit: `62ba2524a82b7b2a8228cf23484a6b45564c077a`.
The implementation is the new commit containing this report. Source tracing and
remaining reference differences are documented in [Python control plane](../../docs/python-control-plane.md).

## Model request

Before: 63 model-facing tools, including direct repository/edit/test/experiment,
orchestration, state, completion and Python tools.
After: one `ipython(code)` tool. All 39 actual live provider requests were inspected
and contain only ipython. Provider-origin direct capability calls are rejected.

- [Complete BEFORE schema](tool-schema-before.json)
- [Complete AFTER schema](tool-schema-after.json)

The AFTER snapshot equals the live provider request exactly. Useful registered
capabilities remain accessible through Python; explicit legacy `control_plane:
direct` is not the default.

## Live subscription run

Executed 2026-09-07 with OPENAI_API_KEY unset, existing ChatGPT login,
`codex-cli 0.153.4`, `uv 0.11.8`, Python 3.12.12, Darwin 25.3.0 arm64 / Apple M4.
Provider: `codex_subscription`; resolved model: `gpt-6-astra`; reasoning: low;
streaming enabled; configured output reservation: 4096 (not a server-enforced cap).
No deterministic provider, Codex agent execution or API billing fallback was used.

The test ran the production CLI against this real repository, using the working
implementation, before the final documentation and optional-policy regression
checks. It did not claim to be a benchmark or a frozen-commit performance result.

| Measurement | Actual result |
| --- | --- |
| Acceptance | PASS |
| Root | `4034ae59b63847bc8a27f49526cdc90e` |
| Persistent children | 2, separate kernels, shared workspace |
| Model calls | 39 |
| Input / output / total tokens | 156,437 / 3,101 / 159,538 |
| Tool calls / Python executions | 31 / 23 |
| Elapsed test wall time | 183.13 seconds |
| Recursive execution seconds (can overlap) | 224.54 seconds |
| Compactions | 1 |
| Retries / harness failure events | 0 / 0 |
| Verifier calls / forced coding baseline | 0 / none |
| Subscription monetary cost | null, not API pricing |

Verified from Python results and durable events:

1. `hi` produced a normal reply, with no coding baseline/tests/build/checkpoint.
2. Python retained `x = 123`; a later invocation printed the existing value.
3. Python inspected this repository and retained 76 source/test paths in `py_files`.
4. Python searched lifecycle code, retaining `lifecycle_evidence`.
5. Two `await rlm(...)` calls returned persistent handles. The parent printed
   `PARENT_CONTINUED` before either child completed: 27.256 and 33.443 seconds earlier.
6. Each child inspected actual files in its own kernel, retained an observation,
   and explicitly sent an evidence-backed parent message.
7. `await bash('uv run pytest -q tests/test_python_control.py tests/test_context.py')`
   executed real tests: exit 0, **9 passed in 2.95s** at that point in the working tree.
8. `/compact` retained L2 values/handles and L3 messages/history. Assertions printed
   `PARITY_STATE_OK`.
9. Detach left the daemon alive. `--continue` reopened the same root and printed
   `REATTACH_OK` after checking the existing values.
10. The test killed the daemon, reopened it with `--continue`, checked recovered
    values/handles/messages without redefining them, and printed `RECOVERY_OK`.

The imported `Counter` class was correctly reported as requiring a recovery recipe;
it was not falsely serialized. The required values and both typed handles recovered.

### Exact retained evidence

Committed, reviewed evidence:

- [Terminal transcript](live-final-20260907/terminal.scrollback.txt), rendered from
  the original terminal bytes without editing the conversation.
- [Run configuration, IDs, prompts and usage](live-final-20260907/summary.json).

Full local evidence remains at `results/parity/live-final-20260907/`:

- `terminal.raw.txt`: original terminal bytes.
- `events.jsonl`: all **923** root/child events.
- `provider-requests.json`: all **39** actual request payloads.
- `config.json`, `tool-schema-after.json`.
- `state/history.sqlite3`, `state/artifacts`, `state/kernels`, `state/processes`:
  durable history, full request/output artifacts, checkpoints and process logs.

Private runtime state and unreviewed full request/artifact contents are retained
locally, not committed to Git.

The earlier `live-20260907` attempt remains recorded as **failed**: its exporter
requested 10,000 events from a recent-history API capped at 500 and omitted an
early assertion marker. `Store.iter_events` now paginates the complete log; a
1,250-event regression test checks this. The successful run used a fresh directory.

## Verification

Before changes: **198 passed, 4 skipped**.

Final checks after production changes:

```text
uv run pytest -q
213 passed, 5 skipped, 1 warning in 48.22s

uv run ruff check src tests
All checks passed!

uv run ruff format --check src tests
76 files already formatted

uv build
Successfully built dist/threadweave-0.2.0.tar.gz
Successfully built dist/threadweave-0.2.0-py3-none-any.whl

git diff --check
passed
```

Separate live run: **1 passed in 183.13s**, three macOS forkpty deprecation
warnings. The full suite's five skipped cases are opt-in live tests; the new live
acceptance was executed separately. The ordinary suite's warning is the same
pexpect/forkpty warning, not a test failure.

New component tests use real workers, temporary Git repositories, real shell/test
processes and actual MCP SDK servers over stdio **and** HTTP. Model doubles are
test-only. Tests cover request filtering, forbidden direct calls, async children,
shared/explicitly isolated workspaces, messages, IPython magic/background tasks,
process cancellation/cleanup/timeouts, relative edits, history export, skill
bootstrap/import errors/execution/reference/versioning/rollback, observation,
MCP permissions/redaction/errors/lifecycle, goal budgets and heartbeat yielding.

## Production file manifest

Added:

```text
src/threadweave/background.py
src/threadweave/host_api.py
src/threadweave/kernel_api.py
src/threadweave/mcp_client.py
src/threadweave/skills.py
```

Changed:

```text
src/threadweave/chat.py
src/threadweave/context.py
src/threadweave/daemon.py
src/threadweave/environment.py
src/threadweave/kernel.py
src/threadweave/kernel_worker.py
src/threadweave/migrations.py
src/threadweave/models.py
src/threadweave/refinement.py
src/threadweave/runtime.py
src/threadweave/storage.py
src/threadweave/terminal.py
src/threadweave/tools.py
configs/session.json
pyproject.toml
uv.lock
```

No production capabilities were deleted. Coding tools moved out of the default
model-facing schema. Schema migrations preserve existing state, adding durable
process jobs and goal budgets.

## Reproduction

```sh
uv sync --extra dev
env -u OPENAI_API_KEY uv run threadweave auth status
env -u OPENAI_API_KEY uv run threadweave
# Later: uv run threadweave --continue

# Fresh directory required; actual subscription calls:
env -u OPENAI_API_KEY THREADWEAVE_LIVE_PARITY=1 \
  THREADWEAVE_PARITY_OUTPUT=/absolute/path/to/fresh-parity-run \
  uv run pytest -s -q tests/test_python_control_live.py
```

The default parallel control planes are removed. Exact parity with every advanced
reference behavior is **not claimed**: see the explicit remaining differences in
shell foreground handling/environment inheritance, MCP OAuth provisioning,
automatic skill dependency installation and deletion/observation scope in the
[implementation notes](../../docs/python-control-plane.md#deliberate-boundaries-and-remaining-differences).
