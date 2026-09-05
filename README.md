# Threadweave

A persistent coding-agent harness for real Git repositories. A model chooses how
to search, edit, execute commands, delegate, and experiment. A separate verifier
checks the resulting repository against configured tests and constraints before
accepting completion.

Python 3.11+ on macOS/Linux. Local execution requires trusted code and tool access.
The default provider uses your existing ChatGPT/Codex subscription login.

## Configure a real model and run

Install the harness and the target repository's dependencies first:

```sh
uv sync --extra dev
uv run threadweave auth status
# Only if not already signed into Codex:
uv run threadweave auth login
# Once per client revision: requires git and Rust/cargo, downloads/builds official libraries.
uv run threadweave auth install-client

uv run threadweave --data /absolute/path/to/agent-state doctor --config configs/coding.json
uv run threadweave --data /absolute/path/to/agent-state run \
  "Fix the failing parser tests without weakening the tests." \
  --workspace /absolute/path/to/repository \
  --config configs/coding.json --attach
```

[configs/coding.json](configs/coding.json) and [configs/kernel.json](configs/kernel.json)
use `codex_subscription`. No `OPENAI_API_KEY` is needed. `auth models` lists the
account's current model catalog. Omit `provider.model` to use Codex's configured/default
model, or select one explicitly with `--model`. Model and reasoning settings are
resolved and stored before CLI run admission. `provider.parameters.reasoning_effort`
can override the Codex setting. There is no automatic API-billing fallback.

Authentication uses the official app-server account protocol. Inference uses the
official Codex Rust Responses client and auth manager, pinned to a source revision.
The bridge makes one model request: it creates no Codex agent or conversation and
executes no tools. Threadweave supplies the context and schemas and executes returned
structured calls. Codex owns credentials and refresh; Threadweave never exports tokens.
`auth logout` signs out of the **shared Codex login**, not just Threadweave.

The subscription backend currently rejects a server `max_output_tokens` parameter.
Threadweave therefore applies a client-observed output-byte guard (four bytes per
configured output token), plus timeouts and cumulative accounting. This is **not a
hard server token cap**; hidden reasoning/in-flight usage can exceed reservations.
Reported usage is authoritative; interrupted usage is marked estimated. Subscription
monetary cost is `null`, and dollar budgets/API prices are rejected for this provider.
See [subscription transport](docs/subscription.md) for installation and limitations.

The optional `chat` provider remains available for intentionally configured API
usage; see [optional API configuration](docs/subscription.md#optional-api-provider).

Configure repository-specific argv arrays in task.test_commands, build_commands,
lint_commands and typecheck_commands. Empty groups use available Python/Cargo/Go/npm
entry-point detection. Missing tests fail preparation unless explicitly configured
or require_tests is deliberately disabled. The default baseline requires a clean
Git working tree. Dependencies are not installed automatically.

## Repository intelligence and precise edits

The persistent incremental index records hashes, languages, symbols and imports,
excluding common generated/vendor directories. Tools include repo_map, repo_search,
symbol_search, references_search, file_outline and dependency_context. Python uses
AST parsing; other supported languages use lexical extraction. References are
likely usages, not a compiler-grade call graph. Search uses ripgrep when available
with a Python fallback.

Editing tools provide strict unified patches, hash-checked line replacement,
creation/deletion/moves, diffs and rollback. Patches validate before mutation;
journals retain pre/post hashes, prior contents and patch artifacts. Interrupted
edits recover or pause on conflicting external changes. Git checkpoints do not
commit or reset the user's repository.
Direct Python/process coding actions also receive before/after checkpoints and
workspace-effect records, including interrupted effects observed during recovery.

## Real execution and independent verification

Before model actions, the coding adapter captures Git state, a file checkpoint,
configured test/build results and optional benchmark measurements. Calling finish
only requests completion. The verifier reruns commands and checks allowed/forbidden
paths, required files, test deletion, optional test protection, baseline regressions,
nonempty changes and benchmark thresholds. Failure returns evidence for another turn.

run_tests, run_targeted_tests, run_build, run_lint and run_typecheck retain exit
status, timing, full stdout/stderr and bounded parsed diagnostics. failure_localize
connects evidence to source definitions, likely references and recent edits. Passing
a weak configured verifier is not proof of arbitrary task correctness.

Local execution is trusted-host execution, not a sandbox. Docker/Podman commands
can use a private workspace mount, read-only container root, resource limits and
network restrictions. Container-only configurations must omit Python permission:
host REPL workers are not sandboxed by the command executor.

## Persistent sessions and isolated coding children

The detached daemon owns sessions independently of clients. Ctrl-C detaches without
cancelling work. SQLite preserves session IDs, recursive relationships, messages,
action journals, goals, contexts, accounting and versioned state.

Python variables persist across turns. Supported codecs and explicit reconstruction
recipes recover state after restart; unsupported objects are reported. Full history
survives compaction and is searchable through FTS5. Model-generated compaction
retains structured facts and provenance, with a recorded extractive fallback.

Coding children receive private Git copies of the parent's captured working state,
including uncommitted inputs. Parents explicitly inspect/apply candidate patches;
nothing merges automatically. Child usage, findings, verification, patch production,
consumption and acceptance are recorded.

```sh
uv run threadweave --data /absolute/path/to/agent-state list
uv run threadweave --data /absolute/path/to/agent-state tree SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state status SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state history SESSION_ID --limit 20
uv run threadweave --data /absolute/path/to/agent-state usage SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state pause SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state diff SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state verify SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state input SESSION_ID "Investigate the remaining failure."
uv run threadweave --data /absolute/path/to/agent-state resume SESSION_ID
uv run threadweave --data /absolute/path/to/agent-state attach SESSION_ID
```

stop cancels a tree; fork creates a separate continuation with explicit ancestry
and an isolated coding workspace. Paused sessions retain messages. Restart recovers
runnable sessions; reboot requires restarting the daemon or a service manager.

## Experiments, performance and refinement

Experiments are durable entities with hypotheses, source checkpoints, changes,
correctness commands, measured results, patch artifacts and conclusions.
experiment_create/run/result/compare/list expose them to the model;
threadweave experiments exposes them to a human.

Benchmarks execute correctness gates, warmups and repeated actual commands, storing
raw values, median, nearest-rank p95 and baseline comparisons. Direction, required
improvement and noise tolerance are explicit. [configs/kernel.json](configs/kernel.json)
expects real make build/correctness/benchmark targets; adapt commands and the metric
regex to your project. Optional ncu, nsys or configured profiler commands retain
reports. Ordinary coding requires no GPU.

Optional automatic refinement runs at configured intervals, verifier failures,
experiment conclusions and completion. Proposals need evidence IDs and intended
effects. Executable skills validate schemas, syntax and declared permissions, track
outcomes and quarantine repeatedly failing versions. Rollback appends a version.
Model weights and foundational policy are unchanged. Validation does not prove
generated code is safe.

## Evaluate supplied workloads

No benchmark dataset or score is bundled. Supply real repository issue,
long-context or kernel instances described in [evaluation docs](docs/evaluation.md):

```sh
uv run threadweave --data /absolute/path/to/eval-state eval /path/to/tasks.jsonl \
  --config configs/coding.json --repetitions 3 --seed 42 --output /path/to/results.jsonl
uv run threadweave analyze /path/to/results.jsonl
```

Each isolated run records its resolved config, verifier outcome, patch and resource/
trajectory metrics in JSONL and SQLite. Feature flags support ablations. Seeds are
run labels, not claimed deterministic remote-model seeds.

## Development and boundaries

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

Providers used for deterministic tests live only under tests/. Tests execute real
temporary repositories, test commands, patches, measurements and daemon death/
restart. HTTP transport tests use in-memory responses. No paid-model quality,
external benchmark score, or GPU performance is claimed by these tests.

Boundaries: single-host daemon; synchronous local indexing/Git metadata; lexical
non-Python navigation; full-copy candidates; text unified patches; no isolation for
host Python/processes; no exactly-once external effects; no statistical significance
claim from benchmark tolerance. Exact artifacts may contain repository secrets.

See [architecture](docs/architecture.md), [operations/security](docs/operations.md),
[extensions](docs/extensions.md), and [evaluation](docs/evaluation.md).
