# Reproduce hardening measurements and coding comparisons

## Local performance

```sh
uv sync --extra dev
uv run python -m tests.perf_mutation_observation --output /absolute/results/mutations.json
uv run python benchmarks/hardening.py --sizes 100 10000 30000 \
  --repetitions 3 --output /absolute/results/components.json
uv run python benchmarks/isolation.py --sizes 100 10000 \
  --output /absolute/results/isolation.json
```

These create temporary synthetic file corpora to measure components, not coding
ability. They invoke no model. Samples include machine metadata. Shared child
admission does not run the child model. Candidate timing includes checkpoint and
worktree admission; four candidates are admitted serially. Object storage is
shared, checkout bytes are reported, and cleanup is measured separately. Mutation
timings exclude cold preparation but include actual persistence/invalidation.
Related-test results depend on repository/test density and watcher backlog.

`isolation.py` separately measures one/four concurrent **backend** requests with
independent SQLite connections and a synchronization barrier. On this machine,
10,001 files (35.1 MB) took 1.10 s for one checkout and 1.26 s for four concurrent
checkouts, excluding 0.35–0.78 s checkpoint capture; checkout storage was 35.1/140.5
MB with shared Git objects. This does not imply parallel runtime admission.

## Real local historical suite

Two confirmed Threadweave bugs at commit `e5c8c41` are the coding tasks: completed
child follow-up/recovery and interactive refinement routing. They are historical
repository defects, not fabricated programs, and not a standard public benchmark.
Evaluator-only regression patches are withheld until independent final evaluation.
Those evaluator tests use model doubles to test APIs; the coding agent uses a real
subscription model. Do not confuse the two.

```sh
unset OPENAI_API_KEY
codex login status
uv run threadweave auth status
uv run python evals/historical/prepare.py /absolute/results/inputs
uv run python -m threadweave.frozen_eval freeze /absolute/results/inputs/tasks.json \
  --config /absolute/results/inputs/config.json --output /absolute/results/frozen
PYTHONPATH=/absolute/results/frozen/runtime uv run python -m threadweave.frozen_eval run \
  /absolute/results/frozen --output /absolute/results/comparison
```

Use new destinations. Freeze refuses an existing bundle and hashes exact Python/
Rust sources, dependency declarations, task/config/evaluator files and base commit.
Workers pin their Python source root to the frozen runtime rather than accidentally
importing a developing editable checkout. Inputs refer to a local source repository;
keep its Git objects available. Runtime code can be developed while the frozen
comparison runs. Dependencies are recorded; reproduce with the frozen lock too.

Both profiles receive the same task, base revision, provider/model/settings,
context policy, independent verifier and limits. `base` is this runtime's basic
filesystem/shell ablation, with repository index/retrieval/subagents/experiments/
automatic refinement disabled. It is **not** an independent raw-agent implementation.
Buffalo exposes its configured Python helpers, never a larger default model schema.
Repetition ordering alternates; seeds are labels unless supported by the provider.

Current sample config: Codex subscription, `gpt-6-astra`, reasoning `low`, 4096
maximum output-token guard, 20 turns, 120,000 total tokens, 600 seconds/task,
60,000 context ceiling, at most two children, concurrency three. Limits include
descendants. Subscription dollar cost is null. No API billing fallback exists.

Accepted solve requires runtime completion **and** independent verifier pass.
A budget-limited patch passing withheld tests is reported separately, not silently
counted as an accepted completed task. Setup failures before any model invocation
are setup failures, not coding scores. The first v2 preparation missed explicit
process permission and was corrected before v3 real runs.

Each evaluation retains `events.jsonl`, `analysis.json`, `final.patch`, repository,
`state/history.sqlite3`, artifacts, config/source identity and final result.
`comparison.json` and per-profile JSONL summarize actual measurements. Actions
inside arbitrary Python cannot all be classified: raw reads/searches/test commands
are not reliable tool counters. No metric fabricates unknown costs or causal
retrieval usefulness. Tiny samples cannot establish superiority.

## External harness adapter

`threadweave eval ... --profile external --external-command PROGRAM [ARGS...]`
starts an explicit adapter with protocol-1 JSON on stdin: objective, workspace,
provider config and resource limits. It returns JSON with `completed` and optional
`usage`/trajectory paths. The evaluator runs its own final verifier. External
usage is labeled self-reported; stderr and stdout are retained, JSON is capped at
4 MB, and timeout/cancellation kills the process group. No Prime adapter has been
executed here and no comparison against Prime is claimed.

Local execution is trusted-host execution, not secure benchmark isolation. Hidden
patches are not intentionally in model context or its workspace, but an unrestricted
host process could inspect other host files. Use OS/container isolation for hostile
or publishable hidden-test evaluation. Existing public-workload adapters still
require externally supplied actual datasets; this work reports none as newly run.
