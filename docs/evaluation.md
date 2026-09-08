# ARC-AGI-3 comparison

The current evaluation phase runs **ARC-AGI-3 only**. ManyIH Coding, ManyIH
Instruction Following, LongBench v2, and Factorio are paused. Their official
integration code remains available for future work; the current CLI cannot run them.

The comparison is `gpt-6-astra` with `xhigh` reasoning through the actual installed,
ChatGPT-authenticated **Codex app-server**, versus the production Buffalo Runtime.
The custom BASE chat/tool loop has been removed. Codex retains its own prompt,
agent loop, native tools, and delegation. A loopback transport proxy forwards
its subscription requests unchanged while enforcing shared inference admission
and recording usage; it does not implement an agent.

## Execute

Install with `uv sync --extra dev --locked`. Authenticate the installed `codex`
runtime using its normal login. Copy `configs/evaluation.example.json` outside the
checkout and set the official ARC source, commit, Python, and environment directory.
Install the [official ARC toolkit](https://github.com/arcprize/arc-agi) in that Python
environment and supply its 25 official local game environments.

```sh
# Setup only, with no inference
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --check

# Two-game validity check for both harnesses
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --limit 2 \
  --games-concurrency 4 --inference-concurrency 16

# Complete 25-game comparison, both harnesses
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json \
  --games-concurrency 16 --inference-concurrency 16
```

`--profile paired` is the default; `codex` and `buffalo` select one side explicitly.
Each output directory must be empty. Existing artifacts are preserved. A failed
setup or invalid execution reports its exact reason; there is no alternate benchmark.
The full run requires exactly 25 official environments. Subset reports are labeled.

For fixed-game protocol diagnostics, add `--protocol-validation /absolute/policy.json`
and `--limit 1` or `--limit 2`, with seed zero. The policy supplies `instructions`,
`guidance`, `continuation_prompt`, `max_continuations`, `max_turns`, `max_tokens`,
and `wall_seconds`. This mode exposes `observe`, `status`, and `act` through a
local Python client, limits each game to 500 actions and each batch to 20, and
disables delegation. Every continuation retains the same environment, workspace,
and native Codex thread or Buffalo session. Terminal observations stop further
inference admission. Infrastructure failures remain failed attempts without
automatic fresh-game retries.

Diagnostic snapshots retain official scores and actual cumulative usage at action,
continuation, and final boundaries. They neither impose output-token thresholds nor
claim to reproduce a published scaling curve. The continuation counter uses
successful root assistant responses and noncached input plus output tokens; the
separate resource ledger includes all calls, including auxiliary inference.
Codex's named permission profile allows only the assigned local game socket, with
an empty network-domain allowlist. Its normal agent and tools execute the task.

## Isolation, matching, and budgets

Every game attempt owns an official worker process, Arcade/environment instance,
scorecard, recordings, workspace, agent state, and result directory. Hashes verify
identical initial observations across harnesses. Observation checks before each
action detect changes outside that worker's action stream. Both harnesses receive
the same task and lossless observation encoding, environment version, seed, model,
reasoning level, cumulative token budget, model-call budget, tool/action budget,
and wall-clock budget.

At most 16 games run concurrently. A single inference gate admits every native
Codex request and every Buffalo root, descendant, compaction, and refinement call.
Transport instability reduces capacity with recorded decisions. The runner never
raises the configured capacity without a separate measured experiment.

Only infrastructure-invalidated attempts can retry, up to three attempts total.
A completed low score and its healthy comparison counterpart are never rerun.
Failed attempts retain their trajectories and separate usage accounting.

## Scoring and provenance

RHAE is computed by the official SDK: disjoint game scorecards are merged, then
`EnvironmentScorecard.from_scorecard` produces the aggregate. Scores are never
replaced by level counts, custom averages, or selected best attempts.

Each run records the official source commit, environment file hashes, game IDs,
starting-state hashes, resolved configuration and budgets, Buffalo source identity,
Codex binary version/hash, UTC timestamps, raw official scorecards, recordings,
and trajectories. Provider journals retain request provenance while opaque
reasoning continuation is stored privately in Buffalo's durable state.

Resource accounting includes input/output/total tokens, model calls, and elapsed
time for the complete agent tree. Interrupted requests without final provider usage
are explicitly marked as estimated; they do not disappear. Subscription API cost
is unavailable and remains null. Final-attempt usage, failed-attempt usage, and
all-attempt totals are distinct. Profile elapsed time and summed game durations
are both retained.

Large run directories are local and ignored by Git. Previously stopped runs remain
provenance only and are not evidence for this Codex comparison. GitHub Actions runs
engineering checks only; it never executes capability benchmarks or inference.
