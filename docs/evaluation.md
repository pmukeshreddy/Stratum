# ARC-AGI-3 comparison

The primary comparison holds **Astra XHigh** (`gpt-6-astra`, `xhigh` reasoning) fixed:

| Harness | Model | Reported score |
| --- | --- | ---: |
| ARC Standard harness | Astra XHigh | ≈ 59% |
| Buffalo harness | Astra XHigh | ≈ 81% |

**59 → 81: approximately +22 percentage points.** Buffalo improves the same
underlying model by replacing the ARC Standard harness.

The rounded values were supplied by the project owner. The underlying score
artifacts for these values are not present in this checkout. The ARC Standard
harness is external; no implementation of that harness is bundled here.

## Run Buffalo

The current evaluation CLI runs **ARC-AGI-3 only**. LongBench v2 and Factorio
integrations remain paused. No replacement dataset is configured yet.

Install with `uv sync --extra dev --locked`. Authenticate the normal subscription
provider as described in [subscription transport](subscription.md). Copy
`configs/evaluation.example.json` outside the checkout and set the official ARC
source, commit, Python, and environment directory. Install the
[official ARC toolkit](https://github.com/arcprize/arc-agi) in that Python
environment and supply its 25 official local game environments.

```sh
# Setup only, with no inference
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --check

# Two-game Buffalo validity check
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --limit 2 \
  --games-concurrency 4 --inference-concurrency 16

# Complete 25-game Buffalo run
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json \
  --games-concurrency 16 --inference-concurrency 16
```

Each output directory must be empty. A failed setup or invalid execution reports
its exact reason. The full run requires exactly 25 official environments. Subset
reports are labeled. No profile selector is needed: the CLI runs Buffalo.

To retain the fixed-game protocol across subset checks and the full run, add
`--protocol-policy /absolute/policy.json` with seed zero. Use `--limit 2` for the
small check and omit `--limit` for all 25 games. `--protocol-validation` remains an
alias. The policy supplies `instructions`, `guidance`, `continuation_prompt`,
`max_continuations`, `max_turns`, `max_tokens`, and `wall_seconds`. This mode exposes
`observe`, `status`, and `act` through a local Python client, limits each game to
500 actions and each batch to 20, and disables delegation. Every continuation
retains the same environment, workspace, and Buffalo session. Terminal observations
stop further inference admission. Infrastructure failures remain failed attempts
without automatic fresh-game retries.

Diagnostic snapshots retain official scores and actual cumulative usage at action,
continuation, and final boundaries. They neither impose output-token thresholds nor
claim to reproduce a published scaling curve. The continuation counter uses
successful root assistant responses and noncached input plus output tokens; the
separate resource ledger includes all calls, including auxiliary inference.

## Isolation and budgets

Every game attempt owns an official worker process, Arcade/environment instance,
scorecard, recordings, workspace, agent state, and result directory. Hashes verify
identical initial observations across retries. Observation checks before each
action detect changes outside that worker's action stream. The run records the
task, lossless observation encoding, environment version, seed, model, reasoning
level, cumulative token budget, model-call budget, tool/action budget, and wall-clock
budget. An independently reproduced ARC Standard comparison must match these
conditions before attributing a measured difference to the harness.

At most 16 games run concurrently. A single inference gate admits every Buffalo
root, descendant, compaction, and refinement call. Transport instability reduces
capacity with recorded decisions. The runner never raises the configured capacity.

Outside fixed-game mode, only infrastructure-invalidated attempts can retry, up to
three attempts total. Completed low scores are never rerun. Failed attempts retain
their trajectories and separate usage accounting.

## Scoring and provenance

RHAE is computed by the official SDK: disjoint game scorecards are merged, then
`EnvironmentScorecard.from_scorecard` produces the aggregate. Scores are never
replaced by level counts, custom averages, or selected best attempts.

Each run records the official source commit, environment file hashes, game IDs,
starting-state hashes, resolved configuration and budgets, Buffalo source identity,
UTC timestamps, raw official scorecards, recordings, and trajectories. Provider
journals retain request provenance while opaque reasoning continuation is stored
privately in Buffalo's durable state.

Resource accounting includes input/output/total tokens, model calls, and elapsed
time for the complete agent tree. Interrupted requests without final provider usage
are explicitly marked as estimated. Subscription API cost is unavailable and remains
null. Final-attempt usage, failed-attempt usage, and all-attempt totals are distinct.
Profile elapsed time and summed game durations are both retained.

Large run directories are local and ignored by Git. GitHub Actions runs engineering
checks only; it never runs capability benchmarks or inference.
