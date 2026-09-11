# ARC-AGI-3

The project owner's reported Buffalo score is **81%** with Astra XHigh
(`gpt-6-astra`, `xhigh`). The underlying scorecards for that reported value are
not bundled in this checkout. Fresh runs record their own measured results.

## Setup and execution

Install Buffalo with `uv sync --extra dev --locked` and authenticate its normal
subscription provider. Copy `configs/evaluation.example.json` outside the checkout
and set the official ARC source checkout, commit, Python interpreter and environment
directory. Install the official ARC toolkit in that interpreter and supply its
25 official local game environments.

```sh
# Validate setup without inference.
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --check

# Explicit two-game subset.
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json --limit 2 \
  --games-concurrency 4 --inference-concurrency 16

# Full 25-game run.
uv run buffalo eval arc-agi-3 --config /absolute/evaluation.json \
  --games-concurrency 16 --inference-concurrency 16
```

Use `--output /absolute/new-directory` to choose a fresh output directory; the
default is `results/evaluation/arc-buffalo-*`. Output directories must be empty.
Setup failures retain their reason, and subset reports are labeled explicitly.

## Protocol and isolation

`--protocol-policy /absolute/policy.json` selects a fixed-game policy with seed
zero; `--protocol-validation` is an alias. The policy supplies `instructions`,
`guidance`, `continuation_prompt`, `max_continuations`, `max_turns`, `max_tokens`
and `wall_seconds`. Use the same policy for subset and full runs.

Fixed-game mode exposes `observe`, `status` and `act` through a local Python client,
limits each game to 500 actions and each batch to 20, and disables delegation.
Continuations retain the same environment, workspace and session. Terminal states
stop further inference admission. Infrastructure failures remain failed attempts.
Outside fixed-game mode, only infrastructure-invalidated attempts can retry, up to
three attempts total; completed low scores are never rerun.

Each attempt owns an official worker, Arcade instance, scorecard, recordings,
workspace and agent state. Initial-state hashes and observation checks detect
unintended environment changes. One inference gate admits root, descendant,
compaction and refinement calls, with a maximum configured concurrency of 16.

## Scoring and artifacts

The official SDK merges disjoint game scorecards and computes RHAE using
`EnvironmentScorecard.from_scorecard`. Level counts, custom averages and selected
best attempts do not replace the official score.

Run reports retain source commits, environment hashes, game IDs, seed, starting
states, model/reasoning configuration, budgets, timestamps, raw official scorecards,
recordings and trajectories. Diagnostic snapshots preserve scores and cumulative
usage at action, continuation and final boundaries.

Accounting separates final-attempt, failed-attempt and total usage. It includes
root, descendant and auxiliary calls, input/output tokens and elapsed time.
Interrupted calls with incomplete provider usage are marked as estimated.
Subscription API cost remains null. Local run directories are ignored by Git.

See [EmulatorBench](emulatorbench-evaluation.md) for the other reported benchmark.
