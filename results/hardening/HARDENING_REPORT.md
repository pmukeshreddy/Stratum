# Hardening result — 2026-09-07

## Executive result

Research harness with stronger measured components, **not demonstrated
production-competitive coding performance**. Every requested stage was addressed
in implementation/tests/measurement or with an explicit remaining limit. This is
not a claim that the entire production-hardening completion standard has been met.

The default provider surface is still only `ipython(code)`; the live trajectories
also used only that model action. Persistent kernels, unrestricted Python,
Environment policy/audit routing, SQLite durability and independent coding
verification remain. No generic agent framework or API-billing fallback was added.

## Implementation record

[Stage-by-stage before/after, exact production files, tests and limitations](../../docs/hardening-ledger.md).
[Python Environment observation](../../docs/python-environment.md).
[Reproduction commands](../../docs/hardening-evaluation.md).

Major new production modules: `change_tracking.py`, `syntax_index.py`,
`tokenization.py`, `test_evidence.py`, `test_selection.py`, `snapshots.py`,
`machine.py`, `frozen_eval.py`, `trajectory_analysis.py`. Step 1's `mutations.py`
was extended. Existing runtime, repository, context, execution, coding/verifier,
Git, experiments, migrations and Python host/kernel modules were improved rather
than replaced. Migrations advance existing databases to version 8 without wiping
trajectories.

## Tests and build

Focused hardening: 32 passed. Step 1 focused Environment suite: 34 passed.
Full suite: 313 passed, 5 skipped. Skips: four opt-in live acceptance tests and
Docker engine/preinstalled-image availability. Two existing forkpty deprecation
warnings remain. Daemon/socket and HTTP MCP tests ran in this environment.
The separate live coding runs below are new verification, not those skipped tests
or older reports. Ruff check, format check, diff whitespace check and package build
pass; wheel includes the native provider source. Exact JUnit: `pytest-final.xml`.

## Component performance

Machine: macOS 26.3 arm64, Python 3.12.12, 10 logical CPUs, 16 GiB RAM.
These are local synthetic-file component workloads, **not coding benchmarks**.
Timing uses real Git, parsers, SQLite, workers and processes. No model was called
by the component benchmarks. Background machine load was not controlled.

Comparable seven-repetition mutation workload, 4 KiB files:

| 10,000-file operation | Step 1 median | Hardened median |
| --- | ---: | ---: |
| No-op observation | 457.36 ms | 22.24 ms |
| Single-file observation | 470.59 ms | 36.63 ms |

Full component run (ms; three samples for most warm operations, single cold/index,
compaction, background and verifier samples explicitly present in raw data):

| Operation | 100 files | 10,000 files | 30,000 files |
| --- | ---: | ---: | ---: |
| Initial syntax index | 25.34 | 2,866.93 | 9,227.65 |
| One-file index update | 0.65 | 0.75 | 0.67 |
| Ten-file index update | 3.69 | 4.28 | 3.71 |
| Definition lookup + watcher fence | 13.93 | 12.78 | 13.84 |
| Reference lookup + fence | 14.04 | 12.66 | 13.26 |
| Dependency lookup + fence | 14.00 | 14.10 | 13.67 |
| No-op mutation observation | 23.16 | 23.05 | 22.29 |
| One-file mutation observation | 21.68 | 36.14 | 83.13 |
| Ten-file mutation observation | 24.40 | 40.09 | 85.78 |
| Background mutation observation | 16.40 | 41.10 | 72.37 |
| Full reconciliation | 30.15 | 312.20 | 886.55 |
| FTS history retrieval | 0.72 | 5.32 | 14.99 |
| Deterministic context compaction | 2.22 | 2.14 | 2.35 |
| Related-test lookup | 13.52 | 19.74 | 236.96 |
| Warm REPL cell | 28.24 | 28.57 | 26.93 |
| Process spawn/admission | 3.04 | 3.32 | 4.01 |
| Shared child admission | 1.47 | 1.61 | 1.73 |
| One candidate checkpoint + admission | 100.90 | 875.83 | 3,437.83 |
| Four candidates, serial admission | 154.07 | 2,715.14 | 8,771.95 |
| Independent verifier, no test commands | 123.39 | 3,857.47 | 18,287.61 |

The 30k related-test samples were 269.93, 236.96 and 32.28 ms: notably variable,
not a reliable 32 ms claim. Final verification remains a major linear-cost boundary.

Snapshot observations on the 10k run: 8 MB immutable bytes initially 29.60 ms,
unchanged 0.052 ms, zero blob bytes rewritten. A 100k-item mutable list: initial
52.63 ms, unchanged 48.80 ms, still 1.89 MB serialized but zero bytes rewritten.
One thousand variables took 4.18 ms. Snapshot latency, not full cell latency.

Separate concurrent-backend worktree measurement (10,001 files / 35.1 MB):
one checkout 1.10 s, four parallel requests 1.26 s, excluding 0.35–0.78 s checkpoint
capture. Checkout bytes 35.1/140.5 MB; Git object database shared. Cleanup 0.41/2.22 s.
This tests backend overlap, not asynchronous runtime admission, which remains serial.

Raw: `mutation-comparable-final.json`, `performance-final.json`,
`isolation-concurrent.json`. Earlier intermediate measurements remain retained.

## Fresh coding evaluation

Two actual historical Threadweave issues at base commit `e5c8c41`: reusable
completed-child follow-up/recovery and interactive `/refine`. Each was run with a
real Codex subscription model, `gpt-6-astra`, reasoning `low`; OPENAI_API_KEY unset.
Codex 0.153.4, official structured Responses client. Reported monetary cost: null.
The configured 4096-output limit is a client guard, not a claimed server token cap.

Both profiles: 20-turn / 120,000-token / 600-second limits, 60,000 context ceiling,
same repository revision, model settings and withheld evaluator. `base` is basic
filesystem/shell Buffalo core, **not** independent raw-agent or Prime. `buffalo`
uses full configured capabilities. No standard external benchmark was run here.

First v2 preparation lacked process permission and failed before model calls;
those are setup errors, not coding scores. Corrected v3 was the first real run.

| Run/profile | Tasks | Accepted solved | Withheld verification passes | Tokens/task | Total model calls | Total wall seconds | Children | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 base | 2 | 0 | 1 | 106,338.5 | 33 | 269.32 | 0 | 0 |
| v3 Buffalo | 2 | 0 | 0 | 104,070.5 | 37 | 200.93 | 1 | 0 |
| v4 base | 2 | 0 | 1 | 109,322.5 | 40 | 289.94 | 0 | 19 |
| v4 Buffalo | 2 | 0 | 1 | 108,540.0 | 40 | 237.71 | 1 | 10 |

All sessions ended resource-limited, so **solve rate is 0%** for both profiles in
both freezes. Passing a withheld check does not override the required completion
condition. Tokens/solved and cost/solved are undefined. There is no superiority
claim or statistically meaningful effect estimate from these two tasks.

Post-fix v4 recognized test-command completions: base 5, Buffalo 4, including
baseline/final evaluator commands. These are lexical argv classifications, not
all subprocesses launched by arbitrary Python. Structured test-tool counters alone
reported only two baseline commands/profile; that is not the entire execution count.
Failed tool results: v3 base 2/Buffalo 1; v4 base 0/Buffalo 1. Propagated Python/RPC
errors can correspond to one root cause. `review-summary.json` preserves definitions.

## Trajectory-driven fixes and outstanding policy failures

v3 exposed a disabled helper advertised without an explicit base-profile allowlist
and input-context replay consuming the token budget before the context-window limit.
The fixes expose capability restrictions and compact against remaining tree-wide
tokens without changing budgets. Nested tool error accounting was corrected.
v4 uses the same tasks/settings/limits and shows real compactions and more calls,
but no accepted-completion improvement. One Buffalo `rlm()` call supplied a
keyword-only argument positionally: a model API-use error, not a provider failure.

The saved calls contain broad repeated source/test-file inspection with changing
arguments; exact-repeat guard counters therefore understate this behavior. Neither
profile used repository/history retrieval RPCs. Compaction did not demonstrate
decision-state sufficiency, and test execution/completion was not reached promptly.
Those are unresolved effectiveness problems, not reasons to weaken the evaluator.

## Remaining major limits and next measured targets

1. Coding completion efficiency: repeated reads, budget/turn awareness, concise
   decision state and discoverable Python helper signatures need further matched
   tests. No mandatory analyze/edit/test graph should be introduced.
2. 18.3 s full verifier boundary at 30k files; retain independent reconciliation
   while reducing repeated checkpoint/diff/artifact work.
3. Mutable snapshot reserialization (48.8 ms unchanged list), actual-change O(paths)
   manifests, and variable related-test latency.
4. Synchronous runtime candidate admission despite parallel-capable Git backend;
   checkpoint-ref/blob cleanup and binary/submodule candidate behavior remain limited.
5. Syntax, not compiler semantics; no LSP/coverage/embedding integration. CUDA uses
   C++ syntax subset, explicitly labeled. This label was corrected after v4; live
   Python task behavior is unaffected, but v4 is identified by its frozen source.
6. Trusted-host Python is not sandboxed. Research read-only is not an OS guarantee;
   detached processes may escape observation before reparenting. No Windows/GPU
   validation here. Secrets are not deliberately logged, but task output can itself
   contain sensitive data.
7. No actual external-harness or public-benchmark comparison. More representative
   frozen tasks and repeated trials are required before competitive claims.

## Exact retained evidence

- `frozen-v3/manifest.json`: freeze `c5fe16b5821eec3ab7e3e5d7e7966a36058709483a0548f09d6649daba78a8ed`.
- `frozen-v4/manifest.json`: freeze `0061923d5b36fed652055469f95274554e4e1ed8899501f391b252580549f9f4`;
  production source identity `453d4ccd5add82a0821eb61afb7c9aad3482efa2869f3eed39c9c20c60c26ada`.
- `live-v3/results.jsonl`, `live-v4/results.jsonl`: per-task configs, outcomes, usage and verifier evidence.
- `live-v3/review-summary.json`, `live-v4/review-summary.json` and matching CSV: normalized descriptive results, recomputed error counts.
- `live-v*/evaluations/<run-id>/events.jsonl`, `analysis.json`, `final.patch`,
  `state/history.sqlite3`, `state/artifacts/`: complete trajectories and outputs.
- `live-v*/<run-id>-review.json`: descriptive repeated-action/error/compaction evidence.

No old live report was counted as verification. No credentials/API credits were
requested or used. No claim of exact upstream equivalence is made.
