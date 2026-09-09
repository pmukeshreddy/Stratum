# Observed behavior

The three-case run `after` completed and passed independent acceptance on all three tasks. The subsequent `final_resolution_check` exercised the corrected pending-work resolution path on the recursive task and also completed and passed. The earlier three-case run still exposed one handled resolution error; that is why the final correction and recheck were necessary.

Provider: local `codex_subscription`, model `gpt-6-astra`, medium reasoning, automatic tool choice. Each case used an isolated git workspace, 24,000-token context, 24-turn cap, 240-second wall cap, and a separate state store. The final evaluator explicitly enables evaluation isolation while keeping within-task refinement enabled. Tasks had no tool-use or delegation quotas. Acceptance ran in an independent Python subprocess.

| Run | Task | Outcome / acceptance | Seconds | Root REPL | Children | Child evidence inputs | Model calls | Verification L1/L2/L3 | L1 compactions | L2 checkpoints |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| before | identifier_repair | completed / pass | 27.6 | 2 | 0 | 0 | 3 | 0/0/1 | 0 | 2 |
| before | ledger_aggregation | failed / fail | 3.5 | 1 | 0 | 0 | 1 | 0/0/0 | 1 | 1 |
| before | independent_components | failed / fail | 42.6 | 5 | 1 | 1 | 7 | 0/0/0 | 5 | 7 |
| after | identifier_repair | completed / pass | 38.2 | 2 | 0 | 0 | 4 | 3/1/1 | 0 | 2 |
| after | ledger_aggregation | completed / pass | 58.4 | 4 | 0 | 0 | 9 | 5/1/1 | 2 | 5 |
| after | independent_components | completed / pass | 120.2 | 11 | 1 | 5 | 26 | 18/4/3 | 21 | 25 |
| final_resolution_check | independent_components | completed / pass | 92.3 | 12 | 1 | 9 | 25 | 20/4/4 | 24 | 27 |

Root REPL entry was already 100% in the baseline coding sample. The relevant improvement is that the previously failing ledger and recursive tasks completed with independently checked code. This small sample is not evidence of a general success-rate or latency improvement. The simplest case became slower (27.6 to 38.2 seconds). The baseline failures cannot be compared as successful task latency.

## Refinement lifecycle

| Run/task | Seen | Prefilter pass/reject | Reviews | Declined | Proposed | Validation pass/fail | Activated versions | Later input receipts |
|---|---:|---|---:|---:|---:|---|---:|---:|
| after/identifier_repair | 12 | 10/2 | 1 | 1 | 0 | 0/0 | 0 | 0 |
| after/ledger_aggregation | 17 | 12/5 | 3 | 1 | 1 | 1/0 | 1 | 1 |
| after/independent_components | 49 | 38/11 | 6 | 4 | 2 | 2/0 | 2 | 3 |
| final_resolution_check/independent_components | 37 | 29/8 | 4 | 3 | 0 | 0/0 | 0 | 0 |

The three-case run activated a bounded-inspection note and two versions of a session interpreter note. The later-input receipt records inclusion in the actual request, not proof of causal improvement. In the final recheck, three reviews declined and one admitted planning; that planner was interrupted by task shutdown and did not activate a version. Its baseline and completed evidence were retained. This is a remaining limitation for learning near task completion.

## What the traces establish

- The root delegated version comparison while implementing interval merging locally. The child used its own REPL, wrote focused tests, returned implementation decisions and validation evidence, and those sources reached later root invocations. The final recheck also resumed the same child for follow-up work.
- Continuous syntax/execution diagnostics, targeted pytest selectors, and independent full gates all ran. The final recursive recheck ran four full gates across twenty tree turns; explicit requests and child completion can still repeat full checks. No final gate was replaced by a cheap check.
- The final recheck used 24 L1 compactions and 27 L2 checkpoints without a model compaction request. The sum of checkpoint serialization timings was 0.0181 seconds. Counts alone do not imply large overhead; full per-purpose model timings are in comparison.json.
- The sample values did not exceed offload thresholds: zero live-sample artifact offloads or prunes. A real 24 MiB string in the integration tests crosses the old 16 MiB limit and verifies artifact-backed recovery. Other lifecycle tests exercise stale retirement, recipes, opaque objects, corrupted checkpoints, failed commits, live-kernel preservation and child isolation.
- The final recheck recovered from one malformed/ambiguous child-message recipient call. The previous automatic-projection revision hit its turn limit by repeatedly rereading files after losing visible completion evidence. Live receipts and authoritative resolution now preserve that state.

## Intermediate runs, including failures

| Revision label | Cases | Completed | Acceptance passed |
|---|---:|---:|---:|
| before | 3 | 1 | 1 |
| first_upgrade | 3 | 2 | 3 |
| evidence_tuning | 3 | 3 | 3 |
| verification_run | 3 | 2 | 2 |
| pre_gate | 3 | 3 | 3 |
| gate_tuning | 1 | 0 | 1 |
| wait_tuning | 1 | 0 | 1 |
| projection_tuning | 3 | 2 | 3 |
| after | 3 | 3 | 3 |
| final_resolution_check | 1 | 1 | 1 |

Intermediate runs include wall-limit failures, a provider transport failure, excessive auxiliary evidence reduction, repeated model compaction, and one turn-limit failure after switching to deterministic projection. These drove the changes described in implementation.md. They are retained instead of reporting only a successful attempt.

## Audit artifacts

comparison.json includes every observed case, event counts, model calls grouped by purpose/status, source state paths and received harness versions. Each run/task trace.tar.gz contains a SQLite history backup, events.jsonl.gz, and complete referenced artifacts indexed by artifact ID. These are audit bundles, not portable kernel-recovery images. source-manifest.json fingerprints the delivered source. commands.txt records verification and evaluation commands. Tests use real IPython workers; scripted providers test transport and causality, while the local evaluations test unforced model behavior.
