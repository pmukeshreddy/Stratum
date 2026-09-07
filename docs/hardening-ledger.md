# Hardening implementation ledger

All stages were worked through; not every ambitious target is complete. This is
a research harness with remaining production and coding-effectiveness limitations,
not verified equivalence to Prime. Component speedups are not solve-rate gains.
The default sole model tool remains `ipython(code)`. Python RPCs use common
Environment policy/hooks/auditing. Raw Python stays unrestricted. Independent
verification is unchanged by model strategy or related-test advice.

## Stage-by-stage record

Files below are under `src/threadweave/` unless specified. Focused tests are in
`tests/test_hardening.py`, plus Step 1's `test_python_environment.py` and existing
runtime/kernel/coding/daemon suites. These use test-only models, not live scores.

| Stage | Before → implemented | Files | Tests and measurements | Remaining limitation |
| --- | --- | --- | --- | --- |
| 1.1 | Per-cell traversal → native candidates, unique fence, full reconciliation on loss/overflow/recovery/verifier/periodic boundary | `change_tracking.py`, `mutations.py`, `environment.py`, `guardrails.py`, `runtime.py` | No warm enumeration, overflow, partial/background writes; 100/10k/30k measurements; comparable 10k no-op 457.36 → 22.24 ms | Actual-change manifest encoding remains O(paths); full scans block daemon; events are hints, not syscall journals. |
| 2 | AST/regex → eight actual Tree-sitter adapters; persisted symbols/imports/identifier references/calls/inheritance; indexed exact names, changed-file parsing | `syntax_index.py`, `repository.py`, `migrations.py`, `coding_tools.py`, `kernel_api.py` | Eight grammar tests; rename/delete/parse error/preserved mtime/corrupt index/reopen; cold index, 1/10-file update, queries | Syntax-derived, not compiler semantics. Dependencies heuristic, overloads ambiguous, no LSP/coverage graph. `changed_symbols` reports current symbols in latest refreshed files, not removed-symbol history. Generated/vendor exclusions and 2 MB file bound remain. |
| 3 | Byte estimates/weak Python focus → tokenizer estimates, actual usage retained, budget-pressure compaction, bounded coding focus, ranked FTS evidence | `tokenization.py`, `context.py`, `retrieval.py`, `refinement.py`, `runtime.py` | Tokenizer/fallback, budget/schema/history preservation; retrieval/compaction timing | Unknown tokenizer conservative fallback. Lexical/type/recency ranking, not embeddings. Model compaction optional; component timings measure deterministic fallback. |
| 4 | Implicit modes → `rlm(..., purpose='research'|'shared'|'candidate')` with durable purpose/base/workspace evidence | `kernel_api.py`, `host_api.py`, `runtime.py`, `environment.py` | Real sessions, purpose/read-only/candidate tests; existing asynchronous admission and concurrent kernels | Research read-only is tool policy, not raw-Python sandboxing. Shared writes can conflict. No separate persisted reject-decision API. |
| 5 | Full copy/recommit → private-index Git tree capture, dirty-parent checkpoint, detached worktrees sharing objects | `gitops.py`, `environment.py` | Four dirty-parent candidates, binary/index preservation, cleanup; 1/4 admission, checkout bytes/diff/cleanup | Synchronous serial admission; four-admission timing is not parallel scheduling. Each worktree checks out files. Old-checkpoint copy fallback; no ref GC/submodule automation/binary-patch acceptance parity. |
| 6 | Text scraping → machine reports where supported; JUnit, Go JSONL, Node JSON, Rust compiler JSON, text fallback | `test_evidence.py`, `coding.py` | pytest/CTest/Jest/Vitest/Go formats, JSONL/unittest/Rust/TS errors, malformed reports; real pytest execution | Not all versions/frameworks installed. Shell-wrapped commands are not rewritten. Some location inference remains heuristic. Raw and structured artifacts retained. |
| 7 | Manual targets → `tests.related_to` with failure/import/name/package reasons and failing/related/module/full tiers | `test_selection.py`, `coding_tools.py`, `kernel_api.py` | Selection evidence and unchanged full verifier; lookup timing | No coverage or compiler-resolved graph. Caller translates framework selectors. Bounded full-tier output reports total. |
| 8 | Mostly pass/fail → baseline-existing vs new/unknown failures plus symbol/diff/edit/command/policy evidence | `coding.py` | Baseline/tampering/forbidden/regression tests; Python-created edits; real final evaluator; verifier timing | Linkage syntax/heuristic; baseline distinction depends on stable failure IDs. Full verifier boundaries remain expensive at scale. |
| 9 | Whole JSON-style snapshots → explicit codecs plus restricted Cloudpickle procedures/classes/instances/arrays, bounded per-value blobs and immutable cache | `snapshots.py`, `kernel_worker.py`, `kernel.py` | Function/class/array restore, resource failure, corrupt blob isolation, stale/current interrupt; large namespace/state timings | Mutable containers reserialize for in-place changes. POSIX timers; interrupts kill/recover last snapshot, not graceful partial-state preservation. Trusted snapshots only. Opaque resources need recipes; isolated forks omit source-bound procedure blobs. |
| 10 | Process groups only → descendant birth-identity tracking, owner-loss cleanup, bounded drain/escalation, durable lost/cancelled/timed-out/failed state | `process_worker.py`, `execution.py`, `background.py`, `runtime.py` | Existing timeout/cancellation/owner-loss/native-output tests; lost recovery; spawn timing | Fast double-fork/setsid may escape discovery. No Windows job-object/cgroup backend. Lost jobs not replayed/falsely reattached. Raw Python subprocesses outside managed supervision. |
| 11 | Median/p95 → p50/p95/p99, variance/stdev, explicit outlier policy, raw samples, machine/compiler metadata and correctness-gated comparison | `benchmarks.py`, `experiments.py`, `machine.py`, `models.py` | Benchmark/experiment correctness/distribution/artifact tests | No GPU here; no GPU optimization result. IQR option is not evidence of statistical superiority; thermal/load controls not enforced. |
| 12 | Generic eval → hashed task/revision/config/runtime freeze, equal-budget base/Buffalo profiles, external JSON adapter, full event/patch/state retention | `evaluation.py`, `frozen_eval.py`, `cli.py`, `kernel.py`, `execution.py` | Real Git two-profile test-double fixture, tamper detection, oversized external response; fresh live runs | Base is basic Buffalo core, not independent raw/Prime. No external harness comparison run. Trusted-host evaluation does not OS-hide evaluator files. |
| 13 | Aggregate counters → per-event repeated actions/searches/reads, edit epochs, nested errors, retrieval/compaction/purpose provenance | `trajectory_analysis.py`, `evaluation.py` | Nested error regression; saved live analyses | Arbitrary Python reads/searches/tests not syscall-traced; tool metrics are lower bounds. Repetition does not prove uselessness. |
| 14 | No fresh matched run → two real historical defects at `e5c8c41`, same subscription model/settings/budgets and withheld tests | `evals/historical/prepare.py`, evaluator tests | Fresh `results/hardening/live-v3/` and `live-v4/`; real agent provider, API key unset | Tiny local suite, not public benchmark/superiority evidence. Limited sessions remain unsolved even if final patch passes withheld tests. |
| 15 | Observed budget exhaustion/unavailable helper → expose capability restrictions; compact against remaining root token budget; fix failed-action metric | `context.py`, `runtime.py`, `evaluation.py`, `trajectory_analysis.py` | Budget/history/schema regression; same live task/config rerun | Generic fixes, no task-specific solutions. Effectiveness must come from rerun outcomes, not tests alone. |

## Programmatic APIs

These are internal Python/RPC capabilities, not extra default model tools:

```python
definitions = repo.definition("Session")
callers = repo.callers("message")
callees = repo.callees("Runtime.message")
references = repo.references("Session")
evidence = repo.context_for_symbol("Runtime.message")
imports = repo.dependencies("src/threadweave/runtime.py")
users = repo.dependents("src/threadweave/runtime.py")
selection = tests.related_to(files=["src/threadweave/runtime.py"], tier="related")
review = await rlm("Inspect persistence and message findings.", purpose="research")
candidate = await rlm("Try a change and verify it.", purpose="candidate")
# Admission returns handles. Inspect/accept after candidate is idle/finished.
patch = await rlm.candidate(candidate)
```

## Choices and security

CUDA uses the C++ grammar with an explicitly labeled CUDA syntax subset, not a
complete CUDA compiler/parser. CUDA-only constructs may report parse errors while
retaining useful declarations. This metadata label was corrected after the live
v4 freeze; it does not change the Python historical-task runtime semantics.

Global-state permissions, task budgets and automatic-refinement defaults were not
increased for results. Subscription credentials remain owned by Codex; dollar cost
is null. Read-only tool policy and container process execution do not sandbox the
unrestricted host REPL. Worktree files are independent, but their object database
is shared. Checkpoints and procedure blobs are trusted private local data.

Warm observer no-op is O(events); fence failure falls back to O(paths). Fuzzy
symbol misses and dependency inference may scan metadata. Actual changed manifests,
mutable snapshots, and full verification still have linear costs. These remain
explicit optimization targets rather than hidden guarantees.

## Evidence

- `results/hardening/mutation-comparable-final.json`: seven-sample comparable workload.
- `results/hardening/performance-final.json`: component raw samples and machine metadata; cold index, child admission, diff, verifier and compaction may have only one sample.
- `results/hardening/pytest-final.xml`: final regression report.
- `results/hardening/frozen-v3/manifest.json`, `frozen-v4/manifest.json`: exact source/dependency/config/evaluator identities.
- `results/hardening/live-v3/`, `live-v4/`: fresh model trajectories, results, patches and SQLite state.
- [Evaluation reproduction](hardening-evaluation.md). Older reports elsewhere under results are not new verification.
