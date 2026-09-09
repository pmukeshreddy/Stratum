# Root-loop and latency validation

The demanding tasks now finish with complete correct reports inside the unchanged **300-second** task budget. Final-source v8 results are **299.25s for the four-component audit** and **285.22s for the optimality assessment**. Both retain substantive asynchronous children, overlapping root work, and child evidence in later root requests. **The audit has only 0.75s of headroom**: this establishes the requested targeted criterion, not a guarantee that every stochastic run or benchmark task will fit.

No ManyIH benchmark was launched. Tasks, fixtures, model (`gpt-6-astra`), `xhigh` reasoning, child inheritance and output allowance were not weakened or increased to obtain these results. Tasks contain no instructions to use Python, RLM, children or delegation. The root still chooses actions through its ordinary loop; completion has no usage quota.

## Final validation

**485 passed, 9 skipped, 2 existing `forkpty()` warnings in 99.52s.** Ruff, formatting (152 files), sdist/wheel build and `git diff --check` pass. The built runtime modules and v8 source manifest match current source. [Source audit](../results/root-loop-validation/source-audit.json), [pytest output](../results/root-loop-validation/pytest.txt), [build output](../results/root-loop-validation/build.txt).

Production-path coverage includes direct completion without execution; persistent root/child variables; original ordered task messages and auxiliary contracts; immediate canonical child admission; parent concurrency; persistent bidirectional messaging; observation; grandchildren and inheritance; recovery/compaction; refinement decline, justified application and later state retrieval; safe background refinement; settled cancellation accounting; token estimates after interleaved messages; awaited-shell notification races; and complete-answer feedback after interrupted completion.

The exact first foundation remains [root-foundation.txt](root-foundation.txt), SHA-256 `1e2b7e250aa6c98bd0e5df9d47ba71c26146d35946145895a62d801cf2d94abd`. Both final first requests match it exactly. The [architecture and API audit](adaptive-orchestration.md) documents the preloaded surface, canonical `rlm` path, lifecycle and actual reference-code comparison. No strategy-router dependency remains in source, configuration or the built wheel.

## Live results

All eight cases completed within 300 seconds in v6 and v7. Output review then found that a late mathematical child message made the root replace its detailed candidate with an acknowledgment in v7. That incomplete deliverable was **not accepted**. Production completion feedback was corrected, tested and validated on both demanding tasks again in v8.

| Case | Latest relevant run | Seconds | Children / peak | Result |
|---|---|---:|---:|---|
| Original-message hierarchy | v7 | 15.26 | 0 / 0 | Correct `five`, preserving system/developer requirements. |
| Trivial arithmetic | v7 | 10.05 | 0 / 0 | Correct `42`, zero Python executions. |
| Transaction inspection | v7 | 44.61 | 0 / 0 | Exact counts and signed sums over 6,000 records. |
| Four-component audit | **v8, final source** | **299.25** | **2 / 2** | Complete observed reproducers, corrections, passing controls and bounded independent oracles for every component. |
| Optimality assessment | **v8, final source** | **285.22** | **1 / 1** | Complete counterexamples with actual returns, feasibility/optimality proofs, exact condition, tight approximation bound and separate scaling analysis. |
| Shared money repair | v7 | 157.59 | 1 / 1 | Supplied checks and independent exact-money/type/quantity checks pass. |
| One-off word ordering | v7 | 14.60 | 0 / 0 | Correct ordering; no execution or children required. |
| Identifier repair | v7 | 131.45 | 0 / 0 | Supplied, expanded and independent Unicode/ASCII checks pass; justified refinement version reaches a later invocation. |

The six v7 ancillary cases validate the latency implementation immediately before the completion-feedback correction; the demanding v8 pair exercises the final source, including that correction. No failed run is substituted for a successful final deliverable.

Final reports are retained as [component audit](../results/root-loop-validation/component-audit-final.md) and [optimality assessment](../results/root-loop-validation/optimality-review-final.md). Independent reproduction checks confirm selected defects and exact optima; implementation files remain unchanged. The complete reports were also reviewed for the task's requested coverage, observed results and qualifications. Lifecycle completion or a nonempty answer alone was not sufficient.

The independent ancillary checks recompute all transaction counts/sums; verify 1,000 generated amounts across both money consumers, 5,000-digit magnitudes and malformed input/quantity handling; and exercise 10,000 generated Unicode/ASCII identifier cases. [Verification artifacts](../results/root-loop-validation/v7-independent-verification.json), [final pair checks](../results/root-loop-validation/v8-independent-verification.json).

## Timing evidence and preserved behavior

The [latency investigation](latency-validation.md) records each concrete cause, production correction and failed/successful rerun. Main corrections:

1. Automatic progress review overlaps normal root/child execution, with one pass per session, immutable evidence/baselines and safe state application.
2. Reported context occupancy survives newly inserted child messages; all new evidence is counted without recounting unchanged opaque continuation bytes.
3. Late proactive compaction can defer while the complete request fits; hard capacity and forced compaction still work.
4. An awaited shell result no longer also creates a pending background notification; genuine background jobs and unrelated messages remain deliverable.
5. Background review cancellation settles before evaluation accounting is saved.
6. Completion feedback explains that a deferred candidate was withheld and that the next final response must contain the complete deliverable.

In the final audit, root inference occupies 297.7s; 54.2s of auxiliary inference overlaps it for 53.9s. The mathematical case has 283.2s of root inference, with 110.2s of review/planning overlapping for 110.1s. There are no final-run compaction calls. Model inference now dominates; no long serialized review or scheduling wait remains in those critical paths. The mathematical case explicitly exercises a late-message continuation and returns a complete 6,645-character report afterward.

Each final demanding case contains **two root Python executions during child inference**. All admitted children finish; findings enter three audit root invocations and five mathematical root invocations. The audit report carries the CSV and graph findings from its children, including independently checked duplicate-edge behavior. The mathematical report incorporates the child's exact optimality criterion and scaling qualifications. [Causal records](../results/root-loop-validation/v8-evidence.json) retain actual child messages, source IDs, receiving invocations, overlapping actions and completion feedback. These records describe evidence flow; they do not manufacture reviewer-backed `child_evidence_used` events when the reviewer did not provide validated attribution.

Automatic refinement remains functional. In v7 the identifier trajectory had three reviews, two declines and one evidence-linked state update, with that version in a later root invocation. V6 also applied useful state in both repair cases. One-off cases decline without edits. Expensive investigation reviews may be deferred or cancelled within their allowance; useful execution and valid completion remain independent of whether learning produces an edit.

## Readiness and retained history

The requested targeted criterion is met: demanding unchanged tasks complete correctly below 300 seconds while preserving useful recursive work and concurrency. There is no identified remaining architectural blocker from this validation. The narrow final audit margin and provider/action variability remain real performance limits; these runs do not establish a 100-task success rate.

Buffalo is ready for a **separately authorized clean ManyIH run** on this evidence. No benchmark run or authorization receipt was created automatically.

Raw runs remain under `/Users/mukeshreddypochamreddy/Downloads/buffalo-root-loop-validation-20260909-v1` through `-v8`. V1–v3 exposed delivery/budget problems. V4's earlier **600-second diagnostic** measured the original 467.4s/310.6s overruns and is not counted as a 300-second pass. V5 still failed the audit at 300 seconds, exposing compaction latency. V6 established same-budget completion; v7 exposed the incomplete follow-up; v8 verifies the final correction. Profiles, activity, source manifests, output checks and reports remain in [results/root-loop-validation](../results/root-loop-validation/).
