# Execution latency validation

The 300-second task budget, neutral task wording, model (`gpt-6-astra`), `xhigh` reasoning, child inheritance and output allowance are unchanged. No ManyIH benchmark has been launched. The source still implements task → root action → evidence → next root action; no preliminary strategy inference or mechanism quota was introduced.

## Measured causes and production changes

Profiles come from actual `model_requests`, action/event timestamps and completed results, using `python -m threadweave.evals.latency RUN_DIRECTORY ... --output PROFILE.json`. Union durations measure elapsed coverage; summed child durations describe work across concurrent sessions and must not be added to task wall time. Complete request timelines are retained in `results/root-loop-validation/v*-latency.json`.

| Cause | Recorded evidence | Production fix |
|---|---|---|
| Automatic review serialized ahead of root execution | In v4 the audit spent 259.3s in root inference and **205.7s** in review/planning with **zero overlap**. Children finished by 173.3s; synthesis started at 370.5s. Optimality spent 136.5s in root inference and **172.3s** in review/planning, also with zero overlap. | Progress review captures a committed snapshot and runs concurrently. One pass per session coalesces checkpoints. Immutable baselines, source validation, bounded budgets and safe application remain. Completion does not wait for an already active review. |
| Preemptive compaction delayed a fitting final context | After background review was introduced, v5 audit compaction ran from **162.8–234.4s**. It cost **71.6s**, followed by two rereading/checking turns costing another **15.0s** before synthesis started at 249.9s and was interrupted at 300s. | While the complete request still fits, late proactive compaction defers when measured inference timings leave insufficient time for summarization plus another action and synthesis. Assembly honors that decision. Forced compaction and hard input capacity still compact; no evidence is discarded to avoid the optional model call. |
| Child message delivery invalidated measured context occupancy | V5 root turn 9 estimated **42,697 input tokens**; the provider reported **21,436**. Messages received inside actions were committed before the response block, breaking a strict-prefix usage match and reverting to a bound that counts opaque continuation bytes. | Match the previous items as an ordered subsequence, count every inserted item, and retain provider-reported input/output occupancy for unchanged items. Rewrites, deletions or changed provider/tool contracts still invalidate the anchor. Private continuation data remains unchanged. |
| Awaited shell commands also generated background-result messages | V5 root received notifications containing shell output already returned by `await bash(...)`, duplicating retained context. Such notifications could also invalidate a just-produced final answer. | Active waiters receive the result directly. If a short process completed first, waiting acknowledges only that process's still-pending notification. Durable events/artifacts remain; unrelated messages and unawaited job notifications are untouched. |
| Background cancellation could escape final evaluation accounting | Concurrent review introduces an auxiliary task that can outlive the last agent turn. Previously the evaluation path settled only agent tasks before taking its usage snapshot. | Cancel and settle active reviews before saving the result and usage, including conservative usage for interrupted provider calls. Regression checks require no running model attempt or reservation afterward. |
| Deferred completion was presented like a conversational follow-up | In v7 optimality, a rigorous 5,847-character candidate finished at 149.0s. Late child messages caused two follow-up turns, ending at 172.4s with only a 297-character acknowledgment. The earlier report was withheld, so the returned result lost required detail despite completing on time. | Completion feedback explicitly tells the root that the candidate has not been returned and that its next final response must be the complete self-contained task answer. The same rule accompanies waiting for active descendants. New evidence still reaches ordinary root turns; the runtime neither discards it nor substitutes an earlier answer. |

The baseline has no compaction or reducer calls. Child work was already concurrent: audit child inference totals 417.0s summed across sessions but covers only 147.1s of elapsed time. Kernel/tool execution occupies less than 1.2s per side, and intervals with no model running total only 1.7s for the audit and 1.4s for optimality. There was no evidence of scheduler polling or kernel startup causing the large overruns. No child runtime or useful delegation was disabled.

The changes do not guarantee that two live runs choose identical actions. Timelines establish which waits were removed; elapsed-time comparisons also include ordinary variation in root/child choices and provider inference latency.

## Validation progression

- **v4, earlier extended diagnostic:** audit 467.4s, optimality 310.6s. This is historical evidence of the failures, not a successful 300-second validation.
- **v5, 300 seconds:** background review overlapped execution for 89.9s, but the audit still timed out because of the compaction path above. Optimality completed in 130.3s.
- **v6, 300 seconds:** all eight cases completed. Audit **240.2s**, optimality **118.2s**, shared money repair **156.0s**, identifier repair **144.5s**. The audit used two concurrent substantive children; the money repair used another child. Both repair trajectories applied a justified versioned update that reached a later root invocation. Independent checks passed.
- **v7, 300 seconds:** audit 242.7s with three concurrent children; optimality 172.4s with one reviewer; money repair 157.6s with one child; identifier repair 131.5s. All eight sessions completed within budget. Output inspection caught the incomplete mathematical follow-up described above, so this run alone was not accepted as readiness evidence.
- **v8:** the demanding cases are rerun after correcting completion feedback; final results are recorded in [adaptive-validation.md](adaptive-validation.md).

The original task fixtures and model/reasoning settings are compared across runs. Final-source hashes, exact request validation, output checks, activity and causal evidence are retained with the results. The model is never instructed to use Python, RLM or children by these validation tasks.

## Regressions

- A held reviewer cannot block root inference; a completed plan cannot mutate an in-flight root context; its version reaches the following turn.
- Evaluation completion settles a held background review and includes its cancellation usage.
- Child findings arriving during preparation enter the next root request.
- Fitting late context stays verbatim; over-capacity context still compacts.
- Inserted child evidence is charged without recounting opaque provider continuation; altered history invalidates the estimate.
- Awaited shell results do not duplicate notifications, including completion-before-wait races; unawaited results and unrelated human messages remain deliverable.
- Existing process ownership/cancellation, recursive recovery, original task contract, persistence, state provenance and refinement-decline tests remain enabled.
- A late child finding causes explicit feedback that the previous candidate was withheld; the next root request retains that candidate and the new evidence and requests a complete deliverable.
