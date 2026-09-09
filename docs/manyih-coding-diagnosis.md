Buffalo's fastest credible route from **62 to 72 overall passes** is to validate the actual final answer against the task's output contract and resolved instructions before accepting completion. The evidence favors a bounded verification-and-repair step. It does not support starting with more subagents, more reasoning tokens, or more repository tools.

This is a diagnosis of the completed 100-task experiment, using its original answers, committed histories, official grades, and executable provider/runtime/evaluator paths. The measured scores remain **Buffalo 62/100 and Codex 66/100**. No new model calls were made, no production behavior was changed, and the original artifacts were checked against their saved hashes. Post-hoc edits described below are explicitly diagnostic, not new agent results.

| Original Buffalo outcome | Tasks | Meaning |
| --- | ---: | --- |
| Functional and style pass | 62 | Already successful |
| Functional pass, style fail | 28 | Working implementations rejected on style |
| Functional fail, style pass | 7 | Functional grade blocks success |
| Both fail | 3 | Both dimensions need attention |

With functional outcomes unchanged, recovering 10 of the 28 style-only failures gets overall pass from 62 to 72. That would move style pass from **69 to 79**, not merely to 72. It requires a net gain of 10: if two current successes regress, twelve failures must be recovered. The evidence establishes room for improvement, not a guaranteed improvement from an untested intervention.

The style failures are concentrated: operator spacing affects 26 tasks, internal blank lines 4, type hints 1, and line length 1. These counts overlap; 31 tasks fail at least one style category. See [the complete task inventory](../results/manyih-coding-diagnosis/task-diagnoses.json) and [computed findings](../results/manyih-coding-diagnosis/findings.json).

**What the production trace actually did**

The evaluation uses [the production harness](../src/threadweave/evals/harness.py#L217), creates real `Runtime` sessions, and selects the coding adapter. It is not a direct model-only substitute. However, the supplied benchmark interface is a final code answer, while the coding verifier is oriented toward repository changes and configured commands.

The ManyIH configuration deliberately sets `require_tests=False`, `require_change=False`, and `capture_baseline=False`, because the official task supplies no visible repository test suite. At completion, [Runtime._run_turn](../src/threadweave/runtime.py#L1113) accepts ordinary final text as a completion candidate, and [Runtime._verify](../src/threadweave/runtime.py#L1923) calls the task adapter. [CodingTask.verify](../src/threadweave/coding.py#L197) checks repository invariants and configured commands. It does not validate the Python implementation in the final response against the natural-language output contract.

The resulting facts are unambiguous:

- 99 tasks had one agent request. Task 19 had three requests; its two Python executions listed the workspace and read `TASK.txt`, rather than testing the implementation.
- There were 102 agent requests and no auxiliary refinement/review requests in the durable model-request tables.
- All 100 coding verifier results reported `passed=true` with **zero configured test/build/lint/typecheck/benchmark checks**. Repository invariants were still checked; those results provide no functional/style evidence about the final code answer.
- The official evaluator ran after agent completion. Its failure details never returned to the solving agent.
- Automatic refinement was enabled but deferred on every task. [auxiliary_admitted](../src/threadweave/refinement.py#L286) reserves `(auxiliary time + agent time) * 1.25 + tool timeout`. The cold auxiliary estimate is the 300-second provider timeout, while the coding tool timeout is 600 seconds. The reserve cannot fit the 300-second task budget.

These facts explain why Buffalo's existing verification and learning mechanisms contributed little to this experiment. They do not establish that a verifier or a second model pass would automatically recover ten tasks. A completion-time harness refinement pass also learns state for later use; it is not the same as checking and repairing the current answer. With fresh state for every benchmark task, simply enabling post-completion learning would not improve the already-submitted answers.

**A real instruction-resolution mistake worth fixing**

The task asks the model to satisfy as many instructions as possible, respecting priority only when instructions conflict. Selecting just the highest-ranked instruction within each topic is insufficient.

Task 21 says argument annotations are required and return annotations optional at priority 3. Priority 5 additionally asks for full annotations. Those requirements are compatible. Buffalo omitted the return annotation and failed. The higher-ranked permission to omit a return annotation does not prohibit the compatible lower-ranked request to include it.

Tasks 7, 23 and 31 show the same issue with spacing. The highest-ranked arithmetic-spacing instruction leaves assignment spacing unconstrained. A compatible lower-ranked instruction still asks for spaces around all operators. Buffalo left assignments unspaced. Task 31 is a particularly clear example, independent of the string/operator-token checker problems discussed below.

The general fix is a task constraint record with source references: preserve every compatible requirement, record which conflicting requirement loses and why, and check the draft against the retained set. This belongs to normal instruction handling, not a parser for ManyIH task IDs or hidden `expected_styles`. Real message authority must remain authoritative; within-task priority labels have effect only where the governing instructions delegate that interpretation.

**The official evaluator also has material defects**

Source inspection and replay distinguish actual instruction failures from sensitivity to raw text:

- The operator checker uses substring search, rather than Python tokens. It treats the hyphen in `element-wise`, the slash in `km/h`, and the minus in the return annotation `->` as operators. It also examines pieces of `==` and `+=` separately. Tasks 77 and 96 fail solely because of docstring wording; task 32 fails on annotation whitespace. Task 97's properly spaced `count += 1` is rejected by the arithmetic checker.
- Some combinations requiring full return annotations plus arithmetic spacing penalize ordinary Python return annotations regardless of the implementation. Splitting `->` into `- >` is invalid Python. This is a checker problem, not evidence that the model misunderstood an arithmetic expression.
- The blank-line checker counts only from the first AST body statement to the last. A blank immediately after `def`, before the only `return`, does not count. Buffalo made that placement on tasks 33, 40, 65 and 82. Codex made the same mistake on Buffalo-only wins 37 and 41.
- The functional evaluator keeps only lines beginning with `assert` and drops fixture setup. Four failed tasks reference `math.isclose` without `math` available. Replaying the saved implementations with the official per-assertion helper shows **NameError**, even though the aggregate output labels them generically as assertion failures.

The pinned executable sources are [operator_spacing.py](https://github.com/JHU-CLSP/ManyIH/blob/3287f82436fff86d7506b7737dbd3b5e2e524d01/manyih/coding/styles/operator_spacing.py), [blank_lines.py](https://github.com/JHU-CLSP/ManyIH/blob/3287f82436fff86d7506b7737dbd3b5e2e524d01/manyih/coding/styles/blank_lines.py), and [eval_utils.py](https://github.com/JHU-CLSP/ManyIH/blob/3287f82436fff86d7506b7737dbd3b5e2e524d01/manyih/coding/eval_utils.py). These findings come from the local checkout at that exact revision and actual replay, not its README.

The ten functional failures break down as follows:

| Task IDs | Diagnosis from source and replay |
| --- | --- |
| 18, 43 | Buffalo returns success/failure strings where boolean predicates are expected. Codex returns booleans. This is a useful output-contract improvement target. |
| 99 | The task supplies an imaginary-valued second argument, while both agents interpret it as a real coefficient. Adding the missing test import still leaves incorrect numerical results. |
| 15, 48, 95 | Pure missing `math` test-fixture dependency in the observed failures. Adding that import in a diagnostic replay makes all assertions pass without changing the implementation. |
| 14 | Public prompt says left rotation; the first expected result corresponds to right rotation. Both agents implement the stated left rotation. |
| 30 | The supplied reference formula multiplies by `4*a` where the standard parabola directrix formula divides by `4*a`; tests enforce that reference behavior. Both agents give the usual mathematical formula. |
| 36 | Prompt says every sublist has two elements; a hidden test requires transposing three-element sublists. Both agents implement the explicitly described two-column contract. |
| 63 | Prompt asks to count uppercase letters, but tests expect 1 for both `PYthon` and `BigData`, which each contain 2. The reference returns inside its first loop iteration. |

Task 99 also has the missing-import problem, so the categories above identify the additional implementation problem separately. Three supplied reference implementations themselves fail the official functional helper on tasks 48, 95 and 99. The complete assertion error records are in [functional-failure-details.json](../results/manyih-coding-diagnosis/functional-failure-details.json).

We should preserve the original official score and report these defects alongside it. A patched evaluator would define a separate diagnostic result. Teaching Buffalo to count only the first character, use a wrong mathematical formula, or add imports needed exclusively by hidden tests would be benchmark overfitting. Likewise, replacing legitimate comparisons with awkward equivalents merely to avoid a substring checker is not a general harness improvement.

**A controlled diagnostic establishes how much is superficial**

I made small, explicitly hand-selected edits to 13 saved failures and passed each edited answer to the unchanged official `judge_response` through the existing official worker. All 13 became functional-and-style passes:

| Edit | Task IDs |
| --- | --- |
| Return annotation whitespace | 3, 24, 32, 53 |
| Add missing return annotation | 21 |
| Put the existing blank inside the AST body by adding a permitted docstring | 33, 65, 82 |
| That blank-line fix plus annotation whitespace | 40 |
| Shorten the overlong docstring | 69 |
| Change docstring wording only | 77, 83, 96 |

Parsing both versions produces identical computation ASTs after excluding docstrings and return annotations. Those metadata fields do change; this is not a claim of equivalence under arbitrary introspection. The official functional tests pass before and after each edit.

**This produces a saved-answer counterfactual of 75/100, not an achieved Buffalo score.** The edits were chosen after seeing failures and expected constraints. No model autonomously produced them in a new run. The experiment shows that ten additional algorithmic breakthroughs are unnecessary; it does not show that a production checker can automatically obtain 75, or even 72. Seven of these cases are annotation-whitespace or wording sensitivity, so a semantically correct checker alone may not reproduce all those gains against the flawed official checker. [Exact edits and grades](../results/manyih-coding-diagnosis/posthoc-edit-results.json) and [the reproduction script](../results/manyih-coding-diagnosis/reproduce.py) are saved.

**How to interpret the four-point Codex gap**

Codex-only wins are 18, 32, 33, 69, 77, 96 and 97. They comprise one predicate-return contract difference, two layout/length differences, and four annotation/docstring/compound-operator checker sensitivities. Buffalo-only wins are 19, 37 and 41: a divisors interpretation difference and two blank-line placements. Thus the paired evidence does not suggest that native Codex has substantially better repository exploration or more capable subagents for these tasks.

With only ten discordant pairs, the exact two-sided McNemar test gives **p = 0.34375**. The observed winner remains Codex; a stable four-point harness effect has not been established from this one run. Repeated paired trials should precede stronger causal claims.

**What relevant research supports**

The ManyIH paper itself identifies style adherence as the main bottleneck while frontier models retain high functional correctness. It also observes that longer reasoning traces do not reliably translate into better accuracy; its model-specific reasoning-effort results should not be extrapolated into a promised gain for our different model. Buffalo already ran at xhigh. This supports targeting constraint handling and final checking first. [ManyIH, sections 6.3 and 6.5](https://arxiv.org/html/2604.09443v3).

Self-Refine demonstrates iterative feedback and revision on several tasks, but those results do not predict a ten-point gain here. A later critical survey finds much stronger support for self-correction with reliable external feedback than for unconstrained model self-criticism. The implementation implication is concrete: give a repair pass a failing check, the relevant source instruction, and the exact candidate, rather than simply asking it to think harder. [Self-Refine](https://arxiv.org/abs/2303.17651), [Kamoi et al., TACL 2024](https://aclanthology.org/2024.tacl-1.78/).

Reflexion uses feedback plus memory to improve later trials. That mechanism needs an actual feedback signal and a later opportunity to use it. Our completion reviews were deferred, hidden grading occurred afterward, and task state was reset, so this run did not test such a learning advantage. [Reflexion](https://arxiv.org/abs/2303.11366).

**Recommended implementation order**

1. **Give verification the actual final candidate and report coverage honestly.** Extend the existing task/verifier path to receive the proposed answer or artifact with its content hash. Distinguish repository-invariant success from functional/output-contract coverage. Zero configured checks should be represented as unverified coverage, rather than suggesting that the answer's correctness was established. Keep all existing coding safeguards.
2. **Resolve and retain the task's compatible constraints.** Build a compact, source-linked output contract before validation: requested artifact, function/API identity, return behavior, and applicable formatting requirements. Preserve compatible lower-priority instructions. Validate the same final candidate that will be submitted. This is generic instruction infrastructure; Python-specific validation stays in the coding capability.
3. **Add a bounded pre-completion check-and-repair cycle.** Run syntax/AST checks, declared line/indentation/annotation/layout checks, and examples or tests derived only from visible requirements. A global default formatter would be wrong because tasks explicitly demand conflicting formatting conventions. Use token-aware operator checks. On a concrete failure, allow a narrowly scoped repair, then recheck the whole retained contract and available tests so a style repair cannot silently break a passing solution. Preserve the last candidate, durable evidence and provenance, and cap attempts and elapsed time.
4. **Use an independent model review only for unresolved semantic contract questions.** For example, whether a predicate returns a boolean or a prose status, or whether an argument is already complex. Ground this review in the original task, draft, and observed checks. Self-generated tests and model judgments are fallible evidence, not hidden ground truth. Keep auxiliary request accounting separate from the agent trajectory and make failures safe. This uses Buffalo's existing request and lifecycle mechanisms.
5. **Repair optional-work budgeting separately.** Use stage-appropriate bounded time allowances and reserve only work that can actually fit the remaining deadline. Completion-time learning should not always reserve a 600-second tool operation. Preserve safe boundaries, review gating, baseline conflict protection, and atomic apply. Do not count this scheduling change as a direct ten-point scoring fix: post-completion learning and current-answer repair serve different purposes.

For short answer-generation tasks, a reasonable first experiment is a small local-check allowance and at most one review plus one repair, all inside the existing 300-second deadline. Those are proposed limits, not measured optimal settings. Record accepted/rejected candidate hashes, check coverage, reasons, extra tokens, time, regressions, and whether the auxiliary step was actually admitted. Every validation rule needs a visible task source or a general language correctness justification; the production path must not import ManyIH grading metadata or branch on task IDs.

**How to test whether this actually reaches 72**

Treat IDs 0–99 as the now-inspected development/regression set. Test improvements on passing tasks as well as failures; selecting only failed cases would conceal regressions. Freeze the implementation and budgets before the next measured run. The concrete development acceptance target is at least 72 overall, with no reduction from the current 90 functional passes, and measured gains minus regressions of at least ten.

Then use the untouched canonical IDs **100–199** as a separate confirmation set, without consulting their hidden tests or styles during implementation. Run unchanged Buffalo, improved Buffalo, and native Codex under matched model/reasoning/environment/deadline conditions, retaining all failures and per-task artifacts. Repeat independently if the first gain is small or unstable. Report official scores separately from any evaluator-correction diagnostic, and never choose the best run after observing results.

A 72 on the original inspected set would be a development result. A matching gain on the untouched set would be stronger evidence of a general harness improvement. We should also prepare reproducible upstream evaluator fixes, while keeping the pinned original score intact for comparability.

The next engineering task should therefore be **candidate-aware verification, compatible-constraint retention, and bounded repair**, with the optional refinement budget change tracked separately. The current research supports that order; it does not justify promising 72 before a prospective run.

Reproduce the diagnostics without any model calls:

```sh
.venv/bin/python results/manyih-coding-diagnosis/reproduce.py \
  --archive /Users/mukeshreddypochamreddy/Downloads/buffalo-manyih-coding-100-results \
  --source /tmp/buffalo-manyih-reference \
  --output results/manyih-coding-diagnosis
```
