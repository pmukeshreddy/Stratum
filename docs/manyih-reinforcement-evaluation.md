# Real-model continual-harness evaluation

> Historical diagnosis and preserved run artifacts. No further evaluations are running.
> Prompt/delivery claims below predate the user's source-parity correction; current
> behavior is tracked in [the Prime source-parity audit](prime-refinement-specification.md).

This evaluation is separate from the canonical ManyIH score. Production refinement still uses a 25-assistant-turn interval, compaction, a 20-minute cooldown, and root-only automatic review. Explicit requests bypass review. Neither evaluator calls a refinement trigger, substitutes reviewer answers, requires edits, or changes those defaults.

## Diagnosis of the short canonical run

The interrupted `results/prime-refinement-100/` run contains 29 completed tasks. Their root trajectories are 2–4 assistant turns long. There are zero explicit refinement calls, compactions, and automatic reviews in those completed records. This is **not a 100-task score** and does not test the interval lifecycle. The partial artifacts were preserved.

The actual task-0 request in `full/buffalo/0/provider-calls.jsonl` exposes `ipython`, `await refine.run()`, `await refine.status()`, local/global scope, deferred application, and RLM delegation in SYSTEM context. The API was not hidden. The root solved the short task without selecting refinement. A transcript cannot establish an unexpressed reason for that choice.

Prime's normal guidance was compared directly with Buffalo's `context.py` and `kernel_api.py`:

- `packages/coding-agent/src/core/prompts/rlm.ts`: focused refinement after an observed repeated failure or reusable tactic; four editable kinds; immediate scheduling; continue ordinary work.
- `packages/coding-agent/skills/refine/SKILL.md` and `src/refine/__init__.py`: signatures, optional instructions, global scope, status, and coalescing.
- `core/system-prompt.ts`, resource loading, and the actual Prime evaluation request: the refine Python skill is installed and model-visible in coding runs.
- Prime's delegation guidance and both actual root prompts expose native RLM calls. Neither evaluation requires the model to spawn a child.

Buffalo now includes Prime's focused-refinement paragraph, concrete instruction examples, full signature, status meanings, and coalescing guidance. There is no ManyIH wording in these production instructions. Prime's automatic-resume claim was **not** ported: the user's explicit system-only/no-forced-turn requirement takes precedence. Prime's current skill documentation claims system rebuilding; its actual session implementation instead delivers notices and preserves the system prefix. This contradiction is retained in the reference and documented, not silently copied into Buffalo.

## Actual exposure probe

`results/manyih-reinforcement/exposure-01/` ran both systems on official ManyIH task 28 with implementation and independently authored tests. Both roots independently chose RLM for test authorship. A subsequent read-only capability question asked about available APIs without requesting refinement or delegation.

Both real model responses inspected live signatures/help, called `await refine.status()`, and explained `await refine.run(instructions=None, global_=False)`, scheduling, status booleans, scope, and typed harness state. Buffalo correctly described later SYSTEM delivery. Neither made an explicit refinement request. This proves model-visible exposure and an observed explanation of the API; **it is not counted as natural explicit-refinement evidence**.

## Running the separate evaluation

```sh
uv run python -m threadweave.evals.manyih_reinforcement \
  --tasks results/prime-refinement-100/full/buffalo \
  --prime-source ../prime-agent-main \
  --scenario interval \
  --output results/manyih-reinforcement/my-interval-run

uv run python -m threadweave.evals.reinforcement_report \
  results/manyih-reinforcement/my-interval-run
```

Modes: `interval`, `compaction`, `natural_explicit`, and the diagnostic `exposure`. All scenario manifests contain the actual work requests and input hashes. Work consists of implementing official task functions, independently authored tests, differential/exhaustive checks, complexity analysis, defect fixes, and durable project documentation. Each follow-up introduces substantive work; there are no empty “continue” turns. The root session, workspace, and harness persist across stages within a scenario. Both engines receive the same staged work. These are not independent canonical score tasks.

The compaction scenario provides a larger backlog of real modules, not padded messages or synthetic failures. The first complex-work pilot compacted in Buffalo before assistant turn 25. That is a successful compaction-path observation but cannot establish the first interval boundary. The subsequent interval workload uses smaller-output algorithm modules and ordinary concise test-reporting requests. No context/refinement threshold was lowered.

## Engines and measurement

Buffalo uses its production `Runtime`, ordinary `interact` follow-ups, and configured provider. Prime uses its unmodified source SDK, session runtime, resource loader, Python kernel, reviewer, planner, and serialized headless scheduling. Source dependencies were installed with `npm ci`; Prime source was not edited.

`prime_reinforcement.mjs` is an evaluation-only inference adapter. It registers the same real `gpt-6-astra`/`xhigh` provider through Prime's public provider registry and uses Buffalo's existing authenticated Codex transport. No reviewer/planner response is mocked. Both refinement auxiliaries use their normal non-reasoning policy and provider-supported minimum effort. The actual model catalog lacks this model in Prime, so the adapter supplies its model metadata, including a 96k context allocation matching Buffalo's configured allocation. **Compaction policies remain each runtime's own implementation and need not compact on the same turn.** This is not a stock-provider comparison. The initial adapter does not carry opaque provider reasoning continuation between Prime turns; this transport difference must be considered when interpreting model decisions.

The driver observes Prime's existing kernel refine requests with a transparent proxy that forwards each already-originated call once. It never originates a refine request. Session events, raw provider requests/responses, work stages, and harness files are retained. Early pilots predate this observer and are explicitly marked as such in reports. Later paired runs wait for Prime's normal descendant/quiescence barrier before admitting the next work stage; the initial pilot admitted follow-ups at ordinary root idle.

`reinforcement_metrics.py` reads Buffalo's committed database, request artifacts, and harness JSON. `reinforcement_report.py` reads Prime's actual provider/event traces. These are observers, not production dependencies. They distinguish:

- `NO_TRIGGER`
- `TRIGGER_REVIEW_DECLINED`
- `TRIGGER_APPROVED_EMPTY_PLAN`
- `TRIGGER_APPLIED_EDIT`
- `REFINEMENT_FAILED`
- `TRIGGER_PENDING` for an unfinished request

Counts include root turns, REPL, RLM, explicit requests, compactions, interval and compaction review requests, approvals/declines, planners, non-empty proposals, applied refinements, and typed edits. Canonical task records also include threshold/compaction reachability, root-turn distributions, and lifecycle outcomes. Root request numbers are one-based in lifecycle reports. Reviewer inputs preserve the production “turns since last review” value.

Delivery proofs link an applied edit's audit event to its harness JSON and the next actual root request. They record the entry/version advertised in SYSTEM, and inspect ordinary messages for notices or audit summaries. A later legitimate root edit can advance the disk version beyond the originally applied version; that is reported separately, not misclassified as a failed write.

## Evidence and acceptance status

Live evidence is in each scenario's `comparison.json`, `provider-calls.jsonl`, event history, and workspace. The complex pilot has already demonstrated actual Buffalo compaction → reviewer approval → planner → memory creation → version 1 in the next legitimate SYSTEM request, with no synthetic user notice. The root subsequently updated that memory to version 2 through native CRUD after completing the audit.

This document is not a claim that all acceptance cases have been observed. Real-model decline, approved-empty, first-25-turn application, natural explicit selection, both-engine compaction, and the final canonical 100 require their own recorded evidence. Missing outcomes must remain missing; unit tests cannot fill those cells.

Regression tests: `test_reinforcement_evaluation.py` checks all four review/application outcomes, 24 → 25 boundary accounting using ordinary runtime requests, immutable observation, and scenario policy invariants. `test_refinement_invariants.py` covers SYSTEM-only delivery/no forced continuation. The full suite after the evaluator addition passed **497 tests, 10 skipped**; later observer changes have focused regression coverage.

The deletion inventory, surviving implementation, Prime mapping, and deliberate source differences remain in [continual-harness-parity.md](continual-harness-parity.md).
