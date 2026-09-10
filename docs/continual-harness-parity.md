# Continual harness replacement

> Historical report, superseded by [the Prime source-parity audit](prime-refinement-specification.md).
> Its SYSTEM-only/no-notice requirements were explicitly reversed by the user.
> The text below is retained as an implementation-history artifact, not a description
> of the current runtime or a current parity claim.

The requested loop is implemented. It is **not identical to the entire supplied Prime checkout**: that checkout deliberately injects model-facing refinement notices and preserves a stable system prompt. Buffalo follows the explicit requirements to remove notices and rebuild SYSTEM context instead. Prime source files were read but not modified.

## Final flow

```text
recent conversation
  → root-only automatic trigger (25 assistant turns or compaction; 20-minute cooldown)
  → reviewer: shouldRefine / rationale / optional instructions
  → if approved, planner
  → typed create/update/delete edits (prompt | memory | skill | subagent)
  → baseline conflict check and safe-boundary application
  → atomic harness JSON save + audit history
  → next legitimate model request loads merged local/global harness
  → reconstruct root SYSTEM prompt
```

Explicit `refine.run()`, CLI/chat refinement, and rollback skip automatic review. A declined review never calls the planner. An approved review may produce zero edits. Neither learning nor an empty plan changes task completion, runnable state, or the number of root turns.

Planning may overlap the current tool execution, as in Prime's serialized mode. Application waits for a safe boundary. A concurrent checkpoint consumes the same plan once; branch changes invalidate stale work; changed target entries fail optimistic checks individually. Unrelated edits remain applicable. Explicit background planning failures get Prime's boundary retry; automatic failures do not bypass cooldown. Pending requests/plans/counters are in memory and reset on runtime restart, as in Prime; applied state/history persist.

## Deleted Buffalo machinery

Entire files removed:

- `src/threadweave/refinement_evidence.py`: `bounded_records`, `trajectory_evidence`, `opportunity`, `opportunity_key`.
- `src/threadweave/harness_migration.py`: `migrate_sqlite_harness`, `migrate_session_refinement_history`; retired SQLite/sidecar adapters, alias conversion, migration provenance and old-policy conversion.
- `src/threadweave/evals/refinement_use.py`: `meaningful`, `source_candidates`, `refinement_use`, `use_summary`; before/after candidate attribution and intended-application classification.
- `tests/test_actionable_refinement.py` and `tests/test_refinement_evidence.py`: evidence-trigger, semantic-requirement, application, and novelty-schema tests.
- `docs/continual-harness-trace.json`: obsolete demonstration of notices and forced continuation.

Removed from retained files:

- `refinement.py`: `current_refinement_opportunity`, `LEARNING_EVIDENCE_PROMPT`, evidence-trigger/cooldown exceptions, original-task extraction for learning, bounded evidence collectors, reviewer-as-verifier instructions, candidate/requirement comparison, and notice/context writes.
- Reviewer/result/planner fields: `requirement_findings`, `learning_assessment`, `reviewer_assessment`, `trajectory_evidence`, `in_task_opportunity`, `original_task`, `application`, `opportunity`, `conversation_omitted_chars`, `refinement_history_coverage`, entry `coverage`, result-level `timestamp`. Entry/history timestamps remain.
- Removed assessment/application subfields: `before`, `new_observation`, `reusable_rule`, `duplicate_comparison`, `later_use`; application `status`, `issue`, `evidence_events`, `next_action`, `validation`. Typed edit `before/after` snapshots remain for rollback/conflict handling.
- `harness.py`: `refinement_notice`; audit summaries/outcomes in model-facing harness rendering. Unknown proposal/edit fields are discarded using Prime's normalization rules.
- `context.py`: `ensure_harness_digest`, `messages(refresh_harness=...)`, user-role harness injection, stale-digest detection/deduplication, and `harness_digest` message field.
- `runtime.py`: `learned` completion/runnable gates, refinement-driven reopening of idle/completed sessions, digest injection during turn admission/resume/fork.
- Event producers removed: `refinement_opportunity`, `refinement_notice`, `refinement_continuation`, `harness_digest`.
- `semantic_state.py`: `harness_versions` in compaction's semantic tree; learned state now arrives through SYSTEM context.
- `auxiliary.py`: `refinement_stage`; `refinement.py`'s provider-helper re-export removed. The evaluation wrapper imports the actual helper directly.
- Obsolete SQLite schema: `state_entries`, `state_versions` and their immutable triggers, `refinements`, `refinement_requests`, `refinement_runs`, `skill_outcomes`. New databases never create them; schema 13 drops them from existing databases. Harness JSON and normal audit events remain intact. There is no legacy learned-state import.
- Obsolete metrics/output: reducer-stage attribution, refinement continuation counts, notice-delivery counts, optional-refinement budget exhaustion/defer counts, retired state-retrieval counters, intended-application/use classifications, and `refinement-use.jsonl`.
- Deleted obsolete tests in otherwise retained files: bounded notices, stable system prefix, untouched/stale digest delivery, forced idle continuation, legacy SQLite import, legacy local-sidecar import. Mixed tests retain unrelated runtime/REPL/RLM assertions and now assert SYSTEM-only visibility.

Historical benchmark artifacts under `results/` are preserved as historical evidence. They are not runtime code and do not describe the new loop. Unrelated pre-existing untracked result directories were not changed.

## Remaining implementation and Prime mapping

Prime paths below are relative to `../prime-agent-main/`.

| Prime source | Buffalo source | Responsibility |
| --- | --- | --- |
| `packages/coding-agent/src/core/refinement/refinement.ts` | `src/threadweave/refinement.py` | Reviewer/planner prompts, inputs, parsing, planning, rollback target |
| Same file | `src/threadweave/harness.py` | Four-kind schema, normalization, CRUD, snapshots, versions, timestamps, scope merge, persistence, conflict validation, rollback, bounded overviews/history |
| `packages/coding-agent/src/core/agent-session.ts` serialized checkpoint/background methods | `src/threadweave/refinement.py`, call sites in `runtime.py` | Counters, cooldown, root restriction, explicit requests, safe boundaries, stale-plan cancellation, exact-plan application |
| `packages/coding-agent/src/core/settings-manager.ts:getAutoRefineSettings` | `src/threadweave/models.py:RefinementPolicy` | Defaults: enabled / 25 turns / compaction / 1200 seconds |
| `refinement.ts:formatHarnessStateForPrompt`, `core/system-prompt.ts`, `agent-session.ts:_rebuildSystemPrompt` | `src/threadweave/context.py:system_prompt/messages`, `harness.py:format_harness_state` | Harness formatting and request-time system construction; deliberate difference described below |
| `core/session-manager.ts` custom entries; `appendGlobalRefinement` | `src/threadweave/storage.py`, `harness.py` | Local session audit and global JSONL history |
| `core/messages.ts:convertToLlm` | `src/threadweave/context.py`, `trajectory.py` | Separation of conversation and refinement audit; notices are omitted entirely |
| `prime-agent-runtime/src/rlm/harness.py`, `skills/refine/src/refine/__init__.py` | `src/threadweave/kernel_api.py`, `host_api.py` | Native explicit refinement/status and four-kind CRUD |
| `refinement.ts` provider calls | `src/threadweave/auxiliary.py` | Existing accounted provider transport and non-reasoning structured inference |
| Prime command/RPC/UI entry points | `src/threadweave/tools.py`, `cli.py`, `chat.py`, `daemon.py`, `terminal.py` | Existing external API entry points and human-facing audit output |

The two core reinforcement files are `refinement.py` and `harness.py`. The integrated files listed above provide scheduling, persistence, system construction, or public APIs; they are not another learner. `migrations.py` only retires dead tables. Measurement remains in `evals/activity.py`, `evals/observability.py`, `evals/manyih_full_harness.py`, and `evals/latency.py`; it never schedules learning. `evals/harness.py` enforces provider settings.

## Executable invariants

| Behavior | Tests |
| --- | --- |
| 25-turn trigger, approve/decline, cooldown, no duplicates | `test_continual_harness.py::test_prime_defaults`, `test_interval_gate_cooldown_and_no_duplicates` |
| Compaction, cooldown pending, failed trigger consumption, disabled automation | `test_compaction_pending_cooldown_disabled_fallback`, `test_idle_compaction_is_serviced_when_safe`, `test_only_prime_triggers_schedule_review`, `test_refinement_invariants.py::test_compaction_failure_consumes_trigger_and_sets_cooldown` |
| Root only, explicit bypass, optional empty edits | `test_refinement_invariants.py::test_children_cannot_trigger_review_or_explicit_refine`, `test_approved_review_may_produce_no_edits`; schedule and wrapper tests |
| Create/update/delete every kind, stable IDs/versions, atomic persistence | `test_create_update_delete_stable_id_versions`, `test_atomic_save_roundtrip_and_failure`; parametrized kernel CRUD tests |
| Local/global merge and scope isolation | `test_merge_preserves_collisions_without_mutating_inputs`, `test_local_cannot_update_or_delete_global_and_can_override`, `test_global_planner_sees_only_global_store` |
| Rollback, conflicts | `test_inverse_rollback_restores_create_update_delete`, `test_rollback_history_across_sessions`, `test_copied_local_history_rollback_targets_original_file`, `test_background_baseline_rejects_same_entry_kernel_write` |
| Stale plans and exact plan applied once | `test_branch_cancellation_does_not_apply_or_requeue_old_explicit_work`, `test_new_explicit_request_supersedes_plan_that_ignores_abort`, `test_ready_plan_waits_for_tools_then_single_consumer_applies`, `test_paused_plan_is_applied_once_without_replanning` |
| No forced continuation; next real request gets new SYSTEM; no audit in conversation, user messages, trajectory, or compaction; restart | `test_refinement_invariants.py::test_completed_turn_learns_without_continuation_and_next_request_uses_system` (with and without restart), `test_idle_refinement_does_not_schedule_root_continuation` |
| All typed edits reflected in SYSTEM | `test_refinement_invariants.py::test_typed_edits_update_system_on_every_request` |
| Provider settings and auxiliary graph separation | `test_refinement_parity.py`, `test_refinement_wrapper.py` |
| Real kernel scheduling, resume, compaction | `test_continual_harness_flow.py`, `test_real_turn_plans_while_tool_waits_and_next_model_observes_system_state` |
| Separate REPL/RLM/review/edit measurement | `test_manyih_observability.py` |

## Remaining differences from the supplied Prime Agent

1. **Required behavioral divergence:** Prime's current `agent-session.ts:_applyRefine` says “The prompt stays byte-identical so the provider prefix cache survives; the notice carries the change.” It calls `_recordRefinementNotice`; `messages.ts:createRefinementNoticeMessage` explicitly says the notice passes `convertToLlm`. Buffalo instead reconstructs SYSTEM context for each request and creates no notice or learner-driven root continuation.
2. **Required audit isolation:** Prime's prompt formatter includes recent refinement summaries/outcomes. Buffalo's system formatter includes current entries only; detailed refinement results remain in audit and reviewer/planner history, not ordinary conversation or its trajectory projection.
3. **Native implementation:** Buffalo uses its Python runtime, SQLite audit events, file locks and existing provider accounting. Prime uses TypeScript sessions and custom entries. Reviewer/planner inputs use JSON-labeled sections and JSON-serialized role-bearing conversation in Buffalo, versus XML-labeled sections and Prime's text conversation serializer. Same input categories and 40k/80k trajectory limits; same 40-entry/240-character refinement overview and last-20 history bounds. Global planning sees global state; local planning sees both scopes.
4. **Provider/runtime limits:** Buffalo uses configured output caps bounded by 4096/32000 and its existing cumulative resource limits. Prime derives these caps from model limits. Buffalo's subscription transport uses the least supported reasoning level when literal `none` is unavailable. No optional-work/evidence budget heuristic admits or denies refinement.
5. **Host integrations:** Prime extension hooks, daemon protocol, TUI, and full non-serialized interactive scheduling are not ported. Buffalo uses one serialized checkpoint path in its existing modes. Applied JSON/history survive resume; incomplete plans are cancelled, not replayed. Legacy Buffalo SQLite learned-state imports were deliberately removed.

No full-product parity or identical model outputs are claimed. The reviewer/planner behavior, typed edits and default serialized scheduling are ported from the supplied source; context reinjection follows the user's explicit system-only requirements.

## Validation record

- `uv run pytest -q`: **492 passed, 10 skipped**, in 121.07 seconds. Two pre-existing Python `forkpty()` deprecation warnings occurred in the terminal test. The earlier full run found one obsolete stable-system assertion; that assertion was replaced with the new system-state invariant and the complete suite rerun successfully.
- After adding Prime's explicit empty-edits instruction to the planner: `uv run pytest tests/test_refinement_invariants.py -q`: **8 passed**.
- `uv run ruff check src tests` and `git diff --check`: passed.
- Source scan: no evidence-gate, semantic-learning-schema, synthetic-notice, forced-continuation, or digest-injection identifiers remain under `src/`. Refinement audit event kinds are excluded from the model-facing trajectory projection.
- Source SHA-256: Prime `refinement.ts` = `81438d5b4acb013c40eae05f12402eb8d7052df3a522baec972c773fd5431f86`; `agent-session.ts` = `c8c3b5b2793472ef5d7d1f949ed65786dc8b44529e0bc8566ea5505112d835f2`; `settings-manager.ts` = `59da74a830ee3b1035d79a3f280e940cc639e9f51c5261bea5206c03612ee942`.

The canonical benchmark uses `threadweave.evals.manyih_full_harness`, evaluator commit `3287f82436fff86d7506b7737dbd3b5e2e524d01`, fixed task IDs 0–99, and unchanged production refinement defaults. The partial `results/prime-refinement-100/` run was stopped at 29 completed tasks at the user's direction; it is not a 100-task score. The runner's obsolete hardcoded previous-score assertion was removed; identity checks for task IDs, prompts, workspaces, and official evaluator revision remain. See [manyih-reinforcement-evaluation.md](manyih-reinforcement-evaluation.md) for the separate real-model paired evaluation, root-facing guidance port, and lifecycle observability. Evaluation-only modules are `manyih_reinforcement.py`, `prime_reinforcement.mjs`, `prime_transport.py`, `reinforcement_metrics.py`, and `reinforcement_report.py`; none changes production refinement policy.
