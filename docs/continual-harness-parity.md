# Prime continual harness parity

Reference: the local `prime-agent-main (3).zip` snapshot, dated September 9, 2026,
extracted to the sibling `prime-agent-main` directory. Reference files were not
modified. Source paths in the Prime column are relative to
`packages/coding-agent/`. This matrix concerns the requested serialized continual
harness lifecycle, using Buffalo's existing Python kernel, provider accounting,
trajectory and child runtime.

| Behavior | Prime source/test | Buffalo source/test | Exact parity? |
| --- | --- | --- | --- |
| JSON canonical state | `src/core/refinement/refinement.ts:loadHarnessState/saveHarnessState`; `test/refinement.test.ts` | `src/threadweave/harness.py`; `test_missing_and_corrupt_state`, `test_atomic_save_roundtrip_and_failure` | Yes |
| JSONL refinement history | `appendGlobalRefinement/loadGlobalRefinementHistory`, session custom entries | `load_refinement_history/append_refinement_history`; `test_jsonl_skips_malformed_and_merges` | Yes; local JSONL lives in the persisted session's harness directory |
| Local default, explicit global | `planRefinement`, `handleRefineHostRequest`; `test/suite/agent-session-refine-skill.test.ts` | `request_refinement`, `Refine.run`; `test_schedule_status_coalesce_and_safe_boundary` | Yes |
| Collision-preserving merge | `mergeHarnessStates`; “without hiding colliding entries” | `merge_harness_states`; `test_merge_preserves_collisions_without_mutating_inputs` | Yes; both entries remain, local key is `local:id` |
| Local cannot modify global | `_planRefine/_applyRefine` scope selection | `HarnessStore.apply`; `test_local_cannot_update_or_delete_global_and_can_override` | Yes |
| Automatic refinement on by default | `settings-manager.ts:getAutoRefineSettings` | `RefinementPolicy`; `test_prime_defaults` | Yes |
| 25 assistant turns | serialized checkpoint and assistant-turn accounting | `refinement_checkpoint`; `test_interval_gate_cooldown_and_no_duplicates` | Yes |
| Compaction trigger | `_scheduleAutoRefineAfterCompaction`; serialized compaction tests | `Context.compact/refinement_compacted`; `test_compaction_pending_cooldown_disabled_fallback`, `test_idle_compaction_is_serviced_when_safe`, flow test | Yes |
| 20 minute cooldown | `_runSerializedRefineCheckpointAfterBackground` | `last_review_at/cooldown_seconds`; interval and compaction tests | Yes; Python setting uses seconds |
| Semantic review gate | `reviewAutoRefine`, `shouldRefine` | `review_refinement`; interval approve/decline tests | Yes; decline never calls planner |
| Bounded trajectories | `planRefinement` 80,000; `reviewAutoRefine` 40,000 | `refinement_input`; `test_planner_input_bounded_state_history_scope_and_empty` | Yes |
| Current harness/history/scope/instructions | `overviewForPrompt/historyForPrompt/planRefinement` | `refinement_input`; planner and schedule tests | Yes |
| Model-triggered `refine.run()` | `skills/refine/src/refine/__init__.py` | `kernel_api.py:Refine`; deterministic flow test | Yes |
| Scheduled turn-boundary execution | `handleRefineHostRequest`, `_runSerializedRefineCheckpoint` | `_run_turn/refinement_checkpoint`; safe-boundary, idle-continuation and flow tests | Yes; no application inside the requesting cell |
| Serialized application / stale work | `_autoRefineBranchVersion`, baseline comparison, in-flight guards | `Checkpoint`, `invalidate_refinement`, file lock and baseline comparison; cancellation/deferred-plan/conflict tests | Yes |
| Pending review / failed review recovery | `_pendingAutoRefineReview`, review/refine failure cooldown | `pending_review`, `finally` cleanup; `test_approved_review_deferred_until_safe_and_failure_unwedges` | Yes |
| Create/update/delete | `applyRefinementProposal`; all-kind CRUD tests | `apply_refinement_proposal`; all-kind CRUD and validation tests | Yes; stable IDs, versions, timestamps and snapshots |
| Rollback | `rollbackProposal`, recorded scope/path; copied-local-history test | `rollback_proposal/plan_refinement`; inverse, cross-session and copied-history tests | Yes; reversed applied edits target recorded store |
| Learned skills | Python `reference` plus `arguments` validation | `validate_edit`; skill validation and module execution tests | Yes; no executable body in JSON |
| Subagents | reusable specification; native `rlm` invocation | planner/root instructions and `harness` subagent entries | Yes; existing native child runtime |
| Immediate durable notices | `_recordRefinementNotice`, `messages.ts`; refine-extension notice tests | `refinement_notice`, persisted conversation message; notice and flow tests | Yes; zero edits have no notice |
| Cold-boundary merged digest | `_ensureHarnessDigestContext`, compaction head; prompt/compaction suite tests | `ensure_harness_digest`; digest/resume/compaction tests | Yes; ~6 entries/kind, ~180 character previews, ~5 refinements |
| Stale digest refresh / deduplication | `_appendHarnessDigestIfStale` | `ensure_harness_digest`; digest tests | Yes |
| Stable system prefix | `_applyRefine` leaves prompt unchanged | `python_instructions` contains no learned data; stable-prefix test | Yes |
| Resume persistence | session custom messages plus local/global files | persisted trajectory plus JSON/JSONL; deterministic restart test | Yes |
| No event-specific trigger graph | interval/compact/manual paths | only `refinement_compacted`, completed-turn count, explicit request | Yes; `test_only_interval_compact_or_explicit_requests_schedule` |

Python storage placement is deliberate: global files are under `DATA/harness`, and
local files under `DATA/sessions/SESSION_ID/harness`. Prime stores local refinement
history as custom records in its session JSONL; Buffalo stores those same records
in the session's `harness/refinements.jsonl`. Ordinary trajectory and model usage
still use Buffalo's SQLite telemetry. They do not own or mirror learned state.
Buffalo uses one serialized checkpoint implementation for its runtime; it does not
add Prime's separate interactive background-planning mode or extension system.

Legacy migration imports current active entries once, retaining IDs and versions.
`memory`, `prompt_note`, and `subagent_spec` map to `memory`, `prompt`, and
`subagent`; skills migrate only when they already have a valid Python callable
reference. Untranslatable embedded bodies are reported in
`harness/.sqlite-migrated.json`. The migration does not execute those bodies.
Historical tables remain readable for external forensic inspection, but the
continual harness never reads them after the migration marker is committed.

The new tests replace old tests whose assertions required lexical selection,
SQL version snapshots, atomic all-or-nothing edit batches, embedded skill bodies,
provenance reducers, completion/failure triggers or durable SQL request queues.
Those assertions conflict with the replacement specification. Unrelated parts
of mixed runtime tests remain in place.

The [deterministic trace](continual-harness-trace.json) contains actual runtime
events from the kernel session: scheduled request, completed refinement, durable
notice, root continuation, restart, digest, compaction review and decline. It uses
a mock provider and real file persistence; no live-model benchmark was run.

Validation completed September 9, 2026:

- 35 focused continual-harness cases pass, covering persistence, edits, scopes,
  scheduling, cancellation, cooldown, compaction, digest delivery, migration and rollback.
- The deterministic runtime/kernel flow passed and produced the linked trace.
  Provider-wire contracts and installed-callable integration also passed.
- Complete suite coverage: 457 passing cases and 10 skipped opt-in cases. The
  initial full invocation exposed a digest/compaction loop under a one-token
  budget and was interrupted. After fixing it, only the unfinished remainder
  and affected failures were run; no second full-suite invocation was used.
- Ruff reports no remaining errors; `git diff --check` passes. `uv build`
  successfully produced the source distribution and wheel.

The model-facing harness API is read-only. All model-driven mutations go through
scheduled refinement; the obsolete immediate CRUD helpers were removed too.
The compaction capacity regression has a bounded no-progress guard and its own
focused regression test.
