# Prime refinement specification and port audit

This supersedes the earlier system-only delivery requirement and the previous
parity report. The supplied Prime checkout is authoritative. Evaluation processes
were stopped on 2026-09-10; their results directories were preserved. No new
real-model evaluation is part of this work.

## Root guidance mapping (Job A)

| Prime source | Buffalo surface | Decision |
| --- | --- | --- |
| `core/prompts/rlm.ts`, `REPL_CONTROL_PROMPT`: continual harness CRUD and `global_=True` | `context.python_instructions` | Same text, product-name substitution only. Deleted Buffalo's extra direct-write and per-request SYSTEM delivery explanations. |
| `core/prompts/rlm.ts`, `buildRlmPrompt`: “Treat continual harness refinement…” | `context.python_instructions`, root only | Retained verbatim, including diagnose/update/validate/record, kind selection, immediate return, normal work, and focused edits. These are Prime instructions, not evaluation additions. |
| `core/prompts/rlm.ts`: read SKILL.md, help/dir/signature discovery | Root's refine skill description and file location | Exposed on demand. No stronger invocation policy. |
| `skills/refine/SKILL.md`, entire file | `builtin_skills/refine/SKILL.md` | Exact copy. The two examples and “One request per turn” belong here, not an extra unconditional SYSTEM block. |
| `skills/refine/src/refine/__init__.py` | `kernel_api.refine_run` / `refine_status` module exports | Python signature and false-flag omission must match. Host omission is distinct from explicit false. |
| `core/system-prompt.ts`: visible skill filtering and catalog | Root-only skill guidance | Child sessions must not advertise the refine skill. |
| `core/resource-loader.ts`; `core/package-manager.ts` built-in skill resolution | Bundled readable refine guide | Prime loads a skill catalog, not the entire refine guide into SYSTEM. |
| No Prime root equivalent: background/tool-overlap mechanics, automatic 25/20-minute policy, repeated no-edit assurances | Removed from Buffalo root prompt | Runtime policy is not additional model encouragement. |
| `core/refinement/refinement.ts`, `formatHarnessStateForPrompt` | Harness digest | Ported the actual guidance and bounded recent-history formatting, without Buffalo additions. |

Prime's own skill guide says “rebuilds the system prompt” and “resumes you
automatically.” That prose is retained for prompt parity, but source and tests
govern runtime behavior: `_applyRefine` preserves SYSTEM and appends a notice;
the notice alone does not start an idle turn.

## Section 16: test inventory and validation

The [complete candidate test map](prime-refinement-test-map.md) currently inventories
317 definitions: 245 applicable to the scoped Buffalo surfaces, 136 mapped passing,
109 explicitly open, and 72 explained structural/out-of-scope exclusions. These
are definitions, not expanded parameter cases. Every candidate has a disposition;
the inventory bookkeeping is complete, but the exact-variant parity proof is not.
Open rows are not automatically runtime bugs, and related coverage is not counted
as an exact pass. There is no claim of 245 passing applicable invariants.

Prime's eight core suites were executed against the supplied checkout:
`refinement.test.ts`, `refinement-outcome-message.test.ts`,
`provider-retry.test.ts`, `compaction-serialization.test.ts`,
`suite/agent-session-refine-skill.test.ts`,
`suite/agent-session-refine-extension.test.ts`,
`suite/agent-session-serialized-refine.test.ts`, and
`suite/serialized-refine-config-integration.test.ts`.
Result: **179 passed, 8 files passed**. These use deterministic providers; no
benchmark or real-model diagnostic was launched.

## Actual delivery and lifecycle now implemented

```text
root trajectory
→ explicit refine.run() OR normal interval/compaction trigger
→ reviewer for automatic requests only
→ approved review or explicit request → planner / extension proposal / rollback
→ exact plan, branch identity, live touched-entry conflict checks
→ apply valid typed edits independently; persist JSON and audit
→ display-only outcome; applied-only model-visible refinement notice
→ preserve SYSTEM prefix; next legitimate root turn sees the notice
```

Cold session/resume contexts and compaction heads also receive the merged harness
digest as Prime-style user context. Presentation outcomes and slash-command
receipts are filtered from LLM context. Refinement notices themselves are **not**
filtered from future reviewer/planner trajectories, because Prime includes them.
Applied edits alone do not manufacture another root turn.

Serialized mode permits planning after the primary model response while tools
execute; the next primary request waits for the boundary consumer. Interactive
public refinement permits planning overlap but blocks new turn admission while
waiting to apply. Independent manual calls serialize without coalescing. The
root skill coalesces requests and preserves omitted scope/instructions.

## Sections 17–22: implemented corrections

| Section | Corrections and remaining audit work |
| --- | --- |
| 17 | Background/safe-boundary exact-plan ownership, single-consumer claims, public planning/apply barriers, interactive path, explicit failure boundary retry. Exact variants still open in the test map. |
| 18 | Root module exposure and docstrings, active-turn checks, immediate receipt, explicit bypass, coalescing, supersession, host omission versus explicit false. Removed the extra native model-facing refine tool and human/rollback arguments from the root scheduling API. Manual CLI/RPC uses public refinement, not root coalescing. |
| 19 | Late planner failures and extension skips are classified as invalidated against the originating branch. Aborted root responses clear pending explicit work even without a separate host abort. Actual task cancellation remains authoritative when providers ignore it. Automatic apply rechecks branch identity before stamping cooldown. Direct apply owns the admission barrier. Drain re-reads live task ownership and compaction/interval settings; explicit disposal errors are best effort, while normal boundary failures remain visible. Claimed application and memoized shutdown regressions pass. Remaining exact variants are explicitly open. |
| 20 | Prime XML input blocks; ordinary-message serializer; SYSTEM/developer exclusion; model-visible notices and compaction summaries; 40,000/80,000 UTF-16 conversation limits; 40 entries/kind, 240-character content, 20 history records. Global planner receives global state only. |
| 21 | Active model maxTokens, not the primary turn budget, controls reviewer/planner caps (4096/32000). Bundled metadata is projected from the supplied Prime catalog; custom definitions use Prime's defaults and explicit overrides. Chat one-shot streaming, developer-role and provider compatibility settings, finish-reason classification, Retry-After and EOF handling are ported. Native refinement now preserves incomplete-response length and retry delays. Non-JSON constants are rejected. Prime only special-cases error/length tags before JSON parsing; actual cancellation is enforced by session/task ownership. No extra reasoning pass or fallback planner. |
| 22 | Typed edits, baseline touched-entry conflicts, partial failures, sequential same-entry edits, scope display IDs, rollback recorded paths, local/global history, atomic symlink-preserving persistence. Removed Buffalo's extra file locks. Python CRUD now uses Prime's Python normalization/validation, distinct from planner validation as in Prime. |

## Production files changed by the source-parity work

- `context.py`, `refinement.py`, `harness.py`, `runtime.py`,
  `auxiliary.py`, `models.py`, `providers.py`, `subscription.py`,
  `kernel_api.py`, `host_api.py`, `tools.py`, `daemon.py`, `chat.py`,
  `cli.py`, and `trajectory.py`.
- Added `refinement_context.py`, `refinement_retry.py`, and the copied
  `builtin_skills/refine/SKILL.md`.
- Existing evaluation artifact readers/pinned-provider bookkeeping were adjusted
  only where the corrected production representation invalidated them. No
  evaluation scenario or scheduling rule was changed and no evaluation was run.

The workspace already contained other user/earlier-turn modifications. Those
were preserved; this list is not a claim of ownership of every dirty file.

## Deleted or restored behavior

Deleted in this source-parity pass: unconditional extra refine examples/policy
explanations in SYSTEM; forced lowest-supported reasoning selection; native
`refine` tool handler, registration and `RefineArgs`; human/rollback compatibility
arguments on `request_refinement`; `HarnessStore.lock` and extra refinement file
locking; Python `create_prompt/update_prompt/delete_prompt` aliases without Prime
equivalents; planner validation applied to generic Python CRUD; obsolete CLI
scheduled-result rendering; the separate paused `pending_plan` cache and its
Buffalo-only preservation tests. Background plans now apply directly under the
single-consumer claim, including same-boundary explicit follow-up requests.

Restored because actual Prime requires it: applied-only model-facing notices,
presentation-only outcomes, cold harness digests, compaction-head digest delivery,
interactive/public refinement, valid final-work draining, Prime extension
proposal/skip hooks, the exact on-demand refine guide, and Python
`Harness.plan_refinement` (a passive string-list helper, **not** an automatic
opportunity gate).

Earlier deleted architecture remains deleted: `refinement_evidence.py`,
`harness_migration.py`, `evals/refinement_use.py`, semantic requirement learning
fields/prompts, evidence/opportunity pre-review gates, and forced learning
continuations. Historical report/run artifacts are retained and labeled as such.

## Verification log

- Earlier clean full Buffalo suite: 541 passed, 10 skipped.
- Following lifecycle/CRUD/command ports: targeted 132 passed.
- Production 25-turn scheduler regression: 3 passed (decline, approved-empty,
  approved-edit), deterministic model responses and actual runtime/tool boundaries.
- Prime reference suites: 179 passed.
- Later full Buffalo suite: 561 passed, 10 skipped, 1 failure. The failure was an
  old daemon-recovery fixture still invoking the deleted native refine tool.
  It now uses the Python skill; the isolated regression passed.
- Full Buffalo run after the provider-wire corrections: 566 passed,
  10 skipped, 2 warnings (141.34 seconds).
- Final full Buffalo run after direct background-plan consumption and removal of
  the paused-plan cache: **567 passed, 10 skipped, 2 warnings** (143.62 seconds).
- `git diff --check`: passed. All test references in the candidate map resolve.
- No benchmark, stress evaluation, paired real-model run, or real-model diagnostic
  was started. Existing diagnostic processes were stopped and their artifacts kept.

### Latest narrowed pass: sections 16, 19 and 21

Only focused regressions were run, as requested. The latest command selected
`test_prime_refinement_races.py`, `test_prime_refinement_provider.py`,
`test_prime_refinement_lifecycle.py`, `test_refinement_parity.py`,
`test_refinement_wrapper.py`, `test_subscription.py`, and
`test_continual_harness_gaps.py`: **166 passed in 5.11 seconds**.
The earlier intermediate runs were 89 passed/1 obsolete non-streaming fixture
failed, then 132 passed, then 142 passed. The fixture now emits the SSE requested
by Prime's actual one-shot provider path. No full-suite rerun was made in this
narrowed pass; the 567-pass full-suite result above predates these changes.

Scoped Ruff checks and `git diff --check` passed. The changed native bridge built
successfully against the already available pinned Codex source; no inference was
performed. Existing binaries and diagnostic artifacts were preserved.

Production changes in this narrowed pass:

- `refinement.py`: stale-error classification, aborted-turn cleanup, cancellation
  checks, apply barrier, post-apply branch check, live drain ownership, disposal
  error handling, exact JSON constant/tag behavior.
- `models.py`, `auxiliary.py`, `runtime.py`: selected-model metadata and caps;
  subscription selection recomputes the cap before request registration.
- New `refinement_model.py` and `prime_refinement_models.json`: only the metadata
  and one-shot request construction needed by the supported transports.
- `providers.py`, `refinement_retry.py`, `subscription.py`,
  `native/inference.rs`: streaming/stop reasons, retry classifications/delays,
  backoff cancellation, native incomplete-response handling.
- `evals/harness.py`: existing pinned-provider bookkeeping follows the corrected
  production cap. No evaluation behavior or scenario was changed or executed.

Deleted behavior in this pass: deriving refinement caps from
`ProviderConfig.max_output_tokens`; unconditional rejection of an `aborted`
provider tag (Prime parses its text, then the session checks authoritative
cancellation); accepting NaN/Infinity as JSON; requiring a Chat `[DONE]` sentinel
for refinement when Prime accepts stream EOF; reporting pending explicit disposal
failures as ordinary command failures. No reinforcement files were deleted here.

Exact section-21 source mapping:

| Prime source | Buffalo |
| --- | --- |
| `refinement/refinement.ts:refinementMaxOutputTokens/autoRefineReviewMaxOutputTokens` | `refinement_model.py:refinement_output_limit` |
| `core/model-registry.ts` custom-model defaults/overrides; `packages/ai/src/models.generated.ts` | `models.py:ModelMetadata`; `refinement_model.py:selected_model`; `prime_refinement_models.json` |
| `packages/ai/src/providers/openai-completions.ts:buildParams/getCompat/mapStopReason` | `refinement_model.py:chat_refinement_body`; `refinement_retry.py:completion_metadata`; `providers.py` |
| `packages/ai/src/utils/stream-failure.ts` | `refinement_retry.py:classify_failure/retry_after_ms` |
| `core/provider-retry.ts:completeWithProviderRetry` | `refinement_retry.py:complete_refinement` |
| Codex event normalization plus `openai-responses-shared.ts:mapStopReason` | `native/inference.rs:error`; `subscription.py:collect` |

Catalog provenance: 649 supported-transport/underlying OpenAI model definitions
across 16 endpoint keys, projected without heuristic model-name matching.
Source `packages/ai/src/models.generated.ts` SHA-256:
`9375ca499f6bf0b01a10dd399f43cb4c0d221c6d8bbd16c503fb45a5feeec974`.
Custom/proxied models can provide `provider.model_metadata` with Prime's
`maxTokens`, `reasoning`, and `compat` fields. This metadata is separate from the
four-kind harness schema. Prime's custom-model default is 16384 output tokens and
`reasoning=false`; it is not a newly invented fallback.

The native Codex and Prime direct-fetch clients still differ in transport-internal
authentication recovery and the detail retained for opaque SDK errors. The port
preserves available classifications and retry delays, but does not claim every
provider/SDK diagnostic is byte-for-byte identical. Prime's other provider APIs
are not implemented by Buffalo; this work did not add unrelated providers.

## Known remaining work — no parity claim

1. Close the 109 explicitly open exact-variant proofs in the completed inventory.
2. Audit/port cold first-context commit rollback/re-arming and cleared-input behavior;
   Buffalo currently persists its initial digest before model-request ownership is
   fully committed, unlike Prime's staged input mechanism.
3. Complete refinement command ownership/order against ordinary queued inputs and
   branch/abort/restart races. The new refinement-command queue is not yet proven
   equivalent to Prime's unified session-input pump.
4. Model/output-cap parity for supported transports is implemented. Remaining
   transport differences concern native authentication recovery and opaque SDK
   error detail, not the 25-turn scheduler or planner output budget.
5. Audit exact formatter/JSON edge behavior: Python ordering is not JavaScript
   localeCompare, and numeric JSON edge formatting is not yet proven identical.
6. Finish built-in skill override/resource filtering and all remaining serialized,
   interactive, disposal, and failure variants listed in the table.
7. Finish Python CRUD edge normalization, including generic reference/arguments/
   metadata dictionary coercion; planner validation and Python helper validation
   are intentionally distinct in Prime and must remain so.

SELF-REINFORCEMENT PARITY: NOT COMPLETE
