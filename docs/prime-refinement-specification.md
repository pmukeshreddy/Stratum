# Prime refinement specification and port audit

Subsequent change: [learning-quality admission](refinement-learning-quality.md)
intentionally tightens reviewer/planner criteria while retaining this lifecycle.
The source-parity audit below describes the preceding implementation.

This supersedes the earlier system-only delivery requirement and the previous
parity report. The supplied Prime checkout is authoritative. Evaluation processes
were stopped on 2026-09-10; their results directories were preserved. No new
real-model evaluation is part of this work.

## Root guidance mapping (Job A)

| Prime source | Buffalo surface | Decision |
| --- | --- | --- |
| `core/prompts/rlm.ts`, `REPL_CONTROL_PROMPT`: continual harness CRUD and `global_=True` | `context.python_instructions` | Same text, product-name substitution only. Deleted Buffalo's extra direct-write and per-request SYSTEM delivery explanations. |
| `core/prompts/rlm.ts`, `buildRlmPrompt`: “Treat continual harness refinement…” | `context.python_instructions`, root only | Retained verbatim, including diagnose/update/validate/record, kind selection, immediate return, normal work, and focused edits. These are Prime instructions, not evaluation additions. |
| `core/prompts/rlm.ts`, `buildSubagentGuidance`: “Persist genuinely reusable delegation patterns…” | Visible refine skill plus enabled delegation | Retained verbatim and only advertised when Prime's equivalent capabilities are visible. |
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

The [complete test map](prime-refinement-test-map.md) inventories **317 test
definitions in 42 files: 235 applicable, 235 mapped and passing, 0 unmapped
applicable, and 82 individually explained structural/out-of-scope exclusions**.
Counts refer to definitions, not expanded parameter cases. All 236 regression
references resolve. Ten formerly open candidates were classified as structural
exclusions after checking their exact APIs, not counted as implemented behavior;
the map identifies each one and its supported-path coverage.

Inspected suites include serialized refinement, refine skill/extension, refinement
units, outcomes/notices, harness Python CRUD, settings, provider retry, serializers,
prompt/digest delivery, action/input races, queues, daemon/RPC/config integration,
history, rollback, conflicts, navigation, abort and disposal. The map lists every
file, source test name and disposition; no applicable candidate remains open.

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
| 17 | Background/safe-boundary exact-plan ownership, single-consumer claims, public planning/apply barriers, interactive path, explicit failure boundary retry; no duplicate apply or forbidden model overlap. Queued input remains behind the current refinement boundary and slash commands share FIFO ownership. |
| 18 | Root module exposure and docstrings, active-turn checks, immediate receipt, explicit bypass, coalescing, supersession, host omission versus explicit false. Removed the extra native model-facing refine tool and human/rollback arguments from the root scheduling API. Manual CLI/RPC uses public refinement, not root coalescing. |
| 19 | Late planner failures and extension skips are classified as invalidated against the originating branch. Aborted root responses clear pending explicit work even without a separate host abort. Actual task cancellation remains authoritative when providers ignore it. Automatic apply rechecks branch identity before stamping cooldown. Direct apply owns the admission barrier. Drain re-reads live task ownership and compaction/interval settings without a speculative checkpoint. Valid work drains once; stale results/failures cannot write. Explicit-boundary cancellation and interactive approved-plan failures consume cooldown exactly where Prime does; stale disposal plans do not. Claimed application, navigation, shutdown and late-result variants pass. |
| 20 | Prime XML input blocks; ordinary-message serializer; SYSTEM/developer exclusion; model-visible notices and compaction summaries; 40,000/80,000 UTF-16 conversation limits; 40 entries/kind, 240-character content, 20 history records. Global planner receives global state only. |
| 21 | Active model maxTokens, not the primary turn budget, controls reviewer/planner caps (4096/32000). Bundled metadata is projected from the supplied Prime catalog; custom definitions use Prime's defaults and explicit overrides. Chat one-shot streaming, developer-role and provider compatibility settings, finish-reason classification, Retry-After and EOF handling are ported. Native one-shot refinement uses WebSocket first and pre-stream-only SSE fallback, bypasses SDK retries/401 recovery, and preserves provider error codes, terminal statuses, usage-limit delays, malformed protocol errors and SSE EOF behavior. Non-JSON constants are rejected. Prime only special-cases error/length tags before JSON parsing; actual cancellation is enforced by session/task ownership. No extra reasoning pass or fallback planner. |
| 22 | Typed edits, baseline touched-entry conflicts, partial failures, sequential same-entry edits, scope display IDs, rollback recorded paths, local/global history, atomic symlink-preserving persistence. Removed Buffalo's extra file locks. Python CRUD uses Prime's dictionary coercion/omission and normalization/validation, distinct from planner validation. Host formatter functions are copied directly into JavaScript, including locale ordering, JSON numbers/property order and history slice behavior. |

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
performed.

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

That earlier transport gap is now closed for refinement's supported transports:
the native bridge retains the shared credential resolver but no longer sends
one-shot refinement through the SDK response parser or 401-recovery loop. Raw
WebSocket/SSE events preserve the provider's error type, status and retry timing.
Underlying Rust/Python exception wording and the host credential-store location
are not a byte-for-byte TypeScript API contract. Other Prime provider APIs remain
structurally outside Buffalo's supported transports; none was added.

## Closure pass: queued input, formatter/resources, native transport and exact variants

- First-input digests are staged, refreshed from disk at request admission, and
  persisted atomically with the request. Failed admission/abort can re-arm them.
  Equal timestamps select the latest digest. A compaction head supersedes the
  staged digest, so a request never receives both copies.
- Slash `/refine` commands use the durable ordinary input queue: they cannot
  overtake earlier user work, disappear on restart before admission, or become
  user evidence. Command receipts/results have Prime's context filtering and
  context-only persistence-failure behavior.
- Refinement runs before handing queued next-turn input to the root. Interactive
  deferred interval scheduling consumes its pending flag before dispatch; an
  under-25-turn check cannot repeatedly reschedule itself.
- Disposal waits for current ownership, re-reads settings, and handles pending
  explicit, automatic, compact and interval work in Prime's order. The regression
  matrix covers late review/plan success/failure, authoritative cancellation,
  supersession, branch navigation, exact-plan claims and already-completed roots.
- User refine skills override the built-in by name; disabled/model-hidden and
  Markdown-only resources do not advertise a pre-imported Python API. Child
  sessions do not advertise the root-only refine API. Kernel reload removes
  only its owned built-in module before loading a user override.
- `harness_format.mjs` directly copies Prime's five formatting functions
  (TypeScript annotations removed and product name translated). A source
  comparison confirms the copy. Node.js **22.8+** is now an explicit prerequisite,
  matching the supplied Prime minimum; validation used Node 24.11.0. No Python
  collation/numeric approximation or fallback formatter remains.
- Native one-shot calls have no session cache or reasoning option. They use
  Prime's headers and WebSocket-first/pre-start-only SSE fallback; API/protocol
  errors never trigger that fallback. Shared retries receive original provider
  codes/status, Retry-After/reset delays, terminal stop/length status, and
  malformed JSON/UTF-8/EOF behavior. Missing credentials fail before retries.
  Primary inference's transport is unchanged.

### Source → implementation → regression for the closure

| Prime source/contract | Buffalo implementation | Direct regression |
| --- | --- | --- |
| `agent-session.ts` staged first input, cold digest, input queue | `context.py` staging/commit methods; `runtime.py` admission; `refinement.py:start_queued_refine_command` | `test_prime_refinement_edges.py` first-digest/FIFO tests; `test_prime_refinement_variants.py` abort/re-arm, command ownership tests |
| `agent-session.ts` serialized/interactive barriers and disposal | `refinement.py` checkpoint, background ownership, drain and deferred scheduling | `test_prime_refinement_races.py`, `test_prime_refinement_variants.py`, production 25-turn test in `test_prime_refinement_lifecycle.py` |
| `refinement/refinement.ts:407–565` formatters and JavaScript serialization | `harness_format.mjs`; `harness.py` thin formatter/JSON wrappers | `test_prime_refinement_edges.py` locale/numeric/property-order/zero-history tests; direct source comparison |
| `prime-agent-runtime/src/rlm/harness.py` dictionary coercion | `harness.py:HarnessStore.mutate` | `test_prime_refinement_edges.py` dictionary/invalid-edit/scope/atomic-write cases |
| `resource-loader.ts`, `package-manager.ts`, `system-prompt.ts`, `agent-session.ts:_modelVisibleSkills` | `skills.py:refine_skill`, `context.py`, `kernel_api.py` | `test_prime_refinement_variants.py` root/resource/module-reload cases; `test_refinement_prompt.py` |
| `openai-codex-responses.ts` WebSocket/SSE normalization and errors | `native/inference.rs`, `refinement_transport.py`, `subscription.py` | Rust loopback transport tests; `test_prime_refinement_edges.py` raw events; `test_prime_refinement_provider.py` caps/retry tests |

### Files changed in this closure

Production: `context.py`, `harness.py`, `kernel_api.py`, `models.py`,
`native/inference.rs`, `native_client.py`, `refinement.py`,
`refinement_context.py`, `runtime.py`, `skills.py`, `subscription.py`;
new `harness_format.mjs` and `refinement_transport.py`.

Tests: new `test_prime_refinement_edges.py` and
`test_prime_refinement_variants.py`; strengthened
`test_prime_refinement_lifecycle.py`, `test_prime_refinement_races.py` and the
existing hardening compaction regression. Documentation: this report, the
complete map, README, subscription transport notes and the stale root-prompt
snapshot's refinement section.

Deleted/replaced in this closure: `queue_refine_command` and its separate
command chain; eager first-digest persistence; Python `compact_text` and
formatter/overview/history/notice approximations; the temporary `js_entries`
locale adapter; the SDK-specific incomplete-response string shim; refinement's
inner native SDK retry/401-recovery path; extra speculative disposal checkpoint;
stale conditional/unconditional cooldown behavior; repeated under-threshold
deferred scheduling. No old reinforcement architecture or compatibility path was
retained.

Remaining dedicated production reinforcement files: `refinement.py`,
`harness.py`, `harness_format.mjs`, `refinement_context.py`,
`refinement_model.py`, `refinement_retry.py`, `refinement_transport.py`,
`prime_refinement_models.json`, `builtin_skills/refine/SKILL.md`.
The shared runtime, context, kernel, providers and control-plane modules contain
only their integration points, not a second reinforcement loop.

### Closure verification

- Focused five-file parity set: **269 passed** before the two full-suite
  integration regressions were found.
- Native bridge built successfully against pinned Codex source; **2 Rust tests
  passed**, including seven local WebSocket framing/fallback variants. No model
  or authentication request was made by these tests.
- First full-suite attempt: **689 passed, 6 skipped, 1 failed** before stopping
  an interactive scheduling stall. This exposed duplicate staged/compaction
  digest delivery and the under-threshold deferred-scheduling loop. Both were
  fixed; the affected regressions plus edge/variant set then **146 passed**.
- Final full Buffalo suite: **770 passed, 10 skipped, 2 warnings in 169.16 seconds**
  (`uv run pytest -q --disable-warnings --timeout=45`). The per-test timeout only
  bounds the offline test runner; it changes no runtime or refinement policy.
- Scoped Ruff and whitespace checks pass. All **236** mapped test references
  resolve; copied formatter source comparison passes.

### Remaining differences / applicability boundary

No known behavioral deviations remain in the self-reinforcement loop on the
supported Buffalo surfaces. The 82 exclusions are not claimed as implemented:
they concern absent ACP/TUI/print adapters, generic non-refinement behavior,
Prime-only extension/custom-input/skip-abort APIs, synchronous disposal (Buffalo
has asynchronous shutdown), non-persisted/standalone environment-bound harness
constructors, and other structurally unavailable operations. Each source test's
specific reason is in the map. This is not a claim that Buffalo implements every
Prime application/API/provider.

Source provenance (supplied checkout has no Git metadata):

- `agent-session.ts` SHA-256:
  `c8c3b5b2793472ef5d7d1f949ed65786dc8b44529e0bc8566ea5505112d835f2`
- `refinement/refinement.ts` SHA-256:
  `81438d5b4acb013c40eae05f12402eb8d7052df3a522baec972c773fd5431f86`
- `prompts/rlm.ts` SHA-256:
  `8df7b497524d33aa4e7aae30bf12604029084cfd8747c00bd6df31c9076b32ef`

SELF-REINFORCEMENT PARITY: NO KNOWN BEHAVIORAL DEVIATIONS
