# Continual harness and refinement

The continual harness stores `prompt`, `memory`, `skill` and `subagent` entries in
`harness_state.json`. Global state lives in `DATA/harness/`; session-local state
lives in `DATA/sessions/SESSION_ID/harness/`. Each scope retains refinement history
in `refinements.jsonl`. `state_changes.jsonl` journals changes before the readable
snapshot advances. Local guidance can override global guidance within a session;
colliding IDs remain visible with scope labels.

## Scheduling and delivery

`await refine.run(instructions=None, global_=False)` schedules explicit refinement
and returns immediately. `await refine.status()` reports pending and in-flight
work. Explicit requests bypass the reviewer. Automatic review runs in root
sessions at 25 assistant turns and compaction, with a 20-minute cooldown; the
reviewer decides whether planning is warranted.

Planning can overlap tools after a model response. The exact completed plan is
applied once at a safe turn boundary, after rechecking branch ownership and
touched-entry conflicts. Serialized mode holds the next primary request until
that boundary. Stale or cancelled plans cannot write; individual invalid edits
are reported separately. Independent manual requests serialize, while root skill
requests within a turn coalesce.

Applying edits preserves the SYSTEM prefix. An applied-only notice exposes changes
at the next legitimate model request. Cold session/resume contexts and compaction
heads carry the merged harness digest as user context. Display-only outcomes do
not enter model context, and applying edits does not restart completed or idle work.

## Learning criteria

Model-proposed edits must identify a concrete future decision that changes because
the entry exists. Progress summaries and tactics already known, executed or planned
normally do not qualify. A newly generalized preventive procedure can qualify.
This admission policy is stricter than the supplied Prime implementation.

Each edit carries `metadata.learningAssessment`: prior evidence, redundancy
comparison, future trigger, behavior change, counterfactual, novelty, expected
benefit and rejection judgments. Missing evidence or affirmative redundant,
progress-only or same-action judgments reject the candidate. An empty plan is
valid, including after reviewer approval. The `refinement_plan` event retains
accepted and rejected assessments. These judgments predict value; later trajectory
evidence is needed to establish actual use or outcome benefit.

Direct harness CRUD, trusted extension proposals and explicit rollback retain
their own contracts. Generic CRUD is not subject to model-plan admission rules.

## Retrieving and inspecting state

The digest and change notices contain bounded previews. Retrieve full content
selectively through `harness.get(kind, id).content` in the persistent kernel, then
inspect the relevant content to expose it to the model. Skills refer to existing
Python callables and their contracts; subagent specifications use native `rlm`.

Implementation lives in `harness.py`, `refinement.py`, `refinement_context.py`,
`refinement_model.py`, `refinement_retry.py` and `refinement_transport.py`.
The bundled `builtin_skills/refine/SKILL.md` is the model-facing API guide.
The `test_prime_refinement_*`, `test_refinement_*` and
`test_continual_harness_*` suites cover scheduling, persistence, conflict handling,
notice delivery, retrieval and provider behavior.
