# Refinement learning quality

Buffalo retains Prime's refinement lifecycle and typed continual harness. Its
reviewer and planner now use a stricter, generic admission criterion: compare a
candidate with the pre-edit trajectory and identify a concrete future decision
that would differ because the entry exists. Progress summaries and tactics
already known, executed, or planned normally do not qualify. A newly generalized
preventive procedure can qualify; recording the earlier fix alone cannot.

Each model-proposed edit includes `metadata.learningAssessment`: prior evidence,
redundancy comparison, future trigger, behavior change, counterfactual, novelty,
expected benefit, and explicit redundant/progress-only/same-action judgments.
Planning excludes candidates with affirmative rejection judgments or missing
assessment evidence. It can return an empty plan even after reviewer approval or
an explicit request. The new `refinement_plan` event retains assessments, rejected
candidates, and the exact accepted plan for audit. Model judgment supplies the
semantic comparison; field validation is not proof of novelty or causal benefit.

This is an intentional learning-policy deviation from the supplied Prime source.
No scheduling, cadence, scope, persistence, RLM, or primary model guidance changes
accompany it. Explicit refinement still bypasses the automatic reviewer. Automatic
reviews still use 25 turns, compaction, and the existing cooldown. The same safe
boundary applies the completed plan once. Trusted extension proposals and explicit
rollback operations retain their existing contracts; direct harness CRUD is not
reclassified as learned refinement.

Full content is already available through the intended selective retrieval path:

`harness_state.json` → `HarnessStore.get` → `harness.get(kind, id).content` in the
persistent kernel → tool observation → the next root model request.

The ordinary digest and refinement notice deliberately contain previews. They
allow discovery without copying every full entry into each request. Retrieval,
formatting, and foundational system content remain unchanged. A focused test uses
a long entry whose final instruction is absent from the preview, retrieves it
through the real kernel/host bridge, and checks the next request contains it.

The seven-task rerun remains **EvoCode-Bench + Prime-style autonomous feedback**,
not an official leaderboard score. Assessment metadata records predicted value;
the separate trajectory audit must establish actual later use and any outcome
benefit. Visibility, repetition of pre-existing actions, and eventual round success
do not establish use or benefit.

Validation on 2026-09-10: five focused checks passed, followed by one existing
regression-suite run (773 passed, 9 skipped). Ruff and diff whitespace checks
passed. These checks establish admission, persistence, retrieval, and lifecycle
contracts; model learning quality remains an empirical evaluation question.
