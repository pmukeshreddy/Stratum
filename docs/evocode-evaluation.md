# Buffalo on EvoCode-Bench

This is **EvoCode-Bench + Prime-style autonomous feedback**, an adapted interaction
protocol. Its scores are not official EvoCode leaderboard scores. The dataset's
instructions, reference solutions, cumulative tests and reward files remain unchanged.

## Source and lifecycle mapping

Inspected supplied `prime-agent-main`: `core/autonomous.ts`, `core/agent-session.ts`,
`core/settings-manager.ts`, `core/prompts/rlm.ts`, `core/refinement/refinement.ts`,
the refine skill and Python wrapper, and `prime-agent-runtime/src/rlm/`.
Source hashes are recorded alongside the integration's upstream pins.

Upstream Harbor 0.22.0, commit `191d1b989bbba1d77c2db23e17aec308d7c08046`, uses
`MultiStepTrial._run_step`, `_run_agent_phase`, `BaseAgent.setup/run/resume`, and
`Verifier.verify`. Setup runs once; resume support must be advertised explicitly.
Its shared verifier still mounts the host verifier logs into the agent container.
The [reported leak](https://github.com/harbor-framework/harbor/pull/1961) remains
open in this inspected source. Merely deleting tests before a new root turn would
also leave a race with Buffalo's resident children.

[EvoCode source](https://github.com/UniPat-AI/EvoCodeBench/tree/f8fcfaa1c9ad1c5b0bbc433323b587e4ddea2f32)
and the official [task-only release](https://huggingface.co/datasets/UnipatAI/EvoCodeBench)
define 26 tasks / 227 ordered steps. GitHub does not contain the executable dataset;
the Hugging Face archive does. No reference trajectories are needed.

| Boundary | Integration behavior |
| --- | --- |
| Trial setup | Start one Buffalo worker in Harbor's original task container. Create one interactive root with the verbatim first instruction staged, serialized refinement enabled. |
| Round 1 | Make the staged root runnable; do not send the first instruction twice. |
| Later rounds | `Runtime.interact` sends the verbatim instruction to the existing, still-active root. |
| Assistant completion | Wait for root, descendant work, background processes and owned refinement to settle. |
| Gate | Snapshot the candidate filesystem into a disposable verifier container; call Harbor's original step verifier there. |
| Failed gate | Send Prime's bounded ordinary user continuation through `Runtime.interact` on that same root. |
| Passed gate / limit | Settle the round. Keep runtime, kernel, counters, children and learned state alive. |
| Trial teardown | Export evidence, shut down the worker, and destroy task containers. No learned state is imported by another task. |

There is no benchmark-specific production refinement policy. The normal
`refinement_message_end` → serialized `refinement_checkpoint` → compaction / next
request ordering stays in production Runtime. The controller operates after that
boundary. It never calls `refine.run`, creates children, or lowers the 25-turn interval.
Explicit refinement schedules a receipt; the exact planned edits apply once. A
reviewer decline remains a decline, including when a verifier has failed.

RLM remains the native concurrent child-session API. The worker and its child
registry live for the entire trial. Children communicate through the existing
message queue and files. Completed child kernels may unload and recover under
Buffalo's normal semantics; an idle root kernel is retained across round boundaries.

The host retains model credentials and serves inference over a private Docker exec
stdio channel. No evaluator tools or paths are added to model context. The agent
container never mounts host verifier logs, tests, solutions, dataset paths, Docker's
socket, another trial's state, or the host home directory. A verifier snapshot never
copies files back. Only a binary result and strictly parsed numeric CASE_SUMMARY
counts cross the grader boundary. Raw test stdout, traceback/source text, case
names, expected values and reward files remain host-side.

## Budgets and deviations

Prime defaults: 3 continuations, 12 assistant responses, 80,000 non-cache-read
tokens, 1,800 seconds; gates have 3 retries and 300 seconds. Gates run sequentially,
stop at the first failure, and exhaust on failure **4** (`attempt > maxRetries`).
Limits are checked after a completion gate, as in Prime, rather than enforcing a
12-response hard cap while the model is still calling tools. Output bounds use
Prime's 6,000 UTF-16 character prefix plus its truncation marker.

Each round explicitly starts a fresh autonomous cycle. Refinement counters do not
reset. Resolved Harbor step timeouts override cycle/gate timeout defaults and are
written to the manifest with their provenance. Buffalo's independent task-wide
resource limits come from the supplied RunConfig and are recorded as well.
An authoritative Harbor timeout aborts the trial, not a fabricated failed test.

Unchanged protection hashes Prime's Git status, HEAD binary diff and untracked
contents, with its exact excluded pathspecs. No usable Git snapshot means no
suppression, matching Prime. Verification happens in a fresh filesystem snapshot;
verifier-installed dependencies or other verifier side effects do not flow back.
This isolation and same-round hidden-verifier feedback are explicit protocol changes.
Only Linux Docker tasks without sidecars or candidate volume mounts are supported;
unsupported isolation configurations fail before inference.

The adapter uses Buffalo's interactive resident-root mode with serialized refinement
explicitly enabled. Its outer `wall_seconds` cap therefore measures accumulated
execution time, including children; Harbor and the autonomous cycle impose elapsed
round deadlines. Grading waits for managed descendant/process work to settle, then
captures files, not live service processes. This is stricter quiescence than Prime's
assistant-only gate boundary. No live daemon state is transferred to the grader.

One existing RLM difference is retained: Buffalo's `delete_subagent` cancels and
unloads a child but keeps its cancelled session metadata, whereas Prime hides a
deleted child from its live registry. The report exposes that actual registry;
the adapter does not claim exact Prime deletion equivalence or alter production RLM.

## Evidence and scoring

The report separates official Harbor reward aggregation from instrumentation and
EvoCode's task score (all declared rounds in the denominator, including unreached
rounds). These differ on aborted tasks: the inspected Harbor mean excludes steps
without verifier results. No score from this adapted runner is leaderboard-comparable.

Per-round audits retain runtime/root/kernel identities, workspace device/inode,
event-prefix hashes, refinement counters/history, exact learned entries and the
retained child registry. Instrumentation counts actual child admissions and applied
typed edits. Model request/response logs provide the exact later invocation where
learned state became visible. Visibility is not behavioral use. Explicit references
in model output are recorded as candidate evidence for review; causal improvement
is never inferred automatically. A before / edit / later visibility / use / outcome
chain must be supported before claiming self-improvement.
Aborted rounds export partial evidence when the worker remains reachable. Official
step rewards are scored independently of that export; the report explicitly marks
missing instrumentation and also totals tokens observed by the host provider proxy.

Install from the source checkout with Python 3.12:

```sh
uv sync --python 3.12 --extra dev --extra evocode
```

The entry point is `buffalo-evocode` (or `python -m threadweave.evals.evocode`).
It validates hashes of the Harbor APIs above before starting a container. A single
task uses `--task /path/to/released/task`; a full dataset uses `--dataset
/path/to/evocodebench_wotraj`. Both require `--config configs/evocode.example.json`
and `--output /path/to/results`. `--max-steps 2` explicitly labels a smoke prefix.

The example RunConfig's task-wide outer caps are 1,000 turns / 8M tokens / 27,000
seconds (15 × the release's 1,800-second per-round cap). They are explicit experiment
settings, separate from unchanged Prime continuation defaults. Every resolved
step timeout and the full RunConfig are preserved in the manifest.

Focused integration tests: `pytest -q tests/test_evocode_integration.py`.
The opt-in Docker smoke uses the actual released
`theme_d8_w5_cloud_devops_integration_e2e_wiring` project, with a scripted transport
probe. Set `BUFFALO_EVOCODE_SMOKE_TASK` to that task path and
`BUFFALO_EVOCODE_SMOKE_OUTPUT` to a result directory, then run
`pytest -q tests/test_evocode_smoke.py`. It verifies the real hidden grader, retries,
unchanged task files, root/kernel persistence and grader absence in the agent
container. Its expected failures are not an evaluation of model capability.
RLM and explicit/25-turn refinement are exercised by the integration fixtures;
actual evaluation leaves both choices to the model.

Model credentials stay with the host's usual Buffalo provider. Package installation
inside each original task image adds isolated Python 3.12.12, uv 0.11.8 and Node
22.16.0 under `/opt/buffalo-evaluation`, outside the project. Task PATH entries are
retained. No project packages or reference solutions are installed by the adapter.

Full evaluation requires a separate user instruction.

## Validation performed

Implementation preceded testing. The focused integration run passed 15 tests;
the existing Buffalo regression suite ran once: 754 passed, 8 skipped. Final
failure-export/scoring/cleanup fixes passed three targeted checks. Ruff and
`git diff --check` passed. The official dataset validator accepted all 26 tasks
and 227 steps.

The released-task Docker smoke passed on 2026-09-10 (278.41 seconds). An initial
Python bootstrap failure was corrected before this successful run. Across two
rounds it recorded the same root `88ca9de12245499196c95a3c6f35c5dc`, kernel PID
127, workspace inode 33593438 and runtime identity. A REPL value of 937 survived;
the refinement counter progressed from 5 to 10. Each round ran four real
cumulative verifiers and delivered three bounded ordinary-user failure messages.
Harbor resolved the round deadline to 1,800 seconds and verifier timeout to 600
seconds; those values appear in the manifest. Task input hashes were unchanged.
No smoke-owned containers or snapshot images remained after cleanup.

The scripted smoke intentionally performed no project implementation and earned
zero round rewards. It establishes execution-path behavior, not RLM/refinement
uptake or self-improvement by a live model. The native integration fixtures cover
recursive child admission/messages/artifacts, applied local refinement, later
state visibility, 25-turn review and refinement barriers. No full evaluation has
been launched. Local logs and audits are under `results/evocode-validation/`.
