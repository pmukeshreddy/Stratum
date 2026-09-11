# EmulatorBench evaluation

The evaluator uses Buffalo's production Runtime and the official EmulatorBench verification implementation. The default signed-controller configuration remains **not ready** at the pinned public release: authenticated grading material is unavailable, and three selected platforms are explicitly ineligible for trusted scoring upstream. The separately selected public-source configuration uses the official runnable source-verifier path, with the same four tasks and Buffalo trajectory controls. Each mode requires its own preflight before inference.

## Source and selection

- Official public source: [PrimeIntellect-ai/prime-envs PR 678](https://github.com/PrimeIntellect-ai/prime-envs/pull/678), commit `9567aed36e3b45db877d5275ae378c1674dbe54d`, package `environments/code/emulatorbench`, distribution version `0.1.0`.
- Verifiers v1 bridge: [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers), commit `9f1417dcc1ea937adf67b81dc3766ffa9b43bacf` (the official dependency identified in Verifiers PR 1985).
- Prime autonomous reference: [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent), commit `81cd5390dbc871afb87be0d2012d205dd633bae3`. The supplied `../prime-agent-main` is an archive without Git metadata; `autonomous.ts`, `agent-session.ts`, and `settings-manager.ts` match that commit byte for byte.
- Buffalo baseline inspected: `f6805efad261c3e84b3db3530a7bad4f20f0543d`. Every experiment records its actual SHA, diff, untracked source hashes/archive, installed dependencies, and the exact source archive installed in the candidate.

`EmulatorBenchTaskset.load()` and `taskset.py::ALL_PLATFORMS` define the order. The lock in `src/threadweave/evals/emulatorbench_sources.json` contains all 16 IDs, package-file hashes, and the selection, fixed before any model was run:

1. **emulatorbench-chip8**
2. **emulatorbench-i8080-space-invaders**
3. **emulatorbench-gameboy-dmg**
4. **emulatorbench-nes**
5. emulatorbench-sms
6. emulatorbench-gameboy-cgb
7. emulatorbench-gba
8. emulatorbench-genesis
9. emulatorbench-snes
10. emulatorbench-ps1
11. emulatorbench-c64
12. emulatorbench-n64
13. emulatorbench-psp
14. emulatorbench-ps2
15. emulatorbench-zx-spectrum
16. emulatorbench-game-gear

Selection rule: `first_four_canonical`, one trajectory per selected task, task concurrency 1. No task difficulty was used to select the subset.

The official `PROVENANCE_RECONCILIATION.md` identifies historical July 7 paper experiments with private PR20 tree `091a506`, source aggregate `7f2d5ee3e6f51cfed7a20a4847e75c13fd1ee7522e00e85e2f5323d76ea2dc0f`. The public replacement intentionally changes APIs, source pins, trust rules, and denominator composition. Reports explicitly set `leaderboard_comparable: false`; this is a separately specified Buffalo experiment, not an exact reproduction of those historical matrices.

## Official task and score boundary

Inputs are the official per-platform `manifest.json`, `public_tests.json`, generated Rust starter, and public runner-v2 contract. `EmulatorBenchTask.setup()` stages the workspace with `anti_cheat_validation=True`, `public_feedback=False`. That official mode installs no grader, public-source manifest, oracle, reference implementation, private corpus, or signing material in the candidate. The initial sanitized workspace archive and its hash are retained on the host.

The candidate uses the official digest-pinned image:

```text
rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4
```

Verification calls the official chain, without copying or approximating its scoring logic:

```text
runtime.export_submission_archive(candidate)
EmulatorBenchTask._validate_controller_submission(..., expose_feedback=False)
  -> _run_fresh_grader(...)
  -> runtime.score_runner_v2_workspace(... official controller capability ...)
  -> runner_v2_scorer.score_runner_v2_corpora(...)
  -> signed controller record, verified by the official controller
```

The official controller creates a distinct Prime/Modal grader, enforces no-egress execution, separates candidate UID from trusted supervisor, authenticates corpus/attempt/result bindings, captures restricted artifacts, and confirms teardown. Docker and subprocess graders are explicitly rejected upstream. The adapter preserves those checks. Corpora and expected observations remain host-owned; only input windows enter the grader's candidate process. The original Buffalo candidate never receives them. The host receives only the candidate snapshot, not an arbitrary command or host path to execute.

`EmulatorBenchTask.solved()` returns the same numeric trusted score; the controller record also carries the official `passed` value. The adapter records the exact final graded score, including partial credit, rather than the best attempt. The aggregate is the arithmetic mean across the four fixed tasks, and is null if any task is ungraded. The official runner-v1 verifier remains available for public development testing but always returns untrusted reward zero. The adapter never promotes that to a benchmark failure or substitutes it for trusted runner-v2 scoring.

Only `controller_protocol.project_controller_feedback(..., "diagnostic_summary")` output crosses back to Buffalo. That official projection permits coarse tenth-bucket aggregates, removes hidden counts/identities/expected observations, and is bounded again with Prime's 6000 UTF-16-character rule. Exact scores, raw verifier output, signatures, corpus identity, and grader artifacts stay in the host audit. A failure enters `Runtime.interact(root_id, feedback)` as an ordinary user message.

The task instruction is the official generated instruction plus neutral descriptions of host verification and the public offline Cargo contract. No instruction asks for delegation, refinement, memory creation, or learning from failure.

## Runtime and autonomous semantics

`autonomous_worker.py` extends the existing `evocode_worker.PersistentWorker`, reusing its real Runtime, root creation, provider proxy, persistent Python kernel, native child registry, refinement checkpoints, usage collection, and audit. It creates one `AutonomousCycle` for the entire reconstruction. It cannot restart the trajectory or reset budgets after a verifier failure. Each new task gets a new container, Runtime state directory, root, kernel, local/global harness, and registry.

No Buffalo production-core file is changed. `src/threadweave/autonomous.py` already ports the relevant Prime controller and is reused unchanged. `serialized_refine=True` preserves the existing autonomous evaluation checkpoint behavior while the root uses interactive mode to accept host feedback. Native 25-turn/compaction refinement, 1200-second cooldown, reviewer declines, redundant-edit rejection, and empty plans remain unchanged.

Prime source functions are `createAutonomousRuntimeState`, `nextAutonomousContinuation`, `autonomousLimitReason`, `refreshAutonomousQualityGates`, `buildAutonomousGateFailureContinuation`, and `addAutonomousUsage` in `packages/coding-agent/src/core/autonomous.ts`, and `AgentSession._getContinuationMessages`/`_snapshotAutonomousRuntimeState` plus message-end accounting and compaction handling in `agent-session.ts`.

| Limit/accounting rule | Pinned source behavior |
|---|---|
| Continuations | 3 |
| Root assistant turns | 12; child/reviewer/planner calls do not consume this counter |
| Root token budget | 80,000; input + output + cache write in Prime; cache reads excluded |
| Buffalo equivalent | `max(0, input_tokens - cached_input_tokens) + output_tokens`; cache writes already included in input |
| Autonomous wall limit | 1800 seconds |
| Gate timeout | 300 seconds, also enforced at the host |
| Gate retries | 3; the fourth failed check stops with `retry_exhausted` |
| Unchanged suppression | A suppressed failed check still consumes a retry/continuation according to Prime |
| Feedback | 6000 UTF-16 characters plus truncation marker |
| Compaction | Same cycle and limits retained; no reset of root/kernel/refinement state |

Prime checks limits after a failed completion gate, not before every tool/model step; a passing gate can finish even after a limit was reached. More than 12 responses within a single work segment therefore does not itself prove a bug. The monitor identifies this boundary policy and flags feedback emitted after exhausted limits. Error/aborted assistant responses are excluded as in Prime. Buffalo's broader native tree-resource ceilings and the official task setup/harness deadlines remain separately recorded in the config/manifest; they are not relabeled Prime limits.

Fingerprinting reuses `git_snapshot()`: Git status, binary HEAD diff, and hashes of untracked files. Exact Prime exclusions are `verification`, `target`, `.vf-prime-agent`, `Cargo.lock`, `submission.tar.gz`, and `runner_args.log`. No HEAD or an incomplete snapshot disables suppression. File access times and external evaluator logs do not produce changes. The Cargo.lock exclusion is an intentional parity detail. Root/kernel/process/workspace device/inode are asserted at every boundary; the verifier must also leave harness contents/history, child registry, compaction history, and refinement counters intact.

## Data fetched and remaining access

The exact executed commands, absolute cache paths, upstream URLs, resolved Git commits, HTTP SHA-256 pins, source licenses, and official download receipts are in [emulatorbench-data-provenance.json](emulatorbench-data-provenance.json). All public sources for the selected four were fetched with the unchanged official `scripts/prefetch_public_sources.py`, including offline cache verification. The cache is `.emulatorbench/public-sources`; checkout is `.emulatorbench/sources/prime-envs`. None of that cache is mounted into Buffalo.

The public `PRIME_EVAL.md` requires an externally supplied `EMULATORBENCH_RUNNER_V2_CORPUS_ROOT`; it provides no corpus-download URL or authentication/downloader command. Repository releases, the intended Hub identity, the official task repository, and public release metadata supplied no authorized signed bundle. The [maintainer's status](https://github.com/PrimeIntellect-ai/prime-envs/pull/678#issuecomment-5076627049) states that CHIP-8 alone is installed and independently validated, and provisional systems cannot enter trusted scoring.

The release's launcher and that document disagree. The actual `scripts/run_prime_eval.sh`, `scripts/preflight_prime_eval.py`, and `configs/prime-agent.template.toml` select public source-verifier mode (`anti_cheat_validation=false`, full public feedback), and require neither signed corpora nor a controller signer. This is a runnable development workflow. The PR's all-16 readiness update explicitly calls its completed Luna runs smoke validation, not benchmark results. Following that launcher does not bypass the grading limitation: `runtime.score_workspace()` applies `fail_closed_runner_v1_score()` unconditionally, preserving the original numeric result as `untrusted_development_score` while setting official reward to zero and `passed` to false. `EmulatorBenchTask.solved()` independently returns zero without a verified oracle. The Buffalo trusted adapter therefore cannot silently use this workflow as a scored four-task experiment. Earlier claims that no official runnable workflow exists were too broad; the missing prerequisite applies to trusted grading.

| Selected platform | Exact missing prerequisite |
|---|---|
| CHIP-8 | Author-signed `chip8:chip8_timendus_suite` bundle matching installed commitments; the public package contains its verification key and commitments, not the bundle |
| i8080 | Trusted independent oracles for `i8080_8080ex1`, `i8080_8080exer`, `i8080_8080pre`, `i8080_cputest`, `i8080_test_tst8080`; no installed declarations |
| Game Boy DMG | `gameboy_sm83_json_cpu` is explicitly provisional; `dmg_blargg`, `dmg_mooneye`, `dmg_samesuite`, `gameboy_tasvideos_accuracy_matrix`, `gameboy_gambatte_suite` need independent oracles; `dmg_acid2`/`dmg_mealybug` need fixed indices/capture schedules |
| NES | `nes_christopherpow_archive` requires independent oracle; `nes_full_palette` requires authorized private input and oracle; neither is installed |

The CHIP-8 signed-manifest commitment is `d3953932aa8019e4606d43ada05e9d7d97b2aaf7679136d934740764bacd9b6f`; input root `db33c8f30306bd72989e7bb7be546ca02de2e6acdb31459499ef8ecb498648b4`; oracle root `348fa1a763daedc2110d9c14574cf2eb758da9185f14799d8203f12dd1f23ea3`; license material `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`.

Required access is a matching benchmark-authorized bundle and, for the other three, a maintainer-approved release that actually installs trustworthy declarations. A GitHub token alone cannot repair provisional or missing oracles. The selected runtime is Modal; the host uses the locally activated `pochamreddymukesh` profile, whose token was verified against Modal's API. `PRIME_API_KEY` is not required for this configuration. Corpus directories must satisfy the official loader's ownership/permissions checks, including secure ancestors; use an owner-controlled location outside the group-writable checkout. Do not loosen the loader or self-sign a replacement corpus. The locally generated controller publication key is a separate role and cannot authorize benchmark corpora.

## Reproduce setup without touching another evaluation

These commands use an independent interpreter, checkout, cache, and result tree. Existing completed setup directories need not be recreated.

```sh
mkdir -p .emulatorbench/sources
git clone --filter=blob:none --no-checkout https://github.com/PrimeIntellect-ai/prime-envs .emulatorbench/sources/prime-envs
git -C .emulatorbench/sources/prime-envs fetch origin refs/pull/678/head
git -C .emulatorbench/sources/prime-envs sparse-checkout set environments/code/emulatorbench tests
git -C .emulatorbench/sources/prime-envs checkout --detach 9567aed36e3b45db877d5275ae378c1674dbe54d
git clone https://github.com/PrimeIntellect-ai/verifiers .emulatorbench/sources/verifiers
git -C .emulatorbench/sources/verifiers checkout --detach 9f1417dcc1ea937adf67b81dc3766ffa9b43bacf
uv venv --python 3.12.12 .emulatorbench/venv
uv pip install --python .emulatorbench/venv/bin/python --constraint configs/emulatorbench-host-constraints.txt -e '.[dev]' '.emulatorbench/sources/verifiers[modal]' .emulatorbench/sources/prime-envs/environments/code/emulatorbench
# Only needed on a host without the profile; prompts for credentials and verifies them.
.emulatorbench/venv/bin/modal token set --profile pochamreddymukesh
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench_setup
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench discover
MODAL_PROFILE=pochamreddymukesh .emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench preflight
```

The setup module runs the official downloader once for each selected slug and creates a host-only publication signer (directory 0700, private key 0600). It downloads no unofficial corpus. Preflight exits 2 when prerequisites are missing, before sandbox allocation or inference. Provider credentials are not copied to candidates. Modal SDK 1.5.5 and its compatible protobuf 6.33.6 are pinned in the independent host environment. Credentials remain in the host's Modal profile store, outside the project. Runtime setup installs Buffalo from its project lock with Python 3.12.12, uv 0.11.8, and Node 22.16.0 (official archive SHA-256 checked). It installs no benchmark modules in the candidate.

## Validation and launch

The authorized public-source experiment uses `configs/emulatorbench-public.json`. Its model, reasoning, selection, runtime, RLM/refinement settings, and autonomous limits are identical to `configs/emulatorbench.json`. It additionally enables public rehearsal. Preflight invokes the pinned upstream source-readiness function for all four tasks. The candidate gets the official starter and public runner contract; input corpora and grading code remain in separate fresh no-egress graders. The original source-verifier score and pass flag drive progress/continuation, while the official wrapper's reward is retained separately without alteration. Reports identify the mode and preserve both `official_score` and `public_source_score`, with separate aggregate fields. This mode does not create or authorize runner-v2 corpora.

Before completion, the root can run `./verify-public.sh` through Buffalo's normal shell/REPL. The command uses a narrowly scoped resident socket and the existing host RPC transport to snapshot the candidate and invoke **the same** `PublicSourceVerifier.verify()` and official fresh-grader path as final grading. It does not implement a second scorer or expose a host filesystem. The source-authorized public manifest/source descriptors and exact upstream `invoke_case`/`replay_determinism` source explain the actual input selection, CLI flags, simultaneous cycle/frame budgets, output parsing, exit behavior, and replay semantics. Oracle functions, reference assets, corpus contents and grader artifacts remain outside the candidate. Bounded feedback includes public input basenames, observed cycle/frame/serial values, runner exit status and compiler/runner errors, without expected states or golden outputs.

Rehearsal returns an ordinary tool result: exit 0 for a passing snapshot, 1 for failure, 2 for unavailable/exhausted service. The public iteration cap is 128, taken from the pinned official `configs/prime-agent.template.toml`; each host invocation retains the 300-second verifier timeout. Repeated failed fingerprints are suppressed. Native children may work concurrently, so a changed-during-rehearsal flag prevents confusing a prior snapshot with the current workspace. Final grading always takes a new snapshot and retains its independent Prime gate/continuation accounting. The root is told to rehearse before completion; no RLM/refinement behavior is requested or altered. Rehearsal attempts and suppression records are saved separately from final verifier attempts and never become final scores.

Prime's defaults and the official benchmark launcher are distinct configurations. This comparison retains Prime source defaults (12 root responses, 80,000 noncached root tokens, 3 continuations/retries, 1,800 seconds, checked at completion boundaries). The official benchmark template overrides those caps with its unlimited sentinel and a six-hour wall budget; those much larger budgets are **not** silently adopted for this comparison. A root may exceed the defaults during uninterrupted tool work, then fail its first final verifier and receive no continuation. Public rehearsal remains usable during that work, just as manually running the public gate does in Prime. Differential tests execute the pinned Prime controller directly, including 11/12/13 response boundaries, usage, unchanged retries, successful gates beyond limits and slow-gate timeout behavior. A generic correction in `src/threadweave/autonomous.py` captures wall eligibility before the gate, matching Prime's `now` parameter; RLM/refinement/core solving behavior is unchanged.

```sh
MODAL_PROFILE=pochamreddymukesh .emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench preflight --config configs/emulatorbench-public.json --output .emulatorbench/preflight-public.json
MODAL_PROFILE=pochamreddymukesh .emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench run --config configs/emulatorbench-public.json --output .emulatorbench/runs/gpt-6-astra-first-four-public
```

Build the isolated test image once. The builder uses the same installer as the production remote worker. Each test starts unique, unmounted, no-network containers and removes only those exact test containers. It never prunes Docker or touches another evaluation's containers/results/state.

```sh
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench_setup --build-test-image buffalo-emulatorbench-validation:local
BUFFALO_EMULATORBENCH_TEST_IMAGE=buffalo-emulatorbench-validation:local .emulatorbench/venv/bin/pytest -q tests/test_emulatorbench.py tests/test_emulatorbench_failures.py tests/test_emulatorbench_isolation.py
```

The suite covers the real resident Runtime and mailbox transport, FAIL→ordinary USER feedback→FIX→PASS, unchanged suppression, exact limits, cancellation/provider/verification failures, persisted kernel variable/harness/registry, fresh tasks, native RLM/refinement and compaction/cooldown, hidden-file shell/Python walks, source discovery, score/null handling, and complete fixture-task serialization. Fixture scores are explicitly test-only. A separate test invokes the actual released verifier on the official Rust starter and downloaded CHIP-8 sources and exercises the actual signed-controller bridge's refusal of an unsupported/untrusted grader. The upstream signed-corpus/scorer tests use the upstream synthetic fixtures; they do not establish availability of the benchmark's real corpus or a successful live trusted run.

After source/data preflight and focused validation genuinely pass, the authorized experiment command is:

```sh
MODAL_PROFILE=pochamreddymukesh .emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench run --config configs/emulatorbench.json --output .emulatorbench/runs/gpt-6-astra-first-four
```

The config specifies the official Modal sandbox runtime, `codex_subscription`, `gpt-6-astra`, `xhigh` reasoning, one task at a time, and unchanged native RLM/refinement. Provider requests have a 900-second timeout, separate from Prime's autonomous and verifier limits. An initial run with the generic coding profile's 180-second request timeout was stopped after repeated implementation-call timeouts; its raw evidence and interruption record are retained. Interrupted calls may lack provider token usage and must not be treated as zero-cost calls. The run output must be a new directory. A failed preflight writes a null aggregate with explicit blockers and no model calls. Do not override the gate to spend tokens on ungradable tasks. Switching the sandbox provider does not change corpus eligibility or the official controller's trust requirements.

## Audit and reporting

Each task writes `raw-events.jsonl`, `model-calls.jsonl`, `verifier-attempts.jsonl`, `rlm-events.jsonl`, `refinement-events.jsonl`, `manifest.json`, `trajectory.json`, `task-result.json`, and `task-audit.json`, plus original signed-controller artifacts and initial workspace snapshot. Start records precede external calls, completed journal writes are flushed/fsynced, conflicting event IDs are rejected, and derived reports remain separate from raw evidence. Failures use `PROVIDER_FAILURE`, `ADAPTER_FAILURE`, `INFRASTRUCTURE_FAILURE`, or `TIMEOUT` with null official scores. A prior legitimate grade is retained separately as `last_graded_score`.

```sh
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench monitor --output .emulatorbench/runs/gpt-6-astra-first-four
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench report --output .emulatorbench/runs/gpt-6-astra-first-four
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench audit --output .emulatorbench/runs/gpt-6-astra-first-four/emulatorbench-chip8
```

The monitor reports root/kernel identity, wall time, root/child model calls, turns and separate cached/noncached/output/reasoning tokens, admissions/nesting/active/completed children, refinement triggers/declines/empty/applied plans, attempts/suppression/latest score, and invariant failures. It does not infer useful RLM results from completion counts.

Automatic audit claims stop at `RLM_CALLED`, `CHILD_COMPLETED`, and `ROOT_RECEIVED_RESULT`. Per-child records include assignment, parent/depth/admission turn, requests/messages, and file-change observations tied to execution event IDs. File observations record overlapping writers; they do not prove causal attribution. Refinement records retain reviews, plan assessments/rejections, exact persisted versions and later context visibility. Retrieval candidates are listed separately. Use and benefit default to **unknown**, not false or zero.

For the requested close audit, inspect raw root responses, child messages, actual actions, learned-entry versions, and subsequent official outcomes. Record generic Buffalo bugs separately with the triggering request/event, reproduction, and failure category. No production-core bug is assumed from a low benchmark score.

Optional host-side `audit-review.json` contains a list of reviewer annotations. Allowed claims are `ROOT_USED_RESULT`, `ROOT_USED_RESULT_WITH_BENEFIT`, `REFINEMENT_USED`, and `REFINEMENT_USED_WITH_OUTCOME_BENEFIT`. Each requires `reviewer`, a behavioral `evidence` explanation, `source_event`, `root_request`, `root_action_event`, and a literal `response_quote`. RLM claims also require `root_received_event`; refinement claims require `entry_id`, `entry_version`, and `visibility_event`. Benefit claims require `before_attempt` and `after_attempt`. The audit command validates event ancestry, temporal order, receipt/version evidence, literal quote, and bracketing official score improvement. It then marks those claims as reviewer-assessed observational evidence. Merely delivered messages, visible memory, matching filenames, or rising scores do not automatically create use/benefit claims; experimental causal attribution would require an additional controlled comparison.

## Recorded Modal compatibility fix

The pinned Verifiers Modal adapter always requested an encrypted port, including when `block_network=True`; Modal rejects that combination. `docs/patches/verifiers-modal-no-egress.patch` omits exposed ports only for a network-blocked sandbox. The setup command applies this exact hash-pinned delta to the independent host interpreter using atomic file replacement. Baseline and patched hashes are recorded in `emulatorbench_sources.json`. The official scorer/verifier remains unchanged. Trace IDs use 24 random hexadecimal characters so composed grader names satisfy Modal's length limit. Evaluator shell configuration explicitly preserves the image's `CARGO_HOME` and `RUSTUP_HOME`. None of these changes modifies Buffalo production core.

For reviewed public-source outcome-benefit claims, specify `score_field: public_source_score`; the audit checks this field against both original verifier attempts. Without this explicit field, reviews require improvement in the official reward.
