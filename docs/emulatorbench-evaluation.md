# EmulatorBench

The project owner's reported Buffalo score is **25**. This is a reported result,
not a value recomputed by the cleanup. Associate any published score with its run
report, selected tasks and score field. The public-source verifier score and the
official trusted reward are recorded separately.

## Sources and selected tasks

The pinned release is PrimeIntellect-ai/prime-envs commit
`9567aed36e3b45db877d5275ae378c1674dbe54d`, package
`environments/code/emulatorbench` version `0.1.0`. Verifiers and Prime reference
commits, package hashes and all 16 platform IDs are recorded in
`src/threadweave/evals/emulatorbench_sources.json`. Download commands, cache paths,
licenses and receipts are retained in [data provenance](emulatorbench-data-provenance.json).

The fixed `first_four_canonical` selection runs one trajectory per task, sequentially:
CHIP-8, i8080 Space Invaders, Game Boy DMG and NES. Reports set
`leaderboard_comparable: false`; the public release differs from the historical
paper protocol.

## Verification modes

`configs/emulatorbench-public.json` selects the released public-source verifier.
It retains `public_source_score` and `official_score`, plus separate aggregates.
The upstream wrapper can return official reward zero while preserving a nonzero
public development score. Never substitute one for the other.

`configs/emulatorbench.json` selects the signed-controller path. At the pinned
release it cannot grade the full selected set: CHIP-8 needs the authorized signed
bundle, while i8080, Game Boy and NES have missing or provisional trusted oracles.
Preflight reports these blockers before inference. A GitHub token, a different
sandbox provider or a self-signed corpus does not supply the missing grading data.

Both modes use official verification in a fresh, separate no-egress grader. The
candidate receives the starter, public runner contract and authorized feedback;
corpora, reference implementations and grader artifacts remain outside it.
The local Modal compatibility patch is retained in
[patches/verifiers-modal-no-egress.patch](patches/verifiers-modal-no-egress.patch).
Setup checks its pinned hashes and omits exposed ports for network-blocked graders.

## Setup

Use the independent host environment and source caches:

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
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench_setup
```

Authenticate Modal on the host and select its profile with `MODAL_PROFILE`.
The setup module uses the official public-source downloader and provisions the
separate controller publication signer; it does not create authorized benchmark
corpora. Credentials stay on the host. Buffalo candidates receive isolated
Python 3.12.12, uv 0.11.8 and Node 22.16.0 outside the task workspace.

```sh
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench preflight --config configs/emulatorbench-public.json --output .emulatorbench/preflight-public.json
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench run --config configs/emulatorbench-public.json --output .emulatorbench/runs/new-public-run
```

Use a fresh output directory. The config fixes `gpt-6-astra`, `xhigh`, Modal,
one task at a time and a 900-second provider timeout.

## Runtime and feedback

`autonomous_worker.py` extends the resident EvoCode worker and uses the production
Buffalo Runtime, Python kernel, native children and refinement. Each task starts
with a fresh container, root and harness state. One autonomous cycle spans that
task's attempts; failed verification returns an ordinary user message to the same
root. Native refinement scheduling, cooldown and model decisions remain in effect.

The Prime-derived cycle defaults are 3 continuations, 12 root assistant responses,
80,000 non-cache-read tokens and 1,800 seconds; verifier gates use 300 seconds and
3 retries. Limits are checked after a failed completion gate. Task-wide runtime
caps remain separate and are recorded in the manifest.

The public config also enables `./verify-public.sh` as a normal tool command. It
snapshots the candidate through the resident transport and calls the same official
verifier as final grading. Its public iteration cap is 128. Unchanged failed
snapshots are suppressed; final grading takes a fresh snapshot. Grader output never
writes back into the candidate workspace.

## Results and validation

```sh
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench monitor --output .emulatorbench/runs/new-public-run
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench report --output .emulatorbench/runs/new-public-run
.emulatorbench/venv/bin/python -m threadweave.evals.emulatorbench audit --output .emulatorbench/runs/new-public-run/emulatorbench-chip8
```

Each task retains raw events, model calls, verifier attempts, RLM/refinement events,
manifest, trajectory, task result and audit. Incomplete grading leaves the aggregate
null; failures preserve their category and any earlier legitimate grade separately.
Counters and state visibility do not establish behavioral use or learning benefit.
Review annotations require linked request/action/entry evidence and, for outcome
benefit, bracketing verifier attempts using the explicitly selected score field.

`tests/test_emulatorbench*.py` covers continuation, isolation, score separation,
reporting, rehearsal and native runtime behavior. Docker, Modal and pinned-source
checks are opt-in or skipped when their prerequisites are absent. Fixtures are
engineering tests, not benchmark scores. Local run directories remain ignored by Git.
