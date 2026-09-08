# Current implementation and verification scope

## Current capability

The normal model tool schema contains only `ipython(code)`. Internal tools retain
their typed schemas, permissions and common Environment action lifecycle. Those
schemas also supply Python method signatures and argument help, without entering
the model's tool list. Raw Python remains available. Use `repo.help("search")`,
`tests.help("related_to")`, `agents.help()` and `context.help()` inside a kernel.

Repository intelligence persists syntax evidence and module bindings in SQLite.
Python imports use AST nodes; Rust, Go, C/C++/CUDA and JS/TS imports use Tree-sitter
nodes. Resolution handles local Python packages, Rust modules, Go module paths,
C/C++ include directories and TS path mappings. `repo.context_for_symbol()` returns
bounded source excerpts as well as definitions and call evidence. Python
`repo.resolve(path, line, column)` uses Jedi static inference. Explicit
`repo.semantic(...)` requests use an installed LSP server; only results returned by
that server are labeled compiler/LSP semantic. Clangd was exercised locally.

Context includes a bounded working focus, source evidence, test/verifier failures,
selected durable state and child messages. Optional locally cached FastEmbed
embeddings combine with FTS/type/recency ranks; no model downloads occur in the
retrieval hot path. Compaction retains critical negative excerpts alongside a
model-generated summary. L2 values and L3 source events remain intact. Root budget
metadata is visible to every child.

Research/review kernels enforce filesystem write denial with macOS Seatbelt or
Linux bubblewrap, failing closed when enforcement is unavailable. Writable
candidate admissions use shared Git objects and independent worktrees, with
thread-local SQLite connections for concurrent preparation. `agents.wait()` defers
the next model turn until a message or timeout, avoiding model-based polling.
An exhausted turn-admission budget does not cancel an already-admitted sibling
turn; token, wall-time and other safety limits still apply.

Per-test coverage JSON can be imported into persistent test/file/line relationships.
Selection combines coverage, resolved imports, symbols, prior failures and layout;
each choice carries its reason. Framework JSON/JUnit and compiler text are retained
as both structured evidence and raw artifacts. Selection does not replace the
independent final verifier.

Mutation observation uses watcher candidates on normal cells, and a full metadata
reconciliation at verification trust boundaries. The verifier caches immutable
baseline/current state identities and compares content only on changed paths.
Structured editing retains conflict detection, journaling and rollback; arbitrary
Python edits have observation/audit guarantees, not transactional rollback.

Targeted kernel interrupts identify the active execution, protect snapshot commits,
and preserve the live namespace when Python can stop safely. Unresponsive native
execution escalates to kernel termination. Checkpoints cache exact comparisons of
flat scalar containers and contiguous numeric arrays, with content-addressed blobs
and isolated serialization failures. Process ownership records PID birth identity,
Python Popen calls and observed descendants; cleanup avoids PID-reuse mistakes.
Package distributions explicitly exclude trajectories, local databases and PDFs.

## Current limitation

This is not evidence of production-grade coding effectiveness. The paired coding
results, runtime measurements and API probes are under `results/current-quality/`.
Small public algorithm-repair tasks have ceiling effects and possible training-data
exposure. Their results do not establish effectiveness on substantial multi-file
issues, refactors, systems work or performance tasks.

* Structural bindings are not whole-program type resolution. Dynamic imports,
  dispatch, re-export chains, conditional compilation and monorepo settings remain
  incomplete. Non-Python LSP enrichments other than clangd are not live-validated.
  Files above 2 MB are not parsed. Full text searches still scale with searched data.
* Semantic retrieval is opt-in, requires the semantic extra and a cached encoder,
  and ranks at most 2,048 recent embedded events alongside full lexical history.
  It is not a semantic index of all artifacts or all durable knowledge.
* Compaction's deterministic critical-evidence retention is bounded/extractive;
  it cannot guarantee retention of every decision-relevant fact. Automatic evidence
  packets do not prove that the model uses their contents.
* Test coverage relationships are imported, not automatically gathered for every
  framework. Static selection is advice. C/C++ declarations, dynamic call graphs
  and framework-specific edge cases are not uniformly compiler-grade.
* Read-only kernels were enforced on macOS; Linux bubblewrap and Windows behavior
  were not exercised here. Read-only local host shell helpers require an isolated
  backend; Python subprocesses inside the read-only worker inherit its OS policy.
  MCP transport discovery/invocation is denied to read-only sessions because host
  and remote servers are outside that policy; server read-only hints are not
  treated as enforcement. The parent can supply retrieved MCP evidence.
  Trusted writable Python is not a security sandbox.
* Native descendants that escape, erase their ownership marker and reparent before
  observation can evade discovery. macOS has no cgroup containment equivalent in
  this implementation; Windows Job Objects are not implemented. A crash during
  candidate creation can leave a worktree requiring inspected cleanup.
* Exact detection of arbitrary mutable changes still requires memory comparison.
  Flat-container caches use bounded shadow copies, not O(1) mutation tracking;
  nested mutable objects may require serialization. Opaque OS resources are not
  blindly restored. Native calls may require hard-kill escalation.
* Full reconciliation is O(file metadata), deliberately retained for trust. It is
  not an O(1) verifier. Warm no-op cells use candidate tracking. Background writes
  are eventually observed; changes created and removed entirely within one cell
  need not appear as net mutations. There is no transactional guarantee for raw writes.
* Subscription output limits are client-observed; the transport does not expose a
  server-side output-token cap. Usage is measured, monetary API cost is unavailable.
* Read-call counts inferred from Python syntax are call sites, not exact filesystem
  read ranges. Evidence-consumption fractions and causal child usefulness are not
  reliably measured; the reports must not invent these quantities.

## Implementation

Key modules are `resolution.py`, `module_imports.py`, `lsp.py`, `repository.py`,
`semantic_retrieval.py`, `retrieval.py`, `context.py`, `kernel_api.py`, `runtime.py`,
`isolation.py`, `process_family.py`, `kernel.py`, `kernel_worker.py`, `snapshots.py`,
`mutations.py`, `coding.py`, `test_selection.py` and `test_evidence.py`. SQLite
migrations add module bindings, coverage relationships and embedding records.
Existing databases are migrated rather than reset.

## Evidence and reproduction

All model-double tests are distinct from the live subscription runs. Generated
repository corpora measure latency only, never coding success.

```sh
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
uv build

uv run python benchmarks/hardening.py --sizes 100 10000 30000 \
  --output /tmp/threadweave-performance.json
uv run python benchmarks/current_quality.py --output /tmp/threadweave-runtime.json

# Source is an official QuixBugs checkout; no correct_* implementations are copied.
uv run python benchmarks/quixbugs_suite.py prepare --source /path/to/QuixBugs \
  --output /tmp/quixbugs-input --count 25
uv run python benchmarks/quixbugs_suite.py freeze --source /tmp/quixbugs-input \
  --output /tmp/quixbugs-run
PYTHONPATH=/tmp/quixbugs-run/frozen uv run python /tmp/quixbugs-run/frozen/run.py \
  run --output /tmp/quixbugs-run --parallel 4
uv run python benchmarks/analyze_pair.py /tmp/quixbugs-run

# Separate real-model API probe, not a scored coding task.
uv run python benchmarks/live_capabilities.py --output /tmp/live-api \
  --turns 24 --tokens 100000
```

The paired profiles use the same provider, task inputs, revision, model parameters
and root budgets. `base` is minimal Python/bash plus the same independent verifier,
with additional Buffalo intelligence disabled. It is not a separate commercial
harness. Both profiles use the configured subscription provider, never a test
provider or API-key billing. The runner records source hashes, task/config snapshots,
events, usage, patches and independent verifier logs.
