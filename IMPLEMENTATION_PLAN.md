# Coding product implementation

Audit before editing covered production source, tests, documentation and configs.
The unchanged baseline passed 70 tests.

Preserved: SQLite sessions/events/messages/versions, pending-action recovery,
Python workers/receipts, asynchronous daemon scheduling, recursive accounting,
provider transport, context boundaries and client detachment.

Replaced: production deterministic providers/defaults and the demonstration CLI.
Test providers and the extension fixture now live under tests only.

Implementation layers:

1. Explicit provider configuration and forward-only coding schema migration.
2. Incremental repository navigation, strict recoverable edits and Git checkpoints.
3. Local/container execution, baselines and independent coding verification.
4. Isolated candidates, measured benchmarks and durable experiments.
5. Searchable history, accounted/routed auxiliary calls, validated automatic
   refinement, executable skills and no-progress evidence.
6. Real external workload evaluation, CLI inspection and trajectory analysis.
7. Temporary-repository/hard-daemon-death tests, documentation and quality checks.

No mandatory planning graph, bundled dataset or invented performance score.
See README and docs for implemented security/recovery boundaries.
