# Runtime upgrade: implementation and source comparison

The baseline is Buffalo commit `1e46e739237d77edf140962da699c40120520048`. Prime source was read from `/tmp/buffalo-prime-reference/prime-agent-main`; the compared prompt, context-tree, refinement and Python snapshot files were verified byte-for-byte against `prime-agent-main (1).zip`.

## Changed subsystems

| Files | Actual implementation |
|---|---|
| `src/threadweave/context.py`, `chat.py`, `coding_adapter.py` | Replaced optional-execution doctrine throughout normal root instructions. IPython is the default control environment for inspectable, transformable, stateful and verifiable work. Added concrete decomposition, recovery and child-profile guidance. Direct responses remain possible; tool selection stays auto. |
| `models.py` | Automatic refinement defaults on; full verification no longer defaults to every turn. Removed the Python-only override that disabled waiting for children. Centralized kernel, verification, refinement pacing and evaluation-isolation policies. |
| `refinement_evidence.py`, `refinement.py`, `trajectory.py`, `storage.py` | Persistent evidence fingerprints and host prefilter; success/failure evidence; separate lifecycle events. Automatic reviews use bounded event packets with complete archives. Explicit manual refinement retains complete-record reduction. Version validation, atomic activation, conflict checks, provenance and rollback remain. Evaluation isolation hides global state and rejects global writes. |
| `kernel_state.py`, `snapshots.py`, `kernel_worker.py`, `kernel.py` | Replaced the worker's old checkpoint loop with one per-variable L2 lifecycle manager. Manifests record type, owner, memory estimate, serialized bytes, last use, importance, codecs, artifacts and recipes. Large useful values become explicit loadable handles. Stale values leave the namespace only after durable commit. Reconstruction recipes support dependencies and source hashes. Corruption warnings and a previous checkpoint allow recovery. Forks copy blobs and rebind safe paths and ownership. |
| `semantic_state.py`, `context.py`, `context_budget.py`, `host_api.py`, `kernel_api.py`, `runtime.py` | Compaction captures live goals, persistent unresolved work, child branches, conclusions, verification, harness versions and kernel manifests. `context.track/resolve/state` expose the ledger. Completion evidence resolves pending child entries. Evidence-backed resolutions remove repeated pending entries from both prior and incoming summaries; compaction-created requirements are resolvable through the same public API as explicitly tracked work. Raw stdout, code and data are no longer mistaken for unresolved instructions during extractive compaction. L1 boundaries synchronize L2 checkpoints without restarting the worker. Automatic compaction first projects the durable tree and REPL state through the existing compaction commit path; explicit or capacity-blocked compaction retains model assistance. A bounded live completion receipt and child status remain visible after compaction, preventing successful verification from being buried in retired transcripts. |
| `verification.py`, `runtime.py`, `host_api.py`, `kernel_api.py` | Level 1 checks execution errors and changed Python/JSON syntax; level 2 schedules configured targeted commands or supported pytest selectors after changes; level 3 preserves the original independent adapter gate. Full gates wait for required children; completion-only waiting yields without spending another model call. Test cache paths remain in provenance but cannot schedule another targeted run. Failures become bounded structured evidence, with complete logs/receipts archived. `verify.run(level=...)` and `verify.latest()` expose them to REPL and delegated work. |
| `runtime.py`, `kernel_api.py`, `host_api.py` | `agents.followup(handle, instruction)` reuses child identity, kernel and context. Selected task-local harness versions are copied into child-owned state with parent-version provenance. Existing asynchronous admission, cancellation, model/thinking overrides, worktree profiles and depth limits remain. |
| `tests/test_runtime_upgrade.py` and existing test modules | Real-worker lifecycle flows, recursive evidence/refinement, oversized state, restart/corruption, fork ownership, targeted/full verification, pending-child gates and prompt doctrine. Existing expectations were changed only for intentional behavior changes; complete manual reduction and its cancellation tests remain. Explicit model-compaction coverage remains alongside the new automatic projection test. |
| `evals/runtime_upgrade.py`, `evals/root_loop_validation.py` | Added unforced real-model repository tasks with independent acceptance checks and trace metrics; updated the old behavioral evaluator's obsolete prompt assertion. |

## Central defaults

- L2 namespace pressure: 128 MiB; individual variable pressure: 16 MiB.
- Inline snapshot records: 64 KiB; aggregate serialized working snapshot allowance: 64 MiB; per-artifact allowance: 512 MiB.
- Mutable serialization cache allowance per value: 8 MiB; stale age: 20 cells; periodic L2 lifecycle checkpoint: 5 cells. Recovery checkpoints still happen after each execution.
- Snapshot serialization time budget: 5 seconds total, 1 second per variable. Size traversal: 10,000 nodes, with approximation recorded.
- Generated cache path components excluded from targeted scheduling: `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache` (configurable). Their mutation evidence is still retained.
- Targeted verification cadence: 3 changed-work turns; at most 8 selected test files. Pending completion waits up to 30 seconds, waking earlier on messages.
- Automatic compaction uses structured projection first when a durable REPL snapshot exists; protected unresolved work is preserved and model reduction remains the fallback.
- Automatic successful-progress review cadence: 3 turns; existing failure, child, experiment, compaction and completion triggers remain. Trivial and duplicate evidence is filtered before inference.

## Prime versus Buffalo

| Capability | Inspected Prime implementation | Buffalo now | Difference |
|---|---|---|---|
| Root control | `prompts/rlm.ts` makes persistent IPython the control environment | Consistent REPL-first root doctrine and auto tool selection | Similar control philosophy; Buffalo retains independent gates |
| Variable snapshots | `prime-agent-runtime/src/rlm/repl.py::_snapshot_state` serializes each variable with dill through a capped writer; 16 MiB variable and 256 MiB aggregate defaults | Explicit codecs plus restricted cloudpickle; per-variable manifests, recipes, artifact handles, ownership and recovery | Buffalo has richer lifecycle decisions; Prime's capped streaming writer provides stronger allocation bounds |
| Oversized values | Explicit compaction prunes names exceeding the per-variable cap, after committing snapshot and manifest | Offloads useful values; retires stale values with recovery metadata; never evicts after a failed manifest write | Buffalo preserves recoverability of large useful values |
| Branch/context state | Native parent-linked session branches, branch summaries and context-tree usage display | Durable semantic work ledger and per-session child-tree projections tied to L2 manifests | Buffalo adds explicit semantic/L2 linkage; Prime retains richer interactive branch navigation |
| Continual refinement | Automatic review gate and session/global continual state; prompt explicitly allows local progress/blocker notes | Default-on evidence filtering, validation, activation, later-input events; task status kept in trajectory state | Buffalo adds host novelty filtering and executable-skill validation/provenance |
| Verification | General execution and task-driven checks; no equivalent Buffalo adapter gate in the compared core | Continuous/targeted/full scheduling over existing coding and official-evaluator gates | Buffalo retains the stronger explicit acceptance/receipt system |
| Children | Asynchronous persistent RLM sessions, messaging, observation, recursion and overrides | Those mechanics plus worktree profiles, child-owned inherited harness versions and follow-up reuse | Buffalo retains workspace/provenance advantages; review profile selection still depends on the model |

## Removed and retained paths

Removed the old worker snapshot/restore loop, optional-execution prompt philosophy, automatic full-record reduction on every large refinement packet, and the Python-specific `wait_for_children=False` override. The explicit `control_plane='direct'` interface remains supported by CLI/extensions and contract tests; it is not the default model execution surface. Manual full-evidence refinement remains an explicit operation rather than competing automatic scheduling.

## Demonstrated limits

- The live model still sometimes prints entire files or invokes unavailable `python` commands. A stronger prompt is a behavioral prior, not a guarantee.
- Refinement receipt of a version means it was in a later model input. It does not prove causal improvement. In one ledger trace, the bounded-inspection correction began before the learned note activated.
- Automatic targeted test selection currently supports known pytest command syntax. Other frameworks and plugin-specific options require `verification.targeted_commands`; configured final gates remain unchanged.
- Memory estimates are approximate. Native allocations and nested object graphs are not measured completely, and JSON serialization can allocate temporary buffers. Cross-variable alias identity is not preserved by independent codecs.
- L3 artifacts are durable and are not automatically garbage-collected. Reconstruction of opaque resources still requires explicit recipes.
- Refinement planning can be interrupted when its root task completes; the final live recheck retained the evidence but activated no version from that late planner. Explicit model-requested gates and child follow-ups can still repeat full verification.
- The live sample is deliberately small and used a 24,000-token context to exercise pressure. Intermediate revisions included wall-limit failures and a provider transport failure. The trace archives retain them; successful smoke cases do not establish a general benchmark improvement.

## Final validation

`pytest`: 503 passed, 10 skipped, 2 forkpty deprecation warnings (122.15 seconds). Focused final resolution tests: 16 passed. Ruff and diff checks passed. Source distribution and wheel built successfully. See commands.txt, tests.log, resolution-tests.log, build.log and behavior.md for the actual commands and observations.
