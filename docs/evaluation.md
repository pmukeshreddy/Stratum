# External workloads and evidence

Evaluation uses the production Runtime, CodingTask, tools and verifier. No dataset,
task download or benchmark score is bundled.

Supply a JSON array/object or JSONL records. Each instance requires id, adapter,
repository and objective. Relative file paths resolve against the definition file.

## Repository issue

```json
{
  "id": "your-project-issue-123",
  "adapter": "repository_issue",
  "repository": "/absolute/path/to/repository",
  "base_commit": "a-real-commit-id",
  "objective": "The actual issue statement and acceptance criteria",
  "test_commands": [["python", "-m", "pytest", "-q"]],
  "verifier_commands": [["python", "-m", "pytest", "-q", "tests"]],
  "test_patch": "/absolute/path/to/evaluator-tests.patch"
}
```

An explicitly supplied HTTPS/SSH repository URL is also supported. Each run clones
the requested revision. An optional evaluator-only test patch is validated and
applied to a separate final verification copy, never to the delivered agent patch.
Missing repositories, patches, tooling or required_package produce explicit errors.

## Long-context coding

Use adapter=long_context with repository, objective, test_commands and context_bundle
containing actual task/context file paths. Files are copied into .task_context and
their references supplied to the model. Normal retrieval/compaction handles them.
Original workspaces and bundles are not modified.

## Kernel workloads

Use adapter=kernel with repository/workspace, objective, build_commands,
test_commands and benchmark. Compiler/correctness/benchmark commands must actually
exist in the supplied environment. GPU tooling is optional for other workloads.

Benchmark configuration contains command, correctness_commands, metric_regex,
repetitions, warmups, direction, required_improvement, noise_tolerance and
timeout_seconds. The regex must capture exactly one finite number per execution.
Median improvement plus explicit tolerance is the threshold, not a confidence
interval. p95 uses nearest rank. All raw measurements and outputs are retained.

## Runs, ablations and results

```sh
uv run threadweave --data /absolute/eval-state eval /path/to/instances.jsonl \
  --config configs/coding.json --repetitions 3 --seed 42 --output /path/to/results.jsonl
uv run threadweave analyze /path/to/results.jsonl
```

The CLI owns the outer evaluation batch; it is not a detached batch service.
Individual run state remains recoverable in its recorded data directory.

For ablations, vary the features block in separate configurations:

```json
{
  "features": {
    "persistent_repl": false, "subagents": false, "history_retrieval": false,
    "automatic_refinement": false, "experiments": false,
    "enhanced_code_index": false, "model_compaction": false
  }
}
```

Automatic refinement additionally requires refinement.automatic=true. Disabling
persistent_repl resets computational state between calls. Disabling enhanced_code_index
removes declaration parsing but retains file/hash tracking needed by verification.

Results record submitted/resolved configs, config hash, repetition and seed label.
The seed is not universally forwarded to a model; seed_applied_to_model is false.
Use explicit provider parameters if a particular endpoint supports seeded sampling.

JSONL records append durably; each run stores an eval_runs SQLite row and final.patch.
Fields include solved/unsolved, external verifier score, errors/outcome, elapsed
time, tokens, optional cost, turns, tool/Python calls, test/build runs, children,
experiments, retries, compactions, diff size and measured benchmark evidence.
Subscription cost is null/unavailable, including cost per solved task. For optional
API providers, zero cost with no configured/reported price is not evidence of free usage.

Analysis reports descriptive success rate, aggregate cost/turns/tool calls per
solved task, repeated actions/failed commands, retrieval/compactions, verifier
failures, child acceptance, experiments, command/model timing, file churn and
reverted edits. Model time includes completed calls; tool time measures completed
outer action spans without double-counting nested Python bridge calls. Command time
is also retained separately. Recovered uncertain durations are not invented.
It makes no causal claims or unsupported system comparisons.
