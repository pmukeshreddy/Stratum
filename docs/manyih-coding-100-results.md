The domain-generality refactor was implemented and verified before model evaluation. All eight Part A gates passed: adapter-owned tools, adapter-owned REPL namespaces, generic context without coding leakage, adapter-specific task configuration, adapter-owned child profiles, generic verification, preserved RLM inheritance, and the existing suite. Production-path details are in [the architecture audit](domain-generality-audit.md); the timestamped [gate](../results/domain-generality/gate.json) was saved before the smoke runs. The architecture suite passed 451 tests with 9 skips. After evaluation integration and reporting tests, the final suite passed **457 tests, with 7 skipped**, plus Ruff lint/format, package build and diff checks. CI defines no separate type checker. Native provider sources were unchanged; the pinned bridge was built and used.

**CODEX WINS this fixed 100-task ManyIH Coding experiment: 66% overall versus Buffalo's 62%.** Difference means Buffalo minus Codex. This is a measured subset comparison, not an estimate of a published full-dataset score.

| Metric | Buffalo | Codex | Difference |
| --- | ---: | ---: | ---: |
| Functional pass | 90/100 = 90% | 90/100 = 90% | 0 percentage points |
| Style pass | 69/100 = 69% | 72/100 = 72% | −3 percentage points |
| Overall pass | 62/100 = 62% | 66/100 = 66% | −4 percentage points |
| Input tokens, reported lower bound | ≥173,384 | ≥1,745,845 | Unavailable |
| Output tokens, reported lower bound | ≥60,412 | ≥44,928 | Unavailable |
| Total tokens, reported lower bound | ≥233,796 | ≥1,790,773 | Unavailable |
| Elapsed system phase, 4 parallel tasks | 565.69 s | 514.06 s | +51.63 s |
| Summed task wall time | 2,168.35 s | 1,958.59 s | +209.76 s |
| Median task wall time | 16.79 s | 15.94 s | +0.85 s |
| Estimated API cost | Unavailable | Unavailable | Unavailable |

Buffalo's relative overall change is `(62 − 66) / 66 = −6.06%`. Both passed 59 tasks; Buffalo alone passed 3; Codex alone passed 7; both failed 31. Buffalo-only task IDs: **19, 37, 41**. Codex-only task IDs: **18, 32, 33, 69, 77, 96, 97**.

Provider usage is incomplete for Buffalo task **50** and Codex tasks **61 and 76**, which recovered from interrupted transports. The measured portions are included above; missing usage is never replaced by Buffalo's output reservations. Exact input/output/total fields are `null` for those tasks and in the aggregate. `known_*` fields preserve reported usage. Native Codex reported 1,367,552 cached input tokens, included in its input count; Buffalo reported zero cached input tokens. Both used the existing ChatGPT subscription, so no reliable per-call API cost was available. The experiment took 1,080.20 seconds overall, excluding the separate smoke run and subsequent audits. Each system's phase timer includes its official grading and aggregation; per-task times cover agent execution.

The dataset and grading code came from [JHU-CLSP/ManyIH](https://github.com/JHU-CLSP/ManyIH), pinned at **3287f82436fff86d7506b7737dbd3b5e2e524d01**. The bundled `manyih/data/coding.json` has SHA-256 `57638c25d6f9254985dd26cc382fd16772cabf57ff2e5db0185603206092b61a`. Selection was the first 100 rows in its canonical order, using official sample IDs, which differ from underlying MBPP problem IDs. No task was selected based on outcomes. Exact IDs are saved in [task-ids.json](../results/manyih-coding-100/task-ids.json):

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
```

The official `format_datapoint_for_llm` constructed both agents' task text using the dataset's `ordinal` hierarchy and `boundary` annotation settings. The official `judge_response` graded the raw final answer, applying functional assertions and the applicable expected styles, with a 3-second timeout per assertion. `analyze_results` produced the official aggregate output. No replacement grader or LLM judge was used. Hidden assertions, expected-style metadata, and reference answers were held by the grading worker; agent workspaces initially contained only the same `TASK.txt`, `.gitignore`, and a clean Git repository. Prompt and file hashes match for all 100 pairs.

The production Buffalo source used for all model runs was **983881a60b4407284e6ba17ef3de9b1186358320**. Buffalo ran through `Runtime` and the real `coding` adapter, using `configs/coding.json` with ordinary capabilities enabled. Because ManyIH requests a final implementation rather than a patch against a supplied repository, baseline capture, required visible tests, and required file changes were disabled for this evaluation task. This does not expose hidden tests or alter the official task instructions. No ManyIH solving strategy was added to the runtime. Each task had fresh session and harness state; state was not transferred between tasks.

Both systems used **gpt-6-astra**, **xhigh** reasoning, a **300-second per-task wall limit**, Python **3.12.12**, the same dependency environment and network access, and one task attempt. Both used the native ChatGPT Responses provider. Native Codex was **0.153.4** at `/opt/homebrew/bin/codex`; Buffalo used its pinned native inference bridge based on revision **3d2ee51ca2d5db578f328aa75e20aa22c0197c9a**. Dependencies are recorded in [dependencies.txt](../results/manyih-coding-100/dependencies.txt). Four tasks ran concurrently within a system. The two paired smoke tasks, IDs 0 and 1, all passed the official functional and style checks; those four attempts are excluded from the full denominators. The fresh full 100 Buffalo runs completed before the full 100 native Codex runs began.

Native Codex ran `codex exec` with a fresh private `CODEX_HOME` for every task, the existing authentication file linked only while running, `--ignore-user-config`, and `--ignore-rules`. It received no Buffalo state, tools, REPL, memory, compaction or refinement. The final native model/effort settings and prompts were checked in saved rollouts. Credential files are not in the artifacts.

Conditions that could not be identical, and behavior observed in the traces:

- The official system text was supplied as a Buffalo system instruction and a native Codex developer instruction. Its wording was identical and above the user task in both systems. Each agent retained its own normal foundational instructions and tools.
- Both retained native defaults for temperature and provider output length. Buffalo reserved 32,768 output tokens per call for accounting and imposed a 3-million-token runtime ceiling; native Codex had no equivalent per-task token ceiling. The reservation is not a transmitted hard output cap.
- Buffalo allowed three accounted transport attempts; Codex retained its native request/stream retry policy. No completed task was rerun. Recovered transport failures did not remove tasks from either denominator.
- Buffalo's normal optional-work budget guard deferred completion-triggered automatic refinement on **all 100 tasks**: the 300-second task budget could not reserve the default coding tool timeout plus auxiliary model time. Automatic refinement was configured on, and explicit harness capabilities remained available, but this experiment did not exercise automatic learning. No configuration was changed after observing results.
- Across Buffalo's full run, traces recorded two Python executions and two tool calls; most tasks ended with a direct final answer. Native Codex used seven web operations across tasks 7, 44 and 78, plus one local command on task 44. Tasks 7 and 44 supplied relevant GeeksForGeeks URLs in their official prompts; task 78 searched the Newman–Shanks–Williams recurrence. These were normal native capabilities. None of those tasks accounts for a discordant overall win: both failed 7 and both passed 44 and 78. No hidden evaluator files were accessed in the observed tool calls.

Exact execution commands, run from the Buffalo repository, are preserved in [commands.txt](../results/manyih-coding-100/commands.txt). The principal commands were:

```sh
.venv/bin/python -m threadweave.evals.manyih_coding --source /tmp/buffalo-manyih-reference --output /tmp/buffalo-manyih-coding-100 --smoke-only
.venv/bin/python -m threadweave.evals.manyih_coding --source /tmp/buffalo-manyih-reference --output /tmp/buffalo-manyih-coding-100 --concurrency 4
.venv/bin/python -m threadweave.evals.manyih_coding --source /tmp/buffalo-manyih-reference --output /tmp/buffalo-manyih-coding-100 --audit-only
```

Each native invocation's exact arguments and stdin are in its `command.json`. The command shape was:

```text
/opt/homebrew/bin/codex exec --ignore-user-config --ignore-rules --json --color never
  -m gpt-6-astra -c model_reasoning_effort="xhigh" -c approval_policy="never"
  --sandbox danger-full-access -c developer_instructions=<exact official system text>
  -C <task workspace> -o <answer.txt> -
```

The [manifest](../results/manyih-coding-100/manifest.json), [summary](../results/manyih-coding-100/summary.json), [Buffalo records](../results/manyih-coding-100/buffalo-results.jsonl), [Codex records](../results/manyih-coding-100/codex-results.jsonl), official analyses and [integrity audit](../results/manyih-coding-100/integrity-audit.json) are checked into the repository. Record artifact paths in that copy are relative to the full local archive. Raw model/session traces remain in the local archive:

```text
/Users/mukeshreddypochamreddy/Downloads/buffalo-manyih-coding-100-results/
  buffalo-results.jsonl
  codex-results.jsonl
  full/buffalo/<task_id>/provider-calls.jsonl
  full/buffalo/<task_id>/state/history.sqlite3
  full/codex/<task_id>/codex.jsonl
  full/codex/<task_id>/native-sessions/
  full/<system>/<task_id>/official-grade.json
  official/coding-buffalo.json
  official/coding-codex.json
  official-buffalo-summary.json
  official-codex-summary.json
  task-ids.json
  validation/
  official-source.bundle
  independent-audit.py
```

Raw traces, answers, commands and SQLite stores were copied without modification from `/private/tmp/buffalo-manyih-coding-100`. Only the copied result records' artifact links were relocated; original historical paths remain in raw traces and `original_run_artifact`. The saved official-source Git bundle preserves the evaluator revision. The final audit verified 100 unique expected IDs per system, identical task sets, all failures retained, per-task and aggregate official grades in agreement, matched model/reasoning, no Buffalo instructions in the observed baseline trajectory, and totals reconciled to per-task records. The two systems produced 200 complete final answers with no task timeout or crash. These results support the winner for this fixed subset under the recorded conditions, including the observed refinement deferral and native tool differences.
