# Buffalo evaluation

The capability report contains exactly these five families. Runtime correctness tests are separate.

| Benchmark | Primary result | Official implementation |
| --- | --- | --- |
| ManyIH Coding | overall_passed_pct (0–100) | [ManyIH](https://github.com/JHU-CLSP/ManyIH) |
| ManyIH Instruction Following | ISR (0–1), with CSR secondary | [ManyIH](https://github.com/JHU-CLSP/ManyIH) |
| LongBench v2 | Overall (%) | [LongBench v2](https://github.com/THUDM/LongBench) |
| ARC-AGI-3 | RHAE (%) | [ARC toolkit](https://github.com/arcprize/arc-agi) |
| Factorio | technologies completed + current research progress (%) | [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment) |

## Run

Install Buffalo with `uv sync --extra dev`. Copy `configs/evaluation.example.json`
to a manifest outside the checkout, fill in the model and official paths, and run:

```sh
buffalo eval manyih-coding --config /absolute/evaluation.json
buffalo eval manyih-if --config /absolute/evaluation.json
buffalo eval longbench-v2 --config /absolute/evaluation.json
buffalo eval arc-agi-3 --config /absolute/evaluation.json
buffalo eval factorio --config /absolute/evaluation.json
buffalo eval all --config /absolute/evaluation.json
```

`uv run buffalo` works without activating the virtual environment. `threadweave eval`
is an alias for the same interface. `all` runs exactly the five rows above.
Default output is a fresh `results/evaluation/<run-id>` directory. `--output` must
be empty. A run containing any NOT RUN result exits with code 2, with exact errors
in `run.json`, `report.txt`, and official stderr logs. No credentials or installation
are silently substituted. `--check` validates setup without inference/gameplay and
writes NOT RUN, explicitly labeled preflight-only.

The default `--profile paired` runs a conventional full-history chat/tool harness
and the production Buffalo Runtime. `--profile base` and `--profile buffalo` run
one side. The base harness exposes fresh Python processes and the same official
environment action tool; filesystem and environment state persist. Buffalo uses
its persistent Python REPL, context management, recursive agents and durable state.
Both sides have the same model/provider/parameters (including reasoning and per-call
output cap), task IDs, raw starting input, seed, token/time/cost/turn/tool limits and
context capacity. Every Buffalo model call, including descendants and compaction,
is checked against the shared provider contract. Routing to another model is rejected.
The seed controls ARC environment initialization; a model seed is used only if
explicitly supported and set in `run.provider.parameters`.

Budgets apply per task to the entire root/descendant tree. Factorio is one task,
so its budget applies to the entire persistent run. Choose a sufficiently large
`wall_seconds`, `token_budget`, `max_turns`, `max_model_calls`, and action limit for
long-horizon experiments. Early final replies resume the same Factorio session
without resetting budgets, Python state, or the world. Agent tool errors are
observations for recovery; unavailable environments are blockers.

## Official checkouts and dependencies

Keep official sources and datasets outside the agent workspace. Each `source` must
be an unmodified Git checkout with the official origin. Set `commit` to pin it;
otherwise the detected exact commit is recorded. Each family can use its own
`python` interpreter with upstream dependencies installed. Relative paths are
resolved against the evaluation manifest, not the current working directory.
The official worker process holds reference answers, tests and scorecards. Only
public task instructions and observations enter the task workspace. The current
host execution model is for trusted agent code; separate users/VMs are necessary
for adversarial isolation of evaluator files from arbitrary host Python.

### ManyIH

Clone https://github.com/JHU-CLSP/ManyIH and install it in a dedicated environment:

```sh
git clone https://github.com/JHU-CLSP/ManyIH /path/to/ManyIH
python3 -m venv /path/to/manyih-env
/path/to/manyih-env/bin/pip install -e /path/to/ManyIH
```

Both families use the bundled official datasets. Coding calls the upstream
formatter and `judge_response` with the official default timeout, then uses
`analyze_results` for the official coding percentage. Per-task unit-test,
assertion and style outcomes are retained. Instruction Following passes the
original `input` role/content messages unchanged, calls the official constraint
evaluator, and uses `compute_accuracy` for ISR/CSR and domain breakdowns. Its
LLM judge uses the optional `judge` provider config, defaulting to the same model
as the agent. Only the judge transport is adapted; checker logic, prompts, null
constraint handling and hierarchy semantics remain upstream. Judge calls and
usage are recorded separately and included in total run resources. Official
checker errors block a headline score; their raw diagnostics are retained.

ManyIH instructions are pinned outside compactable trajectory history and inherited
by descendants. No system messages are flattened into user messages. Long task
instructions cannot be silently truncated to satisfy context capacity.

### LongBench v2

Clone https://github.com/THUDM/LongBench. Download the official dataset at an exact
Hugging Face revision in a separate data environment with `datasets` installed:

```python
from datasets import load_dataset
import hashlib, json
from pathlib import Path
revision = "EXACT_HUGGING_FACE_COMMIT"
data = load_dataset("THUDM/LongBench-v2", revision=revision, split="train")
path = Path("longbench-v2.json")
path.write_text(json.dumps(list(data), ensure_ascii=False))
print(hashlib.sha256(path.read_bytes()).hexdigest())
```

Set `dataset`, `dataset_revision`, and `dataset_sha256`. The official zero-shot
prompt template, exact upstream `extract_answer` function and `result.py` are used.
The latter runs unchanged; its raw table is preserved along with category scores.
The full document remains in Buffalo's `context['task']` and `task.txt`; the agent
must retrieve and reason over it itself. No external RAG or answer preprocessing
is performed. The baseline also receives the complete task, in its message when
it fits or through the same raw file when it exceeds context capacity.

### ARC-AGI-3

Install the official `arc-agi` checkout and its dependencies in Python 3.12 or newer.
For offline runs provide official environment files in `options.environments_dir`.
Use complete game IDs including their versions. Offline provenance hashes the
local environment files. For official online competition scoring select
`operation_mode: COMPETITION`; API/service availability is required. See the
[official toolkit guide](https://docs.arcprize.org/toolkit/overview).

Actions go to actual toolkit environments. The SDK's closed scorecard `score` is
RHAE in percent; it is never recomputed from level counts. Scorecard JSON and
recordings are preserved. Competition mode retains the official all-game denominator.
For budget curves:

```sh
buffalo eval arc-agi-3 --config /absolute/evaluation.json --token-budgets 100000,300000,1000000
```

Each budget point uses a fresh matched pair and records its contract. Two or more
successful points generate `rhae-vs-output_tokens.svg`; cost curves require known
API costs. Unknown costs are never converted to zero.

### Factorio

Install the official FLE checkout in its own environment and provide a licensed
headless Factorio server using the upstream FLE setup. Use a dedicated Docker
container labeled `buffalo.eval=true`. Its command must explicitly include
`--start-server` and the configured `container_save_path`; it must not select an
autosave or an arbitrary latest save. Configure the server's RCON address/port and
upstream FLE RCON password. `world_save` is the immutable host `.zip` starting save;
`world_id` identifies that world. Set the expected `factorio_version` and record
the Docker image through the manifest. See the example for all fields.

Before each profile the runner restores that exact full save and restarts only the
dedicated container, then initializes FLE with research locked and simulation paused between action calls. The same inventory
and initial research are checked for both sides. FLE's Python namespace and world
stay alive for all actions. Research is read from FLE's authoritative research-state
API. Progress is not inferred from model claims. `technologies_completed` counts
researched technologies; initial research is also retained. Checkpoints, research
snapshots, trajectory actions, world/run IDs and a final server-save reference are
persisted. No engine source, admin RCON or scorecard object is exposed as an agent tool.

## Reproducibility and reports

Each run records benchmark/source commit, dataset/environment version/hash, task
IDs, resolved model/provider/reasoning settings, Buffalo source hash/configuration,
budget and comparison-contract hash, UTC start/end times, input/output/total tokens,
API cost (null if unavailable), elapsed wall time, primary result, raw evaluator
output and trajectory paths. Provider-call journals identify root/parent/session
IDs. Accounting comes from the complete tree after descendants settle; internal
subsystem timings never enter the capability report. Conservatively estimated
provider usage is explicitly counted as estimated calls.

Use `--limit` or `task_ids` only for explicitly labeled subsets. They never become
full-benchmark claims. LongBench's official result script requires examples from
both difficulty groups and all three length groups; smaller selections that do not
cover them return NOT RUN at scoring and retain their predictions. Existing results
are never treated as a baseline; comparisons always use the same underlying model.

ManyIH IF repeats some upstream IDs. The runner preserves every row using `id@row_index`
for repeated IDs and records the upstream ID/index mapping; it never deduplicates tasks.
The official IF scorer records null values for conditional exclusions and some checks
on empty answers, including answers cut off by a task budget. These values and the
official ISR/CSR denominators are preserved in the raw results, rather than converted
to invented scores or treated as missing-environment failures.
