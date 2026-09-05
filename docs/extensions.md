# Providers, tools, and task environments

The core depends on three small contracts. The model controls workflow; extensions
add execution capabilities or define what success means in an environment.

## Register an extension

Install a Python module containing `install(runtime)`, then include its import
reference in the run configuration:

```json
{"extensions": ["my_package.integration:install"]}
```

The daemon loads the same references before admission and recovery. These are
trusted host-side extensions, not arbitrary module paths selected by a model. Keep
the module and dependency versions pinned alongside your evaluation configuration.
The example [extension module](../examples/extension.py) can be loaded as
`examples.extension:install` when running from the repository checkout.

## Add a provider

Implement an asynchronous `invoke(ModelRequest, emit) -> ModelResponse` method:

```python
from threadweave.models import Action, ModelResponse, Usage, HarnessError

class MyProvider:
    async def invoke(self, request, emit):
        # Translate request.messages and request.tools into your provider's API.
        # Use request.config.model, parameters, and max_output_tokens.
        # Await your HTTP/SDK operation so cancellation propagates.
        await emit("optional streamed text")
        return ModelResponse(
            text="I will inspect the workspace.",
            actions=[Action(name="workspace_list", arguments={"path": "."})],
            usage=Usage(input_tokens=100, output_tokens=20),
            usage_reported=True,
        )

def install(runtime):
    runtime.providers["my-provider"] = MyProvider()
```

Providers may return zero or several actions. Never treat a failed transport as a
model's explicit finish. Raise `HarnessError("provider", code, message,
retryable=True)` for transient API failures and category `model` for malformed
model-generated actions. Propagate `asyncio.CancelledError`. The runtime owns retry
policy, durable invocation records, call timeouts, and reservations.

Set `usage_reported=False` if actual token counts were unavailable, and return
conservative input/output bounds in `usage`. Do not increment the `model_calls`,
`turns`, or other runtime-owned counters yourself. Cost can be supplied in `usage.cost`.
For a hard preflight cost budget, configure input/output prices as well.

`ChatProvider` demonstrates streamed tool-call fragments, final usage chunks,
non-streamed responses, cancellation, authentication through environment variables,
and structured errors. `ScriptedProvider` indexes deterministic responses by the
persisted session name/turn and is intended for tests. Providers do not get implicit
access to L2/L3; only the assembled request is sent to external APIs.

## Add a tool

```python
from pydantic import Field
from threadweave.models import Record
from threadweave.tools import Tool

class CountArgs(Record):
    path: str
    limit: int = Field(default=1000, ge=1, le=100000)

async def count_lines(context, args):
    path = context.path(args.path)  # Enforces the workspace boundary, including symlinks.
    with path.open() as stream:
        count = sum(1 for _, _line in zip(range(args.limit), stream))
    return {"lines": count, "limit": args.limit}

def install(runtime):
    runtime.tools.register(Tool(
        "count_lines", "Count a bounded number of lines in a workspace file.",
        CountArgs, count_lines, permissions=("workspace.read",),
    ))
```

Arguments are validated by Pydantic and published as JSON Schema. Return
JSON-compatible structured results. The runtime retains full results in artifacts
and exposes bounded previews. Python calls receive full results through the bridge.
Use `python_callable=False` if the tool cannot safely execute inside that bridge;
tools that recursively invoke the same Python worker must be disabled there.

For side effects, `context.action_id` is a stable idempotency key. The journal
records the call before execution and its result afterward. The runtime does not
automatically repeat side-effecting tools after an uncertain interruption. Raise
structured `HarnessError("tool", ...)` failures for predictable tool errors. Respect
cancellation, and put CPU-heavy/blocking work in a process or explicitly managed
thread rather than blocking the event loop. A timed-out thread cannot be forcibly
stopped; use subprocesses for work that needs hard cancellation.

## Add a benchmark or interactive environment

Implement `prepare(context, task_config)` and
`verify(context, task_config) -> Verification | None`:

```python
from threadweave.models import Verification

class ExperimentTask:
    async def prepare(self, context, task):
        # Validate paths/services and return bounded descriptive environment data.
        return {"experiment": task.specification, "metrics": task.success_metrics}

    async def verify(self, context, task):
        measurement = read_measurement(context.path("measurement.json"))
        threshold = task.verifier_options["maximum_error"]
        return Verification(
            passed=measurement["error"] <= threshold,
            details=measurement,
            metrics={"error": measurement["error"]},
        )

def install(runtime):
    runtime.adapters["experiment"] = ExperimentTask()
```

Then configure `task.adapter="experiment"` and a verifier name other than `"none"`,
such as `"measurement"`. Custom verifier names are owned by the adapter. Attach
dataset IDs, task specifications, benchmark revisions, success thresholds, and
environment references in the serialized task configuration. Add tools for actions
specific to the environment. Coding tasks, kernel optimization, research, and
interactive environments all run through the same session loop.

`prepare` and `verify` must be idempotent or read-only because interrupted preparation
or verification can run again after recovery. Raise an exception for infrastructure
failure; return `Verification(passed=False, ...)` for a real task failure. These
have different event categories and retry behavior. `details` is retained in an
artifact and bounded before being returned to L1. `metrics` stores quantitative
measurements independent of the pass/fail gate.

The built-in workspace adapter supports no verifier, file existence/content checks,
and argv-based commands whose zero exit status passes. Command verifiers require
the `process` capability. Child sessions inherit workspace/configuration but receive
their own instruction and have the parent's completion verifier disabled. Register
environment tools to let a child perform additional checks when appropriate.
