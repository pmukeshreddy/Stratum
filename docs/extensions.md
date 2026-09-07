# Extension contracts

Extensions are trusted installed Python modules configured as
"extensions": ["your_package.integration:install"]. They load during admission and
recovery. Pin their source/dependency versions for reproducibility.

## Provider

Implement async invoke(ModelRequest, emit) -> ModelResponse. Translate messages
and tool schemas to the provider protocol; emit optional text fragments and return
model-selected actions plus usage. Propagate cancellation. Register the instance in
runtime.providers under an explicit provider name.

Raise HarnessError with category provider for transport failures and model for
malformed responses; never include credentials/headers in errors. The runtime owns
timeouts, retries, reservations and accounting for normal and auxiliary calls.
Unknown usage must be marked usage_reported=false with conservative bounds.

The default codex_subscription provider uses official Codex client/auth libraries
for individual model requests, without a Codex agent loop. See [transport](subscription.md).
The optional chat provider handles streamed tool fragments, usage, nonstreamed
responses, environment credentials and compatible base URLs. Deterministic response
providers exist only in tests.

## Tool

```python
from threadweave.models import Record
from threadweave.tools import Tool


class InspectArgs(Record):
    path: str


async def inspect(context, args):
    return {"bytes": context.path(args.path).stat().st_size}


def install(runtime):
    runtime.tools.register(
        Tool(
            "inspect_size",
            "Inspect a permitted file.",
            InspectArgs,
            inspect,
            permissions=("workspace.read",),
        )
    )
```

Return JSON-compatible structured results. Full results become artifacts; model
previews are bounded and Python bridge callers receive complete values. Set
python_callable=false for tools that would reenter the same worker.

Use context.path for files, Editor for edits, GitWorkspace for checkpoints and the
configured Executor/run_command for processes. Respect cancellation and use
context.action_id as an external idempotency key where available. The harness cannot
undo arbitrary external effects. Tool metadata controls permissions and feature
availability; a tool does not require selecting a coding task adapter.
In default Python-control mode, newly registered tools are programmatic only:
`tools.call('inspect_size', path='file.py')`. The model schema remains ipython.
Do not add direct model tools to implement an Environment capability.

## Task/environment

Implement async prepare(context, task_config) -> dict and
async verify(context, task_config) -> Verification | None. Preparation and verification
must be idempotent/recoverable. Return passed=false for task failure; raise for
infrastructure/environment failure.

WorkspaceTask is the default Environment policy for both CLI and embedded use.
CodingTask optionally supplies repository baseline/verification. Environment owns
that policy, not the Root Session. Add capabilities without a separate scheduler
or mandatory planner graph. Ordinary interaction must not require a coding task.

External Instance loaders cover repository issues, context bundles and kernel
workspaces. A new dataset loader should translate supplied instances into those
contracts and reuse Runtime.

## Executable skills

Skill content requires name, description and executable code, plus inputs and
required_permissions (default empty-object inputs and python permission). Inputs
support a validated subset of JSON Schema: explicit scalar/object/array types,
properties/required/additionalProperties, items, enum, minimum/maximum and string
minLength/maxLength. Unsupported schema keywords fail instead of being ignored.

Code executes in the trusted worker with skill_inputs and normal output capture.
Use `await skills.run(name, **inputs)` inside an active cell: preparation validates
in the daemon, execution stays in the existing kernel and outcomes return through
the bridge. It does not recursively reenter a busy worker. SKILL.md Python package
discovery and MCP setup are described in [Python APIs](python-control-plane.md).
Versions record provenance and validation status; skill_inspect includes outcome
counts and quarantine state. Repeatedly failing versions require a validated
update/rollback. Syntax and permission declarations do not prove safety or constrain
arbitrary host Python.

Memories/prompt notes require text; subagent specs require an instruction. Every
StateEdit requires source event IDs and intended effect. Updates, deletion and
rollback append versions at controlled boundaries. Foundational policy is separate.
