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

The built-in chat provider handles streamed tool fragments, usage, nonstreamed
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
    runtime.tools.register(Tool(
        "inspect_size", "Inspect a permitted file.",
        InspectArgs, inspect, permissions=("workspace.read",),
    ))
```

Return JSON-compatible structured results. Full results become artifacts; model
previews are bounded and Python bridge callers receive complete values. Set
python_callable=false for tools that would reenter the same worker.

Use context.path for files, Editor for edits, GitWorkspace for checkpoints and the
configured Executor/run_command for processes. Respect cancellation and use
context.action_id as an external idempotency key where available. The harness cannot
undo arbitrary external effects. Tool metadata controls permissions, features and
coding-only exposure.

## Task/environment

Implement async prepare(context, task_config) -> dict and
async verify(context, task_config) -> Verification | None. Preparation and verification
must be idempotent/recoverable. Return passed=false for task failure; raise for
infrastructure/environment failure.

CodingTask supplies repository baseline/verification. WorkspaceTask remains for
embedded non-coding environments; it is not the CLI product workflow. Add new
domain capabilities without a separate scheduler or mandatory planner graph.

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
Versions record provenance and validation status; skill_inspect includes outcome
counts and quarantine state. Repeatedly failing versions require a validated
update/rollback. Syntax and permission declarations do not prove safety or constrain
arbitrary host Python.

Memories/prompt notes require text; subagent specs require an instruction. Every
StateEdit requires source event IDs and intended effect. Updates, deletion and
rollback append versions at controlled boundaries. Foundational policy is separate.
