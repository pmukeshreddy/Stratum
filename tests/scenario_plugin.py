"""Deterministic provider loaded by a real daemon in process-recovery tests."""

import asyncio

from threadweave.models import Action, ModelResponse, Usage

from .test_continual_harness import learning_assessment


def action(tool_name, **arguments):
    return Action(name=tool_name, arguments=arguments)


class RecoveryScenario:
    def __init__(self, runtime):
        self.runtime = runtime

    async def invoke(self, request, emit):
        await asyncio.sleep(0.08)
        if request.metadata.get("purpose") == "refinement":
            import json

            return ModelResponse(
                text=json.dumps(
                    {
                        "summary": "Retained computation",
                        "edits": [
                            {
                                "action": "create",
                                "kind": "memory",
                                "id": "values",
                                "title": "Values",
                                "content": "The working values contain integers 0 through 999.",
                                "metadata": {"learningAssessment": learning_assessment()},
                            }
                        ],
                    }
                )
            )
        if request.metadata.get("purpose") == "refinement_review":
            return ModelResponse(text='{"shouldRefine":false,"rationale":"Already retained"}')
        turn = request.turn
        from pathlib import Path

        workspace = Path(self.runtime.store.session(request.session_id).workspace.path)
        scenario = request.config.parameters.get("scenario")
        if scenario in ("interrupted_python", "interrupted_command"):
            if turn == 0:
                code = (
                    "import os, time\nfrom pathlib import Path\n"
                    "p = Path('effect.txt')\n"
                    "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
                    "Path('worker.pid').write_text(str(os.getpid()))\n"
                    "time.sleep(60)"
                )
                if scenario == "interrupted_python":
                    actions = [action("python", code=code)]
                else:
                    import sys

                    actions = [action("process_run", command=[sys.executable, "-c", code])]
            elif turn == 1:
                actions = [
                    action(
                        "python",
                        code="assert (workspace / 'effect.txt').read_text() == 'x'\n'Inspected uncertain effect'",
                    )
                ]
            else:
                actions = [action("finish", result="Recovered without duplicate effects")]
            return ModelResponse(actions=actions, usage=Usage(input_tokens=80, output_tokens=30))
        if turn >= (2 if request.parent_id else 3) and not (workspace / "continue").exists():
            await asyncio.Event().wait()
        if request.parent_id:
            if turn == 0:
                code = "retained = list(range(50))\nlen(retained)"
                actions = [action("python", code=code)]
                if request.name == "left":
                    actions.append(action("agent_spawn", instruction="Nested task", name="leaf"))
            elif turn == 1:
                actions = [
                    action(
                        "agent_message",
                        recipient_id=request.parent_id,
                        body=f"{request.name} ready with retained state",
                    ),
                    action("agent_wait", seconds=300),
                ]
                if request.name == "left":
                    right = next(
                        s
                        for s in self.runtime.store.sessions(root_id=request.root_id)
                        if s.name == "right"
                    )
                    actions.insert(
                        0, action("agent_message", recipient_id=right.id, body="sibling evidence")
                    )
            elif turn == 2:
                actions = [action("python", code="assert sum(retained) == 1225\nsum(retained)")]
            else:
                actions = [action("finish", result=f"{request.name} recovered")]
        elif turn == 0:
            actions = [action("python", code="values = list(range(1000))\nlen(values)")]
        elif turn == 1:
            actions = [
                action("agent_spawn", instruction="Compute left", name="left"),
                action("agent_spawn", instruction="Compute right", name="right"),
                action("python", code="await refine.run('Remember the retained computation')"),
            ]
        elif turn == 2:
            actions = [action("agent_wait", seconds=300)]
        elif turn == 3:
            # Gate continuation on an explicit post-restart human message. Children may wake
            # the parent first; the persisted provider turn still drives the same computation.
            code = "assert len(values) == 1000\nanswer = sum(values)\nanswer"
            actions = [action("python", code=code)]
        elif turn == 4:
            actions = [
                action(
                    "python",
                    code="tools.call('workspace_write', path='answer.txt', content=str(answer))",
                )
            ]
        else:
            actions = [action("finish", result="Recovered computation verified")]
        return ModelResponse(actions=actions, usage=Usage(input_tokens=80, output_tokens=30))


def install(runtime):
    runtime.providers["recovery_scenario"] = RecoveryScenario(runtime)
