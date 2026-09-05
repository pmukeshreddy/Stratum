"""Test-only model actions for hard process-death recovery on a real coding workload."""

import asyncio

from .conftest import response
from .test_coding import FIX


class CodingRecovery:
    def __init__(self, runtime):
        self.runtime = runtime

    async def invoke(self, request, emit):
        await asyncio.sleep(0.05)
        if request.parent_id:
            actions = [
                response(
                    "apply_patch",
                    patch='--- a/mathops.py\n+++ b/mathops.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n+    """Add operands."""\n     return a + b\n',
                ),
                response("run_tests"),
                response(
                    "agent_message", recipient_id=request.parent_id, body="Candidate is correct"
                ),
                response("finish", result="Documented and tested"),
            ]
            return actions[min(request.turn, len(actions) - 1)]
        if request.turn == 6 and not (self.runtime.store.directory / "continue").exists():
            await asyncio.Event().wait()
        if request.turn == 7:
            child = next(
                s for s in self.runtime.store.sessions(root_id=request.root_id) if s.parent_id
            )
            return response("candidate_apply", child_id=child.id)
        actions = [
            response("repo_map"),
            response("symbol_search", query="add"),
            response("python", code="retained = {'observed': 12}\nretained"),
            response("apply_patch", patch=FIX),
            response(
                "agent_spawn",
                instruction="Document the corrected function and run tests",
                name="candidate",
            ),
            response("agent_wait", seconds=0.1),
            response("python", code="assert retained['observed'] == 12\nretained"),
            response("git_diff"),
            response("run_tests"),
            response("finish", result="Recovered coding run verified"),
        ]
        return actions[min(request.turn, len(actions) - 1)]


def install(runtime):
    runtime.providers["coding_recovery"] = CodingRecovery(runtime)
