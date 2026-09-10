"""A small real runtime/kernel session with deterministic model responses."""

import json
import os
from pathlib import Path

from threadweave.models import ModelResponse, Outcome
from threadweave.runtime import Runtime

from .conftest import eventually, response
from .test_continual_harness import edit, proposal


class LearningSession:
    def __init__(self, resume=False):
        self.requests = []
        self.resume = resume
        self.active = self.peak = 0

    async def invoke(self, request, emit):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            purpose = request.metadata.get("purpose", "agent")
            if purpose == "refinement":
                return ModelResponse(text=json.dumps(proposal(edit())))
            if purpose == "refinement_review":
                return ModelResponse(
                    text=json.dumps(
                        {
                            "shouldRefine": False,
                            "rationale": "The project lesson is already retained.",
                        }
                    )
                )
            if not self.resume and request.turn == 0:
                return response(
                    "ipython",
                    code="""workspace.joinpath('example.py').write_text('def add(a, b): return a + b\\n')
observed_lesson = 'Run checks in the project environment'
receipt = await refine.run('Remember the project validation environment')
assert receipt['scheduled']
assert (await refine.status())['in_flight']
assert not harness.list('memory')
print({'scheduled': receipt['scheduled'], 'state_before_boundary': harness.list('memory')})""",
                )
            text = json.dumps(request.messages)
            assert "Run checks in the project environment" in text
            if self.resume:
                assert "# Continual Harness State" in text
                self.resume = False
                return response(
                    "ipython",
                    code="assert harness.get('memory', 'lesson').content == 'Run checks in the project environment'\nawait compact()\nprint('Resumed lesson verified; compact checkpoint requested')",
                )
            assert "Run checks in the project environment" not in request.messages[0]["content"]
            return ModelResponse(
                text="I retained the project validation lesson and will use the project environment."
            )
        finally:
            self.active -= 1


async def test_root_schedules_learns_continues_resumes_and_auto_reviews(tmp_path, python_config):
    directory = tmp_path / "data"
    python_config.limits.max_turns = 8
    provider = LearningSession()
    runtime = Runtime(directory, providers={"mock": provider})
    root = runtime.create(
        "Write a tiny add module; retain and use the validation lesson.",
        tmp_path,
        config=python_config,
    )
    try:
        await runtime.start()
        await eventually(lambda: runtime.store.session(root.id).outcome != Outcome.ACTIVE)
        assert runtime.store.session(root.id).outcome == Outcome.COMPLETED
        assert (tmp_path / "example.py").exists()
        state_path = runtime.store.harness.path(root.id) / "harness_state.json"
        history_path = state_path.with_name("refinements.jsonl")
        assert json.loads(state_path.read_text())["entries"]["memory"]["lesson"]
        assert history_path.exists()
        assert len(runtime.store.refinement_history(root.id)) == 1
        assert provider.peak == 1
        events = runtime.store.events(root.id, limit=100)
        kinds = [e["type"] for e in events]
        assert kinds.index("refine_scheduled") < kinds.index("refine_complete")
        assert "refinement_notice" in kinds and "refinement_continuation" not in kinds
    finally:
        await runtime.shutdown()
    resumed_provider = LearningSession(resume=True)
    runtime = Runtime(directory, providers={"mock": resumed_provider})
    try:
        runtime.resume(root.id)
        await runtime.start()
        await eventually(lambda: runtime.store.session(root.id).outcome != Outcome.ACTIVE)
        assert runtime.store.session(root.id).outcome == Outcome.COMPLETED
        assert any(
            r.metadata.get("purpose") == "refinement_review" for r in resumed_provider.requests
        )
        assert not any(r.metadata.get("purpose") == "refinement" for r in resumed_provider.requests)
        notices = runtime.store.events(root.id, kind="refinement_notice")
        assert len(notices) == 1
        trace = {
            "session_id": root.id,
            "canonical_state": json.loads(state_path.read_text()),
            "local_history_storage": "session trajectory: harness_refinement",
            "refinements": runtime.store.refinement_history(root.id),
            "events": [
                {"type": e["type"], "payload": e["payload"]}
                for e in runtime.store.events(root.id, limit=1000)
                if e["type"]
                in {
                    "refine_scheduled",
                    "refine_complete",
                    "harness_refinement",
                    "resumed",
                    "context_compaction",
                    "refinement_review",
                    "completion",
                }
            ],
            "provider_calls": [
                {"turn": r.turn, "purpose": r.metadata.get("purpose", "agent")}
                for r in provider.requests + resumed_provider.requests
            ],
        }
        if os.environ.get("BUFFALO_DEMO_TRACE"):
            Path(os.environ["BUFFALO_DEMO_TRACE"]).write_text(json.dumps(trace, indent=2) + "\n")  # noqa: ASYNC240 - deterministic test artifact
    finally:
        await runtime.shutdown()
