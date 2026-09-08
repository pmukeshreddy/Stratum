"""Protocol regressions, not capability tasks or reported benchmark results."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from threadweave.evals.arc_protocol import FixedGameControl, load_validation
from threadweave.evals.harness import run_buffalo
from threadweave.evals.inference_gate import InferenceGate
from threadweave.evals.official_worker import FixedArcGame
from threadweave.evals.schema import NotRun, accounting
from threadweave.models import HarnessError, ModelResponse, Usage

from .conftest import response
from .fakes import ScriptedProvider


def frame(state="NOT_FINISHED", levels=0):
    return SimpleNamespace(
        state=SimpleNamespace(name=state),
        levels_completed=levels,
        frame=[[[0]]],
        available_actions=[1],
    )


class Environment:
    def __init__(self):
        self.resets = self.steps = 0
        self.next = frame()

    def reset(self):
        self.resets += 1
        return frame()

    def step(self, action, data=None):
        assert action == "ACTION1"
        self.steps += 1
        return self.next


ACTION = {"name": "ACTION1", "data": {}}
ACTION_TYPE = {"ACTION1": "ACTION1"}
POLICY = {
    "instructions": "Use the supplied fixed-game client.",
    "guidance": "Retain your working state.",
    "continuation_prompt": "Continue the same task and verify remaining work.",
    "max_continuations": 3,
    "max_turns": 12,
    "max_tokens": 80000,
    "wall_seconds": 1800,
}


def test_fixed_game_counts_reset_caps_actions_and_flushes_batches():
    env = Environment()
    game = FixedArcGame(env, "fixture", ACTION_TYPE)
    assert env.resets == 1 and game.actions == 0
    env.next = frame("GAME_OVER")
    assert game.act([ACTION] * 20)["actions_taken"] == 2
    assert env.steps == 1 and env.resets == 2
    env.next = frame(levels=1)
    game.act([ACTION] * 20)
    assert env.steps == 2  # No queued action crosses a level boundary.
    while game.actions < 500:
        result = game.act([ACTION] * 20)
    assert result["terminal"] == "ACTION_CAP"
    before = env.steps
    assert game.act([ACTION]) == result
    assert env.steps == before and game.actions == 500


def test_fixed_game_win_stops_before_remaining_batch():
    env = Environment()
    env.next = frame("WIN", 1)
    game = FixedArcGame(env, "fixture", ACTION_TYPE)
    assert game.act([ACTION] * 20)["terminal"] == "WIN"
    game.act([ACTION])
    assert env.steps == game.actions == 1


class Worker:
    def __init__(self):
        self.env = Environment()
        self.game = FixedArcGame(self.env, "fixture", ACTION_TYPE)
        self.snapshots = 0

    async def call(self, operation, **kwargs):
        if operation == "snapshot_arc":
            self.snapshots += 1
            return {"scorecard_id": "retained-card", "primary_score": 0}
        assert operation == "game_query"
        return self.game.act(kwargs["actions"]) if kwargs["op"] == "act" else self.game.observe()


async def test_bootstrap_isolation_and_readonly_boundary_snapshots(tmp_path):
    gate = InferenceGate(tmp_path / "gate")
    first = FixedGameControl(Worker(), tmp_path / "first", POLICY, gate, "first")
    second = FixedGameControl(Worker(), tmp_path / "second", POLICY, gate, "second")
    try:
        await asyncio.gather(first.start(), second.start())
        for control in (first, second):
            with pytest.raises(ValueError, match="First game"):
                await control.query({"op": "act", "actions": [ACTION]})
            await control.query({"op": "observe"})
            with pytest.raises(ValueError, match="single genuine"):
                await control.query({"op": "status"})
            await control.query({"op": "act", "actions": [ACTION]})
            await control.query({"op": "act", "actions": [ACTION]})
        await first.query({"op": "act", "actions": [ACTION]})
        assert first.worker.game.actions == 3 and second.worker.game.actions == 2
        first.identity["session_id"] = "retained-session"
        first.primary_usage_reader = lambda: accounting(
            [
                {
                    "model_calls": 2,
                    "input_tokens": 90000,
                    "cached_input_tokens": 89000,
                    "output_tokens": 100,
                }
            ],
            0,
        )
        for _ in range(3):
            assert await first.boundary()
        assert not await first.boundary()
        assert first.stop_reason == "continuation_limit"
        snapshots = [json.loads(p.read_text()) for p in (first.directory / "snapshots").glob("*")]
        boundaries = [s for s in snapshots if s["kind"] == "agent_boundary"]
        assert {s["agent_identity"]["session_id"] for s in boundaries} == {"retained-session"}
        assert {s["game_state"]["actions_taken"] for s in boundaries} == {3}
        assert first.worker.env.resets == second.worker.env.resets == 1
        assert len({s["scorecard_id"] for s in snapshots}) == 1
    finally:
        await first.close()
        await second.close()


async def test_terminal_game_rejects_queued_and_future_inference(tmp_path):
    gate = InferenceGate(tmp_path / "gate", 1)
    entered = asyncio.Event()

    async def queued():
        entered.set()
        async with gate.permit("ended"):
            pytest.fail("A terminal game must never acquire inference")

    async with gate.permit("other"):
        task = asyncio.create_task(queued())
        await entered.wait()
        await gate.stop_admission("ended")
        with pytest.raises(HarnessError, match="Game already ended"):
            await task
    with pytest.raises(HarnessError, match="Game already ended"):
        async with gate.permit("ended"):
            pass
    assert gate.peak == 1 and gate.active == 0


def test_fixed_policy_preserves_semantics_for_subsets_and_all_games(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(POLICY))
    for limit in (1, 2, 25, None):
        assert load_validation(path, limit, 0) == POLICY
    for limit, seed in ((0, 0), (26, 0), (2, 1)):
        with pytest.raises(NotRun):
            load_validation(path, limit, seed)
    path.write_text("{}")
    with pytest.raises(NotRun, match="instructions"):
        load_validation(path, 2, 0)


async def test_buffalo_continuation_retains_live_python_and_session(
    tmp_path, python_config, monkeypatch
):
    import threadweave.evals.harness as harness

    python_config.refinement.automatic = False
    provider = ScriptedProvider(
        {
            "root": [
                response("ipython", code="retained_value = 937\nprint(retained_value)"),
                ModelResponse(
                    text="I think I'm done.", usage=Usage(input_tokens=20, output_tokens=10)
                ),
                response(
                    "ipython", code="assert retained_value == 937\nprint('PERSISTENT_VALUE_OK')"
                ),
                ModelResponse(text="Finished."),
            ]
        }
    )
    monkeypatch.setattr(harness, "default_providers", lambda: {"mock": provider})
    control = FixedGameControl(
        Worker(),
        tmp_path / "game",
        {**POLICY, "max_continuations": 1},
        InferenceGate(tmp_path / "gate"),
        "buffalo",
    )
    try:
        task = await control.start()
        result = await run_buffalo(python_config, task, control.directory, controller=control)
        assert result["continuations"] == 1 and result["usage"]["model_calls"] == 4
        assert len({r.session_id for r in provider.requests}) == 1
        assert POLICY["continuation_prompt"] in json.dumps(provider.requests[2].messages)
        assert "PERSISTENT_VALUE_OK" in json.dumps(provider.requests[3].messages)
        boundaries = [
            json.loads(p.read_text()) for p in (control.directory / "snapshots").glob("*")
        ]
        pids = {
            s["agent_identity"]["python_pid"] for s in boundaries if s["kind"] == "agent_boundary"
        }
        assert len(pids) == 1 and None not in pids
        assert control.worker.env.resets == 1
    finally:
        await control.close()
