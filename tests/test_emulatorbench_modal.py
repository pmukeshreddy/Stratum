"""Opt-in real Modal/public-verifier rehearsal. Scripted provider; no paid inference."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from threadweave.evals.emulatorbench import discover_tasks, evaluate_task
from threadweave.evals.emulatorbench_report import read_jsonl
from threadweave.models import ModelResponse

from .test_evocode_integration import cell

pytestmark = pytest.mark.skipif(
    os.environ.get("BUFFALO_EMULATORBENCH_MODAL_TEST") != "1",
    reason="requires the authenticated Modal profile and official public corpus cache",
)


async def test_real_modal_public_rehearsal_matches_final_grading(tmp_path, monkeypatch):
    config = json.loads(
        await asyncio.to_thread(Path("configs/emulatorbench-public.json").read_text)
    )
    config["run"]["provider"] = {
        "name": "scripted",
        "model": "fixture",
        "model_metadata": {"maxTokens": 8192},
    }
    config["autonomous"]["max_turns"] = 1
    config["public_rehearsal"] = {"enabled": True, "max_attempts": 2}
    cache = await asyncio.to_thread(Path(config["public_source_cache"]).resolve)
    monkeypatch.setenv("EMULATORBENCH_PUBLIC_SOURCE_CACHE", str(cache))
    hidden = tmp_path / "secret-reference-answer.txt"
    hidden.write_text("host-only test marker")

    class Scripted:
        turn = 0

        async def invoke(self, request, emit):
            self.turn += 1
            if self.turn == 1:
                return cell(
                    "persistent_marker = 937\nfrom pathlib import Path\n"
                    f"assert not Path({str(hidden)!r}).exists()\n"
                    "assert not Path('/tmp/emulatorbench_final_verification').exists()\n"
                    "assert not Path('/tmp/emulatorbench_suite_adapters.py').exists()\n"
                    "r = await bash('cargo generate-lockfile --offline')\nassert r.exit_code == 0"
                )
            if self.turn in {2, 3}:
                return cell(
                    "assert persistent_marker == 937\n"
                    "r = await bash('./verify-public.sh', timeout=360)\nprint(r.stdout)\n"
                    "assert r.exit_code == 1, r.stderr"
                )
            assert "public_source_score" in str(request.messages)
            assert "workspace has not changed" in str(request.messages)
            return ModelResponse(text="Public rehearsal completed; starter remains failing.")

    task = discover_tasks()[0]
    output = tmp_path / "official-modal-rehearsal"
    result = await evaluate_task(task, output, config, providers={"scripted": Scripted()})
    assert result["status"] == "BENCHMARK_RESULT", result
    assert result["stop_reason"] == "maxTurns"
    assert result["autonomous"]["continuations"] == 0
    assert result["public_rehearsal_attempts"] == 1 and result["verifier_attempts"] == 1
    rehearsals = read_jsonl(output / "rehearsal-attempts.jsonl")
    final = read_jsonl(output / "verifier-attempts.jsonl")
    assert rehearsals[0]["public_source_score"] == final[0]["public_source_score"]
    assert rehearsals[0]["submission_sha256"] == final[0]["submission_sha256"]
    assert len(read_jsonl(output / "rehearsal-suppressed.jsonl")) == 1
    for attempt in rehearsals + final:
        info = attempt["official_info"]
        assert info["emulatorbench_score"]["case_runs"]
        assert info["emulatorbench_score"]["build"]["cargo_build"]["ok"]
        assert info["emulatorbench_grader_lifecycle"]["stopped"]
        assert info["emulatorbench_grader_lifecycle"]["egress_probe"] == "blocked"
    packet = json.loads((output / "trajectory.json").read_text())
    assert not any(e["type"] == "python_error" for e in packet["events"])
    assert packet["before"]["root_session_id"] == packet["after"]["root_session_id"]
    assert packet["before"]["kernel_id"] == packet["after"]["kernel_id"]
    assert json.loads((output / "manifest.json").read_text())["candidate_teardown_confirmed"]
