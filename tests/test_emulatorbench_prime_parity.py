"""Execute the actual pinned Prime controller, then compare Buffalo decisions."""

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from threadweave.autonomous import AutonomousCycle, AutonomousPolicy, GateResult, git_snapshot
from threadweave.evals.emulatorbench import pins
from threadweave.evals.schema import file_digest
from threadweave.models import ModelResponse, Usage

from .test_evocode_integration import initialize_git


async def test_prime_captures_timeout_before_gate(monkeypatch):
    import threadweave.autonomous as autonomous

    clock = [9.0]
    monkeypatch.setattr(
        autonomous, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000)
    )
    cycle = AutonomousCycle(AutonomousPolicy(timeout_seconds=10), ["gate"], started=0)

    async def snapshot():
        return None

    async def gate(*args):
        clock[0] = 11.0
        return GateResult(False, "failed")

    assert await cycle.next_message(None, snapshot=snapshot, run_gate=gate)
    assert cycle.continuations == 1
    assert await cycle.next_message(None, snapshot=snapshot, run_gate=gate) is None
    assert cycle.stop_reason == "timeoutMs"


async def test_actual_prime_default_boundary_usage_retries_and_slow_gate(tmp_path):
    prime = await asyncio.to_thread(
        Path(os.environ.get("BUFFALO_PRIME_CHECKOUT", "../prime-agent-main")).resolve
    )
    source = prime / "packages/coding-agent/src/core/autonomous.ts"
    bun = shutil.which("bun")
    if not source.is_file() or not bun:
        pytest.skip("requires the pinned Prime source checkout and Bun for differential validation")
    assert file_digest(source) == pins()["prime"]["files"][str(source.relative_to(prime))]
    workspace = tmp_path / "workspace"
    initialize_git(workspace)
    script = tmp_path / "prime-parity.ts"
    script.write_text(
        "const p = await import("
        + json.dumps(str(source))
        + ");\n"
        + "const cwd = "
        + json.dumps(str(workspace))
        + ";\n"
        + """
const results = {};
for (const turns of [11, 12, 13]) {
  const state = p.createAutonomousRuntimeState({enabled:true,gates:{commands:['false']}});
  for(let n=0;n<turns;n++) p.addAutonomousUsage(state,{input:5,output:2,cacheRead:99,cacheWrite:3});
  const message = await p.nextAutonomousContinuation(state,{stopReason:'stop'},{cwd});
  results[turns]={feedback:!!message,turns:state.turnsUsed,tokens:state.tokensUsed,continuations:state.continuationsUsed};
}
const retry = p.createAutonomousRuntimeState({enabled:true,gates:{commands:['false']}});
results.retries=[];
for(let n=0;n<4;n++) {
  const message=await p.nextAutonomousContinuation(retry,{stopReason:'stop'},{cwd});
  results.retries.push({feedback:!!message,attempt:retry.gateAttempts.false,continuations:retry.continuationsUsed,unchanged:retry.lastGateFailure.exitText.startsWith('not rerun')});
}
const passed = p.createAutonomousRuntimeState({enabled:true,gates:{commands:['true']}});
passed.turnsUsed=13;
results.passBeyondLimit=!(await p.nextAutonomousContinuation(passed,{stopReason:'stop'},{cwd})) && passed.gateAttempts.true===0;
const slow=p.createAutonomousRuntimeState({enabled:true,timeoutMs:10,gates:{commands:['sleep 0.05; false']}});
const now=Date.now();slow.startedAt=now;
results.slowGateFeedback=!!(await p.nextAutonomousContinuation(slow,{stopReason:'stop'},{cwd},now));
console.log(JSON.stringify(results));
"""
    )
    env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(tmp_path / "prime-state"),
        "PRIME_AGENT_CODING_AGENT_DIR": str(tmp_path / "prime-state"),
    }
    process = await asyncio.to_thread(
        subprocess.run, [bun, str(script)], capture_output=True, text=True, env=env, timeout=30
    )
    assert process.returncode == 0, process.stderr
    reference = json.loads(process.stdout)

    async def fail(*args):
        return GateResult(False, "exited 1")

    for turns in (11, 12, 13):
        cycle = AutonomousCycle(AutonomousPolicy(), ["false"])
        for _ in range(turns):
            cycle.record_response(
                ModelResponse(
                    usage=Usage(input_tokens=107, cached_input_tokens=99, output_tokens=2)
                )
            )
        message = await cycle.next_message(
            None, snapshot=lambda: git_snapshot(workspace), run_gate=fail
        )
        assert {
            "feedback": bool(message),
            "turns": cycle.turns,
            "tokens": cycle.tokens,
            "continuations": cycle.continuations,
        } == reference[str(turns)]
    cycle = AutonomousCycle(AutonomousPolicy(), ["false"])
    for expected in reference["retries"]:
        message = await cycle.next_message(
            None, snapshot=lambda: git_snapshot(workspace), run_gate=fail
        )
        assert {
            "feedback": bool(message),
            "attempt": cycle.attempts["false"],
            "continuations": cycle.continuations,
            "unchanged": not cycle.checks[-1]["rerun"],
        } == expected
    assert cycle.stop_reason == "retry_exhausted"
    assert reference["passBeyondLimit"] and reference["slowGateFeedback"]
    (tmp_path / "prime-reference-results.json").write_text(json.dumps(reference, indent=2))
