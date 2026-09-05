"""Opt-in real subscription inference. No API keys or test provider are used."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from threadweave.codex_auth import CodexControl
from threadweave.models import RunConfig, new_id
from threadweave.runtime import Runtime

REPOSITORY = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("THREADWEAVE_LIVE_CODEX") != "1",
    reason="opt-in real ChatGPT/Codex subscription test",
)


async def test_live_structured_tools_recovery_and_independent_children(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    async with CodexControl() as control:
        if not (await control.status())["logged_in"]:
            pytest.skip("No existing ChatGPT/Codex login")
    directory = Path(os.environ.get("THREADWEAVE_LIVE_DIRECTORY", tmp_path)) / new_id()
    directory.mkdir(parents=True)
    workspace = directory.resolve() / "workspace"
    await asyncio.to_thread(
        subprocess.run,
        [
            "git",
            "clone",
            "--quiet",
            "--local",
            str(REPOSITORY),
            str(workspace),
        ],
        check=True,
    )
    config = RunConfig(
        provider={
            "name": "codex_subscription",
            "parameters": {"reasoning_effort": "low"},
            "max_output_tokens": 4096,
        },
        context={"max_tokens": 96000},
        task={
            "adapter": "coding",
            "require_change": False,
            "capture_baseline": False,
            "test_commands": [[sys.executable, "-m", "pytest", "-q", "tests/test_context.py"]],
        },
        permissions=["workspace.read", "workspace.write", "python", "process", "agents", "state"],
        tool_allowlist=["repo_search", "python", "finish", "session_inspect"],
        limits={"token_budget": 500000, "wall_seconds": 900, "max_turns": 20, "concurrency": 3},
        refinement={"enabled": False},
    )
    runtime = Runtime(directory / "state")
    root = runtime.create(
        "First call repo_search with pattern 'class ProviderConfig' in src/threadweave/models.py. Do not finish until instructed by a user message. Do not edit files.",
        workspace,
        config=config,
        mode="heartbeat",
    )
    root_id, kernel_id = root.id, root.kernel_id
    try:
        await runtime._run_turn(root_id)
        assert any(
            event["payload"]["name"] == "repo_search"
            for event in runtime.store.events(root_id, kind="tool_call")
        )
        results = runtime.store.events(root_id, kind="tool_result")
        assert results and all(not event["payload"].get("error") for event in results)
        runtime.message(
            None,
            root_id,
            "Now use python to execute subscription_marker = 314159 and print(subscription_marker). Do not finish yet.",
        )
        await runtime._run_turn(root_id)
        assert runtime.store.usage(root_id).python_executions >= 1
        assert runtime.store.usage(root_id).model_calls >= 2
        await runtime.shutdown()

        runtime = Runtime(directory / "state")
        await runtime.recover()
        assert runtime.store.session(root_id).kernel_id == kernel_id
        assert runtime.store.events(root_id, kind="recovery")
        children = [
            runtime.spawn(
                root_id,
                f"Use python to set child_marker = '{name}' and print(child_marker). After receiving the Python result, call finish reporting that marker. Do not edit files.",
                name=name,
            )
            for name in ("child-A", "child-B")
        ]
        assert len({root_id, *(child.id for child in children)}) == 3
        assert len({kernel_id, *(child.kernel_id for child in children)}) == 3
        for _ in range(4):
            active = [
                child for child in children if runtime.store.session(child.id).outcome == "active"
            ]
            if not active:
                break
            await asyncio.gather(*(runtime._run_turn(child.id) for child in active))
        for child in children:
            assert runtime.store.session(child.id).outcome == "completed"
            assert runtime.store.usage(child.id).python_executions >= 1
        runtime.message(
            None,
            root_id,
            "The runtime has restarted. Use python to assert subscription_marker == 314159 and print(subscription_marker + 1). After receiving the successful result, call finish with the recovered value and the search finding.",
        )
        for _ in range(4):
            if runtime.store.session(root_id).outcome != "active":
                break
            await runtime._run_turn(root_id)
        assert runtime.store.session(root_id).outcome == "completed"
        assert runtime.store.usage(root_id).python_executions >= 2
        assert runtime.store.usage(root_id, tree=True).cost is None
        # Concurrent model intervals must overlap, not merely use async syntax.
        intervals = []
        for child in children:
            starts = runtime.store.events(child.id, kind="model_invocation_started")
            ends = runtime.store.events(child.id, kind="model_response")
            intervals.append((starts[0]["timestamp"], ends[0]["timestamp"]))
        assert max(i[0] for i in intervals) < min(i[1] for i in intervals)
    finally:
        summary = {
            "session_id": root_id,
            "config": config.model_dump(mode="json"),
            "sessions": [s.model_dump(mode="json") for s in runtime.store.sessions()],
            "usage": runtime.store.usage(root_id, tree=True).model_dump(),
            "directory": str(directory),
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        await runtime.shutdown()
        print(f"Live trajectory: {directory}")
