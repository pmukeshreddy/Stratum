"""Opt-in real terminal + ChatGPT subscription acceptance. Never uses an API key."""

import asyncio
import json
import os
from pathlib import Path

import pexpect
import pytest

from threadweave.chat import data_directory
from threadweave.codex_auth import CodexControl
from threadweave.daemon import request
from threadweave.storage import Store

from .test_chat_terminal import expect, send

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("THREADWEAVE_LIVE_CHAT") != "1",
    reason="opt-in real subscription interactive terminal test",
)


async def test_live_default_shell_multiturn_and_continue(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    async with CodexControl() as control:
        if not (await control.status())["logged_in"]:
            pytest.skip("Existing ChatGPT/Codex login required")
    directory = Path(os.environ.get("THREADWEAVE_LIVE_CHAT_OUTPUT", tmp_path))
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    data = data_directory(PROJECT)
    child = None
    sid = None
    prompts = [
        "What repository are you in? Reply concisely; do not edit files.",
        "Inspect the runtime: use repo_search to examine Runtime.interact and the interactive branch of _run_turn in src/threadweave/runtime.py. Limit this inspection to three tool calls, then explain in four sentences how messages continue the same session. Do not edit files.",
        "What did you find? Use the same conversation context; summarize in three sentences without further tools.",
        "Check the tests: use run_targeted_tests on tests/test_chat.py and tests/test_chat_terminal.py. Report the actual result; do not edit files.",
    ]
    # Keep the raw PTY stream: includes incremental toolbar redraw, terminal input,
    # tool activity and final Markdown. A separate readable transcript is derived later.
    logpath = directory / "terminal.raw.txt"
    with logpath.open("w") as log:
        os.chmod(logpath, 0o600)

        def launch(continuation=False):
            env = {**os.environ, "PROMPT_TOOLKIT_NO_CPR": "1", "TERM": "xterm-256color"}
            env.pop("OPENAI_API_KEY", None)
            args = ["run", "threadweave", *(["--continue"] if continuation else [])]
            log.write("\n$ uv " + " ".join(args) + "\n")
            process = pexpect.spawn(
                "uv",
                args,
                cwd=str(PROJECT),
                env=env,
                encoding="utf-8",
                timeout=60,
                dimensions=(40, 120),
            )
            process.logfile_read = log
            return process

        try:
            child = launch()
            first = await asyncio.to_thread(
                child.expect_exact, ["Repository has existing changes.", "Type /help for commands."]
            )
            if first == 0:
                await expect(child, "[3] Show git status")
                await send(child, "1")
                await expect(child, "Type /help for commands.")
            roots = await request(data, "list")
            sid = max(roots, key=lambda s: s["created_at"])["id"]
            for prompt in prompts:
                print("LIVE INPUT:", prompt, flush=True)
                await send(child, prompt)
                # Per-message work is bounded by the persisted runtime resource limits.
                async with asyncio.timeout(600):
                    while True:
                        try:
                            await expect(child, "Ready.", wait_seconds=30)
                            break
                        except pexpect.TIMEOUT:
                            status = await request(data, "status", session_id=sid)
                            assert status["session"]["outcome"] == "active", status
                            print("LIVE: still working in", sid, flush=True)
            await send(child, "/diff")
            # Drain the potentially large diff before sending the next acceptance
            # prompt; a PTY whose reader stops consuming output exerts backpressure.
            await expect(child, "End of diff.")
            await send(
                child,
                "Now fix the issue if that test run exposed a real failure. If the tests passed, say so and make no speculative edits. Preserve all existing changes.",
            )
            await expect(child, "Ready.", wait_seconds=180)
            await send(child, "/usage")
            await expect(child, "Model calls:")
            await send(child, "/exit")
            await asyncio.to_thread(child.expect, pexpect.EOF)
            child.close()
            assert child.exitstatus == 0
            child = launch(continuation=True)
            await expect(child, sid)
            await expect(child, "Resumed the same conversation")
            await send(
                child,
                "Which test files did I just ask you to check, and what was the result? Answer from this conversation, without tools.",
            )
            await expect(child, "Ready.", wait_seconds=180)
            await send(child, "/usage")
            await expect(child, "Model calls:")
            await send(child, "/exit")
            await asyncio.to_thread(child.expect, pexpect.EOF)
            child.close()
            assert child.exitstatus == 0
            store = Store(data)
            try:
                events = store.events(sid, after=-1, limit=500)
                assert len(store.events(sid, kind="user_intervention")) == 6
                assert any(
                    e["type"] == "tool_call" and e["payload"]["name"] == "run_targeted_tests"
                    for e in events
                )
                assert any(
                    e["type"] == "coding_command" and e["payload"].get("passed") for e in events
                )
                assert any(e["type"] == "model_stream" for e in events)
                assert store.session(sid).mode == "interactive"
                assert store.session(sid).outcome == "active"
                config = store.config(sid)
                assert config.provider.name == "codex_subscription"
                assert store.usage(sid).cost is None
                summary = {
                    "session_id": sid,
                    "data": str(data),
                    "config": config.model_dump(mode="json"),
                    "usage": store.usage(sid, tree=True).model_dump(),
                    "prompts": prompts,
                    "api_key_used": False,
                    "terminal": str(logpath),
                }
                (directory / "summary.json").write_text(json.dumps(summary, indent=2))
                (directory / "events.jsonl").write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
                )
            finally:
                store.close()
        finally:
            if child and child.isalive():
                child.close(force=True)
            if sid:
                print(
                    "Live interactive session:",
                    sid,
                    "data:",
                    data,
                    "transcript:",
                    logpath,
                    flush=True,
                )
