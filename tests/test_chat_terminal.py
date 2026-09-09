"""Real pseudo-terminal tests; no fake model is reachable outside test config."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pexpect

from threadweave.coding_config import update_coding_options
from threadweave.daemon import request
from threadweave.storage import Store

PROJECT = Path(__file__).resolve().parents[1]


def spawn_chat(args, workspace, *, log=None):
    env = {
        **os.environ,
        "PROMPT_TOOLKIT_NO_CPR": "1",
        "TERM": "xterm-256color",
        "PYTHONPATH": str(PROJECT),
    }
    env.pop("OPENAI_API_KEY", None)
    child = pexpect.spawn(
        sys.executable,
        ["-m", "threadweave", *map(str, args)],
        cwd=str(workspace),
        env=env,
        encoding="utf-8",
        timeout=30,
        dimensions=(32, 110),
    )
    child.logfile_read = log
    return child


async def expect(child, text, wait_seconds=30):
    return await asyncio.to_thread(child.expect_exact, text, timeout=wait_seconds)


async def send(child, text):
    # PTY raw mode uses CR for Enter; LF is a deliberate multiline binding.
    await asyncio.to_thread(child.send, text + "\r")


async def test_terminal_multiturn_ctrl_c_detach_daemon_restart_and_continue(
    tmp_path, repository, coding_config
):
    data = tmp_path / "state"
    config = coding_config.model_copy(deep=True)
    config.provider.name = "chat_scenario"
    config.extensions = ["tests.chat_plugin:install"]
    update_coding_options(config.task, capture_baseline=False)
    config.refinement.enabled = False
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    child = spawn_chat(["--data", data, "--config", path], repository)
    try:
        await expect(child, "Type /help for commands.")
        # Default no-subcommand path, with cwd used as workspace.
        await send(child, "inspect repository")
        await expect(child, "First conversation reply")
        await expect(child, "Ready.")
        await send(child, "remember the marker")
        await expect(child, "Second conversation reply")
        await expect(child, "Ready.")
        await asyncio.to_thread(child.sendcontrol, "c")  # Prompt interrupt doesn't detach.
        await send(child, "/usage")
        await expect(child, "Model calls: 4")
        await send(child, "/exit")
        await expect(child, "Detached.")
        await asyncio.to_thread(child.expect, pexpect.EOF)
        child.close()
        assert child.exitstatus == 0
        store = Store(data)
        sid = store.sessions(roots_only=True)[0].id
        identity = store.session(sid).kernel_id
        assert len(store.events(sid, kind="user_intervention")) == 2
        store.close()
        # Hard daemon death: existing context, IDs, history and Python checkpoint recover.
        pid = (await request(data, "ping"))["pid"]
        os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(0.3)
        child = spawn_chat(["--data", data, "--continue"], repository)
        await expect(child, sid)
        await expect(child, "Resumed the same conversation")
        await send(child, "is the marker still there?")
        await expect(child, "Recovered conversation reply")
        await expect(child, "Ready.")
        await send(child, "/exit")
        await asyncio.to_thread(child.expect, pexpect.EOF)
        child.close()
        assert child.exitstatus == 0
        store = Store(data)
        try:
            assert len(store.sessions(roots_only=True)) == 1
            assert store.session(sid).kernel_id == identity
            assert len(store.events(sid, kind="user_intervention")) == 3
            assert store.events(sid, kind="kernel_recovery")[-1]["payload"]["restored"]
            assert not any(
                e["payload"].get("result", {}).get("error")
                for e in store.events(sid, kind="tool_result")
            )
        finally:
            store.close()
    finally:
        if child.isalive():
            child.close(force=True)
        try:
            await request(data, "shutdown")
        except (OSError, ConnectionError):
            pass


def test_large_event_projections_preserve_human_fields():
    from threadweave.daemon import chat_payload

    payload = {"text": "model answer", "metadata": {"huge": "x" * 10000}}
    assert chat_payload("model_response", payload)["text"] == "model answer"
    assert (
        len(
            json.dumps(
                chat_payload(
                    "tool_call",
                    {"name": "apply_patch", "action_id": "a", "arguments": {"patch": "x" * 100000}},
                )
            )
        )
        < 5000
    )
