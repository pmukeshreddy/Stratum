"""Opt-in real subscription + terminal acceptance of the Python-only control plane."""

import asyncio
import json
import os
import signal
import time
from pathlib import Path

import pexpect
import pytest

from threadweave.artifacts import Artifacts
from threadweave.codex_auth import CodexControl
from threadweave.daemon import request
from threadweave.models import RunConfig
from threadweave.storage import Store

from .test_chat_terminal import expect, send

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("THREADWEAVE_LIVE_PARITY") != "1",
    reason="opt-in real subscription Python control-plane acceptance",
)


async def test_live_programmable_session(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    async with CodexControl() as control:
        assert (await control.status())["logged_in"], "ChatGPT login required; no API-key fallback"
    output = await asyncio.to_thread(
        Path(os.environ.get("THREADWEAVE_PARITY_OUTPUT", tmp_path)).resolve
    )
    output.mkdir(parents=True, exist_ok=True)
    data = output / "state"
    assert not (data / "history.sqlite3").exists(), "Use a fresh acceptance directory"
    config = RunConfig.model_validate_json((PROJECT / "configs/session.json").read_text())
    config.provider.parameters["reasoning_effort"] = "low"
    config.limits.python_timeout_seconds = 180
    config.limits.tool_timeout_seconds = 180
    config.limits.wall_seconds = 1800
    config.features.model_compaction = False  # /compact explicitly exercises the boundary.
    config_path = output / "config.json"
    config_path.write_text(config.model_dump_json(indent=2))
    transcript = output / "terminal.raw.txt"
    root_id, cli, passed = None, None, False
    prompts, started = [], time.monotonic()
    with transcript.open("w") as log:
        transcript.chmod(0o600)

        def launch(continuation=False):
            env = {**os.environ, "TERM": "xterm-256color", "PROMPT_TOOLKIT_NO_CPR": "1"}
            env.pop("OPENAI_API_KEY", None)
            args = [
                "run",
                "threadweave",
                "--data",
                str(data),
                *(["--continue"] if continuation else ["--config", str(config_path)]),
            ]
            log.write("\r\n$ uv " + " ".join(args) + "\r\n")
            process = pexpect.spawn(
                "uv",
                args,
                cwd=str(PROJECT),
                env=env,
                encoding="utf-8",
                timeout=30,
                dimensions=(40, 120),
            )
            process.logfile_read = log
            return process

        async def say(text):
            prompts.append(text)
            print("LIVE INPUT:", text, flush=True)
            await send(cli, text)
            async with asyncio.timeout(300):
                while True:
                    try:
                        await expect(cli, "Ready.", wait_seconds=20)
                        break
                    except pexpect.TIMEOUT:
                        current = await request(data, "status", session_id=root_id)
                        assert current["session"]["outcome"] == "active", current
                        print("LIVE: still running", root_id, flush=True)

        async def detach():
            await send(cli, "/exit")
            await asyncio.to_thread(cli.expect, pexpect.EOF)
            cli.close()
            assert cli.exitstatus == 0

        try:
            cli = launch()
            await expect(cli, "Type /help for commands.")
            root_id = max(await request(data, "list"), key=lambda s: s["created_at"])["id"]
            await say("hi")
            await say("Use Python to set x = 123. Keep it in your persistent REPL.")
            await say(
                "Read x from your existing REPL without assigning it again and tell me its value."
            )
            await say(
                "Using Python, inspect this repository and count Python source/test files. Retain the list as py_files; exclude .venv, results, caches and vendored files. Report the count."
            )
            await say(
                "Search for the session lifecycle implementation using Python and retain the search result as lifecycle_evidence. Briefly report the responsible files."
            )
            await say(
                "From Python use await rlm() twice, retaining the handles as persistence_child and messaging_child. Name them persistence-review and messaging-review. Ask the first to inspect src/threadweave/storage.py persistence; ask the second to inspect src/threadweave/runtime.py messaging. Each must use its own Python REPL, retain a local observation, and explicitly send a brief evidence-backed finding with await agent_message.send(..., receiver_role='parent'). Do not edit files. Immediately after both admissions return, set parent_continued = True and parent_continued_at = time.time() (import time) and print 'PARENT_CONTINUED'. Do not wait for child completion before that local work, and do not create additional children."
            )
            async with asyncio.timeout(180):
                while True:
                    tree = await request(data, "tree", session_id=root_id)
                    if len(tree) == 3 and all(
                        s["outcome"] == "completed" for s in tree if s["parent_id"]
                    ):
                        break
                    try:
                        await expect(cli, "Ready.", wait_seconds=5)
                    except pexpect.TIMEOUT:
                        pass
            await say(
                "Run these relevant tests programmatically through await bash('uv run pytest -q tests/test_python_control.py tests/test_context.py'). Retain the command result as test_evidence and report the actual exit code and summary. Do not change source files."
            )
            await send(cli, "/compact")
            await expect(cli, "Context compacted; full history retained.")
            await say(
                "Use Python to assert existing x == 123, py_files is nonempty, lifecycle_evidence exists, parent_continued is True, test_evidence.exit_code == 0, and persistence_child.session_id differs from messaging_child.session_id. Read history.messages() and confirm both children replied. Do not redefine these variables or recreate children. Print 'PARITY_STATE_OK' only after the assertions pass."
            )
            await detach()
            assert (await request(data, "ping"))["pid"] > 0
            cli = launch(True)
            await expect(cli, "Resumed the same conversation")
            await say(
                "After reattaching, use Python to read the same x, py_files and both child handles. Assert x == 123 and the handles differ; do not redefine them. Print 'REATTACH_OK'."
            )
            await detach()
            pid = (await request(data, "ping"))["pid"]
            os.kill(pid, signal.SIGKILL)
            await asyncio.sleep(0.3)
            cli = launch(True)
            await expect(cli, "Resumed the same conversation")
            await say(
                "The daemon was restarted. Use Python to assert recovered x == 123, py_files is nonempty, persistence_child.session_id and messaging_child.session_id differ, and history.messages() contains the child replies. Do not reconstruct or redefine these variables. Print 'RECOVERY_OK'."
            )
            await detach()

            store = Store(data)
            try:
                sessions = store.sessions(root_id=root_id)
                assert len(sessions) == 3
                children = [s for s in sessions if s.parent_id]
                assert all(s.workspace.path == str(PROJECT) for s in sessions)
                assert len({s.kernel_id for s in sessions}) == 3
                events = list(store.iter_events(root_id, tree=True))
                failures = [e for e in events if e["type"] in {"python_error", "failure"}]
                assert not failures, failures
                assert not any(
                    e["type"] in {"coding_baseline", "verifier_started", "candidate_preparing"}
                    for e in events
                )
                calls = [e for e in events if e["type"] == "tool_call"]
                assert all(
                    e["payload"]["name"] == "ipython" or e["payload"].get("from_python")
                    for e in calls
                )
                assert all(
                    any(m["sender_id"] == s.id for m in store.messages(root_id)) for s in children
                )
                artifacts = Artifacts(store)
                requests = [
                    artifacts.load(e["session_id"], e["payload"]["request_artifact"])
                    for e in events
                    if e["type"] == "model_invocation_started"
                ]
                assert requests and all(
                    [t["function"]["name"] for t in r["tools"]] == ["ipython"] for r in requests
                )
                (output / "provider-requests.json").write_text(json.dumps(requests, indent=2))
                (output / "tool-schema-after.json").write_text(
                    json.dumps(requests[0]["tools"], indent=2)
                )
                for marker in ("PARITY_STATE_OK", "REATTACH_OK", "RECOVERY_OK"):
                    assert any(
                        marker in json.dumps(e["payload"])
                        for e in events
                        if e["type"] == "python_result"
                    ), marker
                parent_execution = next(
                    e
                    for e in events
                    if e["type"] == "python_result"
                    and "PARENT_CONTINUED" in json.dumps(e["payload"])
                )
                assert all(
                    parent_execution["timestamp"]
                    < next(
                        e["timestamp"]
                        for e in events
                        if e["session_id"] == s.id and e["type"] == "completion_attempt"
                    )
                    for s in children
                )
                passed = True
            finally:
                store.close()
        finally:
            if cli and cli.isalive():
                cli.close(force=True)
            if root_id:
                store = Store(data)
                try:
                    events = list(store.iter_events(root_id, tree=True))
                    (output / "events.jsonl").write_text(
                        "".join(json.dumps(e) + "\n" for e in events)
                    )
                    (output / "summary.json").write_text(
                        json.dumps(
                            {
                                "passed": passed,
                                "root_id": root_id,
                                "session_ids": [s.id for s in store.sessions(root_id=root_id)],
                                "elapsed_seconds": time.monotonic() - started,
                                "usage": store.usage(root_id, tree=True).model_dump(),
                                "config": store.config(root_id).model_dump(mode="json"),
                                "prompts": prompts,
                            },
                            indent=2,
                        )
                    )
                finally:
                    store.close()
            try:
                await request(data, "shutdown")
            except (OSError, ConnectionError):
                pass
