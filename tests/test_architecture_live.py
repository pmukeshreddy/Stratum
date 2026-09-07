"""Opt-in legacy direct-tool acceptance over a real terminal and subscription provider.

THREADWEAVE_LIVE_ARCHITECTURE=1 uv run pytest -s tests/test_architecture_live.py
No test provider, external coding agent, API key, or implicit coding preparation.
"""

import asyncio
import json
import os
import signal
from pathlib import Path

import pexpect
import pytest

from threadweave.codex_auth import CodexControl
from threadweave.daemon import request
from threadweave.models import RunConfig
from threadweave.storage import Store

from .test_chat_terminal import expect, send

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("THREADWEAVE_LIVE_ARCHITECTURE") != "1",
    reason="opt-in real subscription architecture acceptance",
)


async def test_live_agents_view_recursive_sessions_compaction_and_recovery(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    async with CodexControl() as auth:
        assert (await auth.status())["logged_in"], "Existing ChatGPT/Codex login required"
    output = await asyncio.to_thread(
        Path(os.environ.get("THREADWEAVE_ARCHITECTURE_OUTPUT", tmp_path)).resolve
    )
    output.mkdir(parents=True, exist_ok=True)
    data, workspace = output / "state", output / "workspace"
    workspace.mkdir(exist_ok=True)
    assert not (workspace / ".git").exists()
    (workspace / "environment.txt").write_text("Shared environment observation: blue cedar.\n")
    config = RunConfig.model_validate_json((PROJECT / "configs/session.json").read_text())
    config.control_plane = "direct"  # Default Python-only acceptance: test_python_control_live.py.
    config.provider.parameters["reasoning_effort"] = "low"
    config.limits.wall_seconds = 1800
    config.tool_allowlist = [
        "python",
        "rlm",
        "agent_message",
        "agent_sessions",
        "agent_receive",
        "agent_wait",
        "workspace_read",
        "workspace_write",
        "workspace_list",
        "finish",
        "refine",
        "state_list",
        "state_read",
        "state_select",
        "skill_run",
        "history_read",
        "information_inspect",
        "schedule_turn",
        "session_inspect",
    ]
    path = output / "config.json"
    path.write_text(config.model_dump_json(indent=2))
    transcript = output / "terminal.raw.txt"
    cli, sid = None, None
    prompts = []
    identities = None
    with transcript.open("w") as log:
        os.chmod(transcript, 0o600)

        def launch(continuation=False):
            env = {**os.environ, "TERM": "xterm-256color", "PROMPT_TOOLKIT_NO_CPR": "1"}
            env.pop("OPENAI_API_KEY", None)
            args = [
                "run",
                "--project",
                str(PROJECT),
                "threadweave",
                "--data",
                str(data),
                *(["--continue"] if continuation else ["--config", str(path)]),
            ]
            log.write("\r\n$ uv " + " ".join(args) + "\r\n")
            child = pexpect.spawn(
                "uv",
                args,
                cwd=str(workspace),
                env=env,
                encoding="utf-8",
                timeout=30,
                dimensions=(40, 120),
            )
            child.logfile_read = log
            return child

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
                        status = await request(data, "status", session_id=sid)
                        assert status["session"]["outcome"] == "active", status
                        print("LIVE: running in", sid, flush=True)

        async def all_idle():
            async with asyncio.timeout(180):
                while True:
                    tree = await request(data, "tree", session_id=sid)
                    if all(s["lifecycle"] != "RUNNING" and not s["runnable"] for s in tree):
                        return tree
                    # Drain actual terminal output while children communicate.
                    try:
                        await expect(cli, "Ready.", wait_seconds=5)
                    except pexpect.TIMEOUT:
                        pass

        async def exit_view():
            await send(cli, "/exit")
            await asyncio.to_thread(cli.expect, pexpect.EOF)
            cli.close()
            assert cli.exitstatus == 0

        try:
            cli = launch()
            await expect(cli, "Type /help for commands.")
            sid = max(await request(data, "list"), key=lambda s: s["created_at"])["id"]
            await say("hi")
            assert "Preparing repository" not in transcript.read_text()
            assert "Opening environment" in transcript.read_text()
            store = Store(data)
            assert store.usage(sid).model_calls == 1
            assert not store.events(sid, kind="coding_baseline")
            assert not store.events(sid, kind="coding_command")
            store.close()
            await say(
                "Use Python to create x = 123 in your persistent REPL and remember it. Reply briefly after executing."
            )
            await say(
                "Read x from your existing Python REPL, without assigning it again, and report it."
            )
            await say(
                "Use your Python rlm() helper twice to create persistent children named child-a and child-b; "
                "retain their handles as child_a and child_b. Give each this instruction: use your own Python "
                "REPL to set child_value to your name and assert x is not in globals(); read environment.txt "
                "using tools.call('workspace_read', path='environment.txt'); write the observed text to "
                "your-name.txt using workspace_write; message the parent your observed text and own child_value "
                "using agent_message; then finish. After both rlm calls return, immediately set "
                "root_continued = True in your REPL and write root-continued.txt with 'continued' using "
                "workspace_write. Do not wait for child answers before doing this local work. "
                "Then respond briefly. Do not create any additional children."
            )
            tree = await all_idle()
            children = [s for s in tree if s["parent_id"]]
            assert len(children) == 2 and all(s["outcome"] == "completed" for s in children), tree
            assert (workspace / "root-continued.txt").read_text() == "continued"
            assert all(
                (workspace / f"{s['name']}.txt").read_text().strip().endswith("blue cedar.")
                for s in children
            )
            store = Store(data)
            identities = {s.id: s.kernel_id for s in store.sessions(root_id=sid)}
            assert len(set(identities.values())) == 3
            assert all(store.events(s["id"], kind="python_result") for s in children)
            assert all(
                any(m["sender_id"] == s["id"] and m["received_at"] for m in store.messages(sid))
                for s in children
            )
            source = store.events(sid, kind="python_result", limit=1)[0]["id"]
            store.close()
            await send(cli, "/tree")
            await expect(cli, "child-b")
            await send(cli, "/compact")
            await expect(cli, "Context compacted; full history retained.")
            await say(
                "Read existing x in Python after compaction without assigning it. Also inspect your existing child_a and child_b handles; do not recreate children."
            )
            await say(
                "Store a reusable executable skill via refine: title 'Inspect saved scalar', kind skill, "
                "content with name inspect_saved_x, description 'Print the existing scalar without dumping working state', "
                "code 'print(x)', required_permissions ['python']; source_events ['"
                + source
                + "']; "
                "intended_effect 'Reuse explicit bounded selection from persistent REPL'. This is version 1."
            )
            store = Store(data)
            skills = [e for e in store.states(sid) if e["kind"] == "skill"]
            assert len(skills) == 1
            entry = skills[0]
            store.close()
            await say(
                f"Update skill entry {entry['id']} using refine with expected_version {entry['version']}, "
                "kind skill, same name/description/permissions, code 'assert x == 123\\nprint(x)', "
                f"source_events ['{source}'], intended_effect 'Check the scalar before printing'. "
                "Use an actual newline between Python statements. Then run that skill with skill_run."
            )
            await all_idle()
            old_pid = (await request(data, "ping"))["pid"]
            await exit_view()
            assert (await request(data, "ping"))["pid"] == old_pid
            cli = launch(continuation=True)
            await expect(cli, sid)
            await expect(cli, "Resumed the same conversation")
            await say(
                "Read existing x and root_continued from Python after reattaching, without assigning either. Reply briefly."
            )
            await exit_view()
            os.kill(old_pid, signal.SIGKILL)
            await asyncio.sleep(0.3)
            cli = launch(continuation=True)
            await expect(cli, sid)
            await expect(cli, "Resumed the same conversation")
            await say(
                "The daemon was restarted. Read existing x, root_continued, child_a and child_b in Python without redefining them. Inspect session identities with agent_sessions. Do not spawn replacements."
            )
            await send(cli, "/state")
            await expect(cli, "L3")
            await send(cli, "/usage")
            await expect(cli, "Model calls:")
            await exit_view()
            store = Store(data)
            try:
                assert {s.id: s.kernel_id for s in store.sessions(root_id=sid)} == identities
                assert len(store.events(sid, kind="user_intervention")) == len(prompts)
                assert store.session(sid).outcome == "active"
                assert store.states(sid)[0]["version"] >= 2
                assert store.events(sid, kind="skill_outcome")[-1]["payload"]["passed"]
                assert "x" in store.events(sid, kind="kernel_recovery")[-1]["payload"]["restored"]
                assert not store.events(sid, kind="python_error", tree=True)
                assert not store.events(sid, kind="coding_baseline", tree=True)
                events, after = [], -1
                while page := store.events(sid, tree=True, after=after, limit=500):
                    events.extend(page)
                    after = page[-1]["seq"]
                (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
                (output / "summary.json").write_text(
                    json.dumps(
                        {
                            "session_id": sid,
                            "sessions": identities,
                            "data": str(data),
                            "workspace": str(workspace),
                            "config": store.config(sid).model_dump(mode="json"),
                            "usage": store.usage(sid, tree=True).model_dump(),
                            "prompts": prompts,
                            "api_key_used": False,
                            "passed": True,
                            "terminal": str(transcript),
                        },
                        indent=2,
                    )
                )
            finally:
                store.close()
        finally:
            if cli and cli.isalive():
                cli.close(force=True)
            print("LIVE ARTIFACTS:", output, "SESSION:", sid, flush=True)
            # Keep the daemon and persistent state available for human reattachment.
