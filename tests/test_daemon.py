import asyncio
import json
import signal
import sys
from pathlib import Path

import pytest

from threadweave.daemon import Daemon, request, socket_path
from threadweave.models import RunConfig
from threadweave.storage import Store

PROJECT = Path(__file__).resolve().parents[1]


async def start_daemon(data, log_path):
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "threadweave.daemon",
            "--data",
            str(data),
            "--idle-seconds",
            "60",
            cwd=PROJECT,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    async with asyncio.timeout(10):
        while True:
            try:
                result = await request(data, "ping")
                assert result["pid"] == process.pid
                return process
            except (FileNotFoundError, ConnectionRefusedError, ConnectionError):
                if process.returncode is not None:
                    pytest.fail(log_path.read_text()[-6000:])
                await asyncio.sleep(0.03)


async def stop_daemon(process, data):
    if process.returncode is None:
        try:
            await request(data, "shutdown")
            async with asyncio.timeout(10):
                await process.wait()
        except (ConnectionError, OSError, TimeoutError):
            process.kill()
            await process.wait()


async def until_status(data, sid, predicate, seconds=15):
    async with asyncio.timeout(seconds):
        while True:
            status = await request(data, "status", session_id=sid)
            if predicate(status):
                return status
            await asyncio.sleep(0.03)


async def test_real_daemon_kill_restart_recovers_full_recursive_trajectory(tmp_path):
    data, log = tmp_path / "data", tmp_path / "daemon.log"
    config = RunConfig(
        control_plane="direct",
        provider={"name": "recovery_scenario", "model": "deterministic", "max_output_tokens": 128},
        context={"max_tokens": 26000, "compact_at": 0.45, "summary_chars": 600, "recent_blocks": 2},
        extensions=["tests.scenario_plugin:install"],
        task={
            "verifier": "file",
            "verifier_options": {"path": "answer.txt", "equals": "499500"},
            "verify_each_turn": False,
            "require_verifier": True,
        },
        limits={"max_turns": 100, "wall_seconds": 60},
    )
    process = await start_daemon(data, log)
    try:
        root = await request(
            data,
            "create",
            instruction="Recover the full lifecycle",
            workspace=str(tmp_path),
            name="root",
            mode="goal",
            config=config.model_dump(mode="json"),
        )
        sid = root["id"]
        await until_status(data, sid, lambda s: s["session"]["turns"] >= 3)
        async with asyncio.timeout(15):
            while True:
                tree = await request(data, "tree", session_id=sid)
                if len(tree) == 4 and all(s["turns"] >= 2 for s in tree):
                    break
                await asyncio.sleep(0.03)
        identities = {s["id"]: (s["parent_id"], s["kernel_id"]) for s in tree}
        before = Store(data)
        event_ids = [r[0] for r in before.db.execute("SELECT id FROM events")]
        assert before.events(sid, kind="context_compaction")
        assert before.states(sid)[0]["kind"] == "memory"
        assert before.db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] >= 1
        before.close()
        queued = await request(
            data, "input", session_id=sid, body="Durable input queued before process death"
        )
        process.send_signal(signal.SIGKILL)
        await process.wait()
        (tmp_path / "continue").write_text("continue")
        process = await start_daemon(data, log)
        recovered_tree = await request(data, "tree", session_id=sid)
        assert {s["id"]: (s["parent_id"], s["kernel_id"]) for s in recovered_tree} == identities
        for child in recovered_tree:
            if child["id"] != sid:
                await request(
                    data, "input", session_id=child["id"], body="Resume your retained computation"
                )
        result = await until_status(data, sid, lambda s: s["session"]["outcome"] != "active")
        assert result["session"]["outcome"] == "completed", result
        assert (tmp_path / "answer.txt").read_text() == "499500"
        assert result["goal"]["status"] == "completed"
        assert result["tree_usage"]["subagent_count"] == 3
        assert result["tree_usage"]["estimated_calls"] >= 1
        persisted = Store(data)
        after_ids = {r[0] for r in persisted.db.execute("SELECT id FROM events")}
        assert set(event_ids) <= after_ids
        assert any(
            m["id"] == queued["message_id"] and m["received_at"] for m in persisted.messages(sid)
        )
        assert persisted.events(sid, kind="recovery")
        assert persisted.events(sid, kind="verifier_result")[-1]["payload"]["passed"]
        assert persisted.states(sid)[0]["version"] == 1
        for member in persisted.sessions(root_id=sid):
            assert member.outcome == "completed"
            assert persisted.events(member.id, kind="kernel_recovery")
        persisted.close()
    finally:
        await stop_daemon(process, data)


async def test_cli_detach_does_not_cancel_and_controls_are_usable(tmp_path):
    data, log = tmp_path / "data", tmp_path / "daemon.log"
    process = await start_daemon(data, log)
    try:
        config = RunConfig(
            control_plane="direct",
            provider={
                "name": "recovery_scenario",
                "model": "deterministic",
                "max_output_tokens": 128,
            },
            extensions=["tests.scenario_plugin:install"],
            limits={"wall_seconds": 60},
        )
        root = await request(
            data,
            "create",
            instruction="CLI lifecycle",
            workspace=str(tmp_path),
            name="root",
            config=config.model_dump(mode="json"),
        )
        sid = root["id"]
        await until_status(data, sid, lambda s: s["session"]["turns"] >= 3)
        client = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "threadweave",
            "--data",
            str(data),
            "attach",
            sid,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT,
        )
        # Synchronize on attachment, not interpreter/import startup latency.
        first_event = await asyncio.wait_for(client.stdout.readline(), 10)
        assert json.loads(first_event)["session_id"] == sid
        client.send_signal(signal.SIGINT)
        _, stderr = await client.communicate()
        assert client.returncode == 0 and b"Detached" in stderr
        assert (await request(data, "ping"))["pid"] == process.pid
        assert (await request(data, "status", session_id=sid))["session"]["outcome"] == "active"
        for command in ("list", "status", "tree", "history", "usage", "config", "states"):
            args = [] if command == "list" else [sid]
            cli = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "threadweave",
                "--data",
                str(data),
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=PROJECT,
            )
            output, error = await cli.communicate()
            assert cli.returncode == 0, error
            assert json.loads(output)
        await request(data, "stop", session_id=sid)
        stopped = await request(data, "tree", session_id=sid)
        assert all(s["outcome"] == "cancelled" for s in stopped)
        assert all(s["lifecycle"] == "INACTIVE" for s in stopped)
        with pytest.raises(RuntimeError, match="Unknown session"):
            await request(data, "resume", session_id="unknown-id")
    finally:
        await stop_daemon(process, data)


async def test_daemon_single_owner_and_private_socket(tmp_path):
    data = tmp_path / "data"
    process = await start_daemon(data, tmp_path / "log")
    try:
        assert socket_path(data).stat().st_mode & 0o777 == 0o600
        with pytest.raises(RuntimeError, match="already owns"):
            Daemon(data)
    finally:
        await stop_daemon(process, data)


@pytest.mark.parametrize("scenario", ["interrupted_python", "interrupted_command"])
async def test_killed_daemon_does_not_replay_uncertain_tool_side_effect(tmp_path, scenario):
    data, log = tmp_path / "data", tmp_path / "log"
    config = RunConfig(
        control_plane="direct",
        provider={
            "name": "recovery_scenario",
            "max_output_tokens": 128,
            "parameters": {"scenario": scenario},
            "model": "deterministic",
        },
        extensions=["tests.scenario_plugin:install"],
        permissions=["workspace.read", "workspace.write", "python", "process"],
        limits={"wall_seconds": 60, "python_timeout_seconds": 90, "tool_timeout_seconds": 90},
    )
    process = await start_daemon(data, log)
    try:
        root = await request(
            data,
            "create",
            instruction="Inspect interrupted effects",
            workspace=str(tmp_path),
            config=config.model_dump(mode="json"),
        )
        sid = root["id"]
        async with asyncio.timeout(10):
            while not (tmp_path / "worker.pid").exists():  # noqa: ASYNC110 - cross-process file signal
                await asyncio.sleep(0.02)
        assert (tmp_path / "effect.txt").read_text() == "x"
        process.kill()
        await process.wait()
        process = await start_daemon(data, log)
        result = await until_status(data, sid, lambda s: s["session"]["outcome"] != "active")
        assert result["session"]["outcome"] == "completed", result
        assert (tmp_path / "effect.txt").read_text() == "x"
        persisted = Store(data)
        recovered = [
            e for e in persisted.events(sid, kind="tool_result") if e["payload"]["recovered"]
        ]
        assert recovered[0]["payload"]["result"]["error"]["uncertain"]
        requests = persisted.events(sid, kind="model_invocation_started")
        assert len(requests) == 3
        following = requests[1]["payload"]["request_artifact"]
        from threadweave.artifacts import Artifacts

        assert "interrupted_action" in str(Artifacts(persisted).load(sid, following)["messages"])
        persisted.close()
    finally:
        await stop_daemon(process, data)
