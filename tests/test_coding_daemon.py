import asyncio
import signal

from threadweave.daemon import request
from threadweave.storage import Store

from .test_daemon import start_daemon, stop_daemon, until_status


async def test_hard_kill_and_recover_real_repository_child_patch_history_and_repl(
    tmp_path, repository, coding_config
):
    data, log = tmp_path / "daemon", tmp_path / "daemon.log"
    coding_config.provider.name = "coding_recovery"
    coding_config.extensions = ["tests.coding_plugin:install"]
    coding_config.context.compact_at = 0.2
    coding_config.context.recent_blocks = 2
    coding_config.context.summary_chars = 512
    process = await start_daemon(data, log)
    try:
        root = await request(
            data,
            "create",
            instruction="Fix arithmetic and verify candidate documentation",
            workspace=str(repository),
            config=coding_config.model_dump(mode="json"),
        )
        sid = root["id"]
        await until_status(data, sid, lambda s: s["session"]["turns"] == 6)
        async with asyncio.timeout(15):
            while True:
                tree = await request(data, "tree", session_id=sid)
                if len(tree) == 2 and tree[1]["outcome"] == "completed":
                    break
                await asyncio.sleep(0.05)
        identities = {
            s["id"]: (s["parent_id"], s["kernel_id"], s["workspace"]["path"]) for s in tree
        }
        queued = await request(
            data, "input", session_id=sid, body="Recover without losing the candidate result"
        )
        before = Store(data)
        assert before.events(sid, kind="context_compaction")
        count = before.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        before.close()
        process.send_signal(signal.SIGKILL)
        await process.wait()
        (data / "continue").touch()
        process = await start_daemon(data, log)
        done = await until_status(
            data, sid, lambda s: s["session"]["outcome"] != "active", seconds=20
        )
        assert done["session"]["outcome"] == "completed", done
        after = Store(data)
        try:
            assert identities == {
                s.id: (s.parent_id, s.kernel_id, s.workspace.path)
                for s in after.sessions(root_id=sid)
            }
            assert after.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] > count
            assert after.events(sid, kind="candidate_consumed")[-1]["payload"]["accepted"]
            assert after.events(sid, kind="verifier_result")[-1]["payload"]["passed"]
            assert after.events(sid, kind="kernel_recovery")[-1]["payload"]["restored"]
            assert any(
                m["id"] == queued["message_id"] and m["received_at"] for m in after.messages(sid)
            )
            assert after.db.execute("SELECT COUNT(*) FROM repository_files").fetchone()[0] > 0
        finally:
            after.close()
        assert "return a + b" in (repository / "mathops.py").read_text()
        assert "Add operands" in (repository / "mathops.py").read_text()
    finally:
        await stop_daemon(process, data)
