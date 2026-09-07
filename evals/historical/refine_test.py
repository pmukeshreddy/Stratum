"""Evaluator-only regression for a real historical chat routing defect."""

from tests.fakes import ScriptedProvider
from tests.test_chat import InputTerminal
from threadweave.chat import Chat
from threadweave.daemon import Daemon
from threadweave.models import RunConfig


async def test_refine_is_a_control_request(tmp_path):
    daemon = Daemon(tmp_path / "state")
    terminal = InputTerminal(tmp_path / "ui")

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    config = RunConfig(
        provider={"name": "mock", "model": "test-only-verifier"},
        refinement={"enabled": True, "automatic": False},
    )
    daemon.runtime.providers["mock"] = ScriptedProvider({})
    chat = Chat(tmp_path / "state", tmp_path, terminal, rpc=rpc)
    try:
        await chat.open(config=config)
        assert await chat.submit("/refine")
        output = terminal.output.getvalue().lower()
        assert "unknown" not in output
        assert "refin" in output
        assert daemon.runtime.store.events(chat.session["id"], kind="refinement_trigger")
    finally:
        await daemon.runtime.shutdown()
        daemon.lock.close()
