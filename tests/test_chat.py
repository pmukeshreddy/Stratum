"""Interactive clients use the production daemon/runtime; test models live here only."""

import asyncio
import io
import json
from pathlib import Path

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from threadweave.chat import Chat, chat_config, data_directory, most_recent
from threadweave.cli import execute, parser
from threadweave.coding_config import coding_options, update_coding_options
from threadweave.daemon import Daemon
from threadweave.gitops import git
from threadweave.models import ModelResponse, Outcome, now
from threadweave.terminal import EventRenderer, Terminal, safe_text

from .conftest import eventually, response
from .fakes import ScriptedProvider
from .test_coding import FIX


class InputTerminal(Terminal):
    def __init__(self, path, *, json_mode=False):
        self.output = io.StringIO()
        super().__init__(
            path,
            json_mode=json_mode,
            console=Console(file=self.output, width=80, color_system=None),
        )
        self.inputs = asyncio.Queue()

    async def read(self, prompt="> "):
        item = await self.inputs.get()
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
async def chat_setup(tmp_path, repository, coding_config):
    daemon = Daemon(tmp_path / "state")
    runtime = daemon.runtime
    coding_config.provider.name = "mock"
    update_coding_options(coding_config.task, capture_baseline=False)
    coding_config.refinement.enabled = False
    runtime.providers["mock"] = ScriptedProvider({})
    terminal = InputTerminal(tmp_path / "ui")

    async def rpc(directory, method, **args):
        return await daemon.dispatch(method, args)

    client = Chat(tmp_path / "state", repository, terminal, rpc=rpc)
    try:
        yield client, runtime, coding_config, terminal
    finally:
        await runtime.shutdown()
        daemon.lock.close()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["chat"],
        ["--workspace", "/repo"],
        ["chat", "--workspace", "/repo"],
        ["--config", "a.json", "chat"],
        ["chat", "--config", "a.json"],
    ],
)
async def test_default_and_explicit_commands_launch_chat(monkeypatch, arguments):
    seen = []

    async def launch(args):
        seen.append(args)
        return 0

    monkeypatch.setattr("threadweave.chat.chat", launch)
    args = parser().parse_args(arguments)
    assert await execute(args) == 0
    assert seen == [args]


def test_default_workspace_config_and_user_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    args = parser().parse_args([])
    assert (args.workspace or Path.cwd()) == tmp_path
    config = chat_config(tmp_path)
    assert config.provider.name == "codex_subscription"
    assert config.task.adapter == "workspace" and "process" in config.permissions
    assert config.context.max_tokens == 96000
    directory = data_directory(tmp_path)
    assert directory == tmp_path / "user-data/threadweave"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/coding.json").write_text('{"task":{"adapter":"coding"}}')
    assert chat_config(tmp_path).task.adapter == "workspace"
    (tmp_path / "configs/session.json").write_text('{"provider":{"model":"account-model"}}')
    assert chat_config(tmp_path).provider.model == "account-model"
    legacy = tmp_path / ".threadweave"
    legacy.mkdir()
    (legacy / "history.sqlite3").touch()
    assert data_directory(tmp_path) == legacy


async def idle(runtime, sid, turns):
    await eventually(lambda: runtime.store.session(sid).turns >= turns and sid not in runtime.tasks)
    assert not runtime.store.session(sid).runnable


async def test_multiple_messages_tools_and_verified_completion_keep_identity(chat_setup):
    client, runtime, config, terminal = chat_setup
    provider = ScriptedProvider(
        {
            "root": [
                ModelResponse(text="I am in your repository."),
                response("repo_search", query="def add"),
                ModelResponse(text="The add function subtracts."),
                response("apply_patch", patch=FIX),
                response("finish", result="Fixed and independently tested."),
                ModelResponse(text="Yes, the same conversation."),
            ]
        }
    )
    runtime.providers["mock"] = provider
    assert await client.open(config=config)
    sid, kernel = client.session["id"], client.session["kernel_id"]
    await runtime.start()
    await asyncio.sleep(0.1)
    assert not provider.requests  # No unsolicited invocation at shell startup.
    for prompt, turns in [
        ("Where are you?", 1),
        ("Inspect add", 3),
        ("Fix it", 5),
        ("Do you remember?", 6),
    ]:
        await client.submit(prompt)
        await idle(runtime, sid, turns)
        await client.poll()
    assert len(runtime.store.sessions(roots_only=True)) == 1
    session = runtime.store.session(sid)
    assert session.kernel_id == kernel and session.outcome == "active"
    assert runtime.store.events(sid, kind="verifier_result")[-1]["payload"]["passed"]
    assert len(runtime.store.events(sid, kind="user_intervention")) == 4
    assert all(m["received_at"] for m in runtime.store.messages(sid))
    assert all(r.session_id == sid for r in provider.requests)
    assert "Where are you?" in json.dumps(provider.requests[-1].messages)
    assert "repo_search" in terminal.output.getvalue()
    assert "Verification passed" in terminal.output.getvalue()
    assert len(provider.requests) == 6  # Plain replies don't spin the autonomous loop.
    assert not any(
        e["payload"]["result"].get("error") for e in runtime.store.events(sid, kind="tool_result")
    )


async def test_intervention_during_model_reply_is_not_stranded(chat_setup):
    client, runtime, config, terminal = chat_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def work(request):
        entered.set()
        await release.wait()
        return ModelResponse(text="First response")

    provider = ScriptedProvider({"root": [work, ModelResponse(text="Applied your intervention")]})
    runtime.providers["mock"] = provider
    await client.open(config=config)
    sid = client.session["id"]
    await runtime.start()
    await client.submit("Inspect")
    await entered.wait()
    await client.submit("Focus on cancellation")
    release.set()
    await idle(runtime, sid, 2)
    assert "Focus on cancellation" in json.dumps(provider.requests[1].messages)
    assert "queued" in terminal.output.getvalue()


async def test_interrupt_preserves_session_and_next_message_resumes(chat_setup):
    client, runtime, config, terminal = chat_setup
    entered = asyncio.Event()

    async def slow(request):
        entered.set()
        await asyncio.Event().wait()

    provider = ScriptedProvider({"root": [slow]})
    runtime.providers["mock"] = provider
    await client.open(config=config)
    sid = client.session["id"]
    await runtime.start()
    shell = asyncio.create_task(client.run())
    terminal.inputs.put_nowait("Work")
    await entered.wait()
    terminal.inputs.put_nowait(KeyboardInterrupt())
    await eventually(lambda: runtime.store.session(sid).paused)
    assert runtime.store.session(sid).outcome == Outcome.ACTIVE
    await eventually(lambda: sid not in runtime.tasks)
    provider.scripts["root"] = [ModelResponse(text="Recovered")]
    terminal.inputs.put_nowait("Change direction")
    await idle(runtime, sid, 1)
    terminal.inputs.put_nowait(KeyboardInterrupt())  # Idle: clear input only.
    terminal.inputs.put_nowait("/exit")
    assert await shell == 0
    assert runtime.store.session(sid).outcome == Outcome.ACTIVE
    assert not runtime.store.session(sid).paused
    assert runtime.store.events(sid, kind="client_detached")
    assert len(runtime.store.sessions()) == 1


async def test_exit_detaches_without_cancelling_active_work(chat_setup):
    client, runtime, config, terminal = chat_setup
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(request):
        entered.set()
        await release.wait()
        return ModelResponse(text="Completed after detachment")

    runtime.providers["mock"] = ScriptedProvider({"root": [slow]})
    await client.open(config=config)
    sid = client.session["id"]
    await runtime.start()
    task = asyncio.create_task(client.run())
    terminal.inputs.put_nowait("Work")
    await entered.wait()
    terminal.inputs.put_nowait(EOFError())
    assert await task == 0
    assert sid in runtime.tasks and not runtime.store.session(sid).paused
    release.set()
    await idle(runtime, sid, 1)


async def test_resume_and_continue_same_workspace(chat_setup, tmp_path):
    client, runtime, config, terminal = chat_setup
    await client.open(config=config)
    sid = client.session["id"]
    await client.submit("Persist me")
    await runtime._run_turn(sid)
    # A root for another workspace must not win --continue.
    runtime.create("Other workspace", tmp_path, config=config, mode="interactive")
    assert (
        most_recent([s.model_dump() for s in runtime.store.sessions()], client.workspace)["id"]
        == sid
    )
    assert await client.open(continue_recent=True)
    assert client.session["id"] == sid
    assert await client.open(resume_id=sid)
    assert client.session["id"] == sid
    assert runtime.store.messages(sid)[0]["body"] == "Persist me"
    with pytest.raises(KeyError, match="Unknown session"):
        await client.open(resume_id="missing")
    with pytest.raises(ValueError, match="No previous"):
        most_recent([], client.workspace)


async def test_slash_commands_do_not_invoke_model(chat_setup):
    client, runtime, config, terminal = chat_setup
    await client.open(config=config)
    first = client.session["id"]
    for command in (
        "help",
        "status",
        "usage",
        "tree",
        "diff",
        "history",
        "experiments",
        "compact",
        "pause",
        "resume",
        "not-a-command",
    ):
        assert await client.submit("/" + command)
    assert runtime.store.usage(first).model_calls == 0
    assert "Model calls:" in terminal.output.getvalue()
    assert "No changes" in terminal.output.getvalue()
    assert await client.submit("/new")
    assert client.session["id"] != first
    assert len(runtime.store.sessions()) == 2
    assert not await client.submit("/exit")


async def test_dirty_continue_preserves_user_changes_and_config(chat_setup):
    client, runtime, config, terminal = chat_setup
    path = client.workspace / "user-note.txt"
    path.write_text("unrelated user content")
    before = git(client.workspace, "status", "--porcelain=v1")
    for item in ("3", "1"):
        terminal.inputs.put_nowait(item)
    assert await client.open(config=config)
    sid = client.session["id"]
    assert not coding_options(runtime.store.config(sid).task).require_clean_baseline
    assert coding_options(config.task).require_clean_baseline
    assert "user-note" in terminal.output.getvalue()
    assert git(client.workspace, "status", "--porcelain=v1") == before
    assert path.read_text() == "unrelated user content"


async def test_dirty_exit_never_admits_session(chat_setup):
    client, runtime, config, terminal = chat_setup
    (client.workspace / "user-note").write_text("keep")
    terminal.inputs.put_nowait("2")
    assert not await client.open(config=config)
    assert runtime.store.sessions() == []


async def test_resume_a_session_blocked_by_old_dirty_gate(chat_setup):
    client, runtime, config, terminal = chat_setup
    old = runtime.create("Old failed run", client.workspace, config=config, mode="interactive")
    runtime.store.finish(old.id, Outcome.FAILED, "Previously blocked by dirty repository")
    (client.workspace / "user-note").write_text("keep")
    terminal.inputs.put_nowait("1")
    assert await client.open(resume_id=old.id)
    assert client.session["id"] == old.id
    assert not coding_options(runtime.store.config(old.id).task).require_clean_baseline
    assert coding_options(runtime.store.config(old.config_id).task).require_clean_baseline
    assert runtime.store.events(old.id, kind="dirty_baseline_consent")


async def test_json_mode_remains_structured(chat_setup):
    client, runtime, config, terminal = chat_setup
    terminal.json_mode = True
    await client.open(config=config)
    await client.submit("Hello")
    await client.slash("/usage")
    await client.poll()
    rows = [json.loads(line) for line in terminal.output.getvalue().splitlines()]
    assert rows[0]["type"] == "chat_started"
    assert any(r["type"] == "user_intervention" for r in rows)
    assert any(r["type"] == "control_result" for r in rows)


async def test_incremental_text_event_visible_before_response_finishes(chat_setup):
    client, runtime, config, terminal = chat_setup
    release = asyncio.Event()

    class Streaming:
        async def invoke(self, request, emit):
            await emit("Live token")
            await release.wait()
            return ModelResponse(text="Live token, finished")

    runtime.providers["mock"] = Streaming()
    await client.open(config=config)
    sid = client.session["id"]
    await runtime.start()
    await client.submit("Stream")
    await eventually(lambda: runtime.store.events(sid, kind="model_stream"))
    assert not runtime.store.events(sid, kind="model_response")
    await client.poll()
    assert "Live token" in terminal.output.getvalue()
    release.set()
    await idle(runtime, sid, 1)


async def test_concurrent_child_streams_are_separate_and_bounded(tmp_path):
    terminal = InputTerminal(tmp_path)
    terminal.tty = True
    renderer = EventRenderer(terminal, "root")

    async def event(sid, kind, payload, parent="call"):
        await renderer.render(
            {"session_id": sid, "type": kind, "payload": payload, "parent_event_id": parent}
        )

    await event("root", "subagent_created", {"child_id": "child", "name": "investigator"})
    await asyncio.gather(
        event("root", "model_stream", {"text": "Root progress"}),
        event("child", "model_stream", {"text": "Child progress"}, "other"),
    )
    assert len(terminal.drafts) == 2
    assert "Child progress" in terminal.toolbar()[0][1]
    await event("root", "model_response", {"text": "Root answer"})
    await event("child", "model_response", {"text": "Child answer"}, "other")
    assert not terminal.drafts
    assert "Root answer" in terminal.output.getvalue()
    assert "investigator" in terminal.output.getvalue()
    await event("root", "tool_call", {"name": "repo_search", "arguments": {"pattern": "x" * 10000}})
    assert "x" * 400 not in terminal.output.getvalue()
    assert safe_text("safe\x1b[31mred\x1b]52;c;secret\x07\x00") == "safered"


async def test_real_prompt_multiline_history_interrupt_and_eof(tmp_path):
    with create_pipe_input() as pipe:
        terminal = Terminal(
            tmp_path, input=pipe, output=DummyOutput(), console=Console(file=io.StringIO())
        )
        task = asyncio.create_task(terminal.read())
        await asyncio.sleep(0.05)
        pipe.send_text("first\x1b\rsecond\r")
        assert await asyncio.wait_for(task, 2) == "first\nsecond"
        task = asyncio.create_task(terminal.read())
        await asyncio.sleep(0.05)
        pipe.send_text("\x1b[A\r")
        assert await asyncio.wait_for(task, 2) == "first\nsecond"

        async def controlled_read():
            try:
                return await terminal.read()
            except KeyboardInterrupt:
                return "interrupted"
            except EOFError:
                return "detached"

        task = asyncio.create_task(controlled_read())
        await asyncio.sleep(0.05)
        pipe.send_bytes(b"\x03")
        assert await task == "interrupted"
        task = asyncio.create_task(controlled_read())
        await asyncio.sleep(0.05)
        pipe.send_bytes(b"\x04")
        assert await task == "detached"
        assert (tmp_path / "input-history").stat().st_mode & 0o777 == 0o600


async def test_idle_conversation_does_not_spend_wall_budget(chat_setup):
    client, runtime, config, terminal = chat_setup
    await client.open(config=config)
    sid = client.session["id"]
    runtime.store.update(sid, started_at=now() - 100000)
    assert runtime._elapsed(sid) == 0
    runtime._check_limits(sid)
    runtime.store.finish(sid, Outcome.LIMITED, "test limit")
    with pytest.raises(ValueError, match="resource limit"):
        await client.submit("Do not reset the budget")


async def test_live_text_is_visible_without_cursor_position_reporting(tmp_path):
    output = io.StringIO()
    with create_pipe_input() as pipe:
        tty = Vt100_Output(output, lambda: Size(rows=25, columns=100), enable_cpr=False)
        terminal = Terminal(
            tmp_path, input=pipe, output=tty, console=Console(file=output, width=100)
        )
        reader = asyncio.create_task(terminal.read())
        await asyncio.sleep(0.05)
        pipe.send_text("unfinished input")
        await terminal.stream("request", "Agent", "LIVE EARLY TOKENS")
        await asyncio.sleep(0.1)
        assert "LIVE EARLY TOKENS" in output.getvalue()  # Before any final response.
        await terminal.write("● Child investigator is searching")
        pipe.send_text("\r")
        assert await reader == "unfinished input"  # Async output didn't corrupt editing.


async def test_auxiliary_model_json_is_not_a_conversation_reply(tmp_path):
    terminal = InputTerminal(tmp_path)
    renderer = EventRenderer(terminal, "root")
    for kind in ("model_invocation_started", "model_stream", "model_response"):
        await renderer.render(
            {
                "session_id": "root",
                "type": kind,
                "payload": {"purpose": "refinement", "text": '{"internal_proposal":123}'},
            }
        )
    assert "Refinement in progress" in terminal.output.getvalue()
    assert "internal_proposal" not in terminal.output.getvalue()


async def test_input_keeps_reading_during_a_slow_slash_command(chat_setup):
    client, runtime, config, terminal = chat_setup
    await client.open(config=config)
    sid = client.session["id"]
    runtime.providers["mock"] = ScriptedProvider({"root": [ModelResponse(text="Message received")]})
    original_rpc, original_read = client.rpc, terminal.read
    entered, release, read_message = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow_rpc(directory, method, **args):
        if method == "diff":
            entered.set()
            await release.wait()
        return await original_rpc(directory, method, **args)

    async def observed_read(prompt="> "):
        item = await original_read(prompt)
        if item == "message while diff runs":
            read_message.set()
        return item

    client.rpc, terminal.read = slow_rpc, observed_read
    await runtime.start()
    shell = asyncio.create_task(client.run())
    try:
        terminal.inputs.put_nowait("/diff")
        await entered.wait()
        terminal.inputs.put_nowait("message while diff runs")
        await asyncio.wait_for(read_message.wait(), 1)
        assert not release.is_set()  # Input was read while the RPC was still blocked.
        release.set()
        await idle(runtime, sid, 1)
        assert runtime.store.messages(sid)[0]["body"] == "message while diff runs"
        terminal.inputs.put_nowait("/exit")
        assert await shell == 0
    finally:
        release.set()
        if not shell.done():
            shell.cancel()
            await asyncio.gather(shell, return_exceptions=True)


async def test_inspection_without_tests_does_not_weaken_finish_gate(chat_setup, monkeypatch):
    client, runtime, config, terminal = chat_setup
    from threadweave import coding

    original_detect = coding.detect

    def no_test_detection(root):
        metadata = original_detect(root)
        metadata["suggested_commands"] = {}
        return metadata

    monkeypatch.setattr(coding, "detect", no_test_detection)
    update_coding_options(config.task, test_commands=[])
    update_coding_options(config.task, require_change=False)
    runtime.providers["mock"] = ScriptedProvider(
        {"root": [ModelResponse(text="Inspection is allowed")]}
    )
    await client.open(config=config)
    sid = client.session["id"]
    await client.submit("Inspect")
    await runtime._run_turn(sid)
    assert runtime.store.session(sid).outcome == "active"
    event = runtime.store.event(sid, "test", {})
    verification, error = await runtime._verify(sid, event)
    assert not error and not verification.passed
    assert "No test commands" in json.dumps(verification.model_dump())


async def test_new_dirty_confirmation_uses_the_existing_input_reader(chat_setup):
    client, runtime, config, terminal = chat_setup
    await client.open(config=config)
    original = client.session["id"]
    (client.workspace / "user-note").write_text("preserve")
    shell = asyncio.create_task(client.run())
    for line in ("/new", "3", "1", "/status", "/exit"):
        terminal.inputs.put_nowait(line)
    assert await asyncio.wait_for(shell, 3) == 0
    assert client.session["id"] != original
    assert len(runtime.store.sessions()) == 2
    assert not runtime.store.messages(client.session["id"])
    assert (client.workspace / "user-note").read_text() == "preserve"
