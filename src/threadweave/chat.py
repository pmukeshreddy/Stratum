"""Interactive client for the existing daemon. No model or tools execute here."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from .coding_config import coding_options, update_coding_options
from .configuration import load_config
from .daemon import request
from .gitops import git
from .models import RunConfig, new_id
from .terminal import EventRenderer, Terminal

HELP = """/help         Show commands
/status       Session status and workspace
/state        Inspect L1 / L2 / L3 metadata (not stored values)
/states       Inspect Continual Harness entries and versions
/usage        Recursive tokens, calls, turns and execution time
/tree         Persistent child sessions
/diff         Current Git diff (pause first if working)
/history      Recent conversation and tool activity
/experiments  Durable experiments
/compact      Compact active context (pause first if working)
/refine       Request evidence-based learning at a safe boundary (not a StateEdit)
/pause        Interrupt this session's current turn; keep state
/resume       Continue paused work
/new          Start a new conversation; keep the old one
/exit         Detach without stopping background work

Enter sends. Alt-Enter or Ctrl-J inserts a newline; paste supports multiline text.
Arrow keys recall history. Type during work to queue an intervention.
Ctrl-C pauses current work, or clears input when idle. Ctrl-D detaches on empty input.
Children keep running when their parent is paused. /tree shows their state.
"""


def data_directory(workspace, explicit=None):
    if explicit:
        return explicit.resolve()
    # Existing local stores remain usable; new chats don't dirty the repository.
    legacy = workspace / ".threadweave"
    if (legacy / "history.sqlite3").is_file():
        return legacy.resolve()
    return (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "threadweave"
    ).resolve()


def chat_config(workspace, explicit=None):
    path = explicit or workspace / "configs/session.json"
    if not explicit and not path.is_file() and (Path.cwd() / "configs/session.json").is_file():
        path = Path.cwd() / "configs/session.json"
    if explicit or path.is_file():
        config = load_config(path)
    else:
        config = RunConfig(
            task={"adapter": "workspace"},
            context={"max_tokens": 96000, "result_chars": 131072, "summary_chars": 6000},
            limits={"token_budget": 3_000_000, "wall_seconds": 7200},
            permissions=[
                "workspace.read",
                "workspace.write",
                "python",
                "process",
                "agents",
                "state",
                "mcp",
            ],
        )
    return config


def most_recent(sessions, workspace):
    compatible = [
        s
        for s in sessions
        if not s["parent_id"] and Path(s["workspace"]["path"]).resolve() == workspace
    ]
    if not compatible:
        raise ValueError(
            "No previous root session for this workspace; omit --continue to start one"
        )
    return max(compatible, key=lambda s: (s["updated_at"], s["created_at"]))


class Chat:
    def __init__(self, directory, workspace, terminal, *, rpc=request, verbose=False):
        self.directory, self.workspace, self.terminal = directory, workspace, terminal
        self.rpc, self.verbose = rpc, verbose
        self.session = None
        self.renderer = None
        self.cursor = -1
        self.busy = False
        self.poll_lock = asyncio.Lock()
        self.config = None
        self.inputs = None

    async def call(self, method, **args):
        return await self.rpc(self.directory, method, **args)

    async def current(self, method, **args):
        return await self.call(method, session_id=self.session["id"], **args)

    async def notice(self, text):
        if self.terminal.json_mode:
            await self.terminal.json({"type": "notice", "text": text})
        else:
            await self.terminal.write(text)

    async def dirty_consent(self, config):
        if (
            config.task.adapter != "coding"
            or not coding_options(config.task).require_clean_baseline
        ):
            return True
        status = await asyncio.to_thread(
            git, self.workspace, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if not coding_options(config.task).require_clean_baseline or not status.strip():
            return True
        await self.notice(
            "Repository has existing changes.\n[1] Continue using current repository state\n[2] Exit\n[3] Show git status"
        )
        while True:
            try:
                value = (
                    await self.inputs.get()
                    if self.inputs is not None
                    else await self.terminal.read("> ")
                )
                if value is None or value.strip() == "/exit":
                    if self.inputs is not None:
                        await self.inputs.put(None)
                    return False
                choice = value.strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                return False
            if choice == "1":
                update_coding_options(config.task, require_clean_baseline=False)
                await self.notice(
                    "Using current files as this session's baseline. Nothing was committed or stashed."
                )
                return True
            if choice in {"2", "/exit"}:
                return False
            if choice == "3":
                await self.notice(status[:12000])
            else:
                await self.notice("Choose 1, 2, or 3.")

    async def open(self, *, config=None, resume_id=None, continue_recent=False):
        if continue_recent:
            resume_id = most_recent(await self.call("list"), self.workspace)["id"]
        if resume_id:
            status = await self.call("status", session_id=resume_id)
            session = status["session"]
            self.workspace = Path(session["workspace"]["path"])  # Stored canonical workspace.
            self.config = RunConfig.model_validate(await self.call("config", session_id=resume_id))
            consent = False
            if not status.get("environment_prepared") and session["lifecycle"] != "RUNNING":
                clean_required = coding_options(self.config.task).require_clean_baseline
                if not await self.dirty_consent(self.config):
                    return False
                consent = (
                    clean_required and not coding_options(self.config.task).require_clean_baseline
                )
            self.session = await self.call(
                "chat_open", session_id=resume_id, accept_current_baseline=consent
            )
            # Replay bounded recent useful activity, not the entire trajectory.
            recent = await self.current("history", limit=100, tree=True)
            self.cursor = recent[-1]["seq"] if recent else -1
        else:
            self.config = config.model_copy(deep=True)
            if not await self.dirty_consent(self.config):
                return False
            self.session = await self.call(
                "create",
                instruction="Assist the user in this persistent session. Follow their latest messages. Use persistent IPython for inspectable, stateful or verifiable work and delegate independent investigations; preserve unrelated workspace changes.",
                workspace=str(self.workspace),
                config=self.config.model_dump(mode="json"),
                name="root",
                mode="interactive",
            )
            if (
                coding_options(config.task).require_clean_baseline
                and not coding_options(self.config.task).require_clean_baseline
            ):
                self.session = await self.call(
                    "chat_open", session_id=self.session["id"], accept_current_baseline=True
                )
            self.cursor = -1
            recent = []
        self.renderer = EventRenderer(self.terminal, self.session["id"], verbose=self.verbose)
        for child in await self.current("tree"):
            if child["id"] != self.session["id"]:
                self.renderer.names[child["id"]] = child["name"]
        provider = self.config.models.get(self.config.routing.default) or self.config.provider
        if self.terminal.json_mode:
            await self.terminal.json(
                {
                    "type": "chat_started",
                    "session_id": self.session["id"],
                    "workspace": str(self.workspace),
                    "provider": provider.name,
                    "model": provider.model,
                }
            )
        else:
            await self.terminal.write(
                "\nThreadweave\n────────────────────────────────", style="bold"
            )
            await self.terminal.write(
                f"Workspace  {self.workspace}\nModel      {provider.model}\nProvider   {'ChatGPT / Codex subscription' if provider.name == 'codex_subscription' else provider.name}\nSession    {self.session['id']}\n────────────────────────────────\nType /help for commands."
            )
            if resume_id:
                await self.notice("Resumed the same conversation. /history shows recent activity.")
                for event in recent:
                    if event["type"] in {"model_response", "conversation_completed"}:
                        # One recent response gives orientation without replaying tools.
                        last = event
                if "last" in locals():
                    await self.renderer.render(last)
        return True

    async def poll(self):
        async with self.poll_lock:
            # Read status BEFORE events so a final response is rendered before Ready.
            status = await self.current("status")
            rows = await self.current(
                "history" if self.terminal.json_mode else "chat_events",
                after=self.cursor,
                limit=100,
                tree=True,
            )
            for event in rows:
                await self.renderer.render(event)
                self.cursor = event["seq"]
            session = status["session"]
            busy = (
                session["outcome"] == "active"
                and (session["runnable"] or session["lifecycle"] == "RUNNING")
                and not session["paused"]
            )
            if self.busy and not busy and len(rows) < 100:
                await self.notice(
                    "Ready."
                    if session["outcome"] == "active"
                    else f"Session {session['outcome']}. /new starts a new run; /resume retries recoverable failures."
                )
            if len(rows) < 100:
                self.busy = busy
                if self.terminal.busy != busy:
                    self.terminal.busy = busy
                    self.terminal.invalidate()
            return status

    async def watch(self):
        unavailable = False
        while True:
            try:
                await self.poll()
                if unavailable:
                    await self.notice("Reconnected to the daemon; session retained.")
                unavailable = False
            except (OSError, RuntimeError, ValueError) as exc:
                if not unavailable:
                    await self.notice(
                        f"Connection interrupted: {exc}. Reconnecting; /exit still detaches."
                    )
                unavailable = True
            await asyncio.sleep(0.1 if not unavailable else 1)

    async def interrupt(self):
        status = await self.current("status")
        session = status["session"]
        if session["runnable"] or session["lifecycle"] == "RUNNING":
            await self.current("pause")
            await self.notice(
                "Current turn interrupted; session retained. Send a message or /resume."
            )

    async def submit(self, text):
        text = text.strip()
        if not text:
            return True
        if text.startswith("/"):
            return await self.slash(text)
        # Don't let an in-flight poll publish its old idle state after this send.
        async with self.poll_lock:
            await self.current("chat_input", body=text)
            if self.busy:
                await self.notice("Message queued for the next safe turn boundary.")
            self.busy = self.terminal.busy = True
            self.terminal.invalidate()
        return True

    async def slash(self, text):
        if text == "/exit":
            return False
        if text == "/help":
            await self.notice(HELP)
            return True
        command = text.removeprefix("/")
        if command == "new":
            # Stop old event rendering while switching identities. Old work keeps running.
            async with self.poll_lock:
                opened = await self.open(config=self.config)
                if opened:
                    self.busy = self.terminal.busy = False
                    self.terminal.drafts.clear()
            return opened
        if command not in {
            "status",
            "state",
            "states",
            "usage",
            "tree",
            "diff",
            "history",
            "experiments",
            "compact",
            "pause",
            "resume",
            "refine",
        }:
            await self.notice("Unknown command. Type /help.")
            return True
        result = await self.current(
            "status" if command == "usage" else "information" if command == "state" else command,
            **(
                {"limit": 40, "tree": True}
                if command == "history"
                else {"request_id": new_id()}
                if command == "refine"
                else {}
            ),
        )
        if self.terminal.json_mode:
            await self.terminal.json(
                {"type": "control_result", "command": command, "result": result}
            )
        elif command == "refine":
            if result["status"] == "requested":
                await self.notice(
                    "Refinement requested; no changes applied yet. "
                    + (
                        "Session is paused; use /resume to process it."
                        if result.get("waiting_for_resume")
                        else "It will run at the next safe boundary."
                    )
                )
            else:
                await self.notice(f"Refinement {result['status']}: {result.get('reason', '')}")
        elif command == "usage":
            usage = result["tree_usage"]
            cost = "subscription / unavailable" if usage["cost"] is None else str(usage["cost"])
            await self.notice(
                f"Model calls: {usage['model_calls']}\nTokens: {usage['input_tokens']} input / {usage['output_tokens']} output / {usage['cached_input_tokens']} cached\nTurns: {usage['turns']}\nTools: {usage['tool_calls']} · Python: {usage['python_executions']} · Subagents: {usage['subagent_count']}\nExecution: {usage['wall_seconds']:.1f}s\nCost: {cost}"
            )
        elif command == "status":
            s = result["session"]
            await self.notice(
                f"{s['name']} · {s['lifecycle']} · {s['outcome']}{' · paused' if s['paused'] else ''}\nWorkspace: {s['workspace']['path']}\nTurns: {s['turns']}"
            )
        elif command == "tree":
            await self.notice(render_tree(result))
        elif command == "state":
            await self.notice(
                f"L1 · {result['L1']['blocks']} active blocks · {result['L1']['selected_entries']} selected entries\n"
                f"L2 · kernel {result['L2']['kernel_id']} · {len(result['L2']['children'])} children · {result['L2']['checkpointed_variables']} checkpointed variables\n"
                f"L3 · {result['L3']['events']} events · {result['L3']['artifacts']} artifacts · {result['L3']['pending_messages']} pending messages"
            )
        elif command == "states":
            for entry in result:
                await self.notice(
                    f"{entry['id']} · {entry['kind']} · v{entry['version']} · {entry['title']}{' · deleted' if entry['deleted'] else ''}"
                )
            if not result:
                await self.notice("No selected or reusable state entries have been stored.")
        elif command == "diff":
            await self.terminal.write(result["patch"] or "No changes.", code="diff")
            if len(result["patch"]) >= 16000:
                await self.notice("Full diff artifact: " + result["artifact_id"])
            await self.notice("End of diff.")
        elif command == "history":
            replay = EventRenderer(self.terminal, self.session["id"])
            for event in result:
                if event["type"] == "user_intervention":
                    await self.notice("You: " + str(event["payload"].get("body", ""))[:1500])
                elif event["type"] != "model_stream":
                    await replay.render(event)
        elif command == "experiments":
            for experiment in result:
                await self.notice(
                    f"{experiment['id']} · {experiment.get('status', '')} · {experiment.get('hypothesis', '')}"[
                        :1200
                    ]
                )
            if not result:
                await self.notice("No experiments in this session.")
        else:
            await self.notice(
                {
                    "compact": "Context compacted; full history retained.",
                    "pause": "Paused; session retained.",
                    "resume": "Continuing this session.",
                }[command]
            )
        return True

    async def run(self):
        watcher = asyncio.create_task(self.watch())
        inputs = asyncio.Queue()
        self.inputs = inputs

        async def read_input():
            # Keep the terminal in editable/raw input mode even while a slash
            # command is waiting for its RPC. Otherwise early Enter can become a
            # literal newline between prompts and leave the next message unsent.
            while True:
                try:
                    line = await self.terminal.read()
                except KeyboardInterrupt:
                    try:
                        await self.interrupt()
                    except (OSError, RuntimeError, ValueError) as exc:
                        await self.notice(f"Could not interrupt: {exc}")
                    continue
                except EOFError:
                    await inputs.put(None)
                    return
                await inputs.put(line)
                if line.strip() == "/exit":
                    return

        reader = asyncio.create_task(read_input())
        try:
            while True:
                try:
                    line = await inputs.get()
                    if line is None:
                        break
                    if not await self.submit(line):
                        break
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    await self.notice(f"Could not complete that request: {exc}")
        finally:
            reader.cancel()
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            self.inputs = None
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            with contextlib.suppress(OSError, RuntimeError):
                await self.current("chat_detach")
            await self.notice(
                "Detached. Your session is saved; running work continues. Use threadweave --continue to return."
            )
        return 0


def render_tree(sessions):
    children = {}
    for session in sessions:
        children.setdefault(session["parent_id"], []).append(session)
    lines = []

    def visit(node, prefix="", connector=""):
        lines.append(
            f"{prefix}{connector}{node['name']} · {node['id'][:8]} · {node['lifecycle']} / {node['outcome']}"
        )
        rows = children.get(node["id"], [])
        for i, child in enumerate(rows):
            last = i == len(rows) - 1
            visit(
                child,
                prefix + ("    " if connector == "└── " else "│   " if connector else ""),
                "└── " if last else "├── ",
            )

    for root in children.get(None, []):
        visit(root)
    return "\n".join(lines)


async def chat(args):
    from .cli import ensure_daemon, resolved_config

    workspace = (args.workspace or Path.cwd()).resolve()
    directory = data_directory(workspace, args.data)
    terminal = Terminal(directory, json_mode=bool(args.json))
    client = Chat(directory, workspace, terminal, verbose=bool(args.verbose))
    config = None
    if not args.resume_id and not args.continue_recent:
        config = await resolved_config(chat_config(workspace, args.config))
    info = await ensure_daemon(directory)
    if not {
        "interactive_chat",
        "information_hierarchy",
        "recursive_sessions",
        "python_control_plane_v1",
    } <= set(info.get("capabilities", [])):
        raise ValueError(
            f"An older daemon owns {directory}. Stop it with threadweave --data {directory} daemon stop, then retry; sessions are preserved."
        )
    if args.resume_id and args.workspace:
        session = (await request(directory, "status", session_id=args.resume_id))["session"]
        if Path(session["workspace"]["path"]) != workspace:
            raise ValueError("--workspace differs from the resumed session's workspace")
    if await client.open(
        config=config, resume_id=args.resume_id, continue_recent=args.continue_recent
    ):
        return await client.run()
    return 0
