"""A scrollback-first terminal: editable input stays live while events arrive."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import is_done
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text


def safe_text(value):
    """Repository/model text must not control the terminal (OSC, ANSI, C0/C1)."""
    value = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", str(value))
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return "".join(c for c in value if c in "\n\t" or ord(c) >= 32 and not 127 <= ord(c) <= 159)


class Terminal:
    def __init__(self, directory: Path, *, json_mode=False, input=None, output=None, console=None):
        self.json_mode = json_mode
        self.console = console or Console(highlight=False)
        self.tty = input is not None or sys.stdin.isatty() and sys.stdout.isatty()
        self.lock = asyncio.Lock()
        self.busy = False
        self.drafts: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self.plain_stream = None
        self.prompt = None
        if self.tty:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            history = directory / "input-history"
            fd = os.open(history, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            os.close(fd)
            history.chmod(0o600)
            bindings = KeyBindings()

            @bindings.add("enter")
            def submit(event):
                event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            @bindings.add("c-j")
            def newline(event):
                event.current_buffer.insert_text("\n")

            @bindings.add("c-c")
            def interrupt(event):
                event.app.exit(exception=KeyboardInterrupt())

            @bindings.add("c-d")
            def detach(event):
                if not event.current_buffer.text:
                    event.app.exit(exception=EOFError())
                else:
                    event.current_buffer.delete()

            self.prompt = PromptSession(
                history=FileHistory(str(history)),
                key_bindings=bindings,
                multiline=True,
                prompt_continuation="  ",
                input=input,
                output=output,
                enable_suspend=False,
            )
            # PromptSession's built-in bottom toolbar is hidden on terminals without
            # cursor-position reports. A small inline window works there too and
            # never enters the alternate screen or consumes the full terminal.
            original = self.prompt.app.layout
            live = ConditionalContainer(
                Window(
                    FormattedTextControl(self.toolbar),
                    height=lambda: 3 if self.drafts else 1,
                    wrap_lines=True,
                    style="class:bottom-toolbar",
                ),
                filter=~is_done,
            )
            self.prompt.app.layout = Layout(
                HSplit([original.container, live]), focused_element=original.current_window
            )

    def toolbar(self):
        hint = (
            "Working · type to intervene · Ctrl-C pauses"
            if self.busy
            else "Alt-Enter: newline · /help"
        )
        if self.json_mode or not self.drafts:
            return [("class:bottom-toolbar", hint)]
        label, text = next(reversed(self.drafts.values()))
        # Incremental model text is visible immediately, in a bounded live region.
        # On completion it moves to scrollback as rendered Markdown. No duplicate
        # scrollback or partial ANSI redraws when children emit concurrently.
        width = max(20, self.console.width - 2)
        tail = text.replace("\n", " ")[-width * 2 :]
        return [("class:bottom-toolbar", f"{hint}\n{label}: {tail}")]

    def invalidate(self):
        if self.prompt:
            self.prompt.app.invalidate()

    async def read(self, prompt="> "):
        if self.prompt:
            return await self.prompt.prompt_async(prompt, handle_sigint=True)
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            raise EOFError
        return line.rstrip("\n")

    async def write(self, value, *, style=None, markdown=False, code=None, end="\n"):
        text = safe_text(value)
        rendered = (
            Syntax(text, code, word_wrap=True, background_color="default")
            if code
            else Markdown(text)
            if markdown
            else Text(text, style=style)
        )
        async with self.lock:

            def output():
                self.console.print(rendered, end=end, soft_wrap=not code)

            if self.prompt and self.prompt.app.is_running:
                await run_in_terminal(output)
            else:
                output()

    async def json(self, value):
        await self.write(json.dumps(value, ensure_ascii=False))

    async def stream(self, key, label, delta):
        previous = self.drafts.get(key, (label, ""))[1]
        self.drafts[key] = (label, (previous + safe_text(delta))[-64000:])
        self.drafts.move_to_end(key)
        while len(self.drafts) > 32:
            self.drafts.popitem(last=False)
        if self.tty:
            self.invalidate()
        else:
            if self.plain_stream != key:
                await self.write(f"\n{label}: ", end="")
                self.plain_stream = key
            await self.write(delta, end="")

    async def response(self, key, label, text):
        draft = self.drafts.pop(key, None)
        text = text or (draft[1] if draft else "")
        self.invalidate()
        if text and (self.tty or not draft):
            await self.write(label + ":", style="bold cyan")
            await self.write(text, markdown=True)
        elif draft:
            suffix = text[len(draft[1]) :] if text.startswith(draft[1]) else ""
            await self.write(suffix)
        self.plain_stream = None


def tool_summary(name, args):
    if name == "apply_patch":
        patch = args.get("patch", "")
        paths = re.findall(r"^\+\+\+ (?:b/)?(.+)$", patch, re.MULTILINE)
        return ", ".join(paths)[:240] or "validated patch"
    fields = ("pattern", "query", "path", "symbol", "targets", "command", "name", "instruction")
    parts = [f"{k}: {args[k]}" for k in fields if k in args]
    if name == "python":
        parts = [args.get("code", "").split("\n")[0]]
    return safe_text(" · ".join(parts))[:320]


def command_summary(result):
    stdout = result.get("stdout", "")
    match = re.search(r"[^\n]*\b\d+ passed\b[^\n]*", stdout)
    if match:
        return match[0].strip("= ")[:240]
    failures = result.get("failures", [])
    if failures:
        return str(failures[0].get("message") or failures[0].get("name"))[:240]
    return f"exit {result.get('exit_code', '?')} · {result.get('duration', 0):.2f}s"


class EventRenderer:
    def __init__(self, terminal, root_id, *, verbose=False):
        self.terminal, self.root_id, self.verbose = terminal, root_id, verbose
        self.names = {root_id: "Agent"}

    async def render(self, event):
        terminal, kind, payload = self.terminal, event["type"], event["payload"]
        if terminal.json_mode:
            await terminal.json(event)
            return
        purpose = payload.get("purpose") or payload.get("metadata", {}).get("purpose", "agent")
        if purpose != "agent" and kind in {
            "model_stream",
            "model_response",
            "model_invocation_started",
        }:
            if kind == "model_invocation_started":
                await terminal.write("● " + str(purpose).capitalize() + " in progress", style="dim")
            return
        sid = event["session_id"]
        label = self.names.get(sid, f"Child {sid[:8]}")
        prefix = "" if sid == self.root_id else f"[{label}] "
        if kind == "model_stream":
            await terminal.stream(event["parent_event_id"], label, payload.get("text", ""))
        elif kind == "model_response":
            await terminal.response(event["parent_event_id"], label, payload.get("text", ""))
        elif kind == "model_invocation_started":
            await terminal.write(prefix + "● Thinking…", style="dim")
        elif kind == "tool_call":
            name = payload.get("name", "tool")
            await terminal.write(prefix + "● " + name, style="cyan")
            summary = tool_summary(name, payload.get("arguments", {}))
            if summary:
                await terminal.write("  " + summary, style="dim")
        elif kind == "tool_result":
            result = payload.get("result", {})
            if error := result.get("error"):
                await terminal.write(
                    prefix + "✗ " + str(error.get("message", error))[:700], style="red"
                )
            elif self.verbose and result.get("artifact_id"):
                await terminal.write("  artifact: " + result["artifact_id"], style="dim")
        elif kind == "coding_command":
            await terminal.write(
                prefix + ("✓ " if payload.get("passed") else "✗ ") + command_summary(payload),
                style="green" if payload.get("passed") else "red",
            )
        elif kind == "subagent_created":
            self.names[payload["child_id"]] = payload.get("name") or payload["child_id"][:8]
            await terminal.write(
                "● Child started: " + self.names[payload["child_id"]], style="cyan"
            )
        elif kind == "verifier_started":
            await terminal.write(prefix + "● Independent verification", style="cyan")
        elif kind == "verifier_result":
            await terminal.write(
                prefix
                + (
                    "✓ Verification passed"
                    if payload.get("passed")
                    else "✗ Verification did not pass"
                )
            )
        elif kind in {"completion", "conversation_completed"}:
            await terminal.write(prefix + str(payload.get("result") or "Completed"), markdown=True)
        elif kind in {"failure", "termination"}:
            await terminal.write(
                prefix + "✗ " + str(payload.get("message") or payload.get("result") or kind)[:1000],
                style="red",
            )
        elif kind in {"interruption", "paused"}:
            # Flush any partial model text; keep it labelled as interrupted.
            for key, (name, _) in list(terminal.drafts.items()):
                if name == label:
                    await terminal.response(key, label + " (interrupted)", "")
            await terminal.write(prefix + "Paused; persistent state retained.", style="yellow")
        elif kind == "context_compaction":
            await terminal.write(prefix + "● Context compacted; full history retained", style="dim")
        elif kind == "environment_prepare":
            description = (
                "Preparing configured coding environment and baseline"
                if payload.get("adapter") == "coding"
                else "Opening environment"
            )
            await terminal.write(prefix + "● " + description, style="cyan")
        elif kind == "coding_baseline":
            await terminal.write(prefix + "✓ Repository baseline ready", style="dim")
        elif kind == "retry":
            await terminal.write(
                prefix + "● Retrying recoverable provider/tool failure", style="yellow"
            )
        elif self.verbose:
            await terminal.write(prefix + "· " + kind, style="dim")
