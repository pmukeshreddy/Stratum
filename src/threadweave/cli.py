from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .configuration import doctor, load_config
from .daemon import request
from .models import HarnessError


def show(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


async def ensure_daemon(directory, *, concurrency=8):
    try:
        return await request(directory, "ping")
    except (FileNotFoundError, ConnectionRefusedError, ConnectionError):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / "daemon.log").open("ab") as log:
            process = subprocess.Popen(  # noqa: ASYNC220 - detached daemon outlives this event loop
                [
                    sys.executable,
                    "-m",
                    "threadweave.daemon",
                    "--data",
                    str(directory),
                    "--concurrency",
                    str(concurrency),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        for _ in range(100):
            await asyncio.sleep(0.05)
            try:
                return await request(directory, "ping")
            except (FileNotFoundError, ConnectionRefusedError, ConnectionError):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Daemon failed to start; inspect {directory / 'daemon.log'}"
                    ) from None
        raise RuntimeError(
            f"Timed out starting daemon; inspect {directory / 'daemon.log'}"
        ) from None


async def attach(directory, sid, *, follow=True, after=0):
    cursor = after
    while True:
        history = await request(directory, "history", session_id=sid, after=cursor, limit=50)
        for event in history:
            cursor = event["seq"]
            print(json.dumps(event, ensure_ascii=False), flush=True)
        status = await request(directory, "status", session_id=sid)
        if not follow or status["session"]["outcome"] != "active":
            show(status)
            return 0 if status["session"]["outcome"] in ("active", "completed") else 1
        await asyncio.sleep(0.3)


async def execute(args):
    command = args.command or "chat"
    if command == "chat":
        from .chat import chat

        return await chat(args)
    directory = (args.data or Path.cwd() / ".threadweave").resolve()
    if command == "auth":
        from .codex_auth import CodexControl
        from .native_client import client_path, install_client

        if args.action == "install-client":
            print(
                "Building pinned official Codex client libraries; logs stay in the user cache.",
                file=sys.stderr,
            )
            show({"client": str(await asyncio.to_thread(install_client)), "installed": True})
            return 0
        async with CodexControl() as control:
            if args.action == "login":
                import webbrowser

                def present(reply):
                    url = reply.get("authUrl") or reply.get("verificationUrl")
                    if url:
                        print(f"Authorize with ChatGPT: {url}", flush=True)
                        if not args.device:
                            webbrowser.open(url)
                    if reply.get("userCode"):
                        print(f"Device code: {reply['userCode']}", flush=True)

                result = await control.login(present, device=args.device)
            elif args.action == "logout":
                result = await control.logout()
                result["notice"] = "Signed out of the shared Codex credential store"
            elif args.action == "models":
                show(await control.models())
                return 0
            else:
                result = await control.status()
            result["client_installed"] = client_path().is_file()
            result["usable"] = result["logged_in"] and result["client_installed"]
            show(result)
            return 0 if result["logged_in"] or args.action == "logout" else 1
    if command == "daemon":
        if args.action == "start":
            show(await ensure_daemon(directory, concurrency=args.concurrency))
        else:
            show(await request(directory, "shutdown" if args.action == "stop" else "ping"))
        return 0
    if command == "doctor":
        config = load_config(args.config) if args.config else None
        status = None
        if config and any(
            p.name == "codex_subscription" for p in [config.provider, *config.models.values()]
        ):
            from .codex_auth import CodexControl
            from .native_client import client_path
            from .subscription import SubscriptionProvider

            try:
                async with CodexControl() as control:
                    status = await control.status()
                provider = next(
                    p
                    for p in [config.provider, *config.models.values()]
                    if p.name == "codex_subscription"
                )
                resolved, _ = await SubscriptionProvider().resolve(provider)
                version = await asyncio.create_subprocess_exec(
                    "codex",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await version.communicate()
                status.update(
                    codex_version=stdout.decode().strip(),
                    model=resolved.model,
                    parameters=resolved.parameters,
                    transport="official_codex_responses_client",
                    client_installed=client_path().is_file(),
                    usable=status["logged_in"] and client_path().is_file(),
                    output_limit="client-observed byte guard; no server token cap",
                )
                if not status["client_installed"]:
                    status["issue"] = "Run threadweave auth install-client"
            except HarnessError as exc:
                status = {"usable": False, "issue": exc.failure.message, "code": exc.failure.code}
        result = await asyncio.to_thread(doctor, directory, config, subscription_status=status)
        show(result)
        return 0 if result["ok"] else 1
    if command == "eval":
        from .evals.runner import execute as evaluate

        return await evaluate(args)
    if command == "run":
        config = await resolved_config(load_config(args.config, model_override=args.model))
        await ensure_daemon(directory)
        session = await request(
            directory,
            "create",
            instruction=args.instruction,
            workspace=str(args.workspace.resolve()),
            config=config.model_dump(mode="json"),
            name=args.name,
            mode=args.mode,
        )
        if args.attach:
            print(
                f"Session {session['id']}; Ctrl-C detaches and leaves it running.", file=sys.stderr
            )
            return await attach(directory, session["id"])
        print(session["id"])
        return 0
    if command == "attach":
        return await attach(directory, args.session_id, after=args.after)
    if command == "list":
        show(await request(directory, "list"))
    elif command in (
        "status",
        "usage",
        "tree",
        "config",
        "states",
        "schedules",
        "pause",
        "resume",
        "compact",
        "diff",
        "experiments",
        "verify",
    ):
        result = await request(
            directory, "status" if command == "usage" else command, session_id=args.session_id
        )
        show(result["tree_usage"] if command == "usage" else result)
    elif command == "history":
        show(
            await request(
                directory,
                command,
                session_id=args.session_id,
                limit=args.limit,
                after=args.after,
                tree=args.tree,
                kind=args.kind,
            )
        )
    elif command == "input":
        show(await request(directory, "input", session_id=args.session_id, body=args.body))
    elif command == "stop":
        show(await request(directory, "stop", session_id=args.session_id, tree=not args.only))
    elif command == "fork":
        show(await request(directory, "fork", session_id=args.session_id, name=args.name))
    elif command == "schedule":
        show(
            await request(
                directory,
                "schedule",
                sid=args.session_id,
                interval_seconds=args.interval,
                cron=args.cron,
                instruction=args.instruction,
            )
        )
    elif command == "unschedule":
        show(await request(directory, "unschedule", schedule_id=args.schedule_id))
    elif command == "refine":
        show(
            await request(
                directory,
                "refine",
                session_id=args.session_id,
                instructions=args.instructions,
                global_=args.global_,
                rollback_id=args.rollback,
            )
        )
    elif command == "artifact":
        show(
            await request(
                directory,
                "artifact",
                session_id=args.session_id,
                artifact_id=args.artifact_id,
                offset=args.offset,
                limit=args.limit,
            )
        )
    return 0


async def resolved_config(config):
    """Pin discovered model/effort in the run snapshot before daemon admission."""
    from .subscription import SubscriptionProvider

    configs = list(config.models.values()) + (
        [config.provider] if not config.routing.default else []
    )
    for provider in configs:
        if provider.name == "codex_subscription":
            resolved, _ = await SubscriptionProvider().resolve(provider)
            provider.model, provider.parameters = resolved.model, resolved.parameters
    return config


def parser():
    p = argparse.ArgumentParser(description="Persistent coding agents for real repositories")
    p.add_argument(
        "--data", type=Path, help="Durable data directory (chat defaults to user data storage)"
    )
    chat_arguments(p)
    sub = p.add_subparsers(dest="command")
    interactive = sub.add_parser("chat", help="Talk continuously in one persistent coding session")
    chat_arguments(interactive, suppress=True)
    interactive.add_argument("--data", type=Path, default=argparse.SUPPRESS)
    auth = sub.add_parser(
        "auth", help="Manage the shared official Codex ChatGPT login; logout also signs Codex out"
    )
    auth.add_argument("action", choices=["status", "login", "logout", "models", "install-client"])
    auth.add_argument(
        "--device", action="store_true", help="Use Codex's official device-code login flow"
    )
    daemon = sub.add_parser("daemon", help="Control the detached local daemon")
    daemon.add_argument("action", choices=["start", "status", "stop"])
    daemon.add_argument("--concurrency", type=int, default=8)
    run = sub.add_parser("run", help="Create a task; prints its stable session ID")
    run.add_argument("instruction")
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--model")
    run.add_argument("--name", default="root")
    run.add_argument("--mode", choices=["autonomous", "goal", "heartbeat"], default="autonomous")
    run.add_argument("--attach", action="store_true")

    sub.add_parser("list", help="List root sessions")
    for name in (
        "status",
        "usage",
        "tree",
        "config",
        "states",
        "schedules",
        "pause",
        "resume",
        "compact",
        "diff",
        "experiments",
        "verify",
    ):
        command = sub.add_parser(name)
        command.add_argument("session_id")
    attach_parser = sub.add_parser(
        "attach", help="Follow events; Ctrl-C detaches without stopping work"
    )
    attach_parser.add_argument("session_id")
    attach_parser.add_argument("--after", type=int, default=0)
    history = sub.add_parser("history")
    history.add_argument("session_id")
    history.add_argument("--limit", type=int, default=30)
    history.add_argument("--after", type=int, default=0)
    history.add_argument("--kind")
    history.add_argument("--tree", action="store_true")
    human = sub.add_parser(
        "input", help="Send durable human input; resume terminal sessions explicitly"
    )
    human.add_argument("session_id")
    human.add_argument("body")
    stop = sub.add_parser("stop")
    stop.add_argument("session_id")
    stop.add_argument(
        "--only", action="store_true", help="Stop this session but leave children running"
    )
    fork = sub.add_parser("fork")
    fork.add_argument("session_id")
    fork.add_argument("--name")
    schedule = sub.add_parser("schedule")
    schedule.add_argument("session_id")
    choice = schedule.add_mutually_exclusive_group(required=True)
    choice.add_argument("--interval", type=float)
    choice.add_argument("--cron")
    schedule.add_argument("--instruction", default="Scheduled continuation")
    unschedule = sub.add_parser("unschedule")
    unschedule.add_argument("schedule_id")
    refine = sub.add_parser("refine", help="Schedule continual harness refinement")
    refine.add_argument("session_id")
    refine.add_argument("instructions", nargs="?")
    refine.add_argument("--global", dest="global_", action="store_true")
    refine.add_argument("--rollback", help="Refinement ID to undo")
    artifact = sub.add_parser("artifact")
    artifact.add_argument("session_id")
    artifact.add_argument("artifact_id")
    artifact.add_argument("--offset", type=int, default=0)
    artifact.add_argument("--limit", type=int, default=16000)
    diagnostics = sub.add_parser("doctor", help="Check provider, credentials, tools and storage")
    diagnostics.add_argument("--config", type=Path)
    from .evals.runner import arguments

    evaluation = sub.add_parser("eval", help="Run Buffalo on the official ARC-AGI-3 environments")
    arguments(evaluation)
    return p


def chat_arguments(parser, *, suppress=False):
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--workspace", type=Path, default=default, help="Repository (default: cwd)")
    parser.add_argument("--config", type=Path, default=default, help="Run config (auto-discovered)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--resume", dest="resume_id", default=default, help="Reopen a root session"
    )
    selection.add_argument(
        "--continue",
        dest="continue_recent",
        action="store_true",
        default=default,
        help="Reopen the most recent root in this workspace",
    )
    parser.add_argument(
        "--json", action="store_true", default=default, help="Emit JSON debug events"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=default, help="Show extra activity"
    )


def main():
    args = parser().parse_args()
    try:
        code = asyncio.run(execute(args))
    except KeyboardInterrupt:
        if args.command == "eval":
            print("Evaluation interrupted. Partial artifacts are preserved.", file=sys.stderr)
            code = 130
        else:
            print("Detached. The daemon and session continue running.", file=sys.stderr)
            code = 0
    except (OSError, ValueError, RuntimeError, KeyError, HarnessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
