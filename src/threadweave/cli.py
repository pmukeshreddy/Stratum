from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .configuration import doctor, load_config
from .daemon import request


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
    directory = args.data.resolve()
    command = args.command
    if command == "daemon":
        if args.action == "start":
            show(await ensure_daemon(directory, concurrency=args.concurrency))
        else:
            show(await request(directory, "shutdown" if args.action == "stop" else "ping"))
        return 0
    if command == "doctor":
        result = doctor(directory, load_config(args.config) if args.config else None)
        show(result)
        return 0 if result["ok"] else 1
    if command == "eval":
        from .evaluation import evaluate

        show(
            await evaluate(
                args.tasks,
                load_config(args.config),
                directory,
                repetitions=args.repetitions,
                seed=args.seed,
                output=args.output,
            )
        )
        return 0
    if command == "analyze":
        from .evaluation import analyze

        show(analyze(args.results))
        return 0
    if command == "run":
        config = load_config(args.config, model_override=args.model)
        config.task.adapter = "coding"
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
        if args.edit is None:
            show(await request(directory, "input", session_id=args.session_id, body="/refine"))
            return 0
        show(
            await request(
                directory,
                "refine",
                session_id=args.session_id,
                edit=json.loads(args.edit.read_text()),
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


def parser():
    p = argparse.ArgumentParser(description="Persistent coding agents for real repositories")
    p.add_argument(
        "--data", type=Path, default=Path.cwd() / ".threadweave", help="Durable data directory"
    )
    sub = p.add_subparsers(dest="command", required=True)
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
    refine = sub.add_parser("refine", help="Queue an auditable StateEdit JSON document")
    refine.add_argument("session_id")
    refine.add_argument(
        "edit",
        type=Path,
        nargs="?",
        help="StateEdit JSON; omit to request a model refinement at the next turn boundary",
    )
    artifact = sub.add_parser("artifact")
    artifact.add_argument("session_id")
    artifact.add_argument("artifact_id")
    artifact.add_argument("--offset", type=int, default=0)
    artifact.add_argument("--limit", type=int, default=16000)
    diagnostics = sub.add_parser("doctor", help="Check provider, credentials, tools and storage")
    diagnostics.add_argument("--config", type=Path)
    evaluation = sub.add_parser("eval", help="Run externally supplied real coding tasks")
    evaluation.add_argument("tasks", type=Path)
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--repetitions", type=int, default=1)
    evaluation.add_argument("--seed", type=int, default=0)
    analysis = sub.add_parser("analyze", help="Aggregate measured evaluation results")
    analysis.add_argument("results", type=Path)
    return p


def main():
    args = parser().parse_args()
    try:
        code = asyncio.run(execute(args))
    except KeyboardInterrupt:
        print("Detached. The daemon and session continue running.", file=sys.stderr)
        code = 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
