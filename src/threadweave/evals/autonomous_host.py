"""Host transport and evidence journal for an isolated resident Buffalo Runtime."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import io
import json
import os
import shlex
import tarfile
import time
import uuid
from pathlib import Path

from ..models import HarnessError, ModelRequest, ProviderConfig
from ..providers import default_providers
from .evocode_rpc import Peer
from .schema import save, timestamp

REMOTE_ROOT = "/opt/buffalo-evaluation"
REMOTE_PYTHON = f"{REMOTE_ROOT}/venv/bin/python"
REMOTE_STATE = f"{REMOTE_ROOT}/state"
REMOTE_MAILBOX = f"{REMOTE_ROOT}/mailbox"
REMOTE_PATH = f"{REMOTE_ROOT}/node/bin:{REMOTE_ROOT}/venv/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class EvaluationFailure(RuntimeError):
    def __init__(self, category, reason):
        super().__init__(reason)
        self.category = category


class EvidenceJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        os.chmod(self.directory, 0o700)
        self.events, self.calls, self.attempts, self.rehearsals = [], [], [], []
        self.event_ids = set()
        self.event_by_id = {}
        self.progress = {}
        for name in (
            "raw-events",
            "model-calls",
            "verifier-attempts",
            "rehearsal-attempts",
            "rlm-events",
            "refinement-events",
        ):
            (self.directory / f"{name}.jsonl").touch(mode=0o600)

    def append(self, name, rows):
        if not rows:
            return
        with (self.directory / f"{name}.jsonl").open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def evidence(self, events):
        fresh = []
        for event in events:
            previous = self.event_by_id.get(event["id"])
            if previous is not None and previous != event:
                raise EvaluationFailure(
                    "ADAPTER_FAILURE", f"Conflicting raw event ID: {event['id']}"
                )
            if previous is None:
                fresh.append(event)
                self.event_by_id[event["id"]] = event
        self.append("raw-events", fresh)
        self.events.extend(fresh)
        self.event_ids.update(e["id"] for e in fresh)
        self.append(
            "rlm-events",
            [
                e
                for e in fresh
                if (
                    e["type"].startswith(("rlm_", "subagent_", "agent_message", "child_"))
                    or e["type"] in {"completion", "termination", "evaluation_file_observation"}
                )
            ],
        )
        self.append(
            "refinement-events",
            [
                e
                for e in fresh
                if (
                    "refine" in e["type"]
                    or "harness" in e["type"]
                    or e["type"]
                    in {"context_compaction", "skill_loaded", "execution_input_consumed"}
                )
            ],
        )

    def status(self, **updates):
        self.progress.update(updates)
        save(self.directory / "live.json", {"updated": timestamp(), **self.progress})


def agent_source_archive():
    """Never install benchmark modules, pins, tests, host state, or graders in a candidate."""
    package = Path(__file__).resolve().parents[1]
    allowed_evals = {
        "__init__.py",
        "autonomous_rehearsal.py",
        "autonomous_worker.py",
        "evocode_worker.py",
        "evocode_rpc.py",
        "schema.py",
    }
    payload = io.BytesIO()

    def normalized(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        return info

    with (
        gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(package.rglob("*")):
            relative = path.relative_to(package)
            if not path.is_file() or "__pycache__" in relative.parts:
                continue
            if relative.parts[0] == "evals" and (
                len(relative.parts) != 2 or relative.name not in allowed_evals
            ):
                continue
            archive.add(
                path,
                arcname=str(Path("src/threadweave") / relative),
                recursive=False,
                filter=normalized,
            )
        for name in ("pyproject.toml", "uv.lock"):
            archive.add(package.parents[1] / name, arcname=name, filter=normalized)
    return payload.getvalue()


async def checked_run(runtime, argv, *, timeout_seconds=60):
    async with asyncio.timeout(timeout_seconds):
        result = await runtime.run(argv, {})
    if result.exit_code != 0:
        raise EvaluationFailure(
            "INFRASTRUCTURE_FAILURE",
            f"Environment command failed: {argv[0]} ({result.exit_code}): {result.stderr[-1000:]}",
        )
    return result


def worker_install_script():
    # The exact project lock resolves Buffalo's production dependencies. Benchmark
    # packages are host-only; the candidate receives neither them nor a host mount.
    return f"""set -eu
mkdir -p {REMOTE_ROOT}/source {REMOTE_ROOT}/uv
tar -xzf {REMOTE_ROOT}/source.tar.gz -C {REMOTE_ROOT}/source
curl -fsSL https://github.com/astral-sh/uv/releases/download/0.11.8/uv-$(uname -m)-unknown-linux-gnu.tar.gz -o {REMOTE_ROOT}/uv.tar.gz
tar -xzf {REMOTE_ROOT}/uv.tar.gz --strip-components=1 -C {REMOTE_ROOT}/uv
case $(uname -m) in
  x86_64) node_arch=x64; node_sha=fb870226119d47378fa9c92c4535389c72dae14fcc7b47e6fdcc82c43de5a547 ;;
  aarch64) node_arch=arm64; node_sha=1725602e9fb150eb8b8220a899085190e1c04d1a5f3862b01c3dc1dfce0157f9 ;;
  *) exit 64 ;;
esac
curl -fsSL https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-$node_arch.tar.gz -o {REMOTE_ROOT}/node.tar.gz
echo "$node_sha  {REMOTE_ROOT}/node.tar.gz" | sha256sum --check --status
mkdir -p {REMOTE_ROOT}/node
tar -xzf {REMOTE_ROOT}/node.tar.gz --strip-components=1 -C {REMOTE_ROOT}/node
cd {REMOTE_ROOT}/source
UV_PYTHON_INSTALL_DIR={REMOTE_ROOT}/python UV_PROJECT_ENVIRONMENT={REMOTE_ROOT}/venv {REMOTE_ROOT}/uv/uv sync --frozen --no-dev --python 3.12.12
"""


async def install_worker(runtime, journal):
    await runtime.write(f"{REMOTE_ROOT}/source.tar.gz", agent_source_archive())
    script = worker_install_script()
    result = await checked_run(
        runtime, ["bash", "--noprofile", "--norc", "-c", script], timeout_seconds=900
    )
    save(
        journal.directory / "worker-install.json",
        {
            "command": script,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    versions = await checked_run(
        runtime,
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            f"{REMOTE_ROOT}/uv/uv pip freeze --python {REMOTE_PYTHON}; {REMOTE_ROOT}/uv/uv --version; {REMOTE_PYTHON} --version; {REMOTE_ROOT}/node/bin/node --version; sha256sum {REMOTE_ROOT}/uv.tar.gz {REMOTE_ROOT}/node.tar.gz {REMOTE_ROOT}/source.tar.gz; rustc --version; cargo --version; uname -a",
        ],
    )
    (journal.directory / "candidate-environment.txt").write_text(versions.stdout)


class MailboxTransport:
    """The same duplex Peer protocol over provider read/write/run primitives.

    Only protocol bytes cross the boundary. No host directory or listening host
    service is exposed to the candidate. Each write has a unique staging path.
    """

    def __init__(self, runtime, handler, *, python=REMOTE_PYTHON):
        self.runtime, self.python = runtime, python
        self.reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
        self.offset = 0
        self.peer = Peer(self.reader, self.send, handler)
        self.poller = None

    async def start(self):
        await self.runtime.write(f"{REMOTE_MAILBOX}/in.jsonl", b"")
        await self.runtime.write(f"{REMOTE_MAILBOX}/out.jsonl", b"")
        await self.runtime.run_background(
            [
                self.python,
                "-u",
                "-m",
                "threadweave.evals.autonomous_worker",
                "--mailbox",
                REMOTE_MAILBOX,
            ],
            {"PATH": REMOTE_PATH},
            f"{REMOTE_ROOT}/worker.log",
        )
        self.poller = asyncio.create_task(self.poll())

    async def send(self, data):
        path = f"{REMOTE_MAILBOX}/pending-{uuid.uuid4().hex}"
        await self.runtime.write(path, data)
        await checked_run(
            self.runtime,
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                f"cat {shlex.quote(path)} >> {REMOTE_MAILBOX}/in.jsonl && rm {shlex.quote(path)}",
            ],
        )

    async def poll(self):
        try:
            while True:
                script = (
                    "import base64,time;"
                    f"f=open({REMOTE_MAILBOX + '/out.jsonl'!r},'rb');f.seek({self.offset});"
                    "data=f.read(1048576);print(base64.b64encode(data).decode())"
                )
                result = await checked_run(self.runtime, [self.python, "-c", script])
                data = base64.b64decode(result.stdout.strip(), validate=True)
                if data:
                    self.offset += len(data)
                    self.reader.feed_data(data)
                else:
                    await asyncio.sleep(0.25)
        except BaseException as exc:
            self.reader.set_exception(
                ConnectionError(f"Resident worker mailbox failed: {type(exc).__name__}")
            )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def close(self):
        if self.poller:
            self.poller.cancel()
            await asyncio.gather(self.poller, return_exceptions=True)
        await self.peer.close()


class ResidentClient:
    def __init__(self, config, journal, gate, *, providers=None):
        self.config, self.journal, self.gate = config, journal, gate
        self.providers = providers if providers is not None else default_providers()
        self.failure = None
        self.transport = self.peer = None

    def provider(self, config):
        allowed = [self.config.provider, *self.config.models.values()]
        if not any(
            (config.name, config.base_url, config.api_key_env)
            == (p.name, p.base_url, p.api_key_env)
            for p in allowed
        ):
            raise EvaluationFailure("ADAPTER_FAILURE", "Unconfigured provider endpoint requested")
        return self.providers[config.name]

    async def handle(self, method, args):
        if method == "evidence":
            self.journal.evidence(args["events"])
            return {"persisted": len(args["events"])}
        if method == "progress":
            self.journal.status(**args["status"])
            return {}
        if method in {"gate", "rehearse"}:
            try:
                operation = self.gate if method == "gate" else self.gate.rehearse
                return await operation(**args)
            except EvaluationFailure as exc:
                if method == "rehearse":
                    return {
                        "exit_code": 2,
                        "status": exc.category,
                        "error": "Public rehearsal unavailable; the host recorded the failure.",
                    }
                self.failure = exc
                raise HarnessError(
                    "environment", exc.category, "Host verification unavailable; see host audit"
                ) from None
        if method == "models":
            async with self.providers["codex_subscription"].control_factory() as control:
                return await control.models()
        if method == "resolve":
            config = ProviderConfig.model_validate(args["config"])
            try:
                provider = self.provider(config)
                details = {}
                if hasattr(provider, "resolve"):
                    config, details = await provider.resolve(
                        config, **({"reasoning_off": True} if args["reasoning_off"] else {})
                    )
            except Exception as exc:
                self.failure = EvaluationFailure(
                    "PROVIDER_FAILURE", f"Provider resolution failed: {exc}"
                )
                self.journal.append(
                    "provider-errors",
                    [{"timestamp": timestamp(), "operation": "resolve", "error": str(exc)}],
                )
                raise
            return {"config": config.model_dump(mode="json"), "details": details}
        if method != "model":
            raise ValueError(f"Unknown host operation: {method}")
        request = ModelRequest.model_validate(args["request"])
        row = {
            "id": request.request_id,
            "request": request.public_dump(),
            "started": timestamp(),
            "started_monotonic": time.monotonic(),
        }
        self.journal.append("model-call-starts", [row])

        async def emit(delta):
            pass

        try:
            response = await self.provider(request.config).invoke(request, emit)
            row["response"] = response.model_dump(mode="json")
            return {**row["response"], "provider_items": response.provider_items}
        except asyncio.CancelledError:
            row["cancelled"] = True
            raise
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.failure = EvaluationFailure("PROVIDER_FAILURE", str(exc))
            raise HarnessError("provider", "evaluation_provider_failure", str(exc)) from exc
        finally:
            row["ended"] = timestamp()
            row["ended_monotonic"] = time.monotonic()
            self.journal.calls.append(row)
            self.journal.append("model-calls", [row])

    async def start(self, runtime, instruction, *, install=True, python=REMOTE_PYTHON):
        if install:
            await install_worker(runtime, self.journal)
        self.transport = MailboxTransport(runtime, self.handle, python=python)
        self.peer = self.transport.peer
        await self.transport.start()
        async with asyncio.timeout(120):
            audit = await self.peer.call(
                "setup",
                instruction=instruction,
                workspace="/workspace",
                directory=REMOTE_STATE,
                config=self.config.model_dump(mode="json"),
            )
        save(self.journal.directory / "setup-audit.json", audit)
        return audit

    async def close(self):
        if self.peer:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(15):
                    await self.peer.call("shutdown")
        if self.transport:
            await self.transport.close()
