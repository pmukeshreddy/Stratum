"""Reproducible build of a thin bridge against pinned official Codex Rust libraries.

The source and build products live outside repositories. No credentials are copied.
Rust is a build-time prerequisite; an installed client needs only the resulting binary.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CODEX_REVISION = "3d2ee51ca2d5db578f328aa75e20aa22c0197c9a"
CODEX_VERSION = "0.153.4"
CODEX_SOURCE = "https://github.com/openai/codex.git"


def source():
    return (Path(__file__).parent / "native" / "inference.rs").read_text()


def client_directory():
    fingerprint = hashlib.sha256(source().encode()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "threadweave" / f"codex-{CODEX_REVISION[:12]}-{fingerprint}"


def client_path():
    return client_directory() / "threadweave-inference"


def install_client(*, checkout=None):
    """Build once, with a cross-process lock. `checkout` is an optional pinned local clone."""
    target = client_path()
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    with (directory / "install.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_file():
            return target
        if not shutil.which("cargo") or not shutil.which("git"):
            raise RuntimeError("Building the Codex inference client requires Rust/cargo and git")
        build = Path(checkout).resolve() if checkout else directory / "source"
        with (directory / "build.log").open("ab") as log:

            def run(argv, *, cwd=None):
                result = subprocess.run(argv, cwd=cwd, stdout=log, stderr=log, check=False)
                if result.returncode:
                    raise RuntimeError(
                        f"Codex inference client build failed; inspect {directory / 'build.log'}"
                    )

            if not build.exists():
                run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        f"rust-v{CODEX_VERSION}",
                        CODEX_SOURCE,
                        str(build),
                    ]
                )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=build, text=True
            ).strip()
            if revision != CODEX_REVISION:
                raise ValueError("Codex source revision does not match the pinned client revision")
            package = build / "codex-rs" / "model-provider"
            manifest = package / "Cargo.toml"
            text = manifest.read_text()
            if 'name = "threadweave-inference"' not in text:
                text = text.replace(
                    "[dependencies]\n",
                    "[dependencies]\nfutures = { workspace = true }\nserde_json = { workspace = true }\n",
                    1,
                )
                text += '\n[[bin]]\nname = "threadweave-inference"\npath = "src/bin/threadweave_inference.rs"\n'
                manifest.write_text(text)
            entry = package / "src" / "bin" / "threadweave_inference.rs"
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text(source())
            run(
                ["cargo", "build", "-p", "codex-model-provider", "--bin", "threadweave-inference"],
                cwd=build / "codex-rs",
            )
            compiled = build / "codex-rs" / "target" / "debug" / "threadweave-inference"
            fd, temporary = tempfile.mkstemp(dir=directory, prefix="client-")
            os.close(fd)
            shutil.copyfile(compiled, temporary)
            Path(temporary).chmod(0o700)
            os.replace(temporary, target)
    return target
