"""Reproducible official source fetch and host-only controller key setup. No inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .emulatorbench import pins
from .schema import save, timestamp


def apply_runtime_patches():
    """Apply only the hash-pinned provider compatibility delta in this interpreter."""
    import verifiers

    resources = Path(__file__).resolve().parent
    package = Path(verifiers.__file__).parent
    for patch in pins()["verifiers"].get("patches", []):
        diff = (resources / patch["patch"]).read_bytes()
        if hashlib.sha256(diff).hexdigest() != patch["patch_sha256"]:
            raise ValueError("Verifiers compatibility patch changed")
        destination = package / patch["file"]
        original = destination.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        if digest == patch["patched_sha256"]:
            continue
        if digest != patch["original_sha256"]:
            raise ValueError("Verifiers compatibility patch has an unexpected baseline")
        with tempfile.TemporaryDirectory(prefix="buffalo-emulatorbench-patch-") as temporary:
            root = Path(temporary).resolve()
            target = root / "verifiers" / patch["file"]
            target.parent.mkdir(parents=True)
            target.write_bytes(original)
            subprocess.run(
                ["git", "apply", "--unsafe-paths", "--directory", str(root)], input=diff, check=True
            )
            updated = target.read_bytes()
        if hashlib.sha256(updated).hexdigest() != patch["patched_sha256"]:
            raise ValueError("Verifiers compatibility patch produced unexpected bytes")
        staged = destination.with_name(destination.name + ".emulatorbench-new")
        staged.write_bytes(updated)
        # Do not mutate a uv cache hardlink shared with another interpreter.
        os.replace(staged, destination)


def controller_environment(directory):
    """Local publication keys are distinct from the benchmark author's corpus keys."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    directory = Path(directory).absolute()
    if directory.is_symlink():
        raise ValueError("Controller directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    private = directory / "publication-private.pem"
    public = directory / "publication-public.pem"
    signer = directory / "signer"
    if any(p.is_symlink() for p in (private, public, signer)):
        raise ValueError("Controller material cannot be a symlink")
    if not private.exists():
        key = Ed25519PrivateKey.generate()
        data = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        with private.open("xb") as stream:
            os.chmod(private, 0o600)
            stream.write(data)
    else:
        if private.is_symlink() or private.stat().st_mode & 0o077:
            raise ValueError("Unsafe controller private key permissions")
        key = serialization.load_pem_private_key(private.read_bytes(), password=None)
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    public.chmod(0o600)
    signer.write_text(
        f"#!{sys.executable}\n"
        "import base64, pathlib, sys\n"
        "from cryptography.hazmat.primitives.serialization import load_pem_private_key\n"
        "key = load_pem_private_key(pathlib.Path(__file__).with_name('publication-private.pem').read_bytes(), password=None)\n"
        "sys.stdout.write(base64.b64encode(key.sign(sys.stdin.buffer.read())).decode('ascii'))\n"
    )
    signer.chmod(0o700)
    return {
        "EMULATORBENCH_CONTROLLER_SIGNER": str(signer),
        "EMULATORBENCH_CONTROLLER_PUBLIC_KEY": str(public),
    }


def configure_controller(config):
    directory = config.get("controller_directory")
    if directory and not os.environ.get("EMULATORBENCH_CONTROLLER_SIGNER"):
        public = Path(directory).resolve() / "publication-public.pem"
        signer = public.with_name("signer")
        if signer.is_file() and public.is_file():
            os.environ["EMULATORBENCH_CONTROLLER_SIGNER"] = str(signer)
            os.environ["EMULATORBENCH_CONTROLLER_PUBLIC_KEY"] = str(public)


def fetch_public_sources(source, cache, receipts):
    """Run the release's downloader unchanged, in the already locked selection order."""
    receipts.mkdir(parents=True, exist_ok=True)
    script = source / pins()["benchmark"]["package"] / "scripts/prefetch_public_sources.py"
    if (
        hashlib.sha256(script.read_bytes()).hexdigest()
        != pins()["benchmark"]["setup_script_sha256"]
    ):
        raise ValueError("Official public-source downloader differs from the pinned release")
    from .emulatorbench import discover_tasks

    commands = []
    for task in discover_tasks()[:4]:
        command = [
            sys.executable,
            str(script),
            "--platform",
            task.manifest.slug,
            "--cache-root",
            str(cache.resolve()),
            "--receipt",
            str((receipts / f"{task.manifest.slug}.json").resolve()),
        ]
        subprocess.run(command, check=True, timeout=1800)
        commands.append(command)
    save(
        receipts / "setup.json",
        {
            "timestamp": timestamp(),
            "source_repository": pins()["benchmark"]["repo"],
            "source_commit": pins()["benchmark"]["commit"],
            "commands": commands,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "data_kind": "official public source cache; NOT authenticated runner-v2 corpora",
        },
    )


def build_test_image(tag, directory):
    """Separate test image; uses the very same production dependency installer."""
    from .autonomous_host import REMOTE_ROOT, agent_source_archive, worker_install_script

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "source.tar.gz").write_bytes(agent_source_archive())
    (directory / "install.sh").write_text(worker_install_script())
    (directory / "Dockerfile").write_text(
        "FROM rust:1.85-bookworm@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4\n"
        f"COPY source.tar.gz {REMOTE_ROOT}/source.tar.gz\n"
        "COPY install.sh /tmp/install-buffalo.sh\n"
        "RUN bash /tmp/install-buffalo.sh\n"
        f"ENV PATH={REMOTE_ROOT}/node/bin:{REMOTE_ROOT}/venv/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "WORKDIR /workspace\n"
    )
    command = ["docker", "build", "--tag", tag, str(directory)]
    subprocess.run(command, check=True, timeout=1800)
    metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))
    save(
        directory / "build-receipt.json",
        {
            "command": command,
            "timestamp": timestamp(),
            "image": metadata,
            "installer_sha256": hashlib.sha256((directory / "install.sh").read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256((directory / "source.tar.gz").read_bytes()).hexdigest(),
            "purpose": "Scripted tests only; never accepted by official trusted controller",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".emulatorbench/sources/prime-envs"))
    parser.add_argument("--cache", type=Path, default=Path(".emulatorbench/public-sources"))
    parser.add_argument("--receipts", type=Path, default=Path(".emulatorbench/source-receipts"))
    parser.add_argument(
        "--controller-directory", type=Path, default=Path(".emulatorbench/controller")
    )
    parser.add_argument("--build-test-image", metavar="TAG")
    parser.add_argument(
        "--build-directory", type=Path, default=Path(".emulatorbench/test-image-build")
    )
    args = parser.parse_args()
    apply_runtime_patches()
    if args.build_test_image:
        build_test_image(args.build_test_image, args.build_directory)
        return
    actual = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != pins()["benchmark"]["commit"]:
        raise ValueError("Refusing setup from an unpinned benchmark checkout")
    fetch_public_sources(args.source, args.cache, args.receipts)
    environment = controller_environment(args.controller_directory)
    save(
        args.receipts / "controller.json",
        {
            "environment": environment,
            "public_key_sha256": hashlib.sha256(
                Path(environment["EMULATORBENCH_CONTROLLER_PUBLIC_KEY"]).read_bytes()
            ).hexdigest(),
            "purpose": "Host publication signatures only. Does not authorize, create, or replace any benchmark corpus.",
        },
    )
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
