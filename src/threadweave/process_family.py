"""Birth-identified ownership for trusted Python subprocesses, not containment.

Popen registrations are immediate. A marker scan recovers exec'd descendants that
detach/reparent. Native forks that discard the marker before observation remain
an explicit limitation outside a container/cgroup. Never match command strings.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import psutil

from .artifacts import atomic_write

MARKER = "THREADWEAVE_PROCESS_OWNER"


def cleanup_registry(directory):
    path = directory / "process-family.json"
    if not path.is_file():
        return
    try:
        rows = json.loads(path.read_text()).get("processes", [])
    except (ValueError, OSError):
        return
    for row in rows:
        with contextlib.suppress(psutil.Error):
            process = psutil.Process(row["pid"])
            if process.create_time() == row["birth"] and process.pid != os.getpid():
                process.kill()


class ProcessFamily:
    def __init__(self, directory):
        self.directory, self.own = directory, psutil.Process()
        self.marker = uuid.uuid4().hex
        os.environ[MARKER] = self.marker
        self.members = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.execution_id = None
        self.subreaper = False
        if sys.platform == "linux":
            import ctypes

            self.subreaper = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
        self.original = subprocess.Popen
        family = self

        class OwnedPopen(self.original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                family.register(self.pid, "python_popen")

        subprocess.Popen = OwnedPopen
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def register(self, pid, source):
        with contextlib.suppress(psutil.Error):
            process = psutil.Process(pid)
            with self.lock:
                if pid not in self.members:
                    self.members[pid] = {
                        "pid": pid,
                        "birth": process.create_time(),
                        "source": source,
                        "execution_id": self.execution_id,
                    }
                    self.persist()

    def persist(self):
        atomic_write(
            self.directory / "process-family.json",
            json.dumps(
                {"processes": list(self.members.values()), "subreaper": self.subreaper}
            ).encode(),
        )

    def discover(self, *, escaped=False):
        with contextlib.suppress(psutil.Error):
            for process in self.own.children(recursive=True):
                self.register(process.pid, "observed_descendant")
        if escaped:
            for process in psutil.process_iter(["pid", "uids"]):
                if process.pid == self.own.pid:
                    continue
                with contextlib.suppress(psutil.Error):
                    if process.info.get("uids") and process.info["uids"].real != os.getuid():
                        continue
                    if process.environ().get(MARKER) == self.marker:
                        self.register(process.pid, "reconciled_owner_marker")

    def watch(self):
        last = 0
        while not self.stop.wait(0.1):
            escaped = time.monotonic() - last > 1
            self.discover(escaped=escaped)
            if escaped:
                last = time.monotonic()

    def observation(self):
        self.discover()
        result = []
        with self.lock:
            for row in self.members.values():
                alive = False
                with contextlib.suppress(psutil.Error):
                    process = psutil.Process(row["pid"])
                    alive = (
                        process.create_time() == row["birth"]
                        and process.status() != psutil.STATUS_ZOMBIE
                    )
                result.append(
                    {
                        **row,
                        "state": "running" if alive else "exited",
                        "management": "owned Python subprocess",
                    }
                )
        return result[-100:]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.discover(escaped=True)
        cleanup_registry(self.directory)
        subprocess.Popen = self.original


def install_shutdown(family):
    def stop(sig, _):
        family.close()
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, stop)
