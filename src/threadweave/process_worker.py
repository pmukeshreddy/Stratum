"""Own a command process group and terminate it if the daemon disappears."""

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

import psutil


def main():
    owner = os.getppid()
    own = psutil.Process()
    descendants = {}
    lock = threading.RLock()

    def discover():
        with contextlib.suppress(psutil.Error):
            with lock:
                for child in own.children(recursive=True):
                    descendants[child.pid] = child

    def cleanup():
        discover()
        with lock:
            children = list(descendants.values())
        for child in reversed(children):
            # psutil compares process birth identity, protecting against PID reuse.
            with contextlib.suppress(psutil.Error):
                child.kill()

    def terminate(sig, _):
        cleanup()
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, terminate)

    def watch_owner():
        while True:
            discover()
            time.sleep(0.05)
            if os.getppid() != owner:
                cleanup()
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_owner, daemon=True).start()
    process = subprocess.Popen(sys.argv[1:])
    try:
        status = process.wait()
    finally:
        cleanup()
    raise SystemExit(status if status >= 0 else 128 - status)


if __name__ == "__main__":
    main()
