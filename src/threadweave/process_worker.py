"""Own a command process group and terminate it if the daemon disappears."""

import os
import signal
import subprocess
import sys
import threading
import time


def main():
    owner = os.getppid()

    def watch_owner():
        while True:
            time.sleep(0.25)
            if os.getppid() != owner:
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_owner, daemon=True).start()
    process = subprocess.Popen(sys.argv[1:])
    status = process.wait()
    raise SystemExit(status if status >= 0 else 128 - status)


if __name__ == "__main__":
    main()
