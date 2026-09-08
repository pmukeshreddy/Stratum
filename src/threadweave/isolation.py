"""OS-enforced research worker write confinement; fail closed if unavailable."""

import json
import shutil
import sys


def readonly_worker(directory):
    directory = str(directory.resolve())
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        profile = (
            "(version 1)(allow default)(deny file-write*)"
            f'(allow file-write* (subpath {json.dumps(directory)}) (literal "/dev/null"))'
        )
        return ["sandbox-exec", "-p", profile]
    if sys.platform == "linux" and shutil.which("bwrap"):
        return [
            "bwrap",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            directory,
            directory,
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ]
    raise RuntimeError(
        "Read-only research requires macOS sandbox-exec or Linux bubblewrap; shared mode is not a substitute"
    )
