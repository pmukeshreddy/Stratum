"""Fixed-game Python interface. The host owns lifecycle, action limits, and state.

Bootstrap: observe first, then submit one non-RESET action, then a second single
non-RESET action if still ACTIVE. After bootstrap, act accepts batches of 1..20.
The host retains this phase across Python invocations; reading this module or
restarting Python does not reset it. Actions use {"name": "ACTION1", "data": {}}.
"""

import argparse
import json
import socket
from pathlib import Path


def _call(op, **arguments):
    connection = json.loads(Path(__file__).with_name(".game-connection.json").read_text())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        if op == "status":
            client.settimeout(5)
        client.connect(connection["socket"])
        client.sendall(
            (json.dumps({"token": connection["token"], "op": op, **arguments}) + "\n").encode()
        )
        with client.makefile("rb") as stream:
            reply = json.loads(stream.readline())
    if not reply["ok"]:
        raise RuntimeError(reply["error"])
    return reply["value"]


def observe():
    return _call("observe")


def status():
    return _call("status")


def act(actions):
    return _call("act", actions=actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["observe", "status", "act"])
    parser.add_argument("--actions", type=json.loads)
    args = parser.parse_args()
    if args.operation == "act":
        value = act(args.actions)
    else:
        if args.actions is not None:
            parser.error("--actions is only valid with act")
        value = observe() if args.operation == "observe" else status()
    print(json.dumps(value))


if __name__ == "__main__":
    main()
