"""Standard-library targetRunner of the VAMOS irace backend.

irace runs ``python _irace_target_runner.py CONFIG_ID INSTANCE_ID SEED INSTANCE --name=value ...``.
The experiment is forwarded to the evaluation server of the running ``ModelBasedTuner``, found
through ``VAMOS_IRACE_PORT`` and ``VAMOS_IRACE_TOKEN``, and the cost is written on stdout, which
is the only output irace accepts from a target runner.
"""

from __future__ import annotations

import json
import os
import socket
import sys


def main(argv: list[str]) -> int:
    request = {"token": os.environ["VAMOS_IRACE_TOKEN"], "argv": argv}
    with socket.create_connection(("127.0.0.1", int(os.environ["VAMOS_IRACE_PORT"]))) as conn:
        conn.sendall(json.dumps(request).encode("utf-8") + b"\n")
        reply = json.loads(conn.makefile("rb").readline())
    if "error" in reply:
        sys.stderr.write(f"{reply['error']}\n")
        return 1
    sys.stdout.write(f"{float(reply['cost'])!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
