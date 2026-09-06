"""Is the Robotiq URCap actually answering on port 63352? Read-only, commands nothing.

    python scripts/ursim/probe_robotiq_urcap.py                 # URSim on localhost
    python scripts/ursim/probe_robotiq_urcap.py 192.168.1.10    # a real controller

An open port proves nothing here, and that is the whole reason this script exists. Measured against
URSim with port 63352 published and no URCap installed: the TCP connect succeeds and the connection
dies on the first byte. Docker forwards the port; nothing behind it listens. So a port scanner says
yes and there is no gripper. Only a parseable reply is evidence, which is why the connect is
printed as a non-result and the reading is the answer.

It is also the instrument for the one question the vendor documentation cannot settle: polarity.
The manual gives 0 as open, and ur_rtde's own header contradicts itself between its unit systems,
`UNIT_DEVICE` 0 as open against `UNIT_NORMALIZED` 0.0 as closed. So the procedure never asks anyone
to trust a number:

    1. Run this. Note POS.
    2. Move the fingers from the teach pendant, or from the Compute Box web UI.
    3. Run it again. If POS went up as the fingers closed, the manual holds for this unit.

Nothing here writes. There is no flag that makes it write.

This exercises the call path. It cannot tell you whether the jaws are wired to the right tool, and
on URSim there are no jaws at all.

Exit codes come from `ProbeReading.exit_code`, so a caller scripting against the library and a
caller reading this exit status get the same verdict: 0 reachable and not faulted, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. `scripts/examples/_common.py` does the same insert for the examples.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.robot.grippers.robotiq_socket import (  # noqa: E402
    DEFAULT_PORT,
    RobotiqSocket,
    RobotiqSocketError,
)

#: The only code this file derives for itself. Before `probe()` there is no `ProbeReading` to read
#: an exit code from, and the connect failing is the same answer as an unreachable daemon.
_EXIT_NO_URCAP = 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/probe_robotiq_urcap.py",
        description="Ask whether a Robotiq URCap daemon answers on the gripper socket. Read-only: "
                    "it commands nothing, on URSim or on a real controller.",
    )
    ap.add_argument("host", nargs="?", default="127.0.0.1",
                    help="controller address (default: the URSim loopback)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"URCap socket port (default: {DEFAULT_PORT})")
    args = ap.parse_args(argv)

    print(f"=== Robotiq URCap socket at {args.host}:{args.port} ===", flush=True)
    client = RobotiqSocket(timeout_s=3.0)
    try:
        client.connect(args.host, args.port)
    except RobotiqSocketError as exc:
        print(f"  no TCP connection: {exc}", flush=True)
        return _EXIT_NO_URCAP

    # The connect succeeding is printed as a non-result, deliberately. It is the step an operator
    # is most likely to read as success, and it is the step that means least.
    print("  TCP connect succeeded, which on its own means nothing. Asking the daemon:", flush=True)
    try:
        reading = client.probe()
        print(reading.render(), flush=True)
        if not reading.reachable:
            print(
                "\n  No URCap is answering. Install Robotiq_Grippers on the controller, or for "
                "URSim\n  drop the .urcap (renamed to .jar) in a directory and start with\n"
                "     URSIM_URCAPS=<that directory> scripts/ursim/ursim.sh up UR5",
                flush=True,
            )
        elif not reading.faulted:
            print(
                "\n  A URCap is answering. Now settle the polarity, which no document can: move "
                "the\n  fingers from the pendant and run this again. POS must go up as they close.",
                flush=True,
            )
        # The verdict is the reading's, never a second computation of the same fact. A CLI that
        # split the codes its own way handed a scripted caller a different number for one state.
        return reading.exit_code
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
