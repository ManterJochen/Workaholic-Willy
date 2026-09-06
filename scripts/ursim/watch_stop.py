"""Watch what this stack sees when a human stops the cell, mid-motion and connected.

This is the case URSim can only produce with a person at the pendant: the controller enters a stop
while the driver is connected and commanding motion. `probe_protective_stop.py` covers the case a
script can produce on its own, a stop provoked through a commanded motion of its own. Neither
covers a cell that boots into a stop, which `probe_protective_stop.py` describes as an approach
it rejected.

It also observes whether the emergency stop still works while the robot is in remote control, where
the pendant is otherwise locked for motion.

It touches no digital I/O and moves nothing until a stop has been seen.

Usage:
    python scripts/ursim/watch_stop.py                            # UR5e profile
    python scripts/ursim/watch_stop.py --profile ursim,ursim_ur3  # UR3e profile

Press the red stop in the pendant UI at http://localhost:6080/vnc.html when prompted.

Exit codes: 0 the status changed and was measured, 1 nothing changed inside the wait, 2 no
controller reachable, or a profile that is not a UR cell.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. `scripts/examples/_common.py` does the same insert for the examples.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config.loader import active_profile, load_robot_config  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402

#: The chain this probe needs when the operator has not named one. It is resolved here rather than
#: written into `WILLY_PROFILE`, which would leak process-global state into whatever runs next. It
#: is not left to the loader's own default either: the base tree points `ur.ip` at a plant address,
#: and a script that commands motion after a stop must never reach a real cell by default.
_DEFAULT_PROFILE = "ursim"

#: How long to wait for a person to press the stop.
WAIT_S = 120.0

_OK, _NO_CHANGE, _BAD_REQUEST = 0, 1, 2


def line(arm: URRobotArm) -> str:
    """The six `RobotStatus` fields on one line, for comparing two samples by eye.

    `probe_protective_stop.py` composes the same six into a two-line form. `RobotStatus` has no
    `render()`, so each caller states the layout it needs; adding one to the class is a library
    decision rather than something a script settles.
    """
    st = arm.get_robot_status()
    return (f"mode={st.robot_mode.value:<12} safety={st.safety_mode.value:<22} "
            f"protective={str(st.protective_stopped):<5} emergency={str(st.emergency_stopped):<5} "
            f"operational={str(st.is_operational):<5} stopped={st.is_stopped}")


def _wait_for_change(arm: URRobotArm, baseline: str) -> str | None:
    """Poll until the status line differs from `baseline`, or the wait runs out.

    The elapsed figure is measured from one start time. Deriving it from the deadline instead means
    two expressions of one clock, and they drift apart the moment either is edited.
    """
    t0 = time.time()
    while time.time() - t0 < WAIT_S:
        now = line(arm)
        if now != baseline:
            print(f"  CHANGED after {time.time() - t0:.0f}s")
            return now
        time.sleep(0.5)
    return None


def _retry_motions(arm: URRobotArm) -> None:
    """Three retried motions, in the shape a pick loop would issue them after a refusal."""
    try:
        q = list(arm.get_joint_positions().values)
    except Exception as exc:  # noqa: BLE001
        print(f"   could not even read joints: {type(exc).__name__}: {str(exc)[:80]}")
        return
    for attempt in (1, 2, 3):
        t0 = time.time()
        try:
            target = list(q)
            target[5] += 0.08 * attempt
            arm.move_joint(JointPositions(target))
            print(f"   attempt {attempt}: ACCEPTED in {time.time() - t0:.1f}s. "
                  f"A stopped cell reporting success is the dangerous answer.")
        except Exception as exc:  # noqa: BLE001
            print(f"   attempt {attempt}: {type(exc).__name__} in {time.time() - t0:.1f}s: "
                  f"{str(exc)[:75]}")
        time.sleep(0.3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/watch_stop.py",
        description="Poll RobotStatus while a person presses the stop at the pendant, then ask "
                    "whether raise_if_stopped notices and what three retried motions do.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ursim' or 'ursim,ursim_ur3'")
    args = ap.parse_args(argv)

    chain = args.profile if args.profile is not None else (active_profile() or _DEFAULT_PROFILE)
    cfg = load_robot_config(profile=chain)
    built = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
    if not isinstance(built, URRobotArm):
        print(f"profile {chain!r} builds {type(built).__name__}; this probe reads UR-only seams")
        return _BAD_REQUEST
    arm = built
    try:
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"arm.connect() failed: {type(exc).__name__}: {exc}")
        print("no controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST

    try:
        print(f"connected to {cfg.ur.model} at {cfg.ur.ip}\n")
        baseline = line(arm)
        print("  BEFORE :", baseline)

        print("\n" + "=" * 78)
        print("  PRESS THE RED STOP NOW, in the pendant UI:  http://localhost:6080/vnc.html")
        print("  (the round red button, top-left of the pendant frame)")
        print("=" * 78 + "\n")

        changed = _wait_for_change(arm, baseline)
        if changed is None:
            print(f"  no change in {WAIT_S:.0f}s: nothing was pressed, or it did not reach the "
                  f"controller")
            print("  AFTER  :", line(arm))
            return _NO_CHANGE
        print("  AFTER  :", changed)

        print("\n=== does the driver's own guard notice? ===")
        try:
            arm.raise_if_stopped()
            print("   raise_if_stopped(): did NOT raise. A pick loop would carry straight on.")
        except Exception as exc:  # noqa: BLE001
            print(f"   raise_if_stopped(): {type(exc).__name__}: {str(exc)[:110]}")

        print("\n=== and what does a commanded motion do, three retries as a pick loop would? ===")
        _retry_motions(arm)
        return _OK
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 (teardown must not mask what the probe measured)
            pass
        print("\nRELEASE the stop in the UI when you are done. WATCH_STOP_DONE")


if __name__ == "__main__":
    raise SystemExit(main())
