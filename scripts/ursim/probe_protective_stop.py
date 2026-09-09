"""What does this stack see when the controller enters a protective stop mid-pick?

`URRobotArm.get_robot_status()` projects `isProtectiveStopped`, `isEmergencyStopped`, the safety
mode and the controller's own text. This probe provokes a genuine stop through the driver's own
motion path and then measures what that projection says, what `raise_if_stopped()` does about
it, and what three retried motions do in the shape a pick loop issues them.

Connect first, then trigger. The realistic case is a stop that happens while the cell is running,
not a cell that boots into a stopped controller. Connecting to an already-stopped controller
measures something real but different: `connect()` fails with "Failed to start control script,
before timeout" and names nothing, the way it does for local mode. `watch_stop.py` covers the third
case, a human at the pendant pressing the stop while the driver is connected.

`raise_if_stopped()` sits on the UR driver and on no Protocol, so the arm is built through the
vendor-neutral `create_arm` and then narrowed. A profile naming another vendor becomes a sentence
rather than a missing attribute.

Run (URSim up, in REMOTE control):  python scripts/ursim/probe_protective_stop.py

Exit codes: 0 a stop was provoked and measured, 1 no stop was provoked so there was nothing to
measure, 2 no controller reachable, or a profile that is not a UR cell.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root. Every file under `scripts/examples/api/` does the same insert inline, for the same reason.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from src.config.loader import active_profile, load_robot_config  # noqa: E402
from src.geometry import Frame, Pose  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402

HOST = "127.0.0.1"
_DASHBOARD_PORT = 29999

#: The chain this probe needs when the operator has not named one. It is resolved here rather than
#: written into `WILLY_PROFILE`, which would leak process-global state into whatever runs next. It
#: is not left to the loader's own default either: the base tree points `ur.ip` at a plant address,
#: and a probe that deliberately provokes a protective stop must never reach a real cell.
_DEFAULT_PROFILE = "ursim"

#: Straight up the base axis is the shoulder singularity, and how far up depends on the robot: a
#: UR3e reaches about 500 mm where a UR5e reaches about 850, both vendor specifications. Asking a
#: UR3e for 900 mm is unreachable rather than singular and would test the reach check instead of
#: the stop. `probe_pickloop_stop.py` reads this table rather than declaring a second height.
SINGULAR_Z_MM = {"ur3e": 430.0, "ur5e": 900.0, "ur10e": 1100.0}

_OK, _NO_STOP, _BAD_REQUEST = 0, 1, 2


def dash(cmds: list[str]) -> list[str]:
    """Speak the dashboard protocol over a raw socket, before any connection exists.

    This is the pre-connect door and there is no library twin for it. `URConnection` builds its
    dashboard client inside `connect()`, so `dashboard_safety_status()`, `is_in_remote_control()`
    and `get_robot_mode()` all need a connection this probe has not made yet, and `robotmode` has
    no pre-connect door at any level. Once connected, the probe uses those methods instead.
    """
    s = socket.create_connection((HOST, _DASHBOARD_PORT), timeout=15)
    s.settimeout(15)
    s.recv(256)
    out = []
    for c in cmds:
        s.sendall((c + "\n").encode())
        time.sleep(0.6)
        try:
            out.append(f"{c:<26} {s.recv(512).decode(errors='replace').strip()}")
        except Exception as exc:  # noqa: BLE001
            out.append(f"{c:<26} <{exc}>")
    s.close()
    return out


def show_status(arm: URRobotArm, tag: str) -> None:
    """Print the six `RobotStatus` fields an operator reads.

    `watch_stop.py` composes the same six into a one-line form. `RobotStatus` has no `render()`, so
    each caller states the layout it needs; adding one to the class is a library decision.
    """
    st = arm.get_robot_status()
    print(f"   [{tag}] mode={st.robot_mode.value} safety={st.safety_mode.value} "
          f"protective_stopped={st.protective_stopped} emergency={st.emergency_stopped}")
    print(f"   [{tag}] is_operational={st.is_operational} is_stopped={st.is_stopped} "
          f"message={st.message!r}")


def _provoke(arm: URRobotArm, model: str) -> None:
    """Command a straight-line move through the shoulder singularity.

    It is chosen because it trips the controller's own safety system rather than any check of
    this file's: the point is a stop nothing here asked for.
    """
    z = SINGULAR_Z_MM.get(model, SINGULAR_Z_MM["ur5e"])
    print(f"   (model={model}: driving straight up the base axis to z={z:.0f} mm)")
    singular = Pose(
        position_mm=np.array([0.0, 0.0, z]),
        quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
        frame=Frame.BASE,
    )
    try:
        res = arm.move(singular)
        print(f"   arm.move(singular): status={res.status.value}")
        if res.message:
            print(f"      message: {res.message[:110]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   arm.move(singular) raised {type(exc).__name__}: {str(exc)[:100]}")
    time.sleep(3.0)


def _retry_motions(arm: URRobotArm) -> None:
    """Three further motions, exactly as a pick loop would retry them."""
    try:
        q = list(arm.get_joint_positions().values)
    except Exception as exc:  # noqa: BLE001
        print(f"   get_joint_positions() failed: {type(exc).__name__}: {str(exc)[:70]}")
        return
    for attempt in (1, 2, 3):
        t0 = time.time()
        try:
            target = list(q)
            target[5] += 0.1 * attempt
            arm.move_joint(JointPositions(target))
            print(f"   attempt {attempt}: ACCEPTED in {time.time() - t0:.1f}s. "
                  f"A stopped cell that reports success is the dangerous answer.")
        except Exception as exc:  # noqa: BLE001
            print(f"   attempt {attempt}: {type(exc).__name__} in {time.time() - t0:.1f}s: "
                  f"{str(exc)[:80]}")
        time.sleep(0.4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python scripts/ursim/probe_protective_stop.py",
        description="Provoke a real controller protective stop through the driver's own "
                    "motion path, then measure what RobotStatus, raise_if_stopped and three "
                    "retried motions report.",
    )
    ap.add_argument("--profile", type=str, default=None,
                    help="config profile chain, e.g. 'ursim' or 'ursim,ursim_ur3'")
    args = ap.parse_args(argv)

    print("=== 0. starting state, before any connection exists ===")
    try:
        for line in dash(["robotmode", "safetystatus", "is in remote control"]):
            print("  ", line)
    except OSError as exc:
        # The dashboard is the cheapest reachability question there is, so a refusal here is a
        # refusal to run rather than a traceback out of a socket call.
        print(f"   dashboard UNREACHABLE at {HOST}:{_DASHBOARD_PORT}: {exc}")
        print("no controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST

    chain = args.profile if args.profile is not None else (active_profile() or _DEFAULT_PROFILE)
    cfg = load_robot_config(profile=chain)
    built = create_arm(RobotVendor.from_string(cfg.vendor), config=cfg)
    if not isinstance(built, URRobotArm):
        print(f"\nprofile {chain!r} builds {type(built).__name__}; this probe reads UR-only seams")
        return _BAD_REQUEST
    arm = built
    try:
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"\narm.connect() failed: {type(exc).__name__}: {exc}")
        print("no controller reachable: is URSim up and in REMOTE control?")
        return _BAD_REQUEST

    stopped = False
    try:
        print("\n=== 1. the driver is connected, cell healthy ===")
        show_status(arm, "before")

        print("\n=== 2. provoke a real protective stop through the driver's own motion path ===")
        _provoke(arm, cfg.ur.model)

        print("\n=== 3. what this stack sees now, on the connection it already had ===")
        show_status(arm, "after")
        stopped = arm.get_robot_status().is_stopped
        try:
            arm.raise_if_stopped()
            print("   raise_if_stopped(): did NOT raise")
        except Exception as exc:  # noqa: BLE001
            print(f"   raise_if_stopped(): {type(exc).__name__}: {str(exc)[:110]}")

        print("\n=== 4. the point: three further motions, exactly as a pick loop would retry ===")
        _retry_motions(arm)

        # The recovery runs while the connection is still open, so it goes through the library
        # rather than through a second raw socket: `recover_from_protective_stop()` is the
        # dashboard `close safety popup` plus `unlock protective stop`, and `get_robot_status()`
        # carries the resulting `safetystatus` text.
        print("\n=== 5. leave the cell as it was found ===")
        print(f"   recover_from_protective_stop(): {arm.recover_from_protective_stop()}")
        time.sleep(2)
        show_status(arm, "recovered")
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 (teardown must not mask what the probe measured)
            pass

    print("PROTECTIVE_STOP_PROBE_DONE")
    return _OK if stopped else _NO_STOP


if __name__ == "__main__":
    raise SystemExit(main())
