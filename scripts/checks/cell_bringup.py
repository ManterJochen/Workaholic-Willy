"""Does the arm answer, and is it standing INSIDE the workspace box this cell declares?

    WILLY_PROFILE=ursim python scripts/checks/cell_bringup.py --live

⛔ **NO CLI IN THIS REPOSITORY ANSWERS THIS.** `real_cell --check` reads the YAML and never opens a
socket, `real_cell --runs 1` connects and then immediately grasps, and `drivers.ur --read` reads I/O
pins. Nothing else connects, reads `get_tcp_pose()` back, and holds that pose against the declared
box. An arm parked outside its own box is a cell that refuses its first commanded move as
WORKSPACE_REJECTED, and that refusal reads at the pendant like a broken guard rather than like a
parked robot, so it is worth one connection to find out before the first pick.

Every bound below is derived from the operator's own YAML rather than hardcoded: the box is
`robot.workspace_limits`, and each of its six faces is pulled in by `safety.limits.
workspace_margin_mm`, which is the same shrink `SafetyPreflight.from_safety_config` applies before
the workspace guard ever sees the box. So the number in the YAML is not the number that refuses, and
a pose sitting in that margin skin is already refused even though it is inside the declared box.

⚠ **THIS CHECK CONNECTS**, which is why it refuses to run without `--live`: reading a controller is
not something a check may do because someone typed its name. That is the only flag it has.

Exit codes: 0 the arm answered and stands inside the box it will be held to, 1 it answered and
disagrees with its own config, 2 there is nothing to connect to (no cell in the config, no driver on
this host, no `--live`, or a controller that did not answer).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.geometry import Frame  # noqa: E402
from src.robot.core import (  # noqa: E402
    RobotConnectionError,
    RobotVendor,
    SupportsRobotStatus,
)
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.host import Host  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_NOT_READY = 0, 1, 2


def _not_ready(what: str, fix: str) -> int:
    print(f"NOT READY: {what}\n  fix: {fix}")
    return EXIT_NOT_READY


def main() -> int:
    argv = sys.argv[1:]
    if [arg for arg in argv if arg != "--live"]:
        return _not_ready(f"unrecognised argument(s): {' '.join(argv)}",
                          "this check takes one flag and no others: --live")

    try:
        config = load_robot_config()
    except ConfigError as error:
        return _not_ready(f"the config tree has no cell to check ({error})",
                          "select a profile that has a `robot` block: WILLY_PROFILE=ur5e")

    if "--live" not in argv:
        return _not_ready(
            f"this check opens a connection to the {config.vendor!r} arm and reads its pose back, "
            "and it will not do that unless you ask for it",
            "power the cell, put the pendant in REMOTE control, then re-run with --live")

    # The same gate `from_robot_config` runs before it builds anything, called here for its message:
    # it names the missing SDK module and what to install, which is a better answer than the
    # ImportError the factory would raise a moment later.
    try:
        Host.local().readiness().require(config.vendor)
    except RobotConnectionError as error:
        return _not_ready(f"this host cannot drive vendor {config.vendor!r}: {error}",
                          "install the vendor extra, or point this at a vendor the host has: "
                          "WILLY_PROFILE=ursim drives a UR against a container")

    # `create_arm`, not a whole cell: `build_real_cell` opens an RGB-D camera and loads two models
    # onto the GPU, and the question here is only what the controller says about itself.
    try:
        arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
    except Exception as error:  # noqa: BLE001  (report, not raise)
        return _not_ready(
            f"the {config.vendor!r} arm could not be built ({type(error).__name__}: {error})",
            "`python -m src.robot.execution.real_cell --check` reads the same config")

    try:
        # `connect()` does more than open a socket on the UR path: it pushes the declared payload
        # and verifies the tool frame against the controller, so this is the step that catches a
        # config describing a different robot than the one on the bench.
        arm.connect()
    except Exception as error:  # noqa: BLE001  (report, not raise)
        return _not_ready(
            f"the controller did not answer ({type(error).__name__}: {error})",
            "is the controller reachable, is the robot in REMOTE control, and is the program "
            "running? For a simulator: scripts/ursim/ursim.sh up MODEL")

    caps = arm.capabilities
    box = config.workspace_limits
    margin = float(config.safety.limits.workspace_margin_mm)
    wrong: list[str] = []
    print(f"{caps.vendor} {caps.model}: connected, box shrunk by {margin:g} mm on every face")

    try:
        pose = arm.get_tcp_pose()
        print(f"  tcp pose             {[round(float(v), 1) for v in pose.position_mm]} mm "
              f"in {pose.frame.value}")
        if pose.frame is not Frame.BASE:
            # The box is declared in BASE. A pose tagged anything else cannot be compared against
            # it, and comparing anyway would answer a question nobody asked.
            wrong.append(f"the arm reports its pose in {pose.frame.value}, but workspace_limits "
                         f"is declared in {Frame.BASE.value}")
            print(f"  workspace            NOT COMPARABLE <- pose frame is not {Frame.BASE.value}")
        else:
            for axis, value, low, high in (
                ("x", float(pose.position_mm[0]), box.x_min, box.x_max),
                ("y", float(pose.position_mm[1]), box.y_min, box.y_max),
                ("z", float(pose.position_mm[2]), box.z_min, box.z_max),
            ):
                face_lo, face_hi = low + margin, high - margin
                if face_lo >= face_hi:
                    wrong.append(f"workspace_margin_mm={margin:g} inverts {axis} "
                                 f"[{low:g}, {high:g}]; the preflight will refuse to build")
                    print(f"  workspace {axis:10s} INVERTED    margin is wider than the axis")
                elif not face_lo <= value <= face_hi:
                    where = ("outside the declared box" if not low <= value <= high else
                             f"inside the declared box but within its {margin:g} mm margin")
                    wrong.append(f"{axis} {value:.1f} mm is {where} [{low:g}, {high:g}]; the first "
                                 f"commanded move is refused as WORKSPACE_REJECTED")
                    print(f"  workspace {axis:10s} OUTSIDE     {value:.1f} not in "
                          f"[{face_lo:g}, {face_hi:g}] <- {where}")
                else:
                    print(f"  workspace {axis:10s} inside      {value:.1f} in "
                          f"[{face_lo:g}, {face_hi:g}]")

        # A capability, not part of `RobotArm`: vendors differ in what they can report, so the
        # stack feature-checks the Protocol instead of assuming, and a driver that does not carry
        # it is UNSTATED rather than healthy.
        if not isinstance(arm, SupportsRobotStatus):
            print(f"  controller state     UNSTATED    {type(arm).__name__} does not implement "
                  "SupportsRobotStatus")
        else:
            status = arm.get_robot_status()
            print(f"  controller state     {status.robot_mode.value} / {status.safety_mode.value}; "
                  f"protective {status.protective_stopped}; operational {status.is_operational}")
            if status.is_stopped:
                wrong.append(f"the cell is STOPPED ({status.safety_mode.value}, protective "
                             f"{status.protective_stopped}); clear it before a pick")
    finally:
        arm.disconnect()

    if wrong:
        print(f"\nFAILED: {len(wrong)} disagreement(s) between this arm and its own config")
        for line in wrong:
            print(f"  {line}")
        return EXIT_FAILED
    print(f"\nOK: {caps.vendor} {caps.model} answered and stands inside the box it is held to")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
