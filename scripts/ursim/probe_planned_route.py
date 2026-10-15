"""Drive URSim through the planned route of 85f082b: a line where it is clear, a plan around a block.

A real controller, the real sidecar on the host's GPU, no hardware. Three claims, each read on what the controller
reports and on the route the arm logged:

* (u1) a clear move runs as one straight joint line: a Cartesian move 60 mm up from the start logs ``direct line`` with
  one leg, cuRobo plans nothing, and the controller ends within 1 mm of the goal;
* (u2) a joint move whose straight line crosses a declared block is planned around to the same target: a 60 mm block
  stands where the hand passes at the middle of a base swing, 130 mm below the flange, in the cell's own config, so the
  guard and the planner both hold it; the move logs ``cuRobo plan around the straight line`` and the controller ends
  on the target. A block and not a wall from the bench: on a UR3e the hand passes 256 mm from the base axis, and a wall
  up to the hand met the upper arm beside the shoulder at the start (measured 2026-09-25, 279.6 mm into it);
* (u3) a station written at home - 200 degrees on wrist 3 runs at home + 160 degrees, its full turn inside the half
  turn about home, and the log names the twin.

The cell is the profile's (``ursim,ursim_curobo`` by default, ``ursim,ursim_curobo,ursim_ur3`` for the UR3e), with three
things added for the probe alone: ``safety.joint_limits.within_half_turn_of_home``, ``safety.planning_world`` on with a
bench, and the block as a declared fixture. The start is WILLY_ROUTE_START (six radians, comma separated), driven to
first through the checked joint move; the default is the bed's start of 2026-09-18. It is the MIDDLE of the swing,
where the block stands, so the arm must not stand there when the probe begins: a fresh UR3e, whose own pose the
planner refuses, is jogged first through the ik route (``WILLY_PROFILE=ursim,ursim_ur3``, ``move_to_joints``) to the
swing's left end, the start with the base 0.5 rad lower.

Run (URSim up, in REMOTE control, from the repository root with the project venv)::

    python scripts/ursim/probe_planned_route.py --json logs/port2/probe_planned_route_ur5e.json
    WILLY_PROFILE=ursim,ursim_curobo,ursim_ur3 python scripts/ursim/probe_planned_route.py --json logs/port2/probe_planned_route_ur3e.json

Exit codes: ``0`` every claim holds, ``1`` one does not, ``2`` the cell could not be built or connected.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("WILLY_PROFILE", "ursim,ursim_curobo")

import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import load_robot_config  # noqa: E402
from src.config.schema.robot import RobotConfig  # noqa: E402
from src.geometry import Frame, Pose  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402

_START = (-2.0, -1.9, 1.9, -1.5708, -1.5708, 0.0)
_SWING_RAD = 1.0
#: Half the block's size: the ends of the swing stand 128 mm (UR3e) to 190 mm (UR5e) aside, the fingers clear of it.
_BLOCK_HALF_MM = 30.0
#: How far below the flange the block's centre stands at the swing's middle: among the fingers of the 2F-85 on its
#: 132 mm tool frame, well below the wrist.
_BLOCK_BELOW_FLANGE_MM = 130.0
_LIFT_MM = 60.0
_DECLINED = "URSim route probe: the container has no camera"


class _Lines(logging.Handler):
    """Every line the arm logs, kept to read the route it chose."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    def since(self, mark: int) -> str:
        return "\n".join(self.lines[mark:])


def _flange_mm(model: str, q: "list[float]") -> np.ndarray:
    frames = ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1][:3, 3], dtype=np.float64)


def _cell(middle: "list[float]") -> RobotConfig:
    """The profile's cell, with the window about home, a planning world and the block where the swing's middle passes."""
    config = load_robot_config()
    data = config.model_dump(mode="json")
    model = str(config.arm.model)
    flange = _flange_mm(model, middle)
    block = {"name": "route_probe_block",
            "center_mm": [float(flange[0]), float(flange[1]), float(flange[2]) - _BLOCK_BELOW_FLANGE_MM],
            "half_extents_mm": [_BLOCK_HALF_MM, _BLOCK_HALF_MM, _BLOCK_HALF_MM]}
    safety = data.setdefault("safety", {})
    safety.setdefault("joint_limits", {})["within_half_turn_of_home"] = True
    collision = safety.setdefault("self_collision", {})
    collision["fixtures"] = [*(collision.get("fixtures") or []), block]
    world = safety.setdefault("planning_world", {})
    world["enabled"] = True
    if not world.get("support_plane"):  # the dump carries the key as None where the profile declares no bench
        world["support_plane"] = {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0}
    return RobotConfig.model_validate(data)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_planned_route", description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default="logs/port2/probe_planned_route.json")
    args = parser.parse_args(argv)

    start_text = os.environ.get("WILLY_ROUTE_START", "").strip()
    start = [float(v) for v in start_text.split(",")] if start_text else list(_START)
    left, middle, right = ([start[0] + turn, *start[1:]] for turn in (-_SWING_RAD / 2.0, 0.0, _SWING_RAD / 2.0))
    record: dict[str, Any] = {"profile": os.environ["WILLY_PROFILE"], "start": start}
    try:
        config = _cell(middle)
    except Exception as exc:  # noqa: BLE001 - a probe reports what refused its cell
        print(f"the probe's cell does not load: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    record["model"] = str(config.arm.model)
    if config.arm.planner != "curobo":
        print(f"this probe needs robot.motion.planner curobo; {record['profile']} says {config.arm.planner!r}",
              file=sys.stderr)
        return 2
    arm: Any = create_arm(RobotVendor.from_string(config.vendor), config=config)
    lines = _Lines()
    arm.logger.addHandler(lines)
    arm.logger.setLevel(logging.INFO)
    try:
        arm.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"the arm did not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    claims: dict[str, Any] = {}
    try:
        with arm.without_camera_world(_DECLINED):
            moved = arm.move_to_joints(JointPositions(left))
            record["to_the_left_end"] = f"{moved.status}: {str(moved.message or '')[:200]}"
            if not moved.ok:
                print(f"the probe could not reach its start: {moved.message}", file=sys.stderr)
                return 2

            # (u1) 60 mm up from where the arm stands, the straight line clear.
            tcp = arm.get_tcp_pose()
            up = Pose(position_mm=np.asarray(tcp.position_mm) + np.array([0.0, 0.0, _LIFT_MM]),
                      quaternion_xyzw=np.asarray(tcp.quaternion_xyzw), frame=Frame.BASE, label="up")
            mark = len(lines.lines)
            result = arm.move(up)
            said = lines.since(mark)
            reached = arm.get_tcp_pose()
            claims["u1_a_clear_move_runs_one_line"] = {
                "status": str(result.status), "direct_line": "direct line" in said,
                "one_leg": "2 executed (1 leg(s))" in said, "planned": "cuRobo plan" in said,
                "end_error_mm": round(float(np.linalg.norm(np.asarray(reached.position_mm)
                                                           - np.asarray(up.position_mm))), 3),
            }
            back = arm.move_to_joints(JointPositions(left))
            record["back_to_the_left_end"] = str(back.status)

            # (u2) across the block to the other end of the swing.
            mark = len(lines.lines)
            result = arm.move_to_joints(JointPositions(right))
            said = lines.since(mark)
            here = [float(v) for v in arm.get_joint_positions().tolist()]
            claims["u2_a_line_into_the_block_is_planned_around"] = {
                "status": str(result.status), "message": str(result.message or "")[:300],
                "planned_around": "cuRobo plan around the straight line" in said,
                "end_error_rad": round(max(abs(a - b) for a, b in zip(here, right)), 5),
                "route": [line for line in said.splitlines() if "move_to_joints:" in line][:2],
            }

            # (u3) a station at home - 200 degrees on wrist 3 runs at home + 160.
            home = [float(v) for v in np.asarray(arm._home_joints, dtype=np.float64).tolist()]
            homed = arm.move_to_home()
            record["home"] = home
            record["to_home"] = str(homed.status)
            station = [*home[:5], home[5] - math.radians(200.0)]
            mark = len(lines.lines)
            result = arm.move_to_joints(JointPositions(station))
            said = lines.since(mark)
            wrist = float(arm.get_joint_positions().tolist()[5])
            claims["u3_a_station_past_the_window_runs_as_its_twin"] = {
                "status": str(result.status), "message": str(result.message or "")[:300],
                "wrist_3_deg": round(math.degrees(wrist), 3),
                "expected_deg": round(math.degrees(home[5]) + 160.0, 3),
                "twin_logged": "twin wrap" in said and "wrist_3_joint" in said,
            }
            arm.move_to_home()
    finally:
        arm.disconnect()
    u1 = claims.get("u1_a_clear_move_runs_one_line", {})
    u2 = claims.get("u2_a_line_into_the_block_is_planned_around", {})
    u3 = claims.get("u3_a_station_past_the_window_runs_as_its_twin", {})
    passed = {
        "u1": u1.get("direct_line") is True and u1.get("one_leg") is True and u1.get("planned") is False
        and u1.get("end_error_mm", 1e9) <= 1.0,
        "u2": u2.get("planned_around") is True and u2.get("end_error_rad", 1e9) <= 1e-3,
        "u3": u3.get("twin_logged") is True and abs(u3.get("wrist_3_deg", 1e9) - u3.get("expected_deg", 0.0)) <= 0.1,
    }
    record["claims"] = claims
    record["passed"] = passed
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    for key, value in claims.items():
        print(f"{key}: {value}")
    ok = all(passed.values())
    print(f"S2 planned route on {record['model']}: {passed} -> {'PASS' if ok else 'FAIL'}")
    print(f"numbers written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
