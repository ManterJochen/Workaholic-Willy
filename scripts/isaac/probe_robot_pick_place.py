"""`Robot.pick` and `Robot.place` on the M1 cell, measured where the cube ends up.

User code picks and places without a pick service: a planned standoff, one straight line in, the hand
verb, one straight line out. This probe runs them on the Isaac M1 cell, booted with a decline because it
plans against no camera world, around the arm and the gripper the cell built, and claims three things:

* (a) every run's descent and retreat read CHECKED and reach the sim arm's checked line (two lines a
  pick, two a place), and every motion carries a DECLINED stamp;
* (b) at least 8 of 10 cubes come to rest within ``--tolerance-mm`` of the place pose in XY; Z is
  recorded, not claimed;
* (c) one control: a place pose outside the reach of the arm is MOTION_REFUSED and the cube stays in
  the hand.

Not claimed: release verification (the Isaac jaw measures no hold) and payload modelling (the sim arm
models none). The cube is read off its rigid prim, as ``run_m1_pick`` reads it. The grasp is the M1
known pose: the cube's centre 12 mm up, tool down, closing along BASE X, closed to 25 mm.

Usage, on the box and alone on it (Isaac python)::

    python.bat scripts/isaac/probe_robot_pick_place.py --runs 10 --json logs/step8/probe_robot_pick_place.json

Exit codes: ``0`` every claim holds, ``1`` one does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

#: The M1 grasp: the centre raised so the fingertips clear the table, and the width the M1 policy closes to.
_GRASP_LIFT_MM = 12.0
_CLOSE_TO_MM = 25.0
_PRE_OPEN_MM = 80.0
_STANDOFF_MM = 50.0
#: Tool down, closing along BASE -X: rotation columns (closing, binormal, approach) = diag(-1, 1, -1).
_TOOL_DOWN_XYZW = (0.0, 1.0, 0.0, 0.0)
#: Where the control asks to place: beyond the reach of a ur5e from its base.
_UNREACHABLE_MM = (1500.0, 0.0, 300.0)


def _pose(position_mm: Any) -> Any:
    from src.geometry import Frame, Pose

    return Pose(position_mm=np.asarray(position_mm, dtype=np.float64),
                quaternion_xyzw=np.asarray(_TOOL_DOWN_XYZW, dtype=np.float64), frame=Frame.BASE)


def _cube_mm(obj: Any) -> np.ndarray:
    return np.asarray(obj.get_world_pose()[0], dtype=np.float64) * 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_robot_pick_place", description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--place-mm", type=str, default="450,-120",
                        help="the place pose in BASE XY, millimetres; Z is the pick's TCP height plus 2 mm")
    parser.add_argument("--tolerance-mm", type=float, default=15.0,
                        help="how far in XY a cube may rest from the place pose and count as placed; fixed before the "
                             "first run and recorded")
    parser.add_argument("--json", type=str, default="logs/step8/probe_robot_pick_place.json")
    args = parser.parse_args(argv)
    place_xy = tuple(float(v) for v in args.place_mm.split(","))

    from src.robot.core.camera_world import CameraWorldDecline
    from src.willy_sim.harness.bootstrap import bootstrap_sim_cell

    decline = CameraWorldDecline("probe_robot_pick_place: robot verbs on the M1 cell, no camera world")
    cell = bootstrap_sim_cell(None, headless=True, camera_world=decline)

    from isaacsim.core.prims import SingleRigidPrim  # type: ignore[import-not-found]

    from src.robot.core import JointPositions
    from src.robot.core.arm_capabilities import LineMotion
    from src.robot.core.camera_world import CameraWorldUse
    from src.robot.execution.handling import HandlingOutcome
    from src.robot.execution.robot import Robot
    from src.willy_sim.config import require_robot
    from src.willy_sim.harness.gate import reset_object_to_home_z0

    arm, gripper, handles = cell.arm, cell.gripper, cell.handles
    robot_cfg = require_robot(cell.cfg)
    home_q = np.asarray(robot_cfg.sim.home_joint_positions, dtype=np.float64)
    obj = SingleRigidPrim(handles.object_prim_path)
    home_pos, home_quat = obj.get_world_pose()

    lines: list[str] = []
    original_line = arm._drive_checked_line

    def counted_line(pose: Any) -> Any:
        lines.append(getattr(pose, "label", "") or "line")
        return original_line(pose)

    arm._drive_checked_line = counted_line  # type: ignore[method-assign]
    robot = Robot.from_parts(arm=arm, gripper=gripper, lock_key=None)

    def reset() -> float:
        gripper.open()
        arm.move_joint(JointPositions(home_q))
        return reset_object_to_home_z0(obj, home_pos, home_quat, arm.session, settle_steps=30)

    report: dict[str, Any] = {
        "runs": args.runs, "place_xy_mm": list(place_xy), "tolerance_mm": args.tolerance_mm,
        "grasp_lift_mm": _GRASP_LIFT_MM, "close_to_mm": _CLOSE_TO_MM, "standoff_mm": _STANDOFF_MM,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": [],
    }
    placed = 0
    all_checked = True
    for run in range(args.runs):
        reset()
        cube = _cube_mm(obj)
        grasp_at = cube + np.array([0.0, 0.0, _GRASP_LIFT_MM])
        place_at = np.array([place_xy[0], place_xy[1], grasp_at[2] + 2.0])
        before = len(lines)
        picked = robot.pick(_pose(grasp_at), _CLOSE_TO_MM + 1.0, standoff_mm=_STANDOFF_MM, squeeze_mm=1.0,
                            pre_open_mm=_PRE_OPEN_MM, camera_world=decline)
        pick_lines = len(lines) - before
        row: dict[str, Any] = {"run": run, "cube_start_mm": [round(float(v), 1) for v in cube],
                               "pick": picked.to_dict(), "pick_lines": pick_lines}
        placed_here = False
        place_lines = 0
        if picked.ok:
            before = len(lines)
            put = robot.place(_pose(place_at), standoff_mm=_STANDOFF_MM, camera_world=decline)
            place_lines = len(lines) - before
            arm.session.step_n(60)
            rest = _cube_mm(obj)
            xy_error = float(np.linalg.norm(rest[:2] - place_at[:2]))
            placed_here = bool(put.ok and xy_error <= args.tolerance_mm)
            row.update({"place": put.to_dict(), "place_lines": place_lines, "cube_rest_mm":
                        [round(float(v), 1) for v in rest], "xy_error_mm": round(xy_error, 2)})
            stamps = list(picked.camera_worlds) + list(put.camera_worlds)
            checked = (picked.line is not None and picked.line.motion is LineMotion.CHECKED
                       and put.line is not None and put.line.motion is LineMotion.CHECKED
                       and pick_lines == 2 and place_lines == 2
                       and bool(stamps) and all(stamp.use is CameraWorldUse.DECLINED for stamp in stamps))
        else:
            checked = False
        row["claim_a"] = checked
        row["placed"] = placed_here
        all_checked = all_checked and checked
        placed += int(placed_here)
        report["results"].append(row)
        print(f"RUN {run}: pick={picked.outcome.value} lines={pick_lines}+{place_lines} "
              f"xy_error_mm={row.get('xy_error_mm')} placed={placed_here} checked={checked}", flush=True)
        if not picked.ok:
            print(f"RUN {run}: {picked.render()}", flush=True)

    # (c) The control: a place the arm cannot reach is refused, and the cube stays in the hand.
    z0 = reset()
    cube = _cube_mm(obj)
    grasp_at = cube + np.array([0.0, 0.0, _GRASP_LIFT_MM])
    picked = robot.pick(_pose(grasp_at), _CLOSE_TO_MM + 1.0, standoff_mm=_STANDOFF_MM, squeeze_mm=1.0,
                        pre_open_mm=_PRE_OPEN_MM, camera_world=decline)
    refused = robot.place(_pose(_UNREACHABLE_MM), standoff_mm=_STANDOFF_MM, camera_world=decline)
    arm.session.step_n(30)
    held_lift_mm = float(_cube_mm(obj)[2] - z0 * 1000.0)
    control = {
        "pick": picked.to_dict(), "place": refused.to_dict(), "cube_lift_mm": round(held_lift_mm, 1),
        "passed": bool(picked.ok and refused.outcome is HandlingOutcome.MOTION_REFUSED and held_lift_mm >= 30.0),
    }
    report["control"] = control
    print(f"CONTROL: pick={picked.outcome.value} place={refused.outcome.value} cube_lift_mm={held_lift_mm:.1f} "
          f"passed={control['passed']}", flush=True)
    print(f"CONTROL: {refused.render()}", flush=True)

    needed = max(1, int(0.8 * args.runs))
    claims = {"a_every_line_checked_and_declined": all_checked, "b_placed": placed, "b_needed": needed,
              "b_passed": placed >= needed, "c_control": control["passed"]}
    report["claims"] = claims
    target = Path(args.json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    ok = bool(claims["a_every_line_checked_and_declined"] and claims["b_passed"] and claims["c_control"])
    print(f"placed {placed}/{args.runs} within {args.tolerance_mm:g} mm (needs {needed}); "
          f"lines checked and declined in every run={all_checked}; control={control['passed']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"numbers written to {target}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
