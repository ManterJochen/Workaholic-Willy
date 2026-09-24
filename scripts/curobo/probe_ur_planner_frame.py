"""The real UR's planner sees the cell where the controller has it, measured through the UR driver's own glue.

The planner is rooted at UR's URDF ``base_link``, the DH base turned half a turn about Z
(``probe_payload_attach.py`` ``frame`` measures 0.0001 mm with the turn, 1019 mm without). A UR cell
declares its geometry and commands its goals in the frame the controller reports in, the DH base.
This builds a :class:`URRobotArm` the way a cell does, with a stand-in RTDE connection that reports
one joint vector and moves nothing, and asks the planner the arm builds (``_curobo_ur_planner``):

* (f1) a box declared in the cell's config, in the controller's frame, 20 mm into the fingers makes
  the arm's own configuration invalid;
* (f2) on a second arm with no declared box, the same box turned half a turn about the base axis and
  handed to the planner's ``set_world`` leaves it valid;
* (f3) on that arm, a joint plan to the configuration nearest the arm of the flange raised 100 mm, solved in
  the controller's frame as the driver solves it, ends on it: the DH flange of the plan's last configuration is
  within 1 mm and 0.1 degrees of the goal. The UR driver never asks cuRobo for a Cartesian plan, so the goal a
  plan is handed is joints, the same numbers in both bases, and this claim holds the joint remap both ways;
* (f4) on that arm, a live scene field written in the controller's frame, the grid the cell reserves
  centred 150 mm along the controller's +X from the fingertips and inside an obstacle only in a
  200 mm cube around them, makes the configuration invalid through the glue, and the same field
  handed to the planner unturned, which is the wire without the turn, leaves it valid. A planner
  that moved the grid's centre and ignored its turn would put the cube 300 mm from the fingertips;
  measured, it does not: the planner honours a grid's orientation.

The cube is 200 mm because an open 2F-85 has nothing at its TCP. Its fingertips stand about 45 mm
either side of it, so a 90 mm cube centred there sits in the gap between them and the planner
correctly sees nothing; a 200 mm cube meets the fingers.

Without the turn the claims measure: (f1) valid, the box in the fingers a metre away in the
planner's frame; (f2) refused, the turned box on the hand; (f3), when it was a Cartesian plan,
planned in 81 waypoints and ending 1019.28 mm and 180 degrees from its goal, the flange sent to the
goal mirrored through the base axis; (f4) valid, the field a metre away, in a run whose 90 mm cube
also fit between the open fingers. That is the measurement the turn answers.

Run from the repository root with the project venv, alone on the box (it starts the sidecar on the
GPU)::

    .venv/Scripts/python.exe scripts/curobo/probe_ur_planner_frame.py --json logs/step8/probe_ur_planner_frame.json

Exit codes: ``0`` every claim holds, ``1`` one does not, ``2`` the planner could not be started.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config.schema.robot import RobotConfig  # noqa: E402
from src.contracts import chosen  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.drivers.ur.planner_frame import PlannerFrameClient  # noqa: E402
from src.robot.safety._ur_ik import nearest_goals, ur_flange_ik  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.planning import CuroboUnavailableError  # noqa: E402
from src.robot.safety.planning.reservation import PlannerReservation  # noqa: E402
from src.robot.safety.planning.self_envelope import hand_spheres  # noqa: E402
from src.robot.safety.planning.world import planner_cuboid  # noqa: E402

_ARM = "ur5e"
_MARGIN_MM = 4.0
_DOWN_Q = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
_INTO_FINGERS_MM = 20.0
_RAISE_MM = 100.0
_FIELD_VOXEL_MM = 20.0
_FIELD_OFFSET_MM = 150.0
_FIELD_CUBE_HALF_MM = 100.0


class _StandingConnection:
    """RTDE as the planner glue reads it: connected, standing at one joint vector, and moving nothing."""

    def __init__(self, joints: "tuple[float, ...]") -> None:
        self._joints = [float(v) for v in joints]

    @property
    def is_connected(self) -> bool:
        return True

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def moveJ(self, joints: Any, vel: Any = None, acc: Any = None) -> bool:  # noqa: N802 (ur_rtde's name)
        raise AssertionError("this probe never moves anything")


def _cell(fixtures: "list[dict[str, Any]]", *, field: bool) -> RobotConfig:
    world: dict[str, Any] = {"enabled": True, "payload": {"enabled": False},
                             "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0}}
    if field:
        world["perceived"] = {"voxel_field_mm": _FIELD_VOXEL_MM}
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": _ARM, "motion_planner": "curobo"},
        "safety": {
            "payload": {"enforce": False},
            "self_collision": {"backend": "fcl", "kinematics_model": _ARM, "planner_margin_mm": _MARGIN_MM,
                               "fixtures": fixtures},
            "planning_world": world,
        },
        "gripper": {"model": "robotiq_2f85", "coupling_plates": [], "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 1.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    })


def _arm(fixtures: "list[dict[str, Any]]", *, field: bool = False) -> URRobotArm:
    arm = URRobotArm(_cell(fixtures, field=field))
    arm._conn = _StandingConnection(_DOWN_Q)  # type: ignore[assignment]
    return arm


def _flange(q: Any) -> np.ndarray:
    frames = ur_link_transforms_mm(_ARM, np.asarray(q, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1], dtype=np.float64)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _field(reservation: PlannerReservation) -> "tuple[np.ndarray, list[float], float]":
    """A field on the reserved grid, inside an obstacle only in a cube at local ``(-offset, 0, 0)``."""
    assert reservation.voxel_grid, "the cell reserves no live scene grid"
    x, y, z, voxel = (float(v) * 1000.0 for v in reservation.voxel_grid.split(","))
    axes = [(np.arange(int(round(side / voxel))) + 0.5) * voxel - side / 2.0 for side in (x, y, z)]
    lx, ly, lz = np.meshgrid(*axes, indexing="ij")
    inside = ((np.abs(lx + _FIELD_OFFSET_MM) < _FIELD_CUBE_HALF_MM)
              & (np.abs(ly) < _FIELD_CUBE_HALF_MM) & (np.abs(lz) < _FIELD_CUBE_HALF_MM))
    # Half a metre outside the cube: a field is a distance, and a link sphere wider than the value it
    # reads collides, so a small positive value outside would refuse the arm's own upper arm wherever
    # the grid reaches it.
    return np.where(inside, -0.03, 0.5).reshape(-1), [x / 1000.0, y / 1000.0, z / 1000.0], voxel / 1000.0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_ur_planner_frame", description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default="logs/step8/probe_ur_planner_frame.json")
    args = parser.parse_args(argv)

    # Where the fingertips stand in the controller's frame, from the hand model the guard places.
    probe = _arm([])
    hand = probe._preflight.planner_hand(probe)
    if not chosen(hand) or not chosen(hand.placement):
        print("the cell places no hand on the flange, so there are no fingertips to measure from", file=sys.stderr)
        return 2
    spheres = hand_spheres(hand, _ARM)
    assert spheres is not None
    approach_tool0 = np.asarray(hand.placement.approach_in_tool0, dtype=np.float64)
    tip_mm = max(float(np.dot(np.asarray(s.start_mm), approach_tool0)) + float(s.radius_mm) for s in spheres)
    flange0 = _flange(_DOWN_Q)
    approach = flange0[:3, :3] @ approach_tool0
    tip = flange0[:3, 3] + approach * tip_mm
    report: dict[str, Any] = {"tip_in_controller_base_mm": [round(float(v), 2) for v in tip],
                              "approach_in_controller_base": [round(float(v), 4) for v in approach]}
    top = float(tip[2]) + _INTO_FINGERS_MM
    into = {"name": "into_the_fingers", "center_mm": (float(tip[0]), float(tip[1]), top - 25.0),
            "half_extents_mm": (50.0, 50.0, 25.0)}
    claims: dict[str, Any] = {}
    try:
        # (f1): the declared world, through the cell's own config and the planner's registration.
        planner = _arm([into])._curobo_ur_planner()
        try:
            verdict = planner.check_joint_path([list(_DOWN_Q)])
            claims["f1_a_declared_box_in_the_fingers"] = {"valid": bool(verdict.valid), "reason": str(verdict.reason)}
        finally:
            planner.close()

        arm = _arm([], field=True)
        planner = arm._curobo_ur_planner()
        try:
            # (f3) first, on a world with nothing near the hand.
            goal = flange0.copy()
            goal[:3, 3] = flange0[:3, 3] - approach * _RAISE_MM
            lower, upper = arm._goal_joint_window()
            solutions = ur_flange_ik(_ARM, goal, q6_if_singular=_DOWN_Q[-1]) or ()
            nearest = nearest_goals(solutions, current=_DOWN_Q, lower=lower, upper=upper, velocity=1.0)
            traj = planner.plan_joint(list(nearest[0].joints)) if nearest else None
            if not traj:
                claims["f3_the_plan_ends_on_its_goal"] = {"planned": False}
            else:
                end = _flange(traj[-1])
                claims["f3_the_plan_ends_on_its_goal"] = {
                    "planned": True, "waypoints": len(traj),
                    "end_error_mm": round(float(np.linalg.norm(end[:3, 3] - goal[:3, 3])), 3),
                    "end_error_deg": round(_angle_deg(end[:3, :3], goal[:3, :3]), 3),
                }
            # (f4): the live scene channel, written as the live world writes it, in the planner's voxel order.
            field, dims_m, voxel_m = _field(PlannerReservation.from_config(robot_cfg=arm.config))
            path = Path(tempfile.gettempdir()) / "willy_probe_ur_planner_frame.npy"
            np.save(path, field.astype(np.float16))
            centre = tip + np.array([_FIELD_OFFSET_MM, 0.0, 0.0])
            registered = planner._client_or_start().set_voxels(
                str(path), dims_m=dims_m, voxel_size_m=voxel_m,
                pose=[*(float(c) / 1000.0 for c in centre), 1.0, 0.0, 0.0, 0.0],
            )
            verdict = planner.check_joint_path([list(_DOWN_Q)], refresh=False)
            claims["f4_a_field_on_the_fingers"] = {"registered": registered, "valid": bool(verdict.valid),
                                                   "reason": str(verdict.reason)}
            # The control half: the same field, the same numbers, handed to the planner past the turn.
            inner = cast(PlannerFrameClient, planner._client_or_start()).inner
            unturned = inner.set_voxels(
                str(path), dims_m=dims_m, voxel_size_m=voxel_m,
                pose=[*(float(c) / 1000.0 for c in centre), 1.0, 0.0, 0.0, 0.0],
            )
            verdict = inner.check_joints([list(_DOWN_Q)])
            claims["f4_control_the_field_unturned"] = {"registered": unturned, "valid": bool(verdict.valid)}
            # (f2): a world handed in at run time; it replaces the field, and the declared plane goes underneath.
            turned = planner_cuboid("turned_half_a_turn", (-float(tip[0]), -float(tip[1]), top - 25.0),
                                    (100.0, 100.0, 50.0))
            planner.set_world([turned])
            verdict = planner.check_joint_path([list(_DOWN_Q)], refresh=False)
            claims["f2_the_same_box_turned"] = {"valid": bool(verdict.valid), "reason": str(verdict.reason)}
        finally:
            planner.close()
    except CuroboUnavailableError as exc:
        print(f"the planner could not be started: {exc}", file=sys.stderr)
        return 2
    report["claims"] = claims
    f3 = claims["f3_the_plan_ends_on_its_goal"]
    f4 = claims["f4_a_field_on_the_fingers"]
    passed = {
        "f1": claims["f1_a_declared_box_in_the_fingers"]["valid"] is False,
        "f2": claims["f2_the_same_box_turned"]["valid"] is True,
        "f3": bool(f3.get("planned")) and f3["end_error_mm"] < 1.0 and f3["end_error_deg"] < 0.1,
        "f4": f4["registered"] is not None and f4["valid"] is False
        and claims["f4_control_the_field_unturned"]["valid"] is True,
    }
    report["passed"] = passed
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for key, value in claims.items():
        print(f"{key}: {value}")
    ok = all(passed.values())
    print(f"planner frame: {passed}, {'PASS' if ok else 'FAIL'}")
    print(f"numbers written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
