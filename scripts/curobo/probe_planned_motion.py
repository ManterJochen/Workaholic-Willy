"""The planned-motion sidecar on the GPU: a clearance query refuses a graze, one request plans one way, never via the retract.

The planning of 85f082b, measured through the UR driver's own glue as ``probe_ur_planner_frame.py`` is: a
:class:`URRobotArm` built the way a cell builds it, with a stand-in RTDE connection that reports one joint vector and
moves nothing, and the planner it builds (``_curobo_planner``), which starts the sidecar on the GPU. Three claims:

* (g1) ``check_js`` at a clearance (S2.1): a box whose top stands 5 mm below the fingertips passes a check at no
  contact and is refused at ``clearance_mm=10``, and the same box 20 mm below passes both. The box is handed to the
  planner's world through the glue, in the controller's frame; the fingertips are placed from the hand model the
  guard places, as the frame probe places them.
* (g2) the seed (S2.2): the same joint plan asked three times in the same world returns the same trajectory, to
  1e-6 rad on every waypoint of every joint. The world holds a wall across the straight line, so cuRobo has to plan
  around it and its search is not the trivial one.
* (g3) the retract (S2.2): none of those plans passes through the retract the sidecar was started with (the pair's
  row of the committed table): no waypoint comes within 0.1 rad on every joint of it, while the start and the goal
  both stand at least 0.5 rad from it on some joint, and no joint swings more than 45 degrees past its span. Before
  85f082b the graph planner seeded every roadmap through the retract, and a plan was measured routed start, retract,
  goal on the owner's UR10 from the second attempt on.

Run from the repository root with the project venv, alone on the box (it starts the sidecar on the GPU)::

    .venv/Scripts/python.exe scripts/curobo/probe_planned_motion.py --json logs/port2/probe_planned_motion.json

Exit codes: ``0`` every claim holds, ``1`` one does not, ``2`` the planner could not be started or the scene could
not be placed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config.schema.robot import RobotConfig  # noqa: E402
from src.contracts import chosen  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: E402
from src.robot.safety.path_samples import span_overshoot_rad  # noqa: E402
from src.robot.safety.planning import CuroboUnavailableError  # noqa: E402
from src.robot.safety.planning.self_envelope import hand_spheres  # noqa: E402
from src.robot.safety.planning.world import planner_cuboid  # noqa: E402

_ARM = "ur5e"
_MARGIN_MM = 4.0
_DOWN_Q = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
#: Where the wall scenario starts and ends: the tool-down pose turned about the base either way.
_BASE_SWING_RAD = 0.9
_GAPS_MM = (20.0, 5.0)
_CLEARANCE_MM = 10.0
_PLANS = 3
_SAME_PLAN_TOL_RAD = 1e-6
_RETRACT_NEAR_RAD = 0.1
_RETRACT_FAR_RAD = 0.5
_DETOUR_DEG = 45.0


class _StandingConnection:
    """RTDE as the planner glue reads it: connected, standing at one joint vector, and moving nothing."""

    def __init__(self, joints: "tuple[float, ...]") -> None:
        self._joints = [float(v) for v in joints]

    @property
    def is_connected(self) -> bool:
        return True

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def moveJ(self, joints: Any, vel: Any = None, acc: Any = None) -> bool:  # noqa: N802 - ur_rtde's name
        raise AssertionError("this probe never moves anything")


def _arm(joints: "tuple[float, ...]") -> URRobotArm:
    config = RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": _ARM}, "motion": {"planner": "curobo"},
        "safety": {
            "self_collision": {"backend": "fcl", "planner_margin_mm": _MARGIN_MM},
            "planning_world": {"support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0],
                                                 "thickness_mm": 50.0}},
        },
        "gripper": {"model": "robotiq_2f85", "coupling_plates": [], "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 1.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    })
    arm = URRobotArm(config)
    arm._conn = _StandingConnection(joints)  # type: ignore[assignment]
    return arm


def _flange(q: Any) -> np.ndarray:
    frames = ur_link_transforms_mm(_ARM, np.asarray(q, dtype=np.float64))
    assert frames is not None
    return np.asarray(frames[-1], dtype=np.float64)


def _turned(q: "tuple[float, ...]", base: float) -> list[float]:
    return [float(q[0]) + base, *(float(v) for v in q[1:])]


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_planned_motion", description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=str, default="logs/port2/probe_planned_motion.json")
    args = parser.parse_args(argv)

    probe = _arm(_DOWN_Q)
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
    if approach[2] > -0.99:
        print(f"the hand does not point down at the probe's pose (approach {approach.round(3).tolist()})",
              file=sys.stderr)
        return 2
    tip = flange0[:3, 3] + approach * tip_mm
    report: dict[str, Any] = {"tip_in_controller_base_mm": [round(float(v), 2) for v in tip]}
    claims: dict[str, Any] = {}
    try:
        arm = _arm(_DOWN_Q)
        planner = arm._curobo_planner()
        try:
            client = planner._client_or_start()
            retract = getattr(getattr(client, "inner", client), "_default_q", None)
            report["retract_rad"] = None if retract is None else [round(float(v), 5) for v in retract]

            # (g1) a box under the fingertips, 20 mm and 5 mm below them, asked at no contact and at 10 mm.
            for gap in _GAPS_MM:
                top = float(tip[2]) - gap
                box = planner_cuboid(f"under_the_fingers_{gap:g}", (float(tip[0]), float(tip[1]), top - 25.0),
                                     (100.0, 100.0, 50.0))
                planner.set_world([box])
                none = planner.check_joint_path([list(_DOWN_Q)], refresh=False)
                cleared = planner.check_joint_path([list(_DOWN_Q)], refresh=False, clearance_mm=_CLEARANCE_MM)
                claims[f"g1_a_box_{gap:g}_mm_under_the_fingertips"] = {
                    "valid_at_no_contact": bool(none.valid), "valid_at_10_mm": bool(cleared.valid),
                    "reason_at_10_mm": str(cleared.reason)[:300],
                }

            # (g2, g3) a wall across the straight line from one side of the base to the other, planned three times.
            start, goal = _turned(_DOWN_Q, -_BASE_SWING_RAD), _turned(_DOWN_Q, _BASE_SWING_RAD)
            # 200 mm square about where the fingertips pass at the middle of the swing, from the bench to 100 mm above
            # them: the start and the goal stand far outside it, and the straight joint line runs through it.
            wall = planner_cuboid("across_the_line", (float(tip[0]), float(tip[1]), (float(tip[2]) + 100.0) / 2.0),
                                  (200.0, 200.0, float(tip[2]) + 100.0))
            planner.set_world([wall])
            samples = [[a + (b - a) * i / 20.0 for a, b in zip(start, goal)] for i in range(21)]
            line = planner.check_joint_path(samples, refresh=False)
            standing = cast(_StandingConnection, arm._conn)
            standing._joints = list(start)  # the glue reads the joints the connection reports at each plan
            plans = [planner.plan_joint(goal, refresh=False) for _ in range(_PLANS)]
            report["wall_line_valid_at_no_contact"] = bool(line.valid)
            report["wall_line_reason"] = str(line.reason)[:300]
            if any(not plan for plan in plans):
                claims["g2_the_same_plan_each_time"] = {"planned": [bool(plan) for plan in plans]}
                claims["g3_no_plan_through_the_retract"] = {"planned": [bool(plan) for plan in plans]}
            else:
                arrays = [np.asarray(plan, dtype=np.float64) for plan in plans]
                same = all(a.shape == arrays[0].shape for a in arrays)
                worst = max((float(np.max(np.abs(a - arrays[0]))) for a in arrays[1:]), default=0.0) if same else None
                claims["g2_the_same_plan_each_time"] = {
                    "waypoints": [len(a) for a in arrays], "same_shape": same, "max_difference_rad": worst,
                }
                g3: dict[str, Any] = {"measured": retract is not None}
                if retract is not None:
                    r = np.asarray(retract, dtype=np.float64)
                    g3["start_from_retract_rad"] = round(float(np.max(np.abs(np.asarray(start) - r))), 4)
                    g3["goal_from_retract_rad"] = round(float(np.max(np.abs(np.asarray(goal) - r))), 4)
                    g3["nearest_to_retract_rad"] = [round(float(np.min(np.max(np.abs(a - r), axis=1))), 4)
                                                    for a in arrays]
                g3["worst_overshoot_deg"] = [round(math.degrees(max(span_overshoot_rad(a.tolist()))), 2)
                                             for a in arrays]
                g3["ends_on_the_goal_rad"] = [round(float(np.max(np.abs(a[-1] - np.asarray(goal)))), 6) for a in arrays]
                claims["g3_no_plan_through_the_retract"] = g3
        finally:
            planner.close()
    except CuroboUnavailableError as exc:
        print(f"the planner could not be started: {exc}", file=sys.stderr)
        return 2
    report["claims"] = claims
    g1_near = claims["g1_a_box_5_mm_under_the_fingertips"]
    g1_far = claims["g1_a_box_20_mm_under_the_fingertips"]
    g2 = claims["g2_the_same_plan_each_time"]
    g3 = claims["g3_no_plan_through_the_retract"]
    passed = {
        "g1": g1_near["valid_at_no_contact"] is True and g1_near["valid_at_10_mm"] is False
        and g1_far["valid_at_no_contact"] is True and g1_far["valid_at_10_mm"] is True,
        "g2": g2.get("same_shape") is True and g2.get("max_difference_rad") is not None
        and g2["max_difference_rad"] <= _SAME_PLAN_TOL_RAD,
        "g3": bool(g3.get("measured")) and g3["start_from_retract_rad"] >= _RETRACT_FAR_RAD
        and g3["goal_from_retract_rad"] >= _RETRACT_FAR_RAD
        and all(near > _RETRACT_NEAR_RAD for near in g3["nearest_to_retract_rad"])
        and all(deg <= _DETOUR_DEG for deg in g3["worst_overshoot_deg"])
        and all(off <= 1e-3 for off in g3["ends_on_the_goal_rad"]),
    }
    report["passed"] = passed
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for key, value in claims.items():
        print(f"{key}: {value}")
    ok = all(passed.values())
    print(f"S2 planned motion: {passed} -> {'PASS' if ok else 'FAIL'}")
    print(f"numbers written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
