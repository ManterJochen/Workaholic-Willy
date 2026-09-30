"""The band admission re-judges with the camera's boxes at their own distance: F1 and F2 together (2026-09-30).

F1: a sample the planner's padded spheres refuse on a self pair the exact guard judges is left to that guard, which
judges the very sample again (``URRobotArm._exact_guard_decides``). F2: the exact guard holds each camera-seen box turned,
as cuRobo holds it, at ``perceived_min_distance_mm`` (5 mm), while a declared fixture and the arm itself keep
``min_distance_mm`` (10 mm). Merged, the re-judgement is the whole exact guard with both distances:

* a seen box 7 mm from the arm at LOOK[0] leaves the band's refusal to the guard, which accepts it;
* a seen box 3 mm from it keeps the refusal standing, on the guard's own sentence naming the box;
* the same 7 mm box declared as a fixture keeps it standing at 10 mm.

On a real move the path gate judges every sample first and refuses those samples before the admission is asked; this
holds the admission itself, which re-judges every sample it lifts whatever the caller judged (review of F1).
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.robot.core import MotionCommand, MotionStatus
from src.robot.safety._capsule import AxisAlignedBox, TurnedBox
from src.robot.safety._fcl_self_collision import mesh_backend_status
from src.robot.safety._ur_kinematics import ur_link_origins_mm, ur_link_transforms_mm
from tests.test_a_pose_only_the_planners_spheres_refuse_is_the_exact_guards import LOOK0, BandPlanner

_HALF_MM = 20.0


def _arm(fixtures: "list[dict] | None" = None) -> Any:
    """The owner-like UR10 with the Hand-E at LOOK[0], its planner the band double, ``fixtures`` declared."""
    if mesh_backend_status("ur10", mesh_name="robotiq_hande") != "ok":
        raise unittest.SkipTest("no exact mesh backend on this box")
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE

    arm: Any = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "workspace_limits": OPEN_WORKSPACE,
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0, "fixtures": list(fixtures or [])}},
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(LOOK0)
    planner = BandPlanner()
    planner.here = list(LOOK0)
    arm._curobo_ur = planner
    return arm


def _box(centre: np.ndarray, yaw_deg: float, name: str) -> AxisAlignedBox:
    """A 40 mm cube at ``centre`` turned ``yaw_deg`` about BASE Z, as the camera world hands it to the guard."""
    half = np.full(3, _HALF_MM)
    c, s = abs(math.cos(math.radians(yaw_deg))), abs(math.sin(math.radians(yaw_deg)))
    enclosure = np.array([c * half[0] + s * half[1], s * half[0] + c * half[1], half[2]])
    turned = None if yaw_deg == 0.0 else TurnedBox(half_extents_mm=half, yaw_rad=math.radians(yaw_deg))
    return AxisAlignedBox(center_mm=np.asarray(centre, dtype=np.float64), half_extents_mm=enclosure, name=name,
                          turned=turned)


def _box_at(arm: Any, distance_mm: float, *, yaw_deg: float, name: str) -> AxisAlignedBox:
    """The cube over the forearm's middle at LOOK[0], raised until the exact meshes keep ``distance_mm`` to it."""
    guard = arm._preflight._path_authority(arm)
    backend = guard._exact_mesh_backend("ur10")
    transforms = ur_link_transforms_mm("ur10", np.asarray(LOOK0, dtype=np.float64))
    origins = ur_link_origins_mm("ur10", np.asarray(LOOK0, dtype=np.float64))
    assert origins is not None and transforms is not None
    middle = (np.asarray(origins[2], dtype=np.float64) + np.asarray(origins[3], dtype=np.float64)) / 2.0

    def nearer(offset: float, limit: float) -> bool:
        box = _box(middle + np.array([0.0, 0.0, offset]), yaw_deg, name)
        return backend.evaluate(transforms, 0.0, (box,), limit, arm_pairs=False) is not None

    low, high = 0.0, 400.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if nearer(mid, distance_mm):
            low = mid
        else:
            high = mid
    assert nearer(high, distance_mm + 0.05) and not nearer(high, distance_mm - 0.05), "the cube is not where it says"
    return _box(middle + np.array([0.0, 0.0, high]), yaw_deg, name)


def _admission(arm: Any) -> Any:
    """The driver's own decision on LOOK[0], which the band double refuses on forearm|wrist_2."""
    planner = arm._curobo_ur
    verdict = planner.check_joint_path([LOOK0], refresh=False)
    assert not verdict.valid
    return arm._exact_guard_decides(planner, [LOOK0], verdict, clearance_mm=0.0, command=MotionCommand.MOVE_JOINTS)


class TheAdmissionJudgesWithTheCamerasBoxesTests(unittest.TestCase):
    def test_a_seen_box_7_mm_from_the_arm_leaves_the_band_to_the_guard(self) -> None:
        """⭐ 7 mm is past the 5 a seen box is kept at: the guard accepts, so it decides the band's pair."""
        arm = _arm()
        arm._preflight.set_perceived_obstacles([_box_at(arm, 7.0, yaw_deg=30.0, name="seen_07")])
        self.assertIsNone(_admission(arm))

    def test_a_seen_box_3_mm_from_the_arm_keeps_the_refusal_standing_on_the_guards_own_sentence(self) -> None:
        """⭐ THE MERGE: the admission's re-judgement holds the camera's turned box at 5 mm, and says so."""
        arm = _arm()
        arm._preflight.set_perceived_obstacles([_box_at(arm, 3.0, yaw_deg=30.0, name="seen_03")])
        standing = _admission(arm)
        assert standing is not None
        for said in ("the exact guard refuses it too", "fixture:seen_03", "< 5.000 mm"):
            self.assertIn(said, standing.reason)
        self.assertIs(standing.status, MotionStatus.SELF_COLLISION_REJECTED)

    def test_the_same_box_declared_keeps_the_refusal_standing_at_10_mm(self) -> None:
        """A declared fixture is measured geometry and keeps the 10 mm the arm keeps: the 7 mm box passes as seen and
        keeps the band's refusal standing once it is declared."""
        seen = _box_at(_arm(), 7.0, yaw_deg=0.0, name="post")
        arm = _arm()
        arm._preflight.set_perceived_obstacles([seen])
        self.assertIsNone(_admission(arm), "as a seen box it is 7 mm away, past 5")
        declared = _arm([{"name": "post", "center_mm": [float(v) for v in seen.center_mm],
                          "half_extents_mm": [float(v) for v in seen.half_extents_mm]}])
        standing = _admission(declared)
        assert standing is not None
        for said in ("the exact guard refuses it too", "fixture:post", "< 10.000 mm"):
            self.assertIn(said, standing.reason)


if __name__ == "__main__":
    unittest.main()
