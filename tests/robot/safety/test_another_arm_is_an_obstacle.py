"""Another arm of the cell is an obstacle every sample of this arm's motion keeps the guard's distance from.

Owner, 2026-10-15: a second arm is optional, and where there is one, the guard's distances hold to it too. Each arm's
guard describes its own body where it stands, every part's convex hull and its base (``body_hulls``); the moving arm's
guard holds the other's parts, stood in its own base by where the two stand in the cell (``partner_bodies``), and the
exact mesh guard judges the hulls, not the boxes about them, since two hands at work come close.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.core import JointPositions, MotionCommand
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety.guard import SafetyContext
from src.robot.safety.partner_arm import own_from_other, partner_bodies
from tests._arm_kit import kit_tree, planned_arm

#: Where both arms stand in these tests, radians: reaching out and down, the hand clear of everything of its own.
_Q = (0.0, -1.2, 1.3, -0.4, 1.5, 0.2)
#: Where a motion to ``_Q`` starts: the arm upright.
_UP = (0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)


def _arm(backend: str = "fcl") -> tuple[Any, Any]:
    config = kit_tree("ik", safety={"self_collision": {"backend": backend}},
                      gripper={"model": "robotiq_2f85", "tool_frame": {"source": "willy", "offset_mm": (0.0, 0.0, 132.0)}})
    arm = planned_arm(config)
    return arm, next(guard for guard in arm.safety_preflight.guards if guard.name == "self_collision")


def _judge(guard: Any, arm: Any, joints: tuple[float, ...]) -> Any:
    return guard.evaluate(SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(list(joints)),
                                        arm=arm))


def _standing(shift_mm: Any) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = shift_mm
    return matrix


class AnotherArmIsAnObstacleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.halter, self.guard = _arm()
        self.werker, werker_guard = _arm()
        self.own = self.guard.body_hulls(self.halter, _Q)
        self.other = werker_guard.body_hulls(self.werker, _Q)

    def where(self, own_part: str, other_part: str) -> np.ndarray:
        """The werker's base in the cell (the halter's base), so its ``other_part`` stands on the halter's ``own_part``."""
        return _standing(self.own[own_part].mean(axis=0) - self.other[other_part].mean(axis=0))

    def hold(self, werker_in_cell: np.ndarray) -> None:
        self.guard.set_partner_bodies(partner_bodies("werker", self.other, own_from_other(np.eye(4), werker_in_cell)))

    def test_each_arm_describes_its_parts_and_its_base(self) -> None:
        self.assertEqual(sorted(self.other), ["base", "forearm", "gripper", "lfinger", "rfinger", "shoulder",
                                              "upper_arm", "wrist_1", "wrist_2", "wrist_3"])

    def test_the_other_arm_where_this_hand_goes_is_refused_naming_its_part(self) -> None:
        self.assertTrue(_judge(self.guard, self.halter, _Q).accepted, "alone, the pose is free")
        self.hold(self.where("gripper", "forearm"))
        decision = _judge(self.guard, self.halter, _Q)
        self.assertFalse(decision.accepted)
        self.assertIn("|fixture:werker:", decision.message)

    def test_the_other_arm_far_off_refuses_nothing_and_none_held_is_none(self) -> None:
        self.hold(_standing((2500.0, 0.0, 0.0)))
        self.assertTrue(_judge(self.guard, self.halter, _Q).accepted)
        self.assertEqual(len(self.guard.partner_bodies), 10)
        self.hold(self.where("gripper", "forearm"))
        self.guard.set_partner_bodies(())
        self.assertTrue(_judge(self.guard, self.halter, _Q).accepted)

    def test_the_hull_is_judged_not_the_box_about_it(self) -> None:
        """A thin rod whose box holds the hand while the rod itself passes 600 mm from it."""
        hand = self.own["gripper"].mean(axis=0)
        centre = hand + np.array([-380.0, 380.0, 380.0])
        ends = [centre + sign * np.array([400.0, 400.0, 400.0]) for sign in (-1.0, 1.0)]
        rod = np.array([end + np.array(corner) for end in ends
                        for corner in ((5, 5, 5), (5, 5, -5), (5, -5, 5), (5, -5, -5),
                                       (-5, 5, 5), (-5, 5, -5), (-5, -5, 5), (-5, -5, -5))], dtype=np.float64)
        box = dict(center_mm=centre, half_extents_mm=np.full(3, 405.0), name="werker:rod")
        self.guard.set_partner_bodies((AxisAlignedBox(**box, hull_mm=rod),))
        self.assertTrue(_judge(self.guard, self.halter, _Q).accepted)
        self.guard.set_partner_bodies((AxisAlignedBox(**box),))
        self.assertFalse(_judge(self.guard, self.halter, _Q).accepted)

    def test_a_path_through_the_other_arm_is_suspect_before_its_end(self) -> None:
        path = np.linspace(_UP, _Q, 24)
        self.assertEqual(self.guard.first_suspect(self.halter, path, len(path)), len(path))
        self.hold(self.where("gripper", "forearm"))
        self.assertLess(self.guard.first_suspect(self.halter, path, len(path)), len(path))

    def test_the_capsule_proxy_judges_the_boxes_about_the_hulls(self) -> None:
        arm, capsule = _arm(backend="capsule")
        self.assertIsNone(capsule.body_hulls(arm, _Q), "the proxy cannot describe its own body")
        capsule.set_partner_bodies(partner_bodies("werker", self.other, self.where("gripper", "forearm")))
        self.assertFalse(_judge(capsule, arm, _Q).accepted)


if __name__ == "__main__":
    unittest.main()
