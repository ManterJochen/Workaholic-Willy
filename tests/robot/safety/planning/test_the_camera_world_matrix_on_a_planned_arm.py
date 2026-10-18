"""The camera-world stamp matrix, run once on the planning half every planned arm shares (P0g slice 12).

What stood behind a motion (``CameraWorldStamp``) is decided in ``PlannedArm``: the ik route stamps UNPLANNED, a
cuRobo motion with no live world is refused MISSING, a live world whose refresh reported nothing stays UNSTATED, and
one whose refresh vouched says PLANNED. Until this slice the UR driver's rows in
``tests/robot/drivers/test_camera_world_on_every_driver.py`` were where that was held, so a second vendor would have needed its own
copy of every row. The matrix runs here on a bare ``PlannedArm`` over the fake transport, the expected stamps as
literals, and a driver's own file asserts only its stance: which route its tree selects.

The arms are the kit's (``tests/_arm_kit.ik_arm``, ``curobo_arm``): the cuRobo rows plan through the kit's planner
double with the plan, path and endpoint gates stood down, because this file reads the stamp and not the plan; the
planned path, the plan's end and the endpoint are held by their own files.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import (
    CameraWorldStamp,
    CameraWorldUse,
    JointPositions,
    MotionResult,
    MotionStatus,
)
from tests._arm_kit import HERE, PlannerDouble, curobo_arm, ik_arm, kit_pose

_JOINTS = JointPositions(HERE)

_IK = CameraWorldStamp.unplanned("robot.motion.planner is 'ik'")
_NO_WORLD = CameraWorldStamp.missing(
    "robot.motion.planner is 'curobo' and no live camera world is wired to this arm")
_VOUCHED = CameraWorldStamp.planned(cameras=("overhead",), captured_at_s=100.0)


class _Refreshing(PlannerDouble):
    """A planner whose world refresh vouched: one fresh world from the overhead camera, captured at 100 s."""

    last_world_refresh: object = None

    def refresh_world(self, *, near_point_mm: object = None, goal_keep_out: object = None) -> None:
        from src.robot.safety.planning.live_world import WorldRefresh, WorldVerdict

        self.last_world_refresh = WorldRefresh(
            verdict=WorldVerdict.FRESH, sent=1, registered=1, build_ms=0.0, register_ms=0.0, age_ms=5.0,
            dropped_obstacles=0, cameras=("overhead",), captured_at_s=100.0,
        )


#: Each cell: its name, the arm it runs on, the motion, and the stamp it must carry, as a literal.
CELLS: tuple[tuple[str, Any, Any, CameraWorldStamp], ...] = (
    ("ik move", ik_arm, lambda arm: arm.move(kit_pose()), _IK),
    ("ik move_to_joints", ik_arm, lambda arm: arm.move_to_joints(_JOINTS), _IK),
    ("curobo move, no live world", curobo_arm, lambda arm: arm.move(kit_pose()), _NO_WORLD),
    ("curobo move_to_joints, no live world", curobo_arm, lambda arm: arm.move_to_joints(_JOINTS), _NO_WORLD),
    ("curobo joint move whose refresh vouched",
     lambda: curobo_arm(_Refreshing(plan_result=[_JOINTS.tolist()]), live_world=object()),
     lambda arm: arm.move_to_joints(_JOINTS), _VOUCHED),
)


class TheStampMatrixOnAPlannedArmTests(unittest.TestCase):
    def test_every_route_and_verb_says_what_stood_behind_the_motion(self) -> None:
        for label, build, command, expected in CELLS:
            with self.subTest(label):
                result = command(build())
                self.assertIsInstance(result, MotionResult)
                self.assertEqual(expected, result.camera_world)

    def test_the_ik_route_moves_what_it_stamps(self) -> None:
        """The control: the ik cells ran their motion to the end, so the stamp is on an executed result."""
        self.assertIs(MotionStatus.EXECUTED, ik_arm().move(kit_pose()).status)
        self.assertIs(MotionStatus.EXECUTED, ik_arm().move_to_joints(_JOINTS).status)

    def test_a_vouched_joint_move_executed(self) -> None:
        arm = curobo_arm(_Refreshing(plan_result=[_JOINTS.tolist()]), live_world=object())
        result = arm.move_to_joints(_JOINTS)
        self.assertIs(MotionStatus.EXECUTED, result.status, result.message)

    def test_a_live_world_whose_planner_reports_no_refresh_stays_unstated(self) -> None:
        """Nothing vouched: this planner double reports no refresh, so the row cannot say PLANNED."""
        result = curobo_arm(live_world=object()).move(kit_pose())
        self.assertIs(MotionStatus.EXECUTED, result.status)
        self.assertIs(CameraWorldUse.UNSTATED, result.camera_world.use)


if __name__ == "__main__":
    unittest.main()
