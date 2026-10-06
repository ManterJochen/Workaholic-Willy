"""A UR arm drives a joint target on its straight joint line and nothing else, where a caller asks for only that.

The owner, 2026-09-29: the one view a wrist pick generates, and the move back to the look that saw its part, never
plan with cuRobo, at any angle. A "safety" move that drives through the retract pose or takes a strange detour costs
time instead of saving it and alarms whoever stands at the cell. So these two motions run only on the straight joint
line from where the arm stands to their target, judged against the camera world as every joint move is, and a line
that is not clear is refused: the caller skips that angle, or approaches from where the arm stands.

``move_to_joints_on_the_line`` (:class:`~src.robot.core.arm_capabilities.DrivesJointLines`) is ``move_to_joints`` less
the plan around the line: the same twin inside the cable window, the same destination guards, the same line judged by
both authorities at ``safety.planned_motion.line_clearance_mm``, and one ``moveJ`` where it passes. A planner double
that fails the test whenever it is asked to plan holds that cuRobo never is. An arm whose paths nobody judges (the ik
planner) refuses the line: a line nobody judged is not the line the owner allows. The control: ``move_to_joints`` still
plans around the same colliding line, as a declared look does. And a line a pick made up is judged against the camera
world holding every frame of the pick or not run: inside a program's ``without_camera_world`` block on an arm with no
world wired, the generated view, the move back and the motion verb that drives them send nothing.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionStatus, RobotArm
from src.robot.core.arm_capabilities import DrivesJointLines
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.kuka.arm import KukaRobotArm
from src.robot.drivers.sim.arm import IsaacRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution import motion
from src.robot.execution.generated_view import aim_at_faces, go_to_generated_view, move_back_to_look
from src.robot.grasping.multiview.unseen_side import JawFace
from tests._plan_end import OPEN_WORKSPACE
from tests._route_planner import RoutePlanner
from tests._wrist_views import CUBE_CENTRE, HEIGHT, WIDTH, K, camera_looking_at, tool_for

_DECLINED = "unit double: this file reads which line a joint target runs on, and no camera world is wired"

#: Clear, and where the fake controller says the arm is standing (tests/test_ur_checked_joint_verbs.py).
_HERE = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
#: Clear, a short joint move away.
_THERE = [0.2, -1.5, 1.5, 0.0, 0.0, 0.0]


class _NeverPlans(RoutePlanner):
    """The planner double of the route tests, which fails the test the moment it is asked to plan."""

    def __init__(self, test: unittest.TestCase, *, here: list[float]) -> None:
        super().__init__(here=here)
        self.test = test

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> Any:
        self.test.fail(f"cuRobo was asked to plan to {list(goal)}: a straight joint line only is never planned around")


def _arm(planner: RoutePlanner, *, motion_planner: str = "curobo") -> URRobotArm:
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": motion_planner},
        "safety": {"payload": {"enforce": False}}, "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": OPEN_WORKSPACE,
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(planner.here)
    arm._conn.moveJ.return_value = True
    arm._conn.fk.return_value = [0.4, 0.0, 0.3, 0.0, 3.14159, 0.0]
    arm._curobo_ur = planner  # type: ignore[assignment]
    # The local path gate has its own tests (tests/test_ur_checked_joint_verbs.py); the destination guards stay real.
    arm._preflight.gate_planned_path = lambda waypoints, *, arm=None, command=None: None  # type: ignore[method-assign]
    return arm


def _sent(arm: URRobotArm) -> list[list[float]]:
    return [list(call.args[0]) for call in arm._conn.moveJ.call_args_list]


class TheStraightJointLineOnlyTests(unittest.TestCase):
    def test_a_clear_line_is_one_movej_to_the_target_and_the_line_judged_is_the_one_run(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints_on_the_line(JointPositions(_THERE))

        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, result.command)
        self.assertEqual([_THERE], _sent(arm), "the executed joint path is one moveJ, the straight joint line")
        (lined,) = planner.lines()
        np.testing.assert_allclose(lined[1][0], _HERE, atol=0.0)
        np.testing.assert_allclose(lined[1][-1], _THERE, atol=0.0)
        self.assertEqual(3.0, lined[3], "the line was not judged at safety.planned_motion.line_clearance_mm")
        self.assertEqual([], planner.named("execute"), "a planned route ran")

    def test_a_line_that_is_not_clear_is_refused_and_nothing_is_planned_around_it(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints_on_the_line(JointPositions(_THERE))

        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status, result.message)
        self.assertIn("the planner refused this straight joint line", result.message or "")
        self.assertIn("nothing is planned around it", result.message or "")
        arm._conn.moveJ.assert_not_called()
        self.assertEqual([], planner.named("execute"))

    def test_move_to_joints_still_plans_around_the_same_line(self) -> None:
        """The control: a declared look keeps today's rule, the line first and cuRobo around it."""
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))

        self.assertTrue(result.ok, result.message)
        self.assertEqual(1, len(planner.named("plan_joint")))

    def test_the_destination_guards_still_refuse_first(self) -> None:
        """A target in degrees is refused by the joint-limit guard before any line is sampled, as on every joint move."""
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)

        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints_on_the_line(JointPositions([0.0, -90.0, 90.0, -90.0, -90.0, 0.0]))

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertEqual([], planner.lines())
        arm._conn.moveJ.assert_not_called()

    def test_an_arm_whose_paths_nobody_judges_refuses_the_line(self) -> None:
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner, motion_planner="ik")

        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints_on_the_line(JointPositions(_THERE))

        self.assertIs(MotionStatus.UNSUPPORTED, result.status, result.message)
        self.assertIn("nobody judges", result.message or "")
        arm._conn.moveJ.assert_not_called()

    def test_inside_a_declined_camera_world_nothing_a_pick_made_up_is_sent(self) -> None:
        """A UR with no camera world wired, inside a program's decline: whatever the caller says the world holds, the
        generated view generates nothing, the move back is refused, and the motion verb refuses the line."""
        planner = _NeverPlans(self, here=_HERE)
        arm = _arm(planner)
        look = camera_looking_at(bearing_deg=0.0)
        far_face = JawFace(side=-1, centre_mm=(-20.0, -700.0, 20.0), normal=(-1.0, 0.0, 0.0), points=0, cells_seen=0,
                           cells_touchable=48)

        with arm.without_camera_world(_DECLINED):
            views = [go_to_generated_view(
                arm, aim=aim_at_faces((far_face,), grasp_centre_mm=CUBE_CENTRE), camera_to_base_mm=look,
                tool_to_base_mm=tool_for(look).to_matrix(), intrinsics=K, image_size=(WIDTH, HEIGHT),
                here=JointPositions(_HERE), home=arm.home_joint_positions, holding=holding) for holding in (False, True)]
            back = move_back_to_look(arm, JointPositions(_THERE), look="the look", here=JointPositions(_HERE),
                                     holding=True)
            moved = motion.move_joints_on_the_line(arm, JointPositions(_THERE))

        for view in views:
            self.assertFalse(view.moved)
            self.assertIsNone(view.stopped)
        self.assertIn("does not hold the frames of this pick", views[0].skipped)
        self.assertIn("declined", views[1].skipped)
        self.assertFalse(back.done)
        self.assertIn("declined", back.refused)
        self.assertIs(motion.MotionOutcome.REFUSED, moved.outcome)
        self.assertIs(MotionStatus.UNSUPPORTED, moved.status)
        arm._conn.moveJ.assert_not_called()
        self.assertEqual([], planner.lines(), "a line a pick made up was judged inside a decline")

    def test_only_the_ur_arm_says_it_drives_joint_lines(self) -> None:
        self.assertIsInstance(_arm(RoutePlanner(here=_HERE)), DrivesJointLines)
        for driver in (KukaRobotArm, IsaacRobotArm, DummyRobotArm):
            with self.subTest(driver=driver.__name__):
                self.assertFalse(hasattr(driver, "move_to_joints_on_the_line"))
        self.assertFalse(hasattr(RobotArm, "move_to_joints_on_the_line"),
                         "a member on RobotArm would reach every driver that subclasses it as a stub")


if __name__ == "__main__":
    unittest.main()
