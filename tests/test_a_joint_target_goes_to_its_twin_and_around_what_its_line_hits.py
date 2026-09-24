"""A joint target on a cuRobo UR runs as its full turn inside the cable window, by its line or by a plan around it.

Three things a joint move did not do before, each held here with a planner double and the real destination guards:

* A joint outside the window the planner and the joint-limit guard admit goes onto its full turn nearest the arm
  inside it: a station stored at -200 degrees on a wrist is the pose +160 is, and with the half turn about home the
  guard would refuse the number and not the pose. A joint already inside the window is left as the caller wrote it.
  A value past a full turn either way is never turned: no UR joint reaches it in radians, so it is a target written in
  degrees, and turning it would make a plausible configuration out of a typing mistake. The destination guard refuses
  it, JOINT_LIMIT_REJECTED, before a line is sampled or a plan asked for.
* The straight joint line is judged by both authorities, the planner at ``safety.planned_motion.line_clearance_mm``,
  and runs as one ``moveJ`` where it passes.
* Where the line collides, cuRobo plans to the same target instead of the move being refused; the plan is held to
  ``safety.planned_motion.max_detour_deg``, its legs are judged, and exactly that list runs. A line refused for anything
  but a collision, a planner that cannot be reached for one, is not planned around.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.errors import RobotMotionRejected
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.safety.planning import CuroboUnavailableError, StateRefusal, StateRefusalKind, StateWhere
from tests._plan_end import OPEN_WORKSPACE
from tests._route_planner import RoutePlanner, line

_DECLINED = "unit double: this file reads how a joint target is judged and run, and no camera world is wired"

#: Clear, and where the fake controller says the arm is standing (tests/test_ur_checked_joint_verbs.py).
_HERE = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0]
#: Clear, a short joint move away.
_THERE = [0.2, -1.5, 1.5, 0.0, 0.0, 0.0]
#: The shipped UR10 home copied off the pendant in degrees and handed over as radians.
_DEGREES = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


def _arm(planner: RoutePlanner, *, safety: "dict[str, Any] | None" = None,
         home: "list[float] | None" = None) -> URRobotArm:
    arm = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
        "safety": {"payload": {"enforce": False}, **(safety or {})}, "gripper": {"model": "robotiq_2f85"},
        "workspace_limits": OPEN_WORKSPACE,
        **({"home_joint_positions": home} if home is not None else {}),
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


class TheTwinTests(unittest.TestCase):
    def test_a_joint_outside_the_cable_window_runs_as_its_twin_inside_and_the_log_says_so(self) -> None:
        home = list(_HERE)
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner, safety={"joint_limits": {"within_half_turn_of_home": True}}, home=home)
        station = [*_THERE[:5], math.radians(-200.0)]
        with arm.without_camera_world(_DECLINED), self.assertLogs("URRobotArm", level="INFO") as logs:
            result = arm.move_to_joints(JointPositions(station))
        self.assertTrue(result.ok, result.message)
        (sent,) = _sent(arm)
        self.assertAlmostEqual(math.radians(160.0), sent[5], places=12)
        np.testing.assert_allclose(sent[:5], station[:5], atol=0.0)
        np.testing.assert_allclose(result.target_joints.values, sent, atol=0.0)  # the result names what ran
        (lined,) = planner.lines()
        self.assertAlmostEqual(math.radians(160.0), lined[1][-1][5], places=12, msg="the line judged is the one run")
        said = "\n".join(logs.output)
        self.assertIn("wrist_3_joint -200.0 -> 160.0 deg", said)
        self.assertIn("twin wrap", said)

    def test_a_joint_inside_the_window_is_left_as_written(self) -> None:
        """Without the half turn the window is two turns wide; a turn inside it is the caller's to choose."""
        planner = RoutePlanner(here=[*_HERE[:5], 5.0])
        arm = _arm(planner)
        self.assertEqual((JointPositions(_THERE), ""), arm._twin_near_the_arm(JointPositions(_THERE), planner.here))
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertTrue(result.ok, result.message)
        (sent,) = _sent(arm)
        np.testing.assert_allclose(sent, _THERE, atol=0.0)

    def test_a_target_written_in_degrees_is_refused_and_never_reaches_movej(self) -> None:
        """The safety gap this closes: wrapped, [0, -90, 90, -90, -90, 0] 'rad' is a plausible configuration."""
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        self.assertEqual("", arm._twin_near_the_arm(JointPositions(_DEGREES), _HERE)[1],
                         "a value past a full turn was turned onto a twin")
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_DEGREES))
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, result.command)
        self.assertIn("reads as degrees", result.message or "")
        np.testing.assert_allclose(result.target_joints.values, _DEGREES, atol=0.0)
        arm._conn.moveJ.assert_not_called()
        self.assertEqual([], planner.named("check"), "a line to a target in degrees was sampled")
        self.assertEqual([], planner.named("plan_joint"))

    def test_the_raising_verb_refuses_the_same_target_the_same_way(self) -> None:
        arm = _arm(RoutePlanner(here=_HERE))
        with arm.without_camera_world(_DECLINED), self.assertRaises(RobotMotionRejected) as caught:
            arm.move_joint(JointPositions(_DEGREES))
        result = caught.exception.result
        assert isinstance(result, MotionResult)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        arm._conn.moveJ.assert_not_called()


class TheLineAndThePlanTests(unittest.TestCase):
    def test_a_clear_line_is_one_movej_and_cuRobo_plans_nothing(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED), self.assertLogs("URRobotArm", level="INFO") as logs:
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertTrue(result.ok, result.message)
        self.assertEqual([_THERE], _sent(arm))
        self.assertEqual([], planner.named("plan_joint"))
        self.assertEqual([], planner.named("execute"))
        (lined,) = planner.lines()
        self.assertEqual(10.0, lined[3])
        self.assertIn("move_to_joints: direct line; 2 waypoint(s) dense, 2 executed (1 leg(s))", "\n".join(logs.output))

    def test_a_line_that_collides_is_planned_around_and_the_judged_list_runs(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        corner = [_HERE[0] + 0.1, _HERE[1] - 0.3, *_HERE[2:]]
        planner.answer = lambda goal: line(_HERE, corner, 11) + line(corner, goal, 11)[1:]
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_JOINTS, result.command)
        self.assertEqual("move_to_joints", result.message)
        (asked,) = planner.named("plan_joint")
        np.testing.assert_allclose(asked[1], _THERE, atol=0.0)
        self.assertIs(False, asked[2], "the plan refreshed the world the line was judged in")
        (execute,) = planner.named("execute")
        self.assertEqual(3, len(execute[1]), "the plan's corner was dropped, or the plan was not shortened")
        self.assertIs(MotionCommand.MOVE_JOINTS, execute[2])
        np.testing.assert_allclose(execute[3].values, _THERE, atol=0.0)
        arm._conn.moveJ.assert_not_called()  # every waypoint went through the planner glue's execute

    def test_home_is_planned_around_too_and_reads_move_home(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner, home=list(_THERE))
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_home()
        self.assertTrue(result.ok, result.message)
        self.assertIs(MotionCommand.MOVE_HOME, result.command)
        (execute,) = planner.named("execute")
        self.assertIs(MotionCommand.MOVE_HOME, execute[2])

    def test_the_raising_verb_runs_a_planned_route_as_well(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED):
            arm.move_joint(JointPositions(_THERE))
        self.assertEqual(1, len(planner.named("execute")))

    def test_a_plan_past_the_detour_bound_is_refused_naming_the_joint(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        swing = [*_HERE[:5], math.radians(80.0)]
        planner.answer = lambda goal: [list(_HERE), swing, list(goal)]
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status, result.message)
        self.assertIn("swings wrist_3_joint 80.0 deg beyond the span", result.message or "")
        self.assertIn("max_detour_deg of 45", result.message or "")
        self.assertEqual([], planner.named("execute"))
        arm._conn.moveJ.assert_not_called()

    def test_no_plan_around_a_colliding_line_is_the_lines_refusal(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        planner.refusal_on_a_line = StateRefusal(where=StateWhere.GOAL, kind=StateRefusalKind.WORLD,
                                                 joints=tuple(_THERE), depth_mm=3.0)
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.SELF_COLLISION_REJECTED, result.status)
        self.assertIn("the planner refused this straight joint line", result.message or "")
        self.assertIn("no plan goes around it", result.message or "")
        arm._conn.moveJ.assert_not_called()

    def test_a_start_the_planner_will_not_leave_is_typed(self) -> None:
        planner = RoutePlanner(here=_HERE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: None
        planner.refusal_on_no_plan = StateRefusal(where=StateWhere.START, kind=StateRefusalKind.JOINT_LIMIT,
                                                  joints=tuple(_HERE))
        arm = _arm(planner)
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("will not start from", result.message or "")

    def test_a_line_refused_for_anything_but_a_collision_is_not_planned_around(self) -> None:
        planner = RoutePlanner(here=_HERE)
        arm = _arm(planner)

        def unavailable(*_: Any, **__: Any) -> Any:
            raise CuroboUnavailableError("no GPU env")

        planner.check_joint_path = unavailable  # type: ignore[method-assign]
        with arm.without_camera_world(_DECLINED):
            result = arm.move_to_joints(JointPositions(_THERE))
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, result.status)
        self.assertEqual([], planner.named("plan_joint"))
        arm._conn.moveJ.assert_not_called()


class TheConfigTests(unittest.TestCase):
    def test_the_keys_default_to_the_planners_clearance_and_45_degrees(self) -> None:
        planned = RobotConfig.model_validate({"vendor": "ur"}).safety.planned_motion
        self.assertEqual((10.0, 45.0), (planned.line_clearance_mm, planned.max_detour_deg))

    def test_a_bound_of_nothing_and_a_clearance_the_planner_cannot_ask_are_refused(self) -> None:
        from pydantic import ValidationError

        for block in ({"max_detour_deg": 0.0}, {"max_detour_deg": -5.0}, {"line_clearance_mm": -1.0},
                      {"line_clearance_mm": 150.0}, {"surplus": 1}):
            with self.subTest(block=block), self.assertRaises(ValidationError):
                RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": block}})
        zero = RobotConfig.model_validate({"vendor": "ur", "safety": {"planned_motion": {"line_clearance_mm": 0.0}}})
        self.assertEqual(0.0, zero.safety.planned_motion.line_clearance_mm, "0 judges a line as a plan is judged")


if __name__ == "__main__":
    unittest.main()
