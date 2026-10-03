"""A blocker's grasp is judged from the look before the arm leaves for it (the owner, 2026-10-03).

On URSim (2026-10-02/03) every blocker grasp the clearing took was refused only once the arm stood over the blocker: the
line down by the guard, or the lift by the planner, judged as if the jaws held the blocker. The arm drove down and back
for nothing. Now each grasp a clearing may take (``recovery.blocker_grasp_tries``, 3 by default) is judged first, from
where the arm stands and with nothing sent:

* ``GraspExecutionPolicy.refusal_ahead`` hands the arm the very poses ``execute`` drives: the standoff, the grasp and
  the lift's top, and the width the lift is judged carrying;
* ``URRobotArm.grasp_refusal_ahead`` judges them in the order the pick runs them, each from where the one before ends:
  the route ``move`` would run to the standoff, the line down from that route's end, and the lift from the grasp's
  configuration as if carrying; the first refusal ends the judgement and is said;
* a line judged from elsewhere is never the line commanded.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.core.arm_capabilities import LineMotion, LineReading
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


def _grasp(frame: GraspFrame = GraspFrame.BASE) -> GraspPoint:
    return GraspPoint(position=np.array([-150.0, -760.0, 95.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([0.0, 1.0, 0.0]), grip_width_mm=30.0, score=0.9, frame=frame, label="blocker")


class _JudgingArm:
    """An arm that keeps its lines and judges grasps ahead, recording what it was asked."""

    def __init__(self, answer: Any = "") -> None:
        self.answer = answer
        self.asked: list[dict[str, Any]] = []

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "every sample judged")

    def grasp_refusal_ahead(self, **asked: Any) -> str:
        self.asked.append(asked)
        if isinstance(self.answer, Exception):
            raise self.answer
        return str(self.answer)


class ThePolicyAsksAboutThePosesItDrivesTests(unittest.TestCase):

    def test_the_standoff_the_grasp_and_the_lift_are_the_ones_execute_drives(self) -> None:
        arm = _JudgingArm()
        policy = GraspExecutionPolicy(arm=arm, gripper=None, standoff_mm=10.0, retreat_mm=100.0)  # type: ignore[arg-type]
        grasp = _grasp()

        self.assertEqual("", policy.refusal_ahead(grasp))

        waypoints = policy._build_waypoints(grasp)  # noqa: SLF001
        asked = arm.asked[0]
        self.assertEqual(tuple(asked["standoff"].position_mm), tuple(waypoints[0].position_mm))
        self.assertEqual(tuple(asked["grasp"].position_mm), tuple(waypoints[-1 - policy.retreat_steps].position_mm))
        self.assertEqual(tuple(asked["lift"].position_mm), tuple(waypoints[-1].position_mm))
        self.assertAlmostEqual(float(asked["standoff"].position_mm[2]), 105.0, places=3)
        self.assertAlmostEqual(float(asked["lift"].position_mm[2]), 195.0, places=3)
        self.assertEqual(asked["grip_width_mm"], policy._judged_width(grasp))  # noqa: SLF001

    def test_the_arms_refusal_is_said(self) -> None:
        policy = GraspExecutionPolicy(arm=_JudgingArm("the lift would be refused (planner)"),  # type: ignore[arg-type]
                                      gripper=None, standoff_mm=10.0)
        self.assertEqual("the lift would be refused (planner)", policy.refusal_ahead(_grasp()))

    def test_a_judgement_that_raises_is_a_refusal_said(self) -> None:
        policy = GraspExecutionPolicy(arm=_JudgingArm(RuntimeError("no planner")), gripper=None)  # type: ignore[arg-type]
        said = policy.refusal_ahead(_grasp())
        self.assertIn("could not be judged ahead", said)
        self.assertIn("no planner", said)

    def test_nothing_is_judged_ahead_where_nobody_can(self) -> None:
        class _Plain:
            def line_motion(self) -> LineReading:
                return LineReading(LineMotion.CHECKED, "")

        self.assertEqual("", GraspExecutionPolicy(arm=_Plain(), gripper=None).refusal_ahead(_grasp()))  # type: ignore[arg-type]
        arm = _JudgingArm("refused")
        self.assertEqual("", GraspExecutionPolicy(arm=arm, gripper=None).refusal_ahead(  # type: ignore[arg-type]
            _grasp(GraspFrame.CAMERA)))
        self.assertEqual([], arm.asked)


def _pose(z: float) -> Pose:
    return Pose.tool_down(-150.0, -760.0, z)


class TheArmJudgesInTheOrderThePickRunsTests(unittest.TestCase):
    """``URRobotArm.grasp_refusal_ahead`` on a stand-in of the arm's own judgements."""

    def _arm(self, *, route: Any = None, line: Any = None, lift: Any = None, planner: str = "curobo") -> Any:
        from src.robot.drivers.ur.arm import _Route

        asked: list[tuple[str, Any]] = []

        def route_to(planner_: Any, goal: Pose, pose: Pose, *, velocity: float) -> Any:
            asked.append(("route", (pose, velocity)))
            return route if route is not None else _Route(waypoints=[[0.0] * 6, [0.1] * 6], dense=2, how="direct line",
                                                           planned=False)

        def judge_line(pose: Pose, *, command: MotionCommand, commanded: bool = True, start: Any = None) -> Any:
            asked.append(("line", (pose, commanded, start)))
            return line

        def ik(pose: Pose, *, seed: Any) -> Any:
            asked.append(("ik", (pose, list(seed.tolist()))))
            return SimpleNamespace(tolist=lambda: [0.2] * 6)

        def carried(pose: Pose, *, grip_width_mm: float, start: Any = None) -> Any:
            asked.append(("lift", (pose, grip_width_mm, start)))
            return lift

        self.asked = asked
        return SimpleNamespace(
            _motion_planner=planner, _preflight=object(), _conn=SimpleNamespace(is_connected=True),
            config=SimpleNamespace(motion_limits=SimpleNamespace(max_velocity=0.5)),
            _curobo_ur_planner=lambda: "planner", _pose_to_flange=lambda pose: pose,
            _route_to_the_nearest_goal=route_to, _judge_linear_move=judge_line, ik=ik, carried_line_refusal=carried,
        )

    def _ask(self, arm: Any) -> str:
        from src.robot.drivers.ur.arm import URRobotArm

        return URRobotArm.grasp_refusal_ahead(arm, standoff=_pose(105.0), grasp=_pose(95.0), lift=_pose(195.0),
                                              grip_width_mm=30.0)

    def test_each_is_judged_from_where_the_one_before_ends(self) -> None:
        self.assertEqual("", self._ask(self._arm()))
        kinds = [kind for kind, _ in self.asked]
        self.assertEqual(["route", "line", "ik", "lift"], kinds)
        _pose_line, commanded, start = self.asked[1][1]
        self.assertFalse(commanded)
        self.assertAlmostEqual(float(start[0].position_mm[2]), 105.0)
        self.assertEqual([0.1] * 6, start[1])                      # the route's end
        self.assertEqual([0.1] * 6, self.asked[2][1][1])           # the grasp solved seeded on it
        lift_pose, width, lift_start = self.asked[3][1]
        self.assertAlmostEqual(float(lift_pose.position_mm[2]), 195.0)
        self.assertEqual(30.0, width)
        self.assertAlmostEqual(float(lift_start[0].position_mm[2]), 95.0)
        self.assertEqual([0.2] * 6, lift_start[1])                 # the grasp's own configuration

    def test_the_first_refusal_ends_the_judgement_and_is_said(self) -> None:
        refused = MotionResult.failed(MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, message="no plan")
        said = self._ask(self._arm(route=refused))
        self.assertIn("the move to the standoff would be refused", said)
        self.assertIn("no plan", said)
        self.assertEqual(["route"], [kind for kind, _ in self.asked])

        line = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="finger 2 mm")
        said = self._ask(self._arm(line=line))
        self.assertIn("the line down to the grasp would be refused", said)
        self.assertEqual(["route", "line"], [kind for kind, _ in self.asked])

        lift = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="world")
        said = self._ask(self._arm(lift=lift))
        self.assertIn("the lift, as if the jaws held a part 30 mm across, would be refused", said)

    def test_a_cell_that_cannot_judge_ahead_says_nothing_and_asks_nothing(self) -> None:
        self.assertEqual("", self._ask(self._arm(planner="ik")))
        self.assertEqual([], self.asked)


class ALineJudgedFromElsewhereIsNeverCommandedTests(unittest.TestCase):

    def test_a_start_with_commanded_is_refused(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        arm = SimpleNamespace(_preflight=object(), _motion_planner="curobo")
        with self.assertRaises(ValueError):
            URRobotArm._judge_linear_move(arm, _pose(95.0), command=MotionCommand.MOVE_TO,  # type: ignore[arg-type]
                                          commanded=True, start=(_pose(105.0), [0.0] * 6))


class TheTriesAreTheCellsTests(unittest.TestCase):

    def test_three_by_default_one_to_six_allowed(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot.grasping_schema import GraspingRecoveryConfig

        self.assertEqual(3, GraspingRecoveryConfig().blocker_grasp_tries)
        self.assertEqual(2, GraspingRecoveryConfig(blocker_grasp_tries=2).blocker_grasp_tries)
        for wrong in (0, 7):
            with self.subTest(wrong=wrong), self.assertRaises(ValidationError):
                GraspingRecoveryConfig(blocker_grasp_tries=wrong)

    def test_the_effective_config_carries_the_cells_count(self) -> None:
        from src.config.schema.robot.grasping_schema import GraspingRecoveryConfig, RobotGraspingConfig
        from src.robot.execution.autonomous_grasp.builders import build_effective_config
        from src.robot.execution.autonomous_grasp.config import GraspMode

        cfg = RobotGraspingConfig(recovery=GraspingRecoveryConfig(enabled=True, blocker_grasp_tries=2))
        effective = build_effective_config(cfg, resolved_mode=GraspMode.DENSE_CLUTTER, resolved_max_attempts=3)
        assert effective is not None
        self.assertEqual(2, effective.recovery_orchestrator.blocker_grasp_tries)


if __name__ == "__main__":
    unittest.main()
