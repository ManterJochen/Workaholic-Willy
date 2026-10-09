"""A blocker's grasp is judged from the look before the arm leaves for it (the owner, 2026-10-03).

On URSim (2026-10-02/03) every blocker grasp the clearing took was refused only once the arm stood over the blocker: the
line down by the guard, or the lift by the planner, judged as if the jaws held the blocker. The arm drove down and back
for nothing. Now each grasp a clearing may take (``recovery.blocker_grasp_tries``, 3 by default) is judged first, from
where the arm stands and with nothing sent:

* ``GraspExecutionPolicy.refusal_ahead`` hands the arm the very poses ``execute`` drives: the standoff, the grasp and
  the lift's top, and the width the lift is judged carrying;
* ``URRobotArm.grasp_refusal_ahead`` judges each once where it can (the judge chain, 2026-10-08): the line down from the
  configuration the route to the standoff ends on as the closed form names it, the lift from the grasp's configuration
  as if carrying, in the world the line down was judged in, and the route ``move`` would run to the standoff last. A
  route that ends there is kept for the move to the standoff; one that ends anywhere else has the line and the lift
  judged again from its end, each in its own world, as before. The first refusal ends the judgement and is said;
* a line judged from elsewhere is never the line commanded.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus, RobotKinematicsError
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


#: The configuration the stand-in's closed form names for the standoff, and a route's end that is not it.
_NEAR = [0.05] * 6
_ELSEWHERE = [0.1] * 6


class TheArmJudgesInTheOrderThePickRunsTests(unittest.TestCase):
    """``URRobotArm.grasp_refusal_ahead`` on a stand-in of the arm's own judgements: its world is one refresh, which
    nothing replaces, and a held world opened on it is recorded as ``("held", reason)``."""

    def _arm(self, *, route: Any = None, line: Any = None, lift: Any = None, planner: str = "curobo") -> Any:
        from src.robot.drivers.ur.arm import _Route

        asked: list[tuple[str, Any]] = []
        kept: list[Any] = []
        world = object()

        def route_to(planner_: Any, goal: Pose, pose: Pose, *, velocity: float) -> Any:
            asked.append(("route", (pose, velocity)))
            return route if route is not None else _Route(waypoints=[[0.0] * 6, list(_ELSEWHERE)], dense=2,
                                                           how="direct line", planned=False)

        def judge_line(pose: Pose, *, command: MotionCommand, commanded: bool = True, start: Any = None) -> Any:
            asked.append(("line", (pose, commanded, start)))
            return line

        def nearest(pose: Pose) -> Any:
            asked.append(("nearest", pose))
            return SimpleNamespace(values=np.array(_NEAR))

        def ik(pose: Pose, *, seed: Any) -> Any:
            asked.append(("ik", (pose, list(seed.tolist()))))
            return SimpleNamespace(tolist=lambda: [0.2] * 6)

        def carried(pose: Pose, *, grip_width_mm: float, start: Any = None) -> Any:
            asked.append(("lift", (pose, grip_width_mm, start)))
            return lift

        @contextmanager
        def held_world(reason: str) -> Iterator[None]:
            asked.append(("held", reason))
            yield

        def keep(planner_: Any, route_: Any, standoff: Pose, *, velocity: float) -> None:
            kept.append((route_, standoff, velocity))

        self.asked = asked
        self.kept = kept
        return SimpleNamespace(
            _motion_planner=planner, _preflight=object(), _conn=SimpleNamespace(is_connected=True),
            config=SimpleNamespace(motion_limits=SimpleNamespace(max_velocity=0.5)),
            _curobo_ur_planner=lambda: "planner", _pose_to_flange=lambda pose: pose,
            _route_to_the_nearest_goal=route_to, _judge_linear_move=judge_line, ik=ik, carried_line_refusal=carried,
            nearest_configuration=nearest, _planner_refresh=lambda: world, held_world=held_world,
            _keep_the_route_ahead=keep,
            # safety.planning_world.hold.* as shipped: no junction holds its world (2026-10-09).
            _holds_at=lambda junction: False,
        )

    @staticmethod
    def _route(end: "list[float]") -> Any:
        from src.robot.drivers.ur.arm import _Route

        return _Route(waypoints=[[0.0] * 6, list(end)], dense=2, how="direct line", planned=False)

    def _ask(self, arm: Any) -> str:
        from src.robot.drivers.ur.arm import URRobotArm

        return URRobotArm.grasp_refusal_ahead(arm, standoff=_pose(105.0), grasp=_pose(95.0), lift=_pose(195.0),
                                              grip_width_mm=30.0)

    def test_each_is_judged_once_and_the_route_last_where_it_ends_where_the_line_was_judged_from(self) -> None:
        """The judge chain, 2026-10-08: on the cell the line down was judged twice and the lift's world refreshed for the
        same question the line's had answered, 2.0 to 2.7 s of a try."""
        route = self._route(_NEAR)
        self.assertEqual("", self._ask(self._arm(route=route)))
        kinds = [kind for kind, _ in self.asked]
        self.assertEqual(["nearest", "line", "ik", "held", "lift", "route"], kinds)
        _line_pose, commanded, start = self.asked[1][1]
        self.assertFalse(commanded)
        self.assertAlmostEqual(float(start[0].position_mm[2]), 105.0)
        self.assertEqual(_NEAR, start[1])                          # the standoff's nearest configuration
        self.assertEqual(_NEAR, self.asked[2][1][1])               # the grasp solved seeded on it
        self.assertIn("the world the line down to its grasp was judged in", self.asked[3][1])
        lift_pose, width, lift_start = self.asked[4][1]
        self.assertAlmostEqual(float(lift_pose.position_mm[2]), 195.0)
        self.assertEqual(30.0, width)
        self.assertAlmostEqual(float(lift_start[0].position_mm[2]), 95.0)
        self.assertEqual([0.2] * 6, lift_start[1])                 # the grasp's own configuration
        ((kept_route, standoff, velocity),) = self.kept
        self.assertIs(kept_route, route, "the very route judged is kept for the move to the standoff")
        self.assertAlmostEqual(float(standoff.position_mm[2]), 105.0)
        self.assertEqual(0.5, velocity)

    def test_a_route_that_ends_elsewhere_has_the_line_and_the_lift_judged_again_from_its_end(self) -> None:
        """A cuRobo plan, another goal or a turn flip: the line and the lift from where the route ends, each in its own
        world, and nothing is kept."""
        self.assertEqual("", self._ask(self._arm(route=self._route(_ELSEWHERE))))
        kinds = [kind for kind, _ in self.asked]
        self.assertEqual(["nearest", "line", "ik", "held", "lift", "route", "line", "ik", "lift"], kinds)
        _pose_line, commanded, start = self.asked[6][1]
        self.assertFalse(commanded)
        self.assertAlmostEqual(float(start[0].position_mm[2]), 105.0)
        self.assertEqual(_ELSEWHERE, start[1])                     # the route's end
        self.assertEqual(_ELSEWHERE, self.asked[7][1][1])          # the grasp solved seeded on it
        self.assertEqual([0.2] * 6, self.asked[8][1][2][1])
        self.assertEqual(1, kinds.count("held"), "the lift judged again refreshes its own world")
        self.assertEqual([], self.kept)

    def test_a_lift_refused_ends_the_judgement_before_the_route(self) -> None:
        lift = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="world")
        said = self._ask(self._arm(lift=lift))
        self.assertIn("the lift, as if the jaws held a part 30 mm across, would be refused", said)
        self.assertIn("world", said)
        self.assertEqual(["nearest", "line", "ik", "held", "lift"], [kind for kind, _ in self.asked])
        self.assertEqual([], self.kept)

    def test_the_first_refusal_ends_the_judgement_and_is_said(self) -> None:
        refused = MotionResult.failed(MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, message="no plan")
        said = self._ask(self._arm(route=refused))
        self.assertIn("the move to the standoff would be refused", said)
        self.assertIn("no plan", said)
        self.assertEqual(["nearest", "line", "ik", "held", "lift", "route"], [kind for kind, _ in self.asked])
        self.assertEqual([], self.kept)

        line = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="finger 2 mm")
        said = self._ask(self._arm(line=line))
        self.assertIn("the line down to the grasp would be refused", said)
        self.assertEqual(["nearest", "line", "route", "line"], [kind for kind, _ in self.asked])

        lift = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="world")
        said = self._ask(self._arm(lift=lift))
        self.assertIn("the lift, as if the jaws held a part 30 mm across, would be refused", said)

    def test_a_line_a_part_on_the_flange_refuses_ends_it_before_the_route(self) -> None:
        """The hand, its camera and wrist 3 stand where the TCP puts them, from every configuration: refused at one of
        them against a fixture, the line is every configuration's refusal and the route is never planned (17 s a grasp
        in a bin's corner, the grasp bench, 2026-10-06)."""
        for part in ("rfinger", "gripper", "wrist_3", "wrist_camera_EIH_Cam"):
            with self.subTest(part=part):
                line = MotionResult.failed(
                    MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                    message=f"[safety:self_collision/self_collision] sample 32 of 174 of a path sampled at 2.963 mm a "
                            f"step: {part}|fixture:seen_02: mesh distance 2.701 mm < 3.000 mm")
                said = self._ask(self._arm(line=line))
                self.assertIn("the line down to the grasp would be refused", said)
                self.assertIn(f"{part}|fixture:seen_02", said)
                self.assertEqual(["nearest", "line"], [kind for kind, _ in self.asked])

    def test_a_line_an_arm_link_refuses_waits_for_the_route(self) -> None:
        """Where an arm link refuses the line, the configuration decides: the route is planned and the line judged from
        its end, as before."""
        line = MotionResult.failed(
            MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
            message="[safety:self_collision/self_collision] sample 3 of 9 of a path sampled at 2.963 mm a step: "
                    "wrist_1|fixture:seen_02: mesh distance -0.015 mm < 3.000 mm")
        said = self._ask(self._arm(line=line))
        self.assertIn("the line down to the grasp would be refused", said)
        self.assertEqual(["nearest", "line", "route", "line"], [kind for kind, _ in self.asked])

    def test_a_standoff_with_no_nearest_configuration_is_the_routes_to_judge(self) -> None:
        arm = self._arm()

        def no_configuration(pose: Pose) -> Any:
            raise RobotKinematicsError("no configuration")

        arm.nearest_configuration = no_configuration
        self.assertEqual("", self._ask(arm))
        self.assertEqual(["route", "line", "ik", "lift"], [kind for kind, _ in self.asked])

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
