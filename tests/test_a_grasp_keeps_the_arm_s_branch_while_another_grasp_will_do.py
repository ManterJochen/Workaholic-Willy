"""A grasp keeps the arm's branch while another grasp of the same look will do: a change of branch is the last resort
(the owner's R2, 2026-10-08).

On the cell (2026-10-07, 11:19) try 10 of a look judged a shoulder and elbow flip, a direct joint line of 4010 samples
from shoulder -/elbow - to shoulder +/elbow +, because the grasp's natural wrist turn had no configuration on the arm's
branch inside the cable window; the operator's halt stopped it. A change of branch was a WARNING, taken by the first
grasp that needed one, whatever the grasps after it would have done.

Now, on the UR driver, :meth:`URRobotArm.keeping_its_branch` opens a block in which every Cartesian move keeps the branch
the arm holds: the other branches are not screened, lined or planned to, a cuRobo plan swinging a joint more than 15
degrees past its span is a detour (``max_detour_deg`` 45 outside), and a move nothing on the branch runs while another
branch, or a plan within 45, was left untried is refused before anything is sent, ``JOINT_LIMIT_REJECTED``, and counted.
``nearest_configuration`` answers inside it among the same goals. A route a grasp judged ahead on the arm's branch is
judged again on that branch alone: a change of branch nobody judged ahead is refused before sending.

The pick loop tries every grasp of a look inside such a block first; a grasp the block refused for that alone waits, and
the ones that waited are tried by the rules outside the block, a change of branch a WARNING as ever, only where every
grasp of the first pass was refused before anything was sent. A grasp that ran, reached the part or failed once sent
ends the tries as it always did.
"""

from __future__ import annotations

import logging
import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus, RobotKinematicsError
from src.robot.grasping.loop.pick_loop import KEPT_BRANCH_OVERSHOOT_DEG, PickOutcome
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.safety._ur_ik import ur_branch
from tests._plan_end import pose_where_it_ends
from tests._route_planner import RoutePlanner, line
from tests._wrist_views import LookingArm
from tests.test_a_refused_grasp_tries_the_next import REFUSED, SENT_FAILED, _Answers, _Cell
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import _poses
from tests.test_the_nearest_goal_is_planned import _CANDLE, _THERE, _arm, _executed_goal, _move, _near, _ranked

#: Wrist 2 at -0.3 rad, and a goal whose quickest configuration puts it at +0.3, the other side of its branch point; on
#: the arm's own branch the same pose is reached with wrist 1 and wrist 3 turned about half a turn instead.
_FROM = [0.3, -1.4, 1.4, -1.6, -0.3, 0.2]
_TO = [0.3, -1.4, 1.4, -1.6, 0.3, 0.2]


def _on_the_branch(arm: Any) -> "tuple[float, ...]":
    held = ur_branch("ur5e", _FROM)
    assert held is not None
    (same,) = [g.joints for g in _ranked(arm, _TO, _FROM) if held.agrees_with(ur_branch("ur5e", g.joints))]
    return same


def _asked(planner: RoutePlanner) -> "list[tuple[float, ...]]":
    """Every configuration the arm put to the planner as a goal: screened alone, lined to, planned to, or run to."""
    goals = [tuple(call[1]) for call in planner.named("plan_joint")]
    goals += [call[1][0] for call in planner.named("check") if len(call[1]) == 1]
    goals += [call[1][-1] for call in planner.lines()]
    goals += [call[1][-1] for call in planner.named("execute")]
    return goals


def _swinging(goal: "list[float]", *, degrees: float) -> "list[list[float]]":
    """A plan to ``goal`` out of the candle home whose wrist 3 swings ``degrees`` past the goal's side of its span."""
    out = line(_CANDLE, goal, 9)
    out[4] = [*out[4][:5], max(_CANDLE[5], goal[5]) + math.radians(degrees)]
    return out


class InsideTheBlockAMoveKeepsTheBranchTests(unittest.TestCase):
    def test_a_move_only_another_branch_runs_is_refused_before_anything_is_sent_and_counted(self) -> None:
        planner = RoutePlanner(here=_FROM)
        arm = _arm(planner)
        same = _on_the_branch(arm)
        planner.line_clear = lambda start, end: not _near(end, same, 1e-9)
        planner.answer = lambda goal: None

        with arm.keeping_its_branch() as kept:
            result = _move(arm, _TO)

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("kept to the branch the arm holds (shoulder +, wrist 2 -, elbow +)", result.message or "")
        self.assertIn("configuration(s) on another branch were left untried", result.message or "")
        self.assertIn("nothing was sent", result.message or "")
        self.assertEqual([], planner.named("execute"))
        self.assertEqual(1, kept.refused)
        held = ur_branch("ur5e", _FROM)
        assert held is not None
        for goal in _asked(planner):
            self.assertTrue(held.agrees_with(ur_branch("ur5e", goal)), f"{goal} off the arm's branch was asked")

    def test_outside_it_the_same_move_changes_branch_with_a_warning(self) -> None:
        """The control, as ``test_the_nearest_goal_is_planned`` pins it: today's rule outside the block."""
        planner = RoutePlanner(here=_FROM)
        arm = _arm(planner)
        same = _on_the_branch(arm)
        planner.line_clear = lambda start, end: not _near(end, same, 1e-9)
        planner.answer = lambda goal: None

        with self.assertLogs("URRobotArm", level="WARNING") as logs:
            result = _move(arm, _TO)

        self.assertTrue(result.ok, result.message)
        np.testing.assert_allclose(_executed_goal(planner), _TO, atol=1e-6)
        self.assertIn("changes the arm's branch", "\n".join(logs.output))

    def test_a_move_its_branch_runs_runs_inside_it(self) -> None:
        planner = RoutePlanner(here=_FROM)
        arm = _arm(planner)

        with arm.keeping_its_branch() as kept:
            result = _move(arm, _TO)

        self.assertTrue(result.ok, result.message)
        np.testing.assert_allclose(_executed_goal(planner), _on_the_branch(arm), atol=1e-9)
        self.assertEqual(0, kept.refused)

    def test_a_plan_swinging_a_joint_past_15_degrees_is_a_detour_inside_it_and_runs_outside(self) -> None:
        """20 degrees past the span: inside the configured 45, past the block's 15."""
        def planner(degrees: float) -> RoutePlanner:
            made = RoutePlanner(here=_CANDLE)
            made.line_clear = lambda start, end: False
            made.answer = lambda goal: _swinging(goal, degrees=degrees)
            return made

        outside = planner(20.0)
        self.assertTrue(_move(_arm(outside), _THERE).ok)
        inside = planner(20.0)
        arm = _arm(inside)
        with arm.keeping_its_branch() as kept:
            result = _move(arm, _THERE)
        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("more than the 15 deg a move kept to the branch the arm holds allows", result.message or "")
        self.assertEqual(1, kept.refused)
        self.assertEqual([], inside.named("execute"))
        within = planner(10.0)
        arm = _arm(within)
        with arm.keeping_its_branch():
            self.assertTrue(_move(arm, _THERE).ok)

    def test_a_plan_past_the_configured_bound_is_no_refusal_for_the_branch(self) -> None:
        """Past 45 degrees outside the block too, and every goal on the arm's branch: the move's own refusal stands."""
        planner = RoutePlanner(here=_CANDLE)
        planner.line_clear = lambda start, end: False
        planner.answer = lambda goal: _swinging(goal, degrees=60.0)
        arm = _arm(planner)

        with arm.keeping_its_branch() as kept:
            result = _move(arm, _THERE)

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("max_detour_deg", result.message or "")
        self.assertEqual(0, kept.refused)

    def test_the_configuration_a_pose_goes_to_is_answered_on_the_arms_branch_alone(self) -> None:
        planner = RoutePlanner(here=_FROM)
        arm = _arm(planner)
        same = _on_the_branch(arm)
        refusal = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, message="the spy")
        arm._screen_planned_config = (  # type: ignore[method-assign]
            lambda pose, joints: refusal if _near(joints.tolist(), same, 1e-9) else None)
        pose = pose_where_it_ends(arm, _TO)

        outside = arm.nearest_configuration(pose)
        with arm.keeping_its_branch(), self.assertRaises(RobotKinematicsError) as raised:
            arm.nearest_configuration(pose)

        self.assertFalse(_near(outside.tolist(), same, 1e-9), "the control: outside, another branch is answered")
        self.assertIn("refused by the endpoint gate", str(raised.exception))

    def test_a_bound_that_is_no_number_of_degrees_is_refused(self) -> None:
        arm = _arm(RoutePlanner(here=_FROM))
        for bound in (-1.0, float("nan"), float("inf")):
            with self.subTest(bound=bound), self.assertRaises(ValueError), arm.keeping_its_branch(bound):
                pass  # pragma: no cover


class ARouteJudgedAheadKeepsItsBranchTests(unittest.TestCase):
    """``grasp_refusal_ahead`` judges the standoff's route on the arm's branch; the move to the standoff, judged again
    after the world changed, would reach it only by a change of branch."""

    @staticmethod
    def _judging(planner: RoutePlanner) -> Any:
        arm = _arm(planner)
        # The line down and the lift pass; this file reads the route alone.
        arm._judge_linear_move = lambda pose, **keywords: None  # type: ignore[method-assign]
        arm.carried_line_refusal = lambda pose, **keywords: None  # type: ignore[method-assign]
        arm._screen_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
        arm.ik = lambda pose, *, seed=None: seed  # type: ignore[method-assign]
        return arm

    def _the_change(self, arm: Any, planner: RoutePlanner) -> None:
        same = _on_the_branch(arm)
        planner.line_clear = lambda start, end: not _near(end, same, 1e-9)
        planner.answer = lambda goal: None

    def test_is_refused_before_anything_is_sent(self) -> None:
        planner = RoutePlanner(here=_FROM)
        arm = self._judging(planner)
        standoff = pose_where_it_ends(arm, _TO)
        grasp = Pose.tool_down(400.0, -100.0, 100.0)

        said = arm.grasp_refusal_ahead(standoff=standoff, grasp=grasp, lift=grasp, grip_width_mm=40.0)
        self.assertEqual("", said)
        self._the_change(arm, planner)
        with arm.without_camera_world("unit double: no camera world is wired"):
            result = arm.move(standoff)

        self.assertIs(MotionStatus.JOINT_LIMIT_REJECTED, result.status)
        self.assertIn("the route judged ahead to this pose was kept to the branch the arm holds", result.message or "")
        self.assertIn("a change of branch nobody judged ahead is not taken", result.message or "")
        self.assertEqual([], planner.named("execute"))

    def test_a_move_judged_ahead_to_another_pose_changes_branch_as_ever(self) -> None:
        """The control: the same move with no judgement ahead of it, or one of another grasp, warns and runs."""
        planner = RoutePlanner(here=_FROM)
        arm = self._judging(planner)
        standoff = pose_where_it_ends(arm, _TO)
        other = pose_where_it_ends(arm, [0.3, -1.4, 1.4, -1.6, -0.35, 0.2])
        grasp = Pose.tool_down(400.0, -100.0, 100.0)

        self.assertEqual("", arm.grasp_refusal_ahead(standoff=other, grasp=grasp, lift=grasp, grip_width_mm=40.0))
        self._the_change(arm, planner)
        with arm.without_camera_world("unit double: no camera world is wired"), \
                self.assertLogs("URRobotArm", level="WARNING") as logs:
            result = arm.move(standoff)

        self.assertTrue(result.ok, result.message)
        self.assertIn("changes the arm's branch", "\n".join(logs.output))


class _KeepingArm(LookingArm):
    """The looking arm, which keeps its branch inside a block as the UR driver does: ``keeping`` is the record of the
    block open now, ``None`` outside one, and ``bounds`` every overshoot bound a block was opened with."""

    def __init__(self) -> None:
        super().__init__(_poses())
        self.keeping: Any = None
        self.bounds: list[float] = []

    @contextmanager
    def keeping_its_branch(self, overshoot_deg: float = 15.0) -> Iterator[Any]:
        self.bounds.append(float(overshoot_deg))
        self.keeping = SimpleNamespace(overshoot_deg=float(overshoot_deg), refused=0)
        try:
            yield self.keeping
        finally:
            self.keeping = None


#: What the arm says of a route to the standoff it refused for its branch, inside the block.
_KEPT = ("this move is kept to the branch the arm holds (shoulder -, wrist 2 +, elbow -), and nothing on it runs: 4 "
         "configuration(s) on another branch were left untried, a change of branch being the last resort; nothing was "
         "sent")


class _BranchAnswers(_Answers):
    """Answers each rank as told; a rank in ``changers`` only another branch reaches, so inside the arm's block its
    standoff is refused before anything is sent and counted, as the UR driver refuses it. ``kept`` says, per call,
    whether the arm kept its branch then. ``ahead`` judges each grasp ahead, the way the real policy asks the arm."""

    def __init__(self, *args: Any, changers: "tuple[int, ...]" = (), ahead: bool = False, **keywords: Any) -> None:
        super().__init__(*args, **keywords)
        self.changers = changers
        self.ahead = ahead
        self.kept: list[bool] = []
        self.judged: list[int] = []

    def _refused_for_its_branch(self, rank: int) -> bool:
        keeping = self.arm.keeping
        if rank in self.changers and keeping is not None:
            keeping.refused += 1
            return True
        return False

    def refusal_ahead(self, grasp: Any) -> str:
        if not self.ahead:
            return ""
        rank = self.calculator.rank_of(grasp)
        self.judged.append(rank)
        return (f"the move to the standoff would be refused (joint_limit_rejected: {_KEPT})"
                if self._refused_for_its_branch(rank) else "")

    def execute(self, grasp: Any) -> PolicyReport:
        rank = self.calculator.rank_of(grasp)
        self.kept.append(self.arm.keeping is not None)
        if not self.ahead and self._refused_for_its_branch(rank):
            self.executed.append(rank)
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, motion_status=MotionStatus.JOINT_LIMIT_REJECTED,
                                motion_message=_KEPT)
        return super().execute(grasp)


def _cell(answers: "dict[int, str]", *, changers: "tuple[int, ...]", ahead: bool = False, count: int = 3) -> _Cell:
    cell = _Cell(answers, count=count, arm=_KeepingArm())
    cell.policy = _BranchAnswers(cell.arm, cell.calculator, answers, standoff_mm=60.0, changers=changers,
                                 ahead=ahead)  # type: ignore[assignment]
    cell.orchestrator.policy = cell.policy  # type: ignore[assignment]
    return cell


class APickLeavesAChangeOfBranchForLastTests(unittest.TestCase):
    def test_a_grasp_that_would_change_branch_waits_for_the_grasps_that_keep_it(self) -> None:
        """Red before: rank 0 changed branch, a WARNING, though rank 1 kept it and ran."""
        cell = _cell({}, changers=(0,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1], cell.policy.executed)
        self.assertEqual([True, True], cell.policy.kept)
        (attempt,) = report.attempts
        self.assertEqual(["joint_limit_rejected", "executed"], [row["motion_status"] for row in attempt.tries])

    def test_the_change_of_branch_is_taken_only_as_the_last_resort_and_warns(self) -> None:
        cell = _cell({1: REFUSED, 2: REFUSED}, changers=(0,))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1, 2, 0], cell.policy.executed)
        self.assertEqual([True, True, True, False], cell.policy.kept, "the last resort kept the branch")
        warnings = [record.getMessage() for record in said.records if record.levelno == logging.WARNING]
        self.assertTrue(any("the last resort" in line for line in warnings), warnings)
        self.assertTrue(any("try 1 of 3" in line and "waits for the grasps that keep the arm's branch" in line
                            for line in warnings), warnings)
        (attempt,) = report.attempts
        self.assertEqual(4, len(attempt.tries))

    def test_so_where_the_grasps_are_judged_ahead(self) -> None:
        cell = _cell({1: REFUSED}, changers=(0,), ahead=True, count=2)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1, 0], cell.policy.judged)
        self.assertEqual([1, 0], cell.policy.executed)
        self.assertEqual([True, False], cell.policy.kept)

    def test_no_change_of_branch_once_a_grasp_of_the_first_pass_was_sent_and_failed(self) -> None:
        cell = _cell({1: SENT_FAILED}, changers=(0,))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([0, 1], cell.policy.executed)

    def test_a_grasp_refused_in_the_block_for_anything_else_does_not_wait(self) -> None:
        cell = _cell({0: REFUSED, 1: REFUSED, 2: REFUSED}, changers=())

        report = cell.run()

        self.assertEqual([0, 1, 2], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual(3, len(attempt.tries))

    def test_every_grasp_of_the_first_pass_keeps_the_branch_at_the_15_degree_overshoot(self) -> None:
        cell = _cell({0: REFUSED, 1: REFUSED}, changers=(), count=3)

        cell.run()

        self.assertEqual(15.0, KEPT_BRANCH_OVERSHOOT_DEG)
        self.assertEqual([15.0, 15.0, 15.0], cell.arm.bounds)


class TheArmKeepsTheBlockOnlyForTheTriesTests(unittest.TestCase):
    def test_the_move_back_to_the_look_is_no_move_of_the_block(self) -> None:
        """The move back between two grasps is a joint move to where the arm stood, outside every block."""
        seen: list[bool] = []

        class _Arm(_KeepingArm):
            def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
                seen.append(self.keeping is not None)
                return super().move_to_joints(joints, **keywords)

        from tests.test_a_refused_grasp_tries_the_next import LINE_IN_REFUSED  # noqa: PLC0415

        cell = _Cell({0: LINE_IN_REFUSED}, count=2, arm=_Arm())
        cell.run()

        self.assertEqual([False, False], seen, "the look, then the move back")


if __name__ == "__main__":
    unittest.main()
