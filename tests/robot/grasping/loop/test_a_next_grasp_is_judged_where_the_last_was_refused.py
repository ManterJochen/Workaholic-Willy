"""The next grasp of a look is judged first against the boxes that refused the grasp before it, with no new frame.

On the owner's crate (2026-10-12 13:45) the exact guard refused nine grasps one after the other, every one by the same
neighbour's box, each after a world of its own built for it, about a second each. The owner's answer: the guard's boxes
decide what the next grasp is spared. Once a grasp was refused judged ahead, nothing sent, the next one's line down is
judged in the world that refusal was made in (the policy's ``line_refusal_held``, the UR arm's ``line_down_refusal``):
a grasp refused there is not judged again and the next one follows; one that passes there gets the whole judgement as
before. The first grasp of a look, and a grasp after one that ran, are judged as ever.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from src.robot.grasping.loop.pick_loop import PickOutcome
from tests.robot.grasping.types.test_a_refused_grasp_tries_the_next import _Answers, _Cell

#: What the policy's whole judgement says of a refused grasp, and what the held world says of a refused line down.
_AHEAD = "the line down to the grasp would be refused (self_collision_rejected: rfinger|fixture:seen_00: 2.7 < 3.0)"
_HELD = "the line down to the grasp would be refused (self_collision_rejected: rfinger|fixture:seen_00: 2.6 < 3.0)"


class _HeldAnswers(_Answers):
    """Judges each grasp ahead as ``ahead_refused`` says and its line down in the held world as ``held_refused`` says;
    ``judged`` and ``held`` keep the ranks each was asked about, in order."""

    def __init__(self, *args: Any, ahead_refused: "tuple[int, ...]", held_refused: "tuple[int, ...]",
                 **keywords: Any) -> None:
        super().__init__(*args, **keywords)
        self.ahead_refused = ahead_refused
        self.held_refused = held_refused
        self.judged: list[int] = []
        self.held: list[int] = []

    def refusal_ahead(self, grasp: Any) -> str:
        rank = self.calculator.rank_of(grasp)
        self.judged.append(rank)
        return _AHEAD if rank in self.ahead_refused else ""

    def line_refusal_held(self, grasp: Any) -> str:
        rank = self.calculator.rank_of(grasp)
        self.held.append(rank)
        return _HELD if rank in self.held_refused else ""


def _cell(*, ahead_refused: "tuple[int, ...]", held_refused: "tuple[int, ...]", count: int = 4) -> _Cell:
    cell = _Cell({}, count=count)
    cell.policy = _HeldAnswers(cell.arm, cell.calculator, {}, standoff_mm=60.0, ahead_refused=ahead_refused,
                               held_refused=held_refused)
    cell.orchestrator.policy = cell.policy  # type: ignore[assignment]
    return cell


class TheNextGraspIsJudgedWhereTheLastWasRefusedTests(unittest.TestCase):
    def test_grasps_the_held_world_refuses_are_not_judged_again(self) -> None:
        cell = _cell(ahead_refused=(0,), held_refused=(1, 2))

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        # Rank 0 judged whole and refused; ranks 1 and 2 refused in its world and never judged whole; rank 3 passed
        # there, judged whole and executed.
        self.assertEqual([0, 3], cell.policy.judged)
        self.assertEqual([1, 2, 3], cell.policy.held)
        self.assertEqual([3], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual([0, 1, 2, 3], [row["rank"] for row in attempt.tries])
        self.assertEqual(["motion_failed", "motion_failed", "motion_failed", "executed"],
                         [row["outcome"] for row in attempt.tries])
        self.assertIn("seen_00", attempt.tries[1]["motion_message"])
        warnings = [record.getMessage() for record in said.records if record.levelno == logging.WARNING]
        self.assertTrue(any("try 2 of 4" in line and "judged in the world the grasp before it was refused in" in line
                            for line in warnings), warnings)

    def test_the_first_grasp_of_a_look_is_judged_whole(self) -> None:
        cell = _cell(ahead_refused=(), held_refused=(0, 1, 2, 3))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0], cell.policy.judged)
        self.assertEqual([], cell.policy.held)
        self.assertEqual([0], cell.policy.executed)

    def test_every_grasp_refused_ends_the_tries_with_none_sent(self) -> None:
        cell = _cell(ahead_refused=(0,), held_refused=(1, 2, 3))

        report = cell.run()

        self.assertIsNot(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0], cell.policy.judged)
        self.assertEqual([], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual([False] * 4, [row["sent"] for row in attempt.tries])


if __name__ == "__main__":
    unittest.main()


class TheArmJudgesTheLineDownInItsHeldWorldTests(unittest.TestCase):
    """``URRobotArm.line_down_refusal``: the line down from the configuration the route would end on, judged inside a
    held world (no new frame), said only where a part on the flange was refused."""

    def _arm(self, answer: Any) -> "tuple[Any, list[Any]]":
        from tests._route_planner import RoutePlanner
        from tests.robot.drivers.planned.test_the_nearest_goal_is_planned import _arm

        arm = _arm(RoutePlanner(here=[0.3, -1.4, 1.4, -1.6, -0.3, 0.2]))
        seen: list[Any] = []

        def judge(pose: Any, **keywords: Any) -> Any:
            seen.append((arm._held_world_reason, keywords.get("start")))
            return answer

        arm._judge_linear_move = judge  # type: ignore[method-assign]
        return arm, seen

    @staticmethod
    def _poses() -> "tuple[Any, Any]":
        from src.geometry import Pose

        return Pose.tool_down(400.0, -100.0, 180.0), Pose.tool_down(400.0, -100.0, 100.0)

    def test_a_finger_refused_by_a_camera_box_is_said_judged_in_the_held_world(self) -> None:
        from src.robot.core import MotionCommand, MotionResult, MotionStatus

        refused = MotionResult(command=MotionCommand.MOVE_TO, status=MotionStatus.SELF_COLLISION_REJECTED,
                               message="rfinger|fixture:seen_00: mesh distance 2.6 mm < 3.0 mm")
        arm, seen = self._arm(refused)
        standoff, grasp = self._poses()

        said = arm.line_down_refusal(standoff=standoff, grasp=grasp)

        self.assertIn("rfinger|fixture:seen_00", said)
        self.assertIn("the line down to the grasp would be refused", said)
        ((reason, start),) = seen
        self.assertTrue(reason, "judged inside a held world")
        self.assertIs(standoff, start[0])
        self.assertIsNone(arm._held_world_reason, "the held world ends with the judgement")

    def test_an_arm_link_refused_is_left_to_the_whole_judgement(self) -> None:
        from src.robot.core import MotionCommand, MotionResult, MotionStatus

        refused = MotionResult(command=MotionCommand.MOVE_TO, status=MotionStatus.SELF_COLLISION_REJECTED,
                               message="forearm_link|fixture:seen_03: mesh distance 1.0 mm < 3.0 mm")
        arm, _ = self._arm(refused)
        self.assertEqual("", arm.line_down_refusal(standoff=self._poses()[0], grasp=self._poses()[1]))

    def test_a_line_that_runs_is_no_refusal(self) -> None:
        arm, seen = self._arm(None)
        self.assertEqual("", arm.line_down_refusal(standoff=self._poses()[0], grasp=self._poses()[1]))
        self.assertEqual(1, len(seen))
