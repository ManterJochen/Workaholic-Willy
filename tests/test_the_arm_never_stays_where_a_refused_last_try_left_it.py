"""The arm never stays where the last try of a look left it, short of the part (S1 of the stuck map, 2026-10-08).

On the cell (2026-10-07, 10:18:45) the line down of try 5 of 5 was refused at its standoff, 0.14 mm short of the 3 mm
guard, and nothing brought the arm back up: the pick ended there, the target went back into the world and the pick's
frames were dropped, so every motion after it (the next pick's first look, Home, Restart) was judged on the standoff's
own frame alone, a wrist camera over a bin with no depth. The operator pressed Home 37.6 s later. The bench had the same
case on try 8 of 9, saved only because try 9 followed. The move back to the look ran only before a NEXT grasp.

Now a wrist pick drives the arm back to where it stood before the first grasp after its last try too, whenever that
try's last commanded pose is its standoff and it ended short of the part: refused before it was sent on, the planner's
own no-plan words, a carried lift the arm would refuse. It runs inside the pick's world (the target still kept out, every
frame still held), whatever the number of grasps, with ``both_faces`` too. Never after a stop asked for, a halt, a
stopped controller, a motion that was sent and failed, or a back-out refused at the part. Where the move back does not
run, the pick says where the arm stands (``PickReport.stands_at``, ``"standoff of grasp N"``, also on its
``attempt_finished`` event) and a task stops on it for a person (``test_a_task_stops_where_a_refused_move_back_left_the_arm``).

The scene and the doubles are those of ``tests/test_a_refused_grasp_tries_the_next.py``: the cube on the bench, one look
from its +x side, the policy answering each rank as a test says.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus, RobotMode, RobotStatus, SafetyMode
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.types.feedback import GraspFailureReason
from tests._wrist_views import LookingArm, joints_key
from tests.test_a_refused_grasp_tries_the_next import (
    CARRIED,
    LINE_IN_REFUSED,
    NO_PLAN,
    REFUSED,
    SENT_FAILED,
    _Answers,
    _Cell,
    _HeldWorld,
    _StoppingArm,
    _tcp,
)
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_MINUS_X, LOOK_PLUS_X, _keys, _poses

#: Further answers of the policy: the line in refused by the halt with nothing sent, and a back-out up the line it came
#: down refused at the part.
HALTED_AT_THE_STANDOFF = "the standoff ran, the line in was cancelled by the halt"
BACK_OUT_REFUSED = "at the part the carried lift would be refused, and the line back up was refused too"
NO_PLAN_AT_THE_STANDOFF = "the standoff ran, the planner found no plan for the line in"


class _Answering(_Answers):
    """The policy of the refused-grasp tests, with the answers this file adds."""

    def execute(self, grasp: Any) -> PolicyReport:
        rank = self.calculator.rank_of(grasp)
        answer = self.answers.get(rank)
        if answer not in (HALTED_AT_THE_STANDOFF, BACK_OUT_REFUSED, NO_PLAN_AT_THE_STANDOFF):
            return super().execute(grasp)
        self.executed.append(rank)
        self.joints_at.append(self.arm.get_joint_positions())
        if self.on_execute is not None:
            self.on_execute(rank)
        standoff = _tcp(grasp, self.standoff_mm)
        self.arm.move(standoff)
        if answer == HALTED_AT_THE_STANDOFF:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, waypoints=(standoff,),
                                motion_status=MotionStatus.CANCELLED, motion_message="halted: nothing was sent")
        if answer == NO_PLAN_AT_THE_STANDOFF:
            from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE  # noqa: PLC0415

            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, waypoints=(standoff,),
                                motion_status=MotionStatus.TIMEOUT, motion_message=NO_PLAN_FAIL_SAFE_MESSAGE)
        return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, waypoints=(standoff, _tcp(grasp, 0.0)),
                            motion_status=MotionStatus.SELF_COLLISION_REJECTED,
                            motion_message="the line back up to the standoff was refused too")


class _Watching(_HeldWorld):
    """The live world, which also says whether a target is held out of it now (``kept_out``)."""

    def __init__(self) -> None:
        super().__init__()
        self.kept_out = False

    def offer_segmentation(self, **offered: Any) -> None:
        super().offer_segmentation(**offered)
        self.kept_out = bool(offered.get("hold"))

    def forget_segmentation(self) -> None:
        self.kept_out = False


class _WatchingArm(LookingArm):
    """The looking arm, which writes down at every joint move whether the world held the pick's frames and kept its
    target out (``seen``). ``back`` answers the joint moves after the first: ``None`` drives them, a status refuses
    them before anything is sent (or, for ``CONTROLLER_REJECTED``, fails them once sent). A wrist frame over the part
    is blind (``blind_over_the_part``): a joint move from there raises, unless the pick holds its frames."""

    def __init__(self, *, back: "MotionStatus | None" = None, blind_over_the_part: bool = False) -> None:
        super().__init__(_poses())
        self.back = back
        self.blind_over_the_part = blind_over_the_part
        self.over_the_part = False
        self.seen: list[tuple[tuple[float, ...], bool, bool]] = []

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        result = super().move(pose, **keywords)
        self.over_the_part = True
        return result

    def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
        world = self.live_planner_world  # type: ignore[attr-defined]
        self.seen.append((joints_key(joints), bool(world.holding), bool(world.kept_out)))
        if self.blind_over_the_part and self.over_the_part and not world.holding:
            self.motions.append(("joints", joints_key(joints)))
            raise CameraWorldUnavailable(camera="wrist", verdict="blind", attempts=4,
                                         reason="no depth on 100% of the wrist camera's frame over the part")
        if self.back is not None and len(self.seen) > 1:
            self.motions.append(("joints", joints_key(joints)))
            return MotionResult.failed(self.back, MotionCommand.MOVE_JOINTS, target_joints=joints,
                                       message="the move back to the look comes within 2.1 mm of seen_03")
        result = super().move_to_joints(joints, **keywords)
        self.over_the_part = False
        return result


def _cell(answers: dict[int, str], *, arm: "LookingArm | None" = None, count: int = 1, **wiring: Any) -> _Cell:
    cell = _Cell(answers, count=count, arm=arm if arm is not None else _WatchingArm(), **wiring)
    cell.policy = _Answering(cell.arm, cell.calculator, answers, standoff_mm=60.0,
                             on_execute=wiring.get("on_execute"))  # type: ignore[assignment]
    cell.orchestrator.policy = cell.policy  # type: ignore[assignment]
    world = _Watching()
    cell.arm.live_planner_world = world  # type: ignore[attr-defined]
    return cell


def _stood(_rank: int) -> None:
    """``on_execute`` for a test that only needs the hook wired."""


class TheLastTryGoesBackToTheLookTests(unittest.TestCase):
    def test_a_last_grasp_refused_at_its_standoff_goes_back_to_the_look_inside_the_pick(self) -> None:
        """Red before: two refused lines in, one move back between them, and the arm left at the second standoff."""
        cell = _cell({0: LINE_IN_REFUSED, 1: LINE_IN_REFUSED}, count=2)

        report = cell.run()

        self.assertEqual([0, 1], cell.policy.executed)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())
        # The move back after the last try ran while the pick held its frames and kept its target out of the world.
        self.assertEqual((joints_key(LOOK_PLUS_X), True, True), cell.arm.seen[-1])
        self.assertEqual(["joints", "move", "joints", "move", "joints"], [kind for kind, _ in cell.arm.motions])
        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), attempt.reasons)
        self.assertEqual("", report.stands_at, "the arm stands at the look again")

    def test_so_does_a_look_with_one_grasp(self) -> None:
        """Red before: where the pick was where it stood before the first grasp was read only for more than one."""
        cell = _cell({0: LINE_IN_REFUSED})

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())
        self.assertEqual("", report.stands_at)

    def test_and_the_planners_own_no_plan_at_the_standoff(self) -> None:
        cell = _cell({0: NO_PLAN_AT_THE_STANDOFF})

        cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())

    def test_and_a_pick_that_asked_for_both_jaw_faces(self) -> None:
        cell = _cell({0: LINE_IN_REFUSED}, looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True)

        cell.run()

        self.assertEqual([0], cell.policy.executed)
        moves = cell.joint_moves()
        self.assertEqual(joints_key(cell.policy.joints_at[0]), moves[-1], "not back where the first grasp started")
        self.assertEqual(["move", "joints"], [kind for kind, _ in cell.arm.motions][-2:])

    def test_a_carried_lift_the_arm_would_refuse_ends_at_the_look(self) -> None:
        """The hand backed out up the line it came down, empty, to the standoff: from there, back to the look."""
        cell = _cell({0: CARRIED})

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())
        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual("", report.stands_at)

    def test_nothing_is_driven_back_where_nothing_was_sent(self) -> None:
        cell = _cell({0: REFUSED, 1: NO_PLAN}, count=2)

        cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())


class NothingMovesBackAfterTests(unittest.TestCase):
    def test_a_stop_asked_for(self) -> None:
        executed: list[int] = []
        cell = _cell({0: LINE_IN_REFUSED}, on_execute=executed.append, should_cancel=lambda: bool(executed))

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())
        self.assertIs(PickOutcome.CANCELLED, report.outcome)
        self.assertEqual("standoff of grasp 1", report.stands_at)

    def test_a_halt_that_refused_the_line_in(self) -> None:
        cell = _cell({0: HALTED_AT_THE_STANDOFF})

        cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())

    def test_a_stopped_controller(self) -> None:
        arm = _StoppingArm()

        def stop(_rank: int) -> None:
            arm.status = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                                     protective_stopped=True, emergency_stopped=False)

        cell = _cell({0: LINE_IN_REFUSED}, arm=arm, on_execute=stop)

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())
        self.assertIs(PickOutcome.CONTROLLER_NOT_OPERATIONAL, report.outcome)
        self.assertEqual("standoff of grasp 1", report.stands_at)

    def test_a_motion_that_was_sent_and_failed(self) -> None:
        cell = _cell({0: SENT_FAILED})

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())
        self.assertEqual("", report.stands_at, "nothing was due to drive it back")

    def test_a_back_out_refused_at_the_part(self) -> None:
        cell = _cell({0: BACK_OUT_REFUSED})

        report = cell.run()

        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())
        self.assertEqual("", report.stands_at)


class WhereTheMoveBackDidNotRunTests(unittest.TestCase):
    def test_a_refused_move_back_says_the_standoff_it_left_the_arm_at(self) -> None:
        events: list[Any] = []
        cell = _cell({0: LINE_IN_REFUSED, 1: LINE_IN_REFUSED}, count=2,
                     arm=_WatchingArm(back=MotionStatus.WORKSPACE_REJECTED), on_progress=events.append)

        report = cell.run()

        self.assertEqual([0], cell.policy.executed, "a next grasp was handed over from the standoff")
        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual("standoff of grasp 1", report.stands_at)
        (finished,) = [event for event in events if str(event.stage) == "attempt_finished"]
        self.assertEqual("standoff of grasp 1", finished.extra["stands_at"])

    def test_so_does_one_after_the_last_try(self) -> None:
        cell = _cell({0: LINE_IN_REFUSED}, arm=_WatchingArm(back=MotionStatus.SELF_COLLISION_REJECTED))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual("standoff of grasp 1", report.stands_at)

    def test_a_move_back_that_failed_once_sent_says_the_way_back(self) -> None:
        cell = _cell({0: LINE_IN_REFUSED}, arm=_WatchingArm(back=MotionStatus.CONTROLLER_REJECTED))

        report = cell.run()

        self.assertEqual("way back from the standoff of grasp 1", report.stands_at)


class TheArmIsNeverBlindAfterAFailedPickTests(unittest.TestCase):
    def test_a_no_depth_frame_at_the_standoff_does_not_stop_the_next_pick(self) -> None:
        """Over the part the wrist camera sees no depth, and a frame no pick holds is blind there. The move back runs
        while the pick holds its frames; the next pick's first look then runs from the look, not from over the part."""
        arm = _WatchingArm(blind_over_the_part=True)
        cell = _cell({0: LINE_IN_REFUSED}, arm=arm)

        cell.run()
        self.assertFalse(arm.over_the_part, "the pick left the arm over the part")
        cell.policy.answers = {0: "executed"}
        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())


if __name__ == "__main__":
    unittest.main()
