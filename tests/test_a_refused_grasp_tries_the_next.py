"""A grasp refused before anything was sent is followed by the next grasp of the same look, while the hand never reached
the part (the owner's "try the next grasps", cell fixes Track C).

On the owner's cell (2026-10-01) the exact guard refused the line in of P1-P4 before anything was sent, and the pick
ended there with grasps two to four never tried. Now the pick loop tries ``result.best`` and then up to three more of
the same look's candidates, four in all, and only while both hold: the try before was refused before anything was sent
(a guard's or the planner's refusal, ``generated_view.REFUSED_BEFORE_SENDING``, or the planner's own no-plan words),
and every motion it did send kept the open hand's jaw region out of the held target's keep-out box. A standoff 60 mm
over the part keeps it out, so the arm goes back to the look on a judged move and the next grasp is tried from there; a
standoff 10 mm over it does not, and the attempt ends there as it always did. So does a carried lift the arm would
refuse, a motion that was sent and failed, a close, a gripper fault. With both jaw faces asked for only the judged
grasp is tried. A stop asked for between two grasps, a controller that stopped, and a stop asked for during the move
back each end the pick before anything else moves. A toggle hand is asked, never switched, before every try. Every try
is said at INFO, every refused one at WARNING, and kept on the attempt (``PickAttempt.tries``, contract 3).

The scene is ``tests/_wrist_views.py``: the cube on the bench, one look from its +x side. The calculator ranks four or
more grasps of the cube, all at its centre and straight down, each closing turned further about the vertical; the
policy answers each rank as the test says and moves the arm to its standoff where a test says it was sent there.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, Callable
from unittest import mock

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_rotation_matrix
from src.robot.core import (
    NO_PLAN_FAIL_SAFE_MESSAGE,
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.motion.trajectory_safety import ApproachPathPolicy
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.safety.planning.perceived import WorldBuildLimits, WorldBuildTuning
from tests._wrist_views import CUBE_CENTRE, LookingArm, WristCamera, camera_to_tool, joints_key
from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _changes, _toggle
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import (
    HAND_E,
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    _keys,
    _poses,
    _World,
)

#: The planner world's limits and tuning as the pick reads the held target's keep-out box by them: the camera world's
#: own margin of 15 mm round the part, carried down to the bench.
LIMITS = WorldBuildLimits(x_mm=(-800.0, 800.0), y_mm=(-1000.0, 800.0), z_mm=(-100.0, 600.0), support_plane_top_mm=0.0)
TUNING = WorldBuildTuning(pixel_stride=2, margin_mm=15.0)

#: What the policy answers a rank.
REFUSED = "refused before anything was sent"
LINE_IN_REFUSED = "the standoff ran, the line in was refused before it was sent"
NO_PLAN = "the planner found no plan for the standoff"
SENT_TIMEOUT = "the standoff was sent and ran out of time"
SENT_FAILED = "the standoff ran, the line in was sent and failed"
CARRIED = "the carried lift would be refused, the hand backed out empty"
EXECUTED = "executed"


class _HeldWorld(_World):
    """The live planner world double, with the limits and tuning the pick reads the part's keep-out box by."""

    def __init__(self, **keywords: Any) -> None:
        super().__init__(**keywords)
        self.limits = LIMITS
        self.tuning = TUNING


class _Ranked:
    """Ranks the cube with ``count`` grasps at its centre, straight down, each closing turned 20 degrees further about
    the vertical than the one before, rank 0 first; any other label gets none. ``latest`` holds the last ranking."""

    render_debug_images = False

    def __init__(self, count: int = 4) -> None:
        self.count = count
        self.latest: tuple[GraspPoint, ...] = ()

    def compute_result(self, seg: Any, *_args: Any, **_kwargs: Any) -> GraspResult:
        if str(getattr(seg, "label", "")) != "part":
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,
                                        GraspFailureReason.RESCAN_RECOMMENDED))
        grasps = []
        for rank in range(self.count):
            turn = math.radians(20.0 * rank)
            grasps.append(GraspPoint(position=np.array(CUBE_CENTRE), approach=np.array([0.0, 0.0, -1.0]),
                                     axis=np.array([math.cos(turn), math.sin(turn), 0.0]), grip_width_mm=40.0,
                                     score=0.9 - 0.05 * rank, frame=GraspFrame.BASE, label="part"))
        self.latest = tuple(grasps)
        return GraspResult(candidates=self.latest, top_score=0.9)

    def rank_of(self, grasp: GraspPoint) -> int:
        return next(rank for rank, candidate in enumerate(self.latest) if candidate is grasp)


def _tcp(grasp: GraspPoint, back_mm: float) -> Pose:
    """The TCP ``back_mm`` back along the grasp's approach, turned as the policy turns it."""
    closing = np.asarray(grasp.axis, dtype=np.float64)
    approach = np.asarray(grasp.approach, dtype=np.float64)
    binormal = np.cross(approach, closing)
    rotation = np.column_stack([closing, binormal / np.linalg.norm(binormal), approach])
    return Pose(position_mm=np.asarray(grasp.position, dtype=np.float64) - approach * back_mm,
                quaternion_xyzw=np.asarray(from_rotation_matrix(rotation), dtype=np.float64), frame=Frame.BASE)


class _Answers:
    """Executes nothing and answers each rank as told (``answers``), moving the arm to the standoff where the answer
    says it was sent there. ``executed`` is the ranks handed over, in order, ``joints_at`` where the arm stood then."""

    def __init__(self, arm: Any, calculator: _Ranked, answers: dict[int, str], *, standoff_mm: float,
                 on_execute: Callable[[int], None] | None = None) -> None:
        self.arm = arm
        self.calculator = calculator
        self.answers = dict(answers)
        self.standoff_mm = float(standoff_mm)
        self.on_execute = on_execute
        self.executed: list[int] = []
        self.joints_at: list[JointPositions] = []

    def execute(self, grasp: GraspPoint) -> PolicyReport:
        rank = self.calculator.rank_of(grasp)
        self.executed.append(rank)
        self.joints_at.append(self.arm.get_joint_positions())
        if self.on_execute is not None:
            self.on_execute(rank)
        answer = self.answers.get(rank, EXECUTED)
        standoff = _tcp(grasp, self.standoff_mm)
        if answer == REFUSED:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, motion_status=MotionStatus.SELF_COLLISION_REJECTED,
                                motion_message="the standoff comes within 3.1 mm of seen_00 (needs 5.0)")
        if answer == NO_PLAN:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, motion_status=MotionStatus.TIMEOUT,
                                motion_message=NO_PLAN_FAIL_SAFE_MESSAGE)
        if answer == SENT_TIMEOUT:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, motion_status=MotionStatus.TIMEOUT,
                                motion_message="the move ran out of its time budget")
        self.arm.move(standoff)
        if answer == LINE_IN_REFUSED:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, waypoints=(standoff,),
                                motion_status=MotionStatus.SELF_COLLISION_REJECTED,
                                motion_message="the line in comes within 3.5 mm of seen_00 (needs 5.0)")
        if answer == SENT_FAILED:
            return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED, waypoints=(standoff,),
                                motion_status=MotionStatus.CONTROLLER_REJECTED,
                                motion_message="Driver reported moveL() failure")
        if answer == CARRIED:
            return PolicyReport(outcome=PolicyOutcome.CARRIED_RETREAT_REFUSED,
                                waypoints=(standoff, _tcp(grasp, 0.0), standoff),
                                motion_status=MotionStatus.SELF_COLLISION_REJECTED,
                                motion_message="the lift carrying the part comes within 2.0 mm of seen_01")
        return PolicyReport(outcome=PolicyOutcome.EXECUTED, motion_status=MotionStatus.EXECUTED)


class _StoppingArm(LookingArm):
    """The looking arm with a controller that can be stopped: ``status`` is what it reports."""

    def __init__(self) -> None:
        super().__init__(_poses())
        self.status = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                                  emergency_stopped=False)

    def get_robot_status(self) -> RobotStatus:
        return self.status

    def recover_from_protective_stop(self) -> bool:
        raise AssertionError("nothing clears a protective stop but a person")


class _Cell:
    """The wrist cell: the looking arm with its live world, the camera, the ranking calculator and the answering policy.
    ``moves_back`` is called as the arm is driven back to a look after a try."""

    def __init__(self, answers: dict[int, str], *, count: int = 4, standoff_mm: float = 60.0,
                 looks: tuple[JointPositions, ...] = (LOOK_PLUS_X,), arm: LookingArm | None = None,
                 on_execute: Callable[[int], None] | None = None, **wiring: Any) -> None:
        self.arm = arm if arm is not None else LookingArm(_poses())
        self.arm.live_planner_world = _HeldWorld()  # type: ignore[attr-defined]
        self.camera = WristCamera(self.arm)
        self.calculator = _Ranked(count)
        self.policy = _Answers(self.arm, self.calculator, answers, standoff_mm=standoff_mm, on_execute=on_execute)
        self.orchestrator = BinPickingOrchestrator(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()), policy=self.policy,  # type: ignore[arg-type]
            max_attempts=1, primary_camera_id="wrist", gripper_model=HAND_E, looks=looks, target_label="part",
            **wiring,
        )

    def run(self) -> Any:
        return self.orchestrator.run()

    def joint_moves(self) -> list[tuple[float, ...]]:
        return [key for kind, key in self.arm.motions if kind == "joints"]


class TheNextGraspOfTheSameLookTests(unittest.TestCase):
    def test_a_grasp_refused_before_anything_was_sent_is_followed_by_the_next(self) -> None:
        """Red before: the attempt ended EXECUTION_FAILED with only rank 0 tried."""
        cell = _Cell({0: REFUSED})

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual("executed", attempt.action)
        self.assertEqual([0, 1], [row["rank"] for row in attempt.tries])
        self.assertEqual(["motion_failed", "executed"], [row["outcome"] for row in attempt.tries])
        self.assertEqual([False, True], [row["sent"] for row in attempt.tries])
        self.assertEqual([False, True], [row["reached_part"] for row in attempt.tries])
        self.assertEqual("self_collision_rejected", attempt.tries[0]["motion_status"])
        self.assertIn("seen_00", attempt.tries[0]["motion_message"])
        # Nothing was sent, so the arm never left the look: no move back before the next grasp.
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())

    def test_the_planner_s_own_no_plan_words_are_a_refusal_before_sending(self) -> None:
        cell = _Cell({0: NO_PLAN})

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1], cell.policy.executed)

    def test_after_a_standoff_outside_the_part_s_box_the_arm_goes_back_to_the_look_first(self) -> None:
        """A 60 mm standoff: the jaws stand over the box the world keeps out for the part (its top + 15 mm)."""
        cell = _Cell({0: LINE_IN_REFUSED}, standoff_mm=60.0)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual([0, 1], cell.policy.executed)
        # The look, the standoff of rank 0, then a judged move back to the look before rank 1 is handed over (whose own
        # standoff comes last).
        self.assertEqual(_keys(LOOK_PLUS_X, LOOK_PLUS_X), cell.joint_moves())
        kinds = [kind for kind, _ in cell.arm.motions]
        self.assertEqual(["joints", "move", "joints", "move"], kinds)
        self.assertEqual(joints_key(LOOK_PLUS_X), joints_key(cell.policy.joints_at[1]))
        (attempt,) = report.attempts
        self.assertEqual([True, True], [row["sent"] for row in attempt.tries])
        self.assertEqual([False, True], [row["reached_part"] for row in attempt.tries])

    def test_no_next_grasp_once_the_hand_stood_in_the_part_s_box(self) -> None:
        """A 10 mm standoff puts the pads 10 mm over the grasp, inside the box the world keeps out for the part."""
        cell = _Cell({0: LINE_IN_REFUSED}, standoff_mm=10.0)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([0], cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual([True], [row["reached_part"] for row in attempt.tries])
        self.assertEqual((), attempt.reasons, "a try that stood at the part is never typed as a refusal before sending")
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves(), "the arm was sent back to the look")

    def test_no_next_grasp_after_a_carried_lift_was_refused(self) -> None:
        cell = _Cell({0: CARRIED})

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([0], cell.policy.executed)

    def test_no_next_grasp_after_a_motion_that_was_sent_and_failed(self) -> None:
        """A pin: a failure once something was sent ends the attempt, as it always did."""
        cell = _Cell({0: SENT_FAILED})

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([0], cell.policy.executed)

    def test_no_next_grasp_after_a_sent_motion_ran_out_of_time(self) -> None:
        """A timeout in any words but the planner's no-plan sentence may have been sent: a pin."""
        cell = _Cell({0: SENT_TIMEOUT})

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        self.assertEqual([0], cell.policy.executed)

    def test_with_both_faces_asked_for_only_the_judged_grasp_is_tried(self) -> None:
        cell = _Cell({0: REFUSED}, looks=(LOOK_PLUS_X, LOOK_MINUS_X), both_faces=True)

        report = cell.run()

        self.assertEqual([0], cell.policy.executed)
        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)

    def test_never_more_than_twelve_grasps_an_attempt(self) -> None:
        """Twelve since 2026-10-06, every candidate the calculator offers; four before: a grasp refused at the hand on
        its line down is refused before its route is planned, so the more cost little
        (``pick_loop.GRASPS_TRIED_PER_ATTEMPT``)."""
        cell = _Cell(dict.fromkeys(range(14), REFUSED), count=14)

        report = cell.run()

        self.assertEqual(list(range(12)), cell.policy.executed)
        (attempt,) = report.attempts
        self.assertEqual(12, len(attempt.tries))


class AStopWinsBetweenTwoGraspsTests(unittest.TestCase):
    def test_a_stop_asked_for_after_a_refused_grasp_hands_over_no_other(self) -> None:
        executed: list[int] = []
        cell = _Cell({0: REFUSED}, on_execute=executed.append, should_cancel=lambda: bool(executed))

        report = cell.run()

        self.assertEqual([0], cell.policy.executed)
        self.assertIs(PickOutcome.CANCELLED, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves())

    def test_a_controller_that_stopped_after_a_refused_grasp_ends_the_pick(self) -> None:
        arm = _StoppingArm()

        def stop(_rank: int) -> None:
            arm.status = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP,
                                     protective_stopped=True, emergency_stopped=False)

        cell = _Cell({0: LINE_IN_REFUSED}, arm=arm, on_execute=stop)

        report = cell.run()

        self.assertEqual([0], cell.policy.executed)
        self.assertIs(PickOutcome.CONTROLLER_NOT_OPERATIONAL, report.outcome)
        self.assertEqual(_keys(LOOK_PLUS_X), cell.joint_moves(), "the arm was driven back on a stopped controller")

    def test_a_stop_asked_for_during_the_move_back_hands_over_no_grasp(self) -> None:
        asked: list[bool] = []

        class _Arm(LookingArm):
            def move_to_joints(self, joints: JointPositions, **keywords: Any) -> MotionResult:
                result = super().move_to_joints(joints, **keywords)
                if len(self.motions) > 2:  # the look, the standoff, then this: the move back
                    asked.append(True)
                return result

        cell = _Cell({0: LINE_IN_REFUSED}, arm=_Arm(_poses()), should_cancel=lambda: bool(asked))

        report = cell.run()

        self.assertEqual([0], cell.policy.executed)
        self.assertIs(PickOutcome.CANCELLED, report.outcome)


class _RefusingArm(LookingArm):
    """An arm whose every typed move to a standoff is refused by the guard before anything is sent."""

    def __init__(self) -> None:
        super().__init__(_poses())
        self.refused = 0

    def move(self, pose: Any, **keywords: Any) -> MotionResult:
        self.motions.append(("move", tuple(round(float(v), 1) for v in pose.position_mm)))
        self.refused += 1
        return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                   message="the standoff comes within 3.1 mm of seen_00 (needs 5.0)")


class AToggleHandAcrossTheTriesTests(unittest.TestCase):
    def test_six_refused_grasps_change_tool_do0_not_once_and_ask_nobody(self) -> None:
        """The owner's Hand-E on tool DO0: the real policy asks the toggle before every try, and never switches it."""
        events: list[Any] = []
        jaws = _toggle(events, Person("open"))
        jaws.connect()
        nobody = Person()
        jaws._ask = nobody  # noqa: SLF001 (any question from here on fails the test)
        connected = len(events)
        arm = _RefusingArm()
        arm.live_planner_world = _HeldWorld()  # type: ignore[attr-defined]
        camera = WristCamera(arm)
        calculator = _Ranked(6)
        orchestrator = BinPickingOrchestrator(
            arm=arm, calculator=calculator, perception=camera,  # type: ignore[arg-type]
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()),
            policy=GraspExecutionPolicy(arm=arm, gripper=jaws, standoff_mm=60.0), gripper=jaws,  # type: ignore[arg-type]
            max_attempts=1, primary_camera_id="wrist", gripper_model=HAND_E, looks=(LOOK_PLUS_X,), target_label="part",
        )

        report = orchestrator.run()

        self.assertEqual(6, arm.refused, "six grasps, one refused standoff each")
        self.assertEqual([], _changes(events[connected:]), "DO0 changed")
        self.assertEqual([], nobody.asked)
        self.assertFalse(jaws.jaws_closed)
        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), attempt.reasons)


class TheSweepAndTheLogTests(unittest.TestCase):
    def test_the_approach_sweep_is_judged_with_the_cell_s_own_hand(self) -> None:
        """Red before: the sweep was judged with the 2F-85's defaults (E2-9)."""
        cell = _Cell({}, approach_path_policy=ApproachPathPolicy(), mode_label="dense_clutter",
                     _approach_path_modes=frozenset({"dense_clutter"}))
        clear = mock.Mock(is_clear=True)
        with mock.patch("src.robot.grasping.loop.pick_loop.validate_approach_and_retreat",
                        return_value=(clear, clear)) as sweep:
            cell.run()

        self.assertTrue(sweep.called)
        self.assertIs(HAND_E, sweep.call_args.kwargs["gripper_model"])

    def test_every_refused_grasp_is_said_at_warning_with_its_status(self) -> None:
        cell = _Cell({0: REFUSED, 1: REFUSED})

        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            cell.run()

        warnings = [record.getMessage() for record in said.records if record.levelname == "WARNING"]
        self.assertTrue(any("try 1 of 4" in line and "self_collision_rejected" in line for line in warnings), warnings)
        self.assertTrue(any("try 2 of 4" in line for line in warnings), warnings)
        infos = [record.getMessage() for record in said.records if record.levelname == "INFO"]
        self.assertTrue(any("try 3 of 4" in line for line in infos), infos)


if __name__ == "__main__":
    unittest.main()
