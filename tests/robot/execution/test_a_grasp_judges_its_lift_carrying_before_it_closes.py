"""A grasp judges its lift before the jaws close, and closes only where the lift would run.

The owner, 2026-10-01 ("Vorher prüfen"), on verifier W's M1. Option 1 lets an empty hand down a straight line into a pose
beside a bin the camera saw, where only the camera's boxes refuse the planner's world. Once the jaws close on a part
nothing is set aside, so the lift out of that pose, which starts there, is refused at its first sample: the arm stood
beside the bin with the part in its jaws until a person released it (GPU, verifier W, 2026-10-01).

Now the grasp asks the arm first, at the part with its jaws still open, whether the lift would run once they closed
(``JudgesCarriedLines.carried_line_refusal``): the lift's own judgement, as the line up will meet it. Where it would be
refused, the jaws stay open, the arm goes back up the line it came down, to the standoff, empty-handed, and the attempt
ends with its own outcome (``PolicyOutcome.CARRIED_RETREAT_REFUSED``, ``HandlingOutcome.CARRIED_RETREAT_REFUSED``), the
reason beside it, so the pick goes on to its next part or reports. Where it would run, the grasp closes and lifts as
before. An arm that cannot judge (the dummy, the sim) grasps as before. Both grasps judge: the pick loop's policy and
``Robot.pick``.

No part in the jaws is modelled (the owner, 2026-10-05: "mehr Griffe"; every cell since 2026-10-17): a closed hand is
judged as an empty one, so the lift beside the bin runs, the camera's boxes set aside and the exact guard deciding, and
what the judgement still refuses is what the line up would meet whatever the jaws hold: the bench, the exact guard.

The hand here is the real ``JawIOGripper`` on a recording tool I/O, as on the owner's cell, so "jaws stayed open"
is read off the outputs. Names new with this change are imported inside the tests.

Verifier W's second round (2026-10-01) holds the rest: ``Robot.pick`` judges its line out with the decline its motions
carry, the controller is the last thing asked before the jaws, and an ik arm screens the lift's end and remembers
nothing.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotCapabilities,
    RobotMode,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.arm_capabilities import LineMotion, LineReading
from src.robot.core.camera_world import CameraWorldDecline
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests._arm_kit import without_guards
from tests.robot.safety.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import (
    _DECLINED,
    AWAY,
    OFF,
    _cell,
    _from,
    _needs_the_engine,
)
from tests.robot.execution.test_a_stopped_controller_moves_no_jaws import _PROTECTIVE, _pulses, _toggle

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False, program_running=True)
_CAPS = RobotCapabilities(
    vendor="ur", model="ur10", dof=6,
    has_native_fk=True,
    is_simulated=False, controller=None, owns_a_controller=True,
)
#: What the arm says of a lift it would refuse: the exact guard, at the lift's first sample.
_HELD = MotionResult.failed(
    MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
    message="the exact guard refused this line at sample 0 of 12: the hand comes 1.2 mm from the bin's rim")


def _mm(pose: Pose) -> "tuple[float, ...]":
    return tuple(round(float(v), 1) for v in pose.position_mm)


class _LineArm:
    """A UR-like arm that keeps its lines (each sample judged before one moveL) and reports a controller that can move.
    Every call lands on the log the hand shares: ``("move", position, linear)``, ``"status"``, ``"steady"``.
    ``refuse_at`` refuses the motion of that index, nothing sent; ``raise_at`` raises out of it."""

    def __init__(self, events: "list[Any]", *, refuse_at: "int | None" = None, raise_at: "int | None" = None) -> None:
        self.events = events
        self.refuse_at = refuse_at
        self.raise_at = raise_at
        self.moves = 0
        self._tcp = Pose(position_mm=np.array([0.0, -600.0, 400.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                         frame=Frame.BASE, label="home")

    @property
    def capabilities(self) -> RobotCapabilities:
        return _CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def get_robot_status(self) -> RobotStatus:
        self.events.append("status")
        return _RUNNING

    def acknowledge_fault(self, fault: object) -> bool:  # pragma: no cover - never called, by design
        raise AssertionError("nothing on the pick path may clear a stop")

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        self.events.append("steady")
        return True

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "every sample judged before one moveL")

    def move(self, pose: Pose, **kwargs: Any) -> MotionResult:
        index = self.moves
        self.moves += 1
        self.events.append(("move", _mm(pose), bool(kwargs.get("linear", False))))
        if self.raise_at is not None and index == self.raise_at:
            raise RuntimeError("the RTDE link dropped")
        if self.refuse_at is not None and index == self.refuse_at:
            return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="the planner refused this line at sample 3 of 9: the bench")
        self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _JudgingArm(_LineArm):
    """The same arm, judging a lift before its jaws close (``JudgesCarriedLines``): ``carried`` is what it answers, a
    refusal, ``None`` (the lift would run) or an exception it raises."""

    def __init__(self, events: "list[Any]", *, carried: "MotionResult | BaseException | None" = None,
                 **kwargs: Any) -> None:
        super().__init__(events, **kwargs)
        self.carried = carried

    def carried_line_refusal(self, pose: Pose) -> "MotionResult | None":
        self.events.append(("judge", _mm(pose)))
        if isinstance(self.carried, BaseException):
            raise self.carried
        return self.carried


class _StopsWhileJudging(_JudgingArm):
    """The judging arm whose controller falls into a protective stop while it judges the lift, which would run: the
    judgement is IK round trips, a world refresh and the planner's check, and a stop can fall inside it (verifier W,
    2026-10-01, who measured about half a second on the development GPU with a carried part's attach and detach in
    it)."""

    def __init__(self, events: "list[Any]") -> None:
        super().__init__(events, carried=None)
        self.stopped = False

    def get_robot_status(self) -> RobotStatus:
        running = super().get_robot_status()
        return _PROTECTIVE if self.stopped else running

    def carried_line_refusal(self, pose: Pose) -> "MotionResult | None":
        judged = super().carried_line_refusal(pose)
        self.stopped = True
        return judged


def _grasp(width_mm: float = 40.0) -> GraspPoint:
    """A part at the owner's work area, closing along base -Y: the standoff 80 mm over it, the lift 100 mm."""
    return GraspPoint(position=np.array([-130.0, -600.0, 80.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([0.0, -1.0, 0.0]), grip_width_mm=width_mm, score=0.9, frame=GraspFrame.BASE,
                      label="part")


_STANDOFF = (-130.0, -600.0, 160.0)
_PART = (-130.0, -600.0, 80.0)
_LIFT = (-130.0, -600.0, 180.0)


def _policy(arm: Any, jaws: Any, **kwargs: Any) -> GraspExecutionPolicy:
    """The service's policy on the owner's cell: a pre-open the toggle skips; the steady gate is the arm's own."""
    return GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99, **kwargs)


def _moves(events: "list[Any]") -> "list[tuple[Any, ...]]":
    return [e for e in events if isinstance(e, tuple) and e[0] == "move"]


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop's grasp
# ---------------------------------------------------------------------------------------------------------------------


class ThePolicyJudgesTheLiftBeforeItClosesTests(unittest.TestCase):
    def test_a_lift_that_would_be_refused_leaves_the_jaws_open_and_backs_out_up_the_line(self) -> None:
        """⭐ Red before: the jaws closed on the part, and the lift was refused at its first sample: the arm stood at the
        part with it in its jaws."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=_HELD)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"), report.motion_message)
        self.assertEqual(0, _pulses(events), "the jaws were told to close")
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(_moves(events), [("move", _STANDOFF, False), ("move", _PART, True), ("move", _STANDOFF, True)],
                         "down the line to the part, and back up the same line to the standoff")
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART, _STANDOFF])
        self.assertIs(report.motion_status, MotionStatus.SELF_COLLISION_REJECTED)
        for said in ("judged at the part before the jaws close", "the bin's rim", "jaws stayed open",
                     "back up the line it came down"):
            self.assertIn(said, report.motion_message)
        self.assertIsNone(report.object_detected)
        self.assertIsNone(report.error)

    def test_the_lift_is_judged_once_at_the_part_and_the_controller_is_asked_last_before_the_close(self) -> None:
        """The controller is the last thing asked before the jaws, as it was before the judgement came (the owner-cell
        audit, 2026-09-23): judged at the part, then the controller, then the close (verifier W, 2026-10-01)."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=None)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        judged = [e for e in events if isinstance(e, tuple) and e[0] == "judge"]
        self.assertEqual(judged, [("judge", _LIFT)], "the lift to its top")
        at_part = events.index(("move", _PART, True))
        judge = events.index(judged[0])
        close = _toggle_change(events)
        self.assertLess(at_part, judge)
        self.assertLess(judge, close, "judged before the jaws were told anything")
        self.assertIn("status", events[judge + 1:close], "the controller is asked after the judgement, before the close")
        self.assertEqual(2, events.count("status"), "before anything is commanded, and at the part")

    def test_a_stop_that_falls_while_the_lift_is_judged_closes_nothing(self) -> None:
        """⭐ Red before: the controller was asked at the part, then the lift was judged, then the jaws closed, so a
        protective stop that fell while the lift was judged closed the jaws on an arm that cannot lift."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _StopsWhileJudging(events)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED, report.motion_message)
        self.assertIs(report.motion_status, MotionStatus.CONTROLLER_REJECTED)
        self.assertIn("protective_stop=True", report.motion_message)
        self.assertEqual(0, _pulses(events), "the jaws closed on an arm that cannot lift")
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART], "the arm stands at the part")

    def test_a_lift_that_would_run_is_closed_on_and_lifted_as_before(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=None)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual(1, _pulses(events))
        self.assertEqual(_moves(events), [("move", _STANDOFF, False), ("move", _PART, True), ("move", _LIFT, True)])

    def test_an_arm_that_cannot_judge_grasps_as_before(self) -> None:
        """The dummy, the sim and every arm without the verb: nothing is judged ahead, and the grasp is unchanged."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _LineArm(events)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual(1, _pulses(events))
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "judge"])
        self.assertEqual(_moves(events), [("move", _STANDOFF, False), ("move", _PART, True), ("move", _LIFT, True)])

    def test_a_back_out_refused_too_ends_as_a_failed_motion_saying_both(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=_HELD, refuse_at=2)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertEqual(0, _pulses(events))
        self.assertIs(report.motion_status, MotionStatus.SELF_COLLISION_REJECTED)
        for said in ("judged at the part before the jaws close", "the bin's rim", "jaws stayed open", "the line back up",
                     "the bench"):
            self.assertIn(said, report.motion_message)
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART], "the back out never ran")

    def test_a_back_out_that_raises_ends_as_a_failed_motion_saying_it_raised(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=_HELD, raise_at=2)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertEqual(0, _pulses(events))
        self.assertIsInstance(report.error, RuntimeError)
        for said in ("judged at the part before the jaws close", "jaws stayed open",
                     "the line back up to the standoff raised", "the RTDE link dropped"):
            self.assertIn(said, report.motion_message)
        self.assertNotIn("empty-handed", report.motion_message, "a back-out that raised is no back-out")
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART])

    def test_a_refusal_that_ends_its_sentence_is_not_ended_twice(self) -> None:
        """The camera world's refusal and the driver's own refusals end with a full stop; what the grasp adds starts a
        sentence of its own after it, and no sentence ends twice."""
        events: list[Any] = []
        ended = MotionResult.failed(MotionStatus.UNSUPPORTED, MotionCommand.MOVE_TO,
                                    message="Refused before planning: nothing declined one. Nothing moved.")
        report = _policy(_JudgingArm(events, carried=ended), _toggle(events)).execute(_grasp())

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"))
        self.assertIn("Nothing moved. The jaws stayed open", report.motion_message)
        self.assertNotIn("..", report.motion_message)

    def test_a_judgement_that_raises_closes_nothing(self) -> None:
        """A lift nobody could judge is no lift the grasp may close for: the jaws stay open and the arm backs out."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=RuntimeError("the planner's sidecar exited"))

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"))
        self.assertEqual(0, _pulses(events))
        self.assertIsInstance(report.error, RuntimeError)
        self.assertIn("could not be judged", report.motion_message)
        self.assertEqual(_moves(events)[-1], ("move", _STANDOFF, True))

    def test_a_camera_that_cannot_vouch_while_it_judges_leaves_the_pick_as_on_every_motion(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=CameraWorldUnavailable(camera="wrist", verdict="no frame", attempts=4,
                                                                 reason="the wrist camera stayed silent"))
        with self.assertRaises(CameraWorldUnavailable):
            _policy(arm, jaws).execute(_grasp())
        self.assertEqual(0, _pulses(events))

    def test_a_gripper_told_widths_is_not_closed_either(self) -> None:
        from tests.robot.grasping.motion.test_grasp_execution_policy import _BasicGripper

        events: list[Any] = []
        hand = _BasicGripper()
        arm = _JudgingArm(events, carried=_HELD)

        report = GraspExecutionPolicy(arm=arm, gripper=hand).execute(_grasp())

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"))
        self.assertEqual(hand.width_calls, [], "no width was commanded")

    def test_a_lift_in_steps_is_judged_once_to_its_top(self) -> None:
        """Recorded as built: a lift in steps is judged once, from the part to its top, and runs as its steps; their
        samples are not the judged line's, which guide 04 says (verifier W, 2026-10-01, an owner question)."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=None)

        report = _policy(arm, jaws, retreat_steps=4).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual([e for e in events if isinstance(e, tuple) and e[0] == "judge"], [("judge", _LIFT)])

    def test_a_policy_with_no_gripper_closes_nothing_and_judges_nothing(self) -> None:
        events: list[Any] = []
        arm = _JudgingArm(events, carried=_HELD)
        report = GraspExecutionPolicy(arm=arm, gripper=None).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "judge"])

    def test_the_pick_loop_reads_it_as_a_failed_execution_with_its_reason(self) -> None:
        """No new pick outcome: the attempt is ``execution_failed`` (the part is remembered as failed, so the pick goes
        on to its next part or reports), and it carries the typed status and the sentence."""
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome, _motion_fields

        outcome = getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED")
        self.assertEqual(BinPickingOrchestrator._map_policy_outcome(None, outcome),  # type: ignore[arg-type]
                         ("execution_failed", PickOutcome.EXECUTION_FAILED))
        events: list[Any] = []
        report = _policy(_JudgingArm(events, carried=_HELD), _toggle(events)).execute(_grasp())
        fields = _motion_fields(report)
        self.assertEqual(fields["motion_status"], "self_collision_rejected")
        self.assertIn("jaws stayed open", fields["motion_message"] or "")


def _toggle_change(events: "list[Any]") -> int:
    """Where on the log DO0 changed first: the close."""
    from tests.robot.execution.test_a_stopped_controller_moves_no_jaws import _changes

    changes = _changes(events)
    assert changes, "the jaws were never told to close"
    return changes[0]


# ---------------------------------------------------------------------------------------------------------------------
# Robot.pick
# ---------------------------------------------------------------------------------------------------------------------


def _handling() -> Any:
    from src.robot.execution import handling

    return handling


#: Where ``Robot.pick``'s doubles pick: tool down, 100 mm over the bench, its standoff 80 mm up.
_PICK_AT = (400.0, 0.0, 100.0)
_PICK_STANDOFF = (400.0, 0.0, 180.0)


def _pick_pose() -> Pose:
    return Pose(position_mm=np.array(_PICK_AT), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE)


class RobotPickJudgesTheLineOutBeforeItClosesTests(unittest.TestCase):
    @staticmethod
    def _robot(carried: "MotionResult | BaseException | None", **hand: Any) -> "tuple[Any, Any]":
        from src.robot.execution.robot import Robot
        from tests.robot.safety.test_robot_pick_and_place import _Log, _LoggedHand, _RecordingArm

        class _Judging(_RecordingArm):
            def carried_line_refusal(self, pose: Pose) -> "MotionResult | None":
                self.log.entries.append(("judge", _mm(pose)))
                if isinstance(carried, BaseException):
                    raise carried
                return carried

        log = _Log()
        arm = _Judging(log)
        arm.connect()
        hand = _LoggedHand(log, **{"hold": HoldEvidence.HELD, "width_mm": 39.0, **hand})
        return Robot.from_parts(arm=arm, gripper=hand, lock_key=None), log

    def test_a_line_out_that_would_be_refused_closes_nothing_and_backs_out(self) -> None:
        """⭐ Red before: the jaws closed at 39 mm, then the line out was refused and the arm stood with the part."""
        handling = _handling()
        robot, log = self._robot(_HELD)

        report = robot.pick(Pose(position_mm=np.array([400.0, 0.0, 100.0]),
                                 quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE), 40.0)

        self.assertIs(report.outcome, getattr(handling.HandlingOutcome, "CARRIED_RETREAT_REFUSED", "missing"),
                      report.render())
        moves = [(e[1], e[2]) for e in log.entries if e[0] == "move"]
        self.assertEqual(moves, [((400.0, 0.0, 180.0), False), ((400.0, 0.0, 100.0), True),
                                 ((400.0, 0.0, 180.0), True)])
        self.assertNotIn(("set_width", 39.0), log.entries, "the jaws were told to close")
        self.assertIn(("judge", (400.0, 0.0, 180.0)), log.entries)
        for said in ("judged at the part before the jaws close", "the bin's rim", "jaws stayed open"):
            self.assertIn(said, report.message)

    def test_a_line_out_that_would_run_picks_as_before(self) -> None:
        handling = _handling()
        robot, log = self._robot(None)
        report = robot.pick(Pose(position_mm=np.array([400.0, 0.0, 100.0]),
                                 quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE), 40.0)
        self.assertIs(report.outcome, handling.HandlingOutcome.EXECUTED, report.render())
        self.assertIn(("set_width", 39.0), log.entries)
        judge = log.entries.index(("judge", (400.0, 0.0, 180.0)))
        self.assertLess(judge, log.entries.index(("set_width", 39.0)))

    def test_a_back_out_refused_too_says_both_and_closes_nothing(self) -> None:
        handling = _handling()
        robot, log = self._robot(_HELD)
        original = robot.arm.move

        def refuse_the_way_back(pose: Pose, **kwargs: Any) -> MotionResult:
            if kwargs.get("linear") and float(pose.position_mm[2]) > 150.0:
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           target_pose=pose, message="the bench")
            return original(pose, **kwargs)

        robot.arm.move = refuse_the_way_back  # type: ignore[method-assign]
        report = robot.pick(Pose(position_mm=np.array([400.0, 0.0, 100.0]),
                                 quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE), 40.0)
        self.assertIs(report.outcome, handling.HandlingOutcome.MOTION_REFUSED)
        self.assertNotIn(("set_width", 39.0), log.entries)
        for said in ("judged at the part before the jaws close", "the bin's rim", "jaws stayed open", "the bench"):
            self.assertIn(said, report.message)

    def test_a_line_out_nobody_could_judge_closes_nothing_and_backs_out(self) -> None:
        """A judgement that raises is no judgement the jaws may close on: they stay open, and the arm goes back up the
        line it came down (verifier W, 2026-10-01: nothing pinned it)."""
        handling = _handling()
        robot, log = self._robot(RuntimeError("the planner's sidecar exited"))

        report = robot.pick(_pick_pose(), 40.0)

        self.assertIs(report.outcome, getattr(handling.HandlingOutcome, "CARRIED_RETREAT_REFUSED", "missing"),
                      report.render())
        self.assertFalse([e for e in log.entries if e[0] == "set_width" and e[1] < 85.0], "the jaws were told to close")
        moves = [(e[1], e[2]) for e in log.entries if e[0] == "move"]
        self.assertEqual(moves, [(_PICK_STANDOFF, False), (_PICK_AT, True), (_PICK_STANDOFF, True)])
        for said in ("could not be judged", "RuntimeError", "the planner's sidecar exited", "jaws stayed open"):
            self.assertIn(said, report.message)

    def test_a_camera_that_cannot_vouch_while_it_judges_ends_the_pick_as_its_own_outcome(self) -> None:
        """As on every motion of the verb: its own outcome, the jaws untold, and nothing commanded after it."""
        handling = _handling()
        robot, log = self._robot(CameraWorldUnavailable(camera="wrist", verdict="no frame", attempts=4,
                                                        reason="the wrist camera stayed silent"))

        report = robot.pick(_pick_pose(), 40.0)

        self.assertIs(report.outcome, handling.HandlingOutcome.CAMERA_WORLD_UNAVAILABLE, report.render())
        self.assertFalse([e for e in log.entries if e[0] == "set_width" and e[1] < 85.0], "the jaws were told to close")
        moves = [(e[1], e[2]) for e in log.entries if e[0] == "move"]
        self.assertEqual(moves, [(_PICK_STANDOFF, False), (_PICK_AT, True)], "nothing after the judgement")
        self.assertIn("the wrist camera stayed silent", report.message)

    def test_a_refusal_that_ends_its_sentence_is_not_ended_twice(self) -> None:
        handling = _handling()
        ended = MotionResult.failed(MotionStatus.UNSUPPORTED, MotionCommand.MOVE_TO,
                                    message="Refused before planning: nothing declined one. Nothing moved.")
        robot, _ = self._robot(ended)

        report = robot.pick(_pick_pose(), 40.0)

        self.assertIs(report.outcome, getattr(handling.HandlingOutcome, "CARRIED_RETREAT_REFUSED", "missing"))
        self.assertIn("Nothing moved. The jaws stayed open", report.message)
        self.assertNotIn("..", report.message)


class RobotPickOnTheUrArmJudgesWithItsOwnDeclineTests(unittest.TestCase):
    """``Robot.pick(..., decline=...)`` on the UR driver with no live camera world, as example 05 and guide 06 run it:
    every motion of the pick carries the decline, and the judgement of its line out carries it too. Red before: the
    judgement was asked with no decline, read the camera world MISSING and refused, so a declined pick never closed and
    blamed the carried part (verifier W, 2026-10-01)."""

    _BENCH_DECLINE = "a known part on a clear table, no camera"

    @staticmethod
    def _pick(joints: "list[float]", **kwargs: Any) -> "tuple[Any, list[Any], Any, Any]":
        """The owner-like UR10 standing at ``joints`` beside the 30 mm bin the camera saw, no live world wired, the
        owner's toggle in hand (as ``connect_cell`` hands it to the arm), picking the part right under its tool."""
        from src.robot.execution.robot import Robot
        from tests.robot.grippers.test_what_the_hand_verbs_read import running_normally

        arm, planner = _cell()
        _standing_at(arm, planner, joints)
        running_normally(arm._conn)
        arm._drive_curobo = lambda pose, **_: MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)
        events: list[Any] = []
        jaws = _toggle(events)
        arm.set_hand(jaws)
        robot = Robot.from_parts(arm=arm, gripper=jaws, lock_key=None)
        report = robot.pick(arm.get_tcp_pose(), 40.0, **kwargs)
        return report, events, arm, jaws

    def test_away_from_the_bin_a_declined_pick_closes_once_and_lifts(self) -> None:
        """⭐ The pick of example 05: away from the bin it runs, one change of DO0, a moveL down and one up."""
        handling = _handling()
        decline = self._BENCH_DECLINE
        for name, how in (("decline=", {"decline": decline}), ("camera_world=", {"camera_world": CameraWorldDecline(
                decline)})):
            with self.subTest(name):
                report, events, arm, jaws = self._pick(AWAY, **how)
                self.assertIs(report.outcome, handling.HandlingOutcome.EXECUTED, report.render())
                self.assertEqual(1, _pulses(events))
                self.assertTrue(jaws.jaws_closed)
                self.assertEqual(arm._motion.move_to.call_count, 2, "a moveL down and a moveL up")

    def test_beside_the_bin_a_declined_pick_closes_and_lifts(self) -> None:
        """⭐ The owner, 2026-10-05 ("mehr Griffe"; every cell since 2026-10-17): the pick beside the bin is judged as an
        empty hand is, carrying too: one change of DO0, a moveL down and one up."""
        handling = _handling()
        report, events, arm, jaws = self._pick(OFF, decline=self._BENCH_DECLINE)

        self.assertIs(report.outcome, handling.HandlingOutcome.EXECUTED, report.render())
        self.assertEqual(1, _pulses(events))
        self.assertTrue(jaws.jaws_closed)
        self.assertEqual(arm._motion.move_to.call_count, 2, "a moveL down and one up")


# ---------------------------------------------------------------------------------------------------------------------
# The UR arm's judgement
# ---------------------------------------------------------------------------------------------------------------------


def _standing_at(arm: Any, planner: Any, joints: "list[float]") -> Pose:
    """Stand the arm at ``joints`` for a tool line: the controller reports the flange there, and every sample of a short
    line solves to them. Returns the TCP 100 mm straight up from where it stands: the lift."""
    _from(arm, planner, joints)
    arm.ik = lambda pose, *, seed=None: JointPositions(tuple(joints))
    arm._motion = MagicMock()
    arm._motion.move_to.return_value = True
    flange = np.asarray(ur_link_transforms_mm("ur10", np.asarray(joints))[-1])
    arm._motion.get_current_pose.return_value = pose_to_urpose(Pose.from_matrix(flange, frame=Frame.BASE))
    here = arm.get_tcp_pose()
    return Pose(position_mm=np.asarray(here.position_mm) + np.array([0.0, 0.0, 100.0]),
                quaternion_xyzw=np.asarray(here.quaternion_xyzw), frame=Frame.BASE, label="lift")


class TheUrArmJudgesALiftAsTheLineUpWillMeetItTests(unittest.TestCase):
    def test_beside_the_bin_the_lift_is_judged_as_an_empty_hand_is(self) -> None:
        """⭐ The owner, 2026-10-05 ("mehr Griffe"; every cell since 2026-10-17): the lift beside the bin runs as it runs
        empty-handed, the camera's boxes set aside and the exact guard deciding. Nothing is sent."""
        arm, planner = _cell()
        lift = _standing_at(arm, planner, OFF)
        with arm.without_camera_world(_DECLINED):
            self.assertIsNone(arm.carried_line_refusal(lift))
        arm._motion.move_to.assert_not_called()
        arm._conn.moveJ.assert_not_called()
        self.assertNotEqual(planner.set_aside(), [], "the camera's boxes were set aside, as for an empty hand")

    def test_away_from_the_bin_the_lift_would_run(self) -> None:
        arm, planner = _cell()
        lift = _standing_at(arm, planner, AWAY)
        with arm.without_camera_world(_DECLINED):
            self.assertIsNone(arm.carried_line_refusal(lift))
        arm._motion.move_to.assert_not_called()

    def test_a_decline_handed_to_the_verb_is_the_decline_its_move_would_carry(self) -> None:
        """⭐ ``carried_line_refusal(..., camera_world=CameraWorldDecline(...))`` answers what ``move(pose, linear=True,
        camera_world=...)`` answers. Red before: the verb took no decline, so with no live world wired every judgement
        read the camera world MISSING and refused (verifier W, 2026-10-01)."""
        decline = CameraWorldDecline(_DECLINED)
        arm, planner = _cell()
        for joints in (AWAY, OFF):
            with self.subTest(joints=joints):
                self.assertIsNone(arm.carried_line_refusal(_standing_at(arm, planner, joints), camera_world=decline))
        self.assertTrue(planner.named("check"), "the lifts were judged")
        arm._motion.move_to.assert_not_called()

        arm._live_world = MagicMock()  # a live world wired: a declined motion is refused, as move refuses it
        planner.calls.clear()
        refused = arm.carried_line_refusal(_standing_at(arm, planner, AWAY), camera_world=decline)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.UNSUPPORTED)
        self.assertFalse(planner.named("check"), "nothing was judged")

    def test_a_lift_with_no_camera_world_is_refused_as_its_move_would_be(self) -> None:
        """The verb answers what ``move(pose, linear=True)`` answers: on cuRobo with no live world wired and nothing
        declined, the camera world's own refusal, before anything is handed over or judged."""
        arm, planner = _cell()
        lift = _standing_at(arm, planner, OFF)
        refused = arm.carried_line_refusal(lift)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.UNSUPPORTED)
        self.assertFalse(planner.named("check"), "nothing was judged")

    def test_a_pose_not_in_base_is_refused_before_anything_is_judged(self) -> None:
        arm, planner = _cell()
        lift = _standing_at(arm, planner, OFF)
        tool = Pose(position_mm=lift.position_mm, quaternion_xyzw=lift.quaternion_xyzw, frame=Frame.TOOL)
        with arm.without_camera_world(_DECLINED):
            refused = arm.carried_line_refusal(tool)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.INVALID_TARGET)
        self.assertFalse(planner.named("check"), "nothing was judged")

    def test_the_policy_on_the_owners_arm_beside_the_bin_closes_and_lifts(self) -> None:
        """⭐ The whole grasp on the UR driver: the line down to the pose beside the bin runs, the lift judged before the
        jaws close would run as it runs empty-handed (Option 1, every cell since 2026-10-17), the jaws close once, and
        the lift runs, one moveL each way. Red on a cell that modelled the carried part: the jaws stayed open and the
        arm backed out."""
        from src.geometry.quaternion import to_rotation_matrix
        from tests.robot.grippers.test_what_the_hand_verbs_read import running_normally

        arm, planner = _cell()
        _standing_at(arm, planner, OFF)
        running_normally(arm._conn)
        arm._drive_curobo = lambda pose, **_: MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)
        events: list[Any] = []
        jaws = _toggle(events)
        # The part right where the tool stands at OFF, approached along the tool's own axis, so the line down to it
        # turns nothing; every sample of a line here solves to OFF.
        part = arm.get_tcp_pose()
        turned = to_rotation_matrix(np.asarray(part.quaternion_xyzw, dtype=np.float64))
        grasp = GraspPoint(position=np.asarray(part.position_mm), approach=turned[:, 2], axis=turned[:, 0],
                           grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="beside the bin")
        with arm.without_camera_world(_DECLINED):
            report = GraspExecutionPolicy(arm=arm, gripper=jaws).execute(grasp)

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual(1, _pulses(events))
        self.assertTrue(jaws.jaws_closed)
        self.assertEqual(arm._motion.move_to.call_count, 2, "one moveL down to the part, one up")


def _ik_arm() -> Any:
    """The owner-like UR10 on the ``ik`` planner, as the URSim profile runs it: the controller draws a line and only its
    end is gated. It stands at (-30, -60, 80, -95, -90, 0) deg, and every pose solves to there."""
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE

    arm: Any = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10"}, "motion": {"planner": "ik", "workspace": {**OPEN_WORKSPACE, "z_max": 1000.0}},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "perceived_min_distance_mm": 4.0,
                                      "planner_margin_mm": 4.0}},
    }))
    # What is read is how a lift is judged; the IK quality guard is no part of it (its tree switched it off until the
    # switch left every tree, 2026-10-17).
    without_guards(arm._preflight, "ik_quality")
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(AWAY)
    arm.ik = lambda pose, *, seed=None: JointPositions(tuple(AWAY))
    return arm


class OnTheIkPlannerTheJudgementRemembersNothingTests(unittest.TestCase):
    """On the ``ik`` planner a line's end is gated and nothing else, and so is the judgement of a lift: through the same
    guards in the same order, screened, so the continuity memo stays where the last commanded target left it. Red before:
    it went through the commanded pipeline, which remembers an accepted target, and the lift's own continuity was then
    judged from its top to its top (verifier W, 2026-10-01)."""

    def _at(self, z_mm: float, label: str) -> Pose:
        return Pose(position_mm=np.array([-130.0, -600.0, z_mm]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                    frame=Frame.BASE, label=label)

    def test_a_lift_it_would_run_leaves_the_memo_on_the_part(self) -> None:
        _needs_the_engine()
        arm = _ik_arm()
        part = self._at(300.0, "the part")
        self.assertIsNone(arm._gate_pose(part, MotionCommand.MOVE_TO), "the line down to the part is gated, commanded")
        self.assertIs(arm._preflight._last_target_pose, part)

        self.assertIsNone(arm.carried_line_refusal(self._at(400.0, "the lift's top")))

        self.assertIs(arm._preflight._last_target_pose, part, "a lift only judged is no place the arm was sent")

    def test_a_lift_it_would_refuse_is_refused_by_the_guard_the_move_meets(self) -> None:
        _needs_the_engine()
        arm = _ik_arm()
        part = self._at(300.0, "the part")
        arm._gate_pose(part, MotionCommand.MOVE_TO)

        refused = arm.carried_line_refusal(self._at(1400.0, "a top past the workspace"))

        assert refused is not None
        self.assertIs(refused.status, MotionStatus.WORKSPACE_REJECTED)
        self.assertIs(arm._preflight._last_target_pose, part)


if __name__ == "__main__":
    unittest.main()
