"""A grasp judges its lift as if the jaws held the part, before they close, and closes only where the lift would run.

The owner, 2026-10-01 ("Vorher prüfen"), on verifier W's M1. Option 1 lets an empty hand down a straight line into a pose
beside a bin the camera saw, where only the camera's boxes refuse the planner's world. Once the jaws close on a part
nothing is set aside, so the lift out of that pose, which starts there, is refused at its first sample: the arm stood
beside the bin with the part in its jaws until a person released it (GPU, verifier W, 2026-10-01).

Now the grasp asks the arm first, at the part with its jaws still open, whether the lift would run carrying the part
(``JudgesCarriedLines.carried_line_refusal``): the lift's own judgement, with the part in the planner's world as the
attach after the close hands it over and nothing of the camera's world set aside. Where it would be refused, the jaws
stay open, the arm goes back up the line it came down, to the standoff, empty-handed, and the attempt ends with its own
outcome (``PolicyOutcome.CARRIED_RETREAT_REFUSED``, ``HandlingOutcome.CARRIED_RETREAT_REFUSED``), the reason beside it,
so the pick goes on to its next part or reports. Where it would run, the grasp closes and lifts as before. An arm that
cannot judge (the dummy, the sim) grasps as before. Both grasps judge: the pick loop's policy and ``Robot.pick``.

The hand here is the real ``JawIOGripper`` on a recording tool I/O, as on the owner's cell, so "jaws stayed open"
is read off the outputs. Names new with this change are imported inside the tests.

Verifier W's second round (2026-10-01) holds the rest: ``Robot.pick`` judges its line out with the decline its motions
carry, the controller is the last thing asked before the jaws, a hand that does not measure is judged at the width its
attach carries, an ik arm screens the lift's end and remembers nothing, and an arm that already holds a part, modelled or
not, is judged as it stands.
"""

from __future__ import annotations

import math
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
from src.robot.core.arm_capabilities import LineMotion, LineReading, PayloadModel
from src.robot.core.camera_world import CameraWorldDecline
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.ur.pose_adapter import pose_to_urpose
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import (
    _BENCH,
    _DECLINED,
    AWAY,
    OFF,
    Q,
    SeenWorldPlanner,
    _cell,
    _from,
    _needs_the_engine,
    _seen,
)
from tests.test_a_stopped_controller_moves_no_jaws import _PROTECTIVE, _pulses, _toggle

_RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                       emergency_stopped=False)
_CAPS = RobotCapabilities(
    vendor="ur", model="ur10", dof=6, supports_joint_move=True, supports_linear_move=True,
    supports_async_move=False, has_native_fk=True, has_native_ik=True, has_force_control=False,
    is_simulated=False,
)
#: What the arm says of a lift it would refuse with the part in its jaws: the planner's world, a part carried.
_HELD = MotionResult.failed(
    MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
    message=("the planner refused this line at sample 0 of 12: the planner's world; it stands: a part is carried, and "
             "the carried part is the planner's alone"))


def _mm(pose: Pose) -> "tuple[float, ...]":
    return tuple(round(float(v), 1) for v in pose.position_mm)


class _LineArm:
    """A UR-like arm that keeps its lines (each sample judged before one moveL) and reports a controller that can move.
    Every call lands on the log the hand shares: ``("move", position, linear)``, ``"status"``, ``"steady"``,
    ``"detach"``, ``("attach", width)``. ``refuse_at`` refuses the motion of that index, nothing sent; ``raise_at`` raises
    out of it."""

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

    def recover_from_protective_stop(self) -> bool:  # pragma: no cover - never called, by design
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

    def detach_payload(self) -> bool:
        self.events.append("detach")
        return True

    def attach_payload(self, width_mm: float) -> bool:
        self.events.append(("attach", round(float(width_mm), 2)))
        return True


class _JudgingArm(_LineArm):
    """The same arm, judging a lift as if its jaws held the part (``JudgesCarriedLines``): ``carried`` is what it
    answers, a refusal, ``None`` (the lift would run) or an exception it raises."""

    def __init__(self, events: "list[Any]", *, carried: "MotionResult | BaseException | None" = None,
                 **kwargs: Any) -> None:
        super().__init__(events, **kwargs)
        self.carried = carried

    def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float) -> "MotionResult | None":
        self.events.append(("judge", _mm(pose), round(float(grip_width_mm), 2)))
        if isinstance(self.carried, BaseException):
            raise self.carried
        return self.carried


class _StopsWhileJudging(_JudgingArm):
    """The judging arm whose controller falls into a protective stop while it judges the lift, which would run: the
    judgement is IK round trips, a world refresh and the planner's attach, check and detach, about half a second on the
    development GPU, and a stop can fall inside it (verifier W, 2026-10-01)."""

    def __init__(self, events: "list[Any]") -> None:
        super().__init__(events, carried=None)
        self.stopped = False

    def get_robot_status(self) -> RobotStatus:
        running = super().get_robot_status()
        return _PROTECTIVE if self.stopped else running

    def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float) -> "MotionResult | None":
        judged = super().carried_line_refusal(pose, grip_width_mm=grip_width_mm)
        self.stopped = True
        return judged


class _MeasuringHand:
    """A gripper that measures its width, as a Robotiq on its socket does: 50 mm open, it stops on the real part, 46 mm
    across where the grasp said 40."""

    min_width_mm = 0.0
    max_width_mm = 50.0
    is_connected = True

    def __init__(self) -> None:
        self.width = 50.0

    def width_is_measured(self) -> bool:
        return True

    def set_width_mm(self, width_mm: float, *, speed: Any = None, force: Any = None) -> None:
        self.width = 50.0 if float(width_mm) >= 50.0 else max(float(width_mm), 46.0)

    def get_width_mm(self) -> float:
        return self.width


def _grasp(width_mm: float = 40.0) -> GraspPoint:
    """A part at the owner's work area, closing along base -Y: the standoff 80 mm over it, the lift 100 mm."""
    return GraspPoint(position=np.array([-130.0, -600.0, 80.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([0.0, -1.0, 0.0]), grip_width_mm=width_mm, score=0.9, frame=GraspFrame.BASE,
                      label="part")


_STANDOFF = (-130.0, -600.0, 160.0)
_PART = (-130.0, -600.0, 80.0)
_LIFT = (-130.0, -600.0, 180.0)


def _policy(arm: Any, jaws: Any, **kwargs: Any) -> GraspExecutionPolicy:
    """The service's policy on the owner's cell: the steady gate on, a pre-open the toggle skips."""
    return GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99, require_steady_before_motion=True,
                                **kwargs)


def _moves(events: "list[Any]") -> "list[tuple[Any, ...]]":
    return [e for e in events if isinstance(e, tuple) and e[0] == "move"]


# ---------------------------------------------------------------------------------------------------------------------
# The pick loop's grasp
# ---------------------------------------------------------------------------------------------------------------------


class ThePolicyJudgesTheLiftBeforeItClosesTests(unittest.TestCase):
    def test_a_lift_refused_carrying_leaves_the_jaws_open_and_backs_out_up_the_line(self) -> None:
        """⭐ Red before: the jaws closed on the part, the part was attached, and the lift was refused at its first
        sample: the arm stood at the part with it in its jaws."""
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=_HELD)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"), report.motion_message)
        self.assertEqual(0, _pulses(events), "the jaws were told to close")
        self.assertFalse(jaws.jaws_closed)
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "attach"], "a part was attached")
        self.assertEqual(_moves(events), [("move", _STANDOFF, False), ("move", _PART, True), ("move", _STANDOFF, True)],
                         "down the line to the part, and back up the same line to the standoff")
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART, _STANDOFF])
        self.assertIs(report.motion_status, MotionStatus.SELF_COLLISION_REJECTED)
        for said in ("as if the jaws held the part", "a part is carried", "jaws stayed open",
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
        self.assertEqual(judged, [("judge", _LIFT, 39.0)], "the lift to its top, at the width the attach carries")
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
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "attach"], "a part was attached")
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART], "the arm stands at the part")

    def test_a_lift_that_would_run_is_closed_on_and_lifted_as_before(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        arm = _JudgingArm(events, carried=None)

        report = _policy(arm, jaws).execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual(1, _pulses(events))
        self.assertIn(("attach", 39.0), events)
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
        for said in ("as if the jaws held the part", "jaws stayed open", "the line back up", "the bench"):
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
        for said in ("as if the jaws held the part", "jaws stayed open", "the line back up to the standoff raised",
                     "the RTDE link dropped"):
            self.assertIn(said, report.motion_message)
        self.assertNotIn("empty-handed", report.motion_message, "a back-out that raised is no back-out")
        self.assertEqual([_mm(p) for p in report.waypoints], [_STANDOFF, _PART])

    def test_a_refusal_that_ends_its_sentence_is_not_ended_twice(self) -> None:
        """The camera world's refusal and the driver's carried-part refusal end with a full stop; what the grasp adds
        starts a sentence of its own after it, and no sentence ends twice."""
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
        from tests.test_grasp_execution_policy import _BasicGripper

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
        self.assertEqual([e for e in events if isinstance(e, tuple) and e[0] == "judge"], [("judge", _LIFT, 39.0)])

    def test_the_lift_is_judged_at_the_width_the_attach_after_the_close_carries(self) -> None:
        """⭐ A hand that does not measure its width is attached after the close at the width it was told, so the lift is
        judged at exactly that: the box the attach hands the planner, the lift's own part. Red before: the toggle's lift
        was judged 40 mm across and lifted 39 (verifier W, 2026-10-01). A hand that measures is attached at what it
        measures, which nobody knows before the close; the part's own width stands in for it there, never a narrower
        one than the jaws are told."""
        cases = (("a toggle", {}, None, 39.0, 39.0), ("a toggle told 46 mm", {"close_width_mm": 46.0}, None, 46.0, 46.0),
                 ("a hand that measures", {}, _MeasuringHand(), 40.0, 46.0))
        for name, kwargs, hand, judged, attached in cases:
            with self.subTest(name):
                events: list[Any] = []
                arm = _JudgingArm(events, carried=None)
                gripper = hand if hand is not None else _toggle(events)
                report = GraspExecutionPolicy(arm=arm, gripper=gripper, **kwargs).execute(_grasp())
                self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
                self.assertEqual([e[2] for e in events if isinstance(e, tuple) and e[0] == "judge"], [judged])
                self.assertEqual([e[1] for e in events if isinstance(e, tuple) and e[0] == "attach"], [attached])

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
    from tests.test_a_stopped_controller_moves_no_jaws import _changes

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
        from tests.test_robot_pick_and_place import _Log, _LoggedHand, _RecordingArm

        class _Judging(_RecordingArm):
            def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float) -> "MotionResult | None":
                self.log.entries.append(("judge", _mm(pose), round(float(grip_width_mm), 2)))
                if isinstance(carried, BaseException):
                    raise carried
                return carried

            def attach_payload(self, width_mm: float) -> bool:
                self.log.entries.append(("attach", round(float(width_mm), 2)))
                return True

            def payload_declined_reason(self) -> "str | None":
                return None

            def payload_model(self) -> PayloadModel:
                return PayloadModel.PLANNER_AND_FILTER

        log = _Log()
        arm = _Judging(log)
        arm.connect()
        hand = _LoggedHand(log, **{"hold": HoldEvidence.HELD, "width_mm": 39.0, **hand})
        return Robot.from_parts(arm=arm, gripper=hand, lock_key=None), log

    def test_a_line_out_refused_carrying_closes_nothing_and_backs_out(self) -> None:
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
        self.assertIn(("judge", (400.0, 0.0, 180.0), 40.0), log.entries)
        for said in ("as if the jaws held the part", "a part is carried", "jaws stayed open"):
            self.assertIn(said, report.message)

    def test_a_line_out_that_would_run_picks_as_before(self) -> None:
        handling = _handling()
        robot, log = self._robot(None)
        report = robot.pick(Pose(position_mm=np.array([400.0, 0.0, 100.0]),
                                 quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]), frame=Frame.BASE), 40.0)
        self.assertIs(report.outcome, handling.HandlingOutcome.EXECUTED, report.render())
        self.assertIn(("set_width", 39.0), log.entries)
        judge = log.entries.index(("judge", (400.0, 0.0, 180.0), 40.0))
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
        for said in ("as if the jaws held the part", "jaws stayed open", "the bench"):
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

    def test_the_line_out_is_judged_at_the_width_the_attach_after_the_close_carries(self) -> None:
        """⭐ A hand that does not measure its width is attached at what it was told, so the line out is judged at that
        width: the box the attach hands the planner. Red before: judged 40 mm across, carried 39. A hand that measures
        keeps the part's own width as the stand-in for what it will measure, never narrower than it is told."""
        for name, hand, judged, attached in (("a hand that does not measure", {"measured": False}, 39.0, 39.0),
                                              ("a hand that measures", {"measured": True}, 40.0, 39.0)):
            with self.subTest(name):
                handling = _handling()
                robot, log = self._robot(None, **hand)
                report = robot.pick(_pick_pose(), 40.0)
                self.assertIs(report.outcome, handling.HandlingOutcome.EXECUTED, report.render())
                self.assertEqual([e[2] for e in log.entries if e[0] == "judge"], [judged])
                self.assertEqual([e[1] for e in log.entries if e[0] == "attach"], [attached])

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
    def _pick(joints: "list[float]", *, modelled: bool = False, **kwargs: Any) -> "tuple[Any, list[Any], Any, Any]":
        """The owner-like UR10 standing at ``joints`` beside the 30 mm bin the camera saw, no live world wired, the
        owner's toggle in hand (as ``connect_cell`` hands it to the arm), picking the part right under its tool;
        ``modelled`` on a cell that models the carried part."""
        from src.robot.execution.robot import Robot
        from tests.test_what_the_hand_verbs_read import running_normally

        arm, planner = _modelled_cell() if modelled else _cell()
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

    def test_beside_the_bin_a_declined_pick_backs_out_on_the_carried_part(self) -> None:
        """The decline lets the judgement run; beside the bin, on a cell that models the carried part, it refuses on the
        part it judged carried, and the sentence says so, not that nothing declined the camera world."""
        handling = _handling()
        report, events, arm, jaws = self._pick(OFF, modelled=True, decline=self._BENCH_DECLINE)

        self.assertIs(report.outcome, getattr(handling.HandlingOutcome, "CARRIED_RETREAT_REFUSED", "missing"),
                      report.render())
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.jaws_closed)
        for said in ("as if the jaws held the part", "at sample 0 of", "a part is carried", "The jaws stayed open"):
            self.assertIn(said, report.message)
        self.assertNotIn("no live camera world", report.message)
        self.assertNotIn("..", report.message)
        self.assertEqual(arm._motion.move_to.call_count, 2, "a moveL down and the same line back up")
        self.assertIs(arm.payload_model(), PayloadModel.NONE)
        self.assertFalse(arm._closed_on_part)

    def test_beside_the_bin_on_a_cell_that_models_no_part_the_pick_closes_and_lifts(self) -> None:
        """⭐ The owner, 2026-10-05 ("mehr Griffe"): the same pick on a cell that models no carried part is judged as an
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


class _CarryingPlanner(SeenWorldPlanner):
    """The planner double, taking the carried part the arm hands it as the glue does."""

    def attach_payload(self, joints: Any, dims_mm: Any, centre_mm: Any) -> bool:
        self.calls.append(("attach", list(joints), list(dims_mm), list(centre_mm)))
        self.carries_part = True
        return True


def _modelled_cell() -> "tuple[Any, _CarryingPlanner]":
    """The owner-like arm at Q beside the 30 mm bin, on a cell that models the carried part (a 120 mm length)."""
    _needs_the_engine()
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE
    from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import KNOWN_OPEN

    arm: Any = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "curobo"},
        "workspace_limits": OPEN_WORKSPACE,
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0},
                   "planning_world": {"payload": {"length_mm": 120.0}}},
    }))
    arm._conn = MagicMock()
    arm._conn.is_connected = True
    arm._conn.get_joint_positions.return_value = list(Q)
    arm._conn.moveJ.return_value = True
    boxes = _seen(30.0)
    arm._preflight.set_perceived_obstacles(boxes)
    planner = _CarryingPlanner(arm, (_BENCH, *boxes))
    planner.here = list(Q)
    arm._curobo_ur = planner
    arm.set_hand(KNOWN_OPEN)
    return arm, planner


class TheUrArmJudgesALiftAsIfItCarriedThePartTests(unittest.TestCase):
    def test_beside_the_bin_the_lift_is_refused_carrying_and_runs_empty_handed(self) -> None:
        """⭐ At OFF beside the bin 30 mm away, on a cell that models the carried part: empty-handed the lift's moveL is
        admitted (Option 1); judged as if the jaws held a part it is refused at its first sample, a part carried, and
        nothing is sent either way."""
        arm, planner = _modelled_cell()
        lift = _standing_at(arm, planner, OFF)

        with arm.without_camera_world(_DECLINED):
            refused = arm.carried_line_refusal(lift, grip_width_mm=40.0)

        self.assertIsNotNone(refused)
        assert refused is not None
        self.assertFalse(refused.ok)
        self.assertIs(refused.status, MotionStatus.SELF_COLLISION_REJECTED)
        for said in ("at sample 0 of", "the planner's world", "a part is carried"):
            self.assertIn(said, refused.message or "")
        arm._motion.move_to.assert_not_called()
        arm._conn.moveJ.assert_not_called()
        self.assertEqual(planner.set_aside(), [], "nothing set aside while it judged carrying")
        self.assertIs(arm.payload_model(), PayloadModel.NONE)
        self.assertIsNone(arm._attached_payload)
        self.assertFalse(arm._closed_on_part, "the arm forgot the part it judged with")

        with arm.without_camera_world(_DECLINED):
            moved = arm.move(lift, linear=True)
        self.assertTrue(moved.ok, moved.message)
        arm._motion.move_to.assert_called_once()

    def test_on_a_cell_that_models_no_part_the_lift_is_judged_as_an_empty_hand_is(self) -> None:
        """⭐ The owner, 2026-10-05 ("mehr Griffe"): the same lift beside the bin, on a cell that models no carried part,
        runs as it runs empty-handed, the camera's boxes set aside and the exact guard deciding. Nothing is sent."""
        arm, planner = _cell()
        self.assertIsNotNone(arm.payload_declined_reason())
        lift = _standing_at(arm, planner, OFF)
        with arm.without_camera_world(_DECLINED):
            self.assertIsNone(arm.carried_line_refusal(lift, grip_width_mm=40.0))
        arm._motion.move_to.assert_not_called()
        arm._conn.moveJ.assert_not_called()
        self.assertFalse(arm._closed_on_part, "the arm forgot the part it judged with")

    def test_away_from_the_bin_the_lift_would_run_carrying(self) -> None:
        arm, planner = _cell()
        lift = _standing_at(arm, planner, AWAY)
        with arm.without_camera_world(_DECLINED):
            self.assertIsNone(arm.carried_line_refusal(lift, grip_width_mm=40.0))
        arm._motion.move_to.assert_not_called()
        self.assertFalse(arm._closed_on_part)

    def test_a_cell_that_models_the_part_hands_it_to_the_planner_for_the_judgement_and_takes_it_back(self) -> None:
        arm, planner = _modelled_cell()
        self.assertIsNone(arm.payload_declined_reason(), "this cell models the carried part")
        lift = _standing_at(arm, planner, OFF)
        carried_while_judged: list[bool] = []
        original = planner.check_joint_path

        def check(samples: Any, **kwargs: Any) -> Any:
            carried_while_judged.append(planner.carries_part)
            return original(samples, **kwargs)

        planner.check_joint_path = check  # type: ignore[method-assign]
        with arm.without_camera_world(_DECLINED):
            refused = arm.carried_line_refusal(lift, grip_width_mm=40.0)
        self.assertIsNotNone(refused)
        self.assertEqual(carried_while_judged, [True], "the planner held the part while it judged the lift")
        names = [call[0] for call in planner.calls if call[0] in ("attach", "detach", "check")]
        self.assertEqual(names, ["attach", "check", "detach"])
        self.assertFalse(planner.carries_part)
        self.assertIs(arm.payload_model(), PayloadModel.NONE)

    def test_an_arm_already_holding_a_part_is_judged_as_it_stands_and_keeps_it(self) -> None:
        """Whatever says a part is held: the arm's model, the jaws closed on a part nobody models, or a part only the
        planner still holds. Nothing is handed over or taken back, and every mark stays (verifier W, 2026-10-01: the
        last two were unpinned; without them the judgement forgot the jaws had closed, or attached a second part)."""
        with self.subTest("a part the arm and the planner model"):
            arm, planner = _modelled_cell()
            lift = _standing_at(arm, planner, OFF)
            self.assertTrue(arm.attach_payload(40.0))
            planner.calls.clear()
            with arm.without_camera_world(_DECLINED):
                self.assertIsNotNone(arm.carried_line_refusal(lift, grip_width_mm=40.0))
            self.assertFalse([call for call in planner.calls if call[0] in ("attach", "detach")])
            self.assertTrue(planner.carries_part)
            self.assertIs(arm.payload_model(), PayloadModel.PLANNER_AND_FILTER)

        with self.subTest("the jaws closed on a part nobody models"):
            arm, planner = _cell()
            lift = _standing_at(arm, planner, OFF)
            self.assertFalse(arm.attach_payload(40.0), "this cell models no carried part")
            self.assertTrue(arm._closed_on_part)
            handed: list[str] = []
            attach, detach = arm.attach_payload, arm.detach_payload
            arm.attach_payload = lambda width: handed.append("attach") or attach(width)
            arm.detach_payload = lambda: handed.append("detach") or detach()
            with arm.without_camera_world(_DECLINED):
                # Judged as an empty hand is (the owner, 2026-10-05): it runs beside the bin.
                self.assertIsNone(arm.carried_line_refusal(lift, grip_width_mm=40.0))
            self.assertEqual(handed, [], "nothing handed over or taken back")
            self.assertTrue(arm._closed_on_part, "the arm still knows its jaws closed on a part")

        with self.subTest("a part only the planner still holds, its detach never confirmed"):
            arm, planner = _modelled_cell()
            lift = _standing_at(arm, planner, OFF)
            planner.carries_part = True
            planner.calls.clear()
            with arm.without_camera_world(_DECLINED):
                self.assertIsNotNone(arm.carried_line_refusal(lift, grip_width_mm=40.0))
            self.assertFalse([call for call in planner.calls if call[0] in ("attach", "detach")])
            self.assertTrue(planner.carries_part, "the planner still holds its part")
            self.assertIsNone(arm._attached_payload)
            self.assertFalse(arm._closed_on_part)

    def test_a_decline_handed_to_the_verb_is_the_decline_its_move_would_carry(self) -> None:
        """⭐ ``carried_line_refusal(..., camera_world=CameraWorldDecline(...))`` answers what ``move(pose, linear=True,
        camera_world=...)`` answers. Red before: the verb took no decline, so with no live world wired every judgement
        read the camera world MISSING and refused (verifier W, 2026-10-01)."""
        decline = CameraWorldDecline(_DECLINED)
        arm, planner = _modelled_cell()
        self.assertIsNone(arm.carried_line_refusal(_standing_at(arm, planner, AWAY), grip_width_mm=40.0,
                                                   camera_world=decline))
        refused = arm.carried_line_refusal(_standing_at(arm, planner, OFF), grip_width_mm=40.0, camera_world=decline)
        assert refused is not None
        self.assertIn("a part is carried", refused.message or "")
        arm._motion.move_to.assert_not_called()

        arm._live_world = MagicMock()  # a live world wired: a declined motion is refused, as move refuses it
        planner.calls.clear()
        refused = arm.carried_line_refusal(_standing_at(arm, planner, AWAY), grip_width_mm=40.0, camera_world=decline)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.UNSUPPORTED)
        self.assertFalse(planner.named("check"), "nothing was judged")
        self.assertFalse(arm._closed_on_part)

    def test_a_lift_with_no_camera_world_is_refused_as_its_move_would_be(self) -> None:
        """The verb answers what ``move(pose, linear=True)`` answers: on cuRobo with no live world wired and nothing
        declined, the camera world's own refusal, before anything is handed over or judged."""
        arm, planner = _cell()
        lift = _standing_at(arm, planner, OFF)
        refused = arm.carried_line_refusal(lift, grip_width_mm=40.0)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.UNSUPPORTED)
        self.assertFalse(planner.named("check"), "nothing was judged")
        self.assertFalse(arm._closed_on_part)

    def test_a_pose_not_in_base_is_refused_before_anything_is_handed_over(self) -> None:
        arm, planner = _modelled_cell()
        lift = _standing_at(arm, planner, OFF)
        tool = Pose(position_mm=lift.position_mm, quaternion_xyzw=lift.quaternion_xyzw, frame=Frame.TOOL)
        with arm.without_camera_world(_DECLINED):
            refused = arm.carried_line_refusal(tool, grip_width_mm=40.0)
        assert refused is not None
        self.assertIs(refused.status, MotionStatus.INVALID_TARGET)
        self.assertFalse([call for call in planner.calls if call[0] == "attach"])

    def test_the_policy_on_the_owners_arm_beside_the_bin_backs_out_without_closing(self) -> None:
        """⭐ The whole grasp on the UR driver: the line down to the pose beside the bin runs empty-handed, the lift
        judged carrying is refused, the jaws stay open, and the line back up runs, one moveL each way: on a cell that
        models the carried part (Option 1)."""
        from src.geometry.quaternion import to_rotation_matrix
        from tests.test_what_the_hand_verbs_read import running_normally

        arm, planner = _modelled_cell()
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

        self.assertIs(report.outcome, getattr(PolicyOutcome, "CARRIED_RETREAT_REFUSED", "missing"),
                      report.motion_message)
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(arm._motion.move_to.call_count, 2, "one moveL down to the part, one back up")
        self.assertIn("a part is carried", report.motion_message)
        self.assertIs(arm.payload_model(), PayloadModel.NONE)
        self.assertTrue(np.allclose(report.waypoints[-1].position_mm,
                                    np.asarray(part.position_mm) - turned[:, 2] * 80.0, atol=1e-6),
                        "back at the standoff")
        self.assertTrue(math.isfinite(float(report.waypoints[-1].position_mm[2])))


def _ik_arm() -> Any:
    """The owner-like UR10 on the ``ik`` planner, as the URSim profile runs it: the controller draws a line and only its
    end is gated. It stands at (-30, -60, 80, -95, -90, 0) deg, and every pose solves to there."""
    from src.config.schema.robot import RobotConfig
    from src.robot.drivers.ur.arm import URRobotArm

    from tests._plan_end import OPEN_WORKSPACE

    arm: Any = URRobotArm(RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur10", "motion_planner": "ik"},
        "workspace_limits": {**OPEN_WORKSPACE, "z_max": 1000.0},
        "gripper": {"model": "robotiq_hande", "coupling_plates": [{"name": "adapter", "thickness_mm": 20.0}],
                    "tool_frame": {"source": "willy", "offset_mm": [0.0, 0.0, 155.75],
                                   "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "safety": {"payload": {"enforce": False}, "ik_quality": {"enforce": False},
                   "self_collision": {"backend": "fcl", "min_distance_mm": 10.0, "kinematics_model": "ur10",
                                      "planner_margin_mm": 4.0}},
    }))
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

        self.assertIsNone(arm.carried_line_refusal(self._at(400.0, "the lift's top"), grip_width_mm=40.0))

        self.assertIs(arm._preflight._last_target_pose, part, "a lift only judged is no place the arm was sent")
        self.assertFalse(arm._closed_on_part)

    def test_a_lift_it_would_refuse_is_refused_by_the_guard_the_move_meets(self) -> None:
        _needs_the_engine()
        arm = _ik_arm()
        part = self._at(300.0, "the part")
        arm._gate_pose(part, MotionCommand.MOVE_TO)

        refused = arm.carried_line_refusal(self._at(1400.0, "a top past the workspace"), grip_width_mm=40.0)

        assert refused is not None
        self.assertIs(refused.status, MotionStatus.WORKSPACE_REJECTED)
        self.assertIs(arm._preflight._last_target_pose, part)


if __name__ == "__main__":
    unittest.main()
