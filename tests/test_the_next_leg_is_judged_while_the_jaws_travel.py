"""The next leg is judged while the jaws travel, and nothing is sent before their stroke is over (map4 motion C7a).

The owner's Hand-E on tool DO0 takes its configured 2.0 s to close and to open, and nothing reads it: the program waits
that long after the one change. With ``robot.motion.judge_next_leg`` the verb that waits judges its next leg meanwhile:

* at the part, the grasp's close leaves its stroke to the policy: the change goes out once, the part is attached, the
  joint move a task declared after the pick (the carry) is judged from where the lift will end, in the world held at
  the part, and the lift is sent only once the stroke is waited out;
* at a place, the release leaves its stroke to the verb: the change goes out once, the part is forgotten, and in the
  world held across the drop (``safety.planning_world.hold.drop``) the line out is judged where the arm stands and the
  joint move declared after the place (the return) from where the line out ends; the line out is sent after the stroke;
* where no stroke is left to the verb, the leg is left to the line before it, which judges it during motion where the
  arm can (``in_settles_and_motion``);
* off, as shipped, every close and release waits out its stroke as it always did, and nothing is judged ahead.

Pinned on the policy and the place verb over a recording arm and the real ``JawIOGripper`` toggle on a recorded I/O
timeline, one list for both, so the order is the order things happened in. Honesty bucket (3): stand-ins for the arm
and the controller's I/O; the arm's own judgements are pinned in
``test_a_leg_judged_ahead_runs_where_nothing_it_was_judged_on_changed``.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.contracts import UNSET
from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult
from src.robot.core.arm_capabilities import LineMotion, LineReading, PayloadModel
from src.robot.execution import handling
from src.robot.execution.handling import HandlingOutcome
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from tests.test_jaw_io_gripper import CLOSE_PIN, HIGH, LOW, SETTLE_S, _Timeline, _toggle


class _JudgingArm:
    """An arm that keeps its lines, holds its world and judges the next leg, each step written onto ``events``."""

    #: A route the place verb reads as PLANNED (``motion.route_of``): cuRobo plans it and the guard judges its lines.
    plans_paths = True

    def __init__(self, events: "list[tuple[str, object]]", *, judges: bool = True,
                 holds: "tuple[str, ...]" = ("standoff", "carry", "drop")) -> None:
        self.events = events
        self.judges = judges
        self.holds = holds
        self.is_connected = True

    @property
    def judges_next_legs(self) -> bool:
        return self.judges

    def line_motion(self) -> LineReading:
        return LineReading(LineMotion.CHECKED, "judged")

    @contextmanager
    def held_world(self, reason: str) -> Iterator[None]:
        self.events.append(("held", reason))
        try:
            yield
        finally:
            self.events.append(("let go", reason))

    def holds_world_at(self, junction: str) -> bool:
        return junction in self.holds

    def move(self, pose: Pose, *, linear: bool = False, **_said: Any) -> MotionResult:
        self.events.append(("move", f"{pose.label}{' line' if linear else ''}"))
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def carried_line_refusal(self, pose: Pose, *, grip_width_mm: float, camera_world: Any = UNSET) -> None:
        self.events.append(("lift judged carrying", pose.label))

    def attach_payload(self, grip_width_mm: float) -> bool:
        self.events.append(("attach", round(float(grip_width_mm), 3)))
        return False

    def detach_payload(self) -> bool:
        self.events.append(("detach", None))
        return True

    def payload_declined_reason(self) -> str | None:
        return "this stand-in models no carried part"

    def payload_model(self) -> PayloadModel:
        return PayloadModel.NONE

    def judge_the_next_leg(self, after: Pose, *, junction: str, now: bool = True) -> str:
        self.events.append(("next leg judged", (after.label, junction, now)))
        return ""

    def judge_line_ahead(self, pose: Pose) -> None:
        self.events.append(("line judged ahead", pose.label))


class _PlainHand:
    """A hand that opens and closes by intent and waits out its own stroke: it leaves nothing to the verb."""

    is_connected = True
    min_width_mm, max_width_mm = 0.0, 50.0

    def __init__(self, events: "list[tuple[str, object]]") -> None:
        self.events = events

    def set_closed(self, closed: bool) -> None:
        self.events.append(("jaws", "closed" if closed else "open"))

    def set_width_mm(self, width_mm: float, *, speed: Any = None, force: Any = None) -> None:
        raise AssertionError("a hand told open and close by intent is never sent a width")

    def get_width_mm(self) -> float:
        return 0.0


def _grasp() -> GraspPoint:
    return GraspPoint(position=np.array([350.0, -150.0, 120.0]), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE,
                      label="part")


def _changes(events: "list[tuple[str, object]]") -> "list[tuple[str, object]]":
    return [event for event in events if event[0] == f"out{CLOSE_PIN}"]


def _at(events: "list[tuple[str, object]]", name: str) -> int:
    return next(i for i, event in enumerate(events) if event[0] == name)


def _toggle_hand(io: _Timeline, *, closed: bool = False) -> Any:
    hand = _toggle(io, close_settle_s=SETTLE_S)
    hand.connect()
    if closed:
        hand.set_closed(True)
    io.events.clear()
    return hand


class AtThePartTheCarryIsJudgedWhileTheJawsCloseTests(unittest.TestCase):
    def test_the_change_goes_out_once_and_the_lift_waits_for_the_stroke(self) -> None:
        """⭐ Red before: the 2.0 s stroke was slept inside the close, and the carry was judged at the top of the lift."""
        io = _Timeline()
        hand = _toggle_hand(io)
        arm = _JudgingArm(io.events)
        report = GraspExecutionPolicy(arm=arm, gripper=hand).execute(_grasp())  # type: ignore[arg-type]
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        events = io.events
        self.assertEqual([HIGH], _changes(events), "the close is exactly one change of DO0")
        change, attach = events.index(HIGH), _at(events, "attach")
        judged = events.index(("next leg judged", ("retreat", "carry", True)))
        slept = _at(events, "sleep")
        lift = events.index(("move", "retreat line"))
        self.assertLess(change, attach)
        self.assertLess(attach, judged, "the carry is judged with the part attached")
        self.assertLess(judged, slept, "the carry is judged while the jaws close, not after")
        self.assertLess(slept, lift, "the lift went out before the stroke was waited out")
        stroke = float(events[slept][1])  # type: ignore[arg-type]
        self.assertGreater(stroke, 0.0)
        self.assertLessEqual(stroke, SETTLE_S + 1e-6)

    def test_off_the_close_waits_out_its_stroke_as_ever_and_nothing_is_judged_ahead(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io)
        arm = _JudgingArm(io.events, judges=False)
        report = GraspExecutionPolicy(arm=arm, gripper=hand).execute(_grasp())  # type: ignore[arg-type]
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        events = io.events
        self.assertEqual([HIGH, ("sleep", SETTLE_S)], events[events.index(HIGH):events.index(HIGH) + 2])
        self.assertFalse([event for event in events if event[0] == "next leg judged"])

    def test_a_hand_that_leaves_no_stroke_leaves_the_carry_to_the_lift(self) -> None:
        events: "list[tuple[str, object]]" = []
        arm = _JudgingArm(events)
        report = GraspExecutionPolicy(arm=arm, gripper=_PlainHand(events)).execute(_grasp())  # type: ignore[arg-type]
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        judged = events.index(("next leg judged", ("retreat", "carry", False)))
        self.assertLess(events.index(("jaws", "closed")), judged)
        self.assertLess(judged, events.index(("move", "retreat line")))

    def test_the_carry_is_judged_inside_the_world_held_at_the_part(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io)
        arm = _JudgingArm(io.events)
        GraspExecutionPolicy(arm=arm, gripper=hand).execute(_grasp())  # type: ignore[arg-type]
        events = io.events
        held = next(i for i, e in enumerate(events) if e[0] == "held")
        let_go = next(i for i, e in enumerate(events) if e[0] == "let go")
        judged = _at(events, "next leg judged")
        self.assertLess(held, judged)
        self.assertLess(judged, let_go)


def _drop() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 150.0]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                frame=Frame.BASE, label="drop")


def _place(arm: _JudgingArm, hand: Any) -> Any:
    return handling.place(SimpleNamespace(arm=arm, gripper=hand), _drop(), standoff_mm=80.0)


class AtAPlaceTheLineOutIsJudgedWhileTheJawsOpenTests(unittest.TestCase):
    def test_the_line_out_and_the_return_are_judged_while_the_jaws_open(self) -> None:
        """⭐ Red before: the line out took its frame after a 2.0 s stroke and was judged then, the return after it."""
        io = _Timeline()
        hand = _toggle_hand(io, closed=True)
        arm = _JudgingArm(io.events)
        placed = _place(arm, hand)
        self.assertIs(HandlingOutcome.EXECUTED, placed.outcome, placed.message)
        events = io.events
        self.assertEqual([LOW], _changes(events), "the release is exactly one change of DO0")
        change, detach = events.index(LOW), _at(events, "detach")
        line = events.index(("line judged ahead", "standoff"))
        judged = events.index(("next leg judged", ("standoff", "drop", True)))
        slept = _at(events, "sleep")
        line_out = events.index(("move", "standoff line"))
        self.assertLess(change, detach)
        self.assertLess(detach, line, "the line out is judged with the part forgotten")
        self.assertLess(line, judged, "the return is judged from where the line out ends")
        self.assertLess(judged, slept)
        self.assertLess(slept, line_out, "the line out went out before the jaws had opened")

    def test_the_release_and_the_line_out_are_inside_the_world_held_across_the_drop(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io, closed=True)
        arm = _JudgingArm(io.events)
        _place(arm, hand)
        events = io.events
        holds = [event[1] for event in events if event[0] == "held"]
        self.assertEqual(2, len(holds), holds)
        self.assertIn("line in", str(holds[0]))
        self.assertIn("line out", str(holds[1]))
        drop_held = [i for i, e in enumerate(events) if e[0] == "held"][1]
        drop_let_go = [i for i, e in enumerate(events) if e[0] == "let go"][1]
        self.assertLess(drop_held, events.index(LOW))
        self.assertLess(events.index(("move", "standoff line")), drop_let_go)

    def test_outside_a_world_held_across_the_drop_nothing_is_judged_ahead(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io, closed=True)
        arm = _JudgingArm(io.events, holds=())
        placed = _place(arm, hand)
        self.assertIs(HandlingOutcome.EXECUTED, placed.outcome, placed.message)
        self.assertFalse([event for event in io.events if event[0] in ("line judged ahead", "held")])
        self.assertEqual([LOW], _changes(io.events))

    def test_off_the_release_waits_out_its_stroke_as_ever(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io, closed=True)
        arm = _JudgingArm(io.events, judges=False, holds=())
        _place(arm, hand)
        events = io.events
        at = events.index(LOW)
        self.assertEqual([LOW, ("sleep", SETTLE_S)], events[at:at + 2])
        self.assertFalse([event for event in events if event[0] in ("line judged ahead", "next leg judged")])

    def test_a_hand_that_leaves_no_stroke_leaves_the_return_to_the_line_out(self) -> None:
        events: "list[tuple[str, object]]" = []
        arm = _JudgingArm(events)
        placed = _place(arm, _PlainHand(events))
        self.assertIs(HandlingOutcome.EXECUTED, placed.outcome, placed.message)
        judged = events.index(("next leg judged", ("standoff", "drop", False)))
        self.assertLess(events.index(("jaws", "open")), judged)
        self.assertLess(judged, events.index(("move", "standoff line")))
        self.assertNotIn(("line judged ahead", "standoff"), events)

    def test_a_line_in_refused_ends_the_place_with_the_jaws_untouched(self) -> None:
        io = _Timeline()
        hand = _toggle_hand(io, closed=True)
        arm = _JudgingArm(io.events)

        def refuse_the_line_in(pose: Pose, *, linear: bool = False, **_said: Any) -> MotionResult:
            if linear:
                from src.robot.core import MotionStatus

                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           target_pose=pose, message="lfinger|seen_00 2.9 < 3.0")
            return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

        arm.move = refuse_the_line_in  # type: ignore[method-assign]
        placed = _place(arm, hand)
        self.assertIs(HandlingOutcome.MOTION_REFUSED, placed.outcome)
        self.assertEqual([], _changes(io.events))


class ATaskDeclaresTheMoveAfterItsVerbTests(unittest.TestCase):
    """``motion.expecting_next``: the seam a task declares its carry and its return through."""

    class _Declaring:
        judges_next_legs = True
        home_joint_positions = (0.0, -1.5, 1.5, -1.5, -1.5, 0.0)

        def __init__(self) -> None:
            self.declared: list[tuple[float, ...]] = []

        @contextmanager
        def expecting_next(self, joints: Any) -> Iterator[None]:
            self.declared.append(tuple(joints.tolist()))
            yield

    def test_the_joints_and_home_reach_an_arm_that_judges_ahead(self) -> None:
        from src.robot.core import JointPositions
        from src.robot.execution.motion import expecting_next

        arm = self._Declaring()
        with expecting_next(arm, JointPositions((0.1, -1.4, 1.4, -1.6, -1.5, 0.2))):
            pass
        with expecting_next(arm, "home"):
            pass
        self.assertEqual([(0.1, -1.4, 1.4, -1.6, -1.5, 0.2), arm.home_joint_positions], arm.declared)

    def test_elsewhere_nothing_is_declared(self) -> None:
        from unittest.mock import MagicMock

        from src.robot.core import JointPositions
        from src.robot.execution.motion import expecting_next

        arm = self._Declaring()
        arm.judges_next_legs = False  # type: ignore[misc]
        with expecting_next(arm, JointPositions((0.0,) * 6)), expecting_next(self._Declaring(), None):
            pass
        double = MagicMock()
        with expecting_next(double, JointPositions((0.0,) * 6)):
            pass
        self.assertEqual([], arm.declared)
        double.expecting_next.assert_not_called()


if __name__ == "__main__":
    unittest.main()
