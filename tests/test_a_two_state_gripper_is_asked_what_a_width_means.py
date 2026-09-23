"""A gripper with two states is never handed a width it could read the wrong way round (owner cell, 2026-09-23).

jaw_io reads every commanded width as open or closed: closed at or below ``closed_below_mm``. A pick grasps at the
part's width less the squeeze and pre-opens to ``max_width_mm``, a release opens to ``max_width_mm``, and nothing in
those numbers says which state they mean on such a gripper. With the schema's default of 5 mm a grasp of any part wider
than 6 mm is an OPEN: the pick drives down, "grasps" by opening, lifts and reports EXECUTED, since a gripper with no
sensor measures nothing.

So the hand verbs tell a gripper that takes it (``OpensAndCloses``, which jaw_io is) what they mean, open or close,
and no width reaches it: every number of ``closed_below_mm`` picks and places the same. A two-state gripper that takes
only widths is still asked: a grasp that would open and a release that would close are refused by name, and a pick and
a place ask both of their widths before the arm moves at all.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core.gripper import HoldEvidence, OpensAndCloses, TwoStateGripper
from src.robot.execution import handling
from src.robot.execution.robot import Robot
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_jaw_io_gripper import CLOSE_PIN, FakeIO
from tests.test_robot_pick_and_place import _Log, _pose, _RecordingArm

_BENCH = "unit double: a bench with no cameras"


class _WidthsOnly:
    """The same toggle, seen through a gripper that takes widths alone: no ``set_closed`` to say what a verb means."""

    def __init__(self, inner: JawIOGripper) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "set_closed":
            raise AttributeError(name)
        return getattr(self._inner, name)


def _robot(*, widths_only: bool = False, **jaw: float) -> tuple[Robot, _Log, FakeIO]:
    log = _Log()
    arm = _RecordingArm(log)
    arm.connect()
    io = FakeIO()
    widths = {"min_width_mm": 5.0, "max_width_mm": 49.99, "closed_below_mm": 49.0, **jaw}
    gripper = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=0.0,
                           close_settle_s=0.0, sleep=lambda _s: None, **widths)
    gripper.connect()
    hand: Any = _WidthsOnly(gripper) if widths_only else gripper
    return Robot.from_parts(arm=arm, gripper=hand, lock_key=None), log, io


def _moves(log: _Log) -> list[object]:
    return [entry for entry in log.entries if entry[0] == "move"]


def _pulses(io: FakeIO) -> int:
    return sum(1 for _pin, high, _port in io.writes if high)


class JawIOIsToldWhatAVerbMeansTests(unittest.TestCase):
    def test_the_double_sees_what_it_should(self) -> None:
        robot, _log, _io = _robot()
        self.assertIsInstance(robot.gripper, OpensAndCloses)
        widths_only, _log, _io = _robot(widths_only=True)
        self.assertNotIsInstance(widths_only.gripper, OpensAndCloses)
        self.assertIsInstance(widths_only.gripper, TwoStateGripper)

    def test_a_pick_closes_at_the_part_whatever_closed_below_mm_says(self) -> None:
        """⛔ The owner's cell: the default 5 mm made the grasp at 39 mm an open, and the pick reported EXECUTED."""
        for below in (5.0, 20.0, 49.0):
            with self.subTest(closed_below_mm=below):
                robot, log, io = _robot(closed_below_mm=below)
                picked = robot.pick(_pose(), 40.0, decline=_BENCH)
                self.assertIs(handling.HandlingOutcome.EXECUTED, picked.outcome, picked.render())
                self.assertEqual(1, _pulses(io), "the pre-open finds the jaws open; the grasp is the one flip")
                self.assertTrue(robot.gripper.jaws_closed)
                self.assertEqual(3, len(_moves(log)))

    def test_a_pre_open_below_the_threshold_still_opens(self) -> None:
        """A pre-open width at or below closed_below_mm used to be a close before the part."""
        robot, _log, io = _robot()
        picked = robot.pick(_pose(), 40.0, pre_open_mm=20.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, picked.outcome, picked.render())
        self.assertEqual(1, _pulses(io), "no pulse at the pre-open, one at the part")

    def test_the_owners_numbers_pick_and_place_with_one_pulse_each(self) -> None:
        """⭐ THE CONTROL: each verb flips the jaws once."""
        robot, _log, io = _robot()
        picked = robot.pick(_pose(), 40.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, picked.outcome, picked.render())
        self.assertEqual(1, _pulses(io), "the pick's pulses")
        placed = robot.place(_pose(x=300.0), decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, placed.outcome, placed.render())
        self.assertEqual(2, _pulses(io), "the place's pulse")
        self.assertFalse(robot.gripper.jaws_closed)

    def test_a_release_opens_whatever_the_hands_width_reads_as(self) -> None:
        robot, _log, io = _robot(max_width_mm=40.0)            # a release to 40 mm, at or below 49
        robot.gripper._closed = True                           # a part in the jaws
        placed = robot.place(_pose(), decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.EXECUTED, placed.outcome, placed.render())
        self.assertEqual(1, _pulses(io))
        self.assertFalse(robot.gripper.jaws_closed)

    def test_grasp_and_release_say_close_and_open(self) -> None:
        robot, _log, io = _robot()
        grasped = robot.grasp(60.0)                            # above closed_below_mm: once an open
        self.assertIs(handling.HandOutcome.GRASPED, grasped.outcome, grasped.render())
        self.assertIs(HoldEvidence.UNMEASURED, grasped.hold)
        self.assertEqual(1, _pulses(io))
        released = robot.release()
        self.assertIs(handling.HandOutcome.RELEASED, released.outcome, released.render())
        self.assertEqual(2, _pulses(io))


class AGripperThatTakesOnlyWidthsIsAskedTests(unittest.TestCase):
    def test_a_grasp_width_that_means_open_is_refused_before_any_motion(self) -> None:
        robot, log, io = _robot(widths_only=True, closed_below_mm=5.0)   # the schema default, and a 40 mm part
        report = robot.pick(_pose(), 40.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
        for part in ("39.0", "closed_below_mm", "5.0", "open"):
            self.assertIn(part, report.message)
        self.assertEqual([], _moves(log))
        self.assertEqual([], io.writes)

    def test_a_pre_open_that_means_close_is_refused_before_any_motion(self) -> None:
        robot, log, io = _robot(widths_only=True)
        report = robot.pick(_pose(), 40.0, pre_open_mm=20.0, decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
        for part in ("20.0", "closed_below_mm", "close"):
            self.assertIn(part, report.message)
        self.assertEqual([], _moves(log))
        self.assertEqual([], io.writes)

    def test_a_release_width_that_means_close_is_refused_before_any_motion(self) -> None:
        robot, log, io = _robot(widths_only=True, max_width_mm=40.0)
        robot.gripper._closed = True
        report = robot.place(_pose(), decline=_BENCH)
        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
        for part in ("40.0", "closed_below_mm", "close"):
            self.assertIn(part, report.message)
        self.assertEqual([], _moves(log))
        self.assertEqual([], io.writes)

    def test_a_grasp_that_would_open_is_refused(self) -> None:
        robot, _log, io = _robot(widths_only=True)
        report = robot.grasp(60.0)
        self.assertIs(handling.HandOutcome.REFUSED, report.outcome)
        self.assertIn("closed_below_mm", report.error)
        self.assertEqual([], io.writes)

    def test_a_release_that_would_close_is_refused(self) -> None:
        robot, _log, io = _robot(widths_only=True, max_width_mm=40.0)
        robot.gripper._closed = True
        report = robot.release()
        self.assertIs(handling.HandOutcome.REFUSED, report.outcome)
        self.assertIn("closed_below_mm", report.error)
        self.assertEqual([], io.writes)

    def test_a_grasp_at_the_threshold_closes(self) -> None:
        """The control at the edge: at or below closes, as the driver reads it."""
        robot, _log, io = _robot(widths_only=True)
        report = robot.grasp(49.0)
        self.assertIs(handling.HandOutcome.GRASPED, report.outcome, report.render())
        self.assertIs(HoldEvidence.UNMEASURED, report.hold)
        self.assertEqual(1, _pulses(io))


if __name__ == "__main__":
    unittest.main()
