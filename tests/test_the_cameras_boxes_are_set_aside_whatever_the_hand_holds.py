"""The camera's boxes are set aside whatever the hand holds, and what the hand says is read where it still matters.

"Am Zustand der Hand", the owner's answer to verifier W's L2 (2026-10-01): Option 1 asks the planner once more with
every box the camera saw set aside, where only those boxes refuse its world, and lets the exact guard decide them, and a
part in the jaws the planner modelled was the planner's alone, so the set-aside was asked only of a hand known empty
and open. A cell that modelled no carried part, the owner's, judged jaws closed on one as an empty hand is judged and
read no hand (the owner, 2026-10-05: "wir nehmen das, wo wir uns sicherer sind, dass wir mehr erhalten ... Also mehr
Griffe"). No cell models a carried part since 2026-10-17, so every cell does what the owner's did: the boxes are set
aside whatever the hand holds, and the exact guard still judges the arm and the hand at full stroke.

What the hand says (``core.gripper.why_not_known_open``) is still read where it matters: a blocker's pick backs out only
a hand known empty and open, the console asks it of a hand it opened, and the arm runs what it judged ahead only while
the hand reads as it did then. Known open are:

* a hand that toggles with no sensor (the owner's Hand-E on tool DO0): connected, its count standing, and the count
  says open;
* a gripper that measures its width: connected, within 2 mm of its widest, and no part measured held.

The arm learns its hand where a cell comes up (``connect_cell``), once the hand connected. The pose screen judges a pose
and not a hand, and says what runs there whatever the hand holds.

The arm, the camera world and the bin are those of ``test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards.py``:
a bin turned 30 degrees, 30 mm from the UR10's meshes, which only the camera's boxes refuse on the planner's world. The
toggle is the real ``JawIOGripper`` on a recording tool I/O, so its count is the driver's own.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from src.robot.core import JointPositions
from src.robot.core.gripper import HoldEvidence, why_not_known_open
from src.robot.safety.planning.band import PoseVerdict
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import (
    OFF,
    OFF_ON,
    _cell,
    _from,
    _line,
)
from tests.test_a_stopped_controller_moves_no_jaws import _IO, _always_open
from tests.test_what_the_hand_verbs_read import running_normally


def _toggle(*, closed: bool = False) -> Any:
    """The owner's hand: a Hand-E on the I/O coupling, single_toggle on tool DO0, no feedback, 49.99 / 5.0 mm,
    connected on jaws a person said stand open, and closed once where ``closed``."""
    from src.robot.grippers.jaw_io import JawIOGripper

    jaws = JawIOGripper(_IO([]), actuation="single_toggle", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                        min_width_mm=5.0, max_width_mm=49.99, ask=_always_open, sleep=lambda _s: None)
    jaws.connect()
    if closed:
        jaws.set_closed(True)
    return jaws


class _WidthHand:
    """A gripper that measures its width, as a Robotiq on its socket does: ``width_mm`` of ``widest`` open."""

    min_width_mm = 0.0

    def __init__(self, width_mm: float = 50.0, *, widest: float = 50.0, connected: bool = True,
                 hold: HoldEvidence = HoldEvidence.UNMEASURED, fails: bool = False) -> None:
        self.width_mm = width_mm
        self.max_width_mm = widest
        self.is_connected = connected
        self.hold = hold
        self.fails = fails
        self.reads = 0

    def width_is_measured(self) -> bool:
        return True

    def get_width_mm(self) -> float:
        self.reads += 1
        if self.fails:
            raise ConnectionError("the gripper's socket did not answer")
        return self.width_mm

    def hold_evidence(self) -> HoldEvidence:
        return self.hold

    def set_width_mm(self, width_mm: float, *, speed: "float | None" = None, force: "float | None" = None) -> None:
        self.width_mm = float(width_mm)


class _OtherHand:
    """A hand that neither counts its changes nor measures its width: a cup, a jaw on a solenoid with no switch."""

    is_connected = True
    min_width_mm = 0.0
    max_width_mm = 50.0

    def set_width_mm(self, width_mm: float, *, speed: "float | None" = None, force: "float | None" = None) -> None:
        return None

    def get_width_mm(self) -> float:
        return 50.0


# ---------------------------------------------------------------------------------------------------------------------
# What the hand says
# ---------------------------------------------------------------------------------------------------------------------


class WhatTheHandSaysTests(unittest.TestCase):
    def test_a_toggle_whose_count_stands_open_is_known_open(self) -> None:
        self.assertEqual(why_not_known_open(_toggle()), "")

    def test_a_toggle_whose_count_says_closed_is_not(self) -> None:
        said = why_not_known_open(_toggle(closed=True))
        self.assertIn("count says the jaws stand closed", said)

    def test_a_toggle_nobody_can_vouch_for_is_not(self) -> None:
        """Switched by hand at the pendant since the program's last change, not connected, a change that failed."""
        switched = _toggle()
        switched._io.do[0] = not switched._io.do[0]  # DO0 switched on the pendant: the jaws moved, the count did not
        failed = _toggle()
        failed._edge_unknown = True
        unconnected = _toggle()
        unconnected.disconnect()
        for name, hand in (("switched by hand", switched), ("a change that failed", failed),
                           ("not connected", unconnected)):
            with self.subTest(name):
                said = why_not_known_open(hand)
                self.assertTrue(said)
                self.assertIn("nobody can say where its jaws stand", said)

    def test_a_gripper_that_measures_its_width_is_known_open_at_its_widest(self) -> None:
        for width in (50.0, 48.5):
            with self.subTest(width=width):
                self.assertEqual(why_not_known_open(_WidthHand(width)), "")

    def test_a_width_short_of_open_or_one_nobody_can_read_is_not(self) -> None:
        for name, hand, words in (
                ("closed on a part", _WidthHand(31.0), "31 mm"),
                ("just short of open", _WidthHand(47.9), "not open"),
                ("no number", _WidthHand(float("nan")), "not open"),
                ("not connected", _WidthHand(connected=False), "not connected"),
                ("a read that fails", _WidthHand(fails=True), "could not be read"),
                ("a part measured at its widest", _WidthHand(hold=HoldEvidence.HELD), "measures a part held")):
            with self.subTest(name):
                self.assertIn(words, why_not_known_open(hand))

    def test_no_hand_and_any_other_hand_are_not(self) -> None:
        self.assertIn("no hand", why_not_known_open(None))
        self.assertIn("_OtherHand neither counts its changes nor measures its width", why_not_known_open(_OtherHand()))

    def test_reading_the_hand_sends_nothing(self) -> None:
        """A read-only question: no change of DO0, and nothing asked of a person."""
        for hand in (_toggle(), _toggle(closed=True)):
            with self.subTest(closed=hand.jaws_closed):
                writes, sent = list(hand._io.events), hand.commands_sent
                why_not_known_open(hand)
                self.assertEqual((hand._io.events, hand.commands_sent), (writes, sent))


# ---------------------------------------------------------------------------------------------------------------------
# What the arm does with it
# ---------------------------------------------------------------------------------------------------------------------


def _beside_the_bin(hand: Any) -> "tuple[Any, Any]":
    """The arm at OFF beside the bin 30 mm away, the camera's boxes held by both, ``hand`` the hand it carries."""
    arm, planner = _cell()
    arm.set_hand(hand)
    _from(arm, planner, OFF)
    return arm, planner


class EveryHandSetsTheBoxesAsideTests(unittest.TestCase):
    def test_a_hand_known_open_sets_them_aside_and_the_line_runs(self) -> None:
        for name, hand in (("a toggle whose count stands open", _toggle()),
                           ("a gripper measured fully open", _WidthHand(49.2))):
            with self.subTest(name):
                arm, planner = _beside_the_bin(hand)
                result = _line(arm, OFF_ON)
                self.assertTrue(result.ok, result.message)
                arm._conn.moveJ.assert_called_once()
                self.assertEqual(len(planner.set_aside()), 1)

    def test_every_other_hand_sets_them_aside_too_and_none_is_read(self) -> None:
        """⭐ Whatever the hand says, the boxes are set aside and the exact guard decides: it holds the hand at full
        stroke, the widest it stands, and no part in the jaws is modelled (the owner, 2026-10-05; every cell since
        2026-10-17). Red on a cell that modelled the carried part: every one of these kept the boxes in and moved
        nothing. A line never reads the hand."""
        switched = _toggle()
        switched._io.do[0] = not switched._io.do[0]
        short = _WidthHand(31.0)
        for name, hand in (("no hand", None), ("the toggle's count says closed", _toggle(closed=True)),
                           ("a toggle switched by hand", switched), ("a width short of open", short),
                           ("a width nobody can read", _WidthHand(fails=True)), ("any other hand", _OtherHand())):
            with self.subTest(name):
                arm, planner = _beside_the_bin(hand)
                result = _line(arm, OFF_ON)
                self.assertTrue(result.ok, result.message)
                self.assertEqual(len(planner.set_aside()), 1)
        self.assertEqual(short.reads, 0, "the hand is not read")

    def test_a_part_the_jaws_closed_on_is_judged_as_an_empty_hand(self) -> None:
        """⭐ THE L2 CASE, as the owner decided it on 2026-10-05. ``Robot.grasp`` closes the jaws; until then the hand's
        count kept the camera's boxes in and the line out of the pose beside the bin was refused, part in the jaws; the
        owner's carried lifts died there. Now the closed hand is judged as an empty one: the boxes set aside, the exact
        guard decides, and the line runs. Released, it runs as before."""
        from src.robot.execution import handling

        jaws = _toggle()
        arm, planner = _beside_the_bin(jaws)
        running_normally(arm._conn)
        robot = SimpleNamespace(arm=arm, gripper=jaws)
        held = handling.grasp(robot, 39.0)
        self.assertTrue(held.ok, held.render())
        self.assertTrue(jaws.jaws_closed)
        result = _line(arm, OFF_ON)
        self.assertTrue(result.ok, result.message)
        arm._conn.moveJ.assert_called_once()
        self.assertEqual(len(planner.set_aside()), 1)

        released = handling.release(robot)
        self.assertTrue(released.ok, released.render())
        self.assertTrue(_line(arm, OFF_ON).ok, "opened, the same line runs")


class TheScreenJudgesThePoseNotTheHandTests(unittest.TestCase):
    def test_the_screen_says_beside_the_boxes_whatever_hand_the_arm_holds(self) -> None:
        """At a desk the hand may not be connected at all: the screen says what runs there whatever the hand holds, and
        asks the planner as a move would."""
        for name, hand in (("no hand", None), ("the count says closed", _toggle(closed=True))):
            with self.subTest(name):
                arm, planner = _cell()
                arm.set_hand(hand)
                screen = arm.screen_configuration(JointPositions(tuple(OFF)))
                self.assertIs(screen.verdict, PoseVerdict.SEEN_BOXES, screen.detail)
                self.assertIn("whatever the hand holds", screen.detail)
                self.assertEqual(len(planner.set_aside()), 1)


# ---------------------------------------------------------------------------------------------------------------------
# Where the arm learns its hand
# ---------------------------------------------------------------------------------------------------------------------


class _Arm:
    def __init__(self, log: "list[Any]", *, takes_a_hand: bool = True) -> None:
        self.log = log
        if takes_a_hand:
            self.set_hand = lambda gripper: self.log.append(("set_hand", gripper))

    def connect(self) -> None:
        self.log.append("arm.connect")

    def disconnect(self) -> None:
        self.log.append("arm.disconnect")


class _Gripper:
    def __init__(self, log: "list[Any]", *, refuses: bool = False) -> None:
        self.log = log
        self.refuses = refuses

    def connect(self) -> None:
        self.log.append("gripper.connect")
        if self.refuses:
            raise RuntimeError("the hand would not come up")

    def disconnect(self) -> None:
        self.log.append("gripper.disconnect")


class ACellComingUpHandsTheArmItsHandTests(unittest.TestCase):
    def test_the_hand_is_handed_over_once_it_connected(self) -> None:
        """⭐ Red before: nothing told the arm which hand it carries."""
        from src.robot.execution.lifecycle import connect_cell

        log: list[Any] = []
        gripper = _Gripper(log)
        connect_cell(_Arm(log), gripper)
        self.assertEqual(log, ["arm.connect", "gripper.connect", ("set_hand", gripper)])

    def test_an_arm_alone_is_told_it_carries_no_hand(self) -> None:
        from src.robot.execution.lifecycle import connect_cell

        log: list[Any] = []
        connect_cell(_Arm(log), None)
        self.assertEqual(log, ["arm.connect", ("set_hand", None)])

    def test_a_hand_that_does_not_come_up_is_never_handed_over(self) -> None:
        from src.robot.execution.lifecycle import connect_cell

        log: list[Any] = []
        with self.assertRaises(RuntimeError):
            connect_cell(_Arm(log), _Gripper(log, refuses=True))
        self.assertEqual(log, ["arm.connect", "gripper.connect", "arm.disconnect"])

    def test_an_arm_that_takes_no_hand_comes_up_as_before(self) -> None:
        from src.robot.execution.lifecycle import connect_cell

        log: list[Any] = []
        connect_cell(_Arm(log, takes_a_hand=False), _Gripper(log))
        self.assertEqual(log, ["arm.connect", "gripper.connect"])

    def test_robot_connected_hands_the_ur_arm_its_hand(self) -> None:
        """Through ``Robot.connected()``: the UR arm reads the very gripper the robot connected."""
        from src.robot.execution.robot import Robot

        arm, _ = _cell()
        arm.set_hand(None)
        jaws = _toggle()
        jaws.disconnect()
        arm.connect = lambda: None  # type: ignore[method-assign]  # the controller is the test's double
        arm.disconnect = lambda: None  # type: ignore[method-assign]
        robot = Robot.from_parts(arm=arm, gripper=jaws, lock_key=None)
        with robot.connected():
            self.assertIs(arm._hand, jaws)
            arm._conn.moveJ.reset_mock()
            _from(arm, arm._curobo, OFF)
            self.assertTrue(_line(arm, OFF_ON).ok, "the line beside the bin runs, the boxes set aside")


if __name__ == "__main__":
    unittest.main()
