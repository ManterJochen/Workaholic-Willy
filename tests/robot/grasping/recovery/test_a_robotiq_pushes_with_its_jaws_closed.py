"""A hand pushes with its jaws closed: a Robotiq (the owner, 2026-10-15: "Ja, Robotiq geschlossen") and a toggle
(2026-10-16: "Ja, geschlossen").

The open Hand-E is 71.6 mm across its fingers' outer faces; on Thursday's cell the descent of that open hand beside a
part in a pile was refused on every side, so no push ever ran. Closed, the fingers are 22 mm across. What is pinned:

* the cell: a hand told a width (a Robotiq) plans its push with the fingers at its closed width, a toggle (a ``jaw_io``
  single_toggle, ``push_gate.toggles``) at the band it reports closed, ``min_width_mm``, and that slack the guard holds
  them with (``PushCell.from_robot_config``); any other hand plans open;
* the push (:func:`execute_push` with ``close_to_mm``): judged ahead with the guard holding the fingers closed, the hand
  closes in the air at P0 and before the down leg, the guard holds the fingers where the hand reads them for the four
  legs, and fully open again once they end; a Robotiq's jaws need not stand open, but it must measure its width and be
  connected; a toggle closes by one change of its output, from a count that stands and says open; a hand that does
  not close leaves the arm in the air at P0 with nothing touched (``refused_down_not_sent``);
* the pick: the hand opens again at the look, after the move back and before the look is taken again, a toggle by one
  more change of its output, and one that does not open ends the pick as a gripper fault with the arm at the look.
"""

from __future__ import annotations

import dataclasses
import unittest
from typing import Any, Callable
from unittest import mock

from src.robot.core import RobotError
from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
from src.robot.grasping.recovery.push_gate import PushCell
from src.robot.grasping.recovery.push_motion import APPROACH_LEG, CONTACT_LEGS, PushOutcomeCode as C
from tests.robot.grasping.recovery import test_a_boxed_in_part_is_pushed_inside_the_pick as boxed
from tests.robot.grasping.recovery.test_a_push_moves_only_when_every_guard_holds import _Cell, _UR

CLOSED = FINGER_OPENING_SLACK_MM  # the fingers closed to 0 mm, as the guard holds them


class _Robotiq:
    """A Robotiq as a push meets it: it stands at the width it is told and reads it back, and every command is noted
    (``note``, ``"open"`` for its widest) before it runs; ``fails`` names the command that raises (``"close"``,
    ``"open"``)."""

    min_width_mm = 0.0
    max_width_mm = 50.0
    closed_width_mm = 0.0

    def __init__(self, *, fails: str = "", note: Callable[[str, Any], None] | None = None) -> None:
        self.fails = fails
        self.note = note or (lambda kind, data: None)
        self.is_connected = True
        self.width_mm = self.max_width_mm

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def width_is_measured(self) -> bool:
        return True

    def get_width_mm(self) -> float:
        return self.width_mm

    def set_width_mm(self, width_mm: float, *, speed: float | None = None, force: float | None = None) -> None:
        opens = float(width_mm) >= self.max_width_mm
        self.note("gripper", "open" if opens else round(float(width_mm), 2))
        if self.fails == ("open" if opens else "close"):
            raise RobotError(f"the hand did not answer the {self.fails}")
        self.width_mm = float(width_mm)


class _Fingered(_UR):
    """The push's UR whose guard is told where the fingers stand, and which judges the push ahead, both noted."""

    def set_finger_opening(self, width_mm: float | None) -> None:
        self.log.write("fingers", None if width_mm is None else round(float(width_mm), 2))

    def lines_refusal_ahead(self, *, approach: Any, lines: Any) -> str:
        self.log.write("ahead", len(lines))
        return ""


class TheCellPlansAClosedPush(unittest.TestCase):
    def _cell(self, profile: str) -> PushCell:
        from src.config.loader import load_robot_config, reload_config  # noqa: PLC0415

        reload_config()
        self.addCleanup(reload_config)
        cell = PushCell.from_robot_config(load_robot_config(None, profile=profile))
        assert isinstance(cell, PushCell), cell
        return cell

    def test_a_robotiq_plans_its_fingers_closed_a_toggle_at_its_band_and_any_other_hand_open(self) -> None:
        robotiq = self._cell("ur10,hande")
        self.assertEqual(0.0, robotiq.closed_width_mm)
        self.assertEqual(CLOSED, robotiq.hand.open_width_mm)
        self.assertAlmostEqual(CLOSED / 2.0 + 10.80, robotiq.hand.half_outer_mm)
        with mock.patch("src.robot.grasping.calculator_factory.opens_as_needed", return_value=False), \
                mock.patch("src.robot.grasping.recovery.push_gate.toggles", return_value=True):
            toggle = self._cell("ur10,hande")
        self.assertEqual(5.0, toggle.closed_width_mm)  # the band the toggle reports closed: its min_width_mm
        self.assertEqual(5.0 + FINGER_OPENING_SLACK_MM, toggle.hand.open_width_mm)
        with mock.patch("src.robot.grasping.calculator_factory.opens_as_needed", return_value=False):
            other = self._cell("ur10,hande")
        self.assertIsNone(other.closed_width_mm)
        self.assertEqual(49.99, other.hand.open_width_mm)

    def test_a_toggle_is_a_jaw_io_single_toggle_and_nothing_else(self) -> None:
        from types import SimpleNamespace  # noqa: PLC0415

        from src.robot.grasping.recovery.push_gate import toggles  # noqa: PLC0415

        def robot(vendor: str, actuation: str = "") -> Any:
            return SimpleNamespace(gripper=SimpleNamespace(vendor=vendor, jaw_io=SimpleNamespace(actuation=actuation)))

        self.assertTrue(toggles(robot("jaw_io", "single_toggle")))
        self.assertTrue(toggles(robot(" JAW_IO ", "single_toggle")))
        self.assertFalse(toggles(robot("jaw_io", "single_solenoid")))
        self.assertFalse(toggles(robot("jaw_io", "double_solenoid")))
        self.assertFalse(toggles(robot("robotiq", "single_toggle")))
        self.assertFalse(toggles(SimpleNamespace()))


class AClosedPush(unittest.TestCase):
    def _push(self, gripper: Any, **keywords: Any) -> tuple[_Cell, Any]:
        cell = _Cell(self, _Fingered)
        if isinstance(gripper, _Robotiq):
            gripper.note = cell.log.write
            return cell, cell.push(gripper=gripper, **keywords)
        return cell, cell.push(**keywords)

    def test_the_hand_closes_in_the_air_at_p0_and_the_guard_holds_its_fingers_there(self) -> None:
        hand = _Robotiq()
        hand.width_mm = 20.0  # neither open nor closed: a closed push asks only that the hand read its width
        cell, outcome = self._push(hand, close_to_mm=0.0)

        self.assertIs(C.PUSHED, outcome.code, outcome.reason)
        self.assertEqual((APPROACH_LEG, *CONTACT_LEGS), outcome.legs_done)
        steps = [(kind, data if kind != "move" else data[0].label)
                 for kind, data in cell.log.events if kind in ("fingers", "ahead", "move", "gripper")]
        self.assertEqual([
            ("fingers", CLOSED), ("ahead", 4), ("fingers", None),  # judged ahead with the fingers closed
            ("move", "push_p0"),
            ("gripper", 0.0), ("fingers", CLOSED),  # closed in the air, the guard told where the hand reads them
            ("move", "push_down"), ("move", "push_push"), ("move", "push_back"), ("move", "push_up"),
            ("fingers", None),  # fully open for the guard again once the legs end; the caller opens the hand
        ], steps)
        kinds = cell.log.kinds()
        self.assertLess(kinds.index("fingers"), kinds.index("move"))
        self.assertLess(len(kinds) - 1 - kinds[::-1].index("fingers"), kinds.index("forget"),
                        "the guard was reset after the keep-out closed")

    def test_a_toggle_closes_by_one_change_of_its_output_in_the_air_at_p0(self) -> None:
        cell, outcome = self._push("jaws", close_to_mm=5.0)

        self.assertIs(C.PUSHED, outcome.code, outcome.reason)
        self.assertEqual((APPROACH_LEG, *CONTACT_LEGS), outcome.legs_done)
        steps = [(kind, data if kind != "move" else data[0].label)
                 for kind, data in cell.log.events if kind in ("fingers", "ahead", "move", "DO")]
        held = 5.0 + FINGER_OPENING_SLACK_MM
        self.assertEqual([("fingers", held), ("ahead", 4), ("fingers", None), ("move", "push_p0")], steps[:4])
        self.assertEqual("DO", steps[4][0], "one change of the tool output closed the hand at P0")
        self.assertEqual([("fingers", held), ("move", "push_down"), ("move", "push_push"), ("move", "push_back"),
                          ("move", "push_up"), ("fingers", None)], steps[5:])
        self.assertEqual(1, cell.jaws.commands_sent - cell.commands_before)
        self.assertTrue(cell.jaws.jaws_closed, "the caller opens the hand at the look, not the push")
        self.assertEqual([], cell.person.asked_after_connect)

    def test_a_toggle_whose_count_says_closed_is_never_pushed(self) -> None:
        cell = _Cell(self, _Fingered)
        cell.jaws.set_closed(True)
        outcome = cell.push(close_to_mm=5.0)
        self.assertIs(C.REFUSED_JAWS_CLOSED, outcome.code)
        self.assertIn("a push closes them from open at P0", outcome.reason)
        cell.assert_nothing_moved(outcome)

    def test_a_hand_that_is_not_connected_is_never_closed_for_a_push(self) -> None:
        hand = _Robotiq()
        hand.is_connected = False
        cell, outcome = self._push(hand, close_to_mm=0.0)
        self.assertIs(C.REFUSED_JAWS_UNKNOWN, outcome.code)
        self.assertEqual([], [kind for kind in cell.log.kinds() if kind in ("move", "fingers", "gripper")])

    def test_a_hand_that_does_not_close_leaves_the_arm_in_the_air_at_p0_and_goes_back(self) -> None:
        cell, outcome = self._push(_Robotiq(fails="close"), close_to_mm=0.0)
        self.assertIs(C.REFUSED_DOWN_NOT_SENT, outcome.code)
        self.assertTrue(outcome.refused_before_motion)
        self.assertEqual((APPROACH_LEG,), outcome.legs_done)
        self.assertEqual(["push_p0"], [pose.label for pose, _keywords in cell.moves()])
        self.assertIn("did not close", outcome.reason)
        self.assertEqual(("fingers", None), [event for event in cell.log.events if event[0] == "fingers"][-1])

    def test_an_open_push_reads_the_jaws_and_never_commands_them(self) -> None:
        hand = _Robotiq()
        cell, outcome = self._push(hand)
        self.assertIs(C.PUSHED, outcome.code, outcome.reason)
        self.assertEqual([], [kind for kind in cell.log.kinds() if kind in ("fingers", "gripper")])


class _FingeredPushingArm(boxed._PushingArm):
    def set_finger_opening(self, width_mm: float | None) -> None:
        self.motions.append(("fingers", None if width_mm is None else round(float(width_mm), 2)))


class APickPushesWithTheRobotiqClosed(unittest.TestCase):
    def _cell(self, hand: _Robotiq) -> Any:
        with mock.patch.object(boxed, "_toggle", return_value=hand):
            cell = boxed._PushCell(self, arm=_FingeredPushingArm)
        hand.note = lambda kind, data: cell.arm.motions.append((kind, data, len(cell.camera.taken)))
        cell.service.push_cell = dataclasses.replace(
            cell.service.push_cell, hand=dataclasses.replace(boxed.PUSH_HAND, open_width_mm=CLOSED),
            closed_width_mm=0.0)
        return cell

    def test_the_hand_opens_at_the_look_before_the_look_is_taken_again(self) -> None:
        hand = _Robotiq()
        cell = self._cell(hand)
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign()

        (attempt,) = run.attempts
        self.assertTrue(attempt.passed, run.render())
        self.assertEqual(boxed.PUSH_LABELS, cell.arm.push_motions())
        motions = cell.arm.motions
        commands = [(index, motion) for index, motion in enumerate(motions) if motion[0] == "gripper"]
        self.assertEqual([("gripper", 0.0, 3), ("gripper", "open", 3)], [motion for _index, motion in commands],
                         "closed at P0, opened before the fourth look was taken")
        p0 = motions.index(("move", "push_p0"))
        up = motions.index(("move", "push_up"))
        back = max(index for index, motion in enumerate(motions) if motion[0] == "joints")
        self.assertTrue(p0 < commands[0][0] < motions.index(("move", "push_down")), motions)
        self.assertTrue(up < back < commands[1][0], "the hand opened before the arm was back at the look")
        self.assertEqual(4, len(cell.camera.taken))
        self.assertEqual(hand.max_width_mm, hand.width_mm)

    def test_a_hand_that_does_not_open_at_the_look_ends_the_pick_as_a_gripper_fault(self) -> None:
        cell = self._cell(_Robotiq(fails="open"))
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign()

        report = run.last
        self.assertEqual("GRIPPER_FAULT", report.pick_report.outcome.name)
        self.assertEqual(boxed.PUSH_LABELS, cell.arm.push_motions())
        self.assertEqual(3, len(cell.camera.taken), "a look was taken with a hand nobody vouches for")
        self.assertIn("did not open again at the look", report.pick_report.attempts[-1].motion_message)


if __name__ == "__main__":
    unittest.main()
