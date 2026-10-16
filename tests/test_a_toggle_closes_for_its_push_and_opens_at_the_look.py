"""A toggle pushes with its jaws closed: one change of DO0 closes it in the air over the part, one more opens it at the
look after the push (the owner, 2026-10-16: "Ja, geschlossen"). The real hand and the real execution policy.

The owner's Hand-E as ``jaw_io`` single_toggle on tool DO0: every change of DO0 moves the jaws once, nothing reads them
back, and the program keeps the count. Open, its fingers stand 72 mm across behind the part, and on Thursday's cell no
push of that open hand ever planned beside a part in a pile; closed, 22 mm. The cell's push cell plans the toggle closed
at the band it reports (``PushCell.from_robot_config``, ``push_gate.toggles``); here the push cell says so itself. The
open push of any other hand is pinned in ``tests/test_a_push_and_its_re_pick_leave_the_toggle_alone.py``.

Pinned, through ``PickRun``, the toggle's three writes from the connect on, each a change and one a command:

* the close in the air at P0, after the move there and before the line down; the open at the look, once the arm is
  back there and before the look is taken again; the re-pick's close, after its approach. The count stands at three
  commands and is still vouched for, nobody asked;
* the same with DO0 standing HIGH with the jaws open at the connect, as the owner leaves it after opening them at the
  pendant: HIGH to LOW, LOW to HIGH, HIGH to LOW;
* a push that stops after the line down was sent leaves the jaws closed as they were closed for it: one write, the
  close at P0, nothing else, and a person decides.
"""

from __future__ import annotations

import dataclasses
import unittest
from typing import Any
from unittest import mock

from src.robot.core import MotionStatus
from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from tests import test_a_boxed_in_part_is_pushed_inside_the_pick as boxed
from tests.test_a_push_and_its_re_pick_leave_the_toggle_alone import _real_cell, _writes
from tests.test_a_toggle_asks_where_its_jaws_stand import _changes

#: The band the owner's toggle reports closed, ``min_width_mm``: where the guard holds its fingers for the push.
CLOSED_BAND_MM = 5.0


def _closed_cell(test: unittest.TestCase, **keywords: Any) -> Any:
    """The owner's toggle cell with the real policy, its push cell planning the fingers closed at the toggle's band."""
    cell = _real_cell(test, **keywords)
    cell.service.push_cell = dataclasses.replace(
        cell.service.push_cell, closed_width_mm=CLOSED_BAND_MM,
        hand=dataclasses.replace(boxed.PUSH_HAND, open_width_mm=CLOSED_BAND_MM + FINGER_OPENING_SLACK_MM))
    return cell


def _index(tail: list[Any], kind: str, label: str) -> int:
    return tail.index((kind, label))


class AToggleClosesForItsPushTests(unittest.TestCase):
    def test_closed_at_p0_opened_at_the_look_and_closed_once_on_the_part(self) -> None:
        for high in (False, True):
            with self.subTest(do0_high_at_the_connect=high):
                cell = _closed_cell(self, do0_high=high)

                with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
                    run = cell.campaign()

                self.assertIs(AutonomousGraspOutcome.SUCCEEDED, run.last.outcome, run.render())
                tail = cell.events[cell.connected_writes:]
                writes = _writes(cell)
                closed, opened = (not high), high
                self.assertEqual([("DO", 0, closed), ("DO", 0, opened), ("DO", 0, closed)],
                                 [tuple(write) for write in writes])
                at = [index for index, event in enumerate(tail) if isinstance(event, tuple) and event[:2] == ("DO", 0)]
                self.assertEqual(at, _changes(tail), "DO0 changed other than by the three writes")
                self.assertTrue(_index(tail, "move", "push_p0") < at[0] < _index(tail, "move", "push_down"),
                                "the hand did not close in the air at P0, before the line down")
                back = max(index for index, event in enumerate(tail[:at[1]])
                           if isinstance(event, tuple) and event[0] == "joints")
                self.assertTrue(_index(tail, "move", "push_up") < back < at[1],
                                "the hand opened before the arm was back at the look")
                approach = max(index for index, event in enumerate(tail)
                               if isinstance(event, tuple) and event[0] == "move"
                               and str(event[1]).startswith("approach"))
                self.assertTrue(at[1] < approach < at[2], "the re-pick's close did not follow its approach")
                self.assertEqual(3, cell.jaws.commands_sent)
                self.assertTrue(cell.jaws.jaws_closed)
                self.assertFalse(cell.jaws.edge_unknown)
                self.assertEqual("", cell.jaws.why_jaws_unknown())
                self.assertEqual([], cell.nobody.asked, "a person was asked")

    def test_a_push_that_stopped_after_the_line_down_leaves_the_jaws_closed_for_a_person(self) -> None:
        refused = boxed._refused(MotionStatus.WORKSPACE_REJECTED, "leaves the box")  # noqa: SLF001
        cell = _closed_cell(self, answers={"push_back": refused})

        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            run = cell.campaign(runs=2)

        self.assertTrue(run.last.needs_person, run.render())
        tail = cell.events[cell.connected_writes:]
        (write,) = _writes(cell)
        self.assertEqual(("DO", 0, True), tuple(write))
        self.assertTrue(_index(tail, "move", "push_p0") < tail.index(write) < _index(tail, "move", "push_down"))
        self.assertTrue(cell.jaws.jaws_closed, "the jaws closed for the push were opened after it stopped")
        self.assertEqual(1, cell.jaws.commands_sent)
        self.assertEqual([], cell.nobody.asked, "a person was asked")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
