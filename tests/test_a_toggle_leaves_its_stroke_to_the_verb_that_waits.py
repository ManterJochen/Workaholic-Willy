"""A hand with no switch leaves its jaws' stroke to the verb that waits for it, and changes its output once all the same.

The map's C7a (the owner, 2026-10-09): ``JawIOGripper.set_closed(closed, wait=False)`` sends the one change a command
is, and comes back with the moment the stroke is over, ``time.monotonic()`` seconds; ``wait_settled`` waits out what is
left of it. In between, the verb judges its next leg, and sends nothing. Pinned here:

* the owner's toggle on tool DO0 changes its output exactly once per command, whether the stroke is waited here or left;
* a command left so returns at once, with the stroke's end, and ``wait_settled`` sleeps the rest, never more;
* a command that finds a stroke left so still running waits it out first, as it waited inside ``set_closed`` before;
* nothing moved, nothing to wait: a close on closed jaws leaves no stroke;
* a wait that reads a switch, a closing solenoid with its reeds, is waited here as ever;
* ``wait`` true, the default, is the command it always was.

Honesty bucket (3): the driver's logic on a recorded stand-in for the controller's I/O, never a gripper.
"""

from __future__ import annotations

import time
import unittest

from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_jaw_io_gripper import CLOSE_PIN, CLOSED_SW, HIGH, LOW, OPEN_SW, SETTLE_S, FakeIO, _Timeline, _toggle


def _slept(io: _Timeline) -> list[float]:
    return [float(seconds) for name, seconds in io.events if name == "sleep"]  # type: ignore[arg-type]


class AToggleLeavesItsStrokeToTheVerbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.io = _Timeline()
        self.g = _toggle(self.io, close_settle_s=SETTLE_S)
        self.g.connect()
        self.io.events.clear()

    def test_a_close_left_to_the_verb_is_one_change_and_no_wait(self) -> None:
        before = time.monotonic()
        settles_at = self.g.set_closed(True, wait=False)
        after = time.monotonic()
        self.assertEqual([HIGH], self.io.events, "one change, and nothing waited here")
        assert settles_at is not None
        self.assertGreaterEqual(settles_at, before + SETTLE_S - 1e-6)
        self.assertLessEqual(settles_at, after + SETTLE_S + 1e-6)
        self.assertTrue(self.g.jaws_closed)

    def test_wait_settled_waits_what_is_left_of_the_stroke_and_no_more(self) -> None:
        settles_at = self.g.set_closed(True, wait=False)
        self.g.wait_settled(settles_at)
        slept = _slept(self.io)
        self.assertEqual(1, len(slept))
        self.assertGreater(slept[0], 0.0)
        self.assertLessEqual(slept[0], SETTLE_S + 1e-6)
        self.assertEqual([HIGH], self.io.outputs_written(), "waiting sent nothing")

    def test_a_stroke_already_over_is_not_waited(self) -> None:
        self.g.wait_settled(time.monotonic() - 1.0)
        self.g.wait_settled(None)
        self.assertEqual([], self.io.events)

    def test_an_open_left_to_the_verb_is_one_change_too(self) -> None:
        self.g.set_closed(True)
        self.io.events.clear()
        settles_at = self.g.set_closed(False, wait=False)
        self.assertIsNotNone(settles_at)
        self.assertEqual([LOW], self.io.events)
        self.assertFalse(self.g.jaws_closed)

    def test_a_close_on_closed_jaws_leaves_no_stroke(self) -> None:
        self.g.set_closed(True)
        self.io.events.clear()
        self.assertIsNone(self.g.set_closed(True, wait=False))
        self.assertEqual([], self.io.events, "a change on closed jaws opens them: nothing went out")

    def test_the_next_command_waits_out_a_stroke_left_to_the_verb_before_it_changes_anything(self) -> None:
        """⭐ A change while the jaws still travel turns them round mid-stroke: the open after a close left so waits out
        the close's stroke first, then changes the output once, and waits out its own."""
        self.g.set_closed(True, wait=False)
        self.g.set_closed(False)
        names = [name for name, _ in self.io.events]
        self.assertEqual([f"out{CLOSE_PIN}", "sleep", f"out{CLOSE_PIN}", "sleep"], names)
        first_wait = _slept(self.io)[0]
        self.assertGreater(first_wait, 0.0)
        self.assertLessEqual(first_wait, SETTLE_S + 1e-6)
        self.assertEqual([HIGH, LOW], self.io.outputs_written(), "exactly one change per command")

    def test_a_waited_command_is_the_one_it_always_was(self) -> None:
        self.assertIsNone(self.g.set_closed(True))
        self.assertEqual([HIGH, ("sleep", SETTLE_S)], self.io.events)


class ASolenoidLeavesOnlyAStrokeNoSwitchReadsTests(unittest.TestCase):
    def test_a_close_that_reads_its_reeds_is_waited_here_as_ever(self) -> None:
        io = FakeIO({CLOSED_SW: False, OPEN_SW: False})
        slept: list[float] = []
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, closed_confirm_input_pin=CLOSED_SW,
                         open_confirm_input_pin=OPEN_SW, close_settle_s=0.4, sleep=slept.append)
        g.connect()
        self.assertIsNone(g.set_closed(True, wait=False), "a wait that reads a switch is no stroke to leave")
        self.assertIn(0.4, slept, "the reeds were read after the travel time, here")

    def test_a_solenoid_with_no_switch_leaves_its_close_stroke(self) -> None:
        io = FakeIO()
        slept: list[float] = []
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, close_settle_s=0.4, sleep=slept.append)
        g.connect()
        settles_at = g.set_closed(True, wait=False)
        self.assertIsNotNone(settles_at)
        self.assertEqual([], slept)
        self.assertEqual([(CLOSE_PIN, True)], [(pin, value) for pin, value, _port in io.writes][-1:])

    def test_an_open_a_switch_confirms_has_no_stroke_to_leave(self) -> None:
        io = FakeIO()
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, open_confirm_input_pin=OPEN_SW, close_settle_s=0.4,
                         sleep=lambda _s: None)
        g.connect()
        g.set_closed(True)
        self.assertIsNone(g.set_closed(False, wait=False))


if __name__ == "__main__":
    unittest.main()
