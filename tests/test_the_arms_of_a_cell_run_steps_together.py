"""Steps of several arms at once (the owner, 2026-10-16: "Python + together()", "Alle Arme halten an").

Pinned here:

* :func:`together` runs its steps at the same time, each on its own thread, and returns what each returned, in order;
* a step that raises, or returns a report whose ``ok`` is false, halts every arm of the cell once, with a reason that
  names the step; every other step still ends, and :class:`TogetherFailed` says what each did;
* outside ``with cell.connected():``, or with a step that is no call, nothing runs;
* the outermost connected block is the cell it halts: a cell of several arms, not one of its arms' own sessions;
* on a real cell of two arms (each on the fake transport): ``arms["werker"]`` is the werker's cell, a failing step
  latches both arms, and after the block :func:`together` refuses again.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.execution.together import StepEnd, TogetherFailed, live_cell, together
from tests.test_a_cell_of_several_arms_comes_up_whole import _arm_cell, _turned


class _Session:
    """A connected cell as together() sees it: what it halts."""

    def __init__(self) -> None:
        self.halts: list[str] = []

    def halt_every_arm(self, reason: str) -> None:
        self.halts.append(reason)


class TogetherRunsStepsAtOnceTests(unittest.TestCase):
    def test_the_steps_run_at_the_same_time_and_come_back_in_order(self) -> None:
        meet = threading.Barrier(2, timeout=5.0)

        def step(value: str) -> str:
            meet.wait()  # one after the other, the first would wait here alone and time out
            return value

        session = _Session()
        with live_cell(session):
            self.assertEqual(("halter", "werker"), together(lambda: step("halter"), lambda: step("werker")))
        self.assertEqual([], session.halts)

    def test_a_step_that_raises_halts_the_cell_once_and_every_other_step_still_ends(self) -> None:
        ended = threading.Event()

        def fails() -> None:
            raise RuntimeError("the werker's line is refused")

        def moves() -> str:
            ended.set()
            return "moved"

        session = _Session()
        with live_cell(session), self.assertRaises(TogetherFailed) as caught:
            together(moves, fails)
        self.assertTrue(ended.is_set())
        self.assertEqual(1, len(session.halts))
        self.assertIn("step 1 raised RuntimeError: the werker's line is refused", session.halts[0])
        steps = caught.exception.steps
        self.assertEqual(["moved", None], [step.result for step in steps])
        self.assertEqual([1], [step.index for step in caught.exception.failed])
        self.assertIsInstance(steps[1].error, RuntimeError)
        self.assertEqual(session.halts[0], caught.exception.reason)

    def test_a_report_that_says_it_did_not_succeed_halts_the_cell(self) -> None:
        refused = SimpleNamespace(ok=False, render=lambda: "move refused: the other arm stands in the way")
        session = _Session()
        with live_cell(session), self.assertRaises(TogetherFailed) as caught:
            together(lambda: SimpleNamespace(ok=True), lambda: refused, lambda: 7)
        self.assertIn("step 1 did not succeed: move refused: the other arm stands in the way", session.halts[0])
        self.assertEqual([False, True, False], [step.failed for step in caught.exception.steps])

    def test_outside_a_connected_cell_and_with_a_step_that_is_no_call_nothing_runs(self) -> None:
        ran: list[int] = []
        with self.assertRaises(RuntimeError) as caught:
            together(lambda: ran.append(1))
        self.assertIn("inside `with cell.connected():`", str(caught.exception))
        with live_cell(_Session()), self.assertRaises(TypeError):
            together(lambda: ran.append(2), "move")  # type: ignore[arg-type]
        self.assertEqual([], ran)

    def test_the_outermost_connected_block_is_the_cell_that_is_halted(self) -> None:
        outer, inner = _Session(), _Session()
        with live_cell(outer), live_cell(inner), self.assertRaises(TogetherFailed):
            together(lambda: SimpleNamespace(ok=False))
        self.assertEqual((1, 0), (len(outer.halts), len(inner.halts)))

    def test_a_step_end_says_whether_it_failed(self) -> None:
        self.assertFalse(StepEnd(0, result=None).failed)
        self.assertTrue(StepEnd(0, error=ValueError("x")).failed)
        self.assertTrue(StepEnd(0, result=SimpleNamespace(ok=False)).failed)


class TogetherOnACellOfTwoArmsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.halter, self.werker = _arm_cell("127.0.0.21"), _arm_cell("127.0.0.22")
        self.halter._in_cell, self.werker._in_cell = np.eye(4), _turned(180.0, shift=(3000.0, 0.0, 0.0))
        self.halter._arms = {"halter": self.halter, "werker": self.werker}

    def test_a_failing_step_latches_both_arms_and_after_the_block_together_refuses(self) -> None:
        arms: Any
        with self.halter.connected() as arms:
            self.assertIs(self.werker, arms["werker"])
            self.assertEqual((1, 2), together(lambda: 1, lambda: 2))
            with self.assertRaises(TogetherFailed):
                together(lambda: 1, lambda: SimpleNamespace(ok=False))
            for name in ("halter", "werker"):
                with self.subTest(arm=name):
                    state = arms[name].arm.halt_state()
                    self.assertIsNotNone(state)
                    self.assertIn("together(): step 1 did not succeed", str(state))
        with self.assertRaises(RuntimeError):
            together(lambda: 1)

    def test_a_cell_of_one_arm_halts_its_arm(self) -> None:
        alone = _arm_cell("127.0.0.23")
        with alone.connected(), self.assertRaises(TogetherFailed):
            together(lambda: SimpleNamespace(ok=False))
        self.assertIsNotNone(alone.arm.halt_state())


if __name__ == "__main__":
    unittest.main()
