"""A pick whose every grasp was refused before anything was sent says so in a typed reason, and its failed motions reach a
file (cell fixes Track C, RC4 and RC6).

On the owner's cell all five Zollstock picks ended "Pick failed with no typed failure reason after 0 recovery action(s)"
(robot.log 672, 726, 745, 797, 845): the guard refused the line in, the pick loop mapped it to ``EXECUTION_FAILED`` with
empty reasons, and the recovery loop had nothing to act on. Now an attempt whose every try was refused before anything
was sent, and none of whose tries reached the part, carries ``MOTION_PLAN_REFUSED``, which the service's adapter hands on
to the recovery loop (a rescan, never a motion). An attempt that ended with the hand at the part is never typed that way:
a person decides, as before. And the pick loop's lines reach ``robot.log`` and ``pick_loop.log``: every refused try at
WARNING, with its status and the driver's sentence.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.service import _RecoveryReportAdapter
from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.types.feedback import GraspFailureReason
from tests.test_a_refused_grasp_tries_the_next import LINE_IN_REFUSED, NO_PLAN, REFUSED, SENT_FAILED, _Cell

_REPO = Path(__file__).resolve().parents[1]


class ATypedRefusalTests(unittest.TestCase):
    def test_four_grasps_refused_before_anything_was_sent_say_motion_plan_refused(self) -> None:
        """Red before: reasons () after rank 0 alone."""
        cell = _Cell(dict.fromkeys(range(4), REFUSED))

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTION_FAILED, report.outcome)
        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), attempt.reasons)
        self.assertEqual("execution_failed", attempt.action)
        self.assertEqual([0, 1, 2, 3], [row["rank"] for row in attempt.tries])
        self.assertTrue(all(not row["sent"] and not row["reached_part"] for row in attempt.tries), attempt.tries)
        # The attempt's motion fields are its last try's.
        self.assertEqual("self_collision_rejected", attempt.motion_status)

    def test_the_service_hands_the_typed_refusal_on_to_the_recovery_loop(self) -> None:
        cell = _Cell(dict.fromkeys(range(4), REFUSED))
        report = cell.run()

        adapter = _RecoveryReportAdapter(SimpleNamespace(  # type: ignore[arg-type]
            outcome="execution_failed", needs_person=False, pick_report=report))

        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), adapter.failure_reasons)

    def test_one_grasp_refused_with_no_other_to_try_is_typed_too(self) -> None:
        cell = _Cell({0: NO_PLAN}, count=1)

        report = cell.run()

        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), attempt.reasons)

    def test_tries_that_were_refused_after_a_standoff_outside_the_box_are_typed(self) -> None:
        cell = _Cell(dict.fromkeys(range(4), LINE_IN_REFUSED), standoff_mm=60.0)

        report = cell.run()

        (attempt,) = report.attempts
        self.assertEqual((GraspFailureReason.MOTION_PLAN_REFUSED,), attempt.reasons)
        self.assertEqual([True] * 4, [row["sent"] for row in attempt.tries])

    def test_an_attempt_that_ended_with_the_hand_at_the_part_is_never_typed_so(self) -> None:
        cell = _Cell({0: LINE_IN_REFUSED}, standoff_mm=10.0)

        report = cell.run()

        (attempt,) = report.attempts
        self.assertEqual((), attempt.reasons)

    def test_a_sent_failure_after_refused_grasps_is_not_typed_as_a_refusal(self) -> None:
        cell = _Cell({0: REFUSED, 1: SENT_FAILED})

        report = cell.run()

        (attempt,) = report.attempts
        self.assertEqual([0, 1], cell.policy.executed)
        self.assertEqual((), attempt.reasons)
        self.assertEqual("controller_rejected", attempt.motion_status)


def _emit_and_read(destination: Path) -> subprocess.CompletedProcess[str]:
    code = "\n".join([
        "import importlib, logging",
        "importlib.import_module('src.robot.grasping.loop.pick_loop')",
        "logging.getLogger('src.robot.grasping.loop.pick_loop').warning('probe from the pick loop')",
        "logging.shutdown()",
    ])
    return subprocess.run([sys.executable, "-c", code], cwd=str(_REPO), capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(_REPO), "WILLY_LOG_DIR": str(destination)}, timeout=300)


class ThePickLoopsLinesReachAFileTests(unittest.TestCase):
    def test_a_line_of_the_pick_loop_lands_in_robot_log_and_its_own_file(self) -> None:
        """Red before: the pick loop logged through a bare logger, which reached no file (RC6, P5)."""
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch, "logs")
            result = _emit_and_read(destination)
            self.assertEqual(0, result.returncode, result.stderr[-800:])
            for name in ("robot.log", "pick_loop.log"):
                found = sorted(destination.rglob(name))
                self.assertTrue(found, f"no {name}: {sorted(p.name for p in destination.rglob('*'))}")
                text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in found)
                self.assertIn("probe from the pick loop", text)


if __name__ == "__main__":
    unittest.main()
