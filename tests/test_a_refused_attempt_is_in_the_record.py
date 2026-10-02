"""A refused attempt is in the record, and its failure line names the last motion and the push.

The owner's P5 (2026-10-01, fix plan RC6): an accepted 82-waypoint plan failed 22 ms later with no line in any file
that said why. The pick loop copies each attempt's ``motion_status`` and ``motion_message`` onto it, and since the cell
fixes every grasp it tried (``PickAttempt.tries``, contract 3), but the record kept none of them. Now the record's
``extra["attempts"]`` holds, for every attempt that commanded or was refused a motion, its action, its typed reasons,
its motion's status and message and the grasps it tried; a pick that commanded nothing writes no such key, as before.
The report's failure line says the last motion (``motion=status: message``) and the code of the last push it
considered (``push=...``).
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.record_logging import to_attempt_record
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport
from src.robot.execution.runtime_pick import PickSessionReport
from src.robot.grasping.loop.pick_loop import PickAttempt, PickOutcome
from src.robot.grasping.types.feedback import GraspFailureReason

#: The tries of one attempt as contract 3 states them: rank 0's line in refused before anything was sent, rank 1 sent
#: and refused by the controller.
TRIES = (
    {"rank": 0, "outcome": "motion_failed", "motion_status": "self_collision_rejected",
     "motion_message": "the exact guard refuses sample 7 of 41: finger_positive_x within 3.1 mm of seen_00",
     "sent": False, "reached_part": False},
    {"rank": 1, "outcome": "motion_failed", "motion_status": "controller_rejected",
     "motion_message": "moveJ was sent and the controller raised", "sent": True, "reached_part": False},
)


def _report(*attempts: Any, outcome: AutonomousGraspOutcome = AutonomousGraspOutcome.EXECUTION_FAILED,
            telemetry: "dict[str, Any] | None" = None) -> AutonomousGraspReport:
    pick = PickSessionReport(outcome=PickOutcome.EXECUTION_FAILED, robot_vendor="ur", is_simulated=False,
                             gripper_present=True, attempts=tuple(attempts))
    return AutonomousGraspReport(outcome=outcome, mode=GraspMode.DENSE_CLUTTER,
                                 profile=_profile_for(GraspMode.DENSE_CLUTTER), pick_report=pick,
                                 telemetry=telemetry or {})


def _attempt(**fields: Any) -> PickAttempt:
    values: dict[str, Any] = dict(attempt_index=0, target_index=0, reasons=(), score=0.8, action="execution_failed")
    values.update(fields)
    return PickAttempt(**values)


class TheRecordHoldsTheMotionTests(unittest.TestCase):
    def test_a_failed_motion_is_in_the_record(self) -> None:
        """P5's case: the attempt's motion status and message reach the record."""
        failed = _attempt(motion_status="controller_rejected",
                          motion_message="moveJ was sent and the controller raised: RTDE control script stopped")
        record = to_attempt_record(_report(failed), attempt_id="p5")
        (row,) = record.extra["attempts"]
        self.assertEqual("controller_rejected", row["motion_status"])
        self.assertIn("RTDE control script stopped", row["motion_message"])
        self.assertEqual("execution_failed", row["action"])
        self.assertEqual([], row["tries"])
        self.assertEqual(json.loads(json.dumps(record.to_dict()))["extra"]["attempts"], [row])

    def test_every_grasp_an_attempt_tried_is_in_the_record(self) -> None:
        """Contract 3: the tries ride along as plain data, in the order they were tried."""
        tried = SimpleNamespace(attempt_index=1, action="execution_failed",
                                reasons=(GraspFailureReason.MOTION_PLAN_REFUSED,),
                                motion_status="controller_rejected", motion_message="moveJ was sent and the controller "
                                                                                    "raised", tries=TRIES)
        record = to_attempt_record(SimpleNamespace(
            outcome=AutonomousGraspOutcome.EXECUTION_FAILED, mode=GraspMode.DENSE_CLUTTER, profile=None, telemetry={},
            pick_report=SimpleNamespace(attempts=(tried,), calculator_telemetry={}, executed_grasp=None)),
            attempt_id="tries")
        (row,) = record.extra["attempts"]
        self.assertEqual([dict(t) for t in TRIES], row["tries"])
        self.assertEqual(["motion_plan_refused"], row["reasons"])
        self.assertEqual(1, row["attempt_index"])

    def test_a_pick_that_commanded_nothing_writes_no_attempts(self) -> None:
        nothing = _attempt(action="rescan", reasons=(GraspFailureReason.ALL_COLLIDED,))
        record = to_attempt_record(_report(nothing, outcome=AutonomousGraspOutcome.NO_VALID_GRASP), attempt_id="none")
        self.assertNotIn("attempts", record.extra)


class TheFailureLineSaysItTests(unittest.TestCase):
    def test_the_last_motion_and_the_push_on_one_line(self) -> None:
        first = _attempt(attempt_index=0, action="push", reasons=(GraspFailureReason.ALL_COLLIDED,))
        last = _attempt(attempt_index=1, motion_status="self_collision_rejected",
                        motion_message="the exact guard refuses sample 7 of 41")
        pushes = [{"trigger": "all_collided", "code": "no_free_direction", "motion_started": False}]
        line = _report(first, last, telemetry={"pushes": pushes}).failure_summary()
        self.assertIn("motion=self_collision_rejected: the exact guard refuses sample 7 of 41", line)
        self.assertIn("push=no_free_direction", line)
        self.assertNotIn("\n", line)
        self.assertTrue(line.isascii())

    def test_no_motion_and_no_push_add_nothing(self) -> None:
        line = _report(_attempt(action="rescan", reasons=(GraspFailureReason.ALL_COLLIDED,)),
                       outcome=AutonomousGraspOutcome.NO_VALID_GRASP).failure_summary()
        self.assertNotIn("motion=", line)
        self.assertNotIn("push=", line)

    def test_a_succeeded_pick_says_nothing(self) -> None:
        report = replace(_report(_attempt(motion_status="ok")), outcome=AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual("", report.failure_summary())


if __name__ == "__main__":
    unittest.main()
