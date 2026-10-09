"""Where a wrist pick's last try left the arm at a standoff, the report the service and the task are handed says so.

The pick loop says where a try left the arm that was due back at its look and was not driven back there
(``PickReport.stands_at``, "standoff of grasp N", S2 of the owner's 2026-10-08 decisions), and a task stops for a
person on it instead of picking again from there. The service's report holds the runtime's ``PickSessionReport``,
not the loop's ``PickReport``, so the session report must carry it on: before 2026-10-09 it did not, and on the
real service path ``needs_person`` and the task never saw where the arm stood (the tests faked the report).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport
from src.robot.execution.runtime_pick import PickSessionReport, RuntimePickService
from src.robot.grasping.loop.pick_loop import PickOutcome, PickReport


def _composed(pick_report: PickReport) -> PickSessionReport:
    """The session report the runtime composes from ``pick_report``, on an orchestrator that ran no policy call."""
    orchestrator = SimpleNamespace(_last_policy_report=None, gripper=None, arm=SimpleNamespace())
    service = RuntimePickService(orchestrator=orchestrator)  # type: ignore[arg-type]
    return service._compose(pick_report, orchestrator_s=0.1)  # noqa: SLF001


def _service_report(session: PickSessionReport) -> AutonomousGraspReport:
    return AutonomousGraspReport(outcome=AutonomousGraspOutcome.EXECUTION_FAILED, mode=GraspMode.EASY,
                                 profile=_profile_for(GraspMode.EASY), pick_report=session)


class TheSessionReportCarriesWhereTheArmStandsTests(unittest.TestCase):
    def test_where_a_try_left_the_arm_reaches_the_session_report(self) -> None:
        session = _composed(PickReport(outcome=PickOutcome.EXECUTION_FAILED, stands_at="standoff of grasp 2"))
        self.assertEqual("standoff of grasp 2", session.stands_at)

    def test_a_pick_that_left_the_arm_where_its_tries_started_says_nothing(self) -> None:
        session = _composed(PickReport(outcome=PickOutcome.EXECUTION_FAILED))
        self.assertEqual("", session.stands_at)

    def test_the_service_report_needs_a_person_where_the_arm_was_left(self) -> None:
        left = _service_report(_composed(PickReport(outcome=PickOutcome.EXECUTION_FAILED,
                                                    stands_at="way back from the standoff of grasp 1")))
        self.assertTrue(left.needs_person, "a person decides where the move back did not run")

    def test_the_service_report_asks_nobody_where_the_arm_is_back(self) -> None:
        back = _service_report(_composed(PickReport(outcome=PickOutcome.EXECUTION_FAILED)))
        self.assertFalse(back.needs_person)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
