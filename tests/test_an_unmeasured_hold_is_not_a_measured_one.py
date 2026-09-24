"""A close nothing measured is reported as unmeasured, judged as unmeasured, and carried at the width it was told.

Owner-cell audit, reproduced on 2026-09-23 (scratchpad ``handE_toggle/repro_verify.py``, ``repro_attach.py``) on the
owner's hand: a Hand-E on the Robotiq I/O coupling, ``jaw_io`` single_toggle on tool DO0, no feedback wired, 49.99 /
5.0 mm. After a close that driver answers ``is_object_detected`` with its own command and ``get_width_mm`` with its
closed band, and says both honestly elsewhere: ``hold_evidence()`` is UNMEASURED and it does not claim ``MeasuresWidth``.
Four readers took the answers as measurements:

* the grasp policy recorded ``object_detected=True`` on every close, so a campaign's successes read as parts held;
* ``ObjectDetectingGripperVerifier`` passed every close as ``gripper_object_detected``;
* ``WidthDeltaGripperVerifier`` read the 5.0 mm band against a 7.0 mm threshold and failed every close, held part or
  not, as ``jaws_collapsed_to_minimum`` (on a cell that turns verification on);
* the policy attached the carried part 5.0 mm wide for a 40 mm grasp, so the planner lifted it about 34 mm too narrow.

The robot's own hand verbs already read the hold evidence and the measured flag; these readers now do too. Each
red-first test says what the code before this change did.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from src.robot.core.gripper import HoldEvidence
from src.robot.grasping.closed_loop.verification import (
    CompositeGraspVerifier,
    GraspVerificationContext,
    GraspVerificationPolicy,
    ObjectDetectingGripperVerifier,
    VerificationOutcome,
    WidthDeltaGripperVerifier,
)
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_a_stopped_controller_moves_no_jaws import (
    _IO,
    _Arm,
    _grasp,
    _pulses,
    _service,
    _toggle,
    two_scan_service,
)

_PART_PIN = 1


def _closed_toggle() -> Any:
    jaws = _toggle([])
    jaws.set_closed(True)
    return jaws


def _part_sensed(*, held: bool, closed: bool = False) -> Any:
    """A solenoid jaw with a part-present input, which the owner's toggle cannot have: a toggle reads no sensor."""
    jaws = JawIOGripper(_IO([], {_PART_PIN: held}), close_output_pin=0, part_present_input_pin=_PART_PIN,
                        close_settle_s=0.0, min_width_mm=5.0, max_width_mm=49.99, sleep=lambda _s: None)
    jaws.connect()
    if closed:
        jaws.set_closed(True)
    return jaws


def _context(gripper: Any, *, policy: "GraspVerificationPolicy | None" = None) -> GraspVerificationContext:
    return GraspVerificationContext(
        grasp=_grasp(), policy=policy or GraspVerificationPolicy(enabled=True, width_delta_max_mm=10.0),
        gripper=gripper, pre_close_width_mm=49.99, post_close_width_mm=float(gripper.get_width_mm()),
        commanded_close_width_mm=39.0,
    )


def _builder_verifier(**verification: Any) -> Any:
    """The composite ``from_robot_config`` wires when ``robot.grasping.verification.enabled`` is on."""
    from src.config.schema.robot.grasping_schema import RobotGraspingConfig
    from src.robot.execution.autonomous_grasp.builders import build_closed_loop_actors

    cfg = RobotGraspingConfig.model_validate({"verification": {"enabled": True, **verification}})
    _refiner, verifier, _strategy = build_closed_loop_actors(
        cfg, refinement_policy=None, verification_policy=GraspVerificationPolicy(enabled=True),
        recovery_policy=None, refiner=None, verifier=None, recovery_strategy=None,
    )
    return verifier


# ---------------------------------------------------------------------------------------------------
# The grasp policy
# ---------------------------------------------------------------------------------------------------


class _AttachingArm(_Arm):
    def __init__(self) -> None:
        super().__init__([])
        self.attached: list[float] = []

    def attach_payload(self, width_mm: float) -> bool:
        self.attached.append(float(width_mm))
        return True


class _MeasuringJaws:
    """A jaw with a position register that reads 38.2 mm after the close, and a hold it measured."""

    min_width_mm = 0.0
    max_width_mm = 50.0
    is_connected = True

    def set_width_mm(self, width_mm: float, *, speed: Any = None, force: Any = None) -> None:
        return None

    def get_width_mm(self) -> float:
        return 38.2

    def width_is_measured(self) -> bool:
        return True

    def is_object_detected(self) -> bool:
        return True

    def hold_evidence(self) -> HoldEvidence:
        return HoldEvidence.HELD


class ThePolicyRecordsWhatWasMeasuredTests(unittest.TestCase):
    def test_a_toggle_with_no_feedback_closes_unmeasured_not_detected(self) -> None:
        """Red before: object_detected=True, the command echo."""
        jaws = _toggle([])

        report = GraspExecutionPolicy(arm=_Arm([]), gripper=jaws, pre_open_width_mm=49.99).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome)
        self.assertIsNone(report.object_detected)

    def test_a_part_pin_is_a_measurement_both_ways(self) -> None:
        held = _part_sensed(held=True)
        empty = _part_sensed(held=False)

        held_report = GraspExecutionPolicy(arm=_Arm([]), gripper=held).execute(_grasp())
        empty_report = GraspExecutionPolicy(arm=_Arm([]), gripper=empty).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, held_report.outcome)
        self.assertIs(True, held_report.object_detected)
        self.assertIs(PolicyOutcome.OBJECT_NOT_DETECTED, empty_report.outcome)
        self.assertIs(False, empty_report.object_detected)

    def test_a_robotiq_still_moving_is_still_not_detected(self) -> None:
        """No weakening: gOBJ 0 reads UNMEASURED as evidence, and is_object_detected False stays final."""
        from src.config.schema.robot import GripperConfig
        from tests.test_gripper_commands_the_number_it_promised import _gripper

        robotiq, driver = _gripper(GripperConfig(min_width_mm=5.0, max_width_mm=50.0, closed_width_mm=0.0))
        driver.obj = 0
        self.assertIs(HoldEvidence.UNMEASURED, robotiq.hold_evidence())

        report = GraspExecutionPolicy(arm=_Arm([]), gripper=robotiq).execute(_grasp())

        self.assertIs(PolicyOutcome.OBJECT_NOT_DETECTED, report.outcome)

    def test_a_detecting_gripper_that_measures_empty_is_not_detected(self) -> None:
        class _Contradicting(_MeasuringJaws):
            def hold_evidence(self) -> HoldEvidence:
                return HoldEvidence.EMPTY

        report = GraspExecutionPolicy(arm=_Arm([]), gripper=_Contradicting()).execute(_grasp())

        self.assertIs(PolicyOutcome.OBJECT_NOT_DETECTED, report.outcome)

    def test_a_detecting_gripper_that_says_nothing_more_is_still_trusted(self) -> None:
        from tests.test_grasp_execution_policy import _DetectingGripper

        report = GraspExecutionPolicy(arm=_Arm([]), gripper=_DetectingGripper(detects=True)).execute(_grasp())

        self.assertIs(True, report.object_detected)


class ThePolicyCarriesTheWidthItKnowsTests(unittest.TestCase):
    def test_a_toggle_attaches_the_commanded_grip_not_its_closed_band(self) -> None:
        """Red before: the part was attached 5.0 mm wide, the band get_width_mm reports while closed."""
        arm = _AttachingArm()

        GraspExecutionPolicy(arm=arm, gripper=_toggle([]), pre_open_width_mm=49.99).execute(_grasp())

        self.assertEqual([39.0], arm.attached)

    def test_a_measured_width_is_still_what_is_attached(self) -> None:
        arm = _AttachingArm()

        GraspExecutionPolicy(arm=arm, gripper=_MeasuringJaws()).execute(_grasp())

        self.assertEqual([38.2], arm.attached)


# ---------------------------------------------------------------------------------------------------
# The verifiers
# ---------------------------------------------------------------------------------------------------


class TheVerifiersJudgeOnlyWhatWasMeasuredTests(unittest.TestCase):
    def test_the_hold_verifier_is_inconclusive_on_an_unmeasured_close(self) -> None:
        """Red before: PASSED gripper_object_detected, on the command echo."""
        report = ObjectDetectingGripperVerifier().verify(_context(_closed_toggle()))

        self.assertIs(VerificationOutcome.INCONCLUSIVE, report.outcome)
        self.assertEqual("hold_not_measured", report.reason)
        self.assertEqual("unmeasured", report.telemetry["hold_evidence"])

    def test_the_width_verifier_is_inconclusive_on_a_band(self) -> None:
        """Red before: FAILED jaws_collapsed_to_minimum, 5.0 mm against 5.0 + 2.0, held part or not."""
        report = WidthDeltaGripperVerifier().verify(_context(_closed_toggle()))

        self.assertIs(VerificationOutcome.INCONCLUSIVE, report.outcome)
        self.assertEqual("width_not_measured", report.reason)
        self.assertEqual(5.0, report.telemetry["post_close_width_mm"])

    def test_the_composite_example_18_wires_learns_nothing_and_says_so(self) -> None:
        """Red before: FAILED child_failed:jaws_collapsed_to_minimum on every close; fixing only the width verifier
        would have PASSED on the echo instead."""
        verifier = _builder_verifier()
        self.assertIsInstance(verifier, CompositeGraspVerifier)

        report = verifier.verify(_context(_closed_toggle()))

        self.assertIs(VerificationOutcome.PASSED, report.outcome)
        self.assertEqual("no_verifier_could_measure", report.reason)
        self.assertFalse(report.telemetry["measured_something"])

    def test_a_cell_that_requires_a_conclusive_verdict_fails_an_unmeasured_close(self) -> None:
        report = _builder_verifier(require_all_conclusive=True).verify(_context(_closed_toggle()))

        self.assertIs(VerificationOutcome.FAILED, report.outcome)
        self.assertEqual("child_inconclusive:hold_not_measured", report.reason)

    def test_a_wired_part_pin_is_judged(self) -> None:
        policy = GraspVerificationPolicy(enabled=True, require_object_detected=True)
        held = _part_sensed(held=True, closed=True)
        empty = _part_sensed(held=False, closed=True)

        self.assertIs(VerificationOutcome.PASSED,
                      ObjectDetectingGripperVerifier().verify(_context(held, policy=policy)).outcome)
        self.assertIs(VerificationOutcome.FAILED,
                      ObjectDetectingGripperVerifier().verify(_context(empty, policy=policy)).outcome)

    def test_a_detected_part_the_gripper_measured_empty_is_empty(self) -> None:
        class _Contradicting(_MeasuringJaws):
            def hold_evidence(self) -> HoldEvidence:
                return HoldEvidence.EMPTY

        policy = GraspVerificationPolicy(enabled=True, require_object_detected=True)
        report = ObjectDetectingGripperVerifier().verify(_context(_Contradicting(), policy=policy))

        self.assertIs(VerificationOutcome.FAILED, report.outcome)
        self.assertEqual("gripper_object_not_detected", report.reason)

    def test_a_measured_hold_and_width_still_pass(self) -> None:
        jaws = _MeasuringJaws()
        for verifier in (ObjectDetectingGripperVerifier(), WidthDeltaGripperVerifier()):
            with self.subTest(type(verifier).__name__):
                self.assertIs(VerificationOutcome.PASSED, verifier.verify(_context(jaws)).outcome)


# ---------------------------------------------------------------------------------------------------
# The service report and the campaign
# ---------------------------------------------------------------------------------------------------


def _report(**overrides: Any) -> Any:
    from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
    from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome, AutonomousGraspReport

    base = AutonomousGraspReport(outcome=AutonomousGraspOutcome.SUCCEEDED, mode=GraspMode.EASY,
                                 profile=_profile_for(GraspMode.EASY))
    return replace(base, **overrides)


class TheReportSaysWhatWasMeasuredTests(unittest.TestCase):
    def test_hold_measured_reads_the_close(self) -> None:
        from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome

        cases = (
            ("measured held", _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=True)), True),
            ("unmeasured", _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=None)), False),
            ("no gripper", _report(pick_report=SimpleNamespace(gripper_present=False, object_detected=None)), None),
            ("two-scan unmeasured", _report(telemetry={"gripper_present": True, "object_detected": None}), False),
            ("two-scan held", _report(telemetry={"gripper_present": True, "object_detected": True}), True),
            ("not a success", _report(outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
                                      pick_report=SimpleNamespace(gripper_present=True, object_detected=None)), None),
        )
        for label, report, expected in cases:
            with self.subTest(label):
                self.assertIs(expected, report.hold_measured)
                self.assertIs(expected, report.to_dict()["hold_measured"])

    def test_an_unmeasured_success_says_so_when_printed(self) -> None:
        unmeasured = _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=None))
        measured = _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=True))

        self.assertIn("hold       not measured", unmeasured.render())
        self.assertNotIn("not measured", measured.render())
        self.assertTrue(unmeasured.render().isascii())


class _Scripted:
    """A service answering scripted reports."""

    def __init__(self, reports: list[Any]) -> None:
        self.reports = reports

    def pick(self) -> Any:
        return self.reports.pop(0)


class TheCampaignCountsWhatWasMeasuredTests(unittest.TestCase):
    def test_a_campaign_of_unmeasured_successes_says_what_its_count_rests_on(self) -> None:
        from src.robot.execution.pick_run import PassRule, PickRun, Recording

        unmeasured = _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=None))
        measured = _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=True))

        run = PickRun.from_service(_Scripted([unmeasured, measured, unmeasured]), runs=3, recording=Recording.off(),
                                   rule=PassRule(fraction=0.8)).execute()

        self.assertEqual(3, run.succeeded)
        self.assertEqual(2, run.unmeasured)
        self.assertTrue(run.passed, "saying a success was unmeasured changes no verdict")
        self.assertIn("2 of 3 success(es) with no hold measured by the gripper", run.summary())
        self.assertIn("close command's word", run.summary())
        self.assertIn("hold not measured", run.attempts[0].render())
        self.assertNotIn("hold not measured", run.attempts[1].render())
        self.assertEqual(2, run.to_dict()["unmeasured"])
        self.assertIs(False, run.to_dict()["attempts"][0]["hold_measured"])
        self.assertTrue(run.render().isascii())

    def test_a_campaign_whose_successes_were_measured_prints_no_such_line(self) -> None:
        from src.robot.execution.pick_run import PickRun, Recording

        measured = _report(pick_report=SimpleNamespace(gripper_present=True, object_detected=True))

        run = PickRun.from_service(_Scripted([measured]), runs=1, recording=Recording.off()).execute()

        self.assertEqual(0, run.unmeasured)
        self.assertNotIn("no hold measured", run.summary())

    def test_the_owner_cell_campaign_counts_every_success_unmeasured(self) -> None:
        """The whole chain: the real pick loop and service, the real toggle driver, a healthy arm."""
        from src.robot.execution.pick_run import PickRun, Recording

        events: list[Any] = []
        jaws = _toggle(events)
        service = _service(_Arm(events), jaws)

        run = PickRun.from_service(service, runs=2, recording=Recording.off()).execute()

        self.assertEqual(2, run.succeeded)
        self.assertEqual(2, run.unmeasured)
        self.assertIsNone(run.last.pick_report.object_detected)
        self.assertIs(False, run.last.hold_measured)
        self.assertGreaterEqual(_pulses(events), 2)

    def test_the_two_scan_path_with_example_18s_verification_is_an_unmeasured_success(self) -> None:
        """Red before: VERIFICATION_FAILED on every pick (the band read as a collapse)."""
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome

        events: list[Any] = []
        jaws = _toggle(events)

        report = two_scan_service(_Arm(events), jaws, verifier=_builder_verifier()).pick()

        self.assertIs(AutonomousGraspOutcome.SUCCEEDED, report.outcome, report.render())
        self.assertEqual("no_verifier_could_measure", report.telemetry["verification_reason"])
        self.assertIsNone(report.telemetry["object_detected"])
        self.assertIs(False, report.hold_measured)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
