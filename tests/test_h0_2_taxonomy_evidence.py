"""H0.2 — record_logging derives the offline failure-taxonomy ``extra.*_evidence`` flags from the
runtime's OWN typed verdicts (calculator GraspFailureReasons + the post-grasp verification reason).

The taxonomy (replay/failure_taxonomy.py) reads these flags but nothing produced them, so coverage was
0.0 on every real-records run. This pins the producer: a typed cause -> the matching flag -> a non-
UNCLASSIFIED taxonomy verdict (coverage > 0); an absent / unmapped cause leaves the record byte-identical.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.record_logging import (
    _stamp_taxonomy_evidence,
)
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.failure_taxonomy import (
    FailureRootCause,
    classify_record,
)


def _evidence(outcome, *, reason_lists=(), verification_reason=None) -> dict:
    """Run the H0.2 derivation over a minimal duck-typed report and return the resulting extra bag."""
    attempts = tuple(SimpleNamespace(reasons=tuple(rs)) for rs in reason_lists)
    report = SimpleNamespace(outcome=outcome, pick_report=SimpleNamespace(attempts=attempts))
    extra: dict = {}
    if verification_reason is not None:
        extra["verification_reason"] = verification_reason
    _stamp_taxonomy_evidence(extra, report)
    return extra


def _classify(final_outcome: str, extra: dict) -> FailureRootCause:
    rec = GraspAttemptRecord.new(
        attempt_id="t", mode="auto", final_outcome=final_outcome, extra=extra
    )
    return classify_record(rec).primary


class TaxonomyEvidenceDerivationTests(unittest.TestCase):
    """The typed-verdict -> evidence-flag mapping (no numeric thresholds)."""

    def test_collision_reason(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            reason_lists=[(GraspFailureReason.ALL_COLLIDED,)],
        )
        self.assertEqual(ev, {"collision_evidence": True})

    def test_heavy_occlusion_reason(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            reason_lists=[(GraspFailureReason.HEAVY_OCCLUSION,)],
        )
        self.assertEqual(ev, {"occlusion_misread_evidence": True})

    def test_deformable_reason(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            reason_lists=[(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,)],
        )
        self.assertEqual(ev, {"deformable_misclass_evidence": True})

    def test_empty_air_verification_reason(self) -> None:
        for reason in ("jaws_collapsed_to_minimum", "gripper_object_not_detected"):
            with self.subTest(reason=reason):
                ev = _evidence(
                    AutonomousGraspOutcome.VERIFICATION_FAILED,
                    verification_reason=reason,
                )
                self.assertEqual(ev, {"verification_reason": reason, "empty_air_evidence": True})

    def test_slip_verification_reason(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.VERIFICATION_FAILED,
            verification_reason="target_still_visible",
        )
        self.assertEqual(ev, {"verification_reason": "target_still_visible", "slip_evidence": True})

    def test_reasons_collected_across_attempt_history(self) -> None:
        # occlusion on an early attempt + collision on a later one -> both flagged.
        ev = _evidence(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            reason_lists=[(GraspFailureReason.HEAVY_OCCLUSION,), (GraspFailureReason.ALL_COLLIDED,)],
        )
        self.assertEqual(ev, {"occlusion_misread_evidence": True, "collision_evidence": True})

    # --- byte-identical (no flag stamped) cases ---

    def test_succeeded_record_carries_no_flag(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.SUCCEEDED,
            reason_lists=[(GraspFailureReason.ALL_COLLIDED,)],
        )
        self.assertEqual(ev, {})

    def test_unmapped_reason_is_byte_identical(self) -> None:
        ev = _evidence(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            reason_lists=[(GraspFailureReason.IK_FAILED,)],
        )
        self.assertEqual(ev, {})

    def test_no_reasons_no_verification_is_byte_identical(self) -> None:
        ev = _evidence(AutonomousGraspOutcome.EXECUTION_FAILED)
        self.assertEqual(ev, {})


class TaxonomyEvidenceClassifiesTests(unittest.TestCase):
    """End-to-end: the produced flag drives a real (non-UNCLASSIFIED) taxonomy verdict -> coverage > 0."""

    def test_collision_classifies(self) -> None:
        self.assertEqual(
            _classify("no_valid_grasp", {"collision_evidence": True}),
            FailureRootCause.COLLISION_REJECTION,
        )

    def test_occlusion_classifies(self) -> None:
        self.assertEqual(
            _classify("no_valid_grasp", {"occlusion_misread_evidence": True}),
            FailureRootCause.OCCLUSION_MISREAD,
        )

    def test_deformable_classifies(self) -> None:
        self.assertEqual(
            _classify("no_valid_grasp", {"deformable_misclass_evidence": True}),
            FailureRootCause.DEFORMABLE_MISCLASSIFICATION,
        )

    def test_empty_air_classifies(self) -> None:
        self.assertEqual(
            _classify("verification_failed", {"empty_air_evidence": True}),
            FailureRootCause.EMPTY_AIR_GRASP,
        )

    def test_slip_classifies(self) -> None:
        self.assertEqual(
            _classify("verification_failed", {"slip_evidence": True}),
            FailureRootCause.SLIP_AFTER_GRASP,
        )

    def test_no_flag_stays_unclassified(self) -> None:
        self.assertEqual(
            _classify("no_valid_grasp", {}),
            FailureRootCause.UNCLASSIFIED,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
