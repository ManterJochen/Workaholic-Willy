"""Phase T7 — RED tests for the telemetry completeness catalog.

Q6=A locks a frozen catalog that maps every supported
``final_outcome`` value to the set of required ``GraspAttemptRecord``
fields (or ``extra`` keys). Any record that fails the audit is an
"incident-grade" telemetry gap.

Contract:

* ``TELEMETRY_CATALOG`` is a ``Mapping[str, frozenset[str]]`` keyed by
  ``final_outcome`` strings. Every value in
  ``AutonomousGraspOutcome`` is a key.
* ``audit_record(record)`` returns a tuple of missing field names
  (empty tuple == record passes).
* ``audit_records(records)`` returns a list of
  ``(attempt_id, missing_fields_tuple)`` for every offending record.

Required-everywhere fields are always validated even if a specific
outcome has no extra requirements.
"""

from __future__ import annotations

import unittest

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_record,
    audit_records,
)


def _r(
    outcome: str,
    *,
    mode: str = "auto",
    execution: dict | None = None,
    verification: dict | None = None,
    recovery_actions: tuple = (),
    extra: dict | None = None,
) -> GraspAttemptRecord:
    return GraspAttemptRecord.new(
        attempt_id=f"a-{outcome}",
        mode=mode,
        final_outcome=outcome,
        execution=execution,
        verification=verification,
        recovery_actions=recovery_actions,
        extra=extra or {},
    )


class CatalogShapeTests(unittest.TestCase):
    def test_catalog_covers_every_outcome(self) -> None:
        missing = [
            v.value
            for v in AutonomousGraspOutcome
            if v.value not in TELEMETRY_CATALOG
        ]
        self.assertEqual(
            missing, [], f"catalog missing outcome keys: {missing!r}"
        )

    def test_catalog_is_frozen(self) -> None:
        # Every required-field set must be a frozenset (immutable).
        for outcome, required in TELEMETRY_CATALOG.items():
            self.assertIsInstance(
                required,
                frozenset,
                f"required set for {outcome!r} is not frozen",
            )


class AuditRecordTests(unittest.TestCase):
    def test_succeeded_requires_execution_and_verification(self) -> None:
        # Missing both -> reported.
        rec = _r("succeeded")
        missing = audit_record(rec)
        self.assertIn("execution", missing)
        self.assertIn("verification", missing)

    def test_succeeded_with_execution_and_verification_passes(self) -> None:
        rec = _r(
            "succeeded",
            execution={"outcome": "executed"},
            verification={"outcome": "passed"},
        )
        self.assertEqual(audit_record(rec), ())

    def test_decision_fail_closed_requires_decision_reason_code(self) -> None:
        rec = _r("decision_fail_closed", extra={"uncertainty_score": 0.6})
        missing = audit_record(rec)
        self.assertIn("extra.decision_reason_code", missing)

    def test_decision_fail_closed_with_reason_passes(self) -> None:
        rec = _r(
            "decision_fail_closed",
            extra={
                "decision_reason_code": "uncertainty_above_threshold",
                "uncertainty_score": 0.62,
                "threshold_used": 0.4,
            },
        )
        self.assertEqual(audit_record(rec), ())

    def test_recovery_exhausted_requires_recovery_actions(self) -> None:
        rec = _r("recovery_exhausted")
        self.assertIn("recovery_actions", audit_record(rec))

    def test_audit_records_returns_offenders_only(self) -> None:
        good = GraspAttemptRecord.new(
            attempt_id="a-good-succeeded",
            mode="auto",
            final_outcome="succeeded",
            execution={"outcome": "executed"},
            verification={"outcome": "passed"},
        )
        bad = GraspAttemptRecord.new(
            attempt_id="a-bad-succeeded",
            mode="auto",
            final_outcome="succeeded",
        )  # missing exec + verif
        offenders = audit_records((good, bad))
        ids = [aid for aid, _ in offenders]
        self.assertNotIn(good.attempt_id, ids)
        self.assertIn(bad.attempt_id, ids)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
