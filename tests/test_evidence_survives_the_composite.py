"""A verification verdict that reached the record through a composite still stamps its evidence flag.

⛔ **MEASURED 2026-09-09, AND THE COUNT IS ZERO.** `CompositeGraspVerifier` wraps every non-passing
child reason: `child_failed:{reason}` and `child_inconclusive:{reason}` are the only two verdicts it
can return, and `builders.py` wires the composite whenever verification is on and no verifier was
supplied. `_VERIFICATION_REASON_TO_EVIDENCE` did an exact lookup on the bare reason. So
`jaws_collapsed_to_minimum` arrived as `child_failed:jaws_collapsed_to_minimum`, matched nothing, and
`empty_air_evidence` was never stamped on a record produced by the shipped wiring.

⚠ **AND THE TEST THAT COVERED IT PASSED THE WHOLE TIME.** `tests/test_h0_2_taxonomy_evidence.py`
constructs the bare reason and hands it to the stamper directly, so the stamper and the string it
will actually receive never met. A test that builds its own input is measuring the half of the system
it did not build. That is why this file drives a real `CompositeGraspVerifier` over a real child
verifier and takes the reason and the telemetry it actually produces, rather than writing either one
down.

⭐ **THE FIX READS STRUCTURE RATHER THAN PARSING A STRING.** The composite already carries each child
report under `telemetry["children"]`, with its own `reason`, and `service.py` already copies the whole
verification telemetry into the record. So the stamper walks those children instead of splitting the
prefix off the wrapped reason, which keeps the prefix a private convention of the composite: change it
tomorrow and nothing here has to know.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.record_logging import (
    _stamp_taxonomy_evidence,
)
from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.grasping.closed_loop.verification import (
    CompositeGraspVerifier,
    GraspVerificationReport,
    VerificationOutcome,
)
from src.robot.grasping.replay.failure_taxonomy import (
    FailureRootCause,
    classify_record,
)
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    verification_metadata_from,
)


class _FixedVerifier:
    """A child that answers what it was told to, so the composite is the only thing under test.

    Deliberately not a mock of the composite's own internals: it satisfies the same call shape a real
    verifier does and returns a real `GraspVerificationReport`, so the wrapping the composite applies
    is the wrapping production applies.
    """

    def __init__(self, outcome: VerificationOutcome, reason: str) -> None:
        self._report = GraspVerificationReport(
            outcome=outcome, reason=reason, telemetry={"verifier": "fixed"}
        )

    def verify(self, context: object) -> GraspVerificationReport:
        del context
        return self._report


def _record_extra_from(verifier, *, outcome=AutonomousGraspOutcome.VERIFICATION_FAILED) -> dict:
    """Everything `service.py` puts in the record extra after a verification, and nothing else.

    The three keys below are copied from `_run_post_grasp_verification`: outcome, reason and the whole
    telemetry bag. Writing them here rather than importing the service keeps the test to one subject,
    and the shape is pinned by `test_the_service_still_stamps_these_three_keys` below.
    """
    report = verifier.verify(context=None)
    metadata = verification_metadata_from(report)
    assert metadata is not None
    extra: dict = {
        "verification_outcome": str(report.outcome),
        "verification_reason": metadata["reason"],
        "verification_telemetry": metadata["telemetry"],
    }
    _stamp_taxonomy_evidence(
        extra,
        SimpleNamespace(outcome=outcome, pick_report=SimpleNamespace(attempts=())),
    )
    return extra


class EvidenceSurvivesTheCompositeTests(unittest.TestCase):
    def test_a_collapsed_jaw_through_a_composite_still_stamps_empty_air(self) -> None:
        """The shipped wiring, end to end: composite in, evidence flag out."""
        composite = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum"),),
            rule="all_must_pass",
        )
        extra = _record_extra_from(composite)
        self.assertTrue(
            extra.get("empty_air_evidence"),
            f"the reason reached the record as {extra['verification_reason']!r} and stamped nothing",
        )

    def test_the_wrapped_reason_is_what_actually_arrives(self) -> None:
        """The premise, asserted before the conclusion rests on it.

        If the composite ever stops wrapping, this goes red and the test above starts passing for a
        different reason than it was written for. That distinction is the whole point of pinning it.
        """
        composite = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum"),),
            rule="all_must_pass",
        )
        extra = _record_extra_from(composite)
        self.assertEqual(
            extra["verification_reason"], "child_failed:jaws_collapsed_to_minimum"
        )

    def test_an_undetected_object_through_a_composite_stamps_it_too(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "gripper_object_not_detected"),),
            rule="all_must_pass",
        )
        self.assertTrue(_record_extra_from(composite).get("empty_air_evidence"))

    def test_a_slip_through_a_composite_stamps_slip(self) -> None:
        composite = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "target_still_visible"),),
            rule="all_must_pass",
        )
        self.assertTrue(_record_extra_from(composite).get("slip_evidence"))

    def test_the_inconclusive_wrapping_stamps_too(self) -> None:
        """The second wrapping form, which is reachable whenever `require_all_conclusive` is set.

        Both of the composite's non-passing returns prefix, so a fix that only understood
        `child_failed:` would leave this one dark. Measured: there are exactly two such returns.
        """
        composite = CompositeGraspVerifier(
            verifiers=(
                _FixedVerifier(VerificationOutcome.INCONCLUSIVE, "jaws_collapsed_to_minimum"),
            ),
            rule="all_must_pass",
            require_all_conclusive=True,
        )
        extra = _record_extra_from(composite)
        self.assertEqual(
            extra["verification_reason"], "child_inconclusive:jaws_collapsed_to_minimum"
        )
        self.assertTrue(extra.get("empty_air_evidence"))

    def test_the_flag_reaches_the_taxonomy(self) -> None:
        """The reason the flag exists at all: an offline run must classify the record.

        Without this the test above would pass on a flag nobody reads, which is the same shape as the
        defect it replaces.
        """
        composite = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum"),),
            rule="all_must_pass",
        )
        record = GraspAttemptRecord.new(
            attempt_id="t",
            mode="auto",
            final_outcome="verification_failed",
            extra=_record_extra_from(composite),
        )
        self.assertIsNot(classify_record(record), FailureRootCause.UNCLASSIFIED)


    def test_a_composite_inside_a_composite_still_reaches_the_child(self) -> None:
        """Nothing wires this today, and the docstring claims it works, so it is asserted.

        A claim about a case nobody exercises is the cheapest kind of stale prose: nobody notices
        when it stops being true, because nobody runs it.
        """
        inner = CompositeGraspVerifier(
            verifiers=(_FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum"),),
            rule="all_must_pass",
        )
        outer = CompositeGraspVerifier(verifiers=(inner,), rule="all_must_pass")
        extra = _record_extra_from(outer)
        self.assertEqual(
            extra["verification_reason"],
            "child_failed:child_failed:jaws_collapsed_to_minimum",
        )
        self.assertTrue(extra.get("empty_air_evidence"))


class ThisTestDoesNotDriftFromTheServiceTests(unittest.TestCase):
    """`_record_extra_from` above hand-copies three keys out of `_run_post_grasp_verification`.

    ⛔ THAT COPY IS ITSELF THE DEFECT SHAPE THIS FILE EXISTS FOR. If the service renames one of the
    three, every test above keeps passing against a record shape that production no longer produces,
    and the file that was written to catch a silent drift becomes an instance of one. So the three
    names are read out of the service's source rather than trusted.
    """

    def test_the_service_still_stamps_these_three_keys(self) -> None:
        import ast
        import pathlib as _pathlib

        source = (
            _pathlib.Path(__file__).resolve().parents[1]
            / "src/robot/execution/autonomous_grasp/service.py"
        ).read_text(encoding="utf-8")
        assigned = {
            node.slice.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "executed_telemetry"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        for key in ("verification_outcome", "verification_reason", "verification_telemetry"):
            with self.subTest(key=key):
                self.assertIn(key, assigned, f"{key} is no longer stamped by the service")


class TheUnwrappedPathIsUntouchedTests(unittest.TestCase):
    """A verifier used without a composite kept working the whole time and must keep working.

    This is the byte-identical half. The defect was that the wrapped path stamped nothing, never that
    the bare path stamped something wrong.
    """

    def test_a_bare_verifier_still_stamps(self) -> None:
        extra = _record_extra_from(
            _FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum")
        )
        self.assertEqual(extra["verification_reason"], "jaws_collapsed_to_minimum")
        self.assertTrue(extra.get("empty_air_evidence"))

    def test_a_reason_nothing_maps_stamps_nothing(self) -> None:
        extra = _record_extra_from(
            _FixedVerifier(VerificationOutcome.FAILED, "some_future_reason")
        )
        self.assertNotIn("empty_air_evidence", extra)
        self.assertNotIn("slip_evidence", extra)

    def test_a_succeeded_record_carries_no_flag(self) -> None:
        extra = _record_extra_from(
            _FixedVerifier(VerificationOutcome.FAILED, "jaws_collapsed_to_minimum"),
            outcome=AutonomousGraspOutcome.SUCCEEDED,
        )
        self.assertNotIn("empty_air_evidence", extra)


if __name__ == "__main__":
    unittest.main()
