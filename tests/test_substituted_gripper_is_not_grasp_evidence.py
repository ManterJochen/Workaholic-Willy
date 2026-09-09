"""A gripper with no jaws cannot be evidence that something was grasped.

MEASURED on this tree: ``from_robot_config`` answers four impossible gripper configurations with a
working :class:`NullGripper` (``robot.vendor: dummy`` + ``gripper.vendor: robotiq`` is one of them,
and it is what ``Cell.rehearsal`` boots). That object accepts ``set_width_mm(5.0)`` and answers
``get_width_mm() -> 85.0``, the configured maximum, forever. Fed to
:class:`WidthDeltaGripperVerifier` that reads as jaws holding 85 mm of something, so the attempt
verifies and the pick reports SUCCEEDED with no end-effector on the flange at all.

These tests are about the VERDICT, not about the number. Whatever a jawless gripper answers, it is
not a measurement: there are no jaws and no encoder behind it. The refusal reads the object's own
``substitution`` record and its ``holds_nothing`` flag, so the failure is named for the empty flange
instead of being inferred from an opening that nothing opened.

Two ways to reach that empty flange, two reason strings, one verdict. A gripper that was ASKED FOR
and could not be built is ``no_end_effector_built``; a cell configured ``gripper.vendor: none`` is
``no_end_effector_configured`` (DECIDED 2026-09-09, inverting the pin the last test in this file used
to carry). Same physics, different thing for an operator to do.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.closed_loop.verification import (
    GraspVerificationContext,
    GraspVerificationPolicy,
    VerificationOutcome,
    WidthDeltaGripperVerifier,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grippers.null import (
    GripperSubstitution,
    NullGripper,
    SubstitutionReason,
)

#: The reason string a REFUSED-TO-BUILD gripper carries. Named for the physical fact rather than for
#: the symptom: an operator reading it should look at the flange, not at the width threshold.
_NO_END_EFFECTOR_BUILT = "no_end_effector_built"

#: The reason string a cell configured WITHOUT an end-effector carries. Same verdict, different
#: operator problem: nobody asked for a gripper here, so there is nothing to repair in the pairing of
#: arm and gripper vendor. Two strings on purpose, so a records rollup can tell the populations apart.
_NO_END_EFFECTOR_CONFIGURED = "no_end_effector_configured"


def _grasp(*, grip_width_mm: float = 40.0) -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=grip_width_mm,
        score=0.9,
        frame=GraspFrame.BASE,
        label="grasp",
    )


def _substituted_gripper() -> NullGripper:
    """What ``from_robot_config`` builds for ``gripper.vendor: robotiq`` on a non-UR arm.

    The widths are the shipped ``robot.yaml`` ones (5.0 policy floor, 85.0 Robotiq 2F-85 opening),
    because the whole point is that this object answers with THAT 85.0 and not with anything it
    was told to do.
    """

    return NullGripper(
        min_width_mm=5.0,
        max_width_mm=85.0,
        substitution=GripperSubstitution(
            reason=SubstitutionReason.ROBOTIQ_NEEDS_UR,
            requested="robotiq",
            detail="gripper.vendor='robotiq' but the arm vendor is 'dummy', not UR.",
            fix="On a real cell, set robot.vendor: ur.",
        ),
    )


def _context(
    gripper: object,
    *,
    post_close_width_mm: float | None,
    commanded_close_width_mm: float | None = 5.0,
    width_delta_max_mm: float | None = None,
) -> GraspVerificationContext:
    return GraspVerificationContext(
        grasp=_grasp(),
        policy=GraspVerificationPolicy(
            enabled=True,
            width_delta_min_mm=2.0,
            width_delta_max_mm=width_delta_max_mm,
        ),
        gripper=gripper,  # type: ignore[arg-type]
        pre_close_width_mm=85.0,
        post_close_width_mm=post_close_width_mm,
        commanded_close_width_mm=commanded_close_width_mm,
    )


class SubstitutedGripperIsNotGraspEvidenceTests(unittest.TestCase):
    def test_a_substituted_gripper_does_not_verify_a_grasp(self) -> None:
        """The measured chain, end to end: command a close, read the gripper back, verify."""
        gripper = _substituted_gripper()
        gripper.connect()
        gripper.set_width_mm(5.0, speed=None, force=None)
        # Read the gripper rather than hard-coding 85.0: the claim under test is that NO answer from
        # a substituted gripper is evidence, so this stays true whatever the readback becomes.
        readback = gripper.get_width_mm()

        report = WidthDeltaGripperVerifier().verify(
            _context(gripper, post_close_width_mm=readback)
        )

        self.assertIs(
            report.outcome,
            VerificationOutcome.FAILED,
            f"a gripper that does not exist reported {readback} mm and the verifier accepted it",
        )
        self.assertEqual(report.reason, _NO_END_EFFECTOR_BUILT)

    def test_the_refusal_names_which_gripper_was_refused(self) -> None:
        """FAILED alone sends an operator to the width thresholds. The telemetry must say 'flange'."""
        report = WidthDeltaGripperVerifier().verify(
            _context(_substituted_gripper(), post_close_width_mm=85.0)
        )
        self.assertEqual(
            report.telemetry.get("substitution_reason"),
            str(SubstitutionReason.ROBOTIQ_NEEDS_UR),
        )
        self.assertEqual(report.telemetry.get("requested_gripper"), "robotiq")

    def test_the_refusal_survives_an_upper_bound_being_configured(self) -> None:
        """With ``width_delta_max_mm`` set, today's arithmetic already fails, for the wrong cause.

        MEASURED: post 85.0 > commanded 5.0 + 10.0 -> FAILED ``jaws_did_not_close_enough``. That is
        the right verdict from a false premise (jaws that never moved, because there are none), and
        it sends the operator to tune a threshold. The named cause must win over the arithmetic.
        """
        report = WidthDeltaGripperVerifier().verify(
            _context(
                _substituted_gripper(),
                post_close_width_mm=85.0,
                width_delta_max_mm=10.0,
            )
        )
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, _NO_END_EFFECTOR_BUILT)

    def test_the_refusal_does_not_need_a_width_sample(self) -> None:
        """No jaws is knowable without a readback, so it must not degrade to INCONCLUSIVE.

        INCONCLUSIVE is "I could not collect the evidence" and the operator's ``fail_closed: false``
        turns that back into a success. There is nothing inconclusive about an empty flange.
        """
        report = WidthDeltaGripperVerifier().verify(
            _context(_substituted_gripper(), post_close_width_mm=None)
        )
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, _NO_END_EFFECTOR_BUILT)


class RealGrippersAreStillJudgedByTheirJawsTests(unittest.TestCase):
    """The refusal is keyed on the two absence attributes and must not reach anything with jaws."""

    class _PlainGripper:
        """A gripper with jaws: no substitution record, no ``holds_nothing``. Every real driver,
        and the sim one."""

        min_width_mm = 0.0
        max_width_mm = 100.0

        def connect(self) -> None: ...
        def disconnect(self) -> None: ...
        def activate(self) -> None: ...
        def set_width_mm(self, width_mm, *, speed=None, force=None) -> None: ...  # noqa: ANN001
        def get_width_mm(self) -> float:
            return 39.0

    def test_a_gripper_with_jaws_still_passes_on_a_plausible_width(self) -> None:
        report = WidthDeltaGripperVerifier().verify(
            _context(
                self._PlainGripper(),
                post_close_width_mm=39.0,
                commanded_close_width_mm=39.0,
            )
        )
        self.assertIs(report.outcome, VerificationOutcome.PASSED)
        self.assertEqual(report.reason, "width_within_bounds")

    def test_a_gripper_with_jaws_still_fails_on_a_collapsed_width(self) -> None:
        report = WidthDeltaGripperVerifier().verify(
            _context(self._PlainGripper(), post_close_width_mm=1.0)
        )
        self.assertIs(report.outcome, VerificationOutcome.FAILED)
        self.assertEqual(report.reason, "jaws_collapsed_to_minimum")


class ADeliberatelyGripperLessCellDoesNotVerifyEitherTests(unittest.TestCase):
    """``gripper.vendor: none``: the same empty flange, reached by a different route."""

    def test_a_cell_with_no_end_effector_on_purpose_does_not_verify_a_grasp_either(
        self,
    ) -> None:
        """DECIDED 2026-09-09 by the repository owner. This test used to pin the opposite.

        The pinned reading, kept as the record of what was weighed: ``gripper.vendor: none`` builds
        the same jawless object with ``substitution=None``, answers the same 85.0 mm, and this
        verifier read that as a held object, so the pick reported SUCCEEDED. It was left alone
        because a motion rehearsal wants exactly that, and whether a deliberately gripper-less cell
        may report a successful pick was called a product decision rather than a defect.

        The decision went the other way. There are no jaws, so nothing was held, and that holds
        whether or not anybody wanted a gripper. A rehearsal that wants motion without a grasp
        verdict switches verification off (``robot.grasping.verification.enabled: false``, which is
        the shipped default) instead of being handed a fabricated pass.
        """
        deliberate = NullGripper(min_width_mm=5.0, max_width_mm=85.0)
        readback = deliberate.get_width_mm()

        report = WidthDeltaGripperVerifier().verify(
            _context(deliberate, post_close_width_mm=readback)
        )

        self.assertIs(
            report.outcome,
            VerificationOutcome.FAILED,
            f"a cell with no end-effector reported {readback} mm and the verifier accepted it",
        )
        self.assertEqual(report.reason, _NO_END_EFFECTOR_CONFIGURED)

    def test_the_two_empty_flanges_stay_two_reasons(self) -> None:
        """Same verdict, two different things for an operator to do.

        A substitution means a gripper was ASKED FOR and the arm could not carry it: the repair is in
        the config, and the record has to name what was requested. ``gripper.vendor: none`` means
        nobody asked: the repair is to fit and configure one, or to stop verifying. Collapsing both
        into one reason string would make a records rollup unable to tell those populations apart.
        """
        built = WidthDeltaGripperVerifier().verify(
            _context(_substituted_gripper(), post_close_width_mm=85.0)
        )
        configured = WidthDeltaGripperVerifier().verify(
            _context(
                NullGripper(min_width_mm=5.0, max_width_mm=85.0),
                post_close_width_mm=85.0,
            )
        )

        self.assertIs(built.outcome, configured.outcome)
        self.assertNotEqual(built.reason, configured.reason)
        # The requested-gripper keys belong to the substitution alone: there is no requested vendor
        # to name when nobody requested one.
        self.assertNotIn("requested_gripper", configured.telemetry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
