"""
Tests for :class:`SafetyDecision` and :class:`SafetyReason`.

Covers the contract:

* the SafetyReason -> MotionStatus mapping is exhaustive for every
  non-``OK`` reason;
* the constructors enforce the accepted/reason invariant;
* the ``motion_status`` property routes correctly through the default
  table and the per-decision override;
* ``unavailable`` decisions are *rejections*, not acceptances.
"""

from __future__ import annotations

import unittest

from src.robot.core import MotionStatus
from src.robot.safety.decision import (
    SafetyDecision,
    SafetyReason,
    safety_reason_to_motion_status,
)


class SafetyReasonMappingTests(unittest.TestCase):
    def test_every_non_ok_reason_maps_to_a_motion_status(self) -> None:
        for reason in SafetyReason:
            if reason is SafetyReason.OK:
                continue
            status = safety_reason_to_motion_status(reason)
            self.assertIsInstance(status, MotionStatus)

    def test_canonical_mapping(self) -> None:
        # Lock the canonical mapping so future contract changes show up
        # as test failures rather than silent behaviour drift.
        expected = {
            SafetyReason.WORKSPACE: MotionStatus.WORKSPACE_REJECTED,
            SafetyReason.JOINT_LIMIT: MotionStatus.JOINT_LIMIT_REJECTED,
            SafetyReason.IK_QUALITY: MotionStatus.IK_QUALITY_REJECTED,
            SafetyReason.SELF_COLLISION: MotionStatus.SELF_COLLISION_REJECTED,
            SafetyReason.PAYLOAD: MotionStatus.PAYLOAD_REJECTED,
            SafetyReason.CONTINUITY: MotionStatus.CONTINUITY_REJECTED,
            SafetyReason.UNAVAILABLE: MotionStatus.CONTROLLER_REJECTED,
        }
        for reason, status in expected.items():
            self.assertIs(safety_reason_to_motion_status(reason), status)

    def test_ok_has_no_mapping(self) -> None:
        with self.assertRaises(ValueError):
            safety_reason_to_motion_status(SafetyReason.OK)


class SafetyDecisionAcceptTests(unittest.TestCase):
    def test_accept_minimal(self) -> None:
        d = SafetyDecision.accept("workspace")
        self.assertTrue(d.accepted)
        self.assertFalse(d.rejected)
        self.assertIs(d.reason, SafetyReason.OK)
        self.assertEqual(d.guard, "workspace")
        self.assertEqual(d.message, "")
        self.assertEqual(d.detail, {})
        self.assertIsNone(d.motion_status)
        # Boolean coercion mirrors ``accepted``.
        self.assertTrue(bool(d))

    def test_accept_with_message(self) -> None:
        d = SafetyDecision.accept("workspace", message="inside box")
        self.assertEqual(d.message, "inside box")

    def test_accept_rejects_empty_guard_name(self) -> None:
        with self.assertRaises(ValueError):
            SafetyDecision.accept("")


class SafetyDecisionRejectTests(unittest.TestCase):
    def test_reject_minimal(self) -> None:
        d = SafetyDecision.reject(
            "workspace", SafetyReason.WORKSPACE,
            message="pose outside box",
            detail={"x_mm": "1000.0"},
        )
        self.assertFalse(d.accepted)
        self.assertTrue(d.rejected)
        self.assertIs(d.reason, SafetyReason.WORKSPACE)
        self.assertEqual(d.detail, {"x_mm": "1000.0"})
        self.assertIs(d.motion_status, MotionStatus.WORKSPACE_REJECTED)

    def test_reject_with_status_override(self) -> None:
        # E.g. an IK-quality guard that wants to surface IK_FAILED
        # instead of the default IK_QUALITY_REJECTED for the NaN case.
        d = SafetyDecision.reject(
            "ik_quality", SafetyReason.IK_QUALITY,
            message="NaN solution",
            motion_status_override=MotionStatus.IK_FAILED,
        )
        self.assertIs(d.motion_status, MotionStatus.IK_FAILED)

    def test_reject_forbids_ok_reason(self) -> None:
        with self.assertRaises(ValueError):
            SafetyDecision.reject("workspace", SafetyReason.OK)


class SafetyDecisionUnavailableTests(unittest.TestCase):
    def test_unavailable_is_a_rejection(self) -> None:
        d = SafetyDecision.unavailable(
            "self_collision",
            message="mesh_dir empty",
        )
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.UNAVAILABLE)
        # Default override falls back to CONTROLLER_REJECTED.
        self.assertIs(d.motion_status, MotionStatus.CONTROLLER_REJECTED)

    def test_unavailable_with_explicit_status(self) -> None:
        d = SafetyDecision.unavailable(
            "self_collision",
            motion_status_override=MotionStatus.SELF_COLLISION_REJECTED,
        )
        self.assertIs(d.motion_status, MotionStatus.SELF_COLLISION_REJECTED)


class SafetyDecisionInvariantTests(unittest.TestCase):
    def test_accepted_requires_ok_reason(self) -> None:
        with self.assertRaises(ValueError):
            SafetyDecision(
                accepted=True,
                reason=SafetyReason.WORKSPACE,
                guard="workspace",
            )

    def test_rejected_forbids_ok_reason(self) -> None:
        with self.assertRaises(ValueError):
            SafetyDecision(
                accepted=False,
                reason=SafetyReason.OK,
                guard="workspace",
            )

    def test_decisions_are_frozen(self) -> None:
        d = SafetyDecision.accept("workspace")
        with self.assertRaises(Exception):  # FrozenInstanceError
            d.guard = "other"  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
