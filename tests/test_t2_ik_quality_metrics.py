"""Phase T2 — typed IK quality metrics on :class:`IKResult`.

The grasping ranker needs a *standardised* surface for IK quality so
the new :mod:`feasibility_score` module can read uniform fields across
every :class:`IKService` implementation. The safety layer already
computes these metrics internally
(:class:`backend.src.robot.safety.ik_quality.IKQualityGuard`) but only
emits them as motion-time rejections, not as ranking inputs.

This test module pins:

1. The shape of the new ``IKQualityMetrics`` dataclass (frozen,
   slotted, every field independently optional so legacy adapters
   that compute only a subset can still emit a partial metric).
2. That ``IKResult`` carries an optional ``quality`` slot defaulting
   to :data:`None`.
3. Backwards compatibility: existing
   ``IKResult(reachable=...)`` construction continues to work
   unchanged.

The metrics themselves are intentionally *not* computed in this
module — the IK adapter (or the safety layer) computes them and
stamps them onto the returned :class:`IKResult`. T2 only standardises
the *carrier*.
"""

from __future__ import annotations

import math
import unittest

from src.robot.grasping.planning.reachability import (
    IKQualityMetrics,
    IKResult,
)


class IKQualityMetricsShapeTests(unittest.TestCase):
    """Locks the frozen / slotted / optional-field contract."""

    def test_defaults_are_all_none(self) -> None:
        m = IKQualityMetrics()
        self.assertIsNone(m.condition_number)
        self.assertIsNone(m.min_singular_value)
        self.assertIsNone(m.joint_margin_deg)

    def test_partial_population_allowed(self) -> None:
        # Adapters that only know one signal must still be able to
        # emit it without manufacturing the others.
        m = IKQualityMetrics(condition_number=42.0)
        self.assertEqual(m.condition_number, 42.0)
        self.assertIsNone(m.min_singular_value)
        self.assertIsNone(m.joint_margin_deg)

    def test_frozen(self) -> None:
        m = IKQualityMetrics(condition_number=10.0)
        with self.assertRaises(Exception):
            m.condition_number = 99.0  # type: ignore[misc]

    def test_slotted(self) -> None:
        m = IKQualityMetrics()
        # Python 3.11 frozen+slots dataclass raises TypeError when
        # setting an unknown attribute (cls captured by the
        # generated ``__setattr__`` predates the slotted rewrite).
        # The codebase-wide pattern accepts that quirk; what matters
        # for T2 is that unknown attrs cannot be set.
        with self.assertRaises((AttributeError, TypeError)):
            m.unexpected_attr = 1  # type: ignore[attr-defined]

    def test_condition_number_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            IKQualityMetrics(condition_number=-1.0)

    def test_min_singular_value_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            IKQualityMetrics(min_singular_value=-0.01)

    def test_joint_margin_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            IKQualityMetrics(joint_margin_deg=-0.5)

    def test_rejects_nan_or_inf(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                IKQualityMetrics(condition_number=bad)
            with self.assertRaises(ValueError):
                IKQualityMetrics(min_singular_value=bad)
            with self.assertRaises(ValueError):
                IKQualityMetrics(joint_margin_deg=bad)


class IKResultQualitySlotTests(unittest.TestCase):
    """Pins backwards compatibility + the new ``quality`` field."""

    def test_legacy_construction_unchanged(self) -> None:
        r = IKResult(reachable=True)
        self.assertTrue(r.reachable)
        self.assertIsNone(r.reason)
        self.assertIsNone(r.joints)
        self.assertIsNone(r.metadata)
        # New field defaults to None so legacy callers see no change.
        self.assertIsNone(r.quality)

    def test_quality_can_be_attached(self) -> None:
        q = IKQualityMetrics(
            condition_number=50.0,
            min_singular_value=0.02,
            joint_margin_deg=15.0,
        )
        r = IKResult(reachable=True, quality=q)
        self.assertIs(r.quality, q)

    def test_quality_rejected_for_unreachable_is_allowed(self) -> None:
        # Adapters may still emit quality metrics even when the pose
        # is unreachable (e.g. the IK converged but lands too close
        # to a joint limit). The dataclass must not constrain this —
        # it's a passive carrier.
        q = IKQualityMetrics(joint_margin_deg=0.5)
        r = IKResult(reachable=False, reason="joint_limit", quality=q)
        self.assertFalse(r.reachable)
        self.assertIs(r.quality, q)


if __name__ == "__main__":
    unittest.main()
