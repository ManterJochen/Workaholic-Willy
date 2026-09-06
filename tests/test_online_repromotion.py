"""Tier-0 re-promotion / snapshot cadence — the pure freshness function + config validation.

No Isaac, no numpy: the cadence is integer arithmetic over update counts. Verifies the three status
transitions (current -> snapshot_due -> repromotion_required), staleness dominance, and that every
decision keeps ``still_shadow=True`` (a stale verdict asks for a fresh promotion, never activates).
"""

from __future__ import annotations

import unittest

from src.robot.grasping.rl.online_repromotion import (
    REPROMOTION_STATUS_CURRENT,
    REPROMOTION_STATUS_SNAPSHOT_DUE,
    REPROMOTION_STATUS_STALE,
    RepromotionCadence,
    RepromotionCadenceError,
    evaluate_repromotion_cadence,
)


class CadenceConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = RepromotionCadence()
        self.assertEqual(c.snapshot_every_n, 100)
        self.assertEqual(c.stale_after_n, 250)

    def test_rejects_nonpositive_snapshot(self) -> None:
        with self.assertRaises(RepromotionCadenceError):
            RepromotionCadence(snapshot_every_n=0)

    def test_rejects_nonpositive_stale(self) -> None:
        with self.assertRaises(RepromotionCadenceError):
            RepromotionCadence(stale_after_n=0)

    def test_rejects_snapshot_gt_stale(self) -> None:
        with self.assertRaises(RepromotionCadenceError):
            RepromotionCadence(snapshot_every_n=300, stale_after_n=250)


class CadenceDecisionTests(unittest.TestCase):
    def _cad(self) -> RepromotionCadence:
        return RepromotionCadence(snapshot_every_n=10, stale_after_n=25)

    def test_zero_updates_is_current(self) -> None:
        d = evaluate_repromotion_cadence(family="recovery", n_updates=0, cadence=self._cad())
        self.assertEqual(d.status, REPROMOTION_STATUS_CURRENT)
        self.assertFalse(d.snapshot_due)
        self.assertFalse(d.repromotion_required)
        self.assertTrue(d.still_shadow)

    def test_snapshot_due_below_stale(self) -> None:
        d = evaluate_repromotion_cadence(family="recovery", n_updates=10, cadence=self._cad())
        self.assertEqual(d.status, REPROMOTION_STATUS_SNAPSHOT_DUE)
        self.assertTrue(d.snapshot_due)
        self.assertFalse(d.repromotion_required)

    def test_stale_dominates_snapshot(self) -> None:
        d = evaluate_repromotion_cadence(family="recovery", n_updates=25, cadence=self._cad())
        self.assertEqual(d.status, REPROMOTION_STATUS_STALE)
        self.assertTrue(d.repromotion_required)
        # even though a snapshot is also due, staleness wins the status.
        self.assertTrue(d.snapshot_due)
        self.assertTrue(d.still_shadow)

    def test_baselines_offset_the_counts(self) -> None:
        # 30 total updates but the last promotion + snapshot were at 28 => only 2 since each => current.
        d = evaluate_repromotion_cadence(
            family="recovery",
            n_updates=30,
            last_promoted_n_updates=28,
            last_snapshot_n_updates=28,
            cadence=self._cad(),
        )
        self.assertEqual(d.status, REPROMOTION_STATUS_CURRENT)
        self.assertEqual(d.updates_since_promotion, 2)
        self.assertEqual(d.updates_since_snapshot, 2)

    def test_negative_updates_rejected(self) -> None:
        with self.assertRaises(RepromotionCadenceError):
            evaluate_repromotion_cadence(family="recovery", n_updates=-1)


if __name__ == "__main__":
    unittest.main()
