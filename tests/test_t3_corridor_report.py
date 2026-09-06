"""Phase T3 — CorridorReport carrier validation.

The :class:`CorridorReport` is the typed output of the directional
free-space analyzer in ``backend.src.robot.grasping.scoring.corridor``.
T3 design intent (operator-locked, see Q&A in this session):

* T3 is *demote-only* by default; hard reject is gated behind a
  separate flag (off by default). The report must therefore carry
  enough information for downstream code to make a soft-or-hard
  decision without re-inspecting the depth map.
* The four canonical modes are :data:`CLEAR`, :data:`PARTIAL`,
  :data:`BLOCKED`, :data:`SKIPPED`. ``SKIPPED`` is emitted when a
  required input (depth map, intrinsics) is missing, so the
  scaffolding can be wired through the calculator before the
  perception stack publishes the data.
* Both clearance fields are non-negative finite millimetres in
  ``[0, max_distance_mm]``; ``blockage_confidence`` is finite in
  ``[0, 1]``.
* The carrier is frozen+slotted so per-pose telemetry never mutates
  in-place during ranking.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.scoring.corridor import (
    CorridorMode,
    CorridorReport,
)


class CorridorReportConstructionTests(unittest.TestCase):
    def test_minimal_clear_report(self) -> None:
        r = CorridorReport(
            approach_clearance_mm=200.0,
            retreat_clearance_mm=200.0,
            blockage_confidence=0.0,
            mode=CorridorMode.CLEAR,
        )
        self.assertEqual(r.mode, CorridorMode.CLEAR)
        self.assertEqual(r.approach_clearance_mm, 200.0)
        self.assertEqual(r.retreat_clearance_mm, 200.0)
        self.assertEqual(r.blockage_confidence, 0.0)
        # Derived convenience flag.
        self.assertFalse(r.blocked)

    def test_blocked_report_sets_derived_flag(self) -> None:
        r = CorridorReport(
            approach_clearance_mm=0.0,
            retreat_clearance_mm=0.0,
            blockage_confidence=0.95,
            mode=CorridorMode.BLOCKED,
        )
        self.assertTrue(r.blocked)
        self.assertEqual(r.mode, CorridorMode.BLOCKED)

    def test_skipped_report_neutral_confidence(self) -> None:
        # SKIPPED means analyzer had no depth/intrinsics input — it
        # must carry the neutral 0.5 confidence so the feasibility
        # subscore collapses to the neutral floor (matches T2's
        # treatment of missing inputs).
        r = CorridorReport(
            approach_clearance_mm=0.0,
            retreat_clearance_mm=0.0,
            blockage_confidence=0.5,
            mode=CorridorMode.SKIPPED,
        )
        self.assertEqual(r.mode, CorridorMode.SKIPPED)
        self.assertFalse(r.blocked)


class CorridorReportValidationTests(unittest.TestCase):
    def test_negative_clearance_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=-1.0,
                retreat_clearance_mm=0.0,
                blockage_confidence=0.5,
                mode=CorridorMode.PARTIAL,
            )
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=0.0,
                retreat_clearance_mm=-0.0001,
                blockage_confidence=0.5,
                mode=CorridorMode.PARTIAL,
            )

    def test_non_finite_clearance_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=float("inf"),
                retreat_clearance_mm=0.0,
                blockage_confidence=0.5,
                mode=CorridorMode.PARTIAL,
            )
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=0.0,
                retreat_clearance_mm=float("nan"),
                blockage_confidence=0.5,
                mode=CorridorMode.PARTIAL,
            )

    def test_confidence_outside_unit_interval_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=0.0,
                retreat_clearance_mm=0.0,
                blockage_confidence=-0.01,
                mode=CorridorMode.PARTIAL,
            )
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=0.0,
                retreat_clearance_mm=0.0,
                blockage_confidence=1.01,
                mode=CorridorMode.PARTIAL,
            )

    def test_non_finite_confidence_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorridorReport(
                approach_clearance_mm=0.0,
                retreat_clearance_mm=0.0,
                blockage_confidence=float("nan"),
                mode=CorridorMode.PARTIAL,
            )

    def test_blocked_flag_consistent_with_mode(self) -> None:
        # ``blocked`` is derived from ``mode == BLOCKED`` so it can
        # never disagree.
        for mode, expected in (
            (CorridorMode.CLEAR, False),
            (CorridorMode.PARTIAL, False),
            (CorridorMode.BLOCKED, True),
            (CorridorMode.SKIPPED, False),
        ):
            with self.subTest(mode=mode):
                r = CorridorReport(
                    approach_clearance_mm=0.0,
                    retreat_clearance_mm=0.0,
                    blockage_confidence=0.5,
                    mode=mode,
                )
                self.assertEqual(r.blocked, expected)


class CorridorReportImmutabilityTests(unittest.TestCase):
    def test_is_frozen(self) -> None:
        r = CorridorReport(
            approach_clearance_mm=10.0,
            retreat_clearance_mm=10.0,
            blockage_confidence=0.2,
            mode=CorridorMode.CLEAR,
        )
        # Python 3.11 frozen+slots dataclasses raise either TypeError
        # (slots) or AttributeError (frozen) depending on resolution
        # order — accept both like the IKQualityMetrics test.
        with self.assertRaises((AttributeError, TypeError)):
            r.approach_clearance_mm = 1.0  # type: ignore[misc]


class CorridorModeEnumTests(unittest.TestCase):
    def test_stable_string_values(self) -> None:
        # Telemetry consumers persist these strings — keep stable.
        self.assertEqual(CorridorMode.CLEAR.value, "clear")
        self.assertEqual(CorridorMode.PARTIAL.value, "partial")
        self.assertEqual(CorridorMode.BLOCKED.value, "blocked")
        self.assertEqual(CorridorMode.SKIPPED.value, "skipped")


if __name__ == "__main__":
    unittest.main()
