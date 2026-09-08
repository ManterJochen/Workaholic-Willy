"""R1.1a — characterization for the two CROSS-CALL latches in GraspCalculator that the R1.0 golden CANNOT see.

The R1.0 golden builds a FRESH calculator per scenario and runs ONE compute() each, so it is structurally
blind to behavior that spans consecutive compute() calls. These tests pin the two such contracts that the
R1.1 decomposition (the GraspCandidateGenerator + SharedGeometryUtil extraction) must preserve byte-identically:

  * the dense runtime-budget COOLDOWN — an overrun in one compute() suppresses auto-dense in the NEXT call
    (``_dense_last_overran``, written in the facade's dense-budget block, read in the dense decision). The
    latch stays facade-owned in R1.1; the generator reads it via an injected getter.
  * the synthetic-intrinsics WARN-ONCE-EVER latch (``_warned_synthetic_K``) — ``camera_matrix=None`` warns
    exactly once across the calculator's lifetime. A duplicated SharedGeometryUtil instance would double-warn
    on hardware, so R1.1c must inject ONE shared instance. (Asserted via the observable WARNING, not the
    private latch, which legitimately moves into SharedGeometryUtil.)
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.robot.grasping import GraspCalculator


def _camera_matrix() -> np.ndarray:
    return np.array([[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _box_seg(shape: tuple[int, int] = (24, 24)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return SimpleNamespace(mask=mask, label="box")


class DenseCooldownLatchTests(unittest.TestCase):
    """An overrun in compute() N must suppress auto-dense in compute() N+1 (the cross-call cooldown)."""

    def test_overrun_suppresses_next_auto_dense(self) -> None:
        calc = GraspCalculator(
            min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=3,
            camera_matrix=_camera_matrix(),
        )
        seg = _box_seg()
        depth = np.full((24, 24), 1000.0, dtype=np.float64)

        # Call 1: force the dense block to overrun its runtime budget by making every monotonic() read jump
        # 1000 s, so elapsed >> dense_runtime_budget_ms. dense_sampling=True enters the dense block.
        ticker = [0.0]

        def _fake_monotonic() -> float:
            ticker[0] += 1000.0
            return ticker[0]

        with mock.patch("time.monotonic", _fake_monotonic):
            calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm", dense_sampling=True)
        self.assertTrue(calc._dense_last_overran, "an overrun must set the degrade latch")
        self.assertTrue(calc.last_telemetry.get("dense_timeout"), "the overrun must stamp dense_timeout=True")

        # Call 2 (auto): the latch must suppress dense AND short-circuit BEFORE the non-convexity
        # probe (so mask_non_convexity is never stamped) — that absence is the proof the READ fired.
        calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm", dense_sampling=None)
        self.assertEqual(calc.last_telemetry.get("dense_auto_reason"), "degraded_after_overrun")
        self.assertFalse(calc.last_telemetry.get("dense_decision"))
        self.assertNotIn(
            "mask_non_convexity", calc.last_telemetry,
            "the latch must short-circuit before the non-convexity probe stamps its key",
        )

        # Call 3, and this is the part the name promised and the code never did. A cooldown ends;
        # this does not. The latch is cleared inside the dense branch it suppresses, so nothing that
        # happens later reaches the reset, and the calculator stays on the silhouette path for the
        # rest of its life unless a caller asks for dense explicitly.
        calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm", dense_sampling=None)
        self.assertEqual(calc.last_telemetry.get("dense_auto_reason"), "degraded_after_overrun")
        self.assertTrue(calc._dense_last_overran, "there is no recovery, and the name now says so")


class SyntheticIntrinsicsWarnOnceTests(unittest.TestCase):
    """camera_matrix=None must warn exactly once across the calculator's lifetime (the warn-once latch)."""

    def test_warns_exactly_once_across_calls(self) -> None:
        calc = GraspCalculator(
            min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=3,
            camera_matrix=None,  # no K -> synthesized intrinsics -> the loud-once warning
        )
        seg = _box_seg()
        depth = np.full((24, 24), 1000.0, dtype=np.float64)
        # pixel_to_mm=None so BOTH _intrinsics AND _estimate_pixel_to_mm run (two warn call-sites, one latch).
        with self.assertLogs("GraspCalculator", level="WARNING") as cm:
            calc.compute(seg, depth, unit="mm")
            calc.compute(seg, depth, unit="mm")
        synth = [r for r in cm.records if "synthesizing intrinsics" in r.getMessage()]
        self.assertEqual(
            len(synth), 1,
            "the synthetic-intrinsics warning must fire exactly once across calls (the warn-once latch)",
        )


if __name__ == "__main__":
    unittest.main()
