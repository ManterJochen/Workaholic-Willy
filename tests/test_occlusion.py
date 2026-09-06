"""Phase O: occlusion ratio + approach clearance tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping import (
    GraspCalculator,
    GraspFailureReason,
    approach_clearance_mm,
    occlusion_ratio,
)
from src.robot.grasping.geometry import CameraIntrinsics


def _rect_mask(H: int, W: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class OcclusionRatioTests(unittest.TestCase):
    def test_no_neighbours_returns_zero(self) -> None:
        mask = _rect_mask(100, 100, 30, 30, 70, 70)
        self.assertEqual(occlusion_ratio(mask, None), 0.0)
        self.assertEqual(occlusion_ratio(mask, []), 0.0)

    def test_empty_target_returns_zero(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        other = _rect_mask(50, 50, 0, 0, 50, 50)
        self.assertEqual(occlusion_ratio(mask, [other]), 0.0)

    def test_full_overlap_returns_one(self) -> None:
        target = _rect_mask(60, 60, 10, 10, 50, 50)
        other = _rect_mask(60, 60, 0, 0, 60, 60)
        self.assertAlmostEqual(occlusion_ratio(target, [other]), 1.0, places=4)

    def test_partial_overlap_lies_in_unit_interval(self) -> None:
        target = _rect_mask(100, 100, 20, 20, 80, 80)
        # Neighbour covers the right half of the target's hull.
        other = _rect_mask(100, 100, 50, 0, 100, 100)
        ratio = occlusion_ratio(target, [other])
        # Right half of a 60x60 square = ~0.5 of hull area.
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.6)

    def test_shape_mismatch_raises(self) -> None:
        target = _rect_mask(50, 50, 0, 0, 30, 30)
        bad = np.zeros((40, 40), dtype=bool)
        with self.assertRaises(ValueError):
            occlusion_ratio(target, [bad])


class ApproachClearanceTests(unittest.TestCase):
    def _intrinsics(self, W: int, H: int) -> CameraIntrinsics:
        return CameraIntrinsics(fx=600.0, fy=600.0, cx=W / 2.0, cy=H / 2.0)

    def test_free_path_returns_full_budget(self) -> None:
        H, W = 200, 200
        # Depth far behind the grasp point so no obstruction is found.
        depth = np.full((H, W), 1000.0, dtype=np.float32)
        K = self._intrinsics(W, H)
        point = np.array([0.0, 0.0, 600.0])
        axis = np.array([0.0, 0.0, 1.0])
        d = approach_clearance_mm(
            point, axis, depth, K, max_distance_mm=200.0, step_mm=5.0
        )
        self.assertAlmostEqual(d, 200.0, places=2)

    def test_obstruction_truncates_clearance(self) -> None:
        H, W = 200, 200
        # Surface sits 50 mm in front of the grasp point (closer to camera).
        depth = np.full((H, W), 550.0, dtype=np.float32)
        K = self._intrinsics(W, H)
        point = np.array([0.0, 0.0, 600.0])
        axis = np.array([0.0, 0.0, 1.0])
        d = approach_clearance_mm(
            point, axis, depth, K, max_distance_mm=200.0, step_mm=5.0
        )
        # We expect the march to stop near 50 mm (the depth at the
        # pixel matches z=550 mm).
        self.assertLess(d, 70.0)
        self.assertGreaterEqual(d, 0.0)

    def test_zero_axis_rejected(self) -> None:
        depth = np.zeros((10, 10), dtype=np.float32)
        K = self._intrinsics(10, 10)
        with self.assertRaises(ValueError):
            approach_clearance_mm(
                np.zeros(3), np.zeros(3), depth, K
            )

    def test_invalid_step_rejected(self) -> None:
        depth = np.zeros((10, 10), dtype=np.float32)
        K = self._intrinsics(10, 10)
        with self.assertRaises(ValueError):
            approach_clearance_mm(
                np.array([0, 0, 100.0]),
                np.array([0, 0, 1.0]),
                depth,
                K,
                step_mm=0.0,
            )


class CalculatorOcclusionGateTests(unittest.TestCase):
    """Confirm Phase O integration is strictly additive."""

    def test_default_constructor_leaves_gate_disabled(self) -> None:
        calc = GraspCalculator()
        self.assertIsNone(calc.heavy_occlusion_threshold)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraspCalculator(heavy_occlusion_threshold=-0.1)
        with self.assertRaises(ValueError):
            GraspCalculator(heavy_occlusion_threshold=1.5)
        with self.assertRaises(ValueError):
            GraspCalculator(heavy_occlusion_threshold=float("nan"))

    def test_threshold_in_range_accepted(self) -> None:
        calc = GraspCalculator(heavy_occlusion_threshold=0.6)
        self.assertAlmostEqual(calc.heavy_occlusion_threshold or 0.0, 0.6)

    def test_heavy_occlusion_short_circuits(self) -> None:
        # Build a tiny segmentation where the target is fully covered
        # by a neighbour and the gate is opted-in.
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        neighbour = _rect_mask(H, W, 0, 0, W, H)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array([[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]])

        class _Seg:
            mask = target
            label = "thing"
            confidence = 0.9

        calc = GraspCalculator(
            heavy_occlusion_threshold=0.5, camera_matrix=K
        )
        candidates = calc.compute(
            segmentation=_Seg(),
            depth_map=depth,
            other_object_masks=[neighbour],
        )
        self.assertEqual(candidates, [])
        reasons = calc.last_failure_reasons
        self.assertIn(GraspFailureReason.HEAVY_OCCLUSION, reasons)
        self.assertIn(
            GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED, reasons
        )
        self.assertGreater(calc.last_telemetry["occlusion_ratio"], 0.9)
        self.assertTrue(calc.last_telemetry.get("heavy_occlusion_rejected"))

    def test_occlusion_metric_reported_without_gate(self) -> None:
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        neighbour = _rect_mask(H, W, 0, 0, W, H)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array([[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]])

        class _Seg:
            mask = target
            label = "thing"
            confidence = 0.9

        # Default calculator: no gate, but metric must still appear.
        calc = GraspCalculator(camera_matrix=K)
        calc.compute(
            segmentation=_Seg(),
            depth_map=depth,
            other_object_masks=[neighbour],
        )
        # No HEAVY_OCCLUSION reason because gate is disabled.
        self.assertNotIn(
            GraspFailureReason.HEAVY_OCCLUSION, calc.last_failure_reasons
        )
        self.assertIn("occlusion_ratio", calc.last_telemetry)

    def test_no_metric_when_no_neighbours(self) -> None:
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array([[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]])

        class _Seg:
            mask = target
            label = "thing"
            confidence = 0.9

        calc = GraspCalculator(camera_matrix=K)
        calc.compute(segmentation=_Seg(), depth_map=depth)
        self.assertNotIn("occlusion_ratio", calc.last_telemetry)


if __name__ == "__main__":
    unittest.main()
