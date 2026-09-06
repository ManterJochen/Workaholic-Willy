"""H1.1a — the silhouette top-referenced grasp depth + penetration option on ``GraspCalculator``.

The calculator referenced the silhouette CENTRE candidate to the mask's MEDIAN depth and applied no
penetration -- the grasp rested on the (median) surface. This adds a default-off ``grasp_depth_reference``
(``"centre"``|``"top"``) + ``grasp_top_penetration_mm`` so a caller can reference the object TOP (a near-
surface depth quantile, no ground-truth needed -- unlike the sim's depth-map override) and descend the
fingers below the surface. Defaults (``"centre"``, ``0.0``) reproduce the historical depth EXACTLY (the R1/R2
goldens are untouched); ``"top"`` only diverges on a non-flat depth distribution.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping import GraspCalculator


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class GraspDepthReferenceTests(unittest.TestCase):
    """The reference math + the penetration descend (default-off byte-identical)."""

    @staticmethod
    def _calc(**kwargs: object) -> GraspCalculator:
        return GraspCalculator(
            min_grip_width_mm=1.0,
            max_grip_width_mm=200.0,
            max_candidates=6,
            camera_matrix=_camera_matrix(),
            **kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _gradient_scene() -> tuple[SimpleNamespace, np.ndarray]:
        # A mask over a depth GRADIENT: nearer (object TOP, ~1000 mm) on the top rows, deeper toward the
        # bottom (~1180 mm). The median sits mid-object; the 10th-percentile near-top is clearly smaller, so
        # "centre" and "top" diverge.
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 6:18] = 1
        depth = np.zeros((24, 24), dtype=np.float64)
        for r in range(24):
            depth[r, :] = 1000.0 + max(0.0, (r - 7)) * 20.0
        seg = SimpleNamespace(mask=mask, label="box")
        return seg, depth

    def _candidate_zs(self, calc: GraspCalculator) -> list[float]:
        seg, depth = self._gradient_scene()
        grasps = calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        self.assertTrue(grasps)
        return sorted(float(g.position[2]) for g in grasps)

    # ---- the reference math (pure) ----

    def test_centre_reference_returns_the_median_unchanged(self) -> None:
        # "centre" is byte-identical: the exact median value flows through untouched.
        calc = self._calc()  # default reference="centre"
        valid = np.array([1000.0, 1000.0, 1000.0, 1100.0, 1200.0])
        self.assertEqual(calc._resolve_grasp_depth(valid, 1234.5, 1.0), 1234.5)

    def test_top_reference_returns_the_near_surface_quantile(self) -> None:
        calc = self._calc(grasp_depth_reference="top", grasp_top_quantile=10.0)
        valid = np.array([1000.0, 1010.0, 1020.0, 1100.0, 1200.0])
        expected = float(np.percentile(valid, 10.0))
        self.assertAlmostEqual(calc._resolve_grasp_depth(valid, 9999.0, 1.0), expected)
        # the scale multiplies the quantile (depth is in raw units * scale -> mm):
        self.assertAlmostEqual(calc._resolve_grasp_depth(valid, 0.0, 2.0), expected * 2.0)

    # ---- penetration descends every silhouette candidate by exactly the penetration ----

    def test_penetration_descends_all_candidates_uniformly(self) -> None:
        z0 = self._candidate_zs(self._calc())  # penetration 0.0
        zp = self._candidate_zs(self._calc(grasp_top_penetration_mm=30.0))
        self.assertEqual(len(z0), len(zp))
        for a, b in zip(z0, zp):
            self.assertAlmostEqual(b - a, 30.0, places=6)

    # ---- the top reference flows through compute() ----

    def test_top_reference_raises_the_centre_candidate(self) -> None:
        z_centre = self._candidate_zs(self._calc())
        z_top = self._candidate_zs(self._calc(grasp_depth_reference="top"))
        self.assertNotEqual(z_centre, z_top)  # the reference changed the output
        # "top" references a nearer (smaller-depth = higher) surface than the median.
        self.assertLess(min(z_top), min(z_centre))

    def test_flat_object_top_equals_centre(self) -> None:
        # Uniform depth -> the quantile equals the median -> "top" is identical to "centre" (and the
        # penetration default keeps it byte-identical).
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 6:18] = 1
        depth = np.full((24, 24), 1000.0, dtype=np.float64)
        seg = SimpleNamespace(mask=mask, label="box")
        z_centre = sorted(
            float(g.position[2])
            for g in self._calc().compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        )
        z_top = sorted(
            float(g.position[2])
            for g in self._calc(grasp_depth_reference="top").compute(
                seg, depth, pixel_to_mm=5.0, unit="mm"
            )
        )
        self.assertEqual(z_centre, z_top)

    # ---- construction validation ----

    def test_invalid_reference_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._calc(grasp_depth_reference="middle")

    def test_negative_penetration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._calc(grasp_top_penetration_mm=-1.0)

    def test_out_of_range_quantile_rejected(self) -> None:
        for q in (0.0, 100.0, -5.0):
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    self._calc(grasp_top_quantile=q)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
