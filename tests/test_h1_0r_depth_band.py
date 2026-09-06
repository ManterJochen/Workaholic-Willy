"""H1.0R-a — the calculator's default-off robust depth-band filter.

On REAL (non-uniform) depth, mask-edge TABLE bleed leaves finite pixels DEEPER than the object top under the
mask, pulling the median (the ``centre`` reference) toward the table. The band keeps only depths within
``[near surface, near + depth_band_mm]`` (the near surface = a robust low percentile), so the centre reference
re-locates onto the object top. Default ``depth_band_mm=None`` -> off -> byte-identical (the R1/R2 goldens are
untouched). This is the enabler that lets ``centre``-rendered survive the table-leak the H1.1b gate exposed.
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


class DepthBandHelperTests(unittest.TestCase):
    """The pure ``_apply_depth_band`` math."""

    @staticmethod
    def _calc(**kwargs: object) -> GraspCalculator:
        return GraspCalculator(
            min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=6,
            camera_matrix=_camera_matrix(), **kwargs,  # type: ignore[arg-type]
        )

    def test_off_is_identity(self) -> None:
        calc = self._calc()  # depth_band_mm=None (default)
        valid = np.array([1000.0, 1000.0, 1100.0, 1100.0])
        out, cap = calc._apply_depth_band(valid, 1.0)
        self.assertIs(out, valid)  # exact same object -> byte-identical
        self.assertIsNone(cap)

    def test_on_rejects_the_deep_table_tail(self) -> None:
        # near percentile(10) ~ 1000 (the top); band 40 -> keep <= 1040 -> drop the 1100 table tail.
        calc = self._calc(depth_band_mm=40.0, depth_band_near_pct=10.0)
        valid = np.concatenate([np.full(10, 1000.0), np.full(10, 1100.0)])
        out, cap = calc._apply_depth_band(valid, 1.0)
        self.assertTrue(np.all(out <= 1040.0))
        self.assertNotIn(1100.0, out.tolist())
        self.assertAlmostEqual(cap, 1040.0)

    def test_scale_applies_to_the_band(self) -> None:
        # raw depths * scale(2.0): near percentile(10)=500 -> *2 = 1000mm; +band 40 -> keep raw <= 520.
        calc = self._calc(depth_band_mm=40.0, depth_band_near_pct=10.0)
        valid = np.concatenate([np.full(10, 500.0), np.full(10, 600.0)])
        out, cap = calc._apply_depth_band(valid, 2.0)
        self.assertAlmostEqual(cap, 1040.0)
        self.assertTrue(np.all(out * 2.0 <= 1040.0))


class DepthBandEndToEndTests(unittest.TestCase):
    """Through compute(): the band re-references the centre grasp onto the object top."""

    @staticmethod
    def _calc(**kwargs: object) -> GraspCalculator:
        return GraspCalculator(
            min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=6,
            camera_matrix=_camera_matrix(), **kwargs,  # type: ignore[arg-type]
        )

    @staticmethod
    def _table_bleed_scene() -> tuple[SimpleNamespace, np.ndarray]:
        # Mask over an object TOP face (near, 1000mm, the top 4 masked rows) with mask-edge TABLE bleed
        # (deep, 1100mm, the lower 6 masked rows) -> the table is the MAJORITY, so the unfiltered median
        # sits on the table (1100). The band must pull it back to ~1000.
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 6:18] = 1
        depth = np.full((24, 24), 1100.0, dtype=np.float64)
        depth[7:11, 6:18] = 1000.0  # object top face (4 of the 10 masked rows)
        seg = SimpleNamespace(mask=mask, label="box")
        return seg, depth

    def _min_candidate_z(self, calc: GraspCalculator) -> float:
        seg, depth = self._table_bleed_scene()
        grasps = calc.compute(seg, depth, pixel_to_mm=5.0, unit="mm")
        self.assertTrue(grasps)
        return min(float(g.position[2]) for g in grasps)

    def test_band_pulls_the_centre_grasp_onto_the_top(self) -> None:
        z_noband = self._min_candidate_z(self._calc())  # off -> centroid sits on the table band
        z_band = self._min_candidate_z(self._calc(depth_band_mm=40.0, depth_band_near_pct=10.0))
        # the band re-references the centroid to the object top -> a measurably nearer (smaller-depth) grasp.
        self.assertLess(z_band, z_noband)
        self.assertLess(z_band, 1050.0)   # near the 1000mm top
        self.assertGreater(z_noband, 1050.0)  # without the band, pulled toward the 1100mm table

    def test_band_off_matches_a_plain_calculator(self) -> None:
        # explicit depth_band_mm=None reproduces the no-arg calculator exactly (byte-identical default).
        z_default = self._min_candidate_z(self._calc())
        z_explicit_off = self._min_candidate_z(self._calc(depth_band_mm=None))
        self.assertEqual(z_default, z_explicit_off)


class DepthBandValidationTests(unittest.TestCase):
    def test_invalid_band_rejected(self) -> None:
        for bad in (0.0, -5.0, float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GraspCalculator(camera_matrix=_camera_matrix(), depth_band_mm=bad)

    def test_invalid_near_pct_rejected(self) -> None:
        for bad in (-1.0, 100.0, 150.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GraspCalculator(camera_matrix=_camera_matrix(), depth_band_near_pct=bad)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
