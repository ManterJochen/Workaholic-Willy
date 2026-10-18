"""H1.2 — the point-cloud outlier filter (``geometry/filters.py``) and the dense sampler's seam for it.

The radius / statistical outlier removers, reached through ``CloudOutlierConfig`` + ``apply_cloud_outlier_filter``,
and the dense surface sampler that takes one before FPS (``dense_surface_samples``). The calculator wired both into its
geometry-first and dense stages until those left with the silhouette stage on 2026-10-16; the library functions stay.

HONESTY: the Isaac sim renderer is essentially noise-free, so this filter is a *no-op there* -- it is a
REAL-HARDWARE robustness lever (stereo / RGB-D speckle, edge-bleed). These tests prove it with SYNTHETIC
speckle; its on-box value is unprovable in the clean sim (Tim-accepted, deferred to real hardware).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.contacts.dense_sampler import dense_surface_samples
from src.robot.grasping.geometry import (
    CloudOutlierConfig,
    apply_cloud_outlier_filter,
)


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 20.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _cluster_plus_outliers() -> tuple[np.ndarray, list[int]]:
    """A tight 20-point cluster near the origin + 3 far spatial outliers (indices 20,21,22)."""
    rng = np.random.default_rng(0)
    cluster = rng.uniform(-2.0, 2.0, size=(20, 3))
    outliers = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]])
    return np.vstack([cluster, outliers]), [20, 21, 22]


class ApplyCloudOutlierFilterTests(unittest.TestCase):
    def test_statistical_drops_spatial_outliers(self) -> None:
        points, outlier_idx = _cluster_plus_outliers()
        keep = apply_cloud_outlier_filter(
            points, CloudOutlierConfig(method="statistical", k_neighbors=8, std_ratio=1.0)
        )
        for i in outlier_idx:
            self.assertNotIn(i, keep.tolist())
        self.assertEqual(len(keep), 20)  # the whole cluster survives

    def test_radius_drops_isolated_points(self) -> None:
        points, outlier_idx = _cluster_plus_outliers()
        keep = apply_cloud_outlier_filter(
            points, CloudOutlierConfig(method="radius", radius_mm=10.0, min_neighbors=3)
        )
        for i in outlier_idx:
            self.assertNotIn(i, keep.tolist())

    def test_method_validation(self) -> None:
        with self.assertRaises(ValueError):
            CloudOutlierConfig(method="bogus")


class DenseSamplerSeamTests(unittest.TestCase):
    """Seam 2: dense_surface_samples applies the filter before FPS."""

    @staticmethod
    def _flat_scene() -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 1
        depth = np.zeros((40, 40), dtype=np.float64)
        depth[8:32, 8:32] = 1000.0
        return mask, depth

    def test_filter_is_applied_before_fps(self) -> None:
        mask, depth = self._flat_scene()
        k = _camera_matrix()
        base = dense_surface_samples(mask, depth, k, voxel_size_mm=3.0)
        self.assertGreater(base.size, 0)
        # an aggressive radius filter (no neighbour within 0.01mm) drops every point -> empty samples.
        filtered = dense_surface_samples(
            mask, depth, k, voxel_size_mm=3.0,
            cloud_outlier_filter=CloudOutlierConfig(method="radius", radius_mm=0.01, min_neighbors=1),
        )
        self.assertEqual(filtered.size, 0)

    def test_none_is_byte_identical(self) -> None:
        mask, depth = self._flat_scene()
        k = _camera_matrix()
        a = dense_surface_samples(mask, depth, k, voxel_size_mm=3.0)
        b = dense_surface_samples(mask, depth, k, voxel_size_mm=3.0, cloud_outlier_filter=None)
        self.assertEqual(a.size, b.size)
        np.testing.assert_array_equal(a.points_mm, b.points_mm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
