"""Geometry hardening + the shared validators: the edge cases the dedup/hardening pass introduced.

These exercise paths the rest of the suite never hits — coincident-cloud farthest-point sampling,
rounded uniform sampling, denormalised/skewed camera matrices, and the shared validation helpers.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.geometry import (
    CameraIntrinsics,
    as_mask_and_depth,
    as_points_nx3,
    as_vec3,
    farthest_point_sample_indices,
    uniform_sample_indices,
)


class FarthestPointSampleTests(unittest.TestCase):
    def test_coincident_cloud_returns_distinct_indices_only(self) -> None:
        # 3 coincident points at the origin + 2 distinct -> only 3 distinct positions exist.
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]
        )
        idx = farthest_point_sample_indices(points, 4)  # ask for more than exist
        self.assertEqual(len(set(idx.tolist())), len(idx))  # no duplicate indices
        self.assertEqual(len(np.unique(points[idx], axis=0)), len(idx))  # all chosen positions distinct
        self.assertEqual(len(idx), 3)

    def test_normal_cloud_unchanged(self) -> None:
        # A fully distinct cloud still returns exactly num_samples (byte-identical to before).
        points = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        self.assertEqual(len(farthest_point_sample_indices(points, 5)), 5)


class UniformSampleTests(unittest.TestCase):
    def test_rounds_not_truncates(self) -> None:
        # linspace(0, 10, 4) = [0, 3.33, 6.67, 10] -> round [0, 3, 7, 10], not truncate [0, 3, 6, 10].
        self.assertEqual(uniform_sample_indices(11, 4).tolist(), [0, 3, 7, 10])

    def test_keeps_endpoints(self) -> None:
        idx = uniform_sample_indices(100, 5)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 99)


class CameraIntrinsicsFromMatrixTests(unittest.TestCase):
    def _k(self, k22: float = 1.0, skew: float = 0.0) -> np.ndarray:
        return np.array([[500.0, skew, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, k22]])

    def test_accepts_normalised_skewless(self) -> None:
        intr = CameraIntrinsics.from_matrix(self._k())
        self.assertEqual((intr.fx, intr.fy, intr.cx, intr.cy), (500.0, 500.0, 320.0, 240.0))

    def test_rejects_denormalised(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics.from_matrix(self._k(k22=2.0))  # would silently 2x-scale points

    def test_rejects_skew(self) -> None:
        with self.assertRaises(ValueError):
            CameraIntrinsics.from_matrix(self._k(skew=5.0))


class SharedValidatorTests(unittest.TestCase):
    def test_as_vec3_copies_and_does_not_alias_caller(self) -> None:
        src = np.array([1.0, 2.0, 3.0])
        out = as_vec3(src, "x")
        src[0] = 9.0
        self.assertEqual(out[0], 1.0)  # out is an independent copy

    def test_as_vec3_rejects_bad_shape_and_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            as_vec3(np.zeros(2), "x")
        with self.assertRaises(ValueError):
            as_vec3(np.array([np.nan, 0.0, 0.0]), "x")

    def test_as_points_nx3_shape_and_empty(self) -> None:
        with self.assertRaises(ValueError):
            as_points_nx3(np.zeros(3))  # not (N, 3)
        self.assertEqual(as_points_nx3(np.empty((0, 3))).shape, (0, 3))  # empty allowed

    def test_as_mask_and_depth_requires_2d_matching(self) -> None:
        mask, depth = as_mask_and_depth(np.ones((4, 5), dtype=bool), np.ones((4, 5)))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth.dtype, np.float64)
        with self.assertRaises(ValueError):
            as_mask_and_depth(np.ones((2, 2, 2)), np.ones((2, 2, 2)))  # 3-D
        with self.assertRaises(ValueError):
            as_mask_and_depth(np.ones((4, 5)), np.ones((5, 4)))  # shape mismatch


if __name__ == "__main__":
    unittest.main()
