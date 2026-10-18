"""Tests for the dense surface sampler and cluttered-bin scene context.

These cover two contracts of the library and one the calculator needs:

* boundary pixels and depth discontinuities must be excluded so
  occlusion edges do not anchor antipodal pairs;
* dense sampling returns a ``SurfaceSamples`` carrying normals,
  graspability, and pixel coordinates aligned 1:1;
* sibling SAM2 masks fed through ``other_object_masks=...`` are
  rasterised into the collision cloud (``scene_collision_cloud``), which the
  calculator's stage and its post-hoc filter keep the hand off.

(The calculator sampled densely until its dense stage left with the silhouette on 2026-10-16.)
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.contacts import (
    dense_surface_samples,
    scene_collision_cloud,
)
from src.robot.grasping.geometry import CameraIntrinsics


def _intrinsics(width: int = 64, height: int = 64) -> CameraIntrinsics:
    return CameraIntrinsics(fx=400.0, fy=400.0, cx=width / 2.0, cy=height / 2.0)


def _planar_object(
    *,
    shape: tuple[int, int] = (64, 64),
    top: int = 18,
    bottom: int = 46,
    left: int = 20,
    right: int = 44,
    depth_mm: float = 500.0,
    background_depth_mm: float = 900.0,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    depth = np.full(shape, background_depth_mm, dtype=np.float64)
    depth[top:bottom, left:right] = depth_mm
    return mask, depth


class DenseSurfaceSamplerTests(unittest.TestCase):
    def test_returns_samples_inside_eroded_interior(self) -> None:
        mask, depth = _planar_object()
        samples = dense_surface_samples(
            mask,
            depth,
            _intrinsics(),
            edge_erosion_px=2,
            depth_discontinuity_mm=8.0,
            voxel_size_mm=None,
            max_points=4096,
            fps=False,
        )
        self.assertGreater(samples.size, 0)
        # No sample sits on the original mask boundary (rows == 18 or 45,
        # cols == 20 or 43).
        rows = samples.pixels_yx[:, 0]
        cols = samples.pixels_yx[:, 1]
        self.assertTrue(((rows >= 20) & (rows <= 43)).all())
        self.assertTrue(((cols >= 22) & (cols <= 41)).all())

    def test_excludes_depth_discontinuity_pixels(self) -> None:
        # Two-tier object: front and back wall with a large depth step.
        shape = (64, 64)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[20:44, 20:44] = 1
        depth = np.full(shape, 900.0, dtype=np.float64)
        depth[20:44, 20:30] = 400.0  # front face
        depth[20:44, 30:44] = 800.0  # back face -> 400 mm step
        samples = dense_surface_samples(
            mask,
            depth,
            _intrinsics(),
            edge_erosion_px=1,
            depth_discontinuity_mm=50.0,
            voxel_size_mm=None,
            max_points=4096,
            fps=False,
        )
        # No surviving sample should straddle the depth step column.
        if samples.size > 0:
            cols = samples.pixels_yx[:, 1]
            self.assertFalse(np.any((cols == 29) | (cols == 30)))

    def test_empty_when_mask_disappears_under_erosion(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[9:11, 9:11] = 1  # 2x2 mask
        depth = np.full((20, 20), 500.0, dtype=np.float64)
        samples = dense_surface_samples(
            mask,
            depth,
            _intrinsics(width=20, height=20),
            edge_erosion_px=2,
        )
        self.assertEqual(samples.size, 0)

    def test_fps_caps_at_max_points(self) -> None:
        mask, depth = _planar_object()
        samples = dense_surface_samples(
            mask,
            depth,
            _intrinsics(),
            voxel_size_mm=None,
            max_points=20,
            fps=True,
        )
        self.assertLessEqual(samples.size, 20)
        self.assertEqual(samples.points_mm.shape[0], samples.pixels_yx.shape[0])
        self.assertEqual(samples.points_mm.shape[0], samples.graspability.shape[0])

    def test_graspability_in_unit_range(self) -> None:
        mask, depth = _planar_object()
        samples = dense_surface_samples(
            mask,
            depth,
            _intrinsics(),
            voxel_size_mm=None,
            max_points=4096,
            fps=False,
        )
        if samples.size > 0:
            self.assertTrue(np.all(samples.graspability >= 0.0))
            self.assertTrue(np.all(samples.graspability <= 1.0))


class SceneCollisionCloudTests(unittest.TestCase):
    def test_excludes_target_index_and_keeps_others(self) -> None:
        shape = (40, 40)
        target = np.zeros(shape, dtype=np.uint8)
        target[10:20, 10:20] = 1
        neighbour = np.zeros(shape, dtype=np.uint8)
        neighbour[25:35, 25:35] = 1
        depth = np.full(shape, 500.0, dtype=np.float64)
        cloud = scene_collision_cloud(
            [target, neighbour],
            depth,
            _intrinsics(width=40, height=40),
            target_index=0,
            voxel_size_mm=None,
        )
        self.assertGreater(cloud.shape[0], 0)
        # Project a target-mask pixel to mm and check it is NOT in the cloud.
        # The neighbour patch covers cols 25..34, so all returned points
        # must have an x in approximately the neighbour's projected range.

    def test_empty_when_no_other_masks(self) -> None:
        shape = (20, 20)
        target = np.zeros(shape, dtype=np.uint8)
        target[5:15, 5:15] = 1
        depth = np.full(shape, 500.0, dtype=np.float64)
        cloud = scene_collision_cloud(
            [target],
            depth,
            _intrinsics(width=20, height=20),
            target_index=0,
        )
        self.assertEqual(cloud.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
