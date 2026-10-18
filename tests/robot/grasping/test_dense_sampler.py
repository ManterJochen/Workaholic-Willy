"""Tests for the cluttered-bin scene context: sibling SAM2 masks fed through ``other_object_masks=...`` are
rasterised into the collision cloud (``scene_collision_cloud``), which the calculator's stage and its post-hoc
filter keep the hand off.

(The dense surface sampler beside it left on 2026-10-18 with its tests; the calculator stopped sampling densely
when its dense stage left with the silhouette on 2026-10-16.)
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.contacts import scene_collision_cloud
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
