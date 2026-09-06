"""L8.8 — minimal smoke test for the 2D grasp-visualization path.

The visualization package had NO automated coverage (visual-inspection only). The 2D path is mock-safe
(pure cv2 array drawing + imwrite, no window / display server / Open3D), so this guards it against import
or signature drift in the CI mock suite. Asserts structural properties (not pixel-exact bytes) to stay
platform-stable. The Open3D 3D path is intentionally out of scope (it may be headless/absent in CI).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.robot.grasping.geometry.pointcloud import CameraIntrinsics
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.visualization.debug_draw import (
    draw_grasp_debug_image,
    save_grasp_debug_image,
)


def _camera_grasp() -> GraspPoint:
    return GraspPoint(
        position=np.array([0.0, 0.0, 200.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=0.9,
        frame=GraspFrame.CAMERA,
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=50.0, fy=50.0, cx=32.0, cy=32.0)


class GraspVisualizationSmokeTests(unittest.TestCase):
    def test_draw_grasp_debug_image_returns_rgb_canvas(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        img = draw_grasp_debug_image(rgb, [_camera_grasp()], intrinsics=_intrinsics())
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        # the metadata/score strips are appended, so height >= the input height
        self.assertGreaterEqual(img.shape[0], 64)

    def test_save_grasp_debug_image_writes_a_file(self) -> None:
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "debug.png"
            save_grasp_debug_image(str(out), rgb, [_camera_grasp()], intrinsics=_intrinsics())
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
