"""The closing axis must lie in the SUPPORT plane, not the image plane.

The defect these tests pin: ``principal_to_3d_axis`` built the axis by zeroing z in the CAMERA frame,
which is the image plane. The two coincide only for a camera looking straight down the support normal.
This project's rig is tilted 36.87 deg, 98.9 % of real jaw grasps close horizontally, and the mismatch
was measured as 42 % of all wrong candidates on objects standing completely free.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.generation._camera_geometry import (
    _SharedGeometryUtil as CameraGeometry,
)


def _rotation_x(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(a), -np.sin(a)],
                     [0.0, np.sin(a), np.cos(a)]])


class LevellingTests(unittest.TestCase):
    def test_a_levelled_axis_is_perpendicular_to_the_support_normal(self) -> None:
        """The whole contract, in one assertion."""
        up = np.array([0.3, -0.2, 0.93])
        up = up / np.linalg.norm(up)
        for raw in ([1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [-0.7, 0.1, 0.0]):
            with self.subTest(axis=raw):
                levelled = CameraGeometry.level_axis_to_support_plane(np.asarray(raw), up)
                self.assertAlmostEqual(float(levelled @ up), 0.0, places=9)
                self.assertAlmostEqual(float(np.linalg.norm(levelled)), 1.0, places=9)

    def test_a_tilted_camera_is_exactly_what_it_corrects(self) -> None:
        """The rig's obliques sit at 36.87 deg. Before levelling the axis is off by that much."""
        tilt = 36.87
        camera_to_base = _rotation_x(tilt)
        up_cam = camera_to_base.T @ np.array([0.0, 0.0, 1.0])
        # An image-plane axis, i.e. what principal_to_3d_axis produces.
        raw = CameraGeometry.principal_to_3d_axis(np.array([0.0, 1.0]))
        before = camera_to_base @ raw
        after = camera_to_base @ CameraGeometry.level_axis_to_support_plane(raw, up_cam)
        angle = lambda v: abs(np.degrees(np.arcsin(np.clip(abs(v[2]), 0.0, 1.0))))  # noqa: E731
        self.assertAlmostEqual(angle(before), tilt, places=3,
                               msg="the untouched axis leaves horizontal by the camera's own tilt")
        self.assertAlmostEqual(angle(after), 0.0, places=9,
                               msg="the levelled axis is horizontal in BASE")

    def test_an_axis_already_in_the_plane_is_left_alone(self) -> None:
        up = np.array([0.0, 0.0, 1.0])
        raw = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(
            CameraGeometry.level_axis_to_support_plane(raw, up), raw, atol=1e-12)

    def test_an_axis_along_the_normal_is_returned_untouched_not_guessed(self) -> None:
        """A grasp closing straight down onto the table has no projection; inventing one is worse."""
        up = np.array([0.0, 0.0, 1.0])
        raw = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            CameraGeometry.level_axis_to_support_plane(raw, up), raw, atol=1e-12)

    def test_a_degenerate_normal_changes_nothing(self) -> None:
        raw = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            CameraGeometry.level_axis_to_support_plane(raw, np.zeros(3)), raw, atol=1e-12)

    def test_it_uses_the_plane_it_is_given_not_vertical(self) -> None:
        """A tilted tray is the caller's business; this function only needs to honour the normal."""
        up = _rotation_x(20.0) @ np.array([0.0, 0.0, 1.0])
        levelled = CameraGeometry.level_axis_to_support_plane(np.array([0.0, 1.0, 0.0]), up)
        self.assertAlmostEqual(float(levelled @ up), 0.0, places=9)
        self.assertNotAlmostEqual(float(levelled[2]), 0.0, places=3,
                                  msg="levelling to a tilted plane must NOT produce a horizontal axis")


class FlagTests(unittest.TestCase):
    def test_the_flag_defaults_on_and_can_restore_the_old_behaviour(self) -> None:
        """Default ON is the decision: a frame error is a bug, and a bug fix left off is a kept bug."""
        from src.robot.grasping.generation.calculator import GraspCalculator

        intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        self.assertTrue(GraspCalculator(camera_matrix=intrinsics)._level_closing_to_support_plane)
        self.assertFalse(GraspCalculator(camera_matrix=intrinsics,
                                         level_closing_to_support_plane=False)
                         ._level_closing_to_support_plane)


if __name__ == "__main__":
    unittest.main()
