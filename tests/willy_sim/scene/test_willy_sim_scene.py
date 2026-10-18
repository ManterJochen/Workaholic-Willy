"""willy_sim.scene — Isaac-free unit tests for the hand-eye transform (run everywhere).

The scene authoring is on-box only; here we lock the pure-numpy ``CAMERA -> BASE`` math the
grasp calculator depends on (the single most important correctness item for the pick).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.willy_sim.scene import (
    ARM_PRIM,
    OBJECT_PRIM,
    top_down_camera_to_base_matrix,
)
from src.geometry import Frame, Transform


class CameraToBaseMatrixTests(unittest.TestCase):
    def test_topdown_rotation_is_yawed_flip(self) -> None:
        # Overhead camera with a 90 deg yaw: optical +Z -> base -Z, +X -> base +Y, +Y -> base +X.
        # The yaw aligns the calculator's default antipodal axis with the UR5e's reachable closing.
        mat = top_down_camera_to_base_matrix(np.array([450.0, 0.0, 1000.0]))
        np.testing.assert_allclose(
            mat[:3, :3], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], atol=1e-9
        )
        np.testing.assert_allclose(mat[:3, 3], [450.0, 0.0, 1000.0])

    def test_centroid_at_table_depth_maps_below_camera(self) -> None:
        # A point on the optical axis 1000 mm deep maps to the table directly below the camera
        # (450, 0, 0) — matches Isaac's own back-projection of the object centroid on-box.
        mat = top_down_camera_to_base_matrix(np.array([450.0, 0.0, 1000.0]))
        T = Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        np.testing.assert_allclose(T.apply_point(np.array([0.0, 0.0, 1000.0])), [450.0, 0.0, 0.0], atol=1.0)

    def test_offaxis_point_swaps_into_base(self) -> None:
        mat = top_down_camera_to_base_matrix(np.array([450.0, 0.0, 1000.0]))
        T = Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        # 90 deg yaw: optical (+50 right, +30 down, 1000 deep) -> base (450+30 X, +50 Y, table)
        np.testing.assert_allclose(T.apply_point(np.array([50.0, 30.0, 1000.0])), [480.0, 50.0, 0.0], atol=1.0)

    def test_base_offset_subtracted(self) -> None:
        mat = top_down_camera_to_base_matrix(
            np.array([450.0, 0.0, 1000.0]), base_pos_mm=np.array([100.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(mat[:3, 3], [350.0, 0.0, 1000.0])

    def test_yields_camera_to_base_transform(self) -> None:
        mat = top_down_camera_to_base_matrix(np.array([450.0, 0.0, 1000.0]))
        T = Transform.from_matrix(mat, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        self.assertIs(T.from_frame, Frame.CAMERA)
        self.assertIs(T.to_frame, Frame.BASE)

    def test_prim_path_constants(self) -> None:
        self.assertEqual(ARM_PRIM, "/World/UR5e")
        self.assertTrue(OBJECT_PRIM.startswith("/World/"))


if __name__ == "__main__":
    unittest.main()
