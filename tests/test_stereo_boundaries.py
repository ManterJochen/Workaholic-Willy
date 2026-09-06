from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from src.calibration import StereoCalibrationError
from src.calibration.stereo import CalibrationResult, StereoCam3D
from src.calibration.stereo.repository import StereoRigRepository
from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator
from src.calibration.stereo.sub_modules.pointcloud import PointCloudReconstructor


def _calibration_result() -> CalibrationResult:
    width, height = 4, 3
    cam = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    proj_left = np.array(
        [[100.0, 0.0, 2.0, 0.0], [0.0, 100.0, 1.5, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    proj_right = proj_left.copy()
    proj_right[0, 3] = -5000.0
    Q = np.array(
        [[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, -1.5], [0.0, 0.0, 0.0, 100.0], [0.0, 0.0, -0.02, 0.0]],
        dtype=np.float64,
    )
    return CalibrationResult(
        camL=cam,
        distL=np.zeros(5),
        rectL=np.eye(3),
        projL=proj_left,
        camR=cam.copy(),
        distR=np.zeros(5),
        rectR=np.eye(3),
        projR=proj_right,
        Q=Q,
        frame_size=(width, height),
    )


class StereoBoundaryTests(unittest.TestCase):
    def test_stereo_repository_roundtrip(self) -> None:
        result = _calibration_result()
        T_cam_to_base = np.eye(4, dtype=np.float64)
        T_cam_to_base[:3, 3] = [10.0, 20.0, 30.0]
        repository = StereoRigRepository()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "stereo_calib.json"
            repository.save(path, result, T_cam_to_base)
            loaded, loaded_transform = repository.load(path)

        self.assertTrue(np.allclose(loaded.Q, result.Q))
        self.assertTrue(np.allclose(loaded.projL, result.projL))
        self.assertTrue(np.allclose(loaded_transform, T_cam_to_base))
        self.assertEqual(loaded.frame_size, result.frame_size)

    def test_rig_index_validation(self) -> None:
        sentinel = object()
        stereo = StereoCam3D.__new__(StereoCam3D)
        stereo.rigs = [sentinel]
        self.assertIs(stereo._rig(0), sentinel)
        for invalid in (-1, 1, True):
            with self.assertRaises(IndexError):
                stereo._rig(invalid)

    @unittest.skipUnless(hasattr(cv, "aruco"), "OpenCV ArUco module is unavailable")
    def test_aruco_no_detection_returns_empty_result(self) -> None:
        estimator = ArucoPoseEstimator(marker_length_mm=50.0, dict_name="DICT_5X5_100")
        blank = np.zeros((80, 80, 3), dtype=np.uint8)
        K = np.array([[100.0, 0.0, 40.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])

        self.assertEqual(estimator.estimate(blank, K, np.zeros(5)), {})
        self.assertIsNone(estimator.estimate(blank, K, np.zeros(5), target_id=1))

    def test_aruco_rejects_missing_intrinsics(self) -> None:
        if not hasattr(cv, "aruco"):
            self.skipTest("OpenCV ArUco module is unavailable")
        estimator = ArucoPoseEstimator(marker_length_mm=50.0, dict_name="DICT_5X5_100")
        with self.assertRaises(StereoCalibrationError):
            estimator.estimate(np.zeros((10, 10, 3), dtype=np.uint8), None, np.zeros(5))

    def test_representative_point_preserves_float64_precision(self) -> None:
        class FakeDisparity:
            def compute(self, left, right):
                return np.full((3, 4), 10.0, dtype=np.float64), np.ones((3, 4), dtype=bool)

        result = _calibration_result()
        reconstructor = PointCloudReconstructor(result.Q)
        image = np.zeros((3, 4, 3), dtype=np.uint8)

        point = reconstructor.representative_point(image, image, FakeDisparity(), x=2, y=1, unit="mm")

        self.assertEqual(point.dtype, np.float64)
        self.assertTrue(np.allclose(point, np.array([-0.0, 2.5, -500.0]), atol=1e-12))


if __name__ == "__main__":
    unittest.main()