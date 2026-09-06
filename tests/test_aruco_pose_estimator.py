"""Regression tests for ArucoPoseEstimator (gap L1.1).

cv.aruco.estimatePoseSingleMarkers was REMOVED in OpenCV >= 4.7 (we pin opencv-contrib 4.13), so the
old code raised AttributeError the moment a marker was detected. The estimator must now pose markers
via cv.solvePnP(IPPE_SQUARE) over the known square corners. These tests are hardware-free (a synthetic
rendered marker) and run in the mock/CI suite.
"""

from __future__ import annotations

import numpy as np
import pytest

cv = pytest.importorskip("cv2")

from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator  # noqa: E402

_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _marker_frame(dict_name: str = "DICT_5X5_100", marker_id: int = 0,
                  marker_px: int = 240, frame: tuple[int, int] = (480, 640)) -> np.ndarray:
    aruco = cv.aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    img = aruco.generateImageMarker(d, marker_id, marker_px)
    h, w = frame
    canvas = np.full((h, w), 255, dtype=np.uint8)  # white quiet-zone around the marker
    y0, x0 = (h - marker_px) // 2, (w - marker_px) // 2
    canvas[y0:y0 + marker_px, x0:x0 + marker_px] = img
    return cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)


def test_solvepnp_path_recovers_a_marker_pose() -> None:
    est = ArucoPoseEstimator(marker_length_mm=50.0, dict_name="DICT_5X5_100")
    T = est.estimate(_marker_frame(), _K, np.zeros(5), target_id=0)
    assert T is not None, "marker should be detected + posed"
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    assert T[2, 3] > 0.0  # marker in front of the camera (positive depth)
    # a centred, frontal marker -> small in-plane translation
    assert abs(T[0, 3]) < 30.0 and abs(T[1, 3]) < 30.0


def test_no_marker_returns_none() -> None:
    blank = np.full((480, 640, 3), 127, dtype=np.uint8)
    est = ArucoPoseEstimator(marker_length_mm=50.0)
    assert est.estimate(blank, _K, np.zeros(5), target_id=0) is None


def test_all_markers_dict_when_no_target() -> None:
    est = ArucoPoseEstimator(marker_length_mm=50.0, dict_name="DICT_5X5_100")
    out = est.estimate(_marker_frame(marker_id=7), _K, np.zeros(5), target_id=None)
    assert isinstance(out, dict) and 7 in out and out[7].shape == (4, 4)
