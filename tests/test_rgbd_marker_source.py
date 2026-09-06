"""S3: the RGB-D ArUco marker source + the intrinsics loader, hardware-free.

Bucket (2) evidence before September: the marker source poses a synthetically-rendered ArUco board
through the same ArucoPoseEstimator the sim path uses, and the loader round-trips a written intrinsics
file. No pyrealsense2, no physical camera.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

cv = pytest.importorskip("cv2")

from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource  # noqa: E402
from src.camera.setup.image_taking.intrinsics import load_intrinsics  # noqa: E402

_K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _marker_bgr(dict_name: str = "DICT_5X5_100", marker_id: int = 0,
                marker_px: int = 240, frame: tuple[int, int] = (480, 640)) -> np.ndarray:
    aruco = cv.aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    img = aruco.generateImageMarker(d, marker_id, marker_px)
    h, w = frame
    canvas = np.full((h, w), 255, dtype=np.uint8)
    y0, x0 = (h - marker_px) // 2, (w - marker_px) // 2
    canvas[y0:y0 + marker_px, x0:x0 + marker_px] = img
    return cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)


@dataclass
class _Frame:
    color: np.ndarray


class _FakeStreamer:
    """grab() -> a fixed marker frame; get_intrinsics/get_distortion -> fixed values. Counts grabs."""

    def __init__(self, bgr: np.ndarray, k: np.ndarray | None = _K, dist: np.ndarray | None = None) -> None:
        self._frame, self._k, self._dist = _Frame(bgr), k, dist
        self.grabs = 0

    def grab(self) -> _Frame:
        self.grabs += 1
        return self._frame

    def get_intrinsics(self) -> np.ndarray | None:
        return None if self._k is None else self._k.copy()

    def get_distortion(self) -> np.ndarray | None:
        return None if self._dist is None else self._dist.copy()


class MarkerSourceTests(unittest.TestCase):
    def test_poses_a_rendered_marker(self) -> None:
        src = RGBDArucoMarkerSource(
            streamer=_FakeStreamer(_marker_bgr()), marker_length_mm=50.0,
            dict_name="DICT_5X5_100", target_id=0, warmup_grabs=0)
        T = src()
        self.assertIsNotNone(T)
        assert T is not None
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertGreater(T[2, 3], 0.0)                 # marker in front of the camera
        self.assertLess(abs(T[0, 3]), 30.0)              # centred -> small in-plane translation
        self.assertEqual(src.intrinsics_source, "factory")

    def test_no_marker_returns_none(self) -> None:
        blank = np.full((480, 640, 3), 127, dtype=np.uint8)
        src = RGBDArucoMarkerSource(streamer=_FakeStreamer(blank), warmup_grabs=0)
        self.assertIsNone(src())

    def test_missing_intrinsics_raises(self) -> None:
        src = RGBDArucoMarkerSource(streamer=_FakeStreamer(_marker_bgr(), k=None), warmup_grabs=0)
        with self.assertRaises(RuntimeError):
            src()

    def test_calibrated_intrinsics_override_is_used_and_recorded(self) -> None:
        # D1: passing intrinsics= overrides the streamer's factory K (which is None here, so the source
        # can only succeed via the override) and records the provenance.
        src = RGBDArucoMarkerSource(
            streamer=_FakeStreamer(_marker_bgr(), k=None), intrinsics=_K, distortion=np.zeros(5),
            warmup_grabs=0)
        self.assertEqual(src.intrinsics_source, "override")
        self.assertIsNotNone(src())

    def test_warmup_grabs_are_discarded(self) -> None:
        st = _FakeStreamer(_marker_bgr())
        RGBDArucoMarkerSource(streamer=st, warmup_grabs=3)()
        self.assertEqual(st.grabs, 4)   # 3 warmup + 1 real

    def test_none_device_distortion_falls_back_to_zeros(self) -> None:
        # get_distortion() is None -> the source must pose with zeros(5), not crash.
        src = RGBDArucoMarkerSource(streamer=_FakeStreamer(_marker_bgr(), dist=None), warmup_grabs=0)
        self.assertIsNotNone(src())


class LoadIntrinsicsTests(unittest.TestCase):
    def _write(self, tmp: str, payload: dict) -> Path:
        p = Path(tmp) / "intrinsics.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_round_trip_with_distortion(self) -> None:
        with TemporaryDirectory() as tmp:
            p = self._write(tmp, {"fx": 600.0, "fy": 610.0, "cx": 320.0, "cy": 240.0,
                                  "dist": [0.1, -0.2, 0.001, 0.002, 0.05]})
            k, dist = load_intrinsics(p)
            self.assertTrue(np.allclose(k, [[600, 0, 320], [0, 610, 240], [0, 0, 1]]))
            self.assertTrue(np.allclose(dist, [0.1, -0.2, 0.001, 0.002, 0.05]))

    def test_missing_dist_defaults_to_zeros(self) -> None:
        with TemporaryDirectory() as tmp:
            p = self._write(tmp, {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0})
            _, dist = load_intrinsics(p)
            self.assertTrue(np.allclose(dist, np.zeros(5)))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_intrinsics(Path("nope") / "intrinsics.json")

    def test_malformed_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            p = self._write(tmp, {"fx": 600.0})  # no fy/cx/cy
            with self.assertRaises(ValueError):
                load_intrinsics(p)

    def test_non_physical_focal_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            p = self._write(tmp, {"fx": -1.0, "fy": 600.0, "cx": 320.0, "cy": 240.0})
            with self.assertRaises(ValueError):
                load_intrinsics(p)


if __name__ == "__main__":
    unittest.main()
