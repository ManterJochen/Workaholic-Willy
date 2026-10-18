"""A stereo rig is calibrated against the robot as an RGB-D rig is: its owner hands out colour, depth and a lens.

The owner, 2026-10-17: a webcam pair or a side by side stereo device sits wherever an RGB-D camera sits, fixed or on the
wrist, and the hand-eye calibration takes it through the same door, ``HandEyeCalibration`` and its CLI: the camera's
owner, a handle of it, and ``RGBDArucoMarkerSource`` posing the board on the frame's colour with the lens the owner
reports. For a stereo rig that colour is the left image rectified and that lens the rectified left one with no
distortion, the frame its depth lies in, so the solved CAMERA to BASE (or TOOL) places that depth.

Proved here on a synthetic pair (``tests/_stereo_scene.py``): a textured plane, a box standing on it and an ArUco marker
printed on it, seen by two cameras of a known lens 60 mm apart, the right one turned three degrees, so a reader that
skipped the rectification or posed the marker with the raw lens would be off by centimetres. The depth is held to the
scene's own, and the marker's pose to where the marker lies in the rectified left camera's frame.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from tests import _stereo_scene as scene
from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
from src.calibration.stereo.sub_modules.aruco_esti import ArucoPoseEstimator
from src.camera.orchestration.camera import Camera, select_rig
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.image_taking.stereo_depth import StereoDepthStreamer


class _OneRig:
    """A camera section of one rig, as ``select_rig`` reads it."""

    def __init__(self, rig: object) -> None:
        self.cameras = type("Cameras", (), {"rigs": [rig], "primary_rig_id": getattr(rig, "rig_id")})()


class AStereoRigSweptAsAnRGBDRigTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.left, cls.right = scene.render()
        cls.calibration = scene.calibration()
        cls._folder = tempfile.TemporaryDirectory()
        cls.rig = scene.webcam_rig(Path(cls._folder.name) / "pair")
        scene.write_calibration(cls.rig, cls.calibration)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._folder.cleanup()

    def _camera(self) -> Camera:
        """The rig's owner, chosen as the sweep chooses it, over the rendered pair instead of two webcams."""
        rig = select_rig(_OneRig(self.rig), rig_id=self.rig.rig_id)
        return Camera.from_rig(rig, streamer=StereoDepthStreamer(rig, pair=scene.PairDouble(self.left, self.right)))

    def test_the_owner_hands_out_the_left_image_rectified_and_its_depth(self) -> None:
        with self._camera() as camera:
            frame = camera.grab()
            lens, distortion = camera.get_intrinsics(), camera.get_distortion()
        self.assertIsInstance(frame, RGBDFrame)
        assert isinstance(frame, RGBDFrame)
        self.assertEqual((480, 640, 3), frame.color.shape)
        self.assertEqual(np.uint8, frame.color.dtype)
        self.assertEqual(np.uint16, frame.depth.dtype)
        np.testing.assert_allclose(self.calibration.K_rect, lens)
        np.testing.assert_array_equal(np.zeros(5), distortion)
        rectified = cv.remap(self.left, self.calibration.stereoMapL_x, self.calibration.stereoMapL_y, cv.INTER_LANCZOS4)
        np.testing.assert_array_equal(rectified, frame.color[..., 0])

    def test_the_depth_is_the_scenes_own_where_it_is_read(self) -> None:
        with self._camera() as camera:
            frame = camera.grab()
        assert isinstance(frame, RGBDFrame)
        truth = scene.true_depth(self.calibration)
        read = frame.depth > 0
        error = frame.depth[read].astype(np.float64) - truth[read]
        self.assertGreater(read.mean(), 0.75)  # measured 0.89: the occluded strips beside the box are holes
        self.assertLess(float(np.median(np.abs(error))), 3.5)  # measured 2.1 mm at 650 to 830 mm
        self.assertLess(float(np.percentile(np.abs(error), 90)), 7.0)  # measured 3.8 mm
        wrong = np.abs(error) > np.maximum(10.0, 0.03 * truth[read])
        self.assertLess(float(wrong.mean()), 0.005)  # measured 0

    def test_the_marker_is_posed_on_the_rectified_image_with_the_rectified_lens(self) -> None:
        estimator = ArucoPoseEstimator(marker_length_mm=scene.MARKER_MM, dict_name=scene.DICTIONARY,
                                       target_id=scene.MARKER_ID)
        with self._camera() as camera:
            source = RGBDArucoMarkerSource(streamer=camera.handle(), estimator=estimator, warmup_grabs=1)
            pose = source()
        self.assertIsNotNone(pose, getattr(source.last_observation, "why_not", ""))
        assert pose is not None
        centre, z_axis = scene.true_marker(self.calibration)
        self.assertLess(float(np.linalg.norm(pose[:3, 3] - centre)), 6.0)  # measured 3.7 mm at 700 mm
        angle = np.degrees(np.arccos(np.clip(float(pose[:3, 2] @ z_axis), -1.0, 1.0)))
        self.assertLess(angle, 4.0)  # measured 2.2 degrees
        self.assertEqual("factory", source.intrinsics_source)

    def test_the_raw_lens_would_have_placed_it_centimetres_away(self) -> None:
        """The control: posed with the left camera's own lens on the same rectified image, the marker is far off."""
        with self._camera() as camera:
            frame = camera.grab()
        assert isinstance(frame, RGBDFrame)
        estimator = ArucoPoseEstimator(marker_length_mm=scene.MARKER_MM, dict_name=scene.DICTIONARY,
                                       target_id=scene.MARKER_ID)
        raw = estimator.observe(frame.color, scene.K, np.zeros(5)).T_cam_to_target
        assert raw is not None
        centre, _ = scene.true_marker(self.calibration)
        self.assertGreater(float(np.linalg.norm(raw[:3, 3] - centre)), 20.0)  # measured 48 mm


if __name__ == "__main__":
    unittest.main()
