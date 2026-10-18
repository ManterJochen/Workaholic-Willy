"""A calibrated stereo rig is a camera the cell takes as it takes an RGB-D one (the owner, 2026-10-17).

A webcam pair or a side by side stereo device becomes an RGB-D camera once its stereo calibration is written: its owner
opens the pair through ``StereoDepthStreamer``, every grab is an ``RGBDFrame``, its lens is the rectified left one with
no distortion, and it carries ``extrinsics`` and a ``body`` as an RGB-D rig does. Before the calibration it gives no
depth, and every door that chooses or opens a camera says so in one sentence naming how to write it. The stereo
calibration tools keep the raw pair.
"""

from __future__ import annotations

import datetime as dt
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydantic import ValidationError

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.extrinsics import Extrinsics
from src.calibration.serialization import save_extrinsics
from src.calibration.stereo.sub_modules.depth import NEAR_LIMIT_MM
from src.camera.orchestration.camera import (
    Camera,
    CameraRefused,
    create_device_streamer,
    create_streamer,
    select_rig,
)
from src.camera.orchestration.frame_provider import FrameProvider
from src.camera.setup.image_taking.frames import RGBDFrame, StereoFrame
from src.camera.setup.image_taking.single import SingleDeviceStreamer
from src.camera.setup.image_taking.stereo_depth import StereoDepthStreamer, StereoNotCalibrated
from src.camera.setup.image_taking.webcam import WebcamPairStreamer
from src.geometry import Frame, Transform
from tests import _stereo_scene as scene


def _section(primary: str, *rigs: object) -> SimpleNamespace:
    return SimpleNamespace(cameras=SimpleNamespace(primary_rig_id=primary, rigs=list(rigs)))


class _Folder(unittest.TestCase):

    def setUp(self) -> None:
        self._folder = tempfile.TemporaryDirectory()
        self.root = Path(self._folder.name)

    def tearDown(self) -> None:
        self._folder.cleanup()


class AStereoRigOpensAsADepthCameraTests(_Folder):

    def test_before_its_stereo_calibration_it_refuses_to_open_saying_how_to_write_it(self) -> None:
        rig = scene.webcam_rig(self.root / "pair")
        pair = scene.PairDouble(*scene.render())
        with self.assertRaises(StereoNotCalibrated) as caught:
            StereoDepthStreamer(rig, pair=pair).open()
        said = str(caught.exception)
        self.assertIn("calibration_paths.stereo_map_file", said)
        self.assertIn(str(Path(str(rig.calibration_paths.stereo_map_file))), said)
        self.assertIn("19_calibrate_a_stereo_pair", said)
        self.assertFalse(pair.opened, "nothing is opened for a rig that gives no depth")

    def test_a_calibration_solved_at_another_size_is_refused(self) -> None:
        rig = scene.webcam_rig(self.root / "pair", size=(320, 240))
        scene.write_calibration(rig, scene.calibration())
        pair = scene.PairDouble(*scene.render())
        with self.assertRaisesRegex(StereoCalibrationError, "640 x 480 and its eyes deliver 320 x 240"):
            StereoDepthStreamer(rig, pair=pair).open()
        self.assertFalse(pair.opened)

    def test_a_grab_is_colour_and_depth_and_the_lens_is_the_rectified_one(self) -> None:
        rig = scene.webcam_rig(self.root / "pair")
        result = scene.calibration()
        scene.write_calibration(rig, result)
        streamer = StereoDepthStreamer(rig, pair=scene.PairDouble(*scene.render()))
        self.assertIsNone(streamer.get_intrinsics(), "no lens before it opens")
        with streamer:
            frame = streamer.grab()
            self.assertIsInstance(frame, RGBDFrame)
            self.assertEqual((480, 640, 3), frame.color.shape)
            self.assertEqual(np.uint16, frame.depth.dtype)
            self.assertGreater(float(np.count_nonzero(frame.depth)) / frame.depth.size, 0.75)
            np.testing.assert_allclose(result.K_rect, streamer.get_intrinsics())
            np.testing.assert_array_equal(np.zeros(5), streamer.get_distortion())
            depth = streamer.stereo_depth
            assert depth is not None
            self.assertEqual((NEAR_LIMIT_MM, math.inf), depth.depth_range_mm)
        self.assertFalse(streamer.is_opened())

    def test_a_look_matches_nothing_and_a_move_leaves_nothing_to_drop(self) -> None:
        rig = scene.webcam_rig(self.root / "pair")
        scene.write_calibration(rig, scene.calibration())
        pair = scene.PairDouble(*scene.render())
        with StereoDepthStreamer(rig, pair=pair) as streamer:
            first = streamer.grab()
            streamer.camera_moved()
            looked = streamer.peek()
            again = streamer.grab()
        self.assertEqual((480, 640, 3), looked.shape)
        np.testing.assert_array_equal(first.color, looked)
        np.testing.assert_array_equal(first.depth, again.depth)  # nothing of the frame before is carried over
        self.assertEqual(3, pair.grabs)

    def test_a_side_by_side_device_is_read_at_its_eye_size(self) -> None:
        rig = scene.single_rig(self.root / "device")
        scene.write_calibration(rig, scene.calibration())
        with StereoDepthStreamer(rig, pair=scene.PairDouble(*scene.render())) as streamer:
            self.assertEqual((480, 640), streamer.grab().depth.shape)

    def test_a_fixed_rig_searches_from_where_its_workspace_box_begins(self) -> None:
        artifact = save_extrinsics(self.root / "eth_pair.json", Extrinsics(
            transform=Transform.from_matrix(scene_looking_down(1300.0), from_frame=Frame.CAMERA, to_frame=Frame.BASE),
            rmse_mm=0.8, max_error_mm=1.5, num_samples=12,
            captured_at=dt.datetime(2026, 10, 17, tzinfo=dt.timezone.utc), rig_id="pair"))
        rig = scene.webcam_rig(self.root / "pair", extrinsics={"mounting_mode": "eye_to_hand",
                                                                "artifact_path": str(artifact)})
        scene.write_calibration(rig, scene.calibration())
        box = SimpleNamespace(x_min=-400.0, x_max=400.0, y_min=-400.0, y_max=400.0, z_min=0.0, z_max=600.0)
        with StereoDepthStreamer(rig, workspace=box, pair=scene.PairDouble(*scene.render())) as streamer:
            depth = streamer.stereo_depth
            assert depth is not None
            self.assertAlmostEqual(700.0, depth.depth_range_mm[0])  # 1300 above the base, the box's top at 600
            self.assertEqual(math.inf, depth.depth_range_mm[1])
        with StereoDepthStreamer(rig, pair=scene.PairDouble(*scene.render())) as streamer:
            assert streamer.stereo_depth is not None
            self.assertEqual(NEAR_LIMIT_MM, streamer.stereo_depth.depth_range_mm[0], "no box: the near limit")


def scene_looking_down(height_mm: float) -> np.ndarray:
    """CAMERA to BASE of a camera at (0, 0, height) whose optical axis points straight down the base z."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[2, 3] = height_mm
    return transform


class EveryDoorTakesACalibratedPairTests(_Folder):

    def test_select_rig_refuses_a_pair_without_its_stereo_calibration_and_takes_it_with_one(self) -> None:
        pair = scene.webcam_rig(self.root / "pair")
        with self.assertRaises(CameraRefused) as caught:
            select_rig(_section("pair", pair))
        self.assertEqual("no_stereo_calibration", caught.exception.reason)
        self.assertIn("camera.cameras.primary_rig_id names 'pair'", str(caught.exception))
        self.assertIn("19_calibrate_a_stereo_pair", str(caught.exception))
        scene.write_calibration(pair, scene.calibration())
        self.assertIs(pair, select_rig(_section("pair", pair)))

    def test_the_refusal_names_the_rigs_that_give_depth(self) -> None:
        pair = scene.webcam_rig(self.root / "pair")
        calibrated = scene.single_rig(self.root / "device")
        scene.write_calibration(calibrated, scene.calibration())
        with self.assertRaises(CameraRefused) as caught:
            select_rig(_section("pair", pair, calibrated))
        self.assertIn("The rigs here that give depth now are ['device']", str(caught.exception))

    def test_the_owner_measures_with_depth_and_the_catalogue_keeps_the_raw_pair(self) -> None:
        pair = scene.webcam_rig(self.root / "pair")
        device = scene.single_rig(self.root / "device")
        self.assertIsInstance(create_streamer(pair), StereoDepthStreamer)
        self.assertIsInstance(create_streamer(device), StereoDepthStreamer)
        self.assertIsInstance(create_device_streamer(pair), WebcamPairStreamer)
        self.assertIsInstance(create_device_streamer(device), SingleDeviceStreamer)
        self.assertTrue(Camera.from_rig(pair).gives_depth)
        self.assertFalse(Camera.from_rig(pair, streamer=create_device_streamer(pair)).gives_depth)
        provider = FrameProvider([pair, device])
        self.assertIsInstance(provider._streamers["pair"], WebcamPairStreamer)  # noqa: SLF001
        self.assertFalse(provider.is_rgbd("pair"))

    def test_an_owner_over_the_raw_pair_hands_out_the_pair(self) -> None:
        pair = scene.webcam_rig(self.root / "pair")
        with Camera.from_rig(pair, streamer=scene.PairDouble(*scene.render())) as camera:
            self.assertIsInstance(camera.grab(), StereoFrame)


class AStereoRigCarriesItsPlacementTests(_Folder):

    def test_a_pair_declares_its_hand_eye_calibration_fixed_or_on_the_wrist(self) -> None:
        fixed = scene.webcam_rig(self.root / "a", extrinsics={"mounting_mode": "eye_to_hand",
                                                               "artifact_path": "calibration/eth_a.json"})
        self.assertEqual("eye_to_hand", fixed.extrinsics.mounting_mode if fixed.extrinsics else None)
        wrist = scene.webcam_rig(self.root / "b", extrinsics={
            "mounting_mode": "eye_in_hand", "artifact_path": "calibration/eih_b.json",
            "shutter_motion_tolerance_mm": 1.0, "shutter_motion_tolerance_deg": 0.2})
        self.assertEqual("eye_in_hand", wrist.extrinsics.mounting_mode if wrist.extrinsics else None)

    def test_a_pair_on_the_wrist_declares_the_body_the_arm_carries(self) -> None:
        rig = scene.webcam_rig(self.root / "wrist_pair", rig_id="wrist_pair", extrinsics={
            "mounting_mode": "eye_in_hand", "artifact_path": "calibration/eih_wrist_pair.json",
            "shutter_motion_tolerance_mm": 1.0, "shutter_motion_tolerance_deg": 0.2,
            "record_tolerance_mm": 1.0, "record_tolerance_deg": 0.5},
            body={"model": "realsense_d415", "margin_mm": 5.0, "bracket": None})
        self.assertIsNotNone(rig.body)
        with self.assertRaises(ValidationError):  # a fixed camera still declares no body, stereo or not
            scene.webcam_rig(self.root / "fixed_pair", extrinsics={
                "mounting_mode": "eye_to_hand", "artifact_path": "calibration/eth.json"},
                body={"model": "realsense_d415", "margin_mm": 5.0, "bracket": None})


if __name__ == "__main__":
    unittest.main()
