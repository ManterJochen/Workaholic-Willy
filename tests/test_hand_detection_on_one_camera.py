"""Asking where a hand is through a camera the program already holds open.

⚠ WHY THIS SEAM EXISTS, AND WHAT IT PREVENTS. `HandFinder` reached its frames through
`FrameProvider`, the multi-rig catalogue, and `FrameProvider.open()` claims every configured
streamer. A program that has already opened one `Camera` -- anything that hands the arm a live
camera world, which is every camera-checked motion -- therefore could not ask where a hand is
without either taking the cell's other devices away from it or opening one device twice, which is
exactly what the camera owner exists to prevent. The capability shipped and no such program could
reach it; `examples/real_robot/15` is the program.

The transform is the other half. `build_hand_finder_on_camera` composes nothing: a fixed rig's
CAMERA to BASE is `RigCalibration.camera_to_base()`, one method with one answer, and a wrist rig is
refused by name rather than re-derived from the tool pose, which `Locator` already owns. A second
spelling of that matrix would be a second source of truth for the number that decides where an arm
moves next to a person's hand.

No mediapipe, no `.task` bundle and no device: the observer is a double and the camera is a stub,
because what is under test is the wiring and the refusals.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.calibration.rig_calibration import RigCalibrationError, RigNotCalibrated
from src.config.schema.models import HandDetectConfig
from src.geometry import Frame
from src.geometry.transform import Transform
from src.models.handdetection.factory import build_hand_finder_on_camera
from src.models.handdetection.hand_finder import OneCamera

_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _camera_to_base() -> Transform:
    return Transform(translation_mm=np.array([100.0, 0.0, 800.0]),
                     quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                     from_frame=Frame.CAMERA, to_frame=Frame.BASE)


class _Calibration:
    """A rig calibration that answers for one mounting and refuses the other, as the real one does."""

    def __init__(self, *, fixed: bool) -> None:
        self._fixed = fixed

    def camera_to_base(self) -> Transform:
        if not self._fixed:
            raise RigCalibrationError("rig 'wrist' is eye_in_hand: a wrist camera has no fixed "
                                      "CAMERA to BASE; ask its frames for the tool pose.")
        return _camera_to_base()


class _Rig:
    """Stands in for the rig config. `OneCamera.is_rgbd` asks its type, as `FrameProvider` does."""


class _Camera:
    def __init__(self, *, rig_id: str = "overhead", intrinsics: Any = _K,
                 calibration: Any = None, rig: Any = None, frame: Any = None) -> None:
        self.rig_id = rig_id
        self.rig = rig if rig is not None else _Rig()
        self._intrinsics = intrinsics
        self._calibration = calibration if calibration is not None else _Calibration(fixed=True)
        self._frame = frame
        self.grabs = 0

    def get_intrinsics(self) -> Any:
        return self._intrinsics

    def calibration(self) -> Any:
        if self._calibration is None:                      # pragma: no cover - set in every case
            raise RigNotCalibrated("rig declares no calibration")
        return self._calibration

    def grab(self) -> Any:
        self.grabs += 1
        return self._frame


def _rgbd_camera(**kwargs: Any) -> _Camera:
    from src.config.schema.camera import RGBDDeviceRigConfig

    rig = RGBDDeviceRigConfig.model_construct()
    return _Camera(rig=rig, **kwargs)


class OneCameraServesTheRigSurfaceTests(unittest.TestCase):

    def test_it_holds_exactly_the_one_rig(self) -> None:
        self.assertEqual(["overhead"], OneCamera(_Camera()).rig_ids)

    def test_a_grab_goes_to_that_camera_and_opens_nothing(self) -> None:
        camera = _Camera(frame="a frame")
        frames = OneCamera(camera)
        self.assertEqual("a frame", frames.grab("overhead"))
        self.assertEqual(1, camera.grabs)

    def test_another_rig_id_is_refused_rather_than_answered_from_this_one(self) -> None:
        """One camera's frames under another rig's name is how a confident wrong answer is made."""
        with self.assertRaises(KeyError) as caught:
            OneCamera(_Camera()).grab("wrist")
        self.assertIn("overhead", str(caught.exception))

    def test_it_reports_rgbd_from_the_rigs_own_type(self) -> None:
        self.assertTrue(OneCamera(_rgbd_camera()).is_rgbd("overhead"))
        self.assertFalse(OneCamera(_Camera()).is_rgbd("overhead"))

    def test_a_stereo_index_is_refused_rather_than_guessed_as_zero(self) -> None:
        """Returning 0 would triangulate against whichever calibration sat first in the file."""
        with self.assertRaises(ValueError) as caught:
            OneCamera(_Camera()).get_stereo_rig_index("overhead")
        self.assertIn("StereoCam3D", str(caught.exception))


class TheCameraDoorReadsTheCamerasOwnFactsTests(unittest.TestCase):

    def _finder(self, camera: Any) -> Any:
        return build_hand_finder_on_camera(
            HandDetectConfig(model_path=__file__), camera, observer=object())

    def test_the_transform_is_the_rigs_declared_camera_to_base(self) -> None:
        finder = self._finder(_rgbd_camera())
        np.testing.assert_allclose(_camera_to_base().to_matrix(), finder.transforms["overhead"])

    def test_the_intrinsics_are_the_devices_own(self) -> None:
        np.testing.assert_allclose(_K, self._finder(_rgbd_camera()).camera_matrices["overhead"])

    def test_the_search_reaches_that_rig_and_no_other(self) -> None:
        finder = self._finder(_rgbd_camera())
        self.assertEqual(["overhead"], finder.rig_ids)
        self.assertIsInstance(finder.provider, OneCamera)

    def test_the_depth_knobs_still_travel(self) -> None:
        config = HandDetectConfig(model_path=__file__, palm_patch_radius_px=33, min_depth_samples=7)
        finder = build_hand_finder_on_camera(config, _rgbd_camera(), observer=object())
        self.assertEqual(33, finder.palm_patch_radius_px)
        self.assertEqual(7, finder.min_depth_samples)


class WhatTheCameraDoorRefusesTests(unittest.TestCase):

    def test_a_wrist_rig_is_refused_by_name_rather_than_composed_here(self) -> None:
        """`Locator` owns `tool_to_base @ camera_to_tool`. A second spelling here is the defect."""
        camera = _rgbd_camera(calibration=_Calibration(fixed=False))
        with self.assertRaises(RigCalibrationError) as caught:
            build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera,
                                        observer=object())
        self.assertIn("eye_in_hand", str(caught.exception))

    def test_an_rgbd_rig_with_no_intrinsics_refuses_instead_of_inventing_a_lens(self) -> None:
        """A back-projection through an invented K completes and returns an unmeasured position."""
        camera = _rgbd_camera(intrinsics=None)
        with self.assertRaises(ValueError) as caught:
            build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera,
                                        observer=object())
        self.assertIn("no intrinsics", str(caught.exception))

    def test_an_uncalibrated_rig_refuses_through_its_own_error(self) -> None:
        class _Uncalibrated:
            def camera_to_base(self) -> Transform:
                raise RigNotCalibrated("rig 'overhead' declares no calibration.")

        camera = _rgbd_camera(calibration=_Uncalibrated())
        with self.assertRaises(RigNotCalibrated):
            build_hand_finder_on_camera(HandDetectConfig(model_path=__file__), camera,
                                        observer=object())


class TheDoorIsExportedTests(unittest.TestCase):

    def test_willy_exports_it(self) -> None:
        import willy

        self.assertIs(build_hand_finder_on_camera, willy.build_hand_finder_on_camera)

    def test_the_package_re_exports_it(self) -> None:
        import src.models.handdetection as package

        self.assertIn("build_hand_finder_on_camera", package.__all__)
        self.assertIs(build_hand_finder_on_camera, package.build_hand_finder_on_camera)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
