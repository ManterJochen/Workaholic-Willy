"""Camera rigs: device setup, the capture pipeline, and the frames a caller streams.

``Camera`` owns one rig's device, and a ``with`` block opens and releases it. Its refusals are
exported beside it: ``CameraRefused`` (the section gives no usable rig: unknown, switched off,
a stereo pair not yet calibrated, or on a capture backend this platform lacks), ``CameraBusy``
(another owner holds the device), ``CameraNotOpen`` (a read before ``open``), and the two raises
of ``camera.calibration()``, ``RigNotCalibrated`` and ``RigCalibrationError``, which live in
``src.calibration.rig_calibration``. ``StereoCalibration`` solves a stereo pair against its
ChArUco board, which is what makes the pair a camera that gives depth.
"""

from __future__ import annotations

from src.calibration.rig_calibration import RigCalibrationError, RigNotCalibrated

from .orchestration import (
    Camera,
    CameraBusy,
    CameraNotOpen,
    CameraRefused,
    FrameProvider,
    FrameProviderStateError,
    UnknownCameraRigError,
)
from .setup.image_taking import AnyFrame, RGBDFrame, StereoFrame
from .stereo_calibration import StereoCalibration, StereoCalibrationReport

__all__ = [
    "AnyFrame",
    "Camera",
    "CameraBusy",
    "CameraNotOpen",
    "CameraRefused",
    "FrameProvider",
    "FrameProviderStateError",
    "RGBDFrame",
    "RigCalibrationError",
    "RigNotCalibrated",
    "StereoCalibration",
    "StereoCalibrationReport",
    "StereoFrame",
    "UnknownCameraRigError",
]
