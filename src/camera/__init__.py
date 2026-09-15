"""Camera rigs: device setup, the capture pipeline, and the frames a caller streams."""

from __future__ import annotations

from .orchestration import (
    Camera,
    CameraBusy,
    FrameProvider,
    FrameProviderStateError,
    UnknownCameraRigError,
)
from .pipeline import StereoCapturePipeline
from .setup.image_taking import AnyFrame, RGBDFrame, StereoFrame

__all__ = [
    "AnyFrame",
    "Camera",
    "CameraBusy",
    "FrameProvider",
    "FrameProviderStateError",
    "RGBDFrame",
    "StereoCapturePipeline",
    "StereoFrame",
    "UnknownCameraRigError",
]
