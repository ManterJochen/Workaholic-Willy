"""Entry points for the owner of one camera rig, the rig-keyed catalogue over those owners, and the
errors they raise."""

from __future__ import annotations

from .camera import Camera, CameraBusy, CameraNotOpen, CameraRefused
from .frame_provider import FrameProvider, FrameProviderStateError, RigHandle, UnknownCameraRigError

__all__ = [
    "Camera",
    "CameraBusy",
    "CameraNotOpen",
    "CameraRefused",
    "FrameProvider",
    "FrameProviderStateError",
    "RigHandle",
    "UnknownCameraRigError",
]
