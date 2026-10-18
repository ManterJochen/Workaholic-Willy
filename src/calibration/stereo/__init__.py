"""Stereo calibration, rectification, reconstruction and marker pose.

`StereoCam3D` is the entry point of the stereo tools: it builds one `StereoRigRunTime` per
configured rig through `StereoRigFactory` and addresses them by index. `StereoDepth` is a
calibrated pair as the pick path takes it, an RGB-D camera: depth on the rectified left image,
searched over the depths `working_depth_range_mm` gives.
"""

from __future__ import annotations

from .config import CalibrationResult, StereoRigConfig
from .factory import StereoRigFactory
from .manager import StereoCam3D
from .repository import StereoRigRepository
from .runtime import StereoRigRunTime
from .sub_modules.depth import NEAR_LIMIT_MM, StereoDepth, working_depth_range_mm

__all__ = [
	"NEAR_LIMIT_MM",
	"CalibrationResult",
	"StereoCam3D",
	"StereoDepth",
	"StereoRigConfig",
	"StereoRigFactory",
	"StereoRigRepository",
	"StereoRigRunTime",
	"working_depth_range_mm",
]
