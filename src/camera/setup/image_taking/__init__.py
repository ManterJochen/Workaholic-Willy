"""The frame dataclasses every rig returns, and the streamers that produce them."""

from __future__ import annotations

from .frames import AnyFrame, CameraFacts, ResearchCapture, RGBDFrame, StereoFrame
from .rgbd import OpenCvRGBDStreamer, RealSenseRGBDStreamer, RGBDStreamerProtocol
from .single import SingleDeviceStreamer
from .stereo_depth import StereoDepthStreamer, StereoNotCalibrated
from .webcam import WebcamPairStreamer

__all__ = [
	"AnyFrame",
	"CameraFacts",
	"OpenCvRGBDStreamer",
	"RGBDFrame",
	"RGBDStreamerProtocol",
	"RealSenseRGBDStreamer",
	"ResearchCapture",
	"SingleDeviceStreamer",
	"StereoDepthStreamer",
	"StereoFrame",
	"StereoNotCalibrated",
	"WebcamPairStreamer",
]
