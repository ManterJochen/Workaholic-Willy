from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_frame_array(value: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3) and not (allow_empty and array.shape == (0,)):
        raise ValueError(f"{name} must be a 2-D or 3-D image array, got shape {array.shape}")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.size and not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must use a numeric dtype, got {array.dtype}")
    return array


@dataclass(frozen=True, slots=True)
class StereoFrame:
    """A left and a right BGR frame from a stereo rig.

    Both eyes must be non-empty numeric image arrays of the same height and width.
    ``captured_at_s`` is the host time the camera owner read just before the device grab, or None
    for a frame that did not come through an owner, such as one from a streamer used directly.
    """

    left: np.ndarray
    right: np.ndarray
    captured_at_s: float | None = None

    def __post_init__(self) -> None:
        left = _validate_frame_array(self.left, name="left")
        right = _validate_frame_array(self.right, name="right")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"left/right frame sizes differ: {left.shape[:2]} vs {right.shape[:2]}")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True, slots=True, eq=False)
class CameraFacts:
    """What a camera was while it recorded for research, read once as it opened (``realsense.record_for_research``).

    ``said`` is plain data a JSON writer takes: the device (name, serial, firmware, USB link), the stream modes, the
    visual preset, emitter and laser power the depth sensor holds, the depth filters in the order they run with each
    one's options, the colour sensor's exposure, gain and white balance as the rig asked for them and as the sensor
    holds them, and each lens's distortion model. ``depth_units_m`` is metres per raw depth unit as the device reads it
    back. ``intrinsics`` and ``distortion`` hold each stream's 3x3 camera matrix, in pixels, and its five distortion
    coefficients, by stream: ``color``, ``depth``, ``ir_left``, ``ir_right``. ``extrinsics_mm`` holds where one
    stream's camera sits in another's as the device reports it, a 4x4 in millimetres that maps a point of the first
    stream's frame into the second's: ``depth_to_color``, ``ir_left_to_color`` and ``ir_left_to_ir_right``.
    """

    said: Mapping[str, Any]
    depth_units_m: float
    intrinsics: Mapping[str, np.ndarray] = field(default_factory=dict)
    distortion: Mapping[str, np.ndarray] = field(default_factory=dict)
    extrinsics_mm: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class ResearchCapture:
    """What a camera recording for research hands with one frame, beside its colour and its depth (the owner,
    2026-10-09: the research recording, ``realsense.record_for_research``).

    ``ir_left`` and ``ir_right`` are the two infrared images (uint8, the depth stream's size) of the frameset the colour
    and the depth came from, taken with the projector as the camera runs. ``raw_depth`` is the depth as the sensor
    streamed it (uint16 in device units, ``camera.depth_units_m`` metres each, 0 where nothing was measured), before the
    filters and the alignment, so in the left infrared camera's frame, the depth's own. Each is ``None`` where the
    frameset carried none. ``color_metadata`` is what the colour frame's metadata says of how it was exposed
    (``actual_exposure``, ``gain_level``, ``white_balance``, ``auto_exposure``), the values the device reports.
    ``camera`` is the camera they were taken with, one object for every frame of one open. The arrays are copies: the
    device's buffers are not held.
    """

    camera: CameraFacts
    ir_left: np.ndarray | None = None
    ir_right: np.ndarray | None = None
    raw_depth: np.ndarray | None = None
    color_metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    """Colour and depth from one grab of an RGB-D camera, aligned pixel for pixel.

    Attributes:
        color (np.ndarray): Height x width x 3 ``uint8``, BGR.
        depth (np.ndarray): Height x width ``uint16``, millimetres; 0 where the sensor measured nothing.
        captured_at_s (float | None): The host time read just before the grab, seconds (default: None).
        research (ResearchCapture | None): With ``record_for_research``: both IR images, the raw depth and the camera
            facts; ``None`` otherwise (default: None).
    """

    color: np.ndarray
    depth: np.ndarray
    captured_at_s: float | None = None
    research: ResearchCapture | None = None

    def __post_init__(self) -> None:
        color = _validate_frame_array(self.color, name="color")
        depth = _validate_frame_array(self.depth, name="depth", allow_empty=True)
        if depth.size and depth.shape[:2] != color.shape[:2]:
            raise ValueError(f"color/depth frame sizes differ: {color.shape[:2]} vs {depth.shape[:2]}")
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "depth", depth)


# Either frame kind a rig-keyed provider or consumer handles: a StereoFrame carrying two eyes, or
# an RGBDFrame carrying colour and depth. Used where a return type or a parameter takes both.
AnyFrame = StereoFrame | RGBDFrame
