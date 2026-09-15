from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    """Colour (BGR uint8) and depth (uint16, millimetres) from an RGB-D camera.

    Depth may be an empty array when the device delivered none. A depth array that is not
    empty must match the colour frame in height and width. ``captured_at_s`` is the host time the
    camera owner read just before the device grab, or None for a frame that did not come through an
    owner, such as one from a streamer used directly.
    """

    color: np.ndarray
    depth: np.ndarray
    captured_at_s: float | None = None

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
