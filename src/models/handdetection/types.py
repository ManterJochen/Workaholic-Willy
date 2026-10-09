"""Value objects for hand detection: a 2-D detection, a 3-D position, and a gesture reading.

Frozen dataclasses that validate in `__post_init__`. Each one crosses the boundary between a C++
inference graph and geometry code that propagates a NaN into a robot pose without complaining, so
the validation rejects non-finite and wrong-shaped values at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np

from src.models.handdetection.landmarks import LANDMARK_COUNT

__all__ = [
    "GestureReading",
    "HandGesture",
    "HandObservation",
    "HandPosition3D",
    "Handedness",
    "LocatedHand",
    "PalmDetection",
    "as_landmark_tuple",
]


class Handedness(StrEnum):
    """Which hand MediaPipe reported.

    The label is from the camera's point of view on a non-mirrored image. This enum records what
    the model said; a caller that mirrors the frame for display has to reinterpret it.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class HandGesture(StrEnum):
    """What a gesture reading may be.

    Attributes:
        THUMB_UP: A thumb up.
        THUMB_DOWN: A thumb down.
        OTHER: A hand, and no gesture of the two.
        NONE: No hand at all; not the same as ``OTHER``.
    """

    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    OTHER = "other"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PalmDetection:
    """One hand found in one frame, in image space.

    `landmarks` holds exactly 21 integer pixel coordinates in MediaPipe's layout. `palm_center_xy`
    is the mean over the palm subset of those, and is float because that mean is sub-pixel.
    """

    palm_center_xy: tuple[float, float]
    landmarks: tuple[tuple[int, int], ...]
    hand_index: int = 0
    handedness: Handedness = Handedness.UNKNOWN
    #: MediaPipe's confidence in the handedness label only, not in the detection. A low value here
    #: is the usual explanation for a left/right call that looks wrong.
    handedness_score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"PalmDetection.landmarks must have {LANDMARK_COUNT} entries (MediaPipe layout), "
                f"got {len(self.landmarks)}"
            )
        for i, landmark in enumerate(self.landmarks):
            if not (isinstance(landmark, tuple) and len(landmark) == 2):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] must be an (x, y) tuple, got {landmark!r}"
                )
            x, y = landmark
            if not (isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer))):
                raise ValueError(
                    f"PalmDetection.landmarks[{i}] entries must be ints, got "
                    f"{(type(x).__name__, type(y).__name__)}"
                )
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"PalmDetection.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        cx, cy = self.palm_center_xy
        if not (np.isfinite(cx) and np.isfinite(cy)):
            raise ValueError(
                f"PalmDetection.palm_center_xy must be finite, got {self.palm_center_xy!r}"
            )
        if self.hand_index < 0:
            raise ValueError(f"PalmDetection.hand_index must be >= 0, got {self.hand_index}")


@dataclass(frozen=True, slots=True)
class GestureReading:
    """What the canned classifier said about one hand.

    `raw_label` carries the classifier's own category name even when `gesture` is `OTHER` or
    `NONE`, so the shapes that map to neither thumb stay distinguishable from each other.
    """

    gesture: HandGesture
    confidence: float
    raw_label: str = ""
    hand_index: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence):
            raise ValueError(f"GestureReading.confidence must be finite, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"GestureReading.confidence must be in [0, 1], got {self.confidence!r}"
            )
        if self.hand_index < 0:
            raise ValueError(f"GestureReading.hand_index must be >= 0, got {self.hand_index}")


@dataclass(frozen=True, slots=True)
class HandObservation:
    """One hand seen once: where it is in the image, and what it was doing.

    The canned gesture model embeds a hand landmarker and returns both halves at once. Pairing them
    in one object keeps a caller from aligning two parallel lists by index and crossing the hands.
    """

    palm: PalmDetection
    gesture: GestureReading


@dataclass(frozen=True, slots=True)
class HandPosition3D:
    """A hand's palm centre in metric space: camera frame and robot base frame, millimetres.

    Both arrays are coerced to `float64` of shape `(3,)` and marked read-only. These are handed to
    motion code, where an in-place write would alter a pose another caller still holds; the
    read-only flag turns such a write into a `ValueError` at the point it is attempted.
    """

    position_base: np.ndarray
    position_cam: np.ndarray
    palm_center_xy: tuple[float, float]
    depth_mm: float
    rig_id: str = ""
    handedness: Handedness = Handedness.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("position_base", "position_cam"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.shape != (3,):
                raise ValueError(
                    f"HandPosition3D.{name} must have shape (3,), got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"HandPosition3D.{name} must be finite, got {array!r}")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if len(self.palm_center_xy) != 2:
            raise ValueError(
                f"HandPosition3D.palm_center_xy must be (x, y), got {self.palm_center_xy!r}"
            )
        if not np.isfinite(self.depth_mm):
            raise ValueError(f"HandPosition3D.depth_mm must be finite, got {self.depth_mm!r}")
        if self.depth_mm <= 0.0:
            raise ValueError(
                f"HandPosition3D.depth_mm must be positive: a hand behind the camera is a "
                f"back-projection bug, not a detection: got {self.depth_mm!r}"
            )


@dataclass(frozen=True, slots=True)
class LocatedHand:
    """The complete answer about one hand: where it is in metric space, and what it was doing.

    One object rather than a tuple: the three parts always travel together, and unpacking them into
    loose variables invites pairing a position with another hand's gesture.
    """

    position: HandPosition3D
    gesture: GestureReading
    palm: PalmDetection


def as_landmark_tuple(landmarks: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Freeze a landmark sequence for `PalmDetection`.

    A frozen dataclass holding a `list` is frozen in name only, so construction sites convert here
    instead of passing the sequence through.
    """
    return tuple((int(x), int(y)) for x, y in landmarks)
