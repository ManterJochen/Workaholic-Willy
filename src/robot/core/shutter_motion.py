"""How far the tool moved while a camera's shutter was open, whether a rig tolerates it, and where the camera stood.

A camera on the wrist is placed frame by frame by where the tool stood when its shutter opened. The pose is read
before and after the grab, and a frame taken while the tool moved beyond the rig's tolerance is placed by no single
pose. One rule decides that for the pick frame (``RealSenseVisionPerceptionSource``) and for the live planner world
(``RigDepthSource``), so the two cannot disagree about whether the arm held still.

The pick frame's stamp is :class:`ShutterStamp`: the pose read before and after the grab, the grab taken again while
the tool moved, and the frame placed by the pose read before it. The hand finder over a wrist camera takes its frames
through the same stamp (``models.handdetection.OneWristCamera``). Where a frame then stood in BASE is
:func:`camera_to_base_at_shutter`, the one composition of a frame's stamped tool pose with its camera's calibration
that the ``Locator`` and the hand finder share: both place every frame by it, so a part and a palm seen at one pixel
of one frame are placed at one point.

At import this loads nothing from ``src.calibration``, ``src.camera`` or ``src.models``: a calibration is read by its
methods.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import angle_between

__all__ = ["PICK_FRAME_WARMUP_GRABS", "ShutterMotion", "ShutterStamp", "StampedGrab", "camera_to_base_at_shutter"]

#: Frames a pick frame grabs and throws away before the frame it keeps (``RealSenseVisionPerceptionSource``'s
#: ``warmup_grabs``), so a physical camera's auto-exposure has settled. On a camera the arm carries they come before
#: the pose read, so a frame the device queued while the arm was still coming to rest is among them, not the one kept.
PICK_FRAME_WARMUP_GRABS = 5


@dataclass(frozen=True, slots=True)
class ShutterMotion:
    """The tool's travel across one grab, and whether it stayed inside a tolerance."""

    moved_mm: float
    turned_deg: float
    within: bool

    @classmethod
    def between(cls, before: object, after: object, *, tolerance_mm: float, tolerance_deg: float) -> "ShutterMotion":
        """The motion from ``before`` to ``after``. A pose that is not a BASE ``Pose`` is never within: no pose places it."""
        if (not isinstance(before, Pose) or not isinstance(after, Pose)
                or before.frame is not Frame.BASE or after.frame is not Frame.BASE):
            return cls(moved_mm=math.inf, turned_deg=math.inf, within=False)
        moved = float(np.linalg.norm(after.position_mm - before.position_mm))
        turned = float(np.degrees(angle_between(before.quaternion_xyzw, after.quaternion_xyzw)))
        return cls(moved_mm=moved, turned_deg=turned,
                   within=moved <= float(tolerance_mm) and turned <= float(tolerance_deg))

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        state = "held still" if self.within else "moved"
        return f"tool {state} across the grab: {self.moved_mm:.3f} mm, {self.turned_deg:.3f} deg"


@dataclass(frozen=True, slots=True)
class StampedGrab:
    """One frame of a camera on the wrist and the tool pose that places it, or none where the tool never held still."""

    #: What the last grab returned: the frame kept, or, where the tool moved across every grab, the last one taken.
    frame: Any
    #: The tool pose read just before the kept grab, in BASE: where the tool stood when its shutter opened. ``None``
    #: where the tool moved beyond the tolerance across every grab, so no single pose places the frame.
    tool_pose: Pose | None
    #: The clock (``time.time``) read just before the last grab, after the pose read before it.
    before_grab_s: float
    #: The tool's travel across the last grab.
    motion: ShutterMotion
    #: How many grabs were taken, the first included.
    grabs: int


@dataclass(frozen=True, slots=True)
class ShutterStamp:
    """How a frame of a camera on the wrist is taken: the arm's TCP reader, the rig's shutter tolerance, and how many
    more grabs a frame taken while the tool moved gets.

    The pick frame's stamp (``RealSenseVisionPerceptionSource.stamp_tool_pose_with``), and the hand finder's over a
    wrist camera. Build it with :meth:`of`, which refuses a reader that cannot be called, a tolerance that is not two
    finite non-negative numbers, and negative attempts.
    """

    reader: Callable[[], object]
    tolerance_mm: float
    tolerance_deg: float
    attempts: int

    @classmethod
    def of(cls, reader: Callable[[], object], *, motion_tolerance: tuple[float, float], attempts: int) -> "ShutterStamp":
        """The stamp of ``reader`` (the arm's ``get_tcp_pose``), the rig's ``(shutter_motion_tolerance_mm,
        shutter_motion_tolerance_deg)`` and ``attempts``, the grabs after the first."""
        if not callable(reader):
            raise TypeError(f"a shutter stamp takes the arm's TCP reader, not {type(reader).__name__}")
        max_mm, max_deg = (float(v) for v in motion_tolerance)
        if not (math.isfinite(max_mm) and math.isfinite(max_deg)) or max_mm < 0.0 or max_deg < 0.0:
            raise ValueError(f"a shutter motion tolerance is two finite non-negative numbers, not {motion_tolerance!r}")
        if int(attempts) < 0:
            raise ValueError(f"attempts counts the grabs after the first and cannot be negative, not {attempts!r}")
        return cls(reader=reader, tolerance_mm=max_mm, tolerance_deg=max_deg, attempts=int(attempts))

    def grab(self, grab: Callable[[], Any]) -> StampedGrab:
        """One frame through ``grab``, with the tool pose read immediately before and after it.

        A grab the tool moved across beyond the tolerance (:class:`ShutterMotion`) is taken again, with nothing thrown
        away in between, up to :attr:`attempts` more times. The frame is placed by the pose read before its grab. Where
        the tool moved across every grab the last frame comes back with no pose, and the caller says what that means:
        the pick frame raises ``PerceptionFrameMoved``, the hand finder places no hand.
        """
        motion = ShutterMotion(moved_mm=math.inf, turned_deg=math.inf, within=False)
        frame: Any = None
        before_grab = math.nan
        for count in range(1, 2 + self.attempts):
            before = self.reader()
            before_grab = time.time()
            frame = grab()
            after = self.reader()
            motion = ShutterMotion.between(before, after, tolerance_mm=self.tolerance_mm,
                                           tolerance_deg=self.tolerance_deg)
            if motion.within:
                assert isinstance(before, Pose)  # within only between two BASE poses
                return StampedGrab(frame=frame, tool_pose=before, before_grab_s=before_grab, motion=motion,
                                   grabs=count)
        return StampedGrab(frame=frame, tool_pose=None, before_grab_s=before_grab, motion=motion,
                           grabs=1 + self.attempts)


def camera_to_base_at_shutter(calibration: Any, tool_pose: Pose | None) -> np.ndarray:
    """Where a camera stood when one frame's shutter opened: its CAMERA to BASE for that frame, 4x4 millimetres.

    A camera on the wrist has no CAMERA to BASE of its own. Its calibration is CAMERA to TOOL, and the frame is placed
    by the tool pose stamped at its shutter (:class:`ShutterStamp`), ``tool_to_base @ camera_to_tool``. A fixed camera
    is placed by its calibration's CAMERA to BASE, the same for every frame: pass ``None`` for it. This is the one
    composition the ``Locator`` and the hand finder share: both place every frame here, so the two cannot drift apart.

    ``calibration`` answers as ``RigCalibration`` does. A wrist calibration handed no pose raises, because
    ``RigCalibration.camera_to_base`` refuses a camera the arm carries: a frame nobody stamped is never placed. A pose
    that is not a ``Pose`` in BASE raises ``ValueError``, because no pose places that frame.
    """
    if tool_pose is None:
        return np.asarray(calibration.camera_to_base().to_matrix(), dtype=np.float64)
    if not isinstance(tool_pose, Pose) or tool_pose.frame is not Frame.BASE:
        where = tool_pose.frame.value if isinstance(tool_pose, Pose) else type(tool_pose).__name__
        raise ValueError(f"a frame is placed by the tool pose at its shutter in BASE, and this one is {where}")
    tool_to_base = np.asarray(tool_pose.to_matrix(), dtype=np.float64)
    return tool_to_base @ np.asarray(calibration.camera_to_tool().to_matrix(), dtype=np.float64)
