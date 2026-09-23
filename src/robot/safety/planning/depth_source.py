"""A camera rig asked for depth and nothing else: the view of a camera the world source takes.

It lives beside the world it feeds, in the safety planning layer, because that world can be handed to
`Robot`, and `Robot` may load neither `execution.autonomous_grasp` nor the camera package. So it knows
a camera only as a handle it can call `grab()` and `get_intrinsics()` on, and imports nothing from
`src.camera`.

A camera on the wrist also reads the arm, through `get_tcp_pose`, the method every arm has,
immediately before and after its grab. Its frame is placed by the pose read before, and a frame taken
while the tool moved further than the rig allows is answered as no frame.

A camera on the wrist has moved since its last frame, and the first frame its stream hands back is not
what it sees now. The pipeline returns a frame it had already queued, which can have been exposed while
the arm was still coming to rest, before the pose was read. And a temporal filter holds the frames
before it: it fills a hole with the depth the pixel had at the pose before and pulls a depth toward it.
So after the pose read the wrist source tells the handle the camera moved (``camera_moved()``, where the
handle has it: a RealSense drops its temporal history), then grabs and throws away
:data:`WRIST_WARMUP_GRABS` frames, as the pick frame does. The frame it keeps was then exposed after the
pose read and filtered with frames from where the camera stands, and the pose read after it bounds how
far the tool moved before its shutter closed. That travel, measured, is what the reading declares as its
placement error.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Pose
from src.robot.core.shutter_motion import ShutterMotion
from src.robot.safety.planning.live_world import DepthSnapshot

__all__ = ["WRIST_WARMUP_GRABS", "RigDepthSource"]

#: Frames a wrist camera's depth source grabs and throws away after it reads the pose and before the frame
#: it keeps.
#:
#: Five, as the pick frame takes (``RealSenseVisionPerceptionSource``). Measured 2026-09-23 in the audit of
#: the owner's cell, against librealsense 2.58.3's own temporal filter at its defaults (alpha 0.4, delta
#: 20, persistency "valid in 2 of the last 4") on frames fed through a software device: after a move, a
#: hole the first frame filled with the depth of the pose before was a hole again from the fourth frame on,
#: and a depth pulled toward the pose before keeps at most 0.6 to the power of the frames taken of that
#: pull. The drain comes with it: a frame the pipeline queued during the move is among the ones thrown
#: away. It costs five frame periods on every grab after a move, about 170 ms at 30 frames a second.
WRIST_WARMUP_GRABS = 5


class RigDepthSource:
    """A camera rig, asked for depth and nothing else.

    Deliberately narrower than the perception source the pick loop drives. That one runs a detector
    and a segmenter to produce named objects, which is the expensive part of a cell and exactly what
    a world refresh does not need: a planner wants to know where geometry is, not what it is called.

    Returns `None` rather than raising, for every reason a camera can fail to answer. The caller is
    about to decide whether a motion may happen, and a decision needs a verdict rather than an
    exception: the world source turns a `None` into a refusal that names the camera.

    ``tool_pose`` and ``motion_tolerance`` make it the source of a camera on the wrist, and come
    together: a pose read with no bound on how far the tool may move during the grab is a pose nobody
    checked. The tolerance is the rig's ``(shutter_motion_tolerance_mm, shutter_motion_tolerance_deg)``,
    the most the tool may move across a grab before the frame is refused.

    Every reading declares the travel it measured across its grab as its placement error, at most
    the tolerance: the frame is placed by the pose read before the grab, and the tool moved that far
    before the pose read after it, so the world widens its bench band by that rather than taking the
    bench seen that far off for an obstacle. Not the tolerance itself: an arm standing still moves
    about nothing, and a band grown by a threshold it never came near drops a part that low
    (review of 2026-09-23: 1 mm and 0.5 degrees at 500 mm grew the band from 5 mm to 10-16 mm, and
    a plate 8 mm tall left the world). The camera's calibration error is not the tool's motion and
    is not declared here: ``perceived.plane_clearance_mm`` holds it.

    ``warmup_grabs`` is how many frames are thrown away between the pose read and the frame kept:
    :data:`WRIST_WARMUP_GRABS` for a camera on the wrist, which moved since its last frame, and none for
    a fixed camera, which did not, when left unset.
    """

    __slots__ = ("_handle", "_motion_tolerance", "_name", "_tool_pose", "_warmup_grabs")

    def __init__(
        self,
        handle: Any,
        *,
        name: str = "",
        tool_pose: Maybe[Callable[[], Pose]] = UNSET,
        motion_tolerance: Maybe[tuple[float, float]] = UNSET,
        warmup_grabs: Maybe[int] = UNSET,
    ) -> None:
        if chosen(tool_pose) != chosen(motion_tolerance):
            raise ValueError(
                "a wrist camera's depth source takes the arm's TCP reader and the rig's shutter motion "
                "tolerance together, or neither: a pose read with no bound on how far the tool may move "
                "during the grab is a pose nobody checked"
            )
        if not chosen(warmup_grabs):
            warmups = WRIST_WARMUP_GRABS if chosen(tool_pose) else 0
        elif isinstance(warmup_grabs, bool) or not isinstance(warmup_grabs, int) or warmup_grabs < 0:
            raise ValueError(f"warmup_grabs is a whole number of frames, 0 or more, not {warmup_grabs!r}")
        else:
            warmups = int(warmup_grabs)
        self._warmup_grabs = warmups
        self._handle = handle
        self._name = name or getattr(handle, "rig_id", "camera")
        self._tool_pose = tool_pose
        self._motion_tolerance = motion_tolerance

    def __repr__(self) -> str:
        return f"RigDepthSource({self._name!r})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def warmup_grabs(self) -> int:
        """How many frames each reading throws away between the pose read and the frame it keeps."""
        return self._warmup_grabs

    def grab_surface_depth(self) -> "DepthSnapshot | None":
        """One depth reading with the time its shutter opened, or `None` when this rig cannot answer.

        In order: the tool pose, on a wrist camera the handle's ``camera_moved()`` where it has one, the
        warm-up frames thrown away, the grab kept, the tool pose again. The stamp is the one the rig's owner
        took before the kept grab, under the rig's lock, when the frame carries it, and otherwise one taken
        here before the kept grab. Either way it is taken before the grab rather than after. Age is measured
        to decide whether the world can still be planned against, and a stamp taken after a slow read
        reports a frame as fresher than it is, which is the one direction that matters. It is a stamp of the
        call and not of the exposure: the exposure can begin up to a frame period before it, and after the
        warm-ups never before the first pose read.
        """
        tool_pose = self._tool_pose
        reader = tool_pose if chosen(tool_pose) else None
        try:
            before = None if reader is None else reader()
            moved = getattr(self._handle, "camera_moved", None) if reader is not None else None
            if callable(moved):
                # The arm moved since this camera's last frame: its temporal history is from the pose before.
                moved()
            for _ in range(self._warmup_grabs):
                self._handle.grab()  # thrown away: a frame queued during the move, or filtered with the pose before
            stamped = time.time()
            frame = self._handle.grab()
            after = None if reader is None else reader()
            intrinsics = self._handle.get_intrinsics()
        except Exception:  # noqa: BLE001 (a camera that cannot answer is a refusal, not a crash)
            return None

        depth = getattr(frame, "depth", None)
        if depth is None or intrinsics is None:
            # A stereo rig has no depth of its own and an uncalibrated one has no matrix. Both are
            # states a cell can be in, and neither is something to guess a way around.
            return None
        array = np.asarray(depth, dtype=np.float64)
        if array.ndim != 2 or array.size == 0:
            return None

        tool_to_base: np.ndarray | None = None
        error_mm = error_deg = 0.0
        if reader is not None:
            motion = self._motion(before, after)
            if motion is None or not motion.within or not isinstance(before, Pose):
                # Taken somewhere between two poses, so no one pose places it. The world source asks
                # again, and after its attempts every refreshing motion stops, naming this camera.
                return None
            tool_to_base = before.to_matrix()
            # Measured, so inside the tolerance: a grab the tool moved further across gave no frame.
            error_mm, error_deg = motion.moved_mm, motion.turned_deg

        captured = getattr(frame, "captured_at_s", None)
        return DepthSnapshot(
            depth_mm=array,
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            timestamp=float(captured) if isinstance(captured, (int, float)) else stamped,
            tool_to_base_mm=tool_to_base,
            placement_error_mm=float(error_mm),
            placement_error_deg=float(error_deg),
        )

    def _motion(self, before: object, after: object) -> "ShutterMotion | None":
        """The tool's travel across the grab and whether it stayed inside the rig's tolerance, by the rule the pick
        frame uses too, or ``None`` without a tolerance.

        Both poses have to be read in BASE (``ShutterMotion``).
        """
        tolerance = self._motion_tolerance
        if not chosen(tolerance):
            return None
        max_mm, max_deg = tolerance
        return ShutterMotion.between(before, after, tolerance_mm=max_mm, tolerance_deg=max_deg)

    def _held_still(self, before: object, after: object) -> bool:
        """Whether the tool stayed inside the rig's tolerance across the grab (:meth:`_motion`)."""
        motion = self._motion(before, after)
        return motion is not None and motion.within
