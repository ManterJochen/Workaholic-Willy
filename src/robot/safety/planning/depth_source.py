"""A camera rig asked for depth and nothing else: the view of a camera the world source takes.

It lives beside the world it feeds, in the safety planning layer, because that world can be handed to
`Robot`, and `Robot` may load neither `execution.autonomous_grasp` nor the camera package. So it knows
a camera only as a handle it can call `grab()` and `get_intrinsics()` on, and imports nothing from
`src.camera`.

A camera on the wrist also reads the arm, through `get_tcp_pose`, the method every arm has,
immediately before and after its grab. Its frame is placed by the pose read before, and a frame taken
while the tool moved further than the rig allows is answered as no frame.
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

__all__ = ["RigDepthSource"]


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
    checked. The tolerance is the rig's ``(shutter_motion_tolerance_mm, shutter_motion_tolerance_deg)``.
    """

    __slots__ = ("_handle", "_motion_tolerance", "_name", "_tool_pose")

    def __init__(
        self,
        handle: Any,
        *,
        name: str = "",
        tool_pose: Maybe[Callable[[], Pose]] = UNSET,
        motion_tolerance: Maybe[tuple[float, float]] = UNSET,
    ) -> None:
        if chosen(tool_pose) != chosen(motion_tolerance):
            raise ValueError(
                "a wrist camera's depth source takes the arm's TCP reader and the rig's shutter motion "
                "tolerance together, or neither: a pose read with no bound on how far the tool may move "
                "during the grab is a pose nobody checked"
            )
        self._handle = handle
        self._name = name or getattr(handle, "rig_id", "camera")
        self._tool_pose = tool_pose
        self._motion_tolerance = motion_tolerance

    def __repr__(self) -> str:
        return f"RigDepthSource({self._name!r})"

    @property
    def name(self) -> str:
        return self._name

    def grab_surface_depth(self) -> "DepthSnapshot | None":
        """One depth reading with the time its shutter opened, or `None` when this rig cannot answer.

        The stamp is the one the rig's owner took before the grab, under the rig's lock, when the frame
        carries it, and otherwise one taken here before the grab. Either way it is taken before the grab
        rather than after. Age is measured to decide whether the world can still be planned against,
        and a stamp taken after a slow read reports a frame as fresher than it is, which is the one
        direction that matters.
        """
        stamped = time.time()
        tool_pose = self._tool_pose
        reader = tool_pose if chosen(tool_pose) else None
        try:
            before = None if reader is None else reader()
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
        if reader is not None:
            if not isinstance(before, Pose) or not isinstance(after, Pose) or not self._held_still(before, after):
                # Taken somewhere between two poses, so no one pose places it. The world source asks
                # again, and after its attempts every refreshing motion stops, naming this camera.
                return None
            tool_to_base = before.to_matrix()

        captured = getattr(frame, "captured_at_s", None)
        return DepthSnapshot(
            depth_mm=array,
            intrinsics=np.asarray(intrinsics, dtype=np.float64),
            timestamp=float(captured) if isinstance(captured, (int, float)) else stamped,
            tool_to_base_mm=tool_to_base,
        )

    def _held_still(self, before: Pose, after: Pose) -> bool:
        """Whether the tool stayed inside the rig's tolerance across the grab, by the rule the pick frame uses too.

        Both poses have to be read in BASE (``ShutterMotion``).
        """
        tolerance = self._motion_tolerance
        if not chosen(tolerance):
            return False
        max_mm, max_deg = tolerance
        return ShutterMotion.between(before, after, tolerance_mm=max_mm, tolerance_deg=max_deg).within
