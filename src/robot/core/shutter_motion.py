"""How far the tool moved while a camera's shutter was open, and whether a rig tolerates it.

A camera on the wrist is placed frame by frame by where the tool stood when its shutter opened. The pose is read
before and after the grab, and a frame taken while the tool moved beyond the rig's tolerance is placed by no single
pose. One rule decides that for the pick frame (``RealSenseVisionPerceptionSource``) and for the live planner world
(``RigDepthSource``), so the two cannot disagree about whether the arm held still.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import angle_between

__all__ = ["ShutterMotion"]


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
