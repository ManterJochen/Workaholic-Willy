"""The one reader of a closing axis: a name ``Pose.tool_down`` takes, or an orientation, read as a heading about base Z.

A pose closes along its tool +X. A program names the closing axis it wants (``GraspMotion(closing_axis=...)``,
``Scene.grasps(closing_axis=...)``, ``Locator.look_around(..., closing_axis=...)``), and a cell names how its hand and
camera naturally stand (``robot.natural_closing_axis``, the owner's decision, 2026-09-30). Both are read here, by
:func:`closing_axis_of`: a name of :data:`~src.geometry.pose.CLOSING_AXES` (a leading "+" allowed), read at each place
through the geometry's own rule, :func:`~src.geometry.pose.closing_axis_heading_deg`; or an orientation, a quaternion
(x, y, z, w) or a BASE ``Pose``, whose tool +X laid onto the base XY plane is the heading. Only that heading counts: a
tool +X pitched out of the horizontal names the heading of its shadow. So
``Pose.tool_down(x, y, z, closing_axis="-y").quaternion_xyzw`` and ``"-y"`` name the same axis.

The reader lives with the poses it reads rather than with the grasps it chooses among
(:mod:`src.robot.grasping.geometry.closing_axis`, which hands these names on), so reading a tree that names the key and
building a pose through the cell (``Robot.tool_down``) load NumPy and this package only: no grasping package, no OpenCV.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .exceptions import GeometryError
from .frame import Frame
from .pose import CLOSING_AXES, Pose, closing_axis_heading_deg
from .quaternion import to_rotation_matrix
from .validation import normalise_quaternion

__all__ = [
    "NO_HEADING_WITHIN_DEG_OF_VERTICAL",
    "ClosingAxis",
    "ClosingAxisLike",
    "closing_axis_of",
    "natural_closing_axis_of",
]

#: An axis within this many degrees of the vertical names no heading about base Z: its shadow on the base XY plane is so
#: short that its heading swings with the last degree of the axis (here one degree of tilt turns it by up to six). A
#: choice, not a measurement.
NO_HEADING_WITHIN_DEG_OF_VERTICAL = 10.0


@dataclass(frozen=True, slots=True)
class ClosingAxis:
    """The closing axis a program wants its grasps to close along, about base Z.

    ``name`` is one of :data:`~src.geometry.pose.CLOSING_AXES` without a leading "+", ``""`` for an orientation's.
    ``heading_deg`` is the heading an orientation's tool +X gave, degrees from base +X toward base +Y in [0, 360),
    ``None`` for a name. Build one with :func:`closing_axis_of`.
    """

    name: str = ""
    heading_deg: float | None = None

    def __post_init__(self) -> None:
        if bool(self.name) == (self.heading_deg is not None):
            raise ValueError("ClosingAxis is a name or a heading, one of the two; closing_axis_of builds it from either")
        if self.name and self.name not in CLOSING_AXES:
            raise ValueError(f"ClosingAxis name {self.name!r} is not one of {', '.join(CLOSING_AXES)}")
        if self.heading_deg is not None and not math.isfinite(float(self.heading_deg)):
            raise ValueError(f"ClosingAxis heading must be finite, got {self.heading_deg!r}")

    def heading_at(self, x_mm: float, y_mm: float) -> float:
        """The wanted heading at (``x_mm``, ``y_mm``) in BASE, degrees. Raises ``ValueError`` for ``radial`` or
        ``tangential`` on the base's vertical axis, where they point nowhere."""
        if self.heading_deg is not None:
            return float(self.heading_deg)
        return closing_axis_heading_deg(self.name, x_mm, y_mm)

    def __str__(self) -> str:
        if self.heading_deg is not None:
            return f"the heading {self.heading_deg:.1f} deg of the orientation's tool +X"
        return repr(self.name)


#: What names a closing axis: a name ``Pose.tool_down`` takes, a quaternion (x, y, z, w), a BASE ``Pose``, or a
#: :class:`ClosingAxis`.
ClosingAxisLike: TypeAlias = "str | Pose | ClosingAxis | Sequence[float] | np.ndarray"


def closing_axis_of(value: object) -> ClosingAxis:
    """The :class:`ClosingAxis` ``value`` names, refused with a sentence when it names none.

    A name of :data:`~src.geometry.pose.CLOSING_AXES` (a leading "+" allowed), a quaternion (x, y, z, w) or a BASE
    ``Pose``: an orientation's tool +X, laid onto the base XY plane, is the wanted heading. Raises ``ValueError`` for an
    unknown name, a quaternion that is not one, a ``Pose`` in another frame, and an orientation whose tool +X points
    within :data:`NO_HEADING_WITHIN_DEG_OF_VERTICAL` of the vertical; ``TypeError`` for anything else.
    """
    if isinstance(value, ClosingAxis):
        return value
    if isinstance(value, str):
        try:
            closing_axis_heading_deg(value, 1.0, 1.0)  # the geometry's own rule says which names it knows
        except ValueError:
            raise ValueError(
                f"closing_axis {value!r} is not a name Pose.tool_down takes ({', '.join(CLOSING_AXES)}, a leading '+' "
                "allowed), nor an orientation") from None
        return ClosingAxis(name=value[1:] if value.startswith("+") else value)
    if isinstance(value, Pose):
        if value.frame is not Frame.BASE:
            raise ValueError(
                f"closing_axis as a Pose is read in BASE, and this one is in {value.frame.value}: carry it into BASE "
                "first, or name the axis")
        quaternion = np.asarray(value.quaternion_xyzw, dtype=np.float64)
    elif isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (bytes, bytearray)):
        try:
            quaternion = normalise_quaternion(np.asarray(value, dtype=np.float64), name="closing_axis")
        except (GeometryError, TypeError, ValueError) as exc:
            raise ValueError(f"closing_axis as an orientation is a quaternion (x, y, z, w): {exc}") from None
    else:
        raise TypeError(
            f"closing_axis is a name Pose.tool_down takes ({', '.join(CLOSING_AXES)}), a quaternion (x, y, z, w) or a "
            f"BASE Pose, not {type(value).__name__}")
    tool_x = to_rotation_matrix(quaternion)[:, 0]
    off_vertical = math.degrees(math.atan2(math.hypot(float(tool_x[0]), float(tool_x[1])), abs(float(tool_x[2]))))
    if off_vertical < NO_HEADING_WITHIN_DEG_OF_VERTICAL:
        raise ValueError(
            f"closing_axis: the orientation's tool +X points {off_vertical:.1f} deg from the vertical, within "
            f"{NO_HEADING_WITHIN_DEG_OF_VERTICAL:g} deg of it, so it names no heading about base Z to close along; name "
            "the axis (\"-y\"), or give an orientation whose tool +X lies nearer the horizontal")
    return ClosingAxis(heading_deg=math.degrees(math.atan2(float(tool_x[1]), float(tool_x[0]))) % 360.0)


def natural_closing_axis_of(robot_config: object) -> "ClosingAxis | None":
    """How the hand and camera of the cell ``robot_config`` describes naturally stand (``robot.natural_closing_axis``),
    read as :func:`closing_axis_of` reads an axis; ``None`` where it names none, a config without the key included."""
    named = getattr(robot_config, "natural_closing_axis", None)
    return None if named is None else closing_axis_of(named)
