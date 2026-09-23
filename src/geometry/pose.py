""":class:`Pose`, a frame-tagged, immutable rigid pose in 3D space.

Public APIs across the stack pass a ``Pose`` rather than an ad-hoc tuple, a
bare ndarray or a vendor struct such as ``URPose``. The numerics contract is:

* ``position_mm``      (3,) float64, millimetres
* ``quaternion_xyzw``  (4,) float64, unit length, canonical sign
* ``frame``            :class:`~src.geometry.frame.Frame`
* ``label``            optional human-readable tag

Both ndarray fields are read-only from construction on, so a consumer cannot
mutate a pose it was handed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .exceptions import FrameMismatchError
from .frame import Frame
from .matrix import matrix_to_position_quaternion, position_quaternion_to_matrix
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    angle_between,
    from_rotation_matrix,
    to_axis_angle,
)
from .validation import (
    normalise_quaternion,
    validate_position_mm,
)

__all__ = ["CLOSING_AXES", "Pose"]

#: Below this the tool and its target are the same point, which names no direction to aim along.
_AIM_MIN_DISTANCE_MM = 1e-6

#: How far from parallel the up hint has to be before the roll it fixes means anything. Below it the
#: cross product is numerical noise and the roll would swing with the last digit of the aim.
_AIM_MIN_UP_CROSS = 1e-6

#: Where the roll comes from when the up hint runs along the aim, most importantly for the default
#: hint and an aim straight down, which is the commonest pose in a cell. The first entry is what
#: makes `aimed_at(x, y, z, target_mm=(x, y, z - d))` equal `tool_down(x, y, z)` exactly, and
#: `tests/test_geometry_precision.py` holds that equality rather than leaving it to be rediscovered.
_AIM_UP_FALLBACKS: tuple[tuple[float, float, float], ...] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0))

#: The horizontal axis a pose's closing axis, the tool's +X, can be lined up with. ``x`` and ``y`` are
#: fixed in the frame, so a UR moving round its base has to turn wrist 3 by exactly as much as the base
#: turns to keep them: measured on the UR3e and UR5e homes, a fixed yaw costs one degree of wrist 3 per
#: degree of base, and moves that go round the base the same way add up. ``radial`` runs from the frame's
#: origin through the point, and ``tangential`` a quarter turn clockwise from it, seen from above. Both
#: follow the point round the base, so wrist 3 keeps its angle from move to move. A leading "-" takes
#: the opposite direction (``-y`` is the frame's -Y, ``-radial`` points at the base), which a two-finger
#: hand grips the same with and which puts wrist 3 half a turn round; a leading "+" changes nothing.
CLOSING_AXES: tuple[str, ...] = (
    "x", "-x", "y", "-y", "radial", "-radial", "tangential", "-tangential",
)

#: Below this a point sits on the frame's vertical axis, where "radial" names no direction.
_RADIAL_MIN_MM = 1e-6


def _closing_axis_deg(closing_axis: str, x_mm: float, y_mm: float) -> float:
    """The heading of ``closing_axis`` about the frame's +Z at (``x_mm``, ``y_mm``), in degrees."""
    axis = str(closing_axis)
    turn = 0.0
    if axis[:1] in ("+", "-"):
        turn = 180.0 if axis[0] == "-" else 0.0
        axis = axis[1:]
    if axis == "x":
        return turn
    if axis == "y":
        return 90.0 + turn
    if axis in ("radial", "tangential"):
        if math.hypot(float(x_mm), float(y_mm)) < _RADIAL_MIN_MM:
            raise ValueError(
                f"Pose: closing_axis {closing_axis!r} at ({float(x_mm)}, {float(y_mm)}) sits on the "
                "frame's vertical axis, which names no radial direction. Use 'x' or 'y' there."
            )
        bearing = math.degrees(math.atan2(float(y_mm), float(x_mm)))
        return (bearing if axis == "radial" else bearing - 90.0) + turn
    raise ValueError(
        f"Pose: unknown closing_axis {closing_axis!r}; expected one of {', '.join(CLOSING_AXES)}"
        " (a leading '+' is allowed too)"
    )


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid 6-DoF pose tagged with its coordinate frame.

    Construction coerces the position to ``float64`` and checks it for shape
    (3,) and finiteness, normalises the quaternion into canonical sign form
    (``w >= 0``), and makes both ndarrays read-only.

    Equality and hashing compare the arrays byte-wise, which is well defined
    only because of that sign convention.
    """

    position_mm: np.ndarray
    quaternion_xyzw: np.ndarray
    frame: Frame
    label: str | None = None

    # Construction and validation

    def __post_init__(self) -> None:
        pos = validate_position_mm(self.position_mm)
        quat = normalise_quaternion(self.quaternion_xyzw, canonicalise=True)
        if not isinstance(self.frame, Frame):
            object.__setattr__(self, "frame", Frame(self.frame))
        pos.setflags(write=False)
        quat.setflags(write=False)
        object.__setattr__(self, "position_mm", pos)
        object.__setattr__(self, "quaternion_xyzw", quat)

    # Constructors

    @classmethod
    def tool_down(
        cls,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        yaw_deg: float = 0.0,
        closing_axis: str = "x",
        frame: Frame = Frame.BASE,
        label: str | None = None,
    ) -> Pose:
        """The tool at (``x_mm``, ``y_mm``, ``z_mm``) with its +Z pointing straight down
        and its +X along ``closing_axis``, turned a further ``yaw_deg`` about the frame's +Z.

        Half a turn about X puts the tool's +Z down and its +X along the frame's +X; the
        yaw then turns the closing axis about the vertical. This is the pose a bench
        move, a known part and a place most often need.

        ``closing_axis`` says which axis the closing axis lines up with before the yaw is
        added (:data:`CLOSING_AXES`). The default ``x`` is the frame's +X, as always. On a
        cell that moves round its base, ``radial`` or ``tangential`` keeps wrist 3 where it
        is instead of turning it by as much as the base turns. On the UR3e and UR5e homes,
        ``tangential`` keeps wrist 3 near its home angle, about 16 to 26 degrees off it,
        and ``radial`` about a quarter turn further. A two-finger hand grips the same with the
        opposite sign, ``-tangential`` or ``-y`` for instance, which puts wrist 3 half a turn round.
        """
        if closing_axis == "x":
            heading = float(yaw_deg)
        else:
            heading = _closing_axis_deg(closing_axis, x_mm, y_mm) + float(yaw_deg)
        half = math.radians(heading) / 2.0
        # Rz(yaw) * Rx(pi), as (x, y, z, w).
        return cls(
            position_mm=np.array(
                [float(x_mm), float(y_mm), float(z_mm)], dtype=np.float64
            ),
            quaternion_xyzw=np.array(
                [math.cos(half), math.sin(half), 0.0, 0.0], dtype=np.float64
            ),
            frame=frame,
            label=label,
        )

    @classmethod
    def aimed_at(
        cls,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        *,
        target_mm: Sequence[float] | np.ndarray,
        roll_deg: float = 0.0,
        up_hint: Sequence[float] = (0.0, 0.0, 1.0),
        closing_axis: str | None = None,
        frame: Frame = Frame.BASE,
        label: str | None = None,
    ) -> Pose:
        """The tool at (``x_mm``, ``y_mm``, ``z_mm``) with its +Z pointing AT ``target_mm``.

        :meth:`tool_down` answers one question: where is the tool, pointing straight down. A great
        deal of a cell is that pose, and two things are not. A wrist camera looking at a board on
        the table sees nothing of it from anywhere but overhead, and a marker board bolted to the
        flange shows a fixed camera nothing but its edge once the arm is off to one side. Both need
        the same thing: an orientation chosen so that one thing faces another, which no yaw about
        the vertical can produce.

        Aiming is all this does. It is not a claim that the target is reachable, in the workspace,
        or even in view: the planner and the cell's guards decide the first two, and the marker
        source reports the third. What it removes is the arithmetic a caller would otherwise write
        with an axis convention of its own.

        ``up_hint`` is which way is up in the WORLD, not in the image; it fixes the roll and does
        not change where the tool points. Where the aim runs along it -- looking straight down with
        the default ``+Z`` -- the hint says nothing, and a fallback settles the roll so that the
        result is exactly ``tool_down`` at the same position. ``roll_deg`` turns the tool about its
        own +Z, the axis it is pointing along. Aimed straight down that axis points at the floor,
        so a positive roll turns the closing axis the opposite way round the vertical from
        :meth:`tool_down`'s ``yaw_deg``; the two agree at zero.

        ``closing_axis`` fixes the roll a second way, instead of ``up_hint``: the tool's +X
        goes as close as it can to that horizontal axis (:data:`CLOSING_AXES`) at the tool's
        own position, tilted only as far as the aim needs. Aimed straight down it gives exactly
        ``tool_down`` with the same ``closing_axis``. With ``up_hint``'s default, the roll
        follows the eye's bearing round the target, which winds wrist 3 on a ring of views;
        ``radial`` or ``tangential`` follows the base instead.

        Raises ``ValueError`` when the tool would stand on its target, which names no
        direction, and when the aim runs along the closing axis, where no roll can put +X
        on it.
        """
        eye = np.array([float(x_mm), float(y_mm), float(z_mm)], dtype=np.float64)
        forward = np.asarray(target_mm, dtype=np.float64).reshape(3) - eye
        distance = float(np.linalg.norm(forward))
        if distance < _AIM_MIN_DISTANCE_MM:
            raise ValueError(
                f"Pose.aimed_at: the tool at {eye.tolist()} is {distance:.3f} mm from its target "
                f"{np.asarray(target_mm, dtype=np.float64).reshape(3).tolist()}, which names no "
                "direction to point along. Stand the tool off from what it looks at.")
        forward /= distance

        if closing_axis is not None:
            heading = math.radians(_closing_axis_deg(closing_axis, x_mm, y_mm))
            wanted = np.array([math.cos(heading), math.sin(heading), 0.0])
            right = wanted - float(np.dot(wanted, forward)) * forward
            if float(np.linalg.norm(right)) < _AIM_MIN_UP_CROSS:
                raise ValueError(
                    f"Pose.aimed_at: the aim from {eye.tolist()} runs along closing_axis "
                    f"{closing_axis!r}, so no roll puts the tool's +X on it. Choose another axis."
                )
        else:
            for candidate in (tuple(up_hint), *_AIM_UP_FALLBACKS):
                up = np.asarray(candidate, dtype=np.float64).reshape(3)
                right = np.cross(up, forward)
                if float(np.linalg.norm(right)) > _AIM_MIN_UP_CROSS:
                    break
            else:  # pragma: no cover (two fallbacks perpendicular to each other cannot both fail)
                raise ValueError("Pose.aimed_at: no usable up axis for this aim")
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        roll = math.radians(float(roll_deg))
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        rotation = np.column_stack([
            right * cos_roll + down * sin_roll,
            down * cos_roll - right * sin_roll,
            forward,
        ])
        return cls(
            position_mm=eye,
            quaternion_xyzw=from_rotation_matrix(rotation),
            frame=frame,
            label=label,
        )

    @classmethod
    def identity(cls, frame: Frame, *, label: str | None = None) -> Pose:
        """Identity pose at the origin of ``frame``."""
        return cls(
            position_mm=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
            frame=frame,
            label=label,
        )

    @classmethod
    def from_matrix(
        cls,
        T: np.ndarray,
        *,
        frame: Frame,
        label: str | None = None,
    ) -> Pose:
        """Build a pose from a validated 4x4 homogeneous matrix in millimetres."""
        t, q = matrix_to_position_quaternion(T)
        return cls(position_mm=t, quaternion_xyzw=q, frame=frame, label=label)

    # Conversions and copies

    def to_matrix(self) -> np.ndarray:
        """Return a fresh 4x4 homogeneous transform (mm) for this pose."""
        return position_quaternion_to_matrix(self.position_mm, self.quaternion_xyzw)

    def with_frame(self, frame: Frame) -> Pose:
        """Return a copy tagged with ``frame``.

        The numbers are unchanged: this relabels the pose, it does not
        transform it. :meth:`Transform.apply_pose` is the coordinate change.
        """
        return Pose(
            position_mm=self.position_mm.copy(),
            quaternion_xyzw=self.quaternion_xyzw.copy(),
            frame=frame,
            label=self.label,
        )

    def with_label(self, label: str | None) -> Pose:
        """Return a copy with a different ``label``."""
        return Pose(
            position_mm=self.position_mm.copy(),
            quaternion_xyzw=self.quaternion_xyzw.copy(),
            frame=self.frame,
            label=label,
        )

    # Geometry helpers

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance in mm between two poses, which must share a frame."""
        if self.frame != other.frame:
            raise FrameMismatchError(
                f"distance_to: frame mismatch {self.frame!r} vs {other.frame!r}"
            )
        return float(np.linalg.norm(self.position_mm - other.position_mm))

    def angle_to(self, other: Pose) -> float:
        """Geodesic angle in radians, 0 to pi, between two orientations in one frame."""
        if self.frame != other.frame:
            raise FrameMismatchError(
                f"angle_to: frame mismatch {self.frame!r} vs {other.frame!r}"
            )
        return angle_between(self.quaternion_xyzw, other.quaternion_xyzw)

    def axis_angle_rad(self) -> np.ndarray:
        """Return the axis-angle vector equivalent to the quaternion.

        The direction is the rotation axis and the norm is the angle in
        radians.
        """
        return to_axis_angle(self.quaternion_xyzw)

    # Equality, hashing and display

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose):
            return NotImplemented
        return (
            self.frame == other.frame
            and self.label == other.label
            and np.array_equal(self.position_mm, other.position_mm)
            and np.array_equal(self.quaternion_xyzw, other.quaternion_xyzw)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.frame,
                self.label,
                self.position_mm.tobytes(),
                self.quaternion_xyzw.tobytes(),
            )
        )

    def __repr__(self) -> str:  # pragma: no cover (cosmetic)
        x, y, z = self.position_mm
        qx, qy, qz, qw = self.quaternion_xyzw
        lbl = f", label={self.label!r}" if self.label else ""
        return (
            f"Pose(frame={self.frame.value!r}, "
            f"pos_mm=[{x:.2f}, {y:.2f}, {z:.2f}], "
            f"quat_xyzw=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]{lbl})"
        )
