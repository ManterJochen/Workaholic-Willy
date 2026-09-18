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
from dataclasses import dataclass

import numpy as np

from .exceptions import FrameMismatchError
from .frame import Frame
from .matrix import matrix_to_position_quaternion, position_quaternion_to_matrix
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    angle_between,
    to_axis_angle,
)
from .validation import (
    normalise_quaternion,
    validate_position_mm,
)

__all__ = ["Pose"]


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
        frame: Frame = Frame.BASE,
        label: str | None = None,
    ) -> Pose:
        """The tool at (``x_mm``, ``y_mm``, ``z_mm``) with its +Z pointing straight down
        and its +X turned ``yaw_deg`` about the frame's +Z.

        Half a turn about X puts the tool's +Z down and its +X along the frame's +X; the
        yaw then turns the closing axis about the vertical. This is the pose a bench
        move, a known part and a place most often need.
        """
        half = math.radians(float(yaw_deg)) / 2.0
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
