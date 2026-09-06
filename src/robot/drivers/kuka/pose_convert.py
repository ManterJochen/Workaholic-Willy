"""KUKA pose and joint conversions.

They bridge between the internal representation, which is a
:class:`src.geometry.Pose` in millimetres with an XYZW quaternion and joints in
radians, and the KUKA controller representation:

* e6pos, ``{X, Y, Z, A, B, C}``, where ``X/Y/Z`` are millimetres and ``A/B/C`` are
  intrinsic Z, Y, X Euler angles in degrees under the KUKA convention
  ``R = Rz(A) * Ry(B) * Rx(C)``.
* e6axis, ``{A1, A2, A3, A4, A5, A6}``, joint angles in degrees.

A conversion never round-trips through axis-angle or Rodrigues, which avoids
gimbal-style cancellations. KUKA is a strictly intrinsic ZYX Euler system, so the
rotation is built directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_euler, to_euler

__all__ = [
    "KukaCartesian",
    "joints_deg_to_rad",
    "joints_rad_to_deg",
    "kuka_cartesian_to_pose",
    "pose_to_kuka_cartesian",
]


@dataclass(frozen=True, slots=True)
class KukaCartesian:
    """A lightweight container mirroring the KUKA e6pos ``{X, Y, Z, A, B, C}``.

    ``x/y/z`` are millimetres and ``a/b/c`` are intrinsic Z, Y, X Euler angles in
    degrees, following the KUKA convention ``R = Rz(A) * Ry(B) * Rx(C)``.
    """

    x: float
    y: float
    z: float
    a: float
    b: float
    c: float

    def to_dict(self) -> dict:
        return {"X": self.x, "Y": self.y, "Z": self.z, "A": self.a, "B": self.b, "C": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> KukaCartesian:
        return cls(
            x=float(d["X"]),
            y=float(d["Y"]),
            z=float(d["Z"]),
            a=float(d["A"]),
            b=float(d["B"]),
            c=float(d["C"]),
        )

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z, self.a, self.b, self.c)


def kuka_cartesian_to_pose(
    e6: KukaCartesian,
    *,
    label: str | None = None,
) -> Pose:
    """Convert a KUKA ``{X, Y, Z, A, B, C}`` reading to a :class:`Pose`.

    The returned pose is tagged :attr:`Frame.BASE`, which is the KUKA ``$BASE`` frame,
    by convention the robot world frame once the controller has resolved any tool and
    base chain.

    Numerics
    --------
    The KUKA convention is ``R = Rz(A) * Ry(B) * Rx(C)`` with the angles in degrees.
    The geometry helper :func:`from_euler` in order ``"xyz"`` builds
    ``Rz(arr[2]) * Ry(arr[1]) * Rx(arr[0])``, so it is passed ``[C, B, A]`` in radians
    to obtain the same rotation matrix.
    """
    abc_rad = np.radians(np.array([e6.c, e6.b, e6.a], dtype=np.float64))
    quat = from_euler(abc_rad, order="xyz")
    return Pose(
        position_mm=np.array([e6.x, e6.y, e6.z], dtype=np.float64),
        quaternion_xyzw=quat,
        frame=Frame.BASE,
        label=label,
    )


def pose_to_kuka_cartesian(pose: Pose) -> KukaCartesian:
    """Convert a :class:`Pose` in Frame.BASE to a KUKA ``E6POS``-style record.

    :func:`kuka_cartesian_to_pose` sets out the array ordering: the geometry helper
    returns ``[x_rad, y_rad, z_rad]``, which is the KUKA ``[C, B, A]`` in radians.
    """
    if pose.frame is not Frame.BASE:
        raise ValueError(
            f"pose_to_kuka_cartesian requires Frame.BASE; got {pose.frame!r}."
        )
    xyz_rad = to_euler(pose.quaternion_xyzw, order="xyz")
    xyz_deg = np.degrees(xyz_rad)
    return KukaCartesian(
        x=float(pose.position_mm[0]),
        y=float(pose.position_mm[1]),
        z=float(pose.position_mm[2]),
        a=float(xyz_deg[2]),
        b=float(xyz_deg[1]),
        c=float(xyz_deg[0]),
    )


def joints_rad_to_deg(joints_rad: Sequence[float]) -> list[float]:
    """Convert joint angles from radians to the degrees of the KUKA wire format."""
    return [float(np.degrees(j)) for j in joints_rad]


def joints_deg_to_rad(joints_deg: Iterable[float]) -> list[float]:
    """Convert joint angles from the degrees of the KUKA wire format to radians."""
    return [float(np.radians(j)) for j in joints_deg]
