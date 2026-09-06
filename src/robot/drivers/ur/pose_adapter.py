"""The vendor boundary adapters between :class:`URPose` and the neutral :class:`Pose`.

This module is the only place in the codebase that translates between the Universal
Robots axis-angle pose representation and the canonical :class:`Pose`, which is
millimetres, an XYZW quaternion and a :class:`Frame`.

Architectural rule
------------------
No module outside ``src.robot.drivers.ur`` imports ``URPose``. A vendor-specific type
stops here, and everything above this boundary speaks ``Pose``.

The geometry package does not host these adapters, because geometry must not know about
UR types. They live here at the vendor boundary instead.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.quaternion import from_axis_angle, to_axis_angle

from .pose import URPose

__all__ = ["urpose_to_pose", "pose_to_urpose"]


def urpose_to_pose(
    urpose: URPose,
    *,
    frame: Frame = Frame.BASE,
    label: str | None = None,
) -> Pose:
    """Convert a :class:`URPose` to a vendor-neutral :class:`Pose`.

    ``URPose`` stores ``(x, y, z)`` in millimetres and the orientation as an axis-angle
    vector ``(rx, ry, rz)`` in radians, and both map directly onto the geometry numerics
    contract. ``frame`` defaults to :attr:`Frame.BASE`, because the UR controller
    reports a TCP pose in the base frame.

    ``label``, where given, overrides the label the URPose carries.
    """
    pos = np.array([urpose.x, urpose.y, urpose.z], dtype=np.float64)
    rvec = np.array([urpose.rx, urpose.ry, urpose.rz], dtype=np.float64)
    quat = from_axis_angle(rvec)
    return Pose(
        position_mm=pos,
        quaternion_xyzw=quat,
        frame=frame,
        label=label if label is not None else (urpose.label or None),
    )


def pose_to_urpose(pose: Pose) -> URPose:
    """Convert a :class:`Pose` back to a :class:`URPose`.

    The pose frame is not checked against :attr:`Frame.BASE`, because ``URPose`` carries
    no frame. The caller is responsible for the pose living in the robot base frame
    before this conversion.
    """
    rvec = to_axis_angle(pose.quaternion_xyzw)
    x, y, z = pose.position_mm
    return URPose(
        x=float(x),
        y=float(y),
        z=float(z),
        rx=float(rvec[0]),
        ry=float(rvec[1]),
        rz=float(rvec[2]),
        label=pose.label or "",
    )
