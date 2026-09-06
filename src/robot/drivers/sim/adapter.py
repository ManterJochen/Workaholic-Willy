"""Pure-Python conversion helpers between this stack types and Isaac shapes.

This module is the explicit translation layer. It holds only helpers that operate on
numeric arrays and stack types, and it never imports Isaac, so it stays safe to load on
a host without it. :class:`IsaacRobotArm` calls these helpers once it has unpacked Isaac
articulation and TCP state into raw numpy arrays.

Keeping the conversion maths out of ``arm.py`` makes the surface exercisable on any
host, with no Isaac needed, and gives the camera and calibration adapters one place to
reuse the same pose and joint conventions.

Conventions
-----------
* Positions are millimetres throughout this stack and metres by default in Isaac, so
  every position crossing the boundary passes through :func:`metres_to_millimetres` or
  :func:`millimetres_to_metres`.
* Quaternions are XYZW on this side and may be WXYZ on the Isaac side, in USD prim
  attributes. :func:`isaac_wxyz_to_xyzw` and :func:`xyzw_to_isaac_wxyz` do the swap
  explicitly, so a call site cannot get the ordering wrong.
* Joint vectors are radians on both sides, so the helpers touch the dtype and the shape
  and never the values.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions

__all__ = [
    "isaac_pose_to_willy",
    "willy_pose_to_isaac",
    "isaac_joints_to_willy",
    "willy_joints_to_isaac",
    "metres_to_millimetres",
    "millimetres_to_metres",
    "isaac_wxyz_to_xyzw",
    "xyzw_to_isaac_wxyz",
    "isaac_rotmat_to_wxyz",
]


# ---------------------------------------------------------------------------
# Scalar and array unit conversions
# ---------------------------------------------------------------------------


def metres_to_millimetres(values: np.ndarray) -> np.ndarray:
    """Return ``values * 1000.0`` as a ``float64`` array."""
    return np.asarray(values, dtype=np.float64) * 1000.0


def millimetres_to_metres(values: np.ndarray) -> np.ndarray:
    """Return ``values / 1000.0`` as a ``float64`` array."""
    return np.asarray(values, dtype=np.float64) / 1000.0


# ---------------------------------------------------------------------------
# Quaternion convention swaps
# ---------------------------------------------------------------------------


def isaac_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from the Isaac ``[w, x, y, z]`` to this stack ``[x, y, z, w]``."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_isaac_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from this stack ``[x, y, z, w]`` to the Isaac ``[w, x, y, z]``."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def isaac_rotmat_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalised Isaac ``[w, x, y, z]`` quaternion.

    The Lula ``compute_forward_kinematics`` returns the orientation as a rotation matrix
    rather than a quaternion, so this is the bridge into :func:`isaac_pose_to_willy`. It
    takes the numerically stable branch of Shepperd's method, the one on the largest
    diagonal entry, and returns a unit quaternion. Validated on-box against an FK and IK
    round-trip on the UR5e.
    """
    R = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Pose and joint round-trips
# ---------------------------------------------------------------------------


def isaac_pose_to_willy(
    position_m: np.ndarray,
    orientation_wxyz: np.ndarray,
    *,
    label: str | None = None,
) -> Pose:
    """Build a :class:`Pose` in mm, XYZW and :attr:`Frame.BASE` from Isaac state.

    Parameters
    ----------
    position_m
        A ``(3,)`` array in metres, as read from Isaac.
    orientation_wxyz
        A ``(4,)`` quaternion in the Isaac ``[w, x, y, z]`` convention.
    label
        Optional label, preserved on the :class:`Pose`.
    """
    return Pose(
        position_mm=metres_to_millimetres(position_m),
        quaternion_xyzw=isaac_wxyz_to_xyzw(orientation_wxyz),
        frame=Frame.BASE,
        label=label,
    )


def willy_pose_to_isaac(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Convert a :class:`Pose` into ``(position_m, orientation_wxyz)``.

    Raises
    ------
    ValueError
        If ``pose.frame`` is not :attr:`Frame.BASE`. An Isaac articulation command is
        expressed in the simulator world frame, which is the robot base frame, so a
        tool-frame pose would mis-target silently.
    """
    if pose.frame is not Frame.BASE:
        raise ValueError(
            f"willy_pose_to_isaac requires Frame.BASE; got {pose.frame!r}."
        )
    return (
        millimetres_to_metres(pose.position_mm),
        xyzw_to_isaac_wxyz(pose.quaternion_xyzw),
    )


def isaac_joints_to_willy(joint_radians: np.ndarray) -> JointPositions:
    """Wrap an Isaac joint vector, in radians, in a :class:`JointPositions`."""
    return JointPositions(np.asarray(joint_radians, dtype=np.float64))


def willy_joints_to_isaac(joints: JointPositions) -> np.ndarray:
    """Return the raw radian vector backing a :class:`JointPositions`."""
    return np.asarray(joints.values, dtype=np.float64)
