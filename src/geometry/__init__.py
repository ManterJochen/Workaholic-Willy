"""Workaholic-Willy geometry subsystem: vendor-neutral, frame-safe pose and
transform primitives.

This package owns the canonical 3D representations used across the stack:

* :class:`Frame`      coordinate frames, as an enum.
* :class:`Pose`       a 6-DoF rigid pose: position, orientation and frame.
* :class:`Transform`  a typed rigid transform between two frames.

It knows nothing about robot vendors, OpenCV, ROS, FastAPI or pipelines. See
``README.md`` for the full ownership and non-ownership contract.

Numerics, on every public value:

* Translation in millimetres, ``float64``.
* Orientation as a unit XYZW quaternion, ``float64``, canonical sign.
* Angles in radians.
* Arrays are read-only once constructed.
"""

from __future__ import annotations

from .exceptions import (
    FrameMismatchError,
    GeometryError,
    InvalidMatrixError,
    InvalidPoseError,
    InvalidQuaternionError,
    InvalidTransformError,
)
from .frame import Frame
from .pose import Pose
from .quaternion import (
    IDENTITY_QUAT_XYZW,
    angle_between,
    canonicalise,
    conjugate,
    from_axis_angle,
    from_euler,
    from_rotation_matrix,
    matrix_to_rotation_vector,
    multiply,
    rotate_vector,
    rotation_vector_to_matrix,
    to_axis_angle,
    to_euler,
    to_rotation_matrix,
)
from .serialization import (
    POSE_SCHEMA,
    TRANSFORM_SCHEMA,
    pose_from_dict,
    pose_to_dict,
    transform_from_dict,
    transform_to_dict,
)
from .transform import Transform

__all__ = [
    "IDENTITY_QUAT_XYZW",
    "POSE_SCHEMA",
    "TRANSFORM_SCHEMA",
    "Frame",
    "FrameMismatchError",
    "GeometryError",
    "InvalidMatrixError",
    "InvalidPoseError",
    "InvalidQuaternionError",
    "InvalidTransformError",
    "Pose",
    "Transform",
    "angle_between",
    "canonicalise",
    "conjugate",
    "from_axis_angle",
    "from_euler",
    "from_rotation_matrix",
    "matrix_to_rotation_vector",
    "multiply",
    "pose_from_dict",
    "pose_to_dict",
    "rotate_vector",
    "rotation_vector_to_matrix",
    "to_axis_angle",
    "to_euler",
    "to_rotation_matrix",
    "transform_from_dict",
    "transform_to_dict",
]
