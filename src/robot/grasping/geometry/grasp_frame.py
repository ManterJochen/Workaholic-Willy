"""A grasp's two directions as the pose a tool takes: +Z the approach, +X the closing axis.

Every parallel-jaw candidate in this package says where the jaw goes as a position and two
directions: the approach, which the tool travels along toward the part, and the closing axis, which
the pads close across. A motion verb takes a pose. The rotation between the two is the grasp frame
of ``src.geometry.Frame.GRASP`` and of ``GraspPose``: the columns are ``[closing, binormal,
approach]``, with the binormal ``approach x closing`` so the frame is right handed. The pick
service's execution policy drives by the same columns, and ``Robot.pick`` backs off along the pose's
own +Z to its standoff.

A parallel jaw turned half a turn about its approach is the same grasp, so a closing axis and its
negation describe one grasp. This function keeps the sign it is given; the generation README records
that both directions plan.

The same rotation is also written in four other places, none of them a public route:
``_rotation_from_axes`` (``planning/pose_generation.py``), ``_quaternion_from_axes``
(``motion/execution_policy.py``), ``_to_camera_pose`` (``generation/_support_footprint_stage.py``)
and the collision check's adapter ``grasp_point_to_pose``. This is the public one the candidates'
``pose()`` methods call. The four stay as they are, because they sit on the pick path the
determinism goldens hash, and folding them in needs its own byte check.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.geometry import Frame, Pose, from_rotation_matrix

from ._validation import as_vec3

__all__ = ["pose_from_grasp_axes"]

#: Below this fraction of its own length a closing axis is taken to lie along the approach.
_PARALLEL_FRACTION = 1e-6
#: Below this length a direction is the zero vector.
_ZERO_LENGTH = 1e-12


def pose_from_grasp_axes(
    position_mm: "np.ndarray | Sequence[float]",
    *,
    approach: "np.ndarray | Sequence[float]",
    closing_axis: "np.ndarray | Sequence[float]",
    frame: Frame,
    label: str | None = "grasp",
) -> Pose:
    """The pose at ``position_mm`` whose +Z is ``approach`` and whose +X is ``closing_axis``, in
    ``frame``.

    ``approach`` is the direction the tool travels toward the part, so a straight-down grasp in BASE
    has ``approach = (0, 0, -1)``, and with ``closing_axis = (-1, 0, 0)`` that is the tool-down
    quaternion ``(0, 1, 0, 0)`` XYZW. ``closing_axis`` is used perpendicular to the approach: a
    component along the approach is removed before it is normalised, as the SFE stage and
    ``generate_grasp_poses`` do. The frame is the caller's and is never guessed.

    Raises ``ValueError`` for a vector that is not three finite numbers, a zero approach, and a
    closing axis that is zero or lies along the approach, where no rotation is defined. Those are
    malformed candidates, not refusals of a cell.
    """
    position = as_vec3(position_mm, "position_mm")
    z_axis = as_vec3(approach, "approach")
    z_length = float(np.linalg.norm(z_axis))
    if z_length < _ZERO_LENGTH:
        raise ValueError("approach is the zero vector, so the grasp has no direction to travel along")
    z_axis = z_axis / z_length
    closing = as_vec3(closing_axis, "closing_axis")
    closing_length = float(np.linalg.norm(closing))
    x_axis = closing - float(closing @ z_axis) * z_axis
    x_length = float(np.linalg.norm(x_axis))
    if closing_length < _ZERO_LENGTH or x_length < _PARALLEL_FRACTION * closing_length:
        raise ValueError(
            "closing_axis is zero or lies along the approach, so the jaw has no direction to close across")
    x_axis = x_axis / x_length
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return Pose(position_mm=position, quaternion_xyzw=from_rotation_matrix(rotation), frame=frame, label=label)
