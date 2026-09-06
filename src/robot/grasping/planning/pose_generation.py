"""Build 6D grasp poses from antipodal contact pairs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.geometry import Frame
from src.robot.grasping.contacts import ContactPair

from .grasp_pose import GraspPose

__all__ = ["generate_grasp_poses", "grasp_pose_from_contact_pair"]

_EPS = 1e-9
_DEFAULT_APPROACH_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_DEFAULT_FALLBACK_APPROACH_AXES = (
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
)


def _unit(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        raise ValueError(f"{name} cannot be the zero vector")
    return vector / norm


def _projected_approach_axis(
    closing_axis: np.ndarray,
    preferred_approach_axis: np.ndarray,
    fallback_approach_axes: Sequence[np.ndarray] | None,
) -> np.ndarray:
    fallbacks = (
        tuple(fallback_approach_axes)
        if fallback_approach_axes is not None
        else _DEFAULT_FALLBACK_APPROACH_AXES
    )
    candidates = (preferred_approach_axis, *fallbacks)
    for candidate in candidates:
        approach = np.asarray(candidate, dtype=np.float64)
        if approach.shape != (3,) or not np.all(np.isfinite(approach)):
            raise ValueError("approach-axis candidates must be finite shape (3,) vectors")
        projected = approach - closing_axis * float(np.dot(approach, closing_axis))
        norm = float(np.linalg.norm(projected))
        if norm >= _EPS:
            return projected / norm
    raise ValueError("no approach-axis candidate is independent of the closing axis")


def _rotation_from_axes(closing_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    x_axis = _unit(closing_axis, "closing_axis")
    projected_approach = approach_axis - x_axis * float(np.dot(approach_axis, x_axis))
    z_axis = _unit(projected_approach, "approach_axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "binormal_axis")
    z_axis = _unit(np.cross(x_axis, y_axis), "approach_axis")
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if float(np.linalg.det(rotation)) < 0.0:
        y_axis = -y_axis
        rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation


def _default_confidence(pair: ContactPair) -> float:
    raw = pair.metadata.get("stability", pair.antipodal_score)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = pair.antipodal_score
    # A non-numeric stability falls back above. A numeric one that is NaN, such as a NaN
    # surface-normal confidence propagated into metadata, would otherwise pass float() and clip to
    # NaN, which the GraspPose constructor rejects, so the always-finite antipodal score is used.
    if not np.isfinite(value):
        value = pair.antipodal_score
    return float(np.clip(value, 0.0, 1.0))


def grasp_pose_from_contact_pair(
    pair: ContactPair,
    *,
    preferred_approach_axis: np.ndarray | Sequence[float] = _DEFAULT_APPROACH_AXIS,
    fallback_approach_axes: Sequence[np.ndarray] | None = None,
    frame: Frame = Frame.CAMERA,
    score: float | None = None,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraspPose:
    """Build a right-handed 6D grasp pose from a contact pair.

    The contact pair fixes the local X/closing axis. The local Z/approach
    axis is the preferred direction projected into the plane perpendicular
    to that closing axis. This makes the roll ambiguity explicit and keeps
    the result executable for arbitrary contact orientations.
    """
    if not isinstance(pair, ContactPair):
        raise TypeError("pair must be a ContactPair")
    closing_axis = pair.closing_axis
    preferred_axis = _unit(preferred_approach_axis, "preferred_approach_axis")
    approach_axis = _projected_approach_axis(
        closing_axis,
        preferred_axis,
        fallback_approach_axes,
    )
    rotation = _rotation_from_axes(closing_axis, approach_axis)
    pose_metadata: dict[str, Any] = {
        "source": "contact_pair",
        "axis_alignment": pair.axis_alignment,
        "normal_opposition": pair.normal_opposition,
    }
    pose_metadata.update(pair.metadata)
    if metadata:
        pose_metadata.update(metadata)
    return GraspPose(
        position_mm=pair.center_mm,
        rotation_matrix=rotation,
        grip_width_mm=pair.distance_mm,
        score=pair.antipodal_score if score is None else float(score),
        confidence=_default_confidence(pair) if confidence is None else float(confidence),
        contacts=(pair.point_a, pair.point_b),
        frame=frame,
        metadata=pose_metadata,
    )


def generate_grasp_poses(
    pairs: Iterable[ContactPair],
    *,
    preferred_approach_axis: np.ndarray | Sequence[float] = _DEFAULT_APPROACH_AXIS,
    fallback_approach_axes: Sequence[np.ndarray] | None = None,
    frame: Frame = Frame.CAMERA,
    max_poses: int | None = None,
) -> list[GraspPose]:
    """Convert contact pairs to grasp poses while preserving input order."""
    if max_poses is not None and max_poses < 1:
        raise ValueError("max_poses must be >= 1 when provided")
    poses: list[GraspPose] = []
    for pair in pairs:
        if max_poses is not None and len(poses) >= max_poses:
            break
        poses.append(
            grasp_pose_from_contact_pair(
                pair,
                preferred_approach_axis=preferred_approach_axis,
                fallback_approach_axes=fallback_approach_axes,
                frame=frame,
            )
        )
    return poses
