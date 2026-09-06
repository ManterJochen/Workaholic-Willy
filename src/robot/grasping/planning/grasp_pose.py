"""6D grasp pose value object for parallel-jaw grasp planning.

The pose is expressed in an external coordinate frame such as camera or
robot base. Its local grasp-frame convention follows
``src.geometry.Frame.GRASP``:

* local X axis: gripper closing direction between the two contacts;
* local Y axis: binormal completing a right-handed frame;
* local Z axis: approach direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.geometry import Frame, Pose, from_rotation_matrix
from src.robot.grasping.geometry import as_vec3

__all__ = ["GraspPose"]

_ORTHONORMAL_ATOL = 1e-6
_DET_ATOL = 1e-6


def _as_rotation_matrix(value: Any) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"GraspPose.rotation_matrix must be shape (3, 3), got {rotation.shape}"
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("GraspPose.rotation_matrix must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ORTHONORMAL_ATOL):
        raise ValueError("GraspPose.rotation_matrix must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > _DET_ATOL:
        raise ValueError(
            f"GraspPose.rotation_matrix must have determinant +1, got {determinant:.6f}"
        )
    return rotation.copy()


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _as_contacts(value: tuple[Any, Any]) -> tuple[np.ndarray, np.ndarray]:
    if len(value) != 2:
        raise ValueError("GraspPose.contacts must contain exactly two contact points")
    contact_a = _readonly(as_vec3(value[0], "contacts[0]"))
    contact_b = _readonly(as_vec3(value[1], "contacts[1]"))
    return contact_a, contact_b


def _score_unit_interval(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(f"GraspPose.{name} must be finite and in [0, 1]")
    return scalar


@dataclass(frozen=True, slots=True)
class GraspPose:
    """Immutable 6D candidate grasp pose.

    ``rotation_matrix`` stores local grasp axes as columns in the pose's
    external ``frame``: ``[closing_axis, binormal_axis, approach_axis]``.
    """

    position_mm: np.ndarray
    rotation_matrix: np.ndarray
    grip_width_mm: float
    score: float
    confidence: float
    contacts: tuple[np.ndarray, np.ndarray]
    frame: Frame = Frame.CAMERA
    metadata: dict[str, Any] = field(default_factory=dict)
    quaternion_xyzw: np.ndarray = field(init=False)
    closing_axis: np.ndarray = field(init=False)
    binormal_axis: np.ndarray = field(init=False)
    approach_axis: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        position = _readonly(as_vec3(self.position_mm, "position_mm"))
        rotation = _readonly(_as_rotation_matrix(self.rotation_matrix))
        quaternion = _readonly(from_rotation_matrix(rotation))
        closing_axis = _readonly(rotation[:, 0].copy())
        binormal_axis = _readonly(rotation[:, 1].copy())
        approach_axis = _readonly(rotation[:, 2].copy())
        contacts = _as_contacts(self.contacts)

        grip_width = float(self.grip_width_mm)
        if not np.isfinite(grip_width) or grip_width < 0.0:
            raise ValueError("GraspPose.grip_width_mm must be finite and >= 0")

        frame = self.frame if isinstance(self.frame, Frame) else Frame(self.frame)

        object.__setattr__(self, "position_mm", position)
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "closing_axis", closing_axis)
        object.__setattr__(self, "binormal_axis", binormal_axis)
        object.__setattr__(self, "approach_axis", approach_axis)
        object.__setattr__(self, "grip_width_mm", grip_width)
        object.__setattr__(self, "score", _score_unit_interval(self.score, "score"))
        object.__setattr__(self, "confidence", _score_unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_pose(self, *, label: str | None = "grasp") -> Pose:
        """Return the generic geometry :class:`Pose` representation."""
        return Pose(
            position_mm=self.position_mm.copy(),
            quaternion_xyzw=self.quaternion_xyzw.copy(),
            frame=self.frame,
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "position_mm": self.position_mm.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
            "closing_axis": self.closing_axis.tolist(),
            "binormal_axis": self.binormal_axis.tolist(),
            "approach_axis": self.approach_axis.tolist(),
            "grip_width_mm": self.grip_width_mm,
            "score": self.score,
            "confidence": self.confidence,
            "contacts": [self.contacts[0].tolist(), self.contacts[1].tolist()],
            "frame": self.frame.value,
            "metadata": dict(self.metadata),
        }
