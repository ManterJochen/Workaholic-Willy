"""GraspPoint, the immutable container for a candidate grasp.

A GraspPoint fully specifies how a parallel-jaw gripper approaches and closes on an
object:

* ``position`` is the 3D anchor point, in mm, in either the camera or the base frame.
* ``approach`` is the unit vector the gripper moves along toward the object before
  closing. For a top-down grasp in the robot base frame it is ``[0, 0, -1]``.
* ``axis`` is the unit vector along the line connecting the two finger pads,
  perpendicular to ``approach``. The gripper closes across this axis.
* ``grip_width_mm`` is the expected finger-pad separation at contact.
* ``score`` is the quality in ``[0, 1]``, where higher is better.
* ``frame`` is the coordinate frame the vectors live in.

Numerics contract
-----------------
Distances are millimetres. Direction vectors are unit-norm, which is enforced at
construction. The dataclass is frozen and the underlying numpy arrays are marked
read-only, so a consumer cannot silently mutate ``gp.position[0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from src.robot.grasping.geometry import as_vec3


class GraspFrame(StrEnum):
    """Coordinate frame a GraspPoint is expressed in."""
    CAMERA = "camera"
    BASE = "base"


# The tolerances for vector validation.
_UNIT_TOL = 1e-3        # acceptable deviation from |v| = 1
_ZERO_TOL = 1e-9        # below this the vector is considered the zero vector


def _normalise_unit(v: np.ndarray, name: str) -> np.ndarray:
    """Return a unit-length copy of ``v`` or raise if it is the zero vector."""
    n = float(np.linalg.norm(v))
    if n < _ZERO_TOL:
        raise ValueError(f"GraspPoint.{name} cannot be the zero vector")
    if abs(n - 1.0) <= _UNIT_TOL:
        return v
    return v / n


@dataclass(frozen=True, slots=True)
class GraspPoint:
    """A single grasp candidate. The module docstring carries the semantics."""

    position: np.ndarray
    approach: np.ndarray
    axis: np.ndarray
    grip_width_mm: float
    score: float
    frame: GraspFrame
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate and coerce the three vectors, normalising approach and axis.
        position = as_vec3(self.position, "position")
        approach = _normalise_unit(as_vec3(self.approach, "approach"), "approach")
        axis = _normalise_unit(as_vec3(self.axis, "axis"), "axis")

        if self.grip_width_mm < 0.0:
            raise ValueError(
                f"GraspPoint.grip_width_mm must be >= 0, got {self.grip_width_mm}",
            )
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"GraspPoint.score must be in [0, 1], got {self.score}",
            )

        # Make the arrays read-only so consumers cannot mutate them.
        position.flags.writeable = False
        approach.flags.writeable = False
        axis.flags.writeable = False

        # A frozen dataclass, so the setattr lock is bypassed to install the validated
        # values.
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "approach", approach)
        object.__setattr__(self, "axis", axis)

    def distance_to(self, other: GraspPoint) -> float:
        """The Euclidean distance in millimetres between two grasp positions."""
        return float(np.linalg.norm(self.position - other.position))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "position": self.position.tolist(),
            "approach": self.approach.tolist(),
            "axis": self.axis.tolist(),
            "grip_width_mm": self.grip_width_mm,
            "score": self.score,
            "frame": self.frame.value,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraspPoint:
        """Deserialise from a dictionary produced by :meth:`to_dict`."""
        try:
            return cls(
                position=np.asarray(d["position"], dtype=np.float64),
                approach=np.asarray(d["approach"], dtype=np.float64),
                axis=np.asarray(d["axis"], dtype=np.float64),
                grip_width_mm=float(d["grip_width_mm"]),
                score=float(d["score"]),
                frame=GraspFrame(d["frame"]),
                label=d.get("label", ""),
                metadata=d.get("metadata", {}),
            )
        except KeyError as exc:
            raise ValueError(f"GraspPoint.from_dict: missing key {exc}") from exc

    def __repr__(self) -> str:
        p = self.position
        return (
            f"GraspPoint(pos=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}] "
            f"{self.frame.value}, width={self.grip_width_mm:.1f}mm, "
            f"score={self.score:.2f}, label={self.label!r})"
        )
