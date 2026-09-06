"""The contact-pair value object for parallel-jaw grasping.

It is two opposing surface points that may fit between the jaws of a gripper. It is
geometry-only: no robot reachability, no collision state, no motion planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.geometry import as_vec3

from .contact_point import ContactPoint, as_unit_normal

__all__ = ["ContactPair"]


@dataclass(frozen=True, slots=True)
class ContactPair:
    """Candidate opposing contacts for a parallel-jaw gripper, as two :class:`ContactPoint` values."""

    point_a: np.ndarray
    point_b: np.ndarray
    normal_a: np.ndarray
    normal_b: np.ndarray
    distance_mm: float
    antipodal_score: float
    axis_alignment: float = 0.0
    normal_opposition: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_a = as_vec3(self.point_a, "point_a")
        point_b = as_vec3(self.point_b, "point_b")
        normal_a = as_unit_normal(self.normal_a, "normal_a")
        normal_b = as_unit_normal(self.normal_b, "normal_b")

        distance = float(self.distance_mm)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("ContactPair.distance_mm must be finite and >= 0")
        actual_distance = float(np.linalg.norm(point_b - point_a))
        if abs(actual_distance - distance) > max(1e-3, distance * 1e-6):
            raise ValueError(
                "ContactPair.distance_mm must match ||point_b - point_a||"
            )

        score = float(self.antipodal_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("ContactPair.antipodal_score must be in [0, 1]")
        axis_alignment = float(self.axis_alignment)
        normal_opposition = float(self.normal_opposition)
        if not 0.0 <= axis_alignment <= 1.0:
            raise ValueError("ContactPair.axis_alignment must be in [0, 1]")
        if not 0.0 <= normal_opposition <= 1.0:
            raise ValueError("ContactPair.normal_opposition must be in [0, 1]")

        for arr in (point_a, point_b, normal_a, normal_b):
            arr.setflags(write=False)
        object.__setattr__(self, "point_a", point_a)
        object.__setattr__(self, "point_b", point_b)
        object.__setattr__(self, "normal_a", normal_a)
        object.__setattr__(self, "normal_b", normal_b)
        object.__setattr__(self, "distance_mm", distance)
        object.__setattr__(self, "antipodal_score", score)
        object.__setattr__(self, "axis_alignment", axis_alignment)
        object.__setattr__(self, "normal_opposition", normal_opposition)

    @property
    def center_mm(self) -> np.ndarray:
        center = 0.5 * (self.point_a + self.point_b)
        center.setflags(write=False)
        return center

    @property
    def closing_axis(self) -> np.ndarray:
        delta = self.point_b - self.point_a
        norm = float(np.linalg.norm(delta))
        if norm < 1e-12:
            raise ValueError("ContactPair has zero contact distance")
        axis = delta / norm
        axis.setflags(write=False)
        return axis

    @property
    def contact_a(self) -> ContactPoint:
        """The first contact, ``point_a`` with ``normal_a``, as the shared :class:`ContactPoint`."""
        return ContactPoint(self.point_a, self.normal_a)

    @property
    def contact_b(self) -> ContactPoint:
        """The second contact, ``point_b`` with ``normal_b``, as the shared :class:`ContactPoint`."""
        return ContactPoint(self.point_b, self.normal_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_a": self.point_a.tolist(),
            "point_b": self.point_b.tolist(),
            "normal_a": self.normal_a.tolist(),
            "normal_b": self.normal_b.tolist(),
            "distance_mm": self.distance_mm,
            "antipodal_score": self.antipodal_score,
            "axis_alignment": self.axis_alignment,
            "normal_opposition": self.normal_opposition,
            "metadata": self.metadata,
        }
