"""One surface contact: a position with its outward unit normal, in millimetres.

The position and normal are in a shared external frame. This is the substrate both
finger-contact grasp types compose, so they validate a contact the same way: a finite 3D
position and a normal within 1% of unit length, stored normalised. The
parallel-jaw :class:`~src.robot.grasping.contacts.ContactPair` holds two of them and the
multi-finger ``MultiContactGrasp`` holds N. Suction is deliberately not built on this,
because its ``approach`` is a pressing direction rather than an outward surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.geometry import as_vec3

__all__ = ["ContactPoint", "as_unit_normal"]

_UNIT_NORMAL_TOLERANCE = 0.01  # accept a normal within 1% of unit length, then normalise


def as_unit_normal(value: Any, name: str) -> np.ndarray:
    """Validate a near-unit normal, within 1% of unit length, and return it exactly unit."""
    arr = as_vec3(value, name)
    norm = float(np.linalg.norm(arr))
    if not 1.0 - _UNIT_NORMAL_TOLERANCE <= norm <= 1.0 + _UNIT_NORMAL_TOLERANCE:
        raise ValueError(f"{name} must be unit length (within 1%); got {norm:.4f}")
    return arr / norm


@dataclass(frozen=True, slots=True)
class ContactPoint:
    """A surface contact: a position in mm with an outward unit ``normal``, in a shared frame."""

    point_mm: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        point = as_vec3(self.point_mm, "point_mm")
        normal = as_unit_normal(self.normal, "normal")
        point.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "point_mm", point)
        object.__setattr__(self, "normal", normal)

    def to_dict(self) -> dict[str, Any]:
        return {"point_mm": self.point_mm.tolist(), "normal": self.normal.tolist()}
