"""Explicit heuristic reachability scoring for grasp candidates.

This module does not claim robot IK reachability. It only scores configured
workspace and approach-direction heuristics until a real robot-specific IK
checker is wired into the planning layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


__all__ = [
    "WorkspaceBox",
]

_EPS = 1e-9


def _vec3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _unit(value: Any, name: str) -> np.ndarray:
    vector = _vec3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        raise ValueError(f"{name} cannot be the zero vector")
    output = vector / norm
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class WorkspaceBox:
    """Axis-aligned workspace box in the same frame as the grasp pose."""

    min_corner_mm: np.ndarray
    max_corner_mm: np.ndarray
    good_margin_mm: float = 100.0

    def __post_init__(self) -> None:
        min_corner = _vec3(self.min_corner_mm, "WorkspaceBox.min_corner_mm")
        max_corner = _vec3(self.max_corner_mm, "WorkspaceBox.max_corner_mm")
        good_margin = float(self.good_margin_mm)
        if not np.all(min_corner < max_corner):
            raise ValueError("WorkspaceBox min_corner_mm must be strictly below max_corner_mm")
        if not np.isfinite(good_margin) or good_margin < 0.0:
            raise ValueError("WorkspaceBox.good_margin_mm must be finite and >= 0")
        min_corner.setflags(write=False)
        max_corner.setflags(write=False)
        object.__setattr__(self, "min_corner_mm", min_corner)
        object.__setattr__(self, "max_corner_mm", max_corner)
        object.__setattr__(self, "good_margin_mm", good_margin)

    def contains(self, point_mm: np.ndarray) -> bool:
        point = _vec3(point_mm, "point_mm")
        return bool(np.all(point >= self.min_corner_mm) and np.all(point <= self.max_corner_mm))



