"""Geometry-only scoring for 6D grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.planning import GraspPose

__all__ = [
    "GeometricScoreConfig",
    "geometric_grasp_score",
    "geometric_score_components",
    "width_fit_score",
]


def _unit_interval(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(scalar, 0.0, 1.0))


def _metadata_score(pose: GraspPose, key: str, fallback: float) -> float:
    raw = pose.metadata.get(key, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = fallback
    if not np.isfinite(value):
        value = fallback
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class GeometricScoreConfig:
    """Weights and width limits for geometry-only grasp quality."""

    min_width_mm: float = 5.0
    max_width_mm: float = 150.0
    antipodal_weight: float = 0.40
    normal_opposition_weight: float = 0.25
    axis_alignment_weight: float = 0.25
    width_fit_weight: float = 0.10

    def __post_init__(self) -> None:
        min_width = float(self.min_width_mm)
        max_width = float(self.max_width_mm)
        if not np.isfinite(min_width) or not np.isfinite(max_width) or min_width < 0.0:
            raise ValueError("width limits must be finite and satisfy min_width_mm >= 0")
        if max_width <= min_width:
            raise ValueError("width limits must satisfy min_width_mm < max_width_mm")
        object.__setattr__(self, "min_width_mm", min_width)
        object.__setattr__(self, "max_width_mm", max_width)
        _validate_weights(
            self.antipodal_weight,
            self.normal_opposition_weight,
            self.axis_alignment_weight,
            self.width_fit_weight,
        )


def _validate_weights(*weights: float) -> None:
    values = [float(weight) for weight in weights]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("score weights must be finite and >= 0")
    if sum(values) <= 0.0:
        raise ValueError("at least one score weight must be positive")


def width_fit_score(width_mm: float, min_width_mm: float, max_width_mm: float) -> float:
    """Score gripper width fit in ``[0, 1]`` with a peak at the range midpoint."""
    width = float(width_mm)
    min_width = float(min_width_mm)
    max_width = float(max_width_mm)
    if not all(np.isfinite(value) for value in (width, min_width, max_width)):
        raise ValueError("width values must be finite")
    if min_width < 0.0 or max_width <= min_width:
        raise ValueError("width limits must satisfy 0 <= min_width_mm < max_width_mm")
    if width < min_width or width > max_width:
        return 0.0
    midpoint = 0.5 * (min_width + max_width)
    half_range = 0.5 * (max_width - min_width)
    return float(np.clip(1.0 - abs(width - midpoint) / half_range, 0.0, 1.0))


def geometric_score_components(
    pose: GraspPose,
    config: GeometricScoreConfig | None = None,
) -> dict[str, float]:
    """Return geometry score components for diagnostics and ranking."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    cfg = config or GeometricScoreConfig()
    return {
        "antipodal": _unit_interval(pose.score, "pose.score"),
        "normal_opposition": _metadata_score(pose, "normal_opposition", pose.score),
        "axis_alignment": _metadata_score(pose, "axis_alignment", pose.score),
        "width_fit": width_fit_score(pose.grip_width_mm, cfg.min_width_mm, cfg.max_width_mm),
    }


def geometric_grasp_score(
    pose: GraspPose,
    config: GeometricScoreConfig | None = None,
) -> float:
    """Return a geometry-only grasp score in ``[0, 1]``."""
    cfg = config or GeometricScoreConfig()
    components = geometric_score_components(pose, cfg)
    weights = {
        "antipodal": cfg.antipodal_weight,
        "normal_opposition": cfg.normal_opposition_weight,
        "axis_alignment": cfg.axis_alignment_weight,
        "width_fit": cfg.width_fit_weight,
    }
    total_weight = sum(weights.values())
    score = sum(components[name] * weight for name, weight in weights.items()) / total_weight
    return float(np.clip(score, 0.0, 1.0))
