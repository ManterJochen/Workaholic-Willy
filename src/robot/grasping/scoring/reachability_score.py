"""Explicit heuristic reachability scoring for grasp candidates.

This module does not claim robot IK reachability. It only scores configured
workspace and approach-direction heuristics until a real robot-specific IK
checker is wired into the planning layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.robot.grasping.planning import GraspPose

__all__ = [
    "ReachabilityScoreConfig",
    "WorkspaceBox",
    "approach_alignment_score",
    "reachability_grasp_score",
    "reachability_score_components",
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


def _validate_weights(*weights: float) -> None:
    values = [float(weight) for weight in weights]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("score weights must be finite and >= 0")
    if sum(values) <= 0.0:
        raise ValueError("at least one score weight must be positive")


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

    def margin_score(self, point_mm: np.ndarray) -> float:
        """Score distance from workspace faces in ``[0, 1]``."""
        point = _vec3(point_mm, "point_mm")
        lower_margin = point - self.min_corner_mm
        upper_margin = self.max_corner_mm - point
        if np.any(lower_margin < 0.0) or np.any(upper_margin < 0.0):
            return 0.0
        min_margin = float(min(np.min(lower_margin), np.min(upper_margin)))
        if self.good_margin_mm == 0.0:
            return 1.0
        return float(np.clip(min_margin / self.good_margin_mm, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ReachabilityScoreConfig:
    """Heuristic reachability settings; ``workspace`` and ``preferred_approach_axis`` are optional, and a missing check contributes ``unknown_score`` rather than pretending IK was run."""

    workspace: WorkspaceBox | None = None
    preferred_approach_axis: np.ndarray | None = None
    max_approach_angle_deg: float = 90.0
    workspace_weight: float = 0.70
    approach_weight: float = 0.30
    unknown_score: float = 0.5

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(self.workspace, WorkspaceBox):
            raise TypeError("workspace must be a WorkspaceBox")
        preferred = None
        if self.preferred_approach_axis is not None:
            preferred = _unit(self.preferred_approach_axis, "preferred_approach_axis")
        max_angle = float(self.max_approach_angle_deg)
        unknown = float(self.unknown_score)
        if not np.isfinite(max_angle) or not 0.0 < max_angle <= 180.0:
            raise ValueError("max_approach_angle_deg must be finite and in (0, 180]")
        if not np.isfinite(unknown) or not 0.0 <= unknown <= 1.0:
            raise ValueError("unknown_score must be finite and in [0, 1]")
        _validate_weights(self.workspace_weight, self.approach_weight)
        object.__setattr__(self, "preferred_approach_axis", preferred)
        object.__setattr__(self, "max_approach_angle_deg", max_angle)
        object.__setattr__(self, "unknown_score", unknown)


def approach_alignment_score(
    approach_axis: np.ndarray,
    preferred_axis: np.ndarray,
    max_angle_deg: float = 90.0,
) -> float:
    """Score directional alignment to a configured approach axis."""
    approach = _unit(approach_axis, "approach_axis")
    preferred = _unit(preferred_axis, "preferred_axis")
    max_angle = float(max_angle_deg)
    if not np.isfinite(max_angle) or not 0.0 < max_angle <= 180.0:
        raise ValueError("max_angle_deg must be finite and in (0, 180]")
    dot = float(np.clip(np.dot(approach, preferred), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    if angle_deg >= max_angle:
        return 0.0
    return float(np.clip(1.0 - angle_deg / max_angle, 0.0, 1.0))


def reachability_score_components(
    pose: GraspPose,
    config: ReachabilityScoreConfig | None = None,
) -> dict[str, float]:
    """Return configured workspace and approach-direction heuristic scores."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    cfg = config or ReachabilityScoreConfig()
    workspace_score = (
        cfg.unknown_score if cfg.workspace is None else cfg.workspace.margin_score(pose.position_mm)
    )
    approach_score = (
        cfg.unknown_score
        if cfg.preferred_approach_axis is None
        else approach_alignment_score(
            pose.approach_axis,
            cfg.preferred_approach_axis,
            cfg.max_approach_angle_deg,
        )
    )
    return {"workspace": workspace_score, "approach_alignment": approach_score}


def reachability_grasp_score(
    pose: GraspPose,
    config: ReachabilityScoreConfig | None = None,
) -> float:
    """Return an explicit heuristic reachability score in ``[0, 1]``."""
    cfg = config or ReachabilityScoreConfig()
    components = reachability_score_components(pose, cfg)
    total_weight = cfg.workspace_weight + cfg.approach_weight
    score = (
        components["workspace"] * cfg.workspace_weight
        + components["approach_alignment"] * cfg.approach_weight
    ) / total_weight
    return float(np.clip(score, 0.0, 1.0))
