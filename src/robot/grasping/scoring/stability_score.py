"""Stability and physical-validation scoring for grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robot.grasping.collision import GraspCollisionResult
from src.robot.grasping.planning import GraspPose

__all__ = [
    "StabilityScoreConfig",
    "stability_grasp_score",
    "stability_score_components",
    "table_clearance_score",
]


def _metadata_score(pose: GraspPose, key: str, fallback: float) -> float:
    raw = pose.metadata.get(key, fallback)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = fallback
    if not np.isfinite(value):
        value = fallback
    return float(np.clip(value, 0.0, 1.0))


def _validate_weights(*weights: float) -> None:
    values = [float(weight) for weight in weights]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("score weights must be finite and >= 0")
    if sum(values) <= 0.0:
        raise ValueError("at least one score weight must be positive")


@dataclass(frozen=True, slots=True)
class StabilityScoreConfig:
    """Weights for confidence, contact stability, and physical-validation results."""

    confidence_weight: float = 0.35
    contact_stability_weight: float = 0.25
    collision_free_weight: float = 0.30
    table_clearance_weight: float = 0.10
    good_table_clearance_mm: float = 50.0
    unknown_physical_score: float = 0.5

    def __post_init__(self) -> None:
        _validate_weights(
            self.confidence_weight,
            self.contact_stability_weight,
            self.collision_free_weight,
            self.table_clearance_weight,
        )
        good_clearance = float(self.good_table_clearance_mm)
        unknown = float(self.unknown_physical_score)
        if not np.isfinite(good_clearance) or good_clearance < 0.0:
            raise ValueError("good_table_clearance_mm must be finite and >= 0")
        if not np.isfinite(unknown) or not 0.0 <= unknown <= 1.0:
            raise ValueError("unknown_physical_score must be finite and in [0, 1]")
        object.__setattr__(self, "good_table_clearance_mm", good_clearance)
        object.__setattr__(self, "unknown_physical_score", unknown)


def table_clearance_score(
    clearance_mm: float | None,
    required_clearance_mm: float = 0.0,
    good_clearance_margin_mm: float = 50.0,
    *,
    unknown_score: float = 0.5,
) -> float:
    """Map support-plane clearance to ``[0, 1]``."""
    unknown = float(unknown_score)
    if not np.isfinite(unknown) or not 0.0 <= unknown <= 1.0:
        raise ValueError("unknown_score must be finite and in [0, 1]")
    if clearance_mm is None:
        return unknown
    clearance = float(clearance_mm)
    required = float(required_clearance_mm)
    good_margin = float(good_clearance_margin_mm)
    if not all(np.isfinite(value) for value in (clearance, required, good_margin)):
        raise ValueError("clearance values must be finite")
    if required < 0.0 or good_margin < 0.0:
        raise ValueError("clearance thresholds must be >= 0")
    if clearance < required:
        return 0.0
    if good_margin == 0.0:
        return 1.0
    return float(np.clip((clearance - required) / good_margin, 0.0, 1.0))


def _required_table_clearance(result: GraspCollisionResult | None) -> float:
    if result is None:
        return 0.0
    raw = result.metadata.get("min_table_clearance_required_mm", 0.0)
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        return 0.0
    return value


def stability_score_components(
    pose: GraspPose,
    *,
    collision_result: GraspCollisionResult | None = None,
    config: StabilityScoreConfig | None = None,
) -> dict[str, float]:
    """Return stability and physical-validation score components."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    if collision_result is not None and not isinstance(collision_result, GraspCollisionResult):
        raise TypeError("collision_result must be a GraspCollisionResult")
    cfg = config or StabilityScoreConfig()
    physical_unknown = cfg.unknown_physical_score
    return {
        "confidence": float(np.clip(pose.confidence, 0.0, 1.0)),
        "contact_stability": _metadata_score(pose, "stability", pose.confidence),
        "collision_free": physical_unknown if collision_result is None else float(collision_result.valid),
        "table_clearance": table_clearance_score(
            None if collision_result is None else collision_result.min_table_clearance_mm,
            _required_table_clearance(collision_result),
            cfg.good_table_clearance_mm,
            unknown_score=physical_unknown,
        ),
    }


def stability_grasp_score(
    pose: GraspPose,
    *,
    collision_result: GraspCollisionResult | None = None,
    config: StabilityScoreConfig | None = None,
) -> float:
    """Return a stability/physical-validation score in ``[0, 1]``."""
    cfg = config or StabilityScoreConfig()
    components = stability_score_components(pose, collision_result=collision_result, config=cfg)
    weights = {
        "confidence": cfg.confidence_weight,
        "contact_stability": cfg.contact_stability_weight,
        "collision_free": cfg.collision_free_weight,
        "table_clearance": cfg.table_clearance_weight,
    }
    total_weight = sum(weights.values())
    score = sum(components[name] * weight for name, weight in weights.items()) / total_weight
    return float(np.clip(score, 0.0, 1.0))
