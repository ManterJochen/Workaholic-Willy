"""Scoring and ranking utilities for 6D grasp candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.collision import GraspCollisionResult
from src.robot.grasping.planning import GraspPose

from .occlusion import approach_clearance_mm, occlusion_ratio
from .reachability_score import WorkspaceBox

__all__ = [
    "GraspScoreBreakdown",
    "WorkspaceBox",
    "approach_clearance_mm",
    "occlusion_ratio",
]


@dataclass(frozen=True, slots=True)
class GraspScoreBreakdown:
    """Transparent score breakdown for one candidate grasp pose."""

    pose: GraspPose
    total_score: float
    geometric_score: float
    stability_score: float
    reachability_score: float
    components: dict[str, dict[str, float]] = field(default_factory=dict)
    collision_result: GraspCollisionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "geometric_score": self.geometric_score,
            "stability_score": self.stability_score,
            "reachability_score": self.reachability_score,
            "components": {
                group: dict(values) for group, values in self.components.items()
            },
            "collision_result": (
                None if self.collision_result is None else self.collision_result.to_dict()
            ),
            "metadata": self.metadata,
            "pose": self.pose.to_dict(),
        }


