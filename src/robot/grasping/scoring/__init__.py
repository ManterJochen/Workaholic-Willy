"""Scoring and ranking utilities for 6D grasp candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.robot.grasping.collision import GraspCollisionResult
from src.robot.grasping.planning import GraspPose

from .force_closure import (
    ForceClosureCertificate,
    certify_contact_pair,
    friction_threshold,
)
from .feasibility_score import (
    FeasibilityInputs,
    FeasibilityScoreConfig,
    feasibility_grasp_score,
    feasibility_score_components,
)
from .occlusion import approach_clearance_mm, occlusion_ratio
from .geometric_score import (
    GeometricScoreConfig,
    geometric_grasp_score,
    geometric_score_components,
    width_fit_score,
)
from .reachability_score import (
    ReachabilityScoreConfig,
    WorkspaceBox,
    approach_alignment_score,
    reachability_grasp_score,
    reachability_score_components,
)
from .stability_score import (
    StabilityScoreConfig,
    stability_grasp_score,
    stability_score_components,
    table_clearance_score,
)

__all__ = [
    "FeasibilityInputs",
    "FeasibilityScoreConfig",
    "ForceClosureCertificate",
    "GeometricScoreConfig",
    "GraspScoreBreakdown",
    "GraspScoreWeights",
    "ReachabilityScoreConfig",
    "StabilityScoreConfig",
    "WorkspaceBox",
    "approach_alignment_score",
    "approach_clearance_mm",
    "certify_contact_pair",
    "feasibility_grasp_score",
    "feasibility_score_components",
    "friction_threshold",
    "geometric_grasp_score",
    "geometric_score_components",
    "occlusion_ratio",
    "rank_grasp_poses",
    "reachability_grasp_score",
    "reachability_score_components",
    "rerank_breakdowns_with_feasibility",
    "score_grasp_pose",
    "stability_grasp_score",
    "stability_score_components",
    "table_clearance_score",
    "width_fit_score",
]


def _validate_weights(*weights: float) -> None:
    values = [float(weight) for weight in weights]
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("score weights must be finite and >= 0")
    if sum(values) <= 0.0:
        raise ValueError("at least one score weight must be positive")


@dataclass(frozen=True, slots=True)
class GraspScoreWeights:
    """Weights for the top-level grasp score. ``feasibility`` is an additive fourth component whose default of ``0.0`` has no effect until ``robot.grasping.feasibility.weight`` raises it."""

    geometric: float = 0.50
    stability: float = 0.35
    reachability: float = 0.15
    feasibility: float = 0.0

    def __post_init__(self) -> None:
        _validate_weights(
            self.geometric, self.stability, self.reachability, self.feasibility
        )


@dataclass(frozen=True, slots=True)
class GraspScoreBreakdown:
    """Transparent score breakdown for one candidate grasp pose. ``feasibility_score`` is always present, which keeps the telemetry layout stable, and contributes ``0.0`` unless feasibility is enabled."""

    pose: GraspPose
    total_score: float
    geometric_score: float
    stability_score: float
    reachability_score: float
    feasibility_score: float = 0.0
    components: dict[str, dict[str, float]] = field(default_factory=dict)
    collision_result: GraspCollisionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "geometric_score": self.geometric_score,
            "stability_score": self.stability_score,
            "reachability_score": self.reachability_score,
            "feasibility_score": self.feasibility_score,
            "components": {
                group: dict(values) for group, values in self.components.items()
            },
            "collision_result": (
                None if self.collision_result is None else self.collision_result.to_dict()
            ),
            "metadata": self.metadata,
            "pose": self.pose.to_dict(),
        }


def score_grasp_pose(
    pose: GraspPose,
    *,
    collision_result: GraspCollisionResult | None = None,
    weights: GraspScoreWeights | None = None,
    geometric_config: GeometricScoreConfig | None = None,
    stability_config: StabilityScoreConfig | None = None,
    reachability_config: ReachabilityScoreConfig | None = None,
    feasibility_inputs: FeasibilityInputs | None = None,
    feasibility_config: FeasibilityScoreConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> GraspScoreBreakdown:
    """Score one grasp pose with transparent component values; the feasibility sub-score is computed and added to ``components["feasibility"]`` only when both ``feasibility_inputs`` and ``feasibility_config`` are provided, otherwise it is ``0.0`` and the component is omitted."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    if collision_result is not None and not isinstance(collision_result, GraspCollisionResult):
        raise TypeError("collision_result must be a GraspCollisionResult")
    score_weights = weights or GraspScoreWeights()
    geometric_value = geometric_grasp_score(pose, geometric_config)
    stability_value = stability_grasp_score(
        pose,
        collision_result=collision_result,
        config=stability_config,
    )
    reachability_value = reachability_grasp_score(pose, reachability_config)

    components: dict[str, dict[str, float]] = {
        "geometric": geometric_score_components(pose, geometric_config),
        "stability": stability_score_components(
            pose,
            collision_result=collision_result,
            config=stability_config,
        ),
        "reachability": reachability_score_components(pose, reachability_config),
    }

    feasibility_value: float = 0.0
    if feasibility_inputs is not None and feasibility_config is not None:
        feasibility_value = feasibility_grasp_score(
            ik_quality=feasibility_inputs.ik_quality,
            approach_path=feasibility_inputs.approach_path,
            config=feasibility_config,
        )
        components["feasibility"] = feasibility_score_components(
            ik_quality=feasibility_inputs.ik_quality,
            approach_path=feasibility_inputs.approach_path,
            config=feasibility_config,
        )

    total_weight = (
        score_weights.geometric
        + score_weights.stability
        + score_weights.reachability
        + score_weights.feasibility
    )
    total = (
        geometric_value * score_weights.geometric
        + stability_value * score_weights.stability
        + reachability_value * score_weights.reachability
        + feasibility_value * score_weights.feasibility
    ) / total_weight
    return GraspScoreBreakdown(
        pose=pose,
        total_score=float(np.clip(total, 0.0, 1.0)),
        geometric_score=geometric_value,
        stability_score=stability_value,
        reachability_score=reachability_value,
        feasibility_score=feasibility_value,
        collision_result=collision_result,
        components=components,
        metadata={} if metadata is None else dict(metadata),
    )


def rank_grasp_poses(
    poses: Iterable[GraspPose],
    *,
    collision_results: Iterable[GraspCollisionResult | None] | None = None,
    weights: GraspScoreWeights | None = None,
    geometric_config: GeometricScoreConfig | None = None,
    stability_config: StabilityScoreConfig | None = None,
    reachability_config: ReachabilityScoreConfig | None = None,
    feasibility_inputs_per_pose: Iterable[FeasibilityInputs | None] | None = None,
    feasibility_config: FeasibilityScoreConfig | None = None,
    max_results: int | None = None,
) -> list[GraspScoreBreakdown]:
    """Score and sort grasp poses from best to worst; ``feasibility_inputs_per_pose`` and ``feasibility_config`` are optional and, when omitted, leave the ranking unchanged."""
    pose_list = list(poses)
    if max_results is not None and max_results < 1:
        raise ValueError("max_results must be >= 1 when provided")
    if collision_results is None:
        result_list: list[GraspCollisionResult | None] = [None] * len(pose_list)
    else:
        result_list = list(collision_results)
        if len(result_list) != len(pose_list):
            raise ValueError("collision_results must match poses length")
    if feasibility_inputs_per_pose is None:
        feas_list: list[FeasibilityInputs | None] = [None] * len(pose_list)
    else:
        feas_list = list(feasibility_inputs_per_pose)
        if len(feas_list) != len(pose_list):
            raise ValueError(
                "feasibility_inputs_per_pose must match poses length"
            )

    indexed_scores = [
        (
            index,
            score_grasp_pose(
                pose,
                collision_result=result,
                weights=weights,
                geometric_config=geometric_config,
                stability_config=stability_config,
                reachability_config=reachability_config,
                feasibility_inputs=feas,
                feasibility_config=feasibility_config,
                metadata={"input_index": index},
            ),
        )
        for index, (pose, result, feas) in enumerate(
            zip(pose_list, result_list, feas_list)
        )
    ]
    indexed_scores.sort(
        key=lambda item: (
            -item[1].total_score,
            -item[1].geometric_score,
            -item[1].stability_score,
            -item[1].reachability_score,
            -item[1].feasibility_score,
            item[0],
        )
    )
    ranked = [score for _, score in indexed_scores]
    return ranked if max_results is None else ranked[:max_results]


def rerank_breakdowns_with_feasibility(
    breakdowns: Sequence[GraspScoreBreakdown],
    feasibility_inputs_per_pose: Sequence[FeasibilityInputs | None],
    *,
    weights: GraspScoreWeights,
    feasibility_config: FeasibilityScoreConfig,
) -> list[GraspScoreBreakdown]:
    """Recompute ``total_score`` with the feasibility axis from existing breakdowns and re-sort.

    Each breakdown's geometric, stability and reachability sub-scores are reused unchanged and only
    the feasibility term is added. This is the re-rank that runs after the IK filter, because
    per-candidate IK quality does not exist before it. A ``None`` carrier, or a sub-signal that is
    disabled or unknown, scores the neutral 0.5 floor and so demotes nothing.
    """
    if len(feasibility_inputs_per_pose) != len(breakdowns):
        raise ValueError("feasibility_inputs_per_pose must match breakdowns length")
    total_weight = (
        weights.geometric + weights.stability + weights.reachability + weights.feasibility
    )
    rescored: list[tuple[int, GraspScoreBreakdown]] = []
    for index, (bd, feas) in enumerate(zip(breakdowns, feasibility_inputs_per_pose)):
        if feas is None:
            feas_value = 0.0
            new_components = bd.components
        else:
            feas_value = feasibility_grasp_score(
                ik_quality=feas.ik_quality,
                approach_path=feas.approach_path,
                config=feasibility_config,
            )
            new_components = dict(bd.components)
            new_components["feasibility"] = feasibility_score_components(
                ik_quality=feas.ik_quality,
                approach_path=feas.approach_path,
                config=feasibility_config,
            )
        total = (
            bd.geometric_score * weights.geometric
            + bd.stability_score * weights.stability
            + bd.reachability_score * weights.reachability
            + feas_value * weights.feasibility
        ) / total_weight
        rescored.append(
            (
                index,
                replace(
                    bd,
                    total_score=float(np.clip(total, 0.0, 1.0)),
                    feasibility_score=feas_value,
                    components=new_components,
                ),
            )
        )
    rescored.sort(
        key=lambda item: (
            -item[1].total_score,
            -item[1].geometric_score,
            -item[1].stability_score,
            -item[1].reachability_score,
            -item[1].feasibility_score,
            item[0],
        )
    )
    return [bd for _, bd in rescored]
