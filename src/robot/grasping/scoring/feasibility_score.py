"""Feasibility-aware ranking signal.

A pure-function module that never rejects a grasp candidate. Rejection
stays with the hard-drop pipeline of the calculator, which is
``_apply_ik_filter`` for ``reachable=False`` poses and the swept-volume
``BLOCKED`` short-circuit. Feasibility only demotes: it moves candidates
with poor execution prospects down the ranking before the commit
decision.

The design is locked by the owner:

* A disabled flag makes that sub-signal contribute the neutral floor
  ``0.5``, so it never demotes a candidate below a peer that is
  enabled but unknown.
* A missing input under an enabled flag scores the same neutral floor,
  so an IK adapter that publishes no quality metrics keeps ranking
  exactly as it did.
* With every flag disabled :func:`feasibility_grasp_score` returns
  ``0.5``, which leaves the upstream weighted average unchanged.
* The apply-modes gate sits at the service layer, and it locks the easy
  mode out. :class:`FeasibilityScoreConfig` is mode-agnostic on purpose,
  and the service builds it only for the modes that opt in.

Every signal flag is wired and defaults to :data:`False`, so turning one
on is a small change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.robot.grasping.planning.reachability import IKQualityMetrics
from src.robot.grasping.scoring.corridor import (
    CorridorMode,
    CorridorReport,
)
from src.robot.grasping.motion.trajectory_safety import (
    ApproachPathOutcome,
    ApproachPathReport,
)

__all__ = [
    "FeasibilityInputs",
    "FeasibilityScoreConfig",
    "feasibility_grasp_score",
    "feasibility_score_components",
]


_NEUTRAL: float = 0.5
"""Neutral feasibility floor used by disabled signals and missing
inputs. It is chosen so that a disabled signal never reorders
candidates that carry any enabled and scored signal."""


@dataclass(frozen=True, slots=True)
class FeasibilityInputs:
    """Per-candidate feasibility carrier; all fields optional so a subset of signals still constructs."""

    ik_quality: Optional[IKQualityMetrics] = None
    approach_path: Optional[ApproachPathReport] = None
    corridor_report: Optional[CorridorReport] = None


@dataclass(frozen=True, slots=True)
class FeasibilityScoreConfig:
    """Operator knobs for the feasibility scorer: per-signal switches and weights."""

    ik_quality_enabled: bool = False
    joint_margin_enabled: bool = False
    swept_approach_enabled: bool = False
    corridor_risk_enabled: bool = False
    ik_quality_weight: float = 0.4
    joint_margin_weight: float = 0.3
    swept_approach_weight: float = 0.3
    corridor_risk_weight: float = 0.25
    condition_number_soft_max: float = 100.0
    # Reserved for an IK-signal refinement; exposed to keep the YAML schema forward-stable.
    min_singular_value_floor: float = 0.01
    joint_margin_full_score_deg: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "ik_quality_weight",
            "joint_margin_weight",
            "swept_approach_weight",
            "corridor_risk_weight",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if (
            self.ik_quality_weight
            + self.joint_margin_weight
            + self.swept_approach_weight
            + self.corridor_risk_weight
            <= 0.0
        ):
            raise ValueError(
                "At least one feasibility sub-weight must be > 0; "
                "got all zero (score would be undefined)."
            )
        for name in (
            "condition_number_soft_max",
            "min_singular_value_floor",
            "joint_margin_full_score_deg",
        ):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")


def _ik_quality_subscore(
    ik_quality: Optional[IKQualityMetrics],
    config: FeasibilityScoreConfig,
) -> float:
    """Map a Jacobian condition number to a [0, 1] score through ``soft_max / (soft_max + cn)``; a missing input scores the neutral floor."""
    if ik_quality is None or ik_quality.condition_number is None:
        return _NEUTRAL
    soft_max = config.condition_number_soft_max
    cn = float(ik_quality.condition_number)
    return max(0.0, min(1.0, soft_max / (soft_max + cn)))


def _joint_margin_subscore(
    ik_quality: Optional[IKQualityMetrics],
    config: FeasibilityScoreConfig,
) -> float:
    """Linear ramp from 0 deg (score 0) to ``full_score_deg`` (score 1)."""
    if ik_quality is None or ik_quality.joint_margin_deg is None:
        return _NEUTRAL
    margin = float(ik_quality.joint_margin_deg)
    ratio = margin / config.joint_margin_full_score_deg
    return max(0.0, min(1.0, ratio))


def _approach_clearance_subscore(
    approach_path: Optional[ApproachPathReport],
) -> float:
    """Map a swept-volume outcome to a discrete [0, 1] score: ``CLEAR`` and ``NO_OBSTACLES`` score 1.0, since no obstacles is treated as clear, ``BLOCKED`` scores 0.0, and ``SKIPPED`` or a missing outcome scores the neutral 0.5."""
    if approach_path is None:
        return _NEUTRAL
    outcome = approach_path.outcome
    if outcome in (ApproachPathOutcome.CLEAR, ApproachPathOutcome.NO_OBSTACLES):
        return 1.0
    if outcome == ApproachPathOutcome.BLOCKED:
        return 0.0
    return _NEUTRAL  # skipped, or an enum value added later


def _corridor_risk_subscore(
    corridor_report: Optional[CorridorReport],
) -> float:
    """Map directional corridor blockage to a [0, 1] feasibility score as ``1 - confidence``; a ``SKIPPED`` report scores the neutral floor."""
    if corridor_report is None:
        return _NEUTRAL
    if corridor_report.mode == CorridorMode.SKIPPED:
        return _NEUTRAL
    return max(
        0.0, min(1.0, 1.0 - float(corridor_report.blockage_confidence))
    )


def feasibility_score_components(
    *,
    ik_quality: Optional[IKQualityMetrics],
    approach_path: Optional[ApproachPathReport],
    config: FeasibilityScoreConfig,
    corridor_report: Optional[CorridorReport] = None,
) -> dict[str, float]:
    """Return the four per-signal sub-scores keyed for telemetry, ``ik_quality``, ``joint_margin``, ``approach_clearance`` and ``corridor_risk``; a disabled signal contributes the neutral floor, so all four are always present."""
    return {
        "ik_quality": (
            _ik_quality_subscore(ik_quality, config)
            if config.ik_quality_enabled
            else _NEUTRAL
        ),
        "joint_margin": (
            _joint_margin_subscore(ik_quality, config)
            if config.joint_margin_enabled
            else _NEUTRAL
        ),
        "approach_clearance": (
            _approach_clearance_subscore(approach_path)
            if config.swept_approach_enabled
            else _NEUTRAL
        ),
        "corridor_risk": (
            _corridor_risk_subscore(corridor_report)
            if config.corridor_risk_enabled
            else _NEUTRAL
        ),
    }


def feasibility_grasp_score(
    *,
    ik_quality: Optional[IKQualityMetrics],
    approach_path: Optional[ApproachPathReport],
    config: FeasibilityScoreConfig,
    corridor_report: Optional[CorridorReport] = None,
) -> float:
    """Weighted feasibility score in [0, 1]. It aggregates the enabled per-signal sub-scores by weight and returns the neutral floor ``0.5`` when none is enabled, which leaves the calculator ranking as it would without the axis."""
    enabled: list[tuple[float, float]] = []
    if config.ik_quality_enabled:
        enabled.append(
            (config.ik_quality_weight, _ik_quality_subscore(ik_quality, config))
        )
    if config.joint_margin_enabled:
        enabled.append(
            (
                config.joint_margin_weight,
                _joint_margin_subscore(ik_quality, config),
            )
        )
    if config.swept_approach_enabled:
        enabled.append(
            (
                config.swept_approach_weight,
                _approach_clearance_subscore(approach_path),
            )
        )
    if config.corridor_risk_enabled:
        enabled.append(
            (
                config.corridor_risk_weight,
                _corridor_risk_subscore(corridor_report),
            )
        )
    if not enabled:
        return _NEUTRAL
    total_weight = sum(w for w, _ in enabled)
    if total_weight <= 0.0:
        return _NEUTRAL
    weighted = sum(w * s for w, s in enabled) / total_weight
    return max(0.0, min(1.0, weighted))
