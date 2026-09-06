"""Shadow-mode runtime adapter + metadata keys for the success-probability model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import logging
import math

import numpy as np
from ._schema import ArtifactSchemaError
from ._features import extract_features
from ._model import (
    SuccessProbabilityModel,
    load_success_probability_model,
    predict_proba,
)



# Shadow mode wires the predictor into the runtime grasping path and no further:
# every candidate ``GraspCalculator`` produces is annotated with a calibrated
# success probability, and that value never feeds back into ranking, selection or
# execution. The carrier types below are the only surface the orchestrator and
# service layers consume, and everything else stays behind the public API here.
#
# The invariants that keeps:
#   * With ``ctx is None``, which is the default, no GraspPoint metadata is
#     mutated.
#   * Candidate ordering never changes, because the annotation reads
#     ``GraspPoint.score`` without writing it.
#   * A missing or invalid artifact produces ``ctx=None`` and a structured
#     warning, and no exception reaches the runtime path.

#: Warnings emit under the module path of the package rather than this _shadow leaf, so
#: every caller sees one logger name however the package is split internally.
_DEFAULT_LOGGER_NAME = "src.robot.grasping.scoring.success_probability"

_VALID_LIFECYCLE_PHASES: frozenset[str] = frozenset({"shadow", "canary", "active"})

#: Per-candidate ``GraspPoint.metadata`` keys the annotator writes. The names are
#: locked, because downstream tooling such as the calibration audit and active
#: ranking reads them verbatim.
SHADOW_METADATA_KEY_PROBABILITY: str = "shadow_predicted_success_probability"
SHADOW_METADATA_KEY_MODEL_VERSION: str = "shadow_success_probability_model_version"
SHADOW_METADATA_KEY_LIFECYCLE_PHASE: str = "shadow_model_lifecycle_phase"

# Blend metadata keys, added only when the bounded probability-blend reranker
# fires. Their absence from a ``GraspPoint.metadata`` is the signal that the
# candidate was ranked by geometry alone.
SHADOW_METADATA_KEY_FINAL_SCORE: str = "ranking_blend_final_score"
SHADOW_METADATA_KEY_GEOMETRIC_PRE_BLEND: str = "ranking_blend_geometric_score"
SHADOW_METADATA_KEY_BLEND_WEIGHT: str = "ranking_blend_weight"

#: Per-candidate uncertainty re-rank audit keys (stamped additively, mirroring the blend keys above).
UNCERTAINTY_RERANK_METADATA_KEY_ADJUSTED: str = "uncertainty_rerank_adjusted_score"
UNCERTAINTY_RERANK_METADATA_KEY_GEOMETRIC: str = "uncertainty_rerank_geometric_score"
UNCERTAINTY_RERANK_METADATA_KEY_WEIGHT: str = "uncertainty_rerank_weight"

#: The locked dense-only subset of grasp modes the reranker may reorder. It mirrors
#: the schema-side ``_DENSE_ONLY`` set in
#: :class:`GraspingSuccessModelConfig._validate`, duplicated here so the runtime
#: carrier validates the same contract without pydantic.
_DENSE_BLEND_MODES: frozenset[str] = frozenset(
    {"dense_clutter", "dense_autonomous"}
)


@dataclass(frozen=True, slots=True)
class ShadowSuccessContext:
    """Immutable runtime carrier for a loaded shadow-mode success model, shareable across attempts and threads. ``mode_label`` is the raw mode string that :func:`normalize_mode` later buckets."""

    model: SuccessProbabilityModel
    version_label: str
    lifecycle_phase: str
    mode_label: str


@dataclass(frozen=True, slots=True)
class ShadowSuccessTelemetry:
    """Telemetry for the shadow probability of the winning grasp.

    ``predicted_success_probability`` is ``None`` only when no candidate could be scored.
    ``features`` is the locked 23-dimensional feature vector of the winner, retained so that a
    logged record carries the features and outcome together, which is the pair the offline
    success-model trainer learns from; see
    :func:`src.robot.grasping.calibration.success_model_calibration.build_dataset_from_records`.
    """

    predicted_success_probability: float | None
    success_probability_model_version: str | None
    model_lifecycle_phase: str | None
    features: tuple[float, ...] | None = None


def extract_features_from_grasp_point(point: Any, *, mode: str | None) -> np.ndarray:
    """Build the locked 23-d feature vector from a runtime ``GraspPoint``, reading subscores off ``metadata`` so the layout stays single-sourced in :func:`extract_features`."""
    md = getattr(point, "metadata", None) or {}
    pose_view = SimpleNamespace(
        grip_width_mm=float(getattr(point, "grip_width_mm", math.nan)),
        confidence=md.get("pose_confidence"),
    )
    components = md.get("score_components")
    if not isinstance(components, Mapping):
        components = {}
    breakdown_view = SimpleNamespace(
        geometric_score=md.get("geometric_score"),
        stability_score=md.get("stability_score"),
        reachability_score=md.get("reachability_score"),
        feasibility_score=md.get("feasibility_score"),
        components=components,
        pose=pose_view,
    )
    return extract_features(breakdown_view, mode=mode)


def try_load_shadow_success_context(
    *,
    enabled: bool,
    artifact_dir: str | Path,
    mode_label: str,
    version_label: str = "v1",
    lifecycle_phase: str = "shadow",
    logger: logging.Logger | None = None,
) -> ShadowSuccessContext | None:
    """Load the shadow model eagerly, falling back safely. It returns ``None`` when the model is disabled, silently, and when the lifecycle_phase is invalid or the artifact is missing or malformed, with one warning. A return that is not ``None`` has passed schema validation."""
    if not enabled:
        return None
    log = logger if logger is not None else logging.getLogger(_DEFAULT_LOGGER_NAME)
    if lifecycle_phase not in _VALID_LIFECYCLE_PHASES:
        log.warning(
            "shadow_success_model load skipped: invalid lifecycle_phase=%r",
            lifecycle_phase,
        )
        return None
    # The promotion gate. A lifecycle phase that grants behavioural influence,
    # "canary" or "active", must point at an artifact carrying a valid and
    # untampered promotion.json. "shadow" stays ungated by design, since it has no
    # behavioural influence. The import is local, so the promotion module is
    # touched on the strict-phase path only and the shadow path is unchanged.
    if lifecycle_phase in ("canary", "active"):
        from src.robot.grasping.calibration.model_promotion import (
            verify_promotion,
        )

        ok, reasons = verify_promotion(artifact_dir)
        if not ok:
            log.warning(
                "shadow_success_model load refused for lifecycle_phase=%s: "
                "promotion gate failed: %s",
                lifecycle_phase,
                "; ".join(reasons),
            )
            return None
    try:
        model = load_success_probability_model(artifact_dir)
    except (ArtifactSchemaError, FileNotFoundError, OSError, ValueError) as exc:
        log.warning(
            "shadow_success_model load failed: artifact_dir=%s, error=%s",
            artifact_dir,
            exc,
        )
        return None
    return ShadowSuccessContext(
        model=model,
        version_label=str(version_label),
        lifecycle_phase=lifecycle_phase,
        mode_label=str(mode_label),
    )


def annotate_grasp_points_with_shadow_probability(
    points: Sequence[Any],
    *,
    ctx: ShadowSuccessContext | None,
) -> ShadowSuccessTelemetry | None:
    """Annotate the ``metadata`` of every entry in ``points`` with shadow telemetry, never touching ``score``, and return the telemetry of the winner at ``points[0]``. It does nothing and returns ``None`` when ``ctx`` is ``None`` or ``points`` is empty."""
    if ctx is None or not points:
        return None

    features_matrix = np.vstack(
        [
            extract_features_from_grasp_point(point, mode=ctx.mode_label)
            for point in points
        ]
    )
    probabilities = predict_proba(ctx.model, features_matrix)

    for point, prob in zip(points, probabilities):
        md = getattr(point, "metadata", None)
        if not isinstance(md, dict):
            continue
        md[SHADOW_METADATA_KEY_PROBABILITY] = float(prob)
        md[SHADOW_METADATA_KEY_MODEL_VERSION] = ctx.version_label
        md[SHADOW_METADATA_KEY_LIFECYCLE_PHASE] = ctx.lifecycle_phase
        # For the RL shadow path only: the same per-candidate probability is aliased under the
        # SAR-named ranking key ``predicted_success_probability``, index 5, which is otherwise a
        # dead 0.0, so the RL ranking rows carry it. It goes into the shadow-precedence sub-dict
        # and touches neither ``score`` nor the frozen success-model feature vector.
        md.setdefault("shadow", {})["predicted_success_probability"] = float(prob)

    return ShadowSuccessTelemetry(
        predicted_success_probability=float(probabilities[0]),
        success_probability_model_version=ctx.version_label,
        model_lifecycle_phase=ctx.lifecycle_phase,
        features=tuple(float(v) for v in features_matrix[0]),
    )
