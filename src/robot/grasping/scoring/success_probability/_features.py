"""Feature extraction for the success-probability model: normalize_mode and extract_features."""

from __future__ import annotations

from typing import Any, Mapping

import math

import numpy as np
from ._schema import MODE_BUCKETS, _FEATURE_INDEX, _MODE_BUCKET_INDEX, _N_FEATURES


def normalize_mode(mode: str | None) -> str:
    """Collapse a raw mode string to one of :data:`MODE_BUCKETS`; an unknown mode and ``None`` both become ``"auto"``."""
    if mode is None:
        return "auto"
    text = str(mode).strip().lower()
    if text == "easy":
        return "easy"
    if text.startswith("dense"):
        return "dense"
    if text in ("auto", "autonomous"):
        return "auto"
    return "auto"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(out):
        return math.nan
    return out


def extract_features(
    breakdown: Any,
    *,
    mode: str | None,
    component_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> np.ndarray:
    """Return the locked v1 feature vector, 23 entries of ``float64``. A missing or non-finite entry becomes NaN, which :func:`predict_proba` later replaces with the training mean."""
    features = np.full(_N_FEATURES, np.nan, dtype=np.float64)

    if breakdown is not None:
        features[_FEATURE_INDEX["geometric_score"]] = _safe_float(
            getattr(breakdown, "geometric_score", math.nan)
        )
        features[_FEATURE_INDEX["stability_score"]] = _safe_float(
            getattr(breakdown, "stability_score", math.nan)
        )
        features[_FEATURE_INDEX["reachability_score"]] = _safe_float(
            getattr(breakdown, "reachability_score", math.nan)
        )
        features[_FEATURE_INDEX["feasibility_score"]] = _safe_float(
            getattr(breakdown, "feasibility_score", math.nan)
        )

        components: Mapping[str, Mapping[str, float]]
        if component_overrides is not None:
            components = component_overrides
        else:
            raw = getattr(breakdown, "components", None)
            components = raw if isinstance(raw, Mapping) else {}

        _fill_group(features, components.get("geometric"), {
            "antipodal": "geo_antipodal",
            "normal_opposition": "geo_normal_opposition",
            "axis_alignment": "geo_axis_alignment",
            "width_fit": "geo_width_fit",
        })
        _fill_group(features, components.get("stability"), {
            "confidence": "stab_confidence",
            "contact_stability": "stab_contact_stability",
            "collision_free": "stab_collision_free",
            "table_clearance": "stab_table_clearance",
        })
        _fill_group(features, components.get("reachability"), {
            "workspace": "reach_workspace",
            "approach_alignment": "reach_approach_alignment",
        })
        _fill_group(features, components.get("feasibility"), {
            "ik_quality": "feas_ik_quality",
            "joint_margin": "feas_joint_margin",
            "approach_clearance": "feas_approach_clearance",
            "corridor_risk": "feas_corridor_risk",
        })

        pose = getattr(breakdown, "pose", None)
        if pose is not None:
            features[_FEATURE_INDEX["pose_grip_width_mm"]] = _safe_float(
                getattr(pose, "grip_width_mm", math.nan)
            )
            features[_FEATURE_INDEX["pose_confidence"]] = _safe_float(
                getattr(pose, "confidence", math.nan)
            )

    bucket = normalize_mode(mode)
    bucket_idx = _MODE_BUCKET_INDEX[bucket]
    for idx, name in enumerate(MODE_BUCKETS):
        features[_FEATURE_INDEX[f"mode_{name}"]] = 1.0 if idx == bucket_idx else 0.0

    return features


def _fill_group(
    features: np.ndarray,
    group: Any,
    mapping: Mapping[str, str],
) -> None:
    if not isinstance(group, Mapping):
        return
    for src_key, dst_key in mapping.items():
        if src_key in group:
            features[_FEATURE_INDEX[dst_key]] = _safe_float(group[src_key])
