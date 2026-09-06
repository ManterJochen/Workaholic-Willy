"""Constructor argument validation for :class:`GraspCalculator`.

:func:`validate_calculator_args` owns the ~20 bounds and type checks and runs as
the first statement of the ``calculator`` constructor, so bad config is rejected
before the object is wired up.
"""

from __future__ import annotations

import numpy as np

from src.robot.grasping.geometry import CloudOutlierConfig
from src.robot.grasping.generation.deformables import DeformableHandlingStrategy
from src.robot.grasping.scoring import FeasibilityScoreConfig


def validate_calculator_args(
    *,
    min_grip_width_mm: float,
    max_grip_width_mm: float,
    normal_radius_mm: float,
    max_candidates: int,
    geometry_voxel_size_mm: float,
    max_geometry_points: int,
    dense_runtime_budget_ms: float,
    dense_non_convexity_threshold: float,
    topology_risk_threshold: float | None,
    friction_coefficient: float | None,
    heavy_occlusion_threshold: float | None,
    deformable_strategy: DeformableHandlingStrategy | None,
    grasp_depth_reference: str,
    grasp_top_penetration_mm: float,
    grasp_top_quantile: float,
    depth_band_mm: float | None,
    depth_band_near_pct: float,
    cloud_outlier_filter: CloudOutlierConfig | None,
    feasibility_weight: float,
    feasibility_config: FeasibilityScoreConfig | None,
) -> None:
    """Raise ``ValueError`` / ``TypeError`` for any out-of-range or wrong-typed constructor argument."""
    if min_grip_width_mm < 0.0 or max_grip_width_mm <= min_grip_width_mm:
        raise ValueError(
            "Gripper limits must satisfy 0 <= min_grip_width_mm < "
            f"max_grip_width_mm; got [{min_grip_width_mm}, {max_grip_width_mm}]"
        )
    if normal_radius_mm <= 0.0 or not np.isfinite(normal_radius_mm):
        raise ValueError("normal_radius_mm must be finite and > 0")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if geometry_voxel_size_mm <= 0.0 or not np.isfinite(geometry_voxel_size_mm):
        raise ValueError("geometry_voxel_size_mm must be finite and > 0")
    if max_geometry_points < 2:
        raise ValueError("max_geometry_points must be >= 2")
    if dense_runtime_budget_ms <= 0.0 or not np.isfinite(dense_runtime_budget_ms):
        raise ValueError("dense_runtime_budget_ms must be finite and > 0")
    if not 0.0 <= dense_non_convexity_threshold <= 1.0:
        raise ValueError(
            "dense_non_convexity_threshold must lie in [0.0, 1.0]; "
            f"got {dense_non_convexity_threshold}"
        )
    if topology_risk_threshold is not None:
        if not (
            np.isfinite(topology_risk_threshold)
            and 0.0 <= float(topology_risk_threshold) <= 1.0
        ):
            raise ValueError(
                "topology_risk_threshold must be in [0.0, 1.0] or None; "
                f"got {topology_risk_threshold}"
            )
    if friction_coefficient is not None:
        # A friction coefficient enables analytical force-closure
        # certification per candidate. ``None`` produces no certificate
        # and no metadata changes.
        if not (
            np.isfinite(friction_coefficient)
            and 0.0 <= float(friction_coefficient) <= 2.0
        ):
            raise ValueError(
                "friction_coefficient must be a finite value in "
                f"[0.0, 2.0] or None; got {friction_coefficient}"
            )
    if heavy_occlusion_threshold is not None:
        # When set, the calculator computes a convex-hull occlusion
        # ratio against ``other_object_masks`` and short-circuits with
        # HEAVY_OCCLUSION when the ratio exceeds the configured value.
        # ``None`` still stamps the ratio on telemetry but never rejects.
        if not (
            np.isfinite(heavy_occlusion_threshold)
            and 0.0 <= float(heavy_occlusion_threshold) <= 1.0
        ):
            raise ValueError(
                "heavy_occlusion_threshold must be in [0.0, 1.0] or None; "
                f"got {heavy_occlusion_threshold}"
            )
    if deformable_strategy is not None and not isinstance(
        deformable_strategy, DeformableHandlingStrategy
    ):
        # The deformable seam is opt-in; validate the Protocol shape at
        # construction so a typo'd strategy surfaces here, not mid-pipeline.
        raise TypeError(
            "deformable_strategy must implement DeformableHandlingStrategy; "
            f"got {type(deformable_strategy).__name__}"
        )
    if grasp_depth_reference not in ("centre", "top"):
        # The silhouette grasp-depth reference. "centre" (default) = the mask's median depth;
        # "top" = a near-surface quantile so the grasp references the object top, robust to
        # angled views of tall objects.
        raise ValueError(
            f"grasp_depth_reference must be 'centre' or 'top'; got {grasp_depth_reference!r}"
        )
    if not (np.isfinite(grasp_top_penetration_mm) and grasp_top_penetration_mm >= 0.0):
        raise ValueError(
            "grasp_top_penetration_mm must be finite and >= 0; "
            f"got {grasp_top_penetration_mm}"
        )
    if not (np.isfinite(grasp_top_quantile) and 0.0 < grasp_top_quantile < 100.0):
        raise ValueError(
            f"grasp_top_quantile must lie in (0, 100); got {grasp_top_quantile}"
        )
    if depth_band_mm is not None and not (
        np.isfinite(depth_band_mm) and depth_band_mm > 0.0
    ):
        raise ValueError(
            f"depth_band_mm must be finite and > 0, or None; got {depth_band_mm}"
        )
    if not (np.isfinite(depth_band_near_pct) and 0.0 <= depth_band_near_pct < 100.0):
        raise ValueError(
            f"depth_band_near_pct must lie in [0, 100); got {depth_band_near_pct}"
        )
    if cloud_outlier_filter is not None and not isinstance(
        cloud_outlier_filter, CloudOutlierConfig
    ):
        raise TypeError(
            "cloud_outlier_filter must be a CloudOutlierConfig or None; "
            f"got {type(cloud_outlier_filter).__name__}"
        )
    if not (np.isfinite(feasibility_weight) and feasibility_weight >= 0.0):
        raise ValueError(
            f"feasibility_weight must be finite and >= 0; got {feasibility_weight}"
        )
    if feasibility_config is not None and not isinstance(
        feasibility_config, FeasibilityScoreConfig
    ):
        raise TypeError(
            "feasibility_config must be a FeasibilityScoreConfig or None; "
            f"got {type(feasibility_config).__name__}"
        )
    if feasibility_weight > 0.0 and feasibility_config is None:
        raise ValueError(
            "feasibility_weight > 0 requires a feasibility_config (which sub-signals to score)"
        )
