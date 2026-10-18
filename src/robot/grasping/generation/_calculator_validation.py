"""Constructor argument validation for :class:`GraspCalculator`.

:func:`validate_calculator_args` owns the bounds and type checks and runs as the first statement of the ``calculator``
constructor, so bad config is rejected before the object is wired up.
"""

from __future__ import annotations

import numpy as np

from src.robot.grasping.generation.deformables import DeformableHandlingStrategy


def validate_calculator_args(
    *,
    min_grip_width_mm: float,
    max_grip_width_mm: float,
    max_candidates: int,
    topology_risk_threshold: float | None,
    heavy_occlusion_threshold: float | None,
    deformable_strategy: DeformableHandlingStrategy | None,
) -> None:
    """Raise ``ValueError`` / ``TypeError`` for any out-of-range or wrong-typed constructor argument."""
    if min_grip_width_mm < 0.0 or max_grip_width_mm <= min_grip_width_mm:
        raise ValueError(
            "Gripper limits must satisfy 0 <= min_grip_width_mm < "
            f"max_grip_width_mm; got [{min_grip_width_mm}, {max_grip_width_mm}]"
        )
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if topology_risk_threshold is not None:
        if not (
            np.isfinite(topology_risk_threshold)
            and 0.0 <= float(topology_risk_threshold) <= 1.0
        ):
            raise ValueError(
                "topology_risk_threshold must be in [0.0, 1.0] or None; "
                f"got {topology_risk_threshold}"
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
