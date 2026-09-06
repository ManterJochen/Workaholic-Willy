"""Small pure bridges the pick loop uses to adapt its inputs.

Two adapters live here rather than in ``pick_loop``: one turns a chosen
``GraspPoint`` into the ``GraspPose`` the swept-volume validator wants; the
other materialises ``TargetCandidate`` records from the per-segmentation
successes the clutter-aware selector orders.
"""

from __future__ import annotations

import numpy as np

from src.robot.grasping.planning.grasp_pose import GraspPose
from src.robot.grasping.collision.candidate_filter import grasp_point_to_pose
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspPoint
from src.robot.grasping.types.perception import SegmentationLike
from src.robot.grasping.loop.target_selector import TargetCandidate


def _grasp_point_to_grasp_pose(grasp: GraspPoint) -> GraspPose:
    """Bridge a `GraspPoint` to a `GraspPose` for the swept-volume validator.

    Delegates to `collision.candidate_filter.grasp_point_to_pose`, the one adapter this loop and the
    deep calculator share. This name is a module-private alias, never a second copy of the contact
    synthesis.
    """
    return grasp_point_to_pose(grasp)


def _build_target_candidates(
    successes: list[tuple[int, GraspResult]],
    depth_map: np.ndarray,
    segmentations: tuple[SegmentationLike, ...],
) -> tuple[TargetCandidate, ...]:
    """Materialise :class:`TargetCandidate` records from per-segmentation successes.

    The centroid depth is the median depth over the segmentation mask (robust to outliers and missing
    ``NaN`` / ``0`` values); the bounding box is the axis-aligned bbox of the mask's true pixels. Both are
    cheap and only run on the disjoint set of successful segmentations.
    """

    out: list[TargetCandidate] = []
    for seg_idx, result in successes:
        seg = segmentations[seg_idx]
        mask_raw = getattr(seg, "mask", None)
        if mask_raw is None:
            continue
        mask = np.asarray(mask_raw).astype(bool, copy=False)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        depth_vals = np.asarray(depth_map, dtype=np.float64)[mask]
        finite = depth_vals[np.isfinite(depth_vals) & (depth_vals > 0.0)]
        centroid_depth = float(np.median(finite)) if finite.size else float("inf")
        out.append(
            TargetCandidate(
                segmentation_index=seg_idx,
                mask=mask,
                centroid_depth_mm=centroid_depth,
                local_score=float(result.top_score),
                bbox_px=bbox,
            )
        )
    return tuple(out)
