"""Occlusion ratio and approach-clearance metrics.

These two metrics let the calculator report when a scene is too
cluttered for the heuristic parallel-jaw pipeline to be trusted, while
keeping the contract that ``compute()`` returns a candidate list, which
may be empty, together with a reason code.

Metric definitions
------------------

* :func:`occlusion_ratio` is the fraction of the convex hull area of
  the target that any of the supplied neighbouring object masks cover.
  Values lie in ``[0, 1]``: ``0`` means the target sits in free space,
  ``1`` means its hull is fully shadowed by other detected objects. The
  convex hull rather than the raw mask is what catches an object
  wrapped by a neighbour, which a strict pixel-overlap check misses.

* :func:`approach_clearance_mm` is the unobstructed distance in
  millimetres along an approach axis, from a 3D point in the camera
  frame to the first depth-map hit closer to the camera than the z
  coordinate of the marched ray. The depth map is queried by
  re-projecting each step through the pinhole intrinsics. It returns
  ``max_distance_mm`` when no obstruction is found within the budget.

Both metrics are pure functions: they depend on neither the calculator
nor grasp pose objects, they are deterministic for fixed inputs, and
they never mutate the supplied arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.robot.grasping.geometry import CameraIntrinsics

__all__ = ["approach_clearance_mm", "occlusion_ratio"]


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Return the convex hull of ``mask`` as a same-shape boolean image (falls back to the raw mask when cv2 is unavailable or there are too few points)."""
    if not mask.any():
        return mask.astype(bool, copy=False)
    try:
        import cv2 as cv
    except ImportError:  # pragma: no cover (cv2 is a hard dep elsewhere)
        return mask.astype(bool, copy=False)
    bin_mask = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    points = np.column_stack(np.nonzero(bin_mask)[::-1])  # (x, y)
    if points.shape[0] < 3:
        return mask.astype(bool, copy=False)
    hull = cv.convexHull(points.reshape(-1, 1, 2))
    out = np.zeros_like(bin_mask)
    cv.fillConvexPoly(out, hull, 255)
    return out.astype(bool)


def occlusion_ratio(
    target_mask: np.ndarray,
    other_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
) -> float:
    """Return the convex-hull occlusion ratio of ``target_mask`` in ``[0, 1]``: the target-hull pixels covered by the union of ``other_masks``, over the target hull area. It is ``0.0`` when the hull is empty or no neighbours are supplied."""
    target = np.asarray(target_mask).astype(bool, copy=False)
    hull = _convex_hull_mask(target)
    hull_area = int(hull.sum())
    if hull_area == 0:
        return 0.0
    if not other_masks:
        return 0.0
    H, W = hull.shape
    union = np.zeros((H, W), dtype=bool)
    for other in other_masks:
        arr = np.asarray(other)
        if arr.shape != (H, W):
            raise ValueError(
                "occlusion_ratio: other mask shape "
                f"{arr.shape} does not match target shape {(H, W)}"
            )
        union |= arr.astype(bool, copy=False)
    # Only a neighbour pixel overlapping the target hull counts as
    # occluding; a pixel outside the hull belongs to an unrelated object.
    covered = int(np.logical_and(hull, union).sum())
    ratio = covered / float(hull_area)
    return float(np.clip(ratio, 0.0, 1.0))


def approach_clearance_mm(
    point_3d_cam_mm: np.ndarray,
    approach_axis_cam: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: "CameraIntrinsics",
    *,
    max_distance_mm: float = 200.0,
    step_mm: float = 5.0,
    scale_to_mm: float = 1.0,
    clearance_tolerance_mm: float = 5.0,
) -> float:
    """Return the unobstructed approach distance in millimetres, ray-marching from ``point_3d_cam_mm`` back along ``-approach_axis_cam``, which points toward the camera along the path the gripper traverses. It stops at the first depth surface closer than the ``z`` of the ray by more than ``clearance_tolerance_mm``, the tolerance that guards against single-pixel depth noise, and otherwise returns ``max_distance_mm``."""
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0.0:
        raise ValueError("max_distance_mm must be finite and > 0")
    if not np.isfinite(step_mm) or step_mm <= 0.0:
        raise ValueError("step_mm must be finite and > 0")

    start = np.asarray(point_3d_cam_mm, dtype=np.float64).reshape(3)
    axis = np.asarray(approach_axis_cam, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        raise ValueError("approach_axis_cam cannot be the zero vector")
    axis = axis / n

    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be 2D")
    H, W = depth.shape

    travelled = 0.0
    steps = int(np.ceil(max_distance_mm / step_mm))
    for k in range(1, steps + 1):
        distance = min(k * step_mm, max_distance_mm)
        # Step toward the camera, along the negated approach direction.
        sample = start - axis * distance
        z = float(sample[2])
        if z <= 0.0:
            return float(travelled)
        u = sample[0] * intrinsics.fx / z + intrinsics.cx
        v = sample[1] * intrinsics.fy / z + intrinsics.cy
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < W and 0 <= iv < H):
            return float(min(distance, max_distance_mm))
        depth_mm = float(depth[iv, iu]) * float(scale_to_mm)
        if np.isfinite(depth_mm) and depth_mm > 0.0:
            if depth_mm + clearance_tolerance_mm < z:
                # A surface in the depth map sits in front of the ray, so
                # the gripper would collide before reaching this step.
                return float(travelled)
        travelled = float(distance)
    return float(min(max_distance_mm, travelled))
