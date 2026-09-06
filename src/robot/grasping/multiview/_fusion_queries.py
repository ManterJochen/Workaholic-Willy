"""Multi-view-fusion read-side queries: corridor evidence and viewpoint information gain.

``corridor_evidence`` and ``viewpoint_information_gain`` are free functions taking the
``SceneFusion`` state explicitly, as cfg, hits, seen, shape and views_accepted, and their float
arithmetic is pinned for determinism. They construct ``CorridorEvidence`` and ``ViewpointGain``, so
they import those carriers at module top from ``fusion``, while the ``SceneFusion`` methods import
these functions inside the method body. That is what breaks the cycle: ``fusion`` never imports this
leaf at module load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from src.geometry import Frame, Transform
from src.robot.grasping.multiview.fusion import CorridorEvidence, ViewpointGain

if TYPE_CHECKING:
    from src.robot.grasping.multiview.fusion import FusionConfig


def corridor_evidence(
    *,
    cfg: "FusionConfig",
    hits: np.ndarray,
    seen: np.ndarray,
    shape: Tuple[int, int, int],
    views_accepted: int,
    position_mm: np.ndarray,
    approach: np.ndarray,
    length_mm: float,
    radius_mm: float,
) -> CorridorEvidence:
    """Evidence inside an approach corridor (see :meth:`SceneFusion.corridor_evidence`)."""

    # ---- input contract ---------------------------------------------------
    p = np.asarray(position_mm, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError("position_mm must have 3 components")
    if not np.all(np.isfinite(p)):
        raise ValueError("position_mm must be finite")
    a = np.asarray(approach, dtype=np.float64).reshape(-1)
    if a.shape != (3,):
        raise ValueError("approach must have 3 components")
    if not np.all(np.isfinite(a)):
        raise ValueError("approach must be finite")
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("approach must have non-zero finite norm")
    a = a / n  # unit vector
    if (
        not isinstance(length_mm, (int, float))
        or isinstance(length_mm, bool)
        or not np.isfinite(float(length_mm))
        or float(length_mm) <= 0.0
    ):
        raise ValueError("length_mm must be finite and > 0")
    if (
        not isinstance(radius_mm, (int, float))
        or isinstance(radius_mm, bool)
        or not np.isfinite(float(radius_mm))
        or float(radius_mm) <= 0.0
    ):
        raise ValueError("radius_mm must be finite and > 0")
    L = float(length_mm)
    R = float(radius_mm)

    # ---- compute axis-aligned voxel bbox enclosing the cylinder ----------
    # The corridor extends from p along ``-a`` for L mm, so expanding by
    # R in every direction bounds the cylinder. The voxel indices inside
    # that bbox are then iterated and the exact in-cylinder test is
    # applied to each.
    end = p + (-a) * L  # far end of the corridor
    # Bounding box in BASE-frame mm.
    lo_mm = np.minimum(p, end) - R
    hi_mm = np.maximum(p, end) + R

    ox, oy, oz = cfg.grid_origin_mm
    vs = float(cfg.voxel_size_mm)
    nx, ny, nz = shape

    # Map mm bounds to voxel index bounds, clipped to grid.
    def _clip_idx(value_mm: float, origin_mm: float, n: int) -> int:
        idx = int(np.floor((value_mm - origin_mm) / vs))
        if idx < 0:
            return 0
        if idx >= n:
            return n
        return idx

    ix_lo = _clip_idx(lo_mm[0], ox, nx)
    iy_lo = _clip_idx(lo_mm[1], oy, ny)
    iz_lo = _clip_idx(lo_mm[2], oz, nz)
    # +1 so the hi index is exclusive after clipping the hi corner.
    ix_hi = _clip_idx(hi_mm[0], ox, nx) + 1
    iy_hi = _clip_idx(hi_mm[1], oy, ny) + 1
    iz_hi = _clip_idx(hi_mm[2], oz, nz) + 1
    ix_hi = min(ix_hi, nx)
    iy_hi = min(iy_hi, ny)
    iz_hi = min(iz_hi, nz)

    if ix_hi <= ix_lo or iy_hi <= iy_lo or iz_hi <= iz_lo:
        return CorridorEvidence(
            queried_voxels=0,
            hit_voxels=0,
            seen_voxels=0,
            hit_fraction=0.0,
            seen_fraction=0.0,
            views_accepted=views_accepted,
        )

    # Voxel center coordinates (BASE-frame mm) inside the bbox.
    xs = ox + (np.arange(ix_lo, ix_hi, dtype=np.float64) + 0.5) * vs
    ys = oy + (np.arange(iy_lo, iy_hi, dtype=np.float64) + 0.5) * vs
    zs = oz + (np.arange(iz_lo, iz_hi, dtype=np.float64) + 0.5) * vs
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    # Position of each voxel center relative to the corridor anchor.
    dx = gx - p[0]
    dy = gy - p[1]
    dz = gz - p[2]
    # Project onto the corridor axis (which points along ``-a``).
    # ``t`` ranges over [0, L] for centers strictly inside the
    # cylinder (along the corridor direction).
    t = -(dx * a[0] + dy * a[1] + dz * a[2])
    # Perpendicular distance squared to the axis line:
    # perp = d - t * (-a), so |perp|^2 = |d|^2 - t^2.
    d2 = dx * dx + dy * dy + dz * dz
    perp2 = d2 - t * t
    # Numerical floor, since the subtraction above can produce tiny negatives.
    np.clip(perp2, 0.0, None, out=perp2)
    inside = (t >= 0.0) & (t <= L) & (perp2 <= R * R)

    queried = int(np.count_nonzero(inside))
    if queried == 0:
        return CorridorEvidence(
            queried_voxels=0,
            hit_voxels=0,
            seen_voxels=0,
            hit_fraction=0.0,
            seen_fraction=0.0,
            views_accepted=views_accepted,
        )

    sub_hits = hits[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    sub_seen = seen[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
    hit_voxels = int(np.count_nonzero((sub_hits > 0) & inside))
    seen_voxels = int(np.count_nonzero((sub_seen > 0) & inside))

    return CorridorEvidence(
        queried_voxels=queried,
        hit_voxels=hit_voxels,
        seen_voxels=seen_voxels,
        hit_fraction=float(hit_voxels) / float(queried),
        seen_fraction=float(seen_voxels) / float(queried),
        views_accepted=views_accepted,
    )


def viewpoint_information_gain(
    *,
    cfg: "FusionConfig",
    seen: np.ndarray,
    t_cam_to_base: Transform,
    intrinsics: np.ndarray,
    depth_shape: Tuple[int, int],
) -> ViewpointGain:
    """Predicted new-information count (see :meth:`SceneFusion.viewpoint_information_gain`)."""

    if not isinstance(t_cam_to_base, Transform):
        raise TypeError(
            "t_cam_to_base must be a Transform"
        )
    if (
        t_cam_to_base.from_frame is not Frame.CAMERA
        or t_cam_to_base.to_frame is not Frame.BASE
    ):
        raise ValueError(
            "t_cam_to_base must map Frame.CAMERA -> Frame.BASE"
        )
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(
            "intrinsics must be a finite (3, 3) float64 matrix"
        )
    if (
        not isinstance(depth_shape, tuple)
        or len(depth_shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in depth_shape)
    ):
        raise ValueError(
            "depth_shape must be a tuple (height, width) of positive ints"
        )
    H, W = depth_shape

    # ---- index every in-ROI voxel that is currently unseen ---------------
    unseen_mask = seen == 0
    unseen_total = int(np.count_nonzero(unseen_mask))
    if unseen_total == 0:
        return ViewpointGain(
            unseen_voxels=0,
            predicted_visible_unseen=0,
            image_shape=(H, W),
        )

    ix, iy, iz = np.nonzero(unseen_mask)
    ox, oy, oz = cfg.grid_origin_mm
    vs = float(cfg.voxel_size_mm)
    xb = ox + (ix.astype(np.float64) + 0.5) * vs
    yb = oy + (iy.astype(np.float64) + 0.5) * vs
    zb = oz + (iz.astype(np.float64) + 0.5) * vs
    pts_base = np.stack([xb, yb, zb], axis=0)  # (3, N)

    # Transform from BASE to CAMERA through the inverse of t_cam_to_base.
    M = np.asarray(t_cam_to_base.to_matrix(), dtype=np.float64)
    R = M[:3, :3]
    t = M[:3, 3]
    # The inverse of a rigid transform is R^T and -R^T t.
    Rt = R.T
    pts_cam = Rt @ (pts_base - t[:, None])
    z_cam = pts_cam[2]

    depth_min = float(cfg.depth_min_mm)
    depth_max = float(cfg.depth_max_mm)
    in_depth = (z_cam >= depth_min) & (z_cam <= depth_max)
    if not np.any(in_depth):
        return ViewpointGain(
            unseen_voxels=unseen_total,
            predicted_visible_unseen=0,
            image_shape=(H, W),
        )

    # Project into the image plane: u = fx * x/z + cx, v = fy * y/z + cy.
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    # Avoid divide-by-zero by masking first.
    zc = z_cam[in_depth]
    xc = pts_cam[0][in_depth]
    yc = pts_cam[1][in_depth]
    u = fx * xc / zc + cx
    v = fy * yc / zc + cy
    in_bounds = (u >= 0.0) & (u < float(W)) & (v >= 0.0) & (v < float(H))
    predicted = int(np.count_nonzero(in_bounds))

    return ViewpointGain(
        unseen_voxels=unseen_total,
        predicted_visible_unseen=predicted,
        image_shape=(H, W),
    )
