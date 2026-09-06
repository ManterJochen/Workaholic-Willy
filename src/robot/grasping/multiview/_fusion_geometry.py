"""Multi-view-fusion backprojection geometry: depth to unique voxels, and nothing else.

This is the ``SceneFusion.ingest`` hot loop. It takes the grid and depth scalars explicitly, with no
``FusionConfig`` dependency, so it stays a clean numpy and geometry leaf with no import cycle against
the fusion carriers. The arithmetic is behaviour-locked: for a given input the hits and seen counters
it feeds must stay bit-stable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.geometry import Transform


def backproject_to_unique_voxels(
    depth: np.ndarray,
    K: np.ndarray,
    t_cam_to_base: Transform,
    *,
    grid_origin_mm: Tuple[float, float, float],
    voxel_size_mm: float,
    shape: Tuple[int, int, int],
    depth_min_mm: float,
    depth_max_mm: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return aggregated ``(flat_idx, counts, n_valid)`` for one view.

    ``flat_idx`` is a 1-D int64 array of unique flat voxel indices,
    ``i*Ny*Nz + j*Nz + k``, and ``counts`` is the parallel per-voxel
    hit count as uint32. ``n_valid`` is the number of in-range, in-ROI
    depth pixels that contributed.
    """

    h, w = depth.shape
    # Canonical (v, u) raster order. ``indexing="ij"`` keeps the
    # memory layout matching the depth raster.
    v_idx, u_idx = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64),
        indexing="ij",
    )

    valid = (
        np.isfinite(depth)
        & (depth >= depth_min_mm)
        & (depth <= depth_max_mm)
    )
    if not np.any(valid):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint32),
            0,
        )

    d_v = depth[valid]
    u_v = u_idx[valid]
    v_v = v_idx[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x_cam = (u_v - cx) * d_v / fx
    y_cam = (v_v - cy) * d_v / fy
    z_cam = d_v

    M = np.asarray(t_cam_to_base.to_matrix(), dtype=np.float64)
    R = M[:3, :3]
    t = M[:3, 3]
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=0)  # (3, N)
    pts_base = R @ pts_cam + t[:, None]
    xb = pts_base[0]
    yb = pts_base[1]
    zb = pts_base[2]

    ox, oy, oz = grid_origin_mm
    vs = float(voxel_size_mm)
    nx, ny, nz = shape

    ix = np.floor((xb - ox) / vs).astype(np.int64)
    iy = np.floor((yb - oy) / vs).astype(np.int64)
    iz = np.floor((zb - oz) / vs).astype(np.int64)

    in_roi = (
        (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )
    n_in_roi = int(np.count_nonzero(in_roi))
    if n_in_roi == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint32),
            0,
        )

    ix = ix[in_roi]
    iy = iy[in_roi]
    iz = iz[in_roi]

    # Flat row-major index in (x, y, z) order.
    flat = (ix * (ny * nz) + iy * nz + iz).astype(np.int64, copy=False)

    # Order-stable aggregation via numpy.unique.
    unique_idx, counts = np.unique(flat, return_counts=True)
    return (
        unique_idx.astype(np.int64, copy=False),
        counts.astype(np.uint32, copy=False),
        n_in_roi,
    )
