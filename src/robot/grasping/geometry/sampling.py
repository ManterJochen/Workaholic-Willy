"""Deterministic point-cloud sampling helpers.

Every one of them returns an integer index array into the input cloud. A caller slices
its own attribute arrays, points, normals and colours, with the same index, which keeps
every per-point side-channel in sync and avoids a hidden allocation.
"""

from __future__ import annotations

import numpy as np

from ._validation import as_points_nx3

__all__ = [
    "voxel_downsample_indices",
]


def voxel_downsample_indices(
    points_mm: np.ndarray,
    voxel_size_mm: float,
) -> np.ndarray:
    """Return one index per occupied voxel, the first encountered.

    It never randomises a tie-break: a voxel and its representative are picked in input
    order, which is what makes the downstream contact-pair search reproducible.
    """
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be finite and > 0")
    points = as_points_nx3(points_mm)
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    voxels = np.floor(points / float(voxel_size_mm)).astype(np.int64)
    _, first_indices = np.unique(voxels, axis=0, return_index=True)
    return np.sort(first_indices.astype(np.int64))


