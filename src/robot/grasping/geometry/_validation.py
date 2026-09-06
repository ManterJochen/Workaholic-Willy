"""The shared input validators for the grasping geometry primitives.

They are one home for the shape and finiteness checks every point-cloud helper would
otherwise re-implement. They are pure NumPy and always raise :class:`ValueError`, never
the ``InvalidPoseError`` of the frame algebra, so the whole ``grasping`` stack keeps one
predictable validation contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["as_mask_and_depth", "as_points_nx3", "as_vec3"]


def as_points_nx3(value: Any, name: str = "points_mm") -> np.ndarray:
    """Coerce ``value`` to a validated, finite ``(N, 3)`` float64 array. An empty cloud is allowed."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be shape (N, 3), got {points.shape}")
    if points.shape[0] > 0 and not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def as_vec3(value: Any, name: str) -> np.ndarray:
    """Coerce ``value`` to a validated, finite, freshly copied ``(3,)`` float64 vector.

    It always copies, so a caller freezes the result on a frozen dataclass without ever
    turning their own array read-only underneath them.
    """
    arr = np.array(value, dtype=np.float64)  # np.array copies, so freezing never reaches the caller
    if arr.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def as_mask_and_depth(mask: Any, depth_map: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce the mask to bool and the depth map to float64, requiring both 2-D and the same shape."""
    mask_arr = np.asarray(mask)
    depth_arr = np.asarray(depth_map)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask_arr.shape}")
    if depth_arr.ndim != 2:
        raise ValueError(f"depth_map must be 2-D, got shape {depth_arr.shape}")
    if mask_arr.shape != depth_arr.shape:
        raise ValueError(
            f"mask and depth_map must have the same shape, got {mask_arr.shape} and {depth_arr.shape}"
        )
    return mask_arr.astype(bool), depth_arr.astype(np.float64, copy=False)
