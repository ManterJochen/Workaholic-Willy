"""The scene a grasp keeps clear of: the other masks of a view as one collision cloud.

:func:`scene_collision_cloud` aggregates the other masks into one Nx3 collision cloud, so a neighbour
object is treated as scene. It consumes already-decoded NumPy masks and never imports a segmentation
model. The dense surface sampler beside it left on 2026-10-18 with the silhouette path it fed.
"""

from __future__ import annotations


import numpy as np

from src.robot.grasping.geometry import (
    CameraIntrinsics,
    masked_point_cloud,
)

__all__ = [
    "scene_collision_cloud",
]


def scene_collision_cloud(
    object_masks: list[np.ndarray],
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    *,
    target_index: int | None = None,
    extra_scene_mask: np.ndarray | None = None,
    unit: str = "mm",
    min_depth_mm: float = 1.0,
    max_depth_mm: float | None = None,
    voxel_size_mm: float | None = 4.0,
) -> np.ndarray:
    """An Nx3 camera-frame collision cloud in mm, of every object but ``target_index``.

    Any ``extra_scene_mask`` is included as well.
    """
    if not object_masks and extra_scene_mask is None:
        return np.empty((0, 3), dtype=np.float64)

    combined = (
        np.zeros_like(depth_map, dtype=bool)
        if not object_masks
        else np.zeros(object_masks[0].shape, dtype=bool)
    )
    for index, mask in enumerate(object_masks):
        if index == target_index:
            continue
        combined |= np.asarray(mask).astype(bool)
    if extra_scene_mask is not None:
        combined |= np.asarray(extra_scene_mask).astype(bool)
    if not np.any(combined):
        return np.empty((0, 3), dtype=np.float64)

    cloud = masked_point_cloud(
        combined,
        depth_map,
        intrinsics,
        unit=unit,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
        voxel_size_mm=voxel_size_mm,
    )
    return np.asarray(cloud.points_mm, dtype=np.float64)


