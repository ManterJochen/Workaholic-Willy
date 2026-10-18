"""Geometry-first grasping primitives.

This subpackage owns the grasp-specific point-cloud and surface-analysis code. The
general frame-safe rigid poses and transforms stay in :mod:`src.geometry`.
"""

from __future__ import annotations

from ._spatial import RadiusIndex
from ._validation import as_mask_and_depth, as_points_nx3, as_vec3
from .grasp_frame import pose_from_grasp_axes
from .normals import NormalEstimationConfig, SurfaceNormals, estimate_surface_normals
from .pointcloud import (
    CameraIntrinsics,
    MaskedPointCloud,
    depth_unit_to_mm,
    masked_point_cloud,
    masked_points,
)
from .sampling import voxel_downsample_indices
from .transforms import validate_transform

__all__ = [
    "CameraIntrinsics",
    "MaskedPointCloud",
    "RadiusIndex",
    "as_mask_and_depth",
    "as_points_nx3",
    "as_vec3",
    "NormalEstimationConfig",
    "SurfaceNormals",
    "depth_unit_to_mm",
    "estimate_surface_normals",
    "masked_point_cloud",
    "masked_points",
    "pose_from_grasp_axes",
    "validate_transform",
    "voxel_downsample_indices",
]
