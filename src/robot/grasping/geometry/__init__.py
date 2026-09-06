"""Geometry-first grasping primitives.

This subpackage owns the grasp-specific point-cloud and surface-analysis code. The
general frame-safe rigid poses and transforms stay in :mod:`src.geometry`.
"""

from __future__ import annotations

from .filters import (
    CloudOutlierConfig,
    apply_cloud_outlier_filter,
    filter_by_depth_range,
    radius_outlier_indices,
    statistical_outlier_indices,
)
from ._spatial import RadiusIndex
from ._validation import as_mask_and_depth, as_points_nx3, as_vec3
from .normals import NormalEstimationConfig, SurfaceNormals, estimate_surface_normals
from .pointcloud import (
    CameraIntrinsics,
    MaskedPointCloud,
    depth_unit_to_mm,
    masked_point_cloud,
    masked_points,
)
from .sampling import (
    farthest_point_sample_indices,
    uniform_sample_indices,
    voxel_downsample_indices,
)
from .transforms import validate_transform

__all__ = [
    "CameraIntrinsics",
    "CloudOutlierConfig",
    "MaskedPointCloud",
    "RadiusIndex",
    "apply_cloud_outlier_filter",
    "as_mask_and_depth",
    "as_points_nx3",
    "as_vec3",
    "NormalEstimationConfig",
    "SurfaceNormals",
    "depth_unit_to_mm",
    "estimate_surface_normals",
    "farthest_point_sample_indices",
    "filter_by_depth_range",
    "masked_point_cloud",
    "masked_points",
    "radius_outlier_indices",
    "statistical_outlier_indices",
    "uniform_sample_indices",
    "validate_transform",
    "voxel_downsample_indices",
]
