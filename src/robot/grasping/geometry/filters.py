"""The point-cloud filters that run before contact-pair search.

A stereo or structured-light depth map delivers outliers: speckle near a mask edge, a
dropout in a low-texture region, and, in a pick-and-place cell, the table itself
bleeding into the object cloud. These filters are deliberately cheap and deterministic,
so they run on every grasp inference without spiking latency.

Every function returns an integer index array, the points to keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spatial import RadiusIndex
from ._validation import as_points_nx3

__all__ = [
    "CloudOutlierConfig",
    "apply_cloud_outlier_filter",
    "filter_by_depth_range",
    "radius_outlier_indices",
    "statistical_outlier_indices",
]


@dataclass(frozen=True, slots=True)
class CloudOutlierConfig:
    """Config for the optional point-cloud outlier filter that runs before contact-pair search."""

    # A robustness lever for real hardware, against stereo and RGB-D speckle and
    # edge-bleed, and a no-op on the noise-free Isaac renderer. It is off unless passed
    # as GraspCalculator(cloud_outlier_filter=...).
    method: str = "statistical"  # "statistical" (kNN-distance) | "radius" (neighbour count)
    k_neighbors: int = 16
    std_ratio: float = 2.0
    radius_mm: float = 10.0
    min_neighbors: int = 4

    def __post_init__(self) -> None:
        if self.method not in ("statistical", "radius"):
            raise ValueError(
                f"CloudOutlierConfig.method must be 'statistical' or 'radius'; got {self.method!r}"
            )


def apply_cloud_outlier_filter(
    points_mm: np.ndarray, config: CloudOutlierConfig
) -> np.ndarray:
    """Return the inlier indices, the points to keep, for ``points_mm`` under ``config``.

    It is a thin dispatcher over :func:`statistical_outlier_indices` and
    :func:`radius_outlier_indices`.
    """
    if config.method == "radius":
        return radius_outlier_indices(
            points_mm, radius_mm=config.radius_mm, min_neighbors=config.min_neighbors
        )
    return statistical_outlier_indices(
        points_mm, k_neighbors=config.k_neighbors, std_ratio=config.std_ratio
    )


def filter_by_depth_range(
    points_mm: np.ndarray,
    *,
    min_depth_mm: float = 0.0,
    max_depth_mm: float | None = None,
    axis: int = 2,
) -> np.ndarray:
    """Keep the points whose depth, on ``axis=2`` by default, lies inside the band."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not np.isfinite(min_depth_mm) or min_depth_mm < 0.0:
        raise ValueError("min_depth_mm must be finite and >= 0")
    if max_depth_mm is not None:
        if not np.isfinite(max_depth_mm) or max_depth_mm <= min_depth_mm:
            raise ValueError("max_depth_mm must be finite and > min_depth_mm")
    points = as_points_nx3(points_mm)
    if points.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    depths = points[:, axis]
    keep = depths >= float(min_depth_mm)
    if max_depth_mm is not None:
        keep &= depths <= float(max_depth_mm)
    return np.flatnonzero(keep).astype(np.int64)


def radius_outlier_indices(
    points_mm: np.ndarray,
    *,
    radius_mm: float,
    min_neighbors: int,
) -> np.ndarray:
    """Return the inlier indices: those with at least ``min_neighbors`` neighbours within
    ``radius_mm``.

    It is the equivalent of the Open3D ``remove_radius_outlier``, kept dependency-free so
    the filter runs inside the inference path with no extras installed.
    """
    if not np.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("radius_mm must be finite and > 0")
    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")
    points = as_points_nx3(points_mm)
    n_points = points.shape[0]
    if n_points == 0:
        return np.empty((0,), dtype=np.int64)
    index = RadiusIndex(points)
    keep = np.zeros(n_points, dtype=bool)
    for i, point in enumerate(points):
        neighbours = index.query_radius(point, float(radius_mm))
        # A KD-tree radius query always returns the point itself.
        if neighbours.size - 1 >= min_neighbors:
            keep[i] = True
    return np.flatnonzero(keep).astype(np.int64)


def statistical_outlier_indices(
    points_mm: np.ndarray,
    *,
    k_neighbors: int = 16,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Return the indices whose mean kNN distance is within ``std_ratio`` standard
    deviations of the global mean.

    It mirrors the semantics of the Open3D ``remove_statistical_outlier``, and it is
    robust to isolated speckle without needing a hard radius parameter.
    """
    if k_neighbors < 2:
        raise ValueError("k_neighbors must be >= 2")
    if not np.isfinite(std_ratio) or std_ratio < 0.0:
        raise ValueError("std_ratio must be finite and >= 0")
    points = as_points_nx3(points_mm)
    n_points = points.shape[0]
    if n_points == 0:
        return np.empty((0,), dtype=np.int64)
    if n_points <= k_neighbors:
        return np.arange(n_points, dtype=np.int64)
    index = RadiusIndex(points)
    mean_distances = np.zeros(n_points, dtype=np.float64)
    for i, point in enumerate(points):
        neighbours = index.query_knn(point, k_neighbors + 1)
        # Drop the point itself from its own kNN result.
        neighbours = neighbours[neighbours != i]
        if neighbours.size == 0:
            mean_distances[i] = np.inf
            continue
        deltas = points[neighbours] - point
        distances = np.linalg.norm(deltas, axis=1)
        mean_distances[i] = float(np.mean(distances))
    finite = np.isfinite(mean_distances)
    if not np.any(finite):
        return np.empty((0,), dtype=np.int64)
    mu = float(np.mean(mean_distances[finite]))
    sigma = float(np.std(mean_distances[finite]))
    threshold = mu + float(std_ratio) * sigma
    keep = finite & (mean_distances <= threshold)
    return np.flatnonzero(keep).astype(np.int64)
