"""Masked point-cloud generation for geometry-first grasping.

This module is deliberately lightweight: NumPy alone, deterministic, with no Open3D
dependency. Open3D can be added later for visualisation or faster large-cloud
processing, and the core projection contract stays small.

Coordinate convention
---------------------
The input intrinsics use the usual pinhole camera model, with pixel coordinates ``u``
for the column and ``v`` for the row::

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth

Every output coordinate is in millimetres in the camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._validation import as_mask_and_depth
from .sampling import voxel_downsample_indices

DepthUnit = Literal["mm", "cm", "m"]

_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

__all__ = [
    "CameraIntrinsics",
    "MaskedPointCloud",
    "depth_unit_to_mm",
    "masked_point_cloud",
    "masked_points",
]


def depth_unit_to_mm(unit: str) -> float:
    """The multiplier that converts ``unit`` to millimetres."""
    try:
        return _UNIT_TO_MM[(unit or "mm").lower()]
    except (KeyError, AttributeError) as exc:  # AttributeError: a non-str unit has no .lower()
        valid = ", ".join(_UNIT_TO_MM)
        raise ValueError(f"unknown depth unit {unit!r}; expected one of: {valid}") from exc


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """The pinhole camera intrinsics used for depth back-projection."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"CameraIntrinsics.{name} must be finite")
            object.__setattr__(self, name, value)
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("CameraIntrinsics.fx and fy must be > 0")

    @classmethod
    def from_matrix(cls, K: np.ndarray) -> CameraIntrinsics:
        """Build intrinsics from a normalised, skewless ``3x3`` pinhole camera matrix.

        It rejects a denormalised matrix, where ``K[2,2]`` is not 1, which would rescale
        every back-projected point silently, and a non-zero skew term ``K[0,1]``, rather
        than reading the four focal and principal entries and ignoring the rest.
        """
        arr = np.asarray(K, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"camera matrix must be shape (3, 3), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("camera matrix must contain only finite values")
        if abs(float(arr[2, 2]) - 1.0) > 1e-6:
            raise ValueError(f"camera matrix must be normalised (K[2,2] == 1), got K[2,2]={arr[2, 2]}")
        if abs(float(arr[0, 1])) > 1e-6:
            raise ValueError(f"camera matrix skew K[0,1] must be 0, got {arr[0, 1]}")
        return cls(fx=arr[0, 0], fy=arr[1, 1], cx=arr[0, 2], cy=arr[1, 2])

    def to_matrix(self) -> np.ndarray:
        """Return a fresh ``3x3`` camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class MaskedPointCloud:
    """The back-projected masked depth pixels.

    Attributes
    ----------
    points_mm
        ``(N, 3)`` float32 camera-frame points, in millimetres.
    pixels_yx
        ``(N, 2)`` int32 image coordinates, as ``(row, col)``.
    depths_mm
        ``(N,)`` float32 depths used for the projection.
    intrinsics
        The camera intrinsics used for the projection.
    """

    points_mm: np.ndarray
    pixels_yx: np.ndarray
    depths_mm: np.ndarray
    intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=np.float32)
        pixels = np.asarray(self.pixels_yx, dtype=np.int32)
        depths = np.asarray(self.depths_mm, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_mm must be shape (N, 3), got {points.shape}")
        if pixels.shape != (points.shape[0], 2):
            raise ValueError(
                f"pixels_yx must be shape ({points.shape[0]}, 2), got {pixels.shape}"
            )
        if depths.shape != (points.shape[0],):
            raise ValueError(
                f"depths_mm must be shape ({points.shape[0]},), got {depths.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points_mm must contain only finite values")
        if not np.all(np.isfinite(depths)):
            raise ValueError("depths_mm must contain only finite values")
        points.setflags(write=False)
        pixels.setflags(write=False)
        depths.setflags(write=False)
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "pixels_yx", pixels)
        object.__setattr__(self, "depths_mm", depths)

    @property
    def size(self) -> int:
        return int(self.points_mm.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.size == 0


def masked_point_cloud(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    *,
    unit: DepthUnit | str = "mm",
    min_depth_mm: float = 1.0,
    max_depth_mm: float | None = None,
    voxel_size_mm: float | None = None,
) -> MaskedPointCloud:
    """Back-project masked depth pixels into a camera-frame point cloud.

    An invalid depth, meaning a NaN, an infinity, a zero or a negative, is discarded
    before any optional band filtering or voxel downsampling.
    """
    mask_bool, depth = as_mask_and_depth(mask, depth_map)
    K = intrinsics if isinstance(intrinsics, CameraIntrinsics) else CameraIntrinsics.from_matrix(intrinsics)

    if min_depth_mm < 0.0 or not np.isfinite(min_depth_mm):
        raise ValueError("min_depth_mm must be finite and >= 0")
    if max_depth_mm is not None:
        if not np.isfinite(max_depth_mm) or max_depth_mm <= min_depth_mm:
            raise ValueError("max_depth_mm must be finite and > min_depth_mm")

    depth_mm = depth * depth_unit_to_mm(unit)
    valid = mask_bool & np.isfinite(depth_mm) & (depth_mm > 0.0) & (depth_mm >= min_depth_mm)
    if max_depth_mm is not None:
        valid &= depth_mm <= max_depth_mm

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return MaskedPointCloud(
            points_mm=np.empty((0, 3), dtype=np.float32),
            pixels_yx=np.empty((0, 2), dtype=np.int32),
            depths_mm=np.empty((0,), dtype=np.float32),
            intrinsics=K,
        )

    z = depth_mm[rows, cols]
    x = (cols.astype(np.float64) - K.cx) * z / K.fx
    y = (rows.astype(np.float64) - K.cy) * z / K.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    pixels = np.column_stack((rows, cols)).astype(np.int32)
    depths = z.astype(np.float32)

    if voxel_size_mm is not None:
        keep = voxel_downsample_indices(points.astype(np.float64), float(voxel_size_mm))
        points = points[keep]
        pixels = pixels[keep]
        depths = depths[keep]

    return MaskedPointCloud(points_mm=points, pixels_yx=pixels, depths_mm=depths, intrinsics=K)


def masked_points(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    **kwargs,
) -> np.ndarray:
    """A wrapper that returns the ``(N, 3)`` float32 points alone."""
    return masked_point_cloud(mask, depth_map, intrinsics, **kwargs).points_mm
