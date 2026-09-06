"""3D point-cloud grasp synthesis from a fused multi-view BASE cloud.

Single-view grasp synthesis, in :class:`GraspCalculator`, works on one camera's image-space silhouette.
That silhouette is foreshortened, because an oblique view compresses the depth axis, and it shows only
the camera-facing surface, so the footprint it yields, and with it the grasp width and closing axis, is
biased. This module fuses every camera's back-projected target points into one BASE-frame cloud, as
localize and neighbour fusion already do for centroids and neighbours, and synthesizes the grasp from
that 3D geometry:

* the closing axis and the grip width come from the object's horizontal footprint over all views, taken
  as PCA on the BASE XY projection and closing across the minor, narrow extent, rather than from a
  single 2D projection;
* the position is the fused centroid at a grasp depth below the top surface of the cloud, where
  symmetric views cancel the surface bias of each single view.

This is the deepest multi-view lever, in that grasp synthesis consumes the geometry of all cameras
rather than localization or collision alone. It is pure numpy, with no Isaac and no torch; the runner
converts the result to a BASE :class:`GraspPoint` and drives it through the validated executor.

Numerics: millimetres, BASE frame. The approach defaults to top-down, along base ``-Z``, which is the
reachable axis the sim and UR5e executor grasp table-supported objects along; a genuinely tilted planned
approach, into a bin wall for instance, is a future extension. Returns ``None`` when the cloud is too
small to fit a footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.types.grasp_point import GraspPoint

__all__ = ["FusedGrasp", "synthesize_grasp_from_cloud", "fused_grasp_to_point"]

# A footprint needs enough points for a stable covariance; below this PCA is noise.
_MIN_POINTS = 8


@dataclass(frozen=True, slots=True)
class FusedGrasp:
    """A top-down parallel-jaw grasp synthesized from a fused 3D BASE point cloud.

    ``closing_axis`` is the horizontal minor, or narrow, footprint axis the jaws close along.
    ``footprint_mm`` is ``(minor_extent, major_extent)``, a diagnostic for how foreshortened or how
    complete the fused cloud is.
    """

    position_mm: np.ndarray
    closing_axis: np.ndarray
    approach: np.ndarray
    width_mm: float
    footprint_mm: tuple[float, float]
    n_points: int


def _xy_principal_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of a 2D point set, returning ``(centroid, minor_axis_xy, major_axis_xy)`` as unit vectors in ascending order of variance."""
    centroid = xy.mean(axis=0)
    centred = xy - centroid
    cov = np.cov(centred.T)
    if not np.all(np.isfinite(cov)):
        # Degenerate, all identical points for instance, so fall back to the canonical axes.
        return centroid, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))  # ascending eigenvalues
    minor = np.asarray(eigvecs[:, 0], dtype=np.float64)
    major = np.asarray(eigvecs[:, 1], dtype=np.float64)
    return centroid, minor, major


def synthesize_grasp_from_cloud(
    points_base_mm: np.ndarray,
    *,
    grasp_depth_mm: float = 8.0,
    min_width_mm: float = 0.0,
    max_width_mm: float = 85.0,
    width_margin_mm: float = 6.0,
) -> FusedGrasp | None:
    """Synthesize a top-down min-width antipodal grasp from a fused BASE point cloud (mm).

    Projects the cloud to the BASE XY plane, runs PCA for the horizontal footprint, and closes the jaws
    across the minor, narrow axis. The width is the minor-axis extent plus ``width_margin_mm``, clipped
    to ``[min_width_mm, max_width_mm]``, and the anchor is the footprint centroid ``grasp_depth_mm``
    below the top surface of the cloud. Returns ``None`` if the cloud has fewer than 8 finite points.
    """
    pts = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < _MIN_POINTS:
        return None

    xy = pts[:, :2]
    centroid_xy, minor_xy, major_xy = _xy_principal_axes(xy)
    centred = xy - centroid_xy
    proj_minor = centred @ minor_xy
    proj_major = centred @ major_xy
    minor_extent = float(proj_minor.max() - proj_minor.min())
    major_extent = float(proj_major.max() - proj_major.min())

    closing = np.array([minor_xy[0], minor_xy[1], 0.0], dtype=np.float64)
    cnorm = float(np.linalg.norm(closing))
    closing = closing / cnorm if cnorm > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    top_z = float(pts[:, 2].max())
    position = np.array([centroid_xy[0], centroid_xy[1], top_z - float(grasp_depth_mm)], dtype=np.float64)
    width = float(np.clip(minor_extent + float(width_margin_mm), float(min_width_mm), float(max_width_mm)))

    return FusedGrasp(
        position_mm=position,
        closing_axis=closing,
        approach=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        width_mm=width,
        footprint_mm=(minor_extent, major_extent),
        n_points=int(pts.shape[0]),
    )


def fused_grasp_to_point(grasp: FusedGrasp, *, score: float = 1.0, label: str = "fused3d") -> "GraspPoint":
    """Convert a :class:`FusedGrasp` to a BASE :class:`GraspPoint` for the execution policy.

    ``grasp_point`` is imported lazily so this module stays a pure-numpy leaf at import time.
    """
    from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

    return GraspPoint(
        position=np.asarray(grasp.position_mm, dtype=np.float64),
        approach=np.asarray(grasp.approach, dtype=np.float64),
        axis=np.asarray(grasp.closing_axis, dtype=np.float64),
        grip_width_mm=float(grasp.width_mm),
        score=float(score),
        frame=GraspFrame.BASE,
        label=label,
    )
