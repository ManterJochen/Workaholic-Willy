"""The container walls, as collision geometry the candidate filter can see.

Nothing else in the grasping stack carries wall geometry. The candidate filter checks a target
cloud, the neighbours' clouds and a support plane, and a plane is a floor, so a grasp whose fingers
swing into the side of a KLT is rejected by nothing upstream of the motion planner and the ranker
puts it first.

The cost is measured. Replaying the datagen reference verdict on the rank-0 winners that lose the
ranking gap on ``finger_collision`` finds 60.9 % of them colliding with a bin wall rather than with
an object, and inside the ``bin`` family it is 14 of 14. That is the largest named item left in the
bin family, and no perception change can reach it: a wall is not an instance any detector segments,
so it can never enter a fused multi-camera cloud through
``fusion.geometry.neighbour_scene_enabled``, which is why that switch moves the ``pile`` gap and not
the ``bin`` one. A wall has to be declared, and this module turns the declaration into points.

Points rather than a primitive, because both consumers downstream already take points:
``validate_grasp_collision(scene_points_mm=...)`` in the CAMERA frame and
``generate_support_footprint_grasps(rigid_obstacle_points_base_mm=...)`` in BASE. Emitting points
means the walls reach every check that already exists, with nothing new to keep in sync.

What declaring the walls buys is not the pick rate. Over the 354 jaw-graspable objects of the D5
reference, the denominator this project tracks, top-1 goes from 55.9 % to 56.2 % with walls and to
55.6 % with walls plus fused neighbours, which is one object either way on an evaluation that is not
bit reproducible. What moves is candidate precision, from 64.6 % to 84.6 %, and the ``bin`` family's
ranking gap, from 15 objects to 0. The walls do not find more grasps, they stop the ranker offering
grasps that drive a finger into the side of the bin, which turns a wrong pick into an honest
no-pick. In simulation that trade is neutral. On real hardware it is the difference between a retry
and a collision, so a cell with a bin declares its walls even though the rate will not thank it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["container_wall_points_base_mm"]


def _axis_samples(low: float, high: float, step: float) -> np.ndarray:
    """Inclusive samples from ``low`` to ``high``, never fewer than the two endpoints."""

    span = high - low
    count = max(int(np.ceil(span / step)), 1) + 1
    return np.linspace(low, high, count, dtype=np.float64)


def container_wall_points_base_mm(
    interior_min_mm: Sequence[float],
    interior_max_mm: Sequence[float],
    *,
    thickness_mm: float = 5.0,
    sample_mm: float = 8.0,
) -> np.ndarray:
    """The four vertical walls around an interior box, as ``(N, 3)`` BASE-frame points.

    The interior box is the empty volume parts sit in: ``interior_min_mm[2]`` is the inside floor and
    ``interior_max_mm[2]`` is the rim. Walls are sampled as a shell of ``thickness_mm`` outward from
    each interior face, between floor and rim.

    The sampling goes outward and through the thickness on purpose. Sampling the inner faces alone
    would leave a finger standing in the middle of the wall material undetected unless the collision
    margin happened to exceed half the thickness, which is a coincidence rather than a check. Sampling
    the shell makes the obstacle the wall rather than the surface of the wall.

    The floor is not included: that is the support plane's job, and it is checked analytically with a
    clearance these points would only approximate. Nothing above the rim is included either, so a
    finger may legally pass over the wall from outside, which is a real grasp on a shallow tray and
    blocking it would trade one wrong answer for another.
    """

    low = np.asarray(interior_min_mm, dtype=np.float64).reshape(3)
    high = np.asarray(interior_max_mm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("container interior corners must be finite")
    if np.any(high <= low):
        raise ValueError(
            f"container interior_max_mm {high.tolist()} must exceed interior_min_mm {low.tolist()} "
            "on every axis"
        )
    if thickness_mm <= 0.0 or sample_mm <= 0.0:
        raise ValueError("thickness_mm and sample_mm must be > 0")

    z = _axis_samples(low[2], high[2], sample_mm)
    depth = _axis_samples(0.0, thickness_mm, sample_mm)
    parts: list[np.ndarray] = []

    # Two walls per horizontal axis. For axis 0 the wall spans y and z and is displaced in x, for
    # axis 1 it spans x and z. Corners are sampled by both axes and land as duplicate points, which
    # costs a few hundred rows that a proximity query does not care about.
    for axis in (0, 1):
        span_axis = 1 - axis
        span = _axis_samples(low[span_axis] - thickness_mm, high[span_axis] + thickness_mm, sample_mm)
        grid_span, grid_z, grid_depth = np.meshgrid(span, z, depth, indexing="ij")
        flat_span = grid_span.reshape(-1)
        flat_z = grid_z.reshape(-1)
        flat_depth = grid_depth.reshape(-1)
        for sign, face in ((-1.0, low[axis]), (1.0, high[axis])):
            points = np.empty((flat_span.size, 3), dtype=np.float64)
            points[:, axis] = face + sign * flat_depth
            points[:, span_axis] = flat_span
            points[:, 2] = flat_z
            parts.append(points)

    return np.vstack(parts)
