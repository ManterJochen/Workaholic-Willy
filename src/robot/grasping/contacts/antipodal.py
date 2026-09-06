"""The deterministic antipodal contact-pair search.

It is geometry-only and deliberately conservative: it finds pairs of surface points
that look physically plausible for a parallel-jaw gripper, before collision and
reachability validation. They are not executable grasps.
"""

from __future__ import annotations

import numpy as np

from src.robot.grasping.geometry import RadiusIndex, SurfaceNormals, as_points_nx3

from .contact_pair import ContactPair

__all__ = ["find_antipodal_pairs"]


def _normal_inputs(
    normals: SurfaceNormals | np.ndarray,
    n_points: int,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    confidence: np.ndarray | None = None
    curvature: np.ndarray | None = None
    if isinstance(normals, SurfaceNormals):
        normal_arr = np.asarray(normals.normals, dtype=np.float64)
        valid = np.asarray(normals.valid_mask, dtype=bool).copy()
        confidence = np.asarray(normals.confidence, dtype=np.float64)
        curvature = np.asarray(normals.curvature, dtype=np.float64)
    else:
        normal_arr = np.asarray(normals, dtype=np.float64)
        valid = np.ones((n_points,), dtype=bool)
    if normal_arr.shape != (n_points, 3):
        raise ValueError(f"normals must be shape ({n_points}, 3), got {normal_arr.shape}")
    if not np.all(np.isfinite(normal_arr)):
        raise ValueError("normals must contain only finite values")
    lengths = np.linalg.norm(normal_arr, axis=1)
    valid &= lengths > 1e-9
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (n_points,):
            raise ValueError(f"valid_mask must be shape ({n_points},), got {mask.shape}")
        valid &= mask
    normal_arr = normal_arr.copy()
    normal_arr[valid] /= lengths[valid, None]
    return normal_arr, valid, confidence, curvature


def _stability(
    i: int,
    j: int,
    confidence: np.ndarray | None,
    curvature: np.ndarray | None,
) -> float:
    """Per-pair stability in ``[0, 1]``, from local normal confidence times surface flatness.

    Only a ``SurfaceNormals`` input carries confidence and curvature. Where
    ``find_antipodal_pairs`` is given a raw ``(N, 3)`` normals array both are ``None``,
    so this collapses to a constant 1.0 and the stability term stops discriminating
    between pairs.
    """
    if confidence is not None:
        conf = float(np.clip(0.5 * (confidence[i] + confidence[j]), 0.0, 1.0))
    else:
        conf = 1.0
    if curvature is not None and np.isfinite(curvature[i]) and np.isfinite(curvature[j]):
        curv = float(0.5 * (curvature[i] + curvature[j]))
        curv_factor = float(np.clip(1.0 - 10.0 * curv, 0.0, 1.0))
    else:
        curv_factor = 1.0
    return float(np.clip(conf * curv_factor, 0.0, 1.0))


def find_antipodal_pairs(
    points_mm: np.ndarray,
    normals: SurfaceNormals | np.ndarray,
    *,
    min_width_mm: float = 5.0,
    max_width_mm: float = 150.0,
    normal_opposition_threshold: float = 0.8,
    axis_alignment_threshold: float = 0.5,
    max_pairs: int = 100,
    valid_mask: np.ndarray | None = None,
) -> list[ContactPair]:
    """Find the valid antipodal contact pairs for a parallel-jaw gripper.

    The alignment between axis and normal is orientation-agnostic, because an upstream
    normal can be camera-facing rather than reliably outward.
    """
    points = as_points_nx3(points_mm)
    n_points = points.shape[0]
    normal_arr, valid, confidence, curvature = _normal_inputs(normals, n_points, valid_mask)

    if min_width_mm < 0.0 or max_width_mm <= min_width_mm:
        raise ValueError("width limits must satisfy 0 <= min_width_mm < max_width_mm")
    if not 0.0 <= normal_opposition_threshold <= 1.0:
        raise ValueError("normal_opposition_threshold must be in [0, 1]")
    if not 0.0 <= axis_alignment_threshold <= 1.0:
        raise ValueError("axis_alignment_threshold must be in [0, 1]")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return []

    index = RadiusIndex(points)
    pairs: list[ContactPair] = []
    for i in valid_indices:
        neighbours = index.query_radius(points[i], float(max_width_mm))
        neighbours = neighbours[(neighbours > i) & valid[neighbours]]
        if neighbours.size == 0:
            continue
        deltas = points[neighbours] - points[i]
        distances = np.linalg.norm(deltas, axis=1)
        in_width = (distances >= min_width_mm) & (distances <= max_width_mm)
        for j, distance, delta in zip(neighbours[in_width], distances[in_width], deltas[in_width]):
            if distance <= 1e-12:
                continue
            axis = delta / distance
            n_a = normal_arr[i]
            n_b = normal_arr[j]
            opposition = float(np.clip(np.dot(n_a, -n_b), 0.0, 1.0))
            if opposition < normal_opposition_threshold:
                continue
            axis_alignment = float(
                np.clip(
                    0.5 * (abs(np.dot(n_a, axis)) + abs(np.dot(n_b, -axis))),
                    0.0,
                    1.0,
                )
            )
            if axis_alignment < axis_alignment_threshold:
                continue
            stability = _stability(int(i), int(j), confidence, curvature)
            score = float(np.clip(0.45 * opposition + 0.35 * axis_alignment + 0.20 * stability, 0.0, 1.0))
            pairs.append(
                ContactPair(
                    point_a=points[i],
                    point_b=points[j],
                    normal_a=n_a,
                    normal_b=n_b,
                    distance_mm=float(distance),
                    antipodal_score=score,
                    axis_alignment=axis_alignment,
                    normal_opposition=opposition,
                    metadata={
                        "index_a": int(i),
                        "index_b": int(j),
                        "stability": stability,
                    },
                )
            )

    pairs.sort(
        key=lambda pair: (
            -pair.antipodal_score,
            pair.distance_mm,
            pair.metadata.get("index_a", -1),
            pair.metadata.get("index_b", -1),
        )
    )
    return pairs[:max_pairs]
