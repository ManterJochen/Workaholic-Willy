"""Directional corridor analysis for grasp candidates.

The occlusion path in :mod:`.occlusion` gives the calculator a scalar
hull-overlap ratio plus a single scalar approach-clearance. That is
enough to report honestly whether a scene is cluttered, but not where
the clutter sits relative to the approach and retreat corridor of the
gripper, so a candidate pointing straight into a wall and a candidate
with an inch of side clearance look identical to the ranker.

This directional, corridor-aware analyzer answers the second question:

* a cylinder of radius ``radius_mm`` is swept along the approach
  axis (and optionally the retreat axis);
* a small number of perimeter rays plus the centerline ray are
  ray-marched against the depth map, mirroring
  :func:`approach_clearance_mm` but reporting the most pessimistic
  ray clearance per direction;
* optionally, the projected corridor is fused with neighbour-object
  masks to catch cases where the depth map is correct but the
  segmentation already knows the corridor will run through another
  detected object;
* the analyzer returns a typed :class:`CorridorReport` (clearance
  per direction, fused blockage confidence in ``[0, 1]``, and a
  stable mode enum).

The module is pure: it depends on neither the calculator nor grasp
pose objects, it is deterministic for fixed inputs, and it never
mutates the supplied arrays. The calculator wires it through behind
feature flags, and the analyzer itself never rejects a candidate. The
optional hard-reject path belongs to the calculator and is gated by
``hard_reject_enabled``, which is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.robot.grasping.geometry.pointcloud import CameraIntrinsics

__all__ = [
    "CorridorAnalysisConfig",
    "CorridorAnalysisInputs",
    "CorridorMode",
    "CorridorReport",
    "analyze_corridor",
]


class CorridorMode(str, Enum):
    """Stable mode strings persisted in telemetry; do not rename."""

    CLEAR = "clear"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


def _validate_finite_nonneg(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class CorridorReport:
    """Typed, JSON-safe output of the directional corridor analyzer; ``mode`` and ``blockage_confidence`` always agree, and ``blocked`` is derived as ``mode == BLOCKED``."""

    approach_clearance_mm: float
    retreat_clearance_mm: float
    blockage_confidence: float
    mode: CorridorMode
    #: False when no ``retreat_axis_cam`` was supplied: no retreat corridor was measured, it
    #: contributed nothing to ``blockage_confidence``, and ``retreat_clearance_mm`` is a
    #: placeholder 0.0 rather than a distance. A consumer must read this flag, because a corridor
    #: that was not measured and one that is blocked immediately are indistinguishable in the
    #: distance alone, and conflating them makes every candidate read as blocked.
    retreat_measured: bool = True

    def __post_init__(self) -> None:
        _validate_finite_nonneg(
            float(self.approach_clearance_mm), "approach_clearance_mm"
        )
        _validate_finite_nonneg(
            float(self.retreat_clearance_mm), "retreat_clearance_mm"
        )
        _validate_unit_interval(
            float(self.blockage_confidence), "blockage_confidence"
        )
        if not isinstance(self.mode, CorridorMode):
            raise TypeError(
                f"CorridorReport.mode must be CorridorMode, got {type(self.mode).__name__}"
            )

    @property
    def blocked(self) -> bool:
        """Derived: True iff ``mode == CorridorMode.BLOCKED``."""
        return self.mode == CorridorMode.BLOCKED


@dataclass(frozen=True, slots=True)
class CorridorAnalysisConfig:
    """Operator knobs for the directional analyzer; defaults mirror the YAML schema."""

    radius_mm: float = 20.0
    step_mm: float = 5.0
    max_distance_mm: float = 200.0
    mask_fusion_weight: float = 0.5
    partial_confidence_threshold: float = 0.3
    blocked_confidence_threshold: float = 0.7
    clearance_tolerance_mm: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "radius_mm",
            "step_mm",
            "max_distance_mm",
            "clearance_tolerance_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        _validate_unit_interval(
            float(self.mask_fusion_weight), "mask_fusion_weight"
        )
        _validate_unit_interval(
            float(self.partial_confidence_threshold),
            "partial_confidence_threshold",
        )
        _validate_unit_interval(
            float(self.blocked_confidence_threshold),
            "blocked_confidence_threshold",
        )
        if (
            self.partial_confidence_threshold
            >= self.blocked_confidence_threshold
        ):
            raise ValueError(
                "partial_confidence_threshold must be strictly less than "
                "blocked_confidence_threshold; got "
                f"partial={self.partial_confidence_threshold!r}, "
                f"blocked={self.blocked_confidence_threshold!r}"
            )


@dataclass(frozen=True, slots=True)
class CorridorAnalysisInputs:
    """Per-candidate corridor analysis inputs: depth + intrinsics, plus optional neighbour masks that fuse with the depth signal via ``mask_fusion_weight``; there is no point-cloud field, so callers pre-render point-cloud clearance into the depth map."""

    target_point_cam_mm: np.ndarray
    approach_axis_cam: np.ndarray
    depth_map: Optional[np.ndarray]
    intrinsics: Optional[CameraIntrinsics]
    retreat_axis_cam: Optional[np.ndarray] = None
    other_object_masks: Optional[tuple[np.ndarray, ...]] = None
    depth_scale_to_mm: float = 1.0


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        raise ValueError("axis vector must be non-zero")
    return arr / n


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors perpendicular to ``axis``."""
    # Pick a seed vector that is not parallel to the axis.
    seed = (
        np.array([1.0, 0.0, 0.0])
        if abs(axis[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(axis, seed)
    u /= float(np.linalg.norm(u))
    v = np.cross(axis, u)
    v /= float(np.linalg.norm(v))
    return u, v


def _ray_clearance_mm(
    *,
    start: np.ndarray,
    march_direction: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_distance_mm: float,
    step_mm: float,
    depth_scale_to_mm: float,
    clearance_tolerance_mm: float,
) -> float:
    """Single-ray clearance: mirror of :func:`approach_clearance_mm`."""
    h, w = depth_map.shape
    steps = int(np.ceil(max_distance_mm / step_mm))
    travelled = 0.0
    for k in range(1, steps + 1):
        distance = min(k * step_mm, max_distance_mm)
        sample = start + march_direction * distance
        z = float(sample[2])
        if z <= 0.0:
            return travelled
        u = sample[0] * intrinsics.fx / z + intrinsics.cx
        v = sample[1] * intrinsics.fy / z + intrinsics.cy
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < w and 0 <= iv < h):
            return min(distance, max_distance_mm)
        depth_mm = float(depth_map[iv, iu]) * depth_scale_to_mm
        if np.isfinite(depth_mm) and depth_mm > 0.0:
            if depth_mm + clearance_tolerance_mm < z:
                return travelled
        travelled = distance
    return min(max_distance_mm, travelled)


def _direction_clearance_and_blockage(
    *,
    target: np.ndarray,
    march_axis_unit: np.ndarray,
    radius_mm: float,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: CorridorAnalysisConfig,
    depth_scale_to_mm: float,
) -> tuple[float, float]:
    """Return ``(clearance_mm, blockage_score)`` for one direction, sampled along the centerline plus eight perimeter rays (four cardinal, four diagonal) at ``radius_mm`` offsets."""
    u, v = _perpendicular_basis(march_axis_unit)
    diag = radius_mm / float(np.sqrt(2.0))
    starts = [
        target,
        target + radius_mm * u,
        target - radius_mm * u,
        target + radius_mm * v,
        target - radius_mm * v,
        target + diag * (u + v),
        target + diag * (u - v),
        target + diag * (-u + v),
        target + diag * (-u - v),
    ]
    clearances: list[float] = []
    for start in starts:
        clearances.append(
            _ray_clearance_mm(
                start=start,
                march_direction=march_axis_unit,
                depth_map=depth_map,
                intrinsics=intrinsics,
                max_distance_mm=config.max_distance_mm,
                step_mm=config.step_mm,
                depth_scale_to_mm=depth_scale_to_mm,
                clearance_tolerance_mm=config.clearance_tolerance_mm,
            )
        )
    # Telemetry uses the most pessimistic ray. The score uses the mean
    # blockage instead, so one blocked perimeter ray cannot promote a
    # partial scene to BLOCKED.
    clearance = float(min(clearances))
    per_ray_blockage = [
        max(0.0, min(1.0, 1.0 - (c / config.max_distance_mm)))
        for c in clearances
    ]
    blockage = float(sum(per_ray_blockage) / len(per_ray_blockage))
    return clearance, blockage


def _mask_corridor_overlap(
    *,
    target: np.ndarray,
    march_axis_unit: np.ndarray,
    radius_mm: float,
    masks: tuple[np.ndarray, ...],
    intrinsics: CameraIntrinsics,
    config: CorridorAnalysisConfig,
) -> float:
    """Fraction of projected corridor samples that fall inside any mask."""
    if not masks:
        return 0.0
    h, w = masks[0].shape
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        arr = np.asarray(m)
        if arr.shape != (h, w):
            raise ValueError(
                "all corridor masks must share the depth map shape; got "
                f"{arr.shape} vs {(h, w)}"
            )
        union |= arr.astype(bool, copy=False)
    u, v = _perpendicular_basis(march_axis_unit)
    diag = radius_mm / float(np.sqrt(2.0))
    starts = [
        target,
        target + radius_mm * u,
        target - radius_mm * u,
        target + radius_mm * v,
        target - radius_mm * v,
        target + diag * (u + v),
        target + diag * (u - v),
        target + diag * (-u + v),
        target + diag * (-u - v),
    ]
    steps = int(np.ceil(config.max_distance_mm / config.step_mm))
    hits = 0
    total = 0
    for start in starts:
        for k in range(1, steps + 1):
            distance = min(k * config.step_mm, config.max_distance_mm)
            sample = start + march_axis_unit * distance
            z = float(sample[2])
            if z <= 0.0:
                continue
            iu = int(round(sample[0] * intrinsics.fx / z + intrinsics.cx))
            iv = int(round(sample[1] * intrinsics.fy / z + intrinsics.cy))
            if not (0 <= iu < w and 0 <= iv < h):
                continue
            total += 1
            if union[iv, iu]:
                hits += 1
    if total == 0:
        return 0.0
    return float(hits) / float(total)


def analyze_corridor(
    inputs: CorridorAnalysisInputs,
    config: CorridorAnalysisConfig,
) -> CorridorReport:
    """Compute a directional :class:`CorridorReport` for one candidate; on missing depth/intrinsics it returns a ``SKIPPED`` report with neutral confidence (never raises) so other candidates still rank."""
    if inputs.depth_map is None or inputs.intrinsics is None:
        return CorridorReport(
            approach_clearance_mm=0.0,
            retreat_clearance_mm=0.0,
            blockage_confidence=0.5,
            mode=CorridorMode.SKIPPED,
        )

    depth = np.asarray(inputs.depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be 2D")

    target = np.asarray(inputs.target_point_cam_mm, dtype=np.float64).reshape(3)
    approach_axis_unit = _unit(inputs.approach_axis_cam)
    # March toward the camera along the approach axis, the same
    # convention :func:`approach_clearance_mm` uses, because the
    # gripper travels from the approach start toward the target.
    approach_march = -approach_axis_unit

    # The retreat corridor is only scored when the caller named a retreat axis.
    #
    # With no axis there is no retreat corridor to measure, and an invented distance is the worst
    # kind of measurement because it looks like evidence. Falling back to the approach axis points
    # away from the camera, straight down through the target and into the support surface on any
    # top-down grasp, which measures ``retr_clear=0`` on 100 % of candidates even on a scene of
    # three separated cubes. The two directions combine as a ``max``, so the confidence is then the
    # retreat term alone and every candidate reads blocked at about 0.86, which is useless as a
    # rejection rule.
    retreat_march = -_unit(inputs.retreat_axis_cam) if inputs.retreat_axis_cam is not None else None

    approach_clearance, approach_blockage = _direction_clearance_and_blockage(
        target=target,
        march_axis_unit=approach_march,
        radius_mm=config.radius_mm,
        depth_map=depth,
        intrinsics=inputs.intrinsics,
        config=config,
        depth_scale_to_mm=float(inputs.depth_scale_to_mm),
    )
    # Not measured when no axis was supplied. The distance then reads 0.0, since the report
    # validates distances as non-negative and no negative sentinel is available, and
    # ``retreat_measured=False`` is what tells a corridor that was not measured apart from one
    # blocked immediately. Read the flag rather than the zero: the two states look identical in
    # the distance alone.
    if retreat_march is not None:
        retreat_clearance, retreat_blockage = _direction_clearance_and_blockage(
            target=target,
            march_axis_unit=retreat_march,
            radius_mm=config.radius_mm,
            depth_map=depth,
            intrinsics=inputs.intrinsics,
            config=config,
            depth_scale_to_mm=float(inputs.depth_scale_to_mm),
        )
    else:
        retreat_clearance, retreat_blockage = 0.0, 0.0

    depth_blockage = (
        max(approach_blockage, retreat_blockage)
        if retreat_march is not None
        else approach_blockage
    )

    if inputs.other_object_masks is None or config.mask_fusion_weight == 0.0:
        confidence = depth_blockage
    else:
        approach_mask = _mask_corridor_overlap(
            target=target,
            march_axis_unit=approach_march,
            radius_mm=config.radius_mm,
            masks=inputs.other_object_masks,
            intrinsics=inputs.intrinsics,
            config=config,
        )
        retreat_mask = (
            _mask_corridor_overlap(
                target=target,
                march_axis_unit=retreat_march,
                radius_mm=config.radius_mm,
                masks=inputs.other_object_masks,
                intrinsics=inputs.intrinsics,
                config=config,
            )
            if retreat_march is not None
            else 0.0
        )
        mask_blockage = max(approach_mask, retreat_mask)
        w = float(config.mask_fusion_weight)
        confidence = (1.0 - w) * depth_blockage + w * mask_blockage

    confidence = max(0.0, min(1.0, confidence))

    if confidence >= config.blocked_confidence_threshold:
        mode = CorridorMode.BLOCKED
    elif confidence >= config.partial_confidence_threshold:
        mode = CorridorMode.PARTIAL
    else:
        mode = CorridorMode.CLEAR

    return CorridorReport(
        approach_clearance_mm=approach_clearance,
        retreat_clearance_mm=retreat_clearance,
        blockage_confidence=confidence,
        mode=mode,
        retreat_measured=retreat_march is not None,
    )
