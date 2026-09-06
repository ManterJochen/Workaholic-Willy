"""Collision validation for 6D grasp poses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.robot.grasping.planning import GraspPose

from .gripper_model import GripperGeometryStrategy, ParallelJawGripperModel, points_to_grasp_frame
from .table_collision import SupportPlane, gripper_table_clearance_mm

__all__ = [
    "GraspCollisionResult",
    "colliding_point_indices",
    "validate_grasp_collision",
    "validate_grasp_collisions",
]


def _points(points_mm: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_mm must be shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_mm must contain only finite values")
    return points


@dataclass(frozen=True, slots=True)
class GraspCollisionResult:
    """Result of table and point-cloud collision validation."""

    pose: GraspPose
    valid: bool
    reasons: tuple[str, ...] = ()
    min_table_clearance_mm: float | None = None
    collision_count: int = 0
    colliding_indices: tuple[int, ...] = ()
    colliding_boxes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "min_table_clearance_mm": self.min_table_clearance_mm,
            "collision_count": self.collision_count,
            "colliding_indices": list(self.colliding_indices),
            "colliding_boxes": list(self.colliding_boxes),
            "metadata": self.metadata,
            "pose": self.pose.to_dict(),
        }


def colliding_point_indices(
    pose: GraspPose,
    points_mm: np.ndarray,
    *,
    gripper_model: GripperGeometryStrategy | None = None,
    margin_mm: float = 0.0,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return scene point indices and box labels intersecting the gripper model."""
    model = gripper_model or ParallelJawGripperModel()
    points = _points(points_mm)
    if points.shape[0] == 0:
        return (), ()
    local = points_to_grasp_frame(points, pose)
    hit_mask = np.zeros(points.shape[0], dtype=bool)
    labels: set[str] = set()
    for box in model.collision_boxes(pose.grip_width_mm):
        inside = box.contains_local_points(local, margin_mm=margin_mm)
        if np.any(inside):
            labels.add(box.label)
            hit_mask |= inside
    return tuple(int(index) for index in np.flatnonzero(hit_mask)), tuple(sorted(labels))


def validate_grasp_collision(
    pose: GraspPose,
    *,
    scene_points_mm: np.ndarray | None = None,
    support_plane: SupportPlane | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    min_table_clearance_mm: float = 5.0,
    collision_margin_mm: float = 0.0,
) -> GraspCollisionResult:
    """Validate one grasp pose against table clearance and optional scene points."""
    if not isinstance(pose, GraspPose):
        raise TypeError("pose must be a GraspPose")
    model = gripper_model or ParallelJawGripperModel()
    min_clearance = float(min_table_clearance_mm)
    if not np.isfinite(min_clearance) or min_clearance < 0.0:
        raise ValueError("min_table_clearance_mm must be finite and >= 0")
    margin = float(collision_margin_mm)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("collision_margin_mm must be finite and >= 0")

    reasons: list[str] = []
    clearance: float | None = None
    if support_plane is not None:
        clearance = gripper_table_clearance_mm(pose, support_plane, gripper_model=model)
        if clearance < min_clearance:
            reasons.append("table_clearance")

    colliding_indices: tuple[int, ...] = ()
    colliding_boxes: tuple[str, ...] = ()
    if scene_points_mm is not None:
        colliding_indices, colliding_boxes = colliding_point_indices(
            pose,
            scene_points_mm,
            gripper_model=model,
            margin_mm=margin,
        )
        if colliding_indices:
            reasons.append("point_cloud_collision")

    return GraspCollisionResult(
        pose=pose,
        valid=not reasons,
        reasons=tuple(reasons),
        min_table_clearance_mm=clearance,
        collision_count=len(colliding_indices),
        colliding_indices=colliding_indices,
        colliding_boxes=colliding_boxes,
        metadata={
            "min_table_clearance_required_mm": min_clearance if support_plane is not None else None,
            "collision_margin_mm": margin,
        },
    )


def validate_grasp_collisions(
    poses: Iterable[GraspPose],
    *,
    scene_points_mm: np.ndarray | None = None,
    support_plane: SupportPlane | None = None,
    gripper_model: GripperGeometryStrategy | None = None,
    min_table_clearance_mm: float = 5.0,
    collision_margin_mm: float = 0.0,
) -> list[GraspCollisionResult]:
    """Validate several poses and preserve input order."""
    return [
        validate_grasp_collision(
            pose,
            scene_points_mm=scene_points_mm,
            support_plane=support_plane,
            gripper_model=gripper_model,
            min_table_clearance_mm=min_table_clearance_mm,
            collision_margin_mm=collision_margin_mm,
        )
        for pose in poses
    ]
