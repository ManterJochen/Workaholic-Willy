"""Collision and physical validation utilities for grasp candidates."""

from .collision_checker import (
    GraspCollisionResult,
    colliding_point_indices,
    validate_grasp_collision,
    validate_grasp_collisions,
)
from .gripper_model import (
    CollisionBox,
    GripperGeometryStrategy,
    ParallelJawGripperModel,
    SuctionCupGripperModel,
    points_to_grasp_frame,
)
from .candidate_filter import (
    REJECTION_KEYS,
    FilterOutcome,
    filter_candidates,
    grasp_point_to_pose,
    rejection_reasons,
)
from .container import container_wall_points_base_mm
from .support_resolver import SupportResolution, resolve_support_plane
from .table_collision import SupportPlane, gripper_table_clearance_mm

__all__ = [
    "FilterOutcome",
    "REJECTION_KEYS",
    "filter_candidates",
    "grasp_point_to_pose",
    "rejection_reasons",
    "CollisionBox",
    "GraspCollisionResult",
    "GripperGeometryStrategy",
    "ParallelJawGripperModel",
    "SuctionCupGripperModel",
    "SupportPlane",
    "SupportResolution",
    "colliding_point_indices",
    "container_wall_points_base_mm",
    "gripper_table_clearance_mm",
    "resolve_support_plane",
    "points_to_grasp_frame",
    "validate_grasp_collision",
    "validate_grasp_collisions",
]
