"""Approach-path and retreat swept-volume validation.

The pipeline exposes :func:`approach_clearance_mm`, which returns one
scalar: the unobstructed distance along the approach cone of the final
grasp pose. That is telemetry rather than a trajectory-level check,
because it inspects the single waypoint at the final grasp and ignores
the pre-grasp standoff, the linear interpolation in between, and the
retreat motion.

This module is the typed swept-volume validator that covers the rest. It:

* generates a deterministic series of intermediate poses between the
  pre-grasp standoff and the final grasp, and between the final grasp
  and the retreat target;
* re-uses the shared :class:`GripperGeometryStrategy`, jaw or suction,
  with the :func:`colliding_point_indices` collision filter at each
  sample, so the moving gripper bounding volume is checked against
  scene points;
* returns a frozen :class:`ApproachPathReport` whose
  :class:`ApproachPathOutcome` is one of ``CLEAR``, ``BLOCKED``,
  ``NO_OBSTACLES``, or ``SKIPPED``.

Scope and non-goals
-------------------

* This is a validator, not a motion planner. It does not command
  :meth:`RobotArm.move`, it returns a report, and the caller decides
  whether to refuse the pick or fall back to a different candidate.
* The check is point-cloud based, re-using the scene representation the
  calculator already builds from ``other_object_masks``. No mesh, urdf
  or swept-volume integration is introduced.
* :class:`SafetyPreflight` is not invoked here. The preflight stack
  operates on joint space and full-arm motion, while this module
  operates on grasp-pose-local gripper geometry against a scene point
  cloud. The two are complementary, and a wrapper may invoke both
  before any motion. The :class:`TrajectorySafetyCheck` Protocol is
  exposed so that a caller can plug a predicate backed by a real
  :class:`SafetyPreflight` in without touching this module.
* No new motion status is added to :class:`MotionStatus`. A caller that
  wires this validator into the orchestrator can map the outcome to
  ``WORKSPACE_REJECTED`` or to a new ``APPROACH_PATH_BLOCKED``
  :class:`GraspFailureReason`; the README names both as candidates.

Public surface
--------------

* :class:`ApproachPathPolicy`, the bounded configuration.
* :class:`ApproachPathOutcome`, the typed outcome enum.
* :class:`TrajectoryStepReport`, the result for one sampled pose.
* :class:`ApproachPathReport`, the aggregate of a full sweep.
* :class:`TrajectorySafetyCheck`, the Protocol for an optional outer
  per-step safety predicate such as :class:`SafetyPreflight`.
* :class:`AcceptAllTrajectorySafetyCheck`, the default that accepts
  every step.
* :func:`validate_approach_path`, from pre-grasp to grasp.
* :func:`validate_retreat_path`, from grasp to retreat.
* :func:`validate_approach_and_retreat`, the combined helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from src.robot.grasping.collision.collision_checker import (
    colliding_point_indices,
)
from src.robot.grasping.collision.gripper_model import (
    GripperGeometryStrategy,
)
from src.robot.grasping.planning import GraspPose

__all__ = [
    "AcceptAllTrajectorySafetyCheck",
    "ApproachPathOutcome",
    "ApproachPathPolicy",
    "ApproachPathReport",
    "TrajectoryStepReport",
    "TrajectorySafetyCheck",
    "validate_approach_and_retreat",
    "validate_approach_path",
    "validate_retreat_path",
]


_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Enums + simple records
# ---------------------------------------------------------------------------


class ApproachPathOutcome(StrEnum):
    """Typed result of a swept-volume sweep."""

    CLEAR = "clear"
    """Every sampled step passed both collision and safety checks."""

    BLOCKED = "blocked"
    """At least one sampled step collided with the scene point cloud
    or was rejected by the trajectory safety check."""

    NO_OBSTACLES = "no_obstacles"
    """The caller provided no scene point cloud and no safety check
    rejected the sweep, so the path is treated as clear. It is a
    distinct outcome so that a service layer can refuse to drive
    motion when no obstacle map was supplied."""

    SKIPPED = "skipped"
    """The policy disabled this validator, e.g. ``num_samples == 0``."""


@dataclass(frozen=True, slots=True)
class TrajectoryStepReport:
    """One sampled pose along an approach or retreat trajectory.

    Attributes
    ----------
    index
        Zero-based sample index along the trajectory.
    fraction
        Normalised position along the trajectory in ``[0.0, 1.0]``.
        For approach: ``0.0`` is the pre-grasp standoff and ``1.0``
        is the final grasp. For retreat: ``0.0`` is the final grasp
        and ``1.0`` is the retreat target.
    position_mm
        World-frame position of the sampled pose.
    collision_count
        Number of scene points found inside the gripper bounding
        volume at this step. ``0`` means the step is clear.
    colliding_boxes
        Sorted, deduplicated tuple of bounding-box labels that
        registered a collision at this step. Empty when clear.
    safety_rejected
        ``True`` iff the outer :class:`TrajectorySafetyCheck`
        predicate refused this step.
    """

    index: int
    fraction: float
    position_mm: tuple[float, float, float]
    collision_count: int = 0
    colliding_boxes: tuple[str, ...] = ()
    safety_rejected: bool = False

    @property
    def is_blocked(self) -> bool:
        return self.safety_rejected or self.collision_count > 0


@dataclass(frozen=True, slots=True)
class ApproachPathReport:
    """Frozen aggregate of a full sweep.

    Attributes
    ----------
    outcome
        :class:`ApproachPathOutcome` summary.
    steps
        Every sampled step, in trajectory order, including the
        blocking one when present.
    first_blocking_index
        Index of the first step that registered a block, or ``None``
        when no step blocked.
    telemetry
        Free-form JSON-safe key/value bag.
    """

    outcome: ApproachPathOutcome
    steps: tuple[TrajectoryStepReport, ...] = ()
    first_blocking_index: Optional[int] = None
    telemetry: dict = field(default_factory=dict)

    @property
    def is_clear(self) -> bool:
        return self.outcome in (
            ApproachPathOutcome.CLEAR,
            ApproachPathOutcome.NO_OBSTACLES,
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApproachPathPolicy:
    """Bounded configuration for swept-volume validation.

    Attributes
    ----------
    standoff_mm
        Pre-grasp standoff distance along ``-approach_axis``.
    retreat_mm
        Retreat distance along :attr:`retreat_direction` after the
        final grasp.
    retreat_direction
        Unit vector for retreat. Defaults to world ``+Z``. Set it to
        the negated approach axis to retreat back along the approach
        line.
    num_approach_samples
        Number of samples between pre-grasp and grasp inclusive.
        Must be ``>= 2``. ``0`` disables the approach validator
        (sweep returns :attr:`ApproachPathOutcome.SKIPPED`).
    num_retreat_samples
        Same for retreat.
    collision_margin_mm
        Margin added to gripper collision boxes. Larger values are
        more conservative (more likely to flag a collision).
    """

    standoff_mm: float = 80.0
    retreat_mm: float = 100.0
    retreat_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    num_approach_samples: int = 6
    num_retreat_samples: int = 4
    collision_margin_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.standoff_mm < 0.0:
            raise ValueError(
                f"standoff_mm must be non-negative; got {self.standoff_mm}"
            )
        if self.retreat_mm < 0.0:
            raise ValueError(
                f"retreat_mm must be non-negative; got {self.retreat_mm}"
            )
        if self.num_approach_samples not in (0,) and self.num_approach_samples < 2:
            raise ValueError(
                "num_approach_samples must be 0 (disabled) or >= 2; "
                f"got {self.num_approach_samples}"
            )
        if self.num_retreat_samples not in (0,) and self.num_retreat_samples < 2:
            raise ValueError(
                "num_retreat_samples must be 0 (disabled) or >= 2; "
                f"got {self.num_retreat_samples}"
            )
        if self.collision_margin_mm < 0.0:
            raise ValueError(
                "collision_margin_mm must be non-negative; "
                f"got {self.collision_margin_mm}"
            )
        if len(self.retreat_direction) != 3:
            raise ValueError(
                f"retreat_direction must have 3 components; got {self.retreat_direction!r}"
            )
        if float(np.linalg.norm(self.retreat_direction)) <= 0.0:
            raise ValueError("retreat_direction must be a non-zero vector")


# ---------------------------------------------------------------------------
# Safety check Protocol + default
# ---------------------------------------------------------------------------


@runtime_checkable
class TrajectorySafetyCheck(Protocol):
    """Predicate: is ``pose`` an acceptable trajectory waypoint?

    Wire this to :class:`SafetyPreflight`, or to a project-specific
    workspace check. The validator calls :meth:`is_safe` once per
    sampled step, and rejecting any one step marks the whole sweep
    :attr:`ApproachPathOutcome.BLOCKED`.
    """

    def is_safe(self, pose: GraspPose) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class AcceptAllTrajectorySafetyCheck:
    """No-op default. Accepts every sampled waypoint."""

    def is_safe(self, pose: GraspPose) -> bool:  # noqa: ARG002
        return True


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_approach_path(
    grasp: GraspPose,
    *,
    obstacle_points_mm: Optional[np.ndarray] = None,
    policy: Optional[ApproachPathPolicy] = None,
    gripper_model: Optional[GripperGeometryStrategy] = None,
    safety_check: Optional[TrajectorySafetyCheck] = None,
) -> ApproachPathReport:
    """Sweep from the standoff to the grasp and report any blocked step.

    The sweep keeps the grasp orientation and the grip width fixed and
    interpolates only the position along ``-approach_axis``. The final
    sample is the grasp itself, so this validator subsumes the
    single-pose collision check at the grasp pose.
    """

    if not isinstance(grasp, GraspPose):
        raise TypeError("grasp must be a GraspPose")
    policy = policy or ApproachPathPolicy()
    safety = safety_check or AcceptAllTrajectorySafetyCheck()
    if policy.num_approach_samples == 0:
        return ApproachPathReport(outcome=ApproachPathOutcome.SKIPPED)
    pre_position = (
        np.asarray(grasp.position_mm, dtype=np.float64)
        - np.asarray(grasp.approach_axis, dtype=np.float64) * float(policy.standoff_mm)
    )
    target_position = np.asarray(grasp.position_mm, dtype=np.float64)
    return _sweep(
        grasp=grasp,
        start_position=pre_position,
        end_position=target_position,
        num_samples=policy.num_approach_samples,
        obstacle_points_mm=obstacle_points_mm,
        gripper_model=gripper_model,
        margin_mm=policy.collision_margin_mm,
        safety_check=safety,
        leg="approach",
    )


def validate_retreat_path(
    grasp: GraspPose,
    *,
    obstacle_points_mm: Optional[np.ndarray] = None,
    policy: Optional[ApproachPathPolicy] = None,
    gripper_model: Optional[GripperGeometryStrategy] = None,
    safety_check: Optional[TrajectorySafetyCheck] = None,
) -> ApproachPathReport:
    """Sweep from the grasp to the retreat target and report any blocked step."""

    if not isinstance(grasp, GraspPose):
        raise TypeError("grasp must be a GraspPose")
    policy = policy or ApproachPathPolicy()
    safety = safety_check or AcceptAllTrajectorySafetyCheck()
    if policy.num_retreat_samples == 0:
        return ApproachPathReport(outcome=ApproachPathOutcome.SKIPPED)
    direction = np.asarray(policy.retreat_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    unit = direction / norm
    start_position = np.asarray(grasp.position_mm, dtype=np.float64)
    end_position = start_position + unit * float(policy.retreat_mm)
    return _sweep(
        grasp=grasp,
        start_position=start_position,
        end_position=end_position,
        num_samples=policy.num_retreat_samples,
        obstacle_points_mm=obstacle_points_mm,
        gripper_model=gripper_model,
        margin_mm=policy.collision_margin_mm,
        safety_check=safety,
        leg="retreat",
    )


def validate_approach_and_retreat(
    grasp: GraspPose,
    *,
    obstacle_points_mm: Optional[np.ndarray] = None,
    policy: Optional[ApproachPathPolicy] = None,
    gripper_model: Optional[GripperGeometryStrategy] = None,
    safety_check: Optional[TrajectorySafetyCheck] = None,
) -> tuple[ApproachPathReport, ApproachPathReport]:
    """Run :func:`validate_approach_path` followed by :func:`validate_retreat_path`.

    Returns a tuple ``(approach_report, retreat_report)``. The caller
    decides how to combine the two outcomes; they are deliberately not
    collapsed here, so that telemetry stays attributable.
    """

    approach = validate_approach_path(
        grasp,
        obstacle_points_mm=obstacle_points_mm,
        policy=policy,
        gripper_model=gripper_model,
        safety_check=safety_check,
    )
    retreat = validate_retreat_path(
        grasp,
        obstacle_points_mm=obstacle_points_mm,
        policy=policy,
        gripper_model=gripper_model,
        safety_check=safety_check,
    )
    return approach, retreat


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sweep(
    *,
    grasp: GraspPose,
    start_position: np.ndarray,
    end_position: np.ndarray,
    num_samples: int,
    obstacle_points_mm: Optional[np.ndarray],
    gripper_model: Optional[GripperGeometryStrategy],
    margin_mm: float,
    safety_check: TrajectorySafetyCheck,
    leg: str,
) -> ApproachPathReport:
    """Sample from ``start_position`` to ``end_position``, running the collision and safety check per step.

    Builds a fresh :class:`GraspPose` at each sample, with the same orientation, contacts, grip width
    and frame. The first blocked step short-circuits the rest of the sweep: later steps add no useful
    information, and an executor would refuse at the first block anyway.
    """

    have_obstacles = (
        obstacle_points_mm is not None
        and np.asarray(obstacle_points_mm).size > 0
    )
    fractions = np.linspace(0.0, 1.0, num=num_samples)
    delta = end_position - start_position
    steps: list[TrajectoryStepReport] = []
    first_blocking: Optional[int] = None
    any_safety_check_run = False
    for index, fraction in enumerate(fractions):
        position = start_position + delta * float(fraction)
        sampled = _grasp_at(grasp, position)
        # Collision side.
        collision_count = 0
        boxes: tuple[str, ...] = ()
        if have_obstacles:
            indices, labels = colliding_point_indices(
                sampled,
                np.asarray(obstacle_points_mm),
                gripper_model=gripper_model,
                margin_mm=margin_mm,
            )
            collision_count = len(indices)
            boxes = labels
        # Safety side.
        any_safety_check_run = True
        safety_rejected = not bool(safety_check.is_safe(sampled))
        step = TrajectoryStepReport(
            index=index,
            fraction=float(fraction),
            position_mm=(float(position[0]), float(position[1]), float(position[2])),
            collision_count=collision_count,
            colliding_boxes=boxes,
            safety_rejected=safety_rejected,
        )
        steps.append(step)
        if step.is_blocked and first_blocking is None:
            first_blocking = index
            break
    if first_blocking is not None:
        outcome = ApproachPathOutcome.BLOCKED
    elif have_obstacles or _is_strict_safety_check(safety_check):
        outcome = ApproachPathOutcome.CLEAR
    else:
        outcome = ApproachPathOutcome.NO_OBSTACLES
    telemetry = {
        "leg": leg,
        "num_samples": int(num_samples),
        "obstacle_points": (
            int(np.asarray(obstacle_points_mm).shape[0])
            if obstacle_points_mm is not None
            else 0
        ),
        "safety_check_ran": any_safety_check_run,
    }
    return ApproachPathReport(
        outcome=outcome,
        steps=tuple(steps),
        first_blocking_index=first_blocking,
        telemetry=telemetry,
    )


def _grasp_at(grasp: GraspPose, position_mm: np.ndarray) -> GraspPose:
    """Return a ``GraspPose`` at ``position_mm`` keeping every other field."""

    return GraspPose(
        position_mm=np.asarray(position_mm, dtype=np.float64),
        rotation_matrix=grasp.rotation_matrix.copy(),
        grip_width_mm=grasp.grip_width_mm,
        score=grasp.score,
        confidence=grasp.confidence,
        contacts=(grasp.contacts[0].copy(), grasp.contacts[1].copy()),
        frame=grasp.frame,
        metadata=dict(grasp.metadata),
    )


def _is_strict_safety_check(check: TrajectorySafetyCheck) -> bool:
    """``True`` when ``check`` is not the no-op default.

    A non-default check is evidence the sweep was validated against a real constraint, so the outcome is
    ``CLEAR`` even when no obstacle points were supplied.
    """

    return not isinstance(check, AcceptAllTrajectorySafetyCheck)
