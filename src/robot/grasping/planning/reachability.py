"""Vendor-neutral reachability filtering for grasp poses.

The grasping package may not import a specific robot driver, whether
UR with RTDE, KUKA with EKI, or ROS 2. Reachability checks therefore
live behind an :class:`IKService` protocol that the execution layer
fulfils: a UR adapter, a KUKA adapter, or a pure-Python stub.

The default ``WorkspaceBoxIKService`` keeps :class:`GraspCalculator`
useful without any robot at all by accepting every candidate inside an
axis-aligned workspace box. Production callers replace it with a
service that consults a real solver.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.geometry import Frame
from src.robot.grasping.constants import (
    REACHABILITY_LOG_FILE,
    create_grasping_logger,
)

from .grasp_pose import GraspPose

#: The IK filter is where a scene full of candidates can quietly become none.
#: Logged once per filter call with counts, never per pose.
logger = create_grasping_logger("Reachability", REACHABILITY_LOG_FILE)

__all__ = [
    "IKQualityMetrics",
    "IKResult",
    "IKService",
    "WorkspaceBoxIKService",
    "filter_reachable_poses",
    "transform_grasp_pose",
]


def _validate_optional_nonneg(value: float | None, name: str) -> float | None:
    """Reject negative/NaN/Inf; pass ``None`` through (adapters may omit a signal)."""
    if value is None:
        return None
    f = float(value)
    if not np.isfinite(f):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if f < 0.0:
        raise ValueError(f"{name} must be >= 0, got {f}")
    return f


@dataclass(frozen=True, slots=True)
class IKQualityMetrics:
    """Optional execution-feasibility signals attached to an IK result.

    A real IK adapter, a UR analytical solver or a KUKA EKI bridge for
    instance, can publish per-pose quality metrics that the calculator
    uses to demote candidates with poor execution prospects. All three
    fields are :data:`None` by default, so an adapter that does not
    compute them keeps working unchanged.

    Attributes
    ----------
    condition_number
        Jacobian condition number at the IK solution. Lower is
        better; the feasibility scorer maps it through a soft cap
        configured via :class:`FeasibilityScoreConfig`.
    min_singular_value
        Smallest singular value of the Jacobian. Higher is better;
        guards against near-singular configurations.
    joint_margin_deg
        Degrees from the closest hard joint limit. Higher is
        better; the scorer scales it against a configured
        full-score threshold (e.g. 30 deg).
    """

    condition_number: float | None = None
    min_singular_value: float | None = None
    joint_margin_deg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "condition_number",
            _validate_optional_nonneg(self.condition_number, "condition_number"),
        )
        object.__setattr__(
            self,
            "min_singular_value",
            _validate_optional_nonneg(self.min_singular_value, "min_singular_value"),
        )
        object.__setattr__(
            self,
            "joint_margin_deg",
            _validate_optional_nonneg(self.joint_margin_deg, "joint_margin_deg"),
        )


@dataclass(frozen=True, slots=True)
class IKResult:
    """Outcome of one IK / reachability query for a grasp pose.

    ``reachable`` decides whether the pose is kept. ``reason`` documents
    why a pose was rejected; ``metadata`` may carry the joint solution
    or implementation-specific diagnostics. None of these fields are
    required to interpret a pass/fail decision so adapters can populate
    only what they have.
    """

    reachable: bool
    reason: str | None = None
    joints: tuple[float, ...] | None = None
    metadata: dict[str, Any] | None = None
    quality: IKQualityMetrics | None = None


class IKService(Protocol):
    """Reachability oracle injected by the execution layer.

    Implementations must be pure: no side effects on the calculator or
    on the candidates. The pose is always given in the same frame the
    calculator returned (camera-frame when no transform was passed,
    base-frame otherwise) so the adapter must know which frame it is
    configured for.
    """

    def query(self, pose: GraspPose) -> IKResult: ...


@dataclass(frozen=True, slots=True)
class WorkspaceBoxIKService:
    """Trivial reachability oracle: pose position inside an aabb box.

    The default when nothing is wired, and a sanity baseline behind a
    real adapter. The direction of the approach axis is not consulted;
    that signal already feeds the heuristic reachability score in
    :mod:`src.robot.grasping.scoring`.
    """

    min_corner_mm: np.ndarray
    max_corner_mm: np.ndarray

    def __post_init__(self) -> None:
        min_corner = np.asarray(self.min_corner_mm, dtype=np.float64)
        max_corner = np.asarray(self.max_corner_mm, dtype=np.float64)
        if min_corner.shape != (3,) or max_corner.shape != (3,):
            raise ValueError("workspace corners must be shape (3,)")
        if not np.all(min_corner < max_corner):
            raise ValueError("min_corner_mm must be strictly less than max_corner_mm")
        min_corner.setflags(write=False)
        max_corner.setflags(write=False)
        object.__setattr__(self, "min_corner_mm", min_corner)
        object.__setattr__(self, "max_corner_mm", max_corner)

    def query(self, pose: GraspPose) -> IKResult:
        position = pose.position_mm
        inside = bool(
            np.all(position >= self.min_corner_mm)
            and np.all(position <= self.max_corner_mm)
        )
        if inside:
            return IKResult(reachable=True)
        return IKResult(reachable=False, reason="outside_workspace_box")


def _nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a near-orthonormal ``3x3`` onto the closest proper rotation, by SVD Procrustes.

    The product of two rigid rotations is itself rigid, but a real calibration matrix from
    camera to base carries more numerical slack than a freshly composed pose: real hand-eye
    extrinsics need it, so ``validate_transform`` admits orthonormality within ``1e-4`` while
    :class:`GraspPose` re-validates at ``1e-6``. Without this cleanup the inherited slack trips
    the strict gate of GraspPose and raises mid-pick. Re-orthonormalising the product absorbs
    the slack, so a legitimately rigid transform yields a constructible pose.
    """
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:  # guard against an accidental reflection
        u[:, -1] = -u[:, -1]
        rotation = u @ vt
    return rotation


def transform_grasp_pose(
    pose: GraspPose,
    T_4x4: np.ndarray,
    *,
    target_frame: Frame,
) -> GraspPose:
    """Return ``pose`` expressed in ``target_frame`` via a 4x4 rigid transform.

    The transform's rotation block applies to the pose orientation,
    contacts, and approach/closing/binormal axes; the translation block
    applies only to the position and the contacts. ``T_4x4`` must already
    be validated by the caller (the calculator does this once before
    invoking the IK filter). The composed rotation is re-orthonormalised
    (:func:`_nearest_rotation`) so a calibration matrix carrying the numerical
    slack ``validate_transform`` allows still yields a valid pose.
    """
    rotation = np.asarray(T_4x4[:3, :3], dtype=np.float64)
    translation = np.asarray(T_4x4[:3, 3], dtype=np.float64)
    new_position = rotation @ pose.position_mm + translation
    new_rotation = _nearest_rotation(rotation @ pose.rotation_matrix)
    new_contact_a = rotation @ pose.contacts[0] + translation
    new_contact_b = rotation @ pose.contacts[1] + translation
    return GraspPose(
        position_mm=new_position,
        rotation_matrix=new_rotation,
        grip_width_mm=pose.grip_width_mm,
        score=pose.score,
        confidence=pose.confidence,
        contacts=(new_contact_a, new_contact_b),
        frame=target_frame,
        metadata={**pose.metadata, "source_frame": pose.frame.value},
    )


def filter_reachable_poses(
    poses: Iterable[GraspPose],
    service: IKService,
) -> tuple[list[GraspPose], list[IKResult]]:
    """Run ``service`` on each pose and return the reachable poses with their diagnostics.

    Order is preserved. The returned ``IKResult`` list is parallel to
    the input iterable rather than to the kept-pose list, so a caller
    can correlate each rejection with its source candidate.
    """
    kept: list[GraspPose] = []
    diagnostics: list[IKResult] = []
    for pose in poses:
        result = service.query(pose)
        if not isinstance(result, IKResult):
            raise TypeError(
                "IKService.query must return an IKResult, "
                f"got {type(result).__name__}"
            )
        diagnostics.append(result)
        if result.reachable:
            kept.append(pose)
    # Aggregate, not per pose: the pose count is the candidate count, and the
    # dominant rejection reason is what an operator needs to act on.
    rejected = len(diagnostics) - len(kept)
    if rejected:
        reasons: dict[str, int] = {}
        for result in diagnostics:
            if not result.reachable:
                key = result.reason or "unspecified"
                reasons[key] = reasons.get(key, 0) + 1
        summary = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])
        )
        log = logger.warning if not kept else logger.info
        log(
            "IK filter (%s): %d/%d poses reachable; rejections: %s",
            type(service).__name__,
            len(kept),
            len(diagnostics),
            summary,
        )
    else:
        logger.debug(
            "IK filter (%s): all %d poses reachable",
            type(service).__name__,
            len(diagnostics),
        )
    return kept, diagnostics
