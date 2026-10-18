"""Vendor-neutral reachability filtering for grasp poses.

The grasping package may not import a specific robot driver, whether
UR with RTDE or another vendor's. Reachability checks therefore
live behind an :class:`IKService` protocol that the execution layer
fulfils: a UR adapter, another vendor's, or a pure-Python stub.

A cell's service consults its arm's solver (``src/robot/execution/ik_service.py``); the
workspace-box stand-in left on 2026-10-18, as nothing built it.
"""

from __future__ import annotations

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
    "IKResult",
    "IKService",
    "transform_grasp_pose",
]


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


class IKService(Protocol):
    """Reachability oracle injected by the execution layer.

    Implementations must be pure: no side effects on the calculator or
    on the candidates. The pose is always given in the same frame the
    calculator returned (camera-frame when no transform was passed,
    base-frame otherwise) so the adapter must know which frame it is
    configured for.
    """

    def query(self, pose: GraspPose) -> IKResult: ...


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


