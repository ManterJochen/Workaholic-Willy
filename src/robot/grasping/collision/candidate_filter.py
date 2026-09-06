"""The geometric rejection stage both grasp generators run.

It rejects candidates that collide with the scene, hit the table, or leave the workspace box. That
filter is what takes the analytic stack from 22 % to 64 % precision, and it is not optional for the
deep path either: `SafetyPreflight` lives in the driver and refuses the motion, so an unfiltered bad
candidate does not fall through to the next one, it fails the pick at the arm. The candidate filter
sits in a different layer from the safety layer, and neither substitutes for the other.

Frame-agnostic, and fail-closed about it. The two callers work in different frames: the analytic
calculator filters CAMERA-frame poses and converts the BASE plane the pick loop hands it, while the
deep calculator emits BASE-frame poses and needs that same plane unconverted. Rather than pick a frame
and make one caller wrong, this function takes everything in one frame and refuses when the pose and
the plane disagree about which. That refusal is the point: a frame defect here is two quantities in
different frames meeting silently, a camera depth read as a height, or a closing axis zeroed in the
camera frame instead of the support plane. It is the first thing checked, before any work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.grasping.collision.collision_checker import validate_grasp_collision
from src.robot.grasping.collision.table_collision import SupportPlane
from src.robot.grasping.planning.grasp_pose import Frame, GraspPose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.collision.gripper_model import GripperGeometryStrategy
    from src.robot.grasping.types.grasp_point import GraspPoint

__all__ = ["FilterOutcome", "REJECTION_KEYS", "filter_candidates", "grasp_point_to_pose",
           "rejection_reasons"]

#: The telemetry keys this stage emits, in the order a histogram should read them.
#:
#: They are the names the analytic path uses in `generation/calculator.py`, on purpose: a rejection
#: histogram is only worth having if the two calculators' histograms can be put side by side, and they
#: cannot if one says `rejected_table` and the other `table_rejected`. `rejected_grip_width` belongs to
#: `_fits_the_jaw` in the deep calculator and keeps its own name for the same reason.
REJECTION_KEYS: tuple[str, ...] = (
    "rejected_workspace",
    "rejected_table",
    "rejected_collision",
)


@dataclass(frozen=True, slots=True)
class FilterOutcome:
    """Which candidates survived, and why the others did not."""

    #: Indices into the input sequence, in input order.
    kept: tuple[int, ...]
    #: One count per `REJECTION_KEYS` entry, plus the clearance stamps when a plane was checked.
    telemetry: dict[str, float]

    @property
    def rejected(self) -> int:
        return int(sum(self.telemetry.get(key, 0.0) for key in REJECTION_KEYS))


def grasp_point_to_pose(grasp: "GraspPoint") -> GraspPose:
    """Bridge a `GraspPoint` to the `GraspPose` the collision validator takes.

    One adapter, not two. This is `loop/_pick_helpers._grasp_point_to_grasp_pose` in a public home, so
    the deep calculator can reach it without importing a private module from another subpackage, and
    `_pick_helpers` delegates here. A copy would be the repo's third gripper description, and two is
    already one more than stays consistent on its own.

    Rebuilds the rotation with the `[closing, binormal, approach]` column convention, the same one
    `execution_policy` uses, so the swept gripper volume is oriented correctly, and synthesizes the two
    contacts at `position +/- grip_width/2` along the closing axis. The frame travels with the pose,
    which is what makes this a bridge between frames rather than another place they meet silently.
    """
    closing = np.asarray(grasp.axis, dtype=np.float64)
    approach = np.asarray(grasp.approach, dtype=np.float64)
    binormal = np.cross(approach, closing)
    binormal_norm = float(np.linalg.norm(binormal))
    if binormal_norm < 1e-9:
        raise ValueError("degenerate grasp axes: approach is parallel to the closing axis")
    binormal /= binormal_norm
    position = np.asarray(grasp.position, dtype=np.float64)
    half = 0.5 * float(grasp.grip_width_mm)
    score = max(0.0, min(1.0, float(grasp.score)))
    return GraspPose(
        position_mm=position,
        rotation_matrix=np.column_stack([closing, binormal, approach]),
        grip_width_mm=float(grasp.grip_width_mm),
        score=score,
        confidence=score,
        contacts=(position - closing * half, position + closing * half),
        frame=Frame(grasp.frame.value),
    )


def rejection_reasons(telemetry: "dict[str, float]") -> "tuple[Any, ...]":
    """Turn a rejection histogram into the failure reasons the recovery orchestrator routes on.

    This is what makes the filter useful beyond accuracy. `recovery/orchestrator.py` routes
    ALL_COLLIDED to a container agitate, and ALL_TABLE_CONFLICT and ALL_OUT_OF_WORKSPACE to
    NEXT_TARGET with a rescan. A generator that reports only NO_VALID_GRASP collapses all of those
    into one generic route, and the cell then retries the same thing instead of shaking the bin. Both
    calculators build their reasons from these counters through this one mapping, so neither routing
    table can drift away from the other.

    RESCAN_RECOMMENDED is appended whenever anything was rejected: a scene where every candidate died
    geometrically is a scene worth perceiving again.
    """
    from src.robot.grasping.types.feedback import GraspFailureReason  # noqa: PLC0415

    reasons: list[Any] = []
    if int(telemetry.get("rejected_workspace", 0)) > 0:
        reasons.append(GraspFailureReason.ALL_OUT_OF_WORKSPACE)
    if int(telemetry.get("rejected_table", 0)) > 0:
        reasons.append(GraspFailureReason.ALL_TABLE_CONFLICT)
    if int(telemetry.get("rejected_collision", 0)) > 0:
        reasons.append(GraspFailureReason.ALL_COLLIDED)
    if not reasons:
        reasons.append(GraspFailureReason.NO_VALID_GRASP)
    reasons.append(GraspFailureReason.RESCAN_RECOMMENDED)
    return tuple(reasons)


def _refuse_mixed_frames(poses: Sequence[GraspPose], plane: SupportPlane | None) -> None:
    """Refuse a plane and a pose that disagree about which frame they are in.

    `SupportPlane` carries a `frame` tag and so does `GraspPose`, so the mismatch is checkable here
    rather than merely commented about. A BASE plane compared against CAMERA-frame poses measures a
    median clearance of 577 mm where the true clearance is -7.01 mm, and rejects 0.4 % of candidates
    instead of 65.5 %. The check runs before any work, because a filter that returns a plausible
    histogram out of mismatched frames is worse than one that refuses.

    Both defaults are `Frame.CAMERA`, so a caller that never thought about frames is unaffected.
    """
    if plane is None or not poses:
        return
    frames = {pose.frame for pose in poses}
    if len(frames) > 1:
        raise ValueError(
            f"candidates are not all in one frame ({sorted(f.value for f in frames)}). A single "
            f"support plane cannot be correct for all of them")
    pose_frame = next(iter(frames))
    if pose_frame is not plane.frame:
        raise ValueError(
            f"support_plane is in {plane.frame.value} frame but the candidates are in "
            f"{pose_frame.value}. Convert one before filtering: `SupportPlane.to_camera_frame()` for "
            f"the analytic path, or hand the BASE plane straight through for a BASE-frame generator. "
            f"Comparing them as-is is how a camera depth was once read as a table height.")


def filter_candidates(
    poses: Sequence[GraspPose],
    *,
    scene_points_mm: np.ndarray | None = None,
    support_plane: SupportPlane | None = None,
    workspace: Any | None = None,
    gripper_model: "GripperGeometryStrategy | None" = None,
    min_table_clearance_mm: float = 5.0,
    collision_margin_mm: float = 0.0,
) -> FilterOutcome:
    """Reject candidates that leave the box, hit the table, or collide with the scene.

    Every argument must be expressed in the same frame as `poses`; see the module docstring. The order
    is workspace, then table, then collision, cheapest first: the box test is three comparisons and the
    swept volume is the expensive one, so a candidate outside the workspace never pays for it.

    Returns the kept indices rather than the poses, so a caller can carry its own richer object, a
    `GraspScoreBreakdown` or a `GraspPoint`, through the filter without this module knowing about it.
    """
    _refuse_mixed_frames(poses, support_plane)

    kept: list[int] = []
    telemetry: dict[str, float] = {key: 0.0 for key in REJECTION_KEYS}
    # How far the table check missed by, not just how often. A bare count cannot separate an object
    # that is too short for a top-down jaw at all from a required clearance that is a millimetre too
    # strict, and those two lead to opposite conclusions.
    best_clearance_mm: float | None = None

    for index, pose in enumerate(poses):
        if workspace is not None and not workspace.contains(pose.position_mm):
            telemetry["rejected_workspace"] += 1.0
            continue
        if support_plane is None and scene_points_mm is None:
            kept.append(index)
            continue

        result = validate_grasp_collision(
            pose,
            scene_points_mm=scene_points_mm,
            support_plane=support_plane,
            gripper_model=gripper_model,
            min_table_clearance_mm=min_table_clearance_mm,
            collision_margin_mm=collision_margin_mm,
        )
        clearance = getattr(result, "min_table_clearance_mm", None)
        if clearance is not None and np.isfinite(clearance):
            best_clearance_mm = (float(clearance) if best_clearance_mm is None
                                 else max(best_clearance_mm, float(clearance)))
        if not result.valid:
            if "table_clearance" in result.reasons:
                telemetry["rejected_table"] += 1.0
            if "point_cloud_collision" in result.reasons:
                telemetry["rejected_collision"] += 1.0
            continue
        kept.append(index)

    if support_plane is not None:
        # The frame belongs in the key name. Stamped untagged, this offset reads as a BASE height of
        # -311 mm against a table at 0, which looks exactly like a frame defect that is not there. The
        # two clearances need no tag: a signed distance to the plane is the same number in either
        # frame.
        telemetry[f"support_plane_offset_{support_plane.frame.value}_mm"] = round(
            float(support_plane.offset_mm), 2)
        telemetry["table_clearance_required_mm"] = round(float(min_table_clearance_mm), 2)
        if best_clearance_mm is not None:
            telemetry["table_clearance_best_mm"] = round(best_clearance_mm, 2)

    return FilterOutcome(kept=tuple(kept), telemetry=telemetry)
