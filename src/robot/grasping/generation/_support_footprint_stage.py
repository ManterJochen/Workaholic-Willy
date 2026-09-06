"""Adapter from Support-Footprint Enumeration into the calculator's candidate pipeline.

Two things this has to get right.

The frame. SFE plans in BASE, because "the support plane" and "down" are BASE concepts. The
calculator's scoring, collision and IK stages run on CAMERA-frame poses and convert to BASE at the
very end. So the candidates are rotated into CAMERA here, once, at the seam.

The order. SFE's own score is what produced its measured top-1 (58.5 % against the calculator's
27.2 %), and the calculator's geometric ranker is anti-correlated with physical validity on this
data: AUC 0.419, worse than a coin. Re-ranking SFE's candidates with it would throw away the result.
So the breakdowns are built directly with SFE's score as ``total_score`` instead of going through
:func:`rank_grasp_poses`.

``stability_score`` and ``reachability_score`` are left at 0.0 and said so: SFE's score is a single
blend of measured physical margins (friction-cone slack, aperture left, table clearance, uprightness,
centredness) and does not decompose into the calculator's three categories. Splitting it into those
buckets would be an invention. The real margins ride along in ``components["support_footprint"]``
instead, as contact angle, support clearance and grip width, where a reader can see them for what
they are; ``metadata`` carries only the stage name and the ``score_is_undecomposed`` flag.
"""

from __future__ import annotations

import numpy as np

from src.geometry import Frame
from src.robot.grasping.planning import GraspPose
from src.robot.grasping.scoring import GraspScoreBreakdown

from .support_footprint import (
    SupportFootprintCandidate,
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)

__all__ = ["support_footprint_breakdowns"]


def _to_camera_pose(candidate: SupportFootprintCandidate,
                    camera_to_base: np.ndarray) -> GraspPose | None:
    rotation_cb, translation_cb = camera_to_base[:3, :3], camera_to_base[:3, 3]
    position = rotation_cb.T @ (candidate.position_mm - translation_cb)
    closing = rotation_cb.T @ candidate.closing_axis
    approach = rotation_cb.T @ candidate.approach
    closing = closing - float(closing @ approach) * approach
    norm = float(np.linalg.norm(closing))
    if norm < 1e-9:
        return None            # degenerate: the closing axis collapsed onto the approach
    closing = closing / norm
    approach = approach / max(float(np.linalg.norm(approach)), 1e-12)
    binormal = np.cross(approach, closing)
    half = 0.5 * candidate.grip_width_mm
    return GraspPose(
        position_mm=position,
        rotation_matrix=np.column_stack([closing, binormal, approach]),
        grip_width_mm=candidate.grip_width_mm,
        score=float(np.clip(candidate.score, 0.0, 1.0)),
        confidence=float(np.clip(candidate.score, 0.0, 1.0)),
        contacts=(position - half * closing, position + half * closing),
        frame=Frame.CAMERA,
    )


def support_footprint_breakdowns(
    target_cloud_base_mm: np.ndarray,
    *,
    camera_to_base: np.ndarray,
    support_height_mm: float,
    jaw: SupportFootprintJaw,
    obstacle_points_base_mm: np.ndarray | None,
    rigid_obstacle_points_base_mm: np.ndarray | None = None,
    max_candidates: int,
    inflate_mm: float = 0.0,
    palm_aware: bool = False,
    score_weights: tuple[float, float, float, float, float] | None = None,
) -> tuple[list[GraspScoreBreakdown], dict]:
    """Run SFE and return camera-frame breakdowns plus its telemetry.

    An empty list is a real answer: the reconstruction admitted no grasp that clears the support.
    Refusing beats proposing one that drives a finger into the table.
    """
    candidates = generate_support_footprint_grasps(
        target_cloud_base_mm,
        support_height_mm=support_height_mm,
        jaw=jaw,
        obstacle_points_base_mm=obstacle_points_base_mm,
        rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm,
        max_candidates=max_candidates,
        inflate_mm=inflate_mm,
        palm_aware=palm_aware,
        score_weights=score_weights,
    )
    breakdowns: list[GraspScoreBreakdown] = []
    for candidate in candidates:
        pose = _to_camera_pose(candidate, camera_to_base)
        if pose is None:
            continue
        breakdowns.append(GraspScoreBreakdown(
            pose=pose,
            total_score=pose.score,
            geometric_score=pose.score,
            stability_score=0.0,
            reachability_score=0.0,
            components={"support_footprint": {
                "contact_angle_deg": float(np.degrees(candidate.contact_angle_rad)),
                "support_clearance_mm": float(candidate.clearance_mm),
                "grip_width_mm": float(candidate.grip_width_mm),
            }},
            metadata={
                "geometry_stage": "support_footprint",
                # The two zeros above are not measurements of zero stability or zero reachability.
                # SFE does not compute those categories; its score is one blend of physical
                # margins, and the margins that exist are in ``components`` above.
                "score_is_undecomposed": True,
            },
        ))
    cloud = np.asarray(target_cloud_base_mm, dtype=np.float64).reshape(-1, 3)
    # float-typed for the millimetre stamps below; the three counters stay ints at runtime.
    telemetry: dict[str, float] = {
        "support_footprint_candidates": len(candidates),
        "support_footprint_kept": len(breakdowns),
        "support_footprint_points": int(cloud.shape[0]),
    }
    # What the stage was looking at, and what it planned: both in BASE, both millimetres.
    #
    # SFE plans the table clearance and the calculator's collision filter re-checks it against the
    # envelope. Two models of one gripper, so the filter's verdict alone cannot say whether a
    # rejection is a model disagreement or a scene fact: the multiview gate reports
    # ``rejected_table: 12`` with ``table_clearance_best_mm: -2.0``, while offline on a clean
    # 30x30x50 cloud the two agree and rank-0 clears by 22.9 mm. These four keys separate the two
    # cases. The cloud's height span says where the object was seen, the anchor says how high SFE
    # aimed, and its own best clearance is directly comparable against ``table_clearance_best_mm``
    # from the filter.
    if cloud.size:
        telemetry["support_footprint_cloud_z_min_mm"] = round(float(cloud[:, 2].min()), 2)
        telemetry["support_footprint_cloud_z_max_mm"] = round(float(cloud[:, 2].max()), 2)
    if candidates:
        # ``candidates`` is score-ordered, so [0] is the one the robot would take.
        telemetry["support_footprint_top_anchor_z_mm"] = round(float(candidates[0].position_mm[2]), 2)
        telemetry["support_footprint_best_clearance_mm"] = round(
            max(float(c.clearance_mm) for c in candidates), 2)
    return breakdowns, telemetry
