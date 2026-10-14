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

Why SFE refused what it refused rides along too: ``support_footprint_refused`` holds one count per
cause (``support_footprint.REFUSAL_CAUSES``), so a pick that got no grasp says whether the camera's
neighbours, a declared body, the part's own fragments or the support stopped it.

And where a caller held SFE's fine search back (``fine_pass``) and the coarse grid found few grasps, and the part's
own body (below) did not make up for them, ``support_footprint_fine`` says ``"deferred"``: the result is the coarse
grid's, and the pick loop asks again in full before it takes such a part (``FINE_SEARCH_KEY``). A search that ran in
full stamps nothing.

The grasps are the part's own body's (``body_grasps``, lever A of 2026-10-13): what the cameras measured of it, a
ring's wall, a star's lobe, a lying cylinder at its equator, which SFE's convex prism cannot offer. SFE's prism is no
longer asked (the owner, 2026-10-14: the body alone). On the owner's three days from each pick's first pose the prism
added no pick to the body's 178 of 202, the first grasp tried was admitted in 169 picks with it and 175 without, and a
pick ranked in 6.8 s with it and 5.0 s without. Its search and the arguments only it read (``inflate_mm``,
``palm_aware``, ``fine_pass``, ``runner``, ``batched``) go with the cleanup after the cell has picked with the body
alone; until then they are taken and not read. ``body_grasps`` and ``body_refused`` in the telemetry say how many the
body offered and why it offered no more.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry import Frame
from src.robot.grasping.planning import GraspPose
from src.robot.grasping.scoring import GraspScoreBreakdown

from .body_grasps import body_grasps
from .support_footprint import (
    DEFAULT_FLOOR_MARGIN_MM,
    FINE_SEARCH_DEFERRED,
    SIDE_APPROACH_SCORE_WEIGHTS,
    CorridorSeen,
    HandFloor,
    SfeRunner,
    SupportFootprintCandidate,
    SupportFootprintJaw,
)

__all__ = ["FINE_SEARCH_DEFERRED", "FINE_SEARCH_KEY", "support_footprint_breakdowns"]

#: The telemetry key under which a result says SFE left its fine search for later: ``FINE_SEARCH_DEFERRED``, and only
#: then.
FINE_SEARCH_KEY = "support_footprint_fine"


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
    floor_margin_mm: float | None = None,
    side_approaches: bool = False,
    corridor_seen: CorridorSeen | None = None,
    seen_envelope: Any = None,
    hand_floor: HandFloor | None = None,
    fine_pass: bool = True,
    runner: SfeRunner | None = None,
    batched: bool = False,
    opens_as_needed: bool = False,
) -> tuple[list[GraspScoreBreakdown], dict]:
    """Run SFE and return camera-frame breakdowns plus its telemetry.

    An empty list is a real answer: the reconstruction admitted no grasp that clears the support.
    Refusing beats proposing one that drives a finger into the table.

    ``floor_margin_mm`` of None is the stage's own floor, ``DEFAULT_FLOOR_MARGIN_MM``, read from where it
    is declared.

    ``side_approaches`` and ``corridor_seen`` are SFE's own (``generate_support_footprint_grasps``). The
    library default stays off, so a direct caller is unchanged; a cell turns them on from
    ``robot.grasping.side_approaches`` with ``corridor_seen`` built from the frame.

    ``seen_envelope`` (``scene_obstacles.SeenEnvelope``) holds the boxes the camera world builds of the neighbours: a
    build whose open hand comes nearer to one than the guard keeps is refused under ``seen_fingers``.

    ``hand_floor`` (``support_footprint.HandFloor``) holds the support solids the guard holds under the hand.

    ``fine_pass`` off holds SFE's fine search back where the coarse grid finds few grasps; the telemetry then says
    ``support_footprint_fine: "deferred"`` (:data:`FINE_SEARCH_KEY`). On, the default, is the search as before, and the
    telemetry carries no such key.

    ``runner`` runs SFE's units in other processes (``src.robot.grasping.workers.SfeWorkers``): the same candidates,
    counts and telemetry as without it, and a runner that cannot answer has the part searched here. ``None``, the
    default, runs every unit in this process.

    ``batched`` makes each of SFE's closing lines build its grasps at once (``generate_support_footprint_grasps``), in
    this process and a runner's alike: the same candidates, counts and telemetry, to the bit. Off, the default, builds
    them one at a time, as before.

    ``opens_as_needed`` plans the body's grasps for a hand that opens only as wide as each span needs
    (``body_grasps``, lever G); off, a hand that opens fully.
    """
    # The part the guard holds out (``perceived.target_body_points``): never a piece of something else its mask ran onto.
    from src.robot.safety.planning.perceived import target_body_points  # noqa: PLC0415 (the stage's only use of it)

    target_cloud_base_mm = target_body_points(target_cloud_base_mm)
    # Scored by the weights SFE scored by: with side approaches upright weighs nothing and the room a grasp keeps counts
    # more (the owner, 2026-10-01). By the plain weights a vertical grasp of the body 16 mm from a wall outranked SFE's
    # tilt 21 mm from it (2026-10-14).
    body_refused: dict[str, int] = {}
    candidates = body_grasps(
        target_cloud_base_mm, support_height_mm=support_height_mm, jaw=jaw,
        obstacle_points_base_mm=obstacle_points_base_mm, rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm,
        seen_envelope=seen_envelope, hand_floor=hand_floor, corridor_seen=corridor_seen,
        floor_margin_mm=DEFAULT_FLOOR_MARGIN_MM if floor_margin_mm is None else float(floor_margin_mm),
        opens_as_needed=opens_as_needed, max_candidates=max_candidates, refusals=body_refused,
        score_weights=score_weights if score_weights is not None or not side_approaches else SIDE_APPROACH_SCORE_WEIGHTS)
    breakdowns: list[GraspScoreBreakdown] = []
    for candidate in candidates:
        pose = _to_camera_pose(candidate, camera_to_base)
        if pose is None:
            continue
        margins: dict[str, Any] = {
            "contact_angle_deg": float(np.degrees(candidate.contact_angle_rad)),
            "support_clearance_mm": float(candidate.clearance_mm),
            "grip_width_mm": float(candidate.grip_width_mm),
            # Whether it holds with the hand ROBUST_ERROR_MM off it across the table; these rank first.
            "robust": bool(candidate.robust),
            # Which stage found it: the part's own body (``body_grasps``).
            "stage": "body",
            # How wide the jaws open before they close, where the body's grasp was planned for jaws that open only
            # as needed (lever G): the span and the room beside each jaw. Absent, the hand opens fully.
            **({"open_width_mm": float(candidate.open_width_mm)} if candidate.open_width_mm is not None else {}),
        }
        breakdowns.append(GraspScoreBreakdown(
            pose=pose,
            total_score=pose.score,
            geometric_score=pose.score,
            stability_score=0.0,
            reachability_score=0.0,
            components={"support_footprint": margins},
            metadata={
                "geometry_stage": "body",
                # The two zeros above are not measurements of zero stability or zero reachability.
                # SFE does not compute those categories; its score is one blend of physical
                # margins, and the margins that exist are in ``components`` above.
                "score_is_undecomposed": True,
            },
        ))
    cloud = np.asarray(target_cloud_base_mm, dtype=np.float64).reshape(-1, 3)
    # Counters and millimetre stamps, and one dict: why every refused build was refused.
    telemetry: dict[str, Any] = {
        "support_footprint_kept": len(breakdowns),
        "support_footprint_points": int(cloud.shape[0]),
        "body_grasps": len(candidates),
        "body_kept": len(breakdowns),
        "body_refused": body_refused,
    }
    if side_approaches:
        telemetry["support_footprint_side_approaches"] = True
        # Without a seen test no tilt past the first that fits is offered, so a reader needs to know which it was.
        telemetry["support_footprint_corridor_seen"] = corridor_seen is not None
    # How many of the grasps kept still hold with the hand ROBUST_ERROR_MM off them; they come first.
    telemetry["support_footprint_robust"] = sum(1 for candidate in candidates if candidate.robust)
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
        if side_approaches:
            # How far off vertical the one the robot would take approaches: a side grasp that won says so here.
            telemetry["support_footprint_top_tilt_deg"] = round(float(np.degrees(np.arccos(
                np.clip(-float(candidates[0].approach[2]), -1.0, 1.0)))), 1)
    return breakdowns, telemetry
