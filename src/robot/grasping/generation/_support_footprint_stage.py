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

Why the body refused what it refused rides along too: ``body_refused`` holds one count per cause, so a
pick that got no grasp says whether the camera's neighbours, a declared body, the part's own fragments
or the support stopped it.

The grasps are the part's own body's (``body_grasps``, lever A of 2026-10-13): what the cameras measured of it, a
ring's wall, a star's lobe, a lying cylinder at its equator, which SFE's convex prism cannot offer. SFE's prism is no
longer asked (the owner, 2026-10-14: the body alone). On the owner's three days from each pick's first pose the prism
added no pick to the body's 178 of 202, the first grasp tried was admitted in 169 picks with it and 175 without, and a
pick ranked in 6.8 s with it and 5.0 s without. What only the prism's search read left with it on 2026-10-16 (the
owner's cleanup): its worker processes, its batched builds, its fine search that waited and its palm switch.
``body_grasps`` in the telemetry says how many the body offered.

Where the calculator asks for them (``inner_grasps``, every cell with a Robotiq hand), the
grasps from inside the part's bores follow (``inner_grasps``): every one after every grasp of the body's outer search,
so a grasp from inside is taken where no outer grasp is. Each says it grips from inside
(``components["support_footprint"]["inner"]``, ``types.grasp_point.grips_from_inside``) and carries what its execution
reads: the jaws' width on the way into the bore, the part's width it is carried as, how deep the fingertips stand and
the bore across; the narrow opening the hand comes down at rides as lever G's ``open_width_mm``.
``inner_grasps``, ``inner_refused`` and ``inner_bores`` in the telemetry say how many it offered, why it offered no
more, and how wide each bore it found was. Off, the stage offers what it offered before, to the bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry import Frame
from src.robot.grasping.planning import GraspPose
from src.robot.grasping.scoring import GraspScoreBreakdown
from src.robot.grasping.types.grasp_point import INNER_GRASP_KEY

from .body_grasps import body_grasps
from .inner_grasps import Bore, InnerGrasp
from .inner_grasps import inner_grasps as find_inner_grasps
from .hand import (
    SIDE_APPROACH_SCORE_WEIGHTS,
    CorridorSeen,
    HandFloor,
    SupportFootprintCandidate,
    SupportFootprintJaw,
)
from .support_prism import DEFAULT_FLOOR_MARGIN_MM

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
    score_weights: tuple[float, float, float, float, float] | None = None,
    floor_margin_mm: float | None = None,
    side_approaches: bool = False,
    corridor_seen: CorridorSeen | None = None,
    seen_envelope: Any = None,
    hand_floor: HandFloor | None = None,
    opens_as_needed: bool = False,
    inner_grasps: bool = False,
    measured_base_mm: np.ndarray | None = None,
    outline_base_mm: np.ndarray | None = None,
    hole_base_mm: np.ndarray | None = None,
) -> tuple[list[GraspScoreBreakdown], dict]:
    """Run SFE and return camera-frame breakdowns plus its telemetry.

    An empty list is a real answer: the reconstruction admitted no grasp that clears the support.
    Refusing beats proposing one that drives a finger into the table.

    ``floor_margin_mm`` of None is the stage's own floor, ``DEFAULT_FLOOR_MARGIN_MM``, read from where it
    is declared.

    ``side_approaches`` ranks the body's grasps by :data:`~src.robot.grasping.generation.hand.SIDE_APPROACH_SCORE_WEIGHTS`
    (the room a lean keeps weighs more, upright nothing); ``corridor_seen`` says what the frame saw, and only a lean
    whose way in a ray of it reached is offered (``body_grasps``). Every cell turns the first on and builds the
    second from the frame.

    ``seen_envelope`` (``scene_obstacles.SeenEnvelope``) holds the boxes the camera world builds of the neighbours: a
    build whose open hand comes nearer to one than the guard keeps is refused under ``seen_fingers``.

    ``hand_floor`` (``hand.HandFloor``) holds the support solids the guard holds under the hand.

    ``opens_as_needed`` plans the body's grasps for a hand that opens only as wide as each span needs
    (``body_grasps``, lever G); off, a hand that opens fully.

    ``inner_grasps`` adds the grasps from inside the part's bores after every outer grasp (``inner_grasps``, the
    module docstring), each planned for a hand told a width; off, the default here, adds nothing and leaves every
    breakdown and every telemetry key as before.

    ``measured_base_mm`` holds the points of ``target_cloud_base_mm`` a camera measured, where the rest stand in the
    part's visual hull (the calculator's ``_with_the_hull``): the body reads where the part is widest from these alone.
    ``None`` reads every point as measured. ``outline_base_mm`` is the part's outline in its colour mask, BASE x and y
    at its top: the body reads on it how square each pad meets the part, and ranks the square grasps first; ``None``
    reads none. ``hole_base_mm`` are the holes of that mask at the part's top: the inner search reads a bore there
    (``inner_grasps.bores``); ``None`` reads the bores off the depth alone.
    """
    # The part the guard holds out (``perceived.target_body_points``): never a piece of something else its mask ran onto.
    from src.robot.safety.planning.perceived import target_body_points  # noqa: PLC0415 (the stage's only use of it)

    target_cloud_base_mm = target_body_points(target_cloud_base_mm)
    if measured_base_mm is not None:
        measured_base_mm = target_body_points(measured_base_mm)
    # Scored by the weights SFE scored by: with side approaches upright weighs nothing and the room a grasp keeps counts
    # more (the owner, 2026-10-01). By the plain weights a vertical grasp of the body 16 mm from a wall outranked SFE's
    # tilt 21 mm from it (2026-10-14).
    body_refused: dict[str, int] = {}
    weights = score_weights if score_weights is not None or not side_approaches else SIDE_APPROACH_SCORE_WEIGHTS
    floor = DEFAULT_FLOOR_MARGIN_MM if floor_margin_mm is None else float(floor_margin_mm)
    candidates = body_grasps(
        target_cloud_base_mm, support_height_mm=support_height_mm, jaw=jaw,
        obstacle_points_base_mm=obstacle_points_base_mm, rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm,
        seen_envelope=seen_envelope, hand_floor=hand_floor, corridor_seen=corridor_seen,
        floor_margin_mm=floor, opens_as_needed=opens_as_needed, max_candidates=max_candidates, refusals=body_refused,
        score_weights=weights, measured_base_mm=measured_base_mm, outline_base_mm=outline_base_mm)
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
    if inner_grasps:
        # The grasps from inside the part's bores, after every outer grasp: taken where no outer grasp is.
        inner_refused: dict[str, int] = {}
        found_bores: list[Bore] = []
        inner = find_inner_grasps(
            target_cloud_base_mm, support_height_mm=support_height_mm, jaw=jaw,
            obstacle_points_base_mm=obstacle_points_base_mm,
            rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm, seen_envelope=seen_envelope,
            hand_floor=hand_floor, corridor_seen=corridor_seen, floor_margin_mm=floor, max_candidates=max_candidates,
            refusals=inner_refused, score_weights=weights, bores_found=found_bores, hole_xy_mm=hole_base_mm)
        kept = [b for b in (_inner_breakdown(grasp, camera_to_base) for grasp in inner) if b is not None]
        breakdowns.extend(kept)
        telemetry["inner_grasps"] = len(inner)
        telemetry["inner_kept"] = len(kept)
        telemetry["inner_refused"] = inner_refused
        telemetry["inner_bores"] = [round(float(bore.diameter_mm), 1) for bore in found_bores]
    return breakdowns, telemetry


def _inner_breakdown(grasp: InnerGrasp, camera_to_base: np.ndarray) -> GraspScoreBreakdown | None:
    """An inner grasp as the calculator's pipeline takes it, in CAMERA, scored by its own score as the body's are; its
    margins say it grips from inside and carry what its execution reads (the module docstring)."""
    candidate = grasp.candidate
    pose = _to_camera_pose(candidate, camera_to_base)
    if pose is None:
        return None
    margins: dict[str, Any] = {
        "contact_angle_deg": float(np.degrees(candidate.contact_angle_rad)),
        "support_clearance_mm": float(candidate.clearance_mm),
        # The jaws' opening where their outer faces meet the bore's wall.
        "grip_width_mm": float(candidate.grip_width_mm),
        "robust": bool(candidate.robust),
        "stage": "inner",
        # It grips from inside: the jaws come down closed or narrow, open onto the wall, and close to let go.
        INNER_GRASP_KEY: True,
        "line_in_width_mm": float(grasp.line_in_width_mm),
        "part_width_mm": float(grasp.part_width_mm),
        "depth_mm": float(grasp.depth_mm),
        "bore_mm": float(grasp.bore_mm),
        # The hand comes down as narrow as this (lever G); the guard holds its fingers at the wall
        # (``types.grasp_point.fingers_held_at``).
        **({"open_width_mm": float(candidate.open_width_mm)} if candidate.open_width_mm is not None else {}),
    }
    return GraspScoreBreakdown(
        pose=pose,
        total_score=pose.score,
        geometric_score=pose.score,
        stability_score=0.0,
        reachability_score=0.0,
        components={"support_footprint": margins},
        metadata={"geometry_stage": "inner", "score_is_undecomposed": True},
    )
