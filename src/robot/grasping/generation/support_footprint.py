"""Support-Footprint Enumeration (SFE): a geometry stage that plans the table clearance.

The shipped silhouette generator anchors a grasp on the visible surface and lets the collision filter
throw it away afterwards. That is the wrong order when the constraint is the support surface: a
filter can only refuse; it cannot move the grasp somewhere legal. On the D5 reference,
``below_table`` is 33.7 % of the calculator's rank-0 candidates and 1 of 949 of SFE's.

SFE reconstructs the target as a convex vertical prism, the visible footprint extruded down onto the
calibrated support plane, then enumerates the grasps that prism admits: the antipodal side-face pairs
are the min-area rectangle's two axes (plus a radial fan when the footprint is round), and for each
axis it solves for the anchor height at which the gripper still clears the table. Everything it uses
is available in a real cell: a masked depth image, the camera pose, the calibrated support plane, the
gripper's own dimensions, and the rest of the depth image as obstacles.

Measured against the D5 reference (n=1365 view-objects with a jaw label, all 270 scenes), top-1, the
rate that decides a pick because the robot takes ``candidates[0]``:

=========  =====  ==============  ========  ==============  ==========
family     n      calculator      SFE       calc + fused    SFE + fused
=========  =====  ==============  ========  ==============  ==========
bin        152    14.5 %          27.6 %    14.5 %          34.2 %
packed     627    27.1 %          59.3 %    27.6 %          75.6 %
pile       178     5.6 %          46.1 %     6.7 %          54.5 %
sparse     408    41.4 %          74.0 %    42.4 %          94.6 %
all        1365   27.2 %          58.5 %    27.7 %          73.9 %
=========  =====  ==============  ========  ==============  ==========

Precision 77.5 % against 8.9 %. Better on every family; ``pile`` by eight times.

This is a geometry stage, not a pick. It does no IK, no reachability, no safety preflight; the
candidates it returns still go through the calculator's scoring, collision, workspace and IK filters
like any other. The numbers above are a ceiling measured offline against ground-truth geometry, not a
delivered pick rate.

The gripper dimensions are taken from :class:`ParallelJawGripperModel` rather than restated here. A
restated copy drifts: an independent one is ~8 mm too lenient toward the table. There is one
measurement and every consumer reads it.

A refused grasp is counted (``refusals``), by cause and by the obstacle set it met: the points a caller
handed as seen, the points it declared, and the target's own low fragments. A pick that gets no grasp
can then say whether a neighbour the camera saw boxed the part in, a declared body did, or the part is
too short for the hand (the owner's cell, 2026-10-01: SFE dropped every obstacle hit uncounted).

Side approaches (``side_approaches``, the owner's "gleichwertig nach Geometrie", 2026-10-01) offer every
tilt the hand fits at, not only the first, and rank them equal by geometry: a tilted grasp wins only
where it keeps more room, vertical wins ties. Only space a depth ray saw counts as clear, because the
camera world takes an occlusion shadow for free space and the guard would judge the approach against
that same free shadow.

Pure and deterministic: numpy, and SciPy's KD-tree where present for the side-approach clearance (a
NumPy scan otherwise, the same answer). BASE millimetres throughout, Z up along the support normal. A part's
search is planned once and run in independent units, one closing line at one height each, merged in the order
they run (:func:`plan_support_footprint`, :func:`run_units`, :func:`rank_found`), so any process may run any of
them and the answer is the same: a cell's worker processes do (``src.robot.grasping.workers``). Where asked
(``batched``), a closing line's builds are made at once, one numpy pass per check (:func:`_build_many`), with the
same answer to the bit.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import dataclasses
from dataclasses import dataclass

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.collision import ParallelJawGripperModel
from src.robot.grasping.geometry.grasp_frame import pose_from_grasp_axes

__all__ = [
    "COARSE",
    "DEFAULT_FLOOR_MARGIN_MM",
    "DEFAULT_MAX_CANDIDATES",
    "FINE",
    "FINE_SEARCH_DEFERRED",
    "REFUSAL_CAUSES",
    "ROBUST_ERROR_MM",
    "ROLLED",
    "SIDE_APPROACH_SCORE_WEIGHTS",
    "CorridorSeen",
    "HandFloor",
    "SfeInputs",
    "SfePlan",
    "SfeRunner",
    "SfeRunnerFailed",
    "SfeUnit",
    "SupportFootprintCandidate",
    "SupportFootprintJaw",
    "SupportPrism",
    "batched_builds_hold",
    "generate_support_footprint_grasps",
    "plan_support_footprint",
    "rank_found",
    "reconstruct_support_prism",
    "robust_to",
    "run_units",
]

_LOG = logging.getLogger(__name__)

_EPS = 1e-9

#: How far above the support a point still counts as the support, millimetres, when the caller does
#: not say. The number the D5 reference measurements above were taken with, in a simulator whose
#: calibration is exact. A real cell's table reads as high as its hand-eye and depth error, and a cell
#: says how high that is in ``safety.planning_world.perceived.plane_clearance_mm``; the config door of
#: ``Scene`` passes the larger of the two.
DEFAULT_FLOOR_MARGIN_MM: Final[float] = 2.0

#: How many candidates :func:`generate_support_footprint_grasps` returns at most when the caller does not say, best
#: first. ``Scene.grasps`` reads it too: a closing axis it is asked for is chosen before the cap, not after it.
DEFAULT_MAX_CANDIDATES: Final[int] = 12

#: The cell of the grid that keeps the hand off the neighbours' points where the boxes the camera world builds of them
#: decide (``seen_envelope``), millimetres, grown by one cell: 4 to 8 mm off a point, under the guard's distance from a
#: box its margin grew round the same point. A choice: the rigid grid's 4 mm.
SEEN_POINTS_CELL_MM: Final[float] = 4.0

#: How close two footprint points have to lie to count as one surface, millimetres, for the fragment
#: trim in :func:`reconstruct_support_prism`. It is the side of a grid cell and two points in touching
#: cells are joined, so points closer than 6 mm are always one surface and pieces more than
#: 6 * 2 * sqrt(2) = 17 mm apart are always two. The 3 mm voxel grid the cloud is thinned to leaves
#: neighbouring samples of one surface at most about 7 mm apart, and the table seen past a 40 mm part's
#: far edge at 45 degrees lies 40 to 46 mm behind it.
_FRAGMENT_LINK_MM: Final[float] = 6.0

#: Approach tilts away from straight-down, in preference order: the first that yields a candidate for
#: an (axis, anchor) wins, so within one anchor a reachable vertical grasp is never traded for a
#: tilted one. Across anchors the final ``found.sort`` by blended score decides, and ``upright`` is
#: only 0.15 of that blend, so a tilted candidate from another anchor can still rank first. With side
#: approaches every tilt that fits is a candidate, and the ladder is the order a tie is broken in.
_TILTS_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
#: Where the coarse grid finds fewer grasps than this, the search runs again on the fine one (the owner, 2026-10-06:
#: "die Orientierung feiner ... den kompletten Freiheitsgrad"). A cylinder 12 mm from a wall: its best grasp scores 0.806
#: on the fine grid against 0.788 on the coarse, at four times the time, which only the parts with few grasps pay.
_FINE_BELOW: Final[int] = 3
#: The fine grid's tilts: every 7.5 degrees.
_FINE_TILTS_DEG: Final[tuple[float, ...]] = tuple(7.5 * k for k in range(13))
#: The fine grid's closing axes beside each face's, turned this far either way, degrees: inside the friction cone
#: (26.6 degrees at the shipped 0.5), whose slack the score weighs, so a face's own axis still ranks first.
_FINE_YAW_DEG: Final[tuple[float, ...]] = (10.0, 20.0)
#: The fine grid's fan for a round footprint: every 15 degrees.
_FINE_RADIAL: Final[int] = 12
#: The fine grid's closing axes tilted out of the horizontal, the hand rolled about its binormal so one finger stands
#: higher than the other, degrees either way (the owner, 2026-10-06: "die geneigte Schliessachse"): a pad on a
#: vertical face then closes off its normal by the roll, inside the friction cone, whose slack the score weighs.
_FINE_ROLL_DEG: Final[tuple[float, ...]] = (10.0, -10.0, 20.0, -20.0)
#: Where along the perpendicular extent to place the closing line: the middle and both thirds.
_FRACS = (0.0, -0.35, 0.35)
#: A footprint whose area fills less of its rectangle than this is round: its closing axes are a fan, and its lines are
#: placed from its centroid (``SupportPrism.middle``). A disc fills pi/4, 0.785; a square 1.
_ROUND_BELOW: Final[float] = 0.88
#: How far inside the part's faces a closing line's span is read where every build on the line is refused for the
#: aperture at once (:func:`_wider_than_the_hand_closes`), millimetres: a rounding in the two ways the span is measured
#: never decides a verdict.
_SPAN_SURE_MM: Final[float] = 1e-6
#: How near its threshold a number may lie that a closing line's builds, made all at once (:func:`_build_many`), decide a
#: verdict by and no candidate carries, millimetres or radians of the friction cone, before that build is made by
#: ``_build`` alone instead: a product over many points rounds its last bit, about 1e-13 mm here, as it likes. Every
#: number a candidate carries is made one build at a time and needs none.
_BATCH_SURE: Final[float] = 1e-6
#: What :func:`generate_support_footprint_grasps` says in ``stages`` where it left the fine search for later
#: (``fine_pass``), under the key ``"fine"``.
FINE_SEARCH_DEFERRED: Final[str] = "deferred"

#: Every cause a refused build is counted under, ``generate_support_footprint_grasps(refusals=...)``. An obstacle
#: refusal names the set it met: ``seen_*`` the points a caller handed as observed (``obstacle_points_base_mm``),
#: ``declared_*`` the points it declared (``rigid_obstacle_points_base_mm``), ``own_fragments`` the target's own low
#: fragments the footprint left out; ``*_fingers`` the closed fingers at the part, ``*_corridor`` the open fingers on
#: their way in. ``table``: the hand reaches within the support clearance. ``span``: the closing line finds no span
#: around the anchor. ``cone``: a contact outside the friction cone. ``prism``: a finger inside the part. ``aperture``:
#: the part is wider than the stroke allows or thinner than the hand closes. ``unseen_corridor``: with side approaches,
#: a tilted corridor through space no depth ray reached.
REFUSAL_CAUSES: Final[tuple[str, ...]] = (
    "seen_fingers", "seen_corridor", "declared_fingers", "declared_corridor", "own_fragments",
    "table", "span", "cone", "prism", "aperture", "unseen_corridor",
)
#: The obstacle-set names :meth:`_ObstacleSet.met` gives, to the cause a finger hit and a corridor hit count under.
_FINGER_CAUSES: Final[Mapping[str, str]] = {"seen": "seen_fingers", "declared": "declared_fingers",
                                            "own": "own_fragments"}
_CORRIDOR_CAUSES: Final[Mapping[str, str]] = {"seen": "seen_corridor", "declared": "declared_corridor",
                                              "own": "own_fragments"}

#: Who says whether a depth ray reached a point: ``(N, 3)`` BASE millimetres in, one ``bool`` per point out, true where
#: the measured depth along that point's pixel ray reaches at least to it. The caller builds it from the frame the part
#: was seen in (the cell-fix plan's contract 5: 5 mm tolerance); SFE never builds one.
CorridorSeen = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HandFloor:
    """How low the hand may come over the solids the guard holds under it: each solid's top face and the guard's distance.

    A support's solid stands over the surface's reading by what the reading stands over its plane, the band and the
    allowance (``support_surfaces.SupportSolid``), and the guard keeps its distance from the box, not from the reading.
    A hand that cleared the reading by the support clearance alone came within the guard's distance of the box: the
    calculator's post-hoc filter refused all 12 grasps of a 30 mm cylinder lying on the mat, whose solid stood 7.3 mm
    over the reading (the grasp bench, 2026-10-06). The caller hands the solids the guard holds; SFE never builds one.
    Each is read as the guard reads it, a box: ``centre_mm``, ``rotation`` (its axes in BASE, the third its up) and
    ``half_extents_mm``.

    ``fingers_to_the_reading``: a finger keeps the distance from each solid's top less its band and its allowance
    (``band_mm``, ``allowance_mm``), the reading and its excess, as the guard holds it
    (``perceived.fingers_to_the_support_reading``, the owner's "wir müssen tiefer gehen" of 2026-10-05); every other
    part of the hand from the whole solid. Ask :meth:`at` and :meth:`under` with ``fingers=True`` for a finger's points.
    """

    solids: tuple[Any, ...]
    distance_mm: float
    fingers_to_the_reading: bool = False
    #: How much nearer than ``distance_mm`` a finger comes to the reading (``scene_obstacles.finger_floor_drop_mm``).
    finger_floor_drop_mm: float = 0.0

    def __post_init__(self) -> None:
        """Read every solid once, as :meth:`at` and :meth:`highest_mm` read them on every call before: its centre, its
        turn, its half extents and its up, a finger's drop under its top (:meth:`_drop`), and the highest top. SFE asks
        the floor of every build, and reading the solids again was 29 % of a boxed-in part's search (2026-10-08); the
        same arrays, read by the same expressions, so the same arithmetic and the same bits."""
        read = []
        for solid in self.solids:
            turn = np.asarray(solid.rotation, dtype=np.float64).reshape(3, 3)
            read.append((np.asarray(solid.centre_mm, dtype=np.float64).reshape(3), turn,
                         np.asarray(solid.half_extents_mm, dtype=np.float64).reshape(3), turn[:, 2],
                         self._drop(solid, True)))
        tops = [float(np.asarray(solid.centre_mm, dtype=np.float64)[2]
                      + (np.abs(np.asarray(solid.rotation, dtype=np.float64).reshape(3, 3))
                         @ np.asarray(solid.half_extents_mm, dtype=np.float64))[2]) for solid in self.solids]
        object.__setattr__(self, "_read", tuple(read))
        object.__setattr__(self, "_highest_mm", max(tops) + float(self.distance_mm) if tops else -math.inf)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickled as what it is built from, and read again where it is unpickled (a worker process,
        ``src.robot.grasping.workers``): what :meth:`__post_init__` reads is views of the solids' arrays, which a pickle
        would turn into copies of another layout."""
        return (type(self), (self.solids, self.distance_mm, self.fingers_to_the_reading, self.finger_floor_drop_mm))

    def _drop(self, solid: Any, fingers: bool) -> float:
        """How far under ``solid``'s top a finger keeps the distance from: its band, its allowance and the finger
        floor's drop where the fingers go to the reading, nothing otherwise."""
        if not (fingers and self.fingers_to_the_reading):
            return 0.0
        return (float(getattr(solid, "band_mm", 0.0) or 0.0) + float(getattr(solid, "allowance_mm", 0.0) or 0.0)
                + max(0.0, float(self.finger_floor_drop_mm)))

    @property
    def highest_mm(self) -> float:
        """No point stands under the floor at or above this height, millimetres; ``-inf`` with no solid."""
        return self._highest_mm  # type: ignore[attr-defined, no-any-return]

    def at(self, xy: np.ndarray, *, fingers: bool = False) -> np.ndarray:
        """The floor over each of ``(N, 2)`` BASE points, millimetres: the highest top face over it and the distance, a
        finger's (``fingers``) the top less the solid's drop (:meth:`_drop`); ``-inf`` where no solid's top face lies
        over it."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(points.shape[0], -math.inf)
        for centre, turn, half, up, finger_drop in self._read:  # type: ignore[attr-defined]
            if up[2] <= 1e-6:
                continue   # a box on its side has no top face to stand over
            offset = points - centre[:2]
            # Where the top face's plane, local z at its half extent, stands over each point.
            z = centre[2] + (half[2] - offset @ up[:2]) / up[2]
            local = np.column_stack([offset, z - centre[2]]) @ turn
            over = np.all(np.abs(local[:, :2]) <= half[:2] + 1e-9, axis=1)
            drop = (finger_drop if fingers else 0.0) / up[2]
            out[over] = np.maximum(out[over], z[over] - drop)
        return out + float(self.distance_mm)

    def under(self, points: np.ndarray, *, fingers: bool = False) -> bool:
        """Whether any of ``(N, 3)`` BASE points stands under the floor, a finger's where ``fingers``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[pts[:, 2] < self.highest_mm]
        return bool(pts.shape[0]) and bool((pts[:, 2] < self.at(pts[:, :2], fingers=fingers)).any())

    def under_many(self, points: np.ndarray, asked: np.ndarray, *, fingers: bool = False) -> np.ndarray:
        """:meth:`under` for many builds at once: ``(N, P, 3)`` BASE points and which of them are asked (``asked``,
        ``(N, P)``) in, one ``int8`` per build out: 0 where none stands under the floor, 1 where one does, 2 where a
        point lies within :data:`_BATCH_SURE` of a solid's top face or of the edge of the face it stands over, which
        only :meth:`under` decides.

        :meth:`at`'s arithmetic, on every point asked at once: its two products may round their last bit another way
        over many points than over one build's (``support_footprint._build_many`` asks every build of a closing line at
        once), and the band covers that.
        """
        out = np.zeros(points.shape[0], dtype=np.int8)
        asked = asked & (points[:, :, 2] < self.highest_mm)
        owner = np.nonzero(asked)[0]
        pts = points[asked]
        if not pts.shape[0]:
            return out
        floor = np.full(pts.shape[0], -math.inf)
        unsure = np.zeros(pts.shape[0], dtype=bool)
        for centre, turn, half, up, finger_drop in self._read:  # type: ignore[attr-defined]
            if up[2] <= 1e-6:
                continue   # a box on its side has no top face to stand over
            offset = pts[:, :2] - centre[:2]
            z = centre[2] + (half[2] - offset @ up[:2]) / up[2]
            local = np.column_stack([offset, z - centre[2]]) @ turn
            edge = np.abs(local[:, :2]) - (half[:2] + 1e-9)
            over = np.all(edge <= 0.0, axis=1)
            unsure |= np.any(np.abs(edge) < _BATCH_SURE, axis=1)
            drop = (finger_drop if fingers else 0.0) / up[2]
            floor[over] = np.maximum(floor[over], z[over] - drop)
        gap = pts[:, 2] - (floor + float(self.distance_mm))
        unsure |= np.abs(gap) < _BATCH_SURE
        out[owner[unsure]] = 2
        out[owner[(gap < 0.0) & ~unsure]] = 1
        return out

#: How the five margins are blended with side approaches on, in the order of ``_SCORE_WEIGHTS``: friction-cone slack
#: 0.35, aperture left 0.20, clearance 0.35, upright 0, centred 0.10. Upright weighs nothing, because the owner wants the
#: approaches equal by geometry (2026-10-01); clearance takes its weight and is the room the grasp keeps, the smaller of
#: the fingertip's height over the support and the open corridor's least distance to an obstacle point. Prototyped on a
#: 40 mm cylinder beside a wall (``R_SCOPE/b_wall_prototype.out``, the cell-fix plan's evidence): no wall, rank 0
#: vertical; a wall 12 mm off, rank 0 tilted 30 degrees away from it with a vertical candidate still listed.
SIDE_APPROACH_SCORE_WEIGHTS: Final[tuple[float, float, float, float, float]] = (0.35, 0.20, 0.35, 0.0, 0.10)
#: The room that scores the full clearance term, millimetres: the table term's 20 mm, the guard's 5 mm four times over.
_CLEARANCE_FULL_MM: Final[float] = 20.0
#: How far across the table the hand may stand off a grasp it is sent to, millimetres, for the grasp to count as robust
#: (:func:`robust_to`) and be ranked first. The owner, 2026-10-09: 3 to 5 mm is what a real cell gets; the two looks of
#: the one recorded pick seen from two poses (2026-10-07) lay 5.4 mm apart at the median, about 3.8 mm each.
ROBUST_ERROR_MM: Final[float] = 4.0
#: A candidate tilted this far off vertical or more is offered, with side approaches, only through space a depth ray saw.
_SEEN_FROM_TILT_DEG: Final[float] = 15.0
#: Scores this near the best of a run are equal by geometry, and the run is ranked vertical first. A face's normal read
#: off a noisy footprint moves the contact angle by hundredths of a degree from one anchor to the next, 0.0008 of score
#: on the straight-down cube of ``tests/test_a_tilted_view_grasps_on_the_part.py``; rounded to 0.001 the two sat either
#: side of a rounding edge, and a 15 degree tilt took the rank from the vertical grasp.
_TIE_SCORE: Final[float] = 0.001


@dataclass(frozen=True, slots=True)
class SupportFootprintJaw:
    """The gripper, as SFE needs to reason about it. Build it with :meth:`from_robot_config` from a
    loaded tree, or with :meth:`from_model` from an envelope and a stroke.

    ``finger_ahead_mm`` is the reach past the grasp centre toward the support: the single number
    that decides a table verdict. ``pad_ahead_mm`` and ``pad_behind_mm`` describe the contact patch,
    which is asymmetric about the grasp centre because the hardware is.
    """

    aperture_mm: float = 85.0
    min_width_mm: float = 5.0
    finger_ahead_mm: float = 28.72
    finger_behind_mm: float = 39.98
    finger_width_mm: float = 27.0
    finger_thickness_mm: float = 31.35
    pad_ahead_mm: float = 23.61
    pad_behind_mm: float = 14.39
    friction: float = 0.5
    table_clearance_mm: float = 5.0
    width_safety_mm: float = 2.0
    #: The palm. It sits behind the fingers, so on a straight-down grasp it is nowhere near the
    #: table; it is 70 mm wide against the fingers' 27, so as the approach tilts and the binormal
    #: swings toward vertical the palm becomes the lowest part of the gripper. The collision
    #: envelope always carries it; SFE reasons about it only under ``palm_aware``, which defaults
    #: off at every layer, with the measurement behind that default beside
    #: ``support_footprint_palm_aware`` in ``calculator.py``.
    palm_width_mm: float = 70.0
    palm_depth_mm: float = 35.0

    @classmethod
    def from_model(
        cls,
        model: ParallelJawGripperModel | None = None,
        *,
        aperture_mm: float = 85.0,
        min_width_mm: float = 5.0,
        friction: float = 0.5,
        table_clearance_mm: float = 5.0,
        width_safety_mm: float = 2.0,
    ) -> "SupportFootprintJaw":
        """Derive the planning dimensions from the collision envelope, so the two cannot drift.

        ``finger_behind_mm`` takes the model's ``finger_length_mm`` (39.98) rather than the measured
        33.37 mm reach: it only lengthens the swept approach corridor, so the larger number is the
        conservative one.

        ⛔ **`aperture_mm` AND `min_width_mm` ARE NOT DERIVED, AND EVERY CALLER MUST PASS THEM.**
        The collision envelope describes the fingers, not the travel, so the stroke cannot come from
        ``model`` and these two keywords fall back to a 2F-85's 85.0 / 5.0. That default was the one
        dimension the sentence above does not cover: between 2026-08-14 and 2026-09-10 the pick path
        omitted them and planned a 49.99 mm Hand-E as though it opened 85.0, while every other
        number in the jaw came out correct. Read ``robot.gripper.max_width_mm``; do not let this
        default stand in for a hand.
        """
        m = model or ParallelJawGripperModel()
        return cls(
            aperture_mm=float(aperture_mm),
            min_width_mm=float(min_width_mm),
            finger_ahead_mm=float(m.fingertip_depth_mm),
            finger_behind_mm=float(m.finger_length_mm),
            finger_width_mm=float(m.finger_width_mm),
            finger_thickness_mm=float(m.finger_thickness_mm),
            pad_ahead_mm=float(m.pad_ahead_mm),
            pad_behind_mm=float(m.pad_length_mm - m.pad_ahead_mm),
            friction=float(friction),
            table_clearance_mm=float(table_clearance_mm),
            width_safety_mm=float(width_safety_mm),
            palm_width_mm=float(m.palm_width_mm),
            palm_depth_mm=float(m.palm_depth_mm),
        )

    @classmethod
    def from_robot_config(cls, robot_config: Any) -> "SupportFootprintJaw":
        """The jaw a loaded tree describes, built from the numbers the cell's pick path builds it from.

        The fingers come from ``grasping.gripper_geometry.parallel_jaw`` through the same collision
        envelope ``build_gripper_geometry`` hands the calculator, field for field. The stroke comes from
        ``robot.gripper``, which the cell hands the calculator as its grip widths, and the table
        clearance from ``grasping.support.min_clearance_mm``, which the pick loop hands it per call.

        Until 2026-09-23 ``Scene.from_robot_config`` built its jaw from the stroke alone, so every
        finger number fell back to ``ParallelJawGripperModel()``, which is a 2F-85. MEASURED on the
        ``hande`` profile: a vertical approach then needs the anchor 62.4 mm above the support instead
        of 25.9 mm, so a 40 mm cube got only 90-degree side approaches 36 mm up, where the 75 mm Hand-E
        housing reaches below the table, and a 30 mm part got no grasp at all.

        A suction hand is refused: this stage plans a parallel jaw, and planning a suction cup's cell as
        a 2F-85 is the substitution this method exists to end.
        """
        geometry = robot_config.grasping.gripper_geometry
        if str(geometry.kind) != "parallel_jaw":
            raise ValueError(
                f"grasping.gripper_geometry.kind is {geometry.kind!r}, and support-footprint grasps are "
                "planned for a parallel jaw: planning this hand as a jaw would put a gripper it does not "
                "have into every candidate")
        j = geometry.parallel_jaw
        model = ParallelJawGripperModel(
            finger_length_mm=j.finger_length_mm,
            finger_thickness_mm=j.finger_thickness_mm,
            finger_width_mm=j.finger_width_mm,
            finger_pad_overlap_mm=j.finger_pad_overlap_mm,
            fingertip_depth_mm=j.fingertip_depth_mm,
            pad_length_mm=j.pad_length_mm,
            pad_ahead_mm=j.pad_ahead_mm,
            palm_depth_mm=j.palm_depth_mm,
            palm_width_mm=j.palm_width_mm,
            outer_margin_mm=geometry.outer_margin_mm,
        )
        # Where the guard holds the jaw: past the TCP by as much as the registry and the plates put it
        # (``planning.hand.hand_past_the_tcp_mm``), as the cell's calculator plans it (``build_gripper_geometry``).
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm  # noqa: PLC0415 (config side only)

        past = hand_past_the_tcp_mm(robot_config)
        if past > 0.0:
            from dataclasses import replace as _replace  # noqa: PLC0415

            model = _replace(model, finger_length_mm=float(model.finger_length_mm) - past,
                             fingertip_depth_mm=float(model.fingertip_depth_mm) + past,
                             pad_ahead_mm=float(model.pad_ahead_mm) + past)
        return cls.from_model(
            model,
            aperture_mm=float(robot_config.gripper.max_width_mm),
            min_width_mm=float(robot_config.gripper.min_width_mm),
            table_clearance_mm=float(robot_config.grasping.support.min_clearance_mm),
        )

    @property
    def finger_reach_mm(self) -> float:
        return self.finger_ahead_mm + self.finger_behind_mm

    @property
    def cone_rad(self) -> float:
        return float(np.arctan(self.friction))


@dataclass(frozen=True, slots=True)
class SupportFootprintCandidate:
    """One enumerated grasp, in BASE millimetres. :meth:`pose` is where the tool goes for it."""

    score: float
    position_mm: np.ndarray
    approach: np.ndarray
    closing_axis: np.ndarray
    grip_width_mm: float
    #: Worst contact-normal deviation from the closing axis, radians. Inside the friction cone.
    contact_angle_rad: float
    #: Lowest point of the closed gripper above the support, millimetres.
    clearance_mm: float
    #: Whether the grasp still holds with the hand :data:`ROBUST_ERROR_MM` off it across the table (:func:`robust_to`):
    #: set where the search ranks it (:func:`rank_found`), which puts these first; ``False`` until then.
    robust: bool = False

    def pose(self) -> Pose:
        """Where the tool goes for this grasp: the BASE pose ``Robot.pick`` takes, with
        ``grip_width_mm``.

        The pose sits at ``position_mm``. Its +Z is ``approach``, the direction the tool travels
        toward the part (straight down is ``(0, 0, -1)``), and its +X is ``closing_axis``, the line
        the pads close across::

            best = Scene.from_cloud(cloud_base_mm, support_height_mm=0.0).grasps().best
            if best is not None:
                robot.pick(best.pose(), best.grip_width_mm)

        ``Robot.pick`` backs off along that +Z to its standoff, runs a line in to ``position_mm``
        and closes to ``grip_width_mm`` less its squeeze. The frame is BASE because this stage
        plans in BASE and in nothing else.
        """
        return pose_from_grasp_axes(self.position_mm, approach=self.approach, closing_axis=self.closing_axis,
                                    frame=Frame.BASE)


# --------------------------------------------------------------------------- planar helpers


def _hull2d(points: np.ndarray) -> np.ndarray:
    """Monotone chain. Deterministic, dependency-free, O(n log n)."""
    p = np.unique(np.round(points, 3), axis=0)
    if p.shape[0] < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts: np.ndarray) -> list:
        out: list = []
        for q in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    return np.asarray(half(p)[:-1] + half(p[::-1])[:-1], dtype=np.float64)


def _min_area_rect(hull: np.ndarray) -> tuple:
    """Rotating calipers. Returns ``(u, v, ext_u, ext_v, centre)`` with ``u`` the short axis."""
    best = None
    n = hull.shape[0]
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        e = edge / length
        f = np.array([-e[1], e[0]])
        a, b = hull @ e, hull @ f
        w, h = float(a.max() - a.min()), float(b.max() - b.min())
        if best is None or w * h < best[0]:
            centre = ((a.max() + a.min()) / 2.0) * e + ((b.max() + b.min()) / 2.0) * f
            best = (w * h, e, f, w, h, centre)
    if best is None:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.0, hull.mean(axis=0)
    _, e, f, w, h, centre = best
    return (e, f, w, h, centre) if w <= h else (f, e, h, w, centre)


def _poly_centroid(hull: np.ndarray) -> np.ndarray:
    """The area centroid of the convex polygon ``hull`` (``(N, 2)``, in order); its vertices' mean where it has no area."""
    x, y = hull[:, 0], hull[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = float(cross.sum()) / 2.0
    if abs(area) < 1e-9:
        return hull.mean(axis=0)
    return np.array([float(((x + xn) * cross).sum()), float(((y + yn) * cross).sum())]) / (6.0 * area)


def _poly_area(hull: np.ndarray) -> float:
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``np.cross`` of two 3-vectors: the same three products and differences in the same order, so the same bits, a
    signed zero included, without numpy's general path, which was a quarter of a boxed-in part's search (2026-10-08)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / max(float(np.linalg.norm(axis)), _EPS)
    return (vector * np.cos(angle) + _cross3(k, vector) * np.sin(angle)
            + k * (k @ vector) * (1.0 - np.cos(angle)))


# --------------------------------------------------------------------------- the reconstruction


class SupportPrism:
    """The visible footprint extruded onto the support plane: a convex vertical prism.

    It answers the two questions a grasp needs, where a line enters and leaves the solid and which
    way the surface faces there, in closed form, on an estimate built only from a depth image and
    the calibrated support height.
    """

    __slots__ = ("hull", "normals", "offsets", "z0", "z1", "centre", "u", "v",
                 "ext_u", "ext_v", "roundness", "inflate", "trimmed", "middle")

    def __init__(self, hull2d: np.ndarray, z0: float, z1: float) -> None:
        self.hull = hull2d
        #: Points the floor let through that the footprint left out as low fragments, ``(N, 3)`` BASE
        #: millimetres. Not forgotten: the generator keeps the fingers off them as declared obstacles.
        self.trimmed = np.zeros((0, 3), dtype=np.float64)
        n = hull2d.shape[0]
        nx, ny, d = [], [], []
        for i in range(n):
            edge = hull2d[(i + 1) % n] - hull2d[i]
            length = float(np.linalg.norm(edge))
            if length < 1e-6:
                continue
            out = np.array([edge[1], -edge[0]]) / length
            nx.append(out[0])
            ny.append(out[1])
            d.append(float(out @ hull2d[i]))
        self.normals = np.stack([np.asarray(nx), np.asarray(ny)], axis=1)
        self.offsets = np.asarray(d)
        # Orient outward: the centroid must be inside every half-plane.
        flip = (self.normals @ hull2d.mean(axis=0)) > self.offsets
        self.normals[flip] *= -1.0
        self.offsets[flip] *= -1.0
        self.z0, self.z1 = float(z0), float(z1)
        self.inflate = 0.0
        self.u, self.v, self.ext_u, self.ext_v, self.centre = _min_area_rect(hull2d)
        self.roundness = _poly_area(hull2d) / max(self.ext_u * self.ext_v, 1e-9)
        #: Where a closing line across the part is placed from: the rectangle's centre, and for a round footprint the
        #: footprint's own centroid. A round footprint's rectangle turns with the noise on its outline, and its centre
        #: with it: a 40 mm cylinder's lay 0.64 mm off the cylinder's axis, and the grasp closing through it came 0.64 mm
        #: nearer a cube 16 mm off, past the room the guard keeps (2026-10-09).
        self.middle = _poly_centroid(hull2d) if self.roundness < _ROUND_BELOW else self.centre

    def line_span(self, origin: np.ndarray, direction: np.ndarray,
                  margin_mm: float = 0.0) -> tuple[float, float] | None:
        """Parameters at which a line enters and leaves, or ``None`` when it misses."""
        lo, hi = -np.inf, np.inf
        a = self.normals @ direction[:2]
        b = self.offsets + margin_mm - self.normals @ origin[:2]
        par = np.abs(a) < 1e-12
        if np.any(par & (b < 0.0)):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(par, 0.0, b / np.where(par, 1.0, a))
        pos, neg = (~par) & (a > 0), (~par) & (a < 0)
        if pos.any():
            hi = min(hi, float(t[pos].min()))
        if neg.any():
            lo = max(lo, float(t[neg].max()))
        dz = float(direction[2])
        if abs(dz) < 1e-12:
            if not (self.z0 - margin_mm <= origin[2] <= self.z1 + margin_mm):
                return None
        else:
            t1 = (self.z0 - margin_mm - origin[2]) / dz
            t2 = (self.z1 + margin_mm - origin[2]) / dz
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        return None if lo > hi else (float(lo), float(hi))

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Outward normal of the nearest face: a side wall, the top, or the footprint."""
        slack = self.offsets - self.normals @ point[:2]
        i = int(np.argmin(np.abs(slack)))
        best = abs(float(slack[i]))
        normal = np.array([self.normals[i, 0], self.normals[i, 1], 0.0])
        if abs(point[2] - self.z1) < best:
            best, normal = abs(point[2] - self.z1), np.array([0.0, 0.0, 1.0])
        if abs(point[2] - self.z0) < best:
            normal = np.array([0.0, 0.0, -1.0])
        return normal

    def contains(self, points: np.ndarray, margin_mm: float = 0.0) -> np.ndarray:
        inside = np.all((points[:, :2] @ self.normals.T) <= self.offsets + margin_mm, axis=1)
        return inside & (points[:, 2] >= self.z0 - margin_mm) & (points[:, 2] <= self.z1 + margin_mm)


#: How the five measured margins are blended into one rank, in the order
#: ``(cone_slack, width_margin, table_margin, upright, centred)``. Nothing here is a tuned constant:
#: each term is a fraction of a physical budget the plan did not spend.
#:
#: Only the first term measurably separates, and the other four may dilute it. `cone_slack` is
#: `1 - contact_angle/cone`, and on 414 `sfe_fused` candidate lists the contact angle alone orders
#: valid before invalid at within-list AUC 0.6907 against the full blend's 0.5698, and reaches top-1
#: 86.23 % against the shipped blend's 83.09 %: 26 lists rescued against 13 lost, +3.14 pp, McNemar
#: chi2 3.69 on 39 discordant lists, p ~ 0.055. That denominator is lists with >=2 candidates and at
#: least one valid, which is 24 % of the graspable units and selected for already having a valid
#: candidate to find.
#:
#: On the population that ships, ranking moves +0.29 pp. Two rungs, three orders over the same
#: candidates, 270 scenes / 1,716 jaw-graspable units: shipped 51.75 %, angle at 0.70 52.04 %, angle
#: alone 52.04 %; paired per unit that is +2 and +3 units, McNemar chi2 0.03 and 0.08.
#:
#: Ranking is worth at most five and a half points here: the share of graspable units where any
#: candidate is valid, the ceiling a perfect ranker reaches, is 57.46 % against top-1's 51.75 %. The
#: other 42.5 % have no valid candidate at all, which is generation and not order. `score_weights`
#: keeps that measurement runnable and stays at None.
_SCORE_WEIGHTS: Final[tuple[float, float, float, float, float]] = (0.35, 0.20, 0.20, 0.15, 0.10)


def _trim_low_fragments(points: np.ndarray, support_height_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a floor-filtered cloud into the footprint and the low fragments lying apart from it.

    The pieces are joined in the XY plane at ``_FRAGMENT_LINK_MM``. The body is the piece with the
    most points. Another piece is a low fragment when it never rises to half the body's height above
    the support: the table seen past an edge, a neighbour's foot bled into the mask. A piece that does
    rise that far is kept in the footprint and taken for the same top face split by a depth hole:
    leaving it out would let a finger into the hole, which is the inside of the part. Returns
    ``(kept, trimmed)``; with one piece nothing is trimmed.
    """
    cells = np.floor(points[:, :2] / _FRAGMENT_LINK_MM).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    count = int(unique_cells.shape[0])
    if count < 2:
        return points, points[:0]
    index = {cell: i for i, cell in enumerate(map(tuple, unique_cells.tolist()))}
    parent = list(range(count))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (cx, cy) in enumerate(unique_cells.tolist()):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            j = index.get((cx + dx, cy + dy))
            if j is not None:
                a, b = root(i), root(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    labels = np.asarray([root(i) for i in range(count)], dtype=np.int64)[inverse]
    sizes = np.bincount(labels, minlength=count)
    body = int(np.argmax(sizes))
    if int(sizes[body]) == points.shape[0]:
        return points, points[:0]
    body_top = float(np.percentile(points[labels == body, 2], 98.0))
    low_line = support_height_mm + 0.5 * (body_top - support_height_mm)
    tops = np.full(count, -np.inf)
    np.maximum.at(tops, labels, points[:, 2])
    low = (np.arange(count) != body) & (tops < low_line)
    trim = low[labels]
    return points[~trim], points[trim]


def reconstruct_support_prism(
    cloud_base_mm: np.ndarray,
    support_height_mm: float,
    *,
    voxel_mm: float = 3.0,
    top_percentile: float = 98.0,
    floor_margin_mm: float = DEFAULT_FLOOR_MARGIN_MM,
    inflate_mm: float = 0.0,
) -> SupportPrism | None:
    """The support-extruded prism from a masked target cloud in BASE. The whole perception step.

    ``inflate_mm`` pushes every face outward. A depth image loses the grazing rim of a curved
    surface, so the measured footprint is systematically smaller than the object and every
    consequence of that error is one-sided (an under-estimated span, a finger that clips a flank on
    the way in). Inflating biases the estimate in the safe direction instead of leaving it unbiased.

    ``floor_margin_mm`` is how far above the support a point still counts as the support.

    The footprint is the convex hull of what survives the floor, and a convex hull has no outlier
    rejection: a band of points anywhere pulls a face out to it. MEASURED 2026-09-23 on a ray-cast of
    a 40 mm cube seen at 45 degrees with its far edge misregistered by 1 px and the table read 3 mm
    high: 63 table points, 0.8 % of the cloud and 2.0 to 3.7 mm above the support, lay up to 45.6 mm
    behind the part and carried the hull's far face out to 65.6 mm from the part centre, against a
    half-size of 20. So low fragments that lie apart from the body are left out of the hull first (see
    :func:`_trim_low_fragments`) and kept on the prism as ``trimmed``. On that scene with a 2 px
    misregistration the far face then sits 23.9 to 24.7 mm from the centre in 5 of 5 noise seeds.
    """
    p = np.asarray(cloud_base_mm, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    p = p[p[:, 2] > support_height_mm + floor_margin_mm]
    if p.shape[0] < 20:
        return None
    _, idx = np.unique(np.round(p / voxel_mm).astype(np.int64), axis=0, return_index=True)
    p = p[np.sort(idx)]
    if p.shape[0] < 12:
        return None
    p, trimmed = _trim_low_fragments(p, support_height_mm)
    if p.shape[0] < 12:
        return None
    top_z = float(np.percentile(p[:, 2], top_percentile))
    if top_z <= support_height_mm + 3.0:
        return None
    hull = _hull2d(p[:, :2])
    if hull.shape[0] < 3:
        return None
    prism = SupportPrism(hull, support_height_mm, top_z)
    prism.trimmed = trimmed
    if inflate_mm:
        prism.offsets = prism.offsets + float(inflate_mm)
        prism.inflate = float(inflate_mm)
    return prism


def closing_axes(prism: SupportPrism, *, radial: int = 6,
                 yaw_offsets_deg: Sequence[float] = ()) -> list[np.ndarray]:
    """The closing directions the reconstruction admits, enumerated rather than searched.

    A vertical prism's antipodal side-face pairs are the min-area rectangle's two axes; a round
    footprint admits every radial direction, so it gets a fan of ``radial`` directions on top of
    those two. ``yaw_offsets_deg`` turns each of the two axes that far either way as well: a jaw
    closing a little off a face's normal still holds inside the friction cone, which ``_build``
    asks, and may fit between neighbours where the face's own axis does not.
    """
    axes = [np.array([prism.u[0], prism.u[1], 0.0]), np.array([prism.v[0], prism.v[1], 0.0])]
    for base in (prism.u, prism.v):
        across = np.array([-base[1], base[0]])
        for offset in yaw_offsets_deg:
            for turned in (np.radians(offset), -np.radians(offset)):
                d = np.cos(turned) * base + np.sin(turned) * across
                axes.append(np.array([d[0], d[1], 0.0]))
    if prism.roundness < _ROUND_BELOW:
        # The fan stands in the base frame, the table's x and y among it. A round footprint's rectangle has no direction
        # of its own: its axes turn with the noise on the outline, and a fan turned with them lost the one closing
        # direction that fits between two neighbours (a 40 mm cylinder with cubes 18 mm off on both sides, its outline
        # read in its visual hull, 2026-10-09: the fan turned 15 degrees and offered no grasp closing toward them).
        for k in range(radial):
            angle = np.pi * k / radial
            axes.append(np.array([np.cos(angle), np.sin(angle), 0.0]))
    return axes


# --------------------------------------------------------------------------- obstacles


class _Obstacles:
    """Occupancy grid over the non-target points, dilated once by one 12 mm cell.

    Built once per object; a query is one raveled ``searchsorted``. The calculator already builds
    this same cloud today and only hands it to a post-hoc filter.
    """

    __slots__ = ("cell", "keys", "origin", "dims", "reach")

    def __init__(self, points: np.ndarray | None,
                 cell_mm: float = 12.0, margin_mm: float = 12.0) -> None:
        self.cell = cell_mm
        self.keys: np.ndarray | None = None
        self.reach = 0
        if points is None or np.size(points) == 0:
            return
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if p.shape[0] == 0:
            return
        r = int(np.ceil(margin_mm / cell_mm))
        self.reach = r
        self.origin = p.min(axis=0) - cell_mm * (r + 1)
        k = np.unique(np.floor((p - self.origin) / cell_mm).astype(np.int64), axis=0)
        self.dims = k.max(axis=0) + r + 2
        offs = np.array([(dx, dy, dz) for dx in range(-r, r + 1)
                         for dy in range(-r, r + 1) for dz in range(-r, r + 1)], dtype=np.int64)
        big = (k[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        big = big[(big >= 0).all(axis=1)]
        self.keys = np.unique(big[:, 0] * self.dims[1] * self.dims[2]
                              + big[:, 1] * self.dims[2] + big[:, 2])

    def hits(self, points: np.ndarray) -> bool:
        if self.keys is None:
            return False
        k = np.floor((points - self.origin) / self.cell).astype(np.int64)
        ok = (k >= 0).all(axis=1) & (k < self.dims).all(axis=1)
        if not ok.any():
            return False
        k = k[ok]
        flat = k[:, 0] * self.dims[1] * self.dims[2] + k[:, 1] * self.dims[2] + k[:, 2]
        idx = np.clip(np.searchsorted(self.keys, flat), 0, self.keys.size - 1)
        return bool((self.keys[idx] == flat).any())

    def keys_of(self, points: np.ndarray | None) -> np.ndarray:
        """The keys of the cells ``points`` fill in this grid, dilated as the grid dilates: ``points`` are a part of what
        it was built from, so the keys of the parts together are exactly :attr:`keys`. Empty where there are none."""
        none = np.zeros(0, dtype=np.int64)
        if self.keys is None or points is None or np.size(points) == 0:
            return none
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if p.shape[0] == 0:
            return none
        r = self.reach
        k = np.unique(np.floor((p - self.origin) / self.cell).astype(np.int64), axis=0)
        offs = np.array([(dx, dy, dz) for dx in range(-r, r + 1)
                         for dy in range(-r, r + 1) for dz in range(-r, r + 1)], dtype=np.int64)
        big = (k[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        big = big[(big >= 0).all(axis=1) & (big < self.dims).all(axis=1)]
        return np.unique(big[:, 0] * self.dims[1] * self.dims[2] + big[:, 1] * self.dims[2] + big[:, 2])

    def hits_among(self, points: np.ndarray, keys: np.ndarray) -> bool:
        """:meth:`hits`, against ``keys`` (from :meth:`keys_of`) instead of every cell of the grid."""
        if self.keys is None or keys.size == 0:
            return False
        k = np.floor((points - self.origin) / self.cell).astype(np.int64)
        ok = (k >= 0).all(axis=1) & (k < self.dims).all(axis=1)
        if not ok.any():
            return False
        k = k[ok]
        flat = k[:, 0] * self.dims[1] * self.dims[2] + k[:, 1] * self.dims[2] + k[:, 2]
        idx = np.clip(np.searchsorted(keys, flat), 0, keys.size - 1)
        return bool((keys[idx] == flat).any())

    def hits_many(self, points: np.ndarray, owner: np.ndarray, count: int,
                  keys: "np.ndarray | None" = None) -> np.ndarray:
        """:meth:`hits` (against ``keys`` where given, :meth:`hits_among`) for ``count`` builds at once: ``(M, 3)``
        points and the build each belongs to (``owner``) in, one ``bool`` per build out. The same cells of the same
        points, so the same answer."""
        out = np.zeros(count, dtype=bool)
        table = self.keys if keys is None else keys
        if self.keys is None or table is None or table.size == 0 or not points.shape[0]:
            return out
        k = np.floor((points - self.origin) / self.cell).astype(np.int64)
        ok = (k >= 0).all(axis=1) & (k < self.dims).all(axis=1)
        k, owner = k[ok], owner[ok]
        if not k.shape[0]:
            return out
        flat = k[:, 0] * self.dims[1] * self.dims[2] + k[:, 1] * self.dims[2] + k[:, 2]
        idx = np.clip(np.searchsorted(table, flat), 0, table.size - 1)
        out[owner[table[idx] == flat]] = True
        return out


class _ObstacleSet:
    """Two obstacle grids, because two kinds of obstacle deserve two dilations.

    ``_Obstacles`` dilates every point by 12 mm on a 12 mm grid, which is right for a sparse depth
    cloud, where the gap between samples is real uncertainty about where the surface is. It is wrong
    for geometry that is declared rather than observed. A container wall is exactly known, and
    dilating it pushes the forbidden zone 12-24 mm into the bin: on the bin-heavy slice, feeding
    walls through the sparse grid moves candidate precision 49.2 -> 98.8 % while 13 objects lose
    every candidate and 3 valid grasps die for 0 rescued.

    So declared geometry gets its own grid at 4 mm with no dilation, and it still reaches the
    generator: a filter can refuse a grasp but cannot move it somewhere legal, and avoiding the wall
    while choosing the grasp is the point.

    The rigid grid holds two sets in one, the caller's declared points and the target's own low
    fragments, and stays one grid so that a verdict is the one it always was. Where refusals are
    counted, :meth:`label` marks which of its cells hold which set, so a hit can say whom it met.
    """

    __slots__ = ("sparse", "rigid", "declared", "own")

    def __init__(self, sparse: _Obstacles, rigid: _Obstacles) -> None:
        self.sparse = sparse
        self.rigid = rigid
        self.declared: np.ndarray | None = None
        self.own: np.ndarray | None = None

    def hits(self, points: np.ndarray) -> bool:
        return self.sparse.hits(points) or self.rigid.hits(points)

    def label(self, declared: np.ndarray | None, own: np.ndarray | None) -> None:
        """Mark the rigid grid's cells by the set that fills them: ``declared`` and ``own`` are the two parts it was
        built from, so their cells together are all of its cells."""
        self.declared = self.rigid.keys_of(declared)
        self.own = self.rigid.keys_of(own)

    def met(self, points: np.ndarray) -> tuple[str, ...]:
        """Which sets ``points`` meet, of ``seen``, ``declared`` and ``own`` in that order, after :meth:`label`.

        Empty exactly where :meth:`hits` is false: the rigid grid's cells are the declared cells and the own cells
        together, because both were cut from one grid with one origin."""
        assert self.declared is not None and self.own is not None, "label() the set before asking whom a hit met"
        met: list[str] = []
        if self.sparse.hits(points):
            met.append("seen")
        if self.rigid.hits_among(points, self.declared):
            met.append("declared")
        if self.rigid.hits_among(points, self.own):
            met.append("own")
        return tuple(met)

    def refuse(self, points: np.ndarray, refusals: dict[str, int] | None, causes: Mapping[str, str]) -> bool:
        """Whether ``points`` meet an obstacle; where ``refusals`` counts, each set met counts once under its cause."""
        if refusals is None:
            return self.hits(points)
        met = self.met(points)
        for name in met:
            refusals[causes[name]] = refusals.get(causes[name], 0) + 1
        return bool(met)

    def met_many(self, points: np.ndarray, asked: np.ndarray, *,
                 counting: bool) -> "tuple[np.ndarray, np.ndarray | None]":
        """:meth:`refuse` for many builds at once, counting nothing yet: ``(N, P, 3)`` points and which of them are
        asked (``(N, P)``) in; whether each build meets an obstacle out and, where ``counting``, which sets it met,
        ``(N, 3)`` in :meth:`met`'s order (``seen``, ``declared``, ``own``), else ``None``. :meth:`hits` where nobody
        counts and :meth:`met` where somebody does, as :meth:`refuse` asks them: the same cells, the same answer."""
        count = points.shape[0]
        owner = np.nonzero(asked)[0]
        flat = points[asked]
        seen = self.sparse.hits_many(flat, owner, count)
        if not counting:
            return seen | self.rigid.hits_many(flat, owner, count), None
        assert self.declared is not None and self.own is not None, "label() the set before asking whom a hit met"
        sets = np.column_stack([seen, self.rigid.hits_many(flat, owner, count, self.declared),
                                self.rigid.hits_many(flat, owner, count, self.own)])
        return np.asarray(sets.any(axis=1), dtype=bool), sets


class _Room:
    """How far a corridor stays from the nearest obstacle point, up to the room that scores in full.

    Built once per object over every obstacle point SFE holds, seen, declared and the target's own
    fragments, for the side-approach clearance. SciPy's KD-tree where it imports, else a scan of the
    points near the corridor, which gives the same distance.
    """

    __slots__ = ("points", "tree")

    def __init__(self, *clouds: np.ndarray | None) -> None:
        parts = []
        for cloud in clouds:
            if cloud is None or np.size(cloud) == 0:
                continue
            p = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
            parts.append(p[np.isfinite(p).all(axis=1)])
        joined = np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float64)
        self.points: np.ndarray | None = joined if joined.shape[0] else None
        self.tree: Any = None
        if self.points is not None:
            try:
                from scipy.spatial import cKDTree  # noqa: PLC0415 (only side approaches pay for the import)
            except Exception:  # pragma: no cover (SciPy is pinned; the scan below gives the same distance)
                self.tree = None
            else:
                self.tree = cKDTree(self.points)

    def least_mm(self, samples: np.ndarray) -> float:
        """The least distance from ``samples`` to an obstacle point; ``inf`` where none lies within
        ``_CLEARANCE_FULL_MM``, which scores the same as any distance past it."""
        if self.points is None or samples.shape[0] == 0:
            return math.inf
        if self.tree is not None:
            distances, _ = self.tree.query(samples, k=1, distance_upper_bound=_CLEARANCE_FULL_MM)
            least = float(np.min(distances))
            return least if least <= _CLEARANCE_FULL_MM else math.inf
        lo = samples.min(axis=0) - _CLEARANCE_FULL_MM
        hi = samples.max(axis=0) + _CLEARANCE_FULL_MM
        near = self.points[np.all((self.points >= lo) & (self.points <= hi), axis=1)]
        least = math.inf
        for start in range(0, near.shape[0], 4096):
            delta = samples[:, None, :] - near[None, start:start + 4096, :]
            least = min(least, float(np.sqrt(np.einsum("ijk,ijk->ij", delta, delta).min())))
        return least if least <= _CLEARANCE_FULL_MM else math.inf

    def least_many(self, samples: np.ndarray, asked: np.ndarray) -> np.ndarray:
        """:meth:`least_mm` for many builds at once: ``(N, P, 3)`` samples and which of them are asked (``(N, P)``)
        in, one distance per build out. The tree answers each point by itself, so asking all at once and taking each
        build's least gives each build's own number."""
        out = np.full(samples.shape[0], math.inf)
        if self.points is None:
            return out
        if self.tree is None:
            for build in range(samples.shape[0]):
                out[build] = self.least_mm(samples[build][asked[build]])
            return out
        flat = samples[asked]
        if not flat.shape[0]:
            return out
        distances, _ = self.tree.query(flat, k=1, distance_upper_bound=_CLEARANCE_FULL_MM)
        per_build = asked.sum(axis=1)
        starts = np.concatenate([[0], np.cumsum(per_build)[:-1]])
        some = per_build > 0
        out[some] = np.minimum.reduceat(distances, starts[some])
        return np.where(out <= _CLEARANCE_FULL_MM, out, math.inf)


@dataclass(frozen=True, slots=True)
class _Side:
    """What a side-approach build needs beyond today's: who says a corridor sample was seen (``None``: nobody can, so
    no tilt is explored past the first that fits) and the obstacle points the clearance is measured to."""

    seen: CorridorSeen | None
    room: _Room


def _upright_first(found: list["SupportFootprintCandidate"]) -> list["SupportFootprintCandidate"]:
    """``found`` by score, each run within :data:`_TIE_SCORE` of its best ranked vertical first: the owner's approaches
    equal by geometry (2026-10-01), so vertical wins a tie. ``approach[2]`` is -cos(tilt) and rises with it; rounded, so
    float noise cannot reorder one rung of the ladder."""
    by_score = sorted(found, key=lambda c: -c.score)
    ranked: list[SupportFootprintCandidate] = []
    start = 0
    while start < len(by_score):
        end = start
        while end < len(by_score) and by_score[end].score >= by_score[start].score - _TIE_SCORE:
            end += 1
        ranked.extend(sorted(by_score[start:end], key=lambda c: round(float(c.approach[2]), 9)))
        start = end
    return ranked


def _refused(refusals: dict[str, int] | None, cause: str) -> "SupportFootprintCandidate | None":
    """Count one refused build under ``cause`` where the caller counts; always ``None``, the refused build's answer."""
    if refusals is not None:
        refusals[cause] = refusals.get(cause, 0) + 1
    return None


def _wider_than_the_hand_closes(prism: SupportPrism, anchor_xy: np.ndarray, mid_z: float, axis: np.ndarray,
                                jaw: SupportFootprintJaw) -> bool:
    """Whether every build anchored at ``anchor_xy`` that closes along the horizontal ``axis`` is refused for the
    aperture, whatever its height, tilt and side.

    ``_build`` measures a grasp's span over nine samples of the pad face, the anchor itself among them, and takes the
    widest, so a grasp never spans less than the line through its anchor; and the anchor stands inside the part, so the
    span check before the aperture's passes. Where that line, read :data:`_SPAN_SURE_MM` inside the faces, spans more
    than the hand closes on, every build on it is refused for the aperture and nothing else. The heights SFE anchors at
    all stand inside the part's height, where a horizontal line's span does not depend on them.
    """
    inner = prism.line_span(np.array([anchor_xy[0], anchor_xy[1], mid_z]), axis, margin_mm=-_SPAN_SURE_MM)
    return (inner is not None and inner[0] <= 0.0 <= inner[1]
            and inner[1] - inner[0] > jaw.aperture_mm - jaw.width_safety_mm)


# --------------------------------------------------------------------------- one candidate


def _pad_contacts(prism: SupportPrism, anchor: np.ndarray, axis: np.ndarray,
                  approach: np.ndarray, binormal: np.ndarray, jaw: SupportFootprintJaw):
    """Where the two pads touch, and the lowest point of the pad face that touches at all.

    Taking the lowest contributing offset rather than the one that happens to win the span is the
    difference between a plan that clears the table and one that ties. All nine samples are clipped
    against the prism in a single vectorised slab pass.
    """
    hw = jaw.finger_width_mm / 2.0
    binormal_offsets = np.array([-hw, 0.0, hw])
    approach_offsets = np.array([-jaw.pad_behind_mm, 0.0, jaw.pad_ahead_mm])
    origins = (anchor[None, None, :]
               + binormal_offsets[:, None, None] * binormal[None, None, :]
               + approach_offsets[None, :, None] * approach[None, None, :]).reshape(9, 3)
    a = prism.normals @ axis[:2]
    b = prism.offsets[None, :] - origins[:, :2] @ prism.normals.T
    par = np.abs(a) < 1e-12
    ok = ~np.any(par[None, :] & (b < 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / np.where(par, np.nan, a)[None, :]
    pos, neg = (~par) & (a > 0), (~par) & (a < 0)
    hi = np.nanmin(np.where(pos[None, :], t, np.inf), axis=1) if pos.any() else np.full(9, np.inf)
    lo = np.nanmax(np.where(neg[None, :], t, -np.inf), axis=1) if neg.any() else np.full(9, -np.inf)
    az = float(axis[2])
    if abs(az) > 1e-12:
        # A closing axis tilted out of the horizontal leaves the part through its top or its foot as well as a side.
        t1, t2 = (prism.z0 - origins[:, 2]) / az, (prism.z1 - origins[:, 2]) / az
        lo, hi = np.maximum(lo, np.minimum(t1, t2)), np.minimum(hi, np.maximum(t1, t2))
    ok &= (lo <= hi) & (origins[:, 2] >= prism.z0) & (origins[:, 2] <= prism.z1)
    if not ok.any():
        return None
    ie = int(np.argmin(np.where(ok, lo, np.inf)))
    ix = int(np.argmax(np.where(ok, hi, -np.inf)))
    return (float(lo[ie]), float(hi[ix]),
            origins[ie] + lo[ie] * axis, origins[ix] + hi[ix] * axis,
            float(origins[ok, 2].min()))


def _finger_points(contact: np.ndarray, approach: np.ndarray, binormal: np.ndarray,
                   jaw: SupportFootprintJaw, length_mm: float | None = None,
                   step: float = 6.0) -> np.ndarray:
    tip = contact + jaw.finger_ahead_mm * approach
    reach = jaw.finger_reach_mm if length_mm is None else length_mm
    length = np.arange(0.0, reach + step, step)
    width = np.linspace(-jaw.finger_width_mm / 2.0, jaw.finger_width_mm / 2.0, 3)
    grid = (tip[None, None, :] - length[:, None, None] * approach[None, None, :]
            + width[None, :, None] * binormal[None, None, :])
    return grid.reshape(-1, 3)


def _rolled(axis: np.ndarray, approach: np.ndarray, roll_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``axis`` and ``approach`` turned ``roll_deg`` about the hand's binormal (``approach x axis``): the closing axis
    tilted out of the horizontal, one finger higher than the other, the approach leaning along the closing axis. No
    roll returns them as they are."""
    if roll_deg == 0.0:
        return axis, approach
    binormal = _cross3(approach, axis)
    binormal /= max(float(np.linalg.norm(binormal)), _EPS)
    angle = math.radians(roll_deg)
    turned_axis = _rodrigues(axis, binormal, angle)
    turned_approach = _rodrigues(approach, binormal, angle)
    return (turned_axis / max(float(np.linalg.norm(turned_axis)), _EPS),
            turned_approach / max(float(np.linalg.norm(turned_approach)), _EPS))


def _on_the_axis(contact: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Where a finger closing on ``contact`` stands: the anchor, moved along the closing axis as far as the contact."""
    return anchor + float((contact - anchor) @ axis) * axis


def _seated(prism: "SupportPrism", anchor: np.ndarray, approach: np.ndarray, jaw: SupportFootprintJaw) -> float:
    """How well a grasp at ``anchor`` sits on the part, 0 to 1: half how much of the pad's height lies on the part, half
    how near the anchor stands to the part's middle height. A grasp at the very top holds the part with the pad's lower
    half alone and far from its middle; the owner saw the hand grip only the top sixth of the parts (2026-10-05)."""
    down = -float(approach[2])
    upper = float(anchor[2]) + jaw.pad_behind_mm * down
    lower = float(anchor[2]) - jaw.pad_ahead_mm * down
    reach = upper - lower
    covered = (max(0.0, min(upper, prism.z1) - max(lower, prism.z0)) / reach) if reach > _EPS else 0.0
    half = (prism.z1 - prism.z0) / 2.0
    middle = 1.0 - min(1.0, abs(float(anchor[2]) - (prism.z0 + half)) / half) if half > _EPS else 0.0
    return 0.5 * covered + 0.5 * middle


def _palm_corners(anchor: np.ndarray, axis: np.ndarray, approach: np.ndarray, binormal: np.ndarray,
                  jaw: SupportFootprintJaw) -> np.ndarray:
    """The open palm box's eight corners in BASE, anchored at the grasp centre as ``collision_boxes`` anchors it: across
    the open fingers' outer faces, its width along the binormal, ``finger_behind`` to ``finger_behind + palm_depth``
    back up the approach."""
    half_x = jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm
    half_y = jaw.palm_width_mm / 2.0
    backs = (-jaw.finger_behind_mm, -(jaw.finger_behind_mm + jaw.palm_depth_mm))
    return np.array([anchor + sx * half_x * axis + sy * half_y * binormal + back * approach
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for back in backs])


def _palm_low_mm(anchor: np.ndarray, approach: np.ndarray, binormal: np.ndarray,
                 jaw: SupportFootprintJaw, axis: "np.ndarray | None" = None) -> float:
    """Lowest point of the palm box, anchored at the grasp centre exactly as the envelope anchors it.

    Mirrors ``ParallelJawGripperModel.collision_boxes``' palm: it spans ``-finger_length`` to
    ``-(finger_length + palm_depth)`` along the approach and ``+/- palm_width/2`` along the binormal.
    The closing extent is omitted because SFE's closing axis is horizontal by construction
    (``axis[2] == 0``), so it contributes nothing to a height.
    """
    back = float(approach[2])
    across = 0.0 if axis is None else (jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm) * abs(float(axis[2]))
    return (float(anchor[2])
            + min(back * -jaw.finger_behind_mm, back * -(jaw.finger_behind_mm + jaw.palm_depth_mm))
            - (jaw.palm_width_mm / 2.0) * abs(float(binormal[2])) - across)


def _build(prism: SupportPrism, anchor: np.ndarray, axis: np.ndarray, approach: np.ndarray,
           jaw: SupportFootprintJaw, obstacles: "_ObstacleSet",
           support_height_mm: float, *, palm_aware: bool = False,
           score_weights: tuple[float, float, float, float, float] | None = None,
           refusals: dict[str, int] | None = None, side: _Side | None = None, tilt_deg: float = 0.0,
           seen: Any = None, floor: HandFloor | None = None,
           ) -> SupportFootprintCandidate | None:
    binormal = _cross3(approach, axis)
    binormal /= max(float(np.linalg.norm(binormal)), _EPS)
    contacts = _pad_contacts(prism, anchor, axis, approach, binormal, jaw)
    if contacts is None:
        return _refused(refusals, "span")
    t_enter, t_exit, contact_b, contact_a, _low_pad_z = contacts
    span = t_exit - t_enter
    if not (t_enter - 2.0 <= 0.0 <= t_exit + 2.0):
        return _refused(refusals, "span")
    if span > jaw.aperture_mm - jaw.width_safety_mm or span < jaw.min_width_mm:
        return _refused(refusals, "aperture")

    cone = jaw.cone_rad
    worst = max(float(np.arccos(np.clip(prism.normal_at(contact_a) @ axis, -1.0, 1.0))),
                float(np.arccos(np.clip(prism.normal_at(contact_b) @ -axis, -1.0, 1.0))))
    if worst > cone:
        return _refused(refusals, "cone")

    half_thickness = jaw.finger_thickness_mm / 2.0
    # Each finger stands where the hand puts it: at the anchor, moved along the closing axis to its contact. The pad's
    # sample that touched may lie a pad's length up or down the approach, and a finger hung from it stood a full
    # ``finger_ahead`` under the real one: every grasp was planned 10.45 mm higher than the hand needs, and parts under
    # 28 mm got none (the grasp bench and the owner, 2026-10-05: "wir müssen tiefer gehen"). The width tips the tip
    # further down where the approach is tilted, which the finger's own samples carry.
    closed = np.concatenate([
        _finger_points(_on_the_axis(contact_a, anchor, axis) + half_thickness * axis, approach, binormal, jaw),
        _finger_points(_on_the_axis(contact_b, anchor, axis) - half_thickness * axis, approach, binormal, jaw)])
    low = float(closed[:, 2].min())
    if palm_aware:
        # Taken as a minimum against the fingers, never as a replacement for them: adding the palm can only lower
        # ``low``, so this can refuse a candidate and can never admit one that was refused before.
        low = min(low, _palm_low_mm(anchor, approach, binormal, jaw, axis))
    if low < support_height_mm + jaw.table_clearance_mm:
        return _refused(refusals, "table")
    # The solid the guard holds under the hand: every finger point over its top by the guard's distance, and the palm,
    # which the guard holds whether or not ``palm_aware`` plans it against the reading.
    if floor is not None and (floor.under(closed, fingers=True)
                              or floor.under(_palm_corners(anchor, axis, approach, binormal, jaw))):
        return _refused(refusals, "table")

    # A real jaw arrives open and closes at the end, so the corridor is checked at the aperture, not
    # at the final width. That also removes the reconstruction's one-sided thinness from the
    # question: fingers held wider than the object cannot clip a flank the depth image missed.
    open_half = jaw.aperture_mm / 2.0
    reach = jaw.finger_reach_mm + 80.0
    swept = np.concatenate([
        _finger_points(anchor + open_half * axis, approach, binormal, jaw, reach, 8.0),
        _finger_points(anchor - open_half * axis, approach, binormal, jaw, reach, 8.0)])
    swept = swept[swept[:, 2] >= support_height_mm]
    # The open fingers come down beside the closed ones, wider: over a tilted solid or one beside the part's they may
    # stand where the closed ones do not.
    if floor is not None and floor.under(swept, fingers=True):
        return _refused(refusals, "table")
    # The way in is asked before the closed fingers, so a refusal counts where the hand meets something
    # first: a part boxed in by its neighbours counts its corridor. A candidate needs both clear and the
    # checks change nothing, so the order decides only what a refusal is counted under.
    if obstacles.refuse(swept, refusals, _CORRIDOR_CAUSES):
        return None
    if obstacles.refuse(closed, refusals, _FINGER_CAUSES):
        return None
    # The boxes the camera world builds of the neighbours, as the guard holds them (``scene_obstacles.SeenEnvelope``):
    # the open hand keeps the guard's distance from them at the grasp and on its way in, or the guard would refuse the
    # grasp once the arm stood over it (the owner, 2026-10-03).
    if seen is not None and seen.refuses(anchor, np.column_stack([axis, binormal, approach])):
        return _refused(refusals, "seen_fingers")
    if prism.contains(closed, margin_mm=-1.0 - prism.inflate).any():
        return _refused(refusals, "prism")
    if prism.contains(swept, margin_mm=-1.0 - prism.inflate).any():
        return _refused(refusals, "prism")

    # The same corridor at the pre-grasp width, against the object only. A controller that
    # pre-positions at span + margin instead of fully open must also get in, so the plan does not
    # depend on which of the two the driver implements.
    narrow = min(jaw.aperture_mm, span + 10.0) / 2.0
    if narrow < open_half - 0.5:
        swept_narrow = np.concatenate([
            _finger_points(anchor + narrow * axis, approach, binormal, jaw, reach, 8.0),
            _finger_points(anchor - narrow * axis, approach, binormal, jaw, reach, 8.0)])
        swept_narrow = swept_narrow[swept_narrow[:, 2] >= support_height_mm]
        if prism.contains(swept_narrow, margin_mm=-1.0 - prism.inflate).any():
            return _refused(refusals, "prism")

    # With side approaches, only space a depth ray saw counts as clear: the camera world takes an
    # occlusion shadow for free space, and the guard would judge this approach against that same free
    # shadow. A vertical grasp is not asked, as it never was.
    room = math.inf
    if side is not None:
        if side.seen is not None and tilt_deg >= _SEEN_FROM_TILT_DEG - 1e-6 and swept.shape[0]:
            seen = np.asarray(side.seen(swept), dtype=bool).reshape(-1)
            if seen.shape[0] != swept.shape[0]:
                raise ValueError(f"corridor_seen answered {seen.shape[0]} verdict(s) for {swept.shape[0]} point(s)")
            if not bool(seen.all()):
                return _refused(refusals, "unseen_corridor")
        room = side.room.least_mm(swept)

    # Rank on measured margins. The weights and their measurement are at ``_SCORE_WEIGHTS``; with side approaches at
    # ``SIDE_APPROACH_SCORE_WEIGHTS``. The third term is how the grasp sits on the part (``_seated``), and with side
    # approaches as much the room it keeps from every obstacle. A fingertip high over the support scored in full until
    # 2026-10-05 and drove every grasp to the part's top; the table check keeps the clearance, the term no longer pays
    # for more of it.
    cone_slack = 1.0 - worst / cone                                 # depth inside the friction cone
    width_margin = 1.0 - span / jaw.aperture_mm                     # aperture left over
    seated = _seated(prism, anchor, approach, jaw)
    if side is None:
        table_margin = seated
    else:
        table_margin = 0.5 * seated + 0.5 * min(1.0, room / _CLEARANCE_FULL_MM)
    upright = max(0.0, float(-approach @ np.array([0.0, 0.0, 1.0])))
    centred = 1.0 - min(1.0, abs(t_enter + t_exit) / max(span, 1.0))
    if score_weights is not None:
        w = score_weights
    else:
        w = _SCORE_WEIGHTS if side is None else SIDE_APPROACH_SCORE_WEIGHTS
    score = (w[0] * cone_slack + w[1] * width_margin + w[2] * table_margin
             + w[3] * upright + w[4] * centred)
    return SupportFootprintCandidate(
        score=float(score), position_mm=anchor.copy(), approach=approach.copy(),
        closing_axis=axis.copy(), grip_width_mm=float(span + 2.0),
        contact_angle_rad=float(worst), clearance_mm=float(low - support_height_mm))


# --------------------------------------------------------------------------- a closing line's builds at once


#: What a batched answer says of one build (:meth:`HandFloor.under_many`, ``SeenEnvelope.refuses_many``,
#: ``SeenInTheFrame.unseen_many``): no, yes, or a number within its guard band of the threshold, which ``_build``
#: decides alone.
_NO, _YES, _UNSURE = 0, 1, 2
#: Why :func:`_build_many` refused a build, by code: ``_KEPT`` it did not; ``_CORRIDOR`` and ``_FINGERS`` an obstacle
#: on the way in or at the closed fingers, counted by the sets it met; ``_BY_ITSELF`` a number within :data:`_BATCH_SURE`
#: of its threshold, so ``_build`` makes it alone; the rest the cause they count under (:data:`_MANY_CAUSES`).
_KEPT, _SPAN, _APERTURE, _CONE, _TABLE, _CORRIDOR, _FINGERS, _SEEN, _PRISM, _UNSEEN, _BY_ITSELF = range(11)
_MANY_CAUSES: Final[Mapping[int, str]] = {_SPAN: "span", _APERTURE: "aperture", _CONE: "cone", _TABLE: "table",
                                          _SEEN: "seen_fingers", _PRISM: "prism", _UNSEEN: "unseen_corridor"}


def _dots(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each row's dot product of ``(N, 3)`` ``a`` and ``b``, one row at a time through a stacked ``np.matmul``: the
    BLAS call ``a[n] @ b[n]`` makes, so its bits. An elementwise sum agreed with it in two cases of three
    (2026-10-08)."""
    return np.matmul(a[:, None, :], b[:, :, None])[:, 0, 0]


def _cross_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:func:`_cross3` of each row of ``(N, 3)`` ``a`` and ``b``: the same three products and differences."""
    return np.column_stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]])


def _rodrigues_many(vectors: np.ndarray, axes: np.ndarray, angles: Sequence[Any]) -> np.ndarray:
    """:func:`_rodrigues` of each row, ``vectors[n]`` turned about ``axes[n]`` by ``angles[n]``: the same elementwise
    arithmetic, each dot product and length the row's own (:func:`_dots`), and each angle's cosine and sine taken one
    at a time as numpy scalars, as there, so that no vectorised sine of this machine's numpy can round otherwise."""
    cos = np.array([np.cos(angle) for angle in angles], dtype=np.float64)
    sin = np.array([np.sin(angle) for angle in angles], dtype=np.float64)
    k = axes / np.maximum(np.sqrt(_dots(axes, axes)), _EPS)[:, None]
    return (vectors * cos[:, None] + _cross_many(k, vectors) * sin[:, None]
            + k * _dots(k, vectors)[:, None] * (1.0 - cos)[:, None])


class _Ladder:
    """Every build one closing axis makes in one search, in :func:`_run_unit`'s order (each tilt, its sides, each roll),
    worked out at once by :func:`_ladder`: a row per build, the same axis, approach and binormal ``_build`` works with,
    the prism's faces against the axis (``_pad_contacts``' ``a``), the tilt off vertical the build is asked with, and the
    rows of each tilt (:attr:`rungs`), the ladder's rungs in order."""

    __slots__ = ("axis", "approach", "binormal", "back", "facing", "off_vertical", "rungs", "size")

    def __init__(self, axis: np.ndarray, approach: np.ndarray, binormal: np.ndarray, facing: np.ndarray,
                 off_vertical: list[float], rungs: list[np.ndarray]) -> None:
        self.axis = axis
        self.approach = approach
        self.binormal = binormal
        #: The axis the far pad closes along, for its contact's angle (``-axis`` in ``_build``).
        self.back = -axis
        self.facing = facing
        self.off_vertical = off_vertical
        self.rungs = rungs
        self.size = int(axis.shape[0])


def _ladder(prism: SupportPrism, axis: np.ndarray, tilts: Sequence[float], rolls: Sequence[float]) -> _Ladder:
    """Every build of the closing ``axis`` over ``tilts``, both sides and ``rolls``, to the bit what :func:`_run_unit`
    and ``_build`` make of each: ``_run_unit``'s tilt (:func:`_rodrigues`) and :func:`_rolled`, then ``_build``'s
    binormal and ``_pad_contacts``' faces, on every row at once."""
    order = [(index, tilt, sign, roll) for index, tilt in enumerate(tilts)
             for sign in ((1.0,) if tilt == 0.0 else (1.0, -1.0)) for roll in rolls]
    count = len(order)
    axes = np.tile(np.asarray(axis, dtype=np.float64), (count, 1))
    tilted = _rodrigues_many(np.tile(np.array([0.0, 0.0, -1.0]), (count, 1)), axes,
                             [sign * np.radians(tilt) for _, tilt, sign, _ in order])
    tilted = tilted / np.sqrt(_dots(tilted, tilted))[:, None]
    turned, approach = axes.copy(), tilted.copy()
    rolled = np.array([roll != 0.0 for *_, roll in order], dtype=bool)
    if rolled.any():
        # ``_rolled``: the axis and the approach turned about the hand's binormal, each made a unit again.
        about = _cross_many(tilted[rolled], axes[rolled])
        about = about / np.maximum(np.sqrt(_dots(about, about)), _EPS)[:, None]
        angles = [math.radians(roll) for *_, roll in order if roll != 0.0]
        turned_axis = _rodrigues_many(axes[rolled], about, angles)
        turned_approach = _rodrigues_many(tilted[rolled], about, angles)
        turned[rolled] = turned_axis / np.maximum(np.sqrt(_dots(turned_axis, turned_axis)), _EPS)[:, None]
        approach[rolled] = turned_approach / np.maximum(np.sqrt(_dots(turned_approach, turned_approach)), _EPS)[:, None]
    binormal = _cross_many(approach, turned)
    binormal = binormal / np.maximum(np.sqrt(_dots(binormal, binormal)), _EPS)[:, None]
    facing = np.matmul(prism.normals[None], turned[:, :2, None])[:, :, 0]
    # The ladder's own tilt where nothing rolls, read back off the approach where it does (``_run_unit``).
    off_vertical = [float(tilt) if roll == 0.0 else
                    math.degrees(math.acos(max(-1.0, min(1.0, -float(approach[row, 2])))))
                    for row, (_, tilt, _, roll) in enumerate(order)]
    tilt_of = np.array([index for index, *_ in order])
    return _Ladder(turned, approach, binormal, facing, off_vertical,
                   [np.nonzero(tilt_of == index)[0] for index in range(len(tilts))])


def _normals_at(prism: SupportPrism, points: np.ndarray) -> np.ndarray:
    """:meth:`SupportPrism.normal_at` of each row of ``(N, 3)`` points: each row's own product (a stacked
    ``np.matmul``), the same face chosen."""
    slack = prism.offsets[None, :] - np.matmul(prism.normals[None], points[:, :2, None])[:, :, 0]
    nearest = np.argmin(np.abs(slack), axis=1)
    best = np.abs(slack[np.arange(points.shape[0]), nearest])
    normal = np.zeros((points.shape[0], 3))
    normal[:, 0] = prism.normals[nearest, 0]
    normal[:, 1] = prism.normals[nearest, 1]
    top = np.abs(points[:, 2] - prism.z1)
    on_top = top < best
    normal[on_top] = np.array([0.0, 0.0, 1.0])
    best = np.where(on_top, top, best)
    normal[np.abs(points[:, 2] - prism.z0) < best] = np.array([0.0, 0.0, -1.0])
    return normal


def _fingers_many(contacts: np.ndarray, approach: np.ndarray, binormal: np.ndarray, jaw: SupportFootprintJaw,
                  lengths: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """:func:`_finger_points` of each row: ``(N, len(lengths) * 3, 3)``, the same elementwise arithmetic and order."""
    tip = contacts + jaw.finger_ahead_mm * approach
    grid = (tip[:, None, None, :] - lengths[None, :, None, None] * approach[:, None, None, :]
            + widths[None, None, :, None] * binormal[:, None, None, :])
    return grid.reshape(contacts.shape[0], -1, 3)


def _inside_many(prism: SupportPrism, normals_t: np.ndarray, points: np.ndarray, asked: np.ndarray,
                 margin_mm: float) -> np.ndarray:
    """Whether one of each build's ``(N, P, 3)`` points that are ``asked`` lies inside the prism grown by ``margin_mm``
    (:meth:`SupportPrism.contains`): :data:`_NO`, :data:`_YES`, or :data:`_UNSURE` where a point's nearest face lies
    within :data:`_BATCH_SURE` of it, since a product over many builds' points rounds its last bit as it likes."""
    out = np.zeros(points.shape[0], dtype=np.int8)
    asked = asked & (points[:, :, 2] >= prism.z0 - margin_mm) & (points[:, :, 2] <= prism.z1 + margin_mm)
    owner = np.nonzero(asked)[0]
    flat = points[asked]
    if not flat.shape[0]:
        return out
    worst = ((flat[:, :2] @ normals_t) - (prism.offsets + margin_mm)).max(axis=1)
    unsure = np.abs(worst) <= _BATCH_SURE
    out[owner[unsure]] = _UNSURE
    out[owner[(worst <= 0.0) & ~unsure]] = _YES
    return out


def _unseen_one(seen: CorridorSeen, points: np.ndarray) -> int:
    """``_build``'s question to a seen test of another kind than SFE's own, for one build: :data:`_YES` where a point is
    unseen."""
    said = np.asarray(seen(points), dtype=bool).reshape(-1)
    if said.shape[0] != points.shape[0]:
        raise ValueError(f"corridor_seen answered {said.shape[0]} verdict(s) for {points.shape[0]} point(s)")
    return _NO if bool(said.all()) else _YES


def _name_the_sets(met: dict[int, tuple[str, ...]], index: np.ndarray, hit: np.ndarray, sets: "np.ndarray | None",
                   causes: Mapping[str, str]) -> None:
    """Into ``met``, by build, the causes each build ``index[k]`` that ``hit`` an obstacle counts under: one per set it
    met (``sets``, :meth:`_ObstacleSet.met_many`), in :meth:`_ObstacleSet.met`'s order. Nothing where nobody counts."""
    if sets is None:
        return
    for k in np.nonzero(hit)[0].tolist():
        met[int(index[k])] = tuple(causes[name] for name, on in zip(("seen", "declared", "own"), sets[k]) if on)


@functools.lru_cache(maxsize=None)
def _stacked_products_hold(faces: int, closed_points: int) -> str:
    """Why this process's numpy cannot make a closing line's builds at once to the bit, or ``""`` where it can.

    :func:`_build_many` makes every number a candidate carries by the products ``_build`` makes, one build at a time
    through a stacked ``np.matmul``: a 3-vector's dot product and length, the prism's ``faces`` against a closing axis
    and a contact, the pads' nine samples and the ``closed_points`` of the closed fingers against the faces. That is
    the same BLAS call where numpy hands every item of a stack to it as it hands one product, as numpy 2.4 with OpenBLAS
    does on the desk it was measured on (2026-10-08, 100 % of every shape); asked once per process and shape here, on
    seeded numbers, because another machine's BLAS may take another path.
    """
    rng = np.random.default_rng(20261009 + 1000 * faces + closed_points)
    a, b = rng.standard_normal((48, 3)), rng.standard_normal((48, 3))
    normals = rng.standard_normal((faces, 2))
    samples = rng.standard_normal((48, 9, 3)) * 50.0
    closed = rng.standard_normal((12, closed_points, 3)) * 50.0
    checks = (
        ("a 3-vector's dot product", _dots(a, b), [x @ y for x, y in zip(a, b)]),
        ("a 3-vector's length", np.sqrt(_dots(a, a)), [np.linalg.norm(x) for x in a]),
        ("the faces against an axis", np.matmul(normals[None], a[:, :2, None])[:, :, 0], [normals @ x[:2] for x in a]),
        ("the pads against the faces", np.matmul(samples[:, :, :2], normals.T), [s[:, :2] @ normals.T for s in samples]),
        ("the fingers against the faces", np.matmul(closed[:, :, :2], normals.T),
         [c[:, :2] @ normals.T for c in closed]),
    )
    for what, stacked, one_at_a_time in checks:
        if np.asarray(stacked, dtype=np.float64).tobytes() != np.asarray(one_at_a_time, dtype=np.float64).tobytes():
            return f"numpy's stacked product of {what} rounds otherwise than one at a time here"
    return ""


def batched_builds_hold(jaw: SupportFootprintJaw | None = None) -> str:
    """Why this process cannot make SFE's builds at once to the bit (``generate_support_footprint_grasps(batched=)``),
    or ``""`` where it can: numpy's stacked products asked on seeded numbers of the shapes a part's search stacks, the
    closed fingers of ``jaw`` (the library's own where ``None``) against footprints of 4 to 24 faces. A cell asks it
    when its calculator is built (``robot.grasping.batched_builds``) and says the answer; every part's own shapes are
    asked again before its first build, once per process, and where they do not hold its builds are made one at a time.
    """
    jaw = jaw or SupportFootprintJaw.from_model()
    closed_points = 6 * int(np.arange(0.0, jaw.finger_reach_mm + 6.0, 6.0).size)
    for faces in (4, 6, 8, 12, 16, 24):
        why = _stacked_products_hold(faces, closed_points)
        if why:
            return why
    return ""


class _Batch:
    """What every build :func:`_build_many` makes of one plan shares, worked out once (:meth:`SfePlan.batch`): the
    pads' and fingers' offsets and lengths as ``_build`` makes them, the score's weights and the friction cone, which of
    the floor, the camera's boxes and the seen test answer for many builds at once (SFE's own kinds; another kind is
    asked one build at a time), and why the builds are made one at a time here instead (``refused``, ``""`` where not).
    """

    __slots__ = ("normals_t", "across", "along", "closed_lengths", "open_lengths", "widths", "weights", "cone",
                 "floor_many", "envelope_many", "rays_many", "refused")

    def __init__(self, plan: "SfePlan") -> None:
        from src.robot.grasping.generation.scene_obstacles import SeenEnvelope, SeenInTheFrame  # noqa: PLC0415

        jaw, inputs = plan.jaw, plan.inputs
        _obstacles, side, _every_tilt = plan.grids()
        self.normals_t = plan.prism.normals.T
        half_width = jaw.finger_width_mm / 2.0
        self.across = np.array([-half_width, 0.0, half_width])
        self.along = np.array([-jaw.pad_behind_mm, 0.0, jaw.pad_ahead_mm])
        self.closed_lengths = np.arange(0.0, jaw.finger_reach_mm + 6.0, 6.0)
        self.open_lengths = np.arange(0.0, (jaw.finger_reach_mm + 80.0) + 8.0, 8.0)
        self.widths = np.linspace(-jaw.finger_width_mm / 2.0, jaw.finger_width_mm / 2.0, 3)
        self.weights = (inputs.score_weights if inputs.score_weights is not None
                        else _SCORE_WEIGHTS if side is None else SIDE_APPROACH_SCORE_WEIGHTS)
        self.cone = jaw.cone_rad
        self.floor_many = type(inputs.hand_floor) is HandFloor
        self.envelope_many = type(inputs.seen_envelope) is SeenEnvelope
        self.rays_many = side is not None and type(side.seen) is SeenInTheFrame
        self.refused = _stacked_products_hold(int(plan.prism.normals.shape[0]), 6 * int(self.closed_lengths.size))


def _build_many(plan: "SfePlan", anchors: np.ndarray, ladder: _Ladder, rows: np.ndarray, *,
                counting: bool) -> "tuple[list[SupportFootprintCandidate | None], list[tuple[str, ...]]]":
    """``_build`` for many builds at once, build ``n`` anchored at ``anchors[n]`` on rung ``rows[n]`` of ``ladder``: each
    build's candidate or ``None``, and where ``counting`` the causes each refused one counts under, in ``_build``'s
    order.

    ``_build``'s checks run in ``_build``'s order, each over every build still standing: the pads' span and the
    aperture, the friction cone, the closed fingers over the support and the floor with the palm, the open fingers over
    the floor, the obstacles on the way in and at the fingers, the camera's boxes, a finger inside the part, the unseen
    corridor. Every number a candidate carries is made by ``_build``'s own arithmetic, each product one build at a time
    through a stacked ``np.matmul``, and the score by ``_build``'s own expressions, per survivor: the same bits. The
    numbers that only decide a verdict and are made over many builds' points at once (the floor, the open fingers in
    the part, the corridor's pixels and depth, the camera's boxes, the cone's angle) are held :data:`_BATCH_SURE` off
    their thresholds; a build nearer one is made by ``_build`` alone, which decides it as it always did. A floor, an
    envelope or a seen test of another kind than SFE's own is asked one build at a time, as ``_build`` asks it.
    """
    count = int(anchors.shape[0])
    if not count:
        return [], []
    batch = plan.batch()
    prism, jaw, inputs = plan.prism, plan.jaw, plan.inputs
    obstacles, side, _every_tilt = plan.grids()
    support, floor, envelope = inputs.support_height_mm, inputs.hand_floor, inputs.seen_envelope
    cause = np.zeros(count, dtype=np.int8)
    met: dict[int, tuple[str, ...]] = {}
    axis, approach, binormal = ladder.axis[rows], ladder.approach[rows], ladder.binormal[rows]
    facing = ladder.facing[rows]

    # The pads (``_pad_contacts``): every build's nine samples clipped against the prism at once.
    origins = (anchors[:, None, None, :] + batch.across[None, :, None, None] * binormal[:, None, None, :]
               + batch.along[None, None, :, None] * approach[:, None, None, :]).reshape(count, 9, 3)
    reach = prism.offsets[None, None, :] - np.matmul(origins[:, :, :2], batch.normals_t)
    level = np.abs(facing) < 1e-12
    ok = ~np.any(level[:, None, :] & (reach < 0.0), axis=2)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = reach / np.where(level, np.nan, facing)[:, None, :]
        hi = np.min(np.where(((~level) & (facing > 0))[:, None, :], t, np.inf), axis=2)
        lo = np.max(np.where(((~level) & (facing < 0))[:, None, :], t, -np.inf), axis=2)
        heights = origins[:, :, 2]
        tipped = np.abs(axis[:, 2]) > 1e-12
        if tipped.any():
            # A closing axis tilted out of the horizontal leaves the part through its top or its foot as well.
            t1 = (prism.z0 - heights[tipped]) / axis[tipped, 2][:, None]
            t2 = (prism.z1 - heights[tipped]) / axis[tipped, 2][:, None]
            lo[tipped] = np.maximum(lo[tipped], np.minimum(t1, t2))
            hi[tipped] = np.minimum(hi[tipped], np.maximum(t1, t2))
        ok &= (lo <= hi) & (heights >= prism.z0) & (heights <= prism.z1)
        every = np.arange(count)
        enter = np.argmin(np.where(ok, lo, np.inf), axis=1)
        leave = np.argmax(np.where(ok, hi, -np.inf), axis=1)
        t_enter, t_exit = lo[every, enter], hi[every, leave]
        contact_b = origins[every, enter] + t_enter[:, None] * axis
        contact_a = origins[every, leave] + t_exit[:, None] * axis
        span = t_exit - t_enter
        # A build whose samples all miss the part has no span: its numbers above are a missing contact's, never read.
        cause[~ok.any(axis=1)] = _SPAN
        alive = cause == _KEPT
        bad = alive & ~((t_enter - 2.0 <= 0.0) & (0.0 <= t_exit + 2.0))
        cause[bad] = _SPAN
        alive &= ~bad
        bad = alive & ((span > jaw.aperture_mm - jaw.width_safety_mm) | (span < jaw.min_width_mm))
        cause[bad] = _APERTURE
        alive &= ~bad

    # The friction cone: the contacts' faces against the axis, each build's own product.
    toward_a = np.full(count, np.nan)
    toward_b = np.full(count, np.nan)
    index = np.nonzero(alive)[0]
    if index.size:
        toward_a[index] = _dots(_normals_at(prism, contact_a[index]), axis[index])
        toward_b[index] = _dots(_normals_at(prism, contact_b[index]), ladder.back[rows[index]])
        worst = np.maximum(np.arccos(np.clip(toward_a[index], -1.0, 1.0)),
                           np.arccos(np.clip(toward_b[index], -1.0, 1.0)))
        unsure = np.abs(worst - batch.cone) < _BATCH_SURE
        cause[index[unsure]] = _BY_ITSELF
        cause[index[(worst > batch.cone) & ~unsure]] = _CONE
        index = index[(worst <= batch.cone) & ~unsure]

    # The closed fingers, each standing on the axis at its contact, over the support (and the palm where it is asked).
    low = np.full(count, np.nan)
    closed = np.zeros((0, 0, 3))
    if index.size:
        x, a, b, at = axis[index], approach[index], binormal[index], anchors[index]
        half_thickness = jaw.finger_thickness_mm / 2.0
        on_a = _dots(contact_a[index] - at, x)
        on_b = _dots(contact_b[index] - at, x)
        closed = np.concatenate([
            _fingers_many((at + on_a[:, None] * x) + half_thickness * x, a, b, jaw, batch.closed_lengths, batch.widths),
            _fingers_many((at + on_b[:, None] * x) - half_thickness * x, a, b, jaw, batch.closed_lengths, batch.widths),
        ], axis=1)
        lowest = closed[:, :, 2].min(axis=1)
        if inputs.palm_aware:
            back = a[:, 2]
            across = (jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm) * np.abs(x[:, 2])
            lowest = np.minimum(lowest, at[:, 2] + np.minimum(back * -jaw.finger_behind_mm,
                                                               back * -(jaw.finger_behind_mm + jaw.palm_depth_mm))
                                - (jaw.palm_width_mm / 2.0) * np.abs(b[:, 2]) - across)
        low[index] = lowest
        bad = lowest < support + jaw.table_clearance_mm
        cause[index[bad]] = _TABLE
        index, closed = index[~bad], closed[~bad]

    # The solid the guard holds under the hand: the closed fingers and the palm's corners.
    if floor is not None and index.size:
        x, a, b, at = axis[index], approach[index], binormal[index], anchors[index]
        half_x = jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm
        half_y = jaw.palm_width_mm / 2.0
        backs = (-jaw.finger_behind_mm, -(jaw.finger_behind_mm + jaw.palm_depth_mm))
        corners = np.stack([((at + (sx * half_x) * x) + (sy * half_y) * b) + back * a
                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for back in backs], axis=1)
        if batch.floor_many:
            fingers_say = floor.under_many(closed, np.ones(closed.shape[:2], dtype=bool), fingers=True)
            palm_says = floor.under_many(corners, np.ones(corners.shape[:2], dtype=bool))
        else:
            fingers_say = np.array([_YES if floor.under(closed[k].copy(), fingers=True) else _NO
                                    for k in range(index.size)], dtype=np.int8)
            palm_says = np.array([_NO if fingers_say[k] == _YES else _YES if floor.under(corners[k].copy()) else _NO
                                  for k in range(index.size)], dtype=np.int8)
        under = (fingers_say == _YES) | (palm_says == _YES)
        unsure = ~under & ((fingers_say == _UNSURE) | (palm_says == _UNSURE))
        cause[index[under]] = _TABLE
        cause[index[unsure]] = _BY_ITSELF
        keep = ~(under | unsure)
        index, closed = index[keep], closed[keep]

    # The open fingers on their way in, the points over the support asked: over the floor first.
    swept = np.zeros((0, 0, 3))
    kept = np.zeros((0, 0), dtype=bool)
    if index.size:
        x, a, b, at = axis[index], approach[index], binormal[index], anchors[index]
        open_half = jaw.aperture_mm / 2.0
        swept = np.concatenate([_fingers_many(at + open_half * x, a, b, jaw, batch.open_lengths, batch.widths),
                                _fingers_many(at - open_half * x, a, b, jaw, batch.open_lengths, batch.widths)], axis=1)
        kept = swept[:, :, 2] >= support
        if floor is not None:
            says = (floor.under_many(swept, kept, fingers=True) if batch.floor_many else
                    np.array([_YES if floor.under(swept[k][kept[k]], fingers=True) else _NO
                              for k in range(index.size)], dtype=np.int8))
            cause[index[says == _YES]] = _TABLE
            cause[index[says == _UNSURE]] = _BY_ITSELF
            keep = says == _NO
            index, closed, swept, kept = index[keep], closed[keep], swept[keep], kept[keep]

    # The obstacles: the way in first, then the closed fingers, as ``_build`` asks them.
    if index.size:
        hit, sets = obstacles.met_many(swept, kept, counting=counting)
        cause[index[hit]] = _CORRIDOR
        _name_the_sets(met, index, hit, sets, _CORRIDOR_CAUSES)
        index, closed, swept, kept = index[~hit], closed[~hit], swept[~hit], kept[~hit]
    if index.size:
        hit, sets = obstacles.met_many(closed, np.ones(closed.shape[:2], dtype=bool), counting=counting)
        cause[index[hit]] = _FINGERS
        _name_the_sets(met, index, hit, sets, _FINGER_CAUSES)
        index, closed, swept, kept = index[~hit], closed[~hit], swept[~hit], kept[~hit]

    # The boxes the camera world builds of the neighbours.
    if envelope is not None and index.size:
        if batch.envelope_many:
            says = envelope.refuses_many(anchors[index], np.stack([axis[index], binormal[index], approach[index]],
                                                                  axis=2))
        else:
            says = np.array([_YES if envelope.refuses(anchors[k].copy(), np.column_stack([axis[k], binormal[k],
                                                                                         approach[k]])) else _NO
                             for k in index.tolist()], dtype=np.int8)
        cause[index[says == _YES]] = _SEEN
        cause[index[says == _UNSURE]] = _BY_ITSELF
        keep = says == _NO
        index, closed, swept, kept = index[keep], closed[keep], swept[keep], kept[keep]

    # A finger inside the part: closed (one build's product at a time, so exact), open, then at the pre-grasp width.
    margin = -1.0 - prism.inflate
    if index.size:
        inside = np.all(np.matmul(closed[:, :, :2], batch.normals_t) <= prism.offsets + margin, axis=2)
        inside &= (closed[:, :, 2] >= prism.z0 - margin) & (closed[:, :, 2] <= prism.z1 + margin)
        bad = inside.any(axis=1)
        cause[index[bad]] = _PRISM
        index, swept, kept = index[~bad], swept[~bad], kept[~bad]
    if index.size:
        says = _inside_many(prism, batch.normals_t, swept, kept, margin)
        cause[index[says == _YES]] = _PRISM
        cause[index[says == _UNSURE]] = _BY_ITSELF
        keep = says == _NO
        index, swept, kept = index[keep], swept[keep], kept[keep]
    if index.size:
        narrow = np.minimum(jaw.aperture_mm, span[index] + 10.0) / 2.0
        asked_narrow = narrow < jaw.aperture_mm / 2.0 - 0.5
        if asked_narrow.any():
            some = index[asked_narrow]
            x, a, b, at = axis[some], approach[some], binormal[some], anchors[some]
            width = narrow[asked_narrow][:, None]
            pre = np.concatenate([_fingers_many(at + width * x, a, b, jaw, batch.open_lengths, batch.widths),
                                  _fingers_many(at - width * x, a, b, jaw, batch.open_lengths, batch.widths)], axis=1)
            says = _inside_many(prism, batch.normals_t, pre, pre[:, :, 2] >= support, margin)
            cause[some[says == _YES]] = _PRISM
            cause[some[says == _UNSURE]] = _BY_ITSELF
            keep = np.ones(index.size, dtype=bool)
            keep[np.nonzero(asked_narrow)[0][says != _NO]] = False
            index, swept, kept = index[keep], swept[keep], kept[keep]

    # With side approaches, only space a depth ray saw counts as clear for a tilted build; and the room each keeps.
    room = np.full(count, math.inf)
    if side is not None and index.size:
        if side.seen is not None:
            off_vertical = np.array([ladder.off_vertical[row] for row in rows[index].tolist()])
            asked_seen = (off_vertical >= _SEEN_FROM_TILT_DEG - 1e-6) & kept.any(axis=1)
            if asked_seen.any():
                some = index[asked_seen]
                rays: Any = side.seen   # SFE's own seen test where ``rays_many`` says so (``SeenInTheFrame``)
                says = (rays.unseen_many(swept[asked_seen], kept[asked_seen]) if batch.rays_many else
                        np.array([_unseen_one(side.seen, swept[k][kept[k]])
                                  for k in np.nonzero(asked_seen)[0].tolist()], dtype=np.int8))
                cause[some[says == _YES]] = _UNSEEN
                cause[some[says == _UNSURE]] = _BY_ITSELF
                keep = np.ones(index.size, dtype=bool)
                keep[np.nonzero(asked_seen)[0][says != _NO]] = False
                index, swept, kept = index[keep], swept[keep], kept[keep]
        if index.size:
            room[index] = side.room.least_many(swept, kept)

    # The candidates, each finished by ``_build``'s own expressions on its own numbers.
    found: list[SupportFootprintCandidate | None] = [None] * count
    w = batch.weights
    for build in index.tolist():
        row = int(rows[build])
        worst_angle = max(float(np.arccos(np.clip(toward_a[build], -1.0, 1.0))),
                          float(np.arccos(np.clip(toward_b[build], -1.0, 1.0))))
        t_in, t_out = float(t_enter[build]), float(t_exit[build])
        width_mm = t_out - t_in
        anchor, turned, leaning = anchors[build], ladder.axis[row], ladder.approach[row]
        cone_slack = 1.0 - worst_angle / batch.cone
        width_margin = 1.0 - width_mm / jaw.aperture_mm
        seated = _seated(prism, anchor, leaning, jaw)
        if side is None:
            table_margin = seated
        else:
            table_margin = 0.5 * seated + 0.5 * min(1.0, float(room[build]) / _CLEARANCE_FULL_MM)
        upright = max(0.0, float(-leaning @ np.array([0.0, 0.0, 1.0])))
        centred = 1.0 - min(1.0, abs(t_in + t_out) / max(width_mm, 1.0))
        score = (w[0] * cone_slack + w[1] * width_margin + w[2] * table_margin
                 + w[3] * upright + w[4] * centred)
        found[build] = SupportFootprintCandidate(
            score=float(score), position_mm=anchor.copy(), approach=leaning.copy(),
            closing_axis=turned.copy(), grip_width_mm=float(width_mm + 2.0),
            contact_angle_rad=float(worst_angle), clearance_mm=float(float(low[build]) - support))

    # What the band left undecided ``_build`` makes alone; what was refused says why.
    causes: list[tuple[str, ...]] = [()] * count
    for build in range(count):
        code = int(cause[build])
        if code == _BY_ITSELF:
            row = int(rows[build])
            counted: "dict[str, int] | None" = {} if counting else None
            found[build] = _build(prism, anchors[build].copy(), ladder.axis[row], ladder.approach[row], jaw, obstacles,
                                  support, palm_aware=inputs.palm_aware, score_weights=inputs.score_weights,
                                  refusals=counted, side=side, tilt_deg=ladder.off_vertical[row], seen=envelope,
                                  floor=floor)
            if counted:
                causes[build] = tuple(name for name, times in counted.items() for _ in range(times))
        elif counting and code in (_CORRIDOR, _FINGERS):
            causes[build] = met[build]
        elif counting and code != _KEPT:
            causes[build] = (_MANY_CAUSES[code],)
    return found, causes


# --------------------------------------------------------------------------- one part's search, in units


#: The three searches of a part, in the order SFE runs them (:attr:`SfePlan.searches`): the coarse grid, whose refusals
#: are counted; the fine grid, every 7.5 degrees of tilt with each face's closing axis turned inside the friction cone;
#: and the coarse grid with each closing axis rolled out of the horizontal. The last two run only where the coarse grid
#: found fewer than ``_FINE_BELOW`` grasps and the fine pass is not left for later.
COARSE: Final[int] = 0
FINE: Final[int] = 1
ROLLED: Final[int] = 2
#: How many heights one closing line is tried at, at most; each is a unit of its own (:data:`SfeUnit`).
_MOST_HEIGHTS: Final[int] = 6

#: One unit of a part's search: ``(search, axis, place, height)``, the indices of :data:`COARSE`, :data:`FINE` or
#: :data:`ROLLED`, of that search's closing axis, of the place along the part (``_FRACS``) and of the line's height.
#: Every unit runs on its own, in any process, and the units are merged in this order, the order SFE runs them in.
SfeUnit = tuple[int, int, int, int]


class SfeRunnerFailed(RuntimeError):
    """A runner could not answer for a part's units (``generate_support_footprint_grasps(runner=...)``): the part is
    searched in this process, from the start, and nothing the runner answered is kept. The runner says why, once."""


#: Who runs a part's units elsewhere (``src.robot.grasping.workers.SfeWorkers``): handed the plan and some of its units,
#: it answers every unit's grasps, by unit, and the refusals the coarse ones counted where the plan counts; or it raises
#: :class:`SfeRunnerFailed`. SFE merges what it answers as it merges its own units.
SfeRunner = Callable[["SfePlan", "Sequence[SfeUnit]"],
                     "tuple[Mapping[SfeUnit, Sequence[SupportFootprintCandidate]], Mapping[str, int]]"]


@dataclass(frozen=True, eq=False)
class SfeInputs:
    """What one part's search is worked out from, as :func:`generate_support_footprint_grasps` was handed it: the
    target's cloud and every keyword that shapes the search, and whether its refusals are counted (``counting``).

    Plain data, picklable where ``corridor_seen`` is (``scene_obstacles.SeenInTheFrame``), so that a worker process
    works out the same plan from it (:func:`plan_support_footprint`).
    """

    cloud_base_mm: np.ndarray
    support_height_mm: float = 0.0
    jaw: SupportFootprintJaw | None = None
    obstacle_points_base_mm: np.ndarray | None = None
    rigid_obstacle_points_base_mm: np.ndarray | None = None
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    height_offsets_mm: tuple[float, ...] = (4.0, 14.0)
    inflate_mm: float = 0.0
    palm_aware: bool = False
    score_weights: tuple[float, float, float, float, float] | None = None
    floor_margin_mm: float = DEFAULT_FLOOR_MARGIN_MM
    counting: bool = False
    side_approaches: bool = False
    corridor_seen: CorridorSeen | None = None
    seen_envelope: Any = None
    hand_floor: HandFloor | None = None
    #: Whether a closing line's builds are made at once (:func:`_build_many`) rather than one at a time: the same
    #: grasps, counts and stages, to the bit, in any process that runs the units.
    batched: bool = False


class _Line:
    """One closing line of a search, at one place along the part: its axis, its middle, the heights its grasps are
    anchored at, highest first, and whether the hand closes across it at all (:func:`_wider_than_the_hand_closes`)."""

    __slots__ = ("axis", "mid", "heights", "wider")

    def __init__(self, axis: np.ndarray, mid: np.ndarray, heights: tuple[float, ...], wider: bool) -> None:
        self.axis = axis
        self.mid = mid
        self.heights = heights
        self.wider = wider


class SfePlan:
    """One part's search, worked out before its first unit (:func:`plan_support_footprint`): the prism, the declared
    points with the part's own low fragments, and each search's closing axes, tilts and rolls (:attr:`searches`).

    The obstacle grids and the side approaches' room are built by the first unit that builds a grasp
    (:meth:`grids`), each line once (:meth:`line`): a process that only plans and ranks the part builds neither.
    :attr:`inputs` are what it was worked out from, so any process works out the same plan from them.
    """

    __slots__ = ("inputs", "jaw", "prism", "rigid", "searches", "_grids", "_lines", "_batch", "_ladders")

    def __init__(self, inputs: SfeInputs, jaw: SupportFootprintJaw, prism: SupportPrism,
                 rigid: np.ndarray | None) -> None:
        self.inputs = inputs
        self.jaw = jaw
        self.prism = prism
        self.rigid = rigid
        #: By :data:`COARSE`, :data:`FINE` and :data:`ROLLED`: the closing axes, the tilts and the rolls.
        self.searches: tuple[tuple[list[np.ndarray], tuple[float, ...], tuple[float, ...]], ...] = (
            (closing_axes(prism), _TILTS_DEG, (0.0,)),
            (closing_axes(prism, radial=_FINE_RADIAL, yaw_offsets_deg=_FINE_YAW_DEG), _FINE_TILTS_DEG, (0.0,)),
            (closing_axes(prism), _TILTS_DEG, _FINE_ROLL_DEG),
        )
        self._grids: "tuple[_ObstacleSet, _Side | None, bool] | None" = None
        self._lines: dict[tuple[int, int, int], _Line | None] = {}
        self._batch: "_Batch | None" = None
        self._ladders: dict[tuple[int, int], _Ladder] = {}

    def units(self, search: int) -> list[SfeUnit]:
        """Every unit of ``search``, in the order SFE runs them: each closing axis, each place along the part, each
        height. A height the line does not have is a unit that builds nothing."""
        axes = self.searches[search][0]
        return [(search, i, j, k) for i in range(len(axes)) for j in range(len(_FRACS)) for k in range(_MOST_HEIGHTS)]

    def grids(self) -> "tuple[_ObstacleSet, _Side | None, bool]":
        """The obstacle grids, the side approaches' room (``None`` without them), and whether every tilt that fits is a
        candidate: built once, by the first unit that asks."""
        if self._grids is None:
            inputs = self.inputs
            # Where the boxes the camera world builds of the neighbours are known (``seen_envelope``), the guard's own
            # distance from them decides (``seen_fingers`` in ``_build``): every point is inside its box, the world's
            # margin from its sides. The sparse grid then keeps the hand off the points alone, on a fine grid: grown 12
            # mm on 12 mm cells it refused every finger within 12 to 24 mm of a neighbour, twice what the guard keeps,
            # and lost 1100 of 1250 builds of a cylinder whose posts stood 15 mm clear of its fingers (the grasp bench,
            # 2026-10-05). Without the boxes it stands in for them as it always did.
            obstacles = _ObstacleSet(
                _Obstacles(inputs.obstacle_points_base_mm) if inputs.seen_envelope is None
                else _Obstacles(inputs.obstacle_points_base_mm, cell_mm=SEEN_POINTS_CELL_MM,
                                margin_mm=SEEN_POINTS_CELL_MM),
                _Obstacles(self.rigid, cell_mm=4.0, margin_mm=0.0),
            )
            if inputs.counting:
                obstacles.label(declared=inputs.rigid_obstacle_points_base_mm, own=self.prism.trimmed)
            side = (_Side(seen=inputs.corridor_seen, room=_Room(inputs.obstacle_points_base_mm, self.rigid))
                    if inputs.side_approaches else None)
            # Every tilt that fits is a candidate only where a depth ray can vouch for the space a tilt sweeps.
            self._grids = (obstacles, side, side is not None and side.seen is not None)
        return self._grids

    def line(self, search: int, axis_index: int, frac_index: int) -> _Line | None:
        """The closing line of ``search`` along its closing axis ``axis_index`` at place ``frac_index``, worked out once;
        ``None`` where the line misses the part."""
        key = (search, axis_index, frac_index)
        if key not in self._lines:
            self._lines[key] = self._line(search, axis_index, frac_index)
        return self._lines[key]

    def batch(self) -> _Batch:
        """What every build made at once shares (:func:`_build_many`): worked out once, by the first line that asks."""
        if self._batch is None:
            self._batch = _Batch(self)
        return self._batch

    def ladder(self, search: int, axis_index: int, axis: np.ndarray) -> _Ladder:
        """Every build of ``search`` along its closing axis ``axis_index`` (``axis``, as its lines hold it), worked out
        once for every place along the part (:func:`_ladder`)."""
        key = (search, axis_index)
        if key not in self._ladders:
            _axes, tilts, rolls = self.searches[search]
            self._ladders[key] = _ladder(self.prism, axis, tilts, rolls)
        return self._ladders[key]

    def _line(self, search: int, axis_index: int, frac_index: int) -> _Line | None:
        axes, tilts, rolls = self.searches[search]
        prism, jaw, inputs = self.prism, self.jaw, self.inputs
        support_height_mm, hand_floor = inputs.support_height_mm, inputs.hand_floor
        reach_across = jaw.aperture_mm / 2.0 + jaw.finger_thickness_mm
        raw_axis = axes[axis_index]
        a2 = raw_axis[:2] / max(float(np.linalg.norm(raw_axis[:2])), _EPS)
        axis = np.array([a2[0], a2[1], 0.0])
        perp = np.array([-a2[1], a2[0]])
        ext_perp = float((prism.hull @ perp).max() - (prism.hull @ perp).min())
        base_xy = prism.middle + _FRACS[frac_index] * (ext_perp / 2.0) * perp
        mid_z = (prism.z0 + prism.z1) / 2.0
        span_xy = prism.line_span(np.array([base_xy[0], base_xy[1], mid_z]), axis)
        if span_xy is None:
            return None
        mid = base_xy + ((span_xy[0] + span_xy[1]) / 2.0) * a2
        # Where the guard's solid under the grasp lets the hand come down to (``hand_floor``).
        floor_fingers = -math.inf if hand_floor is None else float(hand_floor.at(mid[None, :], fingers=True)[0])
        floor_palm = -math.inf if hand_floor is None else float(hand_floor.at(mid[None, :])[0])
        # Near the top, and at the part's middle height, where a grasp sits best (``_seated``).
        heights = [prism.z1 - dz for dz in inputs.height_offsets_mm] + [(prism.z0 + prism.z1) / 2.0]
        # Solve for the height at which the gripper still clears the support, per tilt. This is the step the silhouette
        # generator has no equivalent of: it anchors first and is filtered afterwards, and a filter cannot move a grasp
        # somewhere legal.
        for tilt in tilts:
            c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
            # The solve has to know what the check knows: teaching ``_build`` about the palm without teaching this
            # would only make SFE lose candidates, and the stage exists because it can move a grasp somewhere legal
            # where a filter can only refuse. The palm's own requirement is (palm_width/2)*sin - finger_behind*cos: it
            # is far below the finger term when the approach is vertical (the palm sits behind the fingers) and
            # overtakes it as the grasp tilts. The finger's own: its tip ``finger_ahead`` down the approach, and half
            # its width further where the approach tilts (``_build``'s fingers).
            base = support_height_mm + jaw.table_clearance_mm
            need = max(base, floor_fingers) + jaw.finger_ahead_mm * c + (jaw.finger_width_mm / 2.0) * s
            if inputs.palm_aware or hand_floor is not None:
                # Over the held solid the palm is checked whatever ``palm_aware`` says (``_build``), against the whole
                # solid where a finger keeps the distance from the reading.
                need = max(need, max(base, floor_palm) + (jaw.palm_width_mm / 2.0) * s - jaw.finger_behind_mm * c)
            # A rolled hand stands its lower open finger lower by its reach across the axis.
            for roll in rolls:
                rolled_need = need + reach_across * abs(math.sin(math.radians(roll)))
                if support_height_mm + 2.0 < rolled_need <= prism.z1 - 2.0:
                    heights.append(rolled_need + 0.5)
        # A height is only worth trying once; the solve often lands on the same millimetre.
        heights = sorted({round(z, 1) for z in heights
                          if support_height_mm + 2.0 < z < prism.z1}, reverse=True)[:_MOST_HEIGHTS]
        # A line the hand cannot close across is refused for the aperture whole (:func:`_wider_than_the_hand_closes`):
        # only where nothing rolls, so the closing axis stays the line.
        unrolled = all(roll == 0.0 for roll in rolls)
        wider = bool(unrolled and heights and _wider_than_the_hand_closes(prism, mid, mid_z, axis, jaw))
        return _Line(axis, mid, tuple(heights), wider)


def plan_support_footprint(inputs: SfeInputs) -> SfePlan | None:
    """Work out one part's search from ``inputs``: the prism, and the target's low fragments joined to the declared
    points. ``None`` where the cloud is too sparse to reconstruct, which a search answers with no grasp."""
    jaw = inputs.jaw or SupportFootprintJaw.from_model()
    prism = reconstruct_support_prism(inputs.cloud_base_mm, inputs.support_height_mm, inflate_mm=inputs.inflate_mm,
                                      floor_margin_mm=inputs.floor_margin_mm)
    if prism is None:
        return None
    rigid = inputs.rigid_obstacle_points_base_mm
    if prism.trimmed.shape[0]:
        rigid = (prism.trimmed if rigid is None or np.size(rigid) == 0
                 else np.vstack([np.asarray(rigid, dtype=np.float64).reshape(-1, 3), prism.trimmed]))
    return SfePlan(inputs, jaw, prism, rigid)


def _run_unit(plan: SfePlan, unit: SfeUnit, counted: "dict[str, int] | None") -> list[SupportFootprintCandidate]:
    """The grasps one unit of ``plan`` builds, in the order it builds them: every tilt, side and roll at the unit's
    height on its line, a tilt's grasps ending the ladder where it found one (unless every tilt that fits is a
    candidate). Each refused build is counted into ``counted`` where it is given."""
    search, axis_index, frac_index, height_index = unit
    line = plan.line(search, axis_index, frac_index)
    if line is None or height_index >= len(line.heights):
        return []
    _axes, tilts, rolls = plan.searches[search]
    if line.wider:
        # Every build at this height is refused for the aperture, and counted as it would have been.
        if counted is not None:
            builds = sum(1 if tilt == 0.0 else 2 for tilt in tilts) * len(rolls)
            counted["aperture"] = counted.get("aperture", 0) + builds
        return []
    obstacles, side, every_tilt = plan.grids()
    inputs, prism, jaw, axis = plan.inputs, plan.prism, plan.jaw, line.axis
    found: list[SupportFootprintCandidate] = []
    anchor = np.array([line.mid[0], line.mid[1], line.heights[height_index]])
    for tilt in tilts:
        signs = (1.0,) if tilt == 0.0 else (1.0, -1.0)
        hit = False
        for sign in signs:
            tilted = _rodrigues(np.array([0.0, 0.0, -1.0]), axis, sign * np.radians(tilt))
            tilted /= float(np.linalg.norm(tilted))
            for roll in rolls:
                rolled_axis, approach = _rolled(axis, tilted, roll)
                # The ladder's own tilt where nothing rolls: read back off the approach, 15 degrees came out 14.999999,
                # and the space the tilt sweeps went unasked.
                off_vertical = (float(tilt) if roll == 0.0 else
                                math.degrees(math.acos(max(-1.0, min(1.0, -float(approach[2]))))))
                candidate = _build(prism, anchor, rolled_axis, approach, jaw, obstacles, inputs.support_height_mm,
                                   palm_aware=inputs.palm_aware, score_weights=inputs.score_weights, refusals=counted,
                                   side=side, tilt_deg=off_vertical, seen=inputs.seen_envelope,
                                   floor=inputs.hand_floor)
                if candidate is not None:
                    found.append(candidate)
                    hit = True
        if hit and not every_tilt:
            break   # tilts are in preference order: the first that works is the one
    return found


def _line_built_many(plan: SfePlan, line_key: tuple[int, int, int], heights: Sequence[int], *,
                     counting: bool) -> "dict[SfeUnit, tuple[list[SupportFootprintCandidate], list[str]]]":
    """The units of one closing line at ``heights``, all their builds made at once (:func:`_build_many`): each unit's
    grasps, and where ``counting`` the causes its refused builds count under, in the order :func:`_run_unit` builds and
    counts them. Every build at once where every tilt that fits is a candidate; else the ladder one tilt at a time over
    the heights that found no grasp yet, so that no build is made which :func:`_run_unit`'s ladder ends before."""
    search, axis_index, frac_index = line_key
    answer: dict[SfeUnit, tuple[list[SupportFootprintCandidate], list[str]]] = {
        (search, axis_index, frac_index, height): ([], []) for height in heights}
    line = plan.line(search, axis_index, frac_index)
    if line is None:
        return answer
    standing = [height for height in heights if height < len(line.heights)]
    if not standing:
        return answer
    _axes, tilts, rolls = plan.searches[search]
    if line.wider:
        # Every build at these heights is refused for the aperture, and counted as it would have been.
        builds = sum(1 if tilt == 0.0 else 2 for tilt in tilts) * len(rolls)
        for height in standing:
            answer[(search, axis_index, frac_index, height)] = ([], ["aperture"] * builds if counting else [])
        return answer
    _obstacles, _side, every_tilt = plan.grids()
    ladder = plan.ladder(search, axis_index, line.axis)
    anchor_at = {height: np.array([line.mid[0], line.mid[1], line.heights[height]]) for height in standing}
    made: dict[int, list[tuple[SupportFootprintCandidate | None, tuple[str, ...]]]] = {h: [] for h in standing}
    open_heights = list(standing)
    for rung in ([np.arange(ladder.size)] if every_tilt else ladder.rungs):
        if not open_heights:
            break
        anchors = np.repeat(np.stack([anchor_at[height] for height in open_heights]), rung.size, axis=0)
        candidates, causes = _build_many(plan, anchors, ladder, np.tile(rung, len(open_heights)), counting=counting)
        still_open = []
        for k, height in enumerate(open_heights):
            mine = range(k * rung.size, (k + 1) * rung.size)
            made[height].extend((candidates[n], causes[n]) for n in mine)
            if all(candidates[n] is None for n in mine):
                still_open.append(height)   # tilts are in preference order: the first that works is the one
        open_heights = still_open
    for height in standing:
        answer[(search, axis_index, frac_index, height)] = (
            [candidate for candidate, _ in made[height] if candidate is not None],
            [cause for candidate, why in made[height] if candidate is None for cause in why])
    return answer


def _run_units_many(plan: SfePlan, units: Sequence[SfeUnit],
                    refusals: "dict[str, int] | None") -> dict[SfeUnit, list[SupportFootprintCandidate]]:
    """:func:`run_units` with each closing line's builds made at once (:func:`_line_built_many`): every line's heights
    asked are made first, then each unit is answered and its refused builds counted in the order given, as
    :func:`run_units` answers and counts them. Nothing is counted before every build is made, so a pass that could
    not be made leaves ``refusals`` as it was."""
    counting = plan.inputs.counting and refusals is not None
    lines: dict[tuple[int, int, int], list[int]] = {}
    for search, axis_index, frac_index, height in units:
        asked = lines.setdefault((search, axis_index, frac_index), [])
        if height not in asked:
            asked.append(height)
    made: dict[SfeUnit, tuple[list[SupportFootprintCandidate], list[str]]] = {}
    for line_key, heights in lines.items():
        made.update(_line_built_many(plan, line_key, sorted(heights), counting=counting and line_key[0] == COARSE))
    answered: dict[SfeUnit, list[SupportFootprintCandidate]] = {}
    for unit in units:
        found, causes = made[unit]
        if refusals is not None and counting and unit[0] == COARSE:
            for cause in causes:
                refusals[cause] = refusals.get(cause, 0) + 1
        answered[unit] = list(found)
    return answered


#: Why this process made SFE's builds one at a time where it was asked to make them at once, each said once.
_SAID: set[str] = set()


def run_units(plan: SfePlan, units: Sequence[SfeUnit],
              refusals: "dict[str, int] | None" = None) -> dict[SfeUnit, list[SupportFootprintCandidate]]:
    """Run ``units`` of ``plan`` here, in the order given: every unit's grasps, by unit. The refused builds of the coarse
    search are counted into ``refusals`` where the plan counts (``SfeInputs.counting``); the fine and rolled searches
    count none, so a count stays the coarse grid's, whichever part it is asked of.

    Where the plan asks for it (``SfeInputs.batched``), each closing line's builds are made at once
    (:func:`_build_many`): the same grasps and counts, to the bit. Where this process's numpy cannot make them so
    (:func:`_stacked_products_hold`), or making them so fails, the units are run one build at a time from the start,
    said once in the log, and the answer is the one of before."""
    counting = plan.inputs.counting and refusals is not None
    if plan.inputs.batched:
        try:
            why = plan.batch().refused
            if not why:
                return _run_units_many(plan, units, refusals if counting else None)
        except Exception as exc:  # noqa: BLE001 (one build at a time answers instead, from the start)
            why = f"making them at once failed ({type(exc).__name__}: {exc})"
        if why not in _SAID:
            _SAID.add(why)
            _LOG.warning("SFE makes its builds one at a time: %s; the grasps are the same", why)
    return {unit: _run_unit(plan, unit, refusals if counting and unit[0] == COARSE else None) for unit in units}


def _holds_at(prism: SupportPrism, anchor: np.ndarray, axis: np.ndarray, approach: np.ndarray,
              binormal: np.ndarray, jaw: SupportFootprintJaw) -> bool:
    """Whether a closing line at ``anchor`` passes the checks a build starts with (:func:`_build`): the pads meet the
    part, the anchor stands between them, the span fits the stroke, and both contacts lie inside the friction cone."""
    contacts = _pad_contacts(prism, anchor, axis, approach, binormal, jaw)
    if contacts is None:
        return False
    t_enter, t_exit, contact_b, contact_a, _low = contacts
    span = t_exit - t_enter
    if not (t_enter - 2.0 <= 0.0 <= t_exit + 2.0):
        return False
    if span > jaw.aperture_mm - jaw.width_safety_mm or span < jaw.min_width_mm:
        return False
    worst = max(float(np.arccos(np.clip(prism.normal_at(contact_a) @ axis, -1.0, 1.0))),
                float(np.arccos(np.clip(prism.normal_at(contact_b) @ -axis, -1.0, 1.0))))
    return worst <= jaw.cone_rad


def robust_to(prism: SupportPrism, jaw: SupportFootprintJaw, candidate: SupportFootprintCandidate,
              error_mm: float = ROBUST_ERROR_MM) -> bool:
    """Whether ``candidate`` still holds with the hand ``error_mm`` off it across the table.

    Two ways a hand that stands off its grasp loses it. Along the closing axis an open finger comes down on the part:
    each must stand ``error_mm`` clear of the part's face it closes on, half the stroke beyond the contact. Across the
    closing axis the line slides along the faces: moved ``error_mm`` either way, level, it must still meet the part
    within the stroke and the friction cone (:func:`_holds_at`). The neighbours are not asked again: the guard keeps
    its distance from them at the grasp, and the build kept the open hand off them.

    Args:
        prism (SupportPrism): The part's prism, as the search built its grasps on it.
        jaw (SupportFootprintJaw): The hand the search planned for.
        candidate (SupportFootprintCandidate): The grasp, BASE millimetres.
        error_mm (float): How far off across the table the hand may stand, millimetres; default
            :data:`ROBUST_ERROR_MM`.

    Returns:
        bool: ``True`` where both hold, ``False`` where either does not or the grasp meets the part nowhere.
    """
    anchor = np.asarray(candidate.position_mm, dtype=np.float64)
    axis = np.asarray(candidate.closing_axis, dtype=np.float64)
    approach = np.asarray(candidate.approach, dtype=np.float64)
    binormal = _cross3(approach, axis)
    binormal /= max(float(np.linalg.norm(binormal)), _EPS)
    contacts = _pad_contacts(prism, anchor, axis, approach, binormal, jaw)
    if contacts is None:
        return False
    t_enter, t_exit = contacts[0], contacts[1]
    if jaw.aperture_mm / 2.0 - max(-t_enter, t_exit) < error_mm:
        return False
    across = np.array([-axis[1], axis[0], 0.0])
    length = float(np.linalg.norm(across))
    if length < _EPS:
        return False
    across /= length
    return all(_holds_at(prism, anchor + side * error_mm * across, axis, approach, binormal, jaw)
               for side in (1.0, -1.0))


def rank_found(plan: SfePlan, found: Sequence[SupportFootprintCandidate]) -> list[SupportFootprintCandidate]:
    """The search's end: ``found`` ranked, by score or with side approaches each run within :data:`_TIE_SCORE` of its
    best vertical first (:func:`_upright_first`), then the robust grasps first (:func:`robust_to`), each in that order,
    a grasp near one kept before it left out, ``max_candidates`` at most; each kept says whether it is robust
    (:attr:`SupportFootprintCandidate.robust`).

    Robust first, the owner's "Griffe robust ordnen" (2026-10-09): at a real cell the hand stands 3 to 5 mm off the
    grasp it was sent to, and among the grasps SFE finds valid the order barely mattered without that error (+0.29
    points on the reference, ``_SCORE_WEIGHTS``). MEASURED on the desk the same day: on the held-out Isaac shard (273
    object views with a grasp ranked first, a gripper opening 150 mm) and on the 23 cubes the owner's cell recorded
    (the Hand-E's 49.99 mm), it changed the grasp ranked first in none; it reorders the grasps after it, which the pick
    loop takes where the guard refuses the first. 1.7 ms a search on a cube.
    """
    ranked = sorted(found, key=lambda c: -c.score) if not plan.inputs.side_approaches else _upright_first(list(found))
    most = plan.inputs.max_candidates
    kept: list[SupportFootprintCandidate] = []
    fragile: list[SupportFootprintCandidate] = []
    # Each grasp is asked as it comes, and none once the robust ones fill the list: the same list as asking every grasp
    # first, and where a part's best grasps all hold (a cube's, nearly always) a dozen are asked instead of hundreds.
    for candidate in ranked:
        if len(kept) >= most:
            break
        if robust_to(plan.prism, plan.jaw, candidate):
            candidate = dataclasses.replace(candidate, robust=True)
            if not _near_one_of(candidate, kept):
                kept.append(candidate)
        else:
            fragile.append(candidate)
    for candidate in fragile:
        if len(kept) >= most:
            break
        if not _near_one_of(candidate, kept):
            kept.append(candidate)
    if plan.inputs.side_approaches:
        kept = _with_a_vertical(plan, kept, ranked)
    return kept


#: A grasp whose approach stands within 1 degree of straight down is vertical (``approach[2]`` is -cos(tilt)).
_VERTICAL_APPROACH_Z: Final[float] = -float(np.cos(np.radians(1.0)))


def _with_a_vertical(plan: SfePlan, kept: list[SupportFootprintCandidate],
                     ranked: Sequence[SupportFootprintCandidate]) -> list[SupportFootprintCandidate]:
    """``kept`` with the best vertical grasp of ``ranked`` among them: where none made the list, it takes the last place.

    A tilt the arm cannot take is often refused with every other tilt of its line (the wrist, its camera, the reach),
    and the grasp straight down is the one the pick loop then still has to try. The prototype beside a wall (2026-10-01)
    ranked a grasp tilted away from it first with a vertical one still listed, and the list kept it by its score alone
    until the fan of a round footprint stood in the base frame (2026-10-09): the cylinder 12 mm off a wall then filled
    its twelve places with tilts 30 to 75 degrees of the line closing along the wall, and the vertical grasp ranked 13th
    by 0.009 of score, the noise of the cylinder's sampled facets. Rank 0 never changes.
    """
    if not kept or any(float(c.approach[2]) <= _VERTICAL_APPROACH_Z for c in kept):
        return kept
    best = next((c for c in ranked if float(c.approach[2]) <= _VERTICAL_APPROACH_Z and not _near_one_of(c, kept)), None)
    if best is None:
        return kept
    best = dataclasses.replace(best, robust=robust_to(plan.prism, plan.jaw, best))
    most = plan.inputs.max_candidates
    return (kept[:most - 1] if len(kept) >= most else kept) + [best]


def _near_one_of(candidate: SupportFootprintCandidate, kept: Sequence[SupportFootprintCandidate]) -> bool:
    """Whether ``candidate`` is a grasp near one of ``kept``: within 4 mm, its closing axis and approach the same."""
    return any(
        float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
        and abs(float(candidate.closing_axis @ k.closing_axis)) > 0.995
        and float(candidate.approach @ k.approach) > 0.995
        for k in kept
    )


def _merged(by_unit: Mapping[SfeUnit, Sequence[SupportFootprintCandidate]]) -> list[SupportFootprintCandidate]:
    """Every unit's grasps, the units in their own order: the order SFE builds them in, whoever ran which."""
    return [candidate for unit in sorted(by_unit) for candidate in by_unit[unit]]


def _units_run(plan: SfePlan, units: list[SfeUnit], refusals: "dict[str, int] | None",
               runner: "SfeRunner | None") -> dict[SfeUnit, list[SupportFootprintCandidate]]:
    """``units`` run here (``runner`` ``None``) or by ``runner``, whose counts join ``refusals``."""
    if runner is None:
        return run_units(plan, units, refusals)
    by_unit, counted = runner(plan, units)
    missing = [unit for unit in units if unit not in by_unit]
    if missing:
        # Never a part of a search: the part is searched here in full instead.
        raise SfeRunnerFailed(f"the runner answered {len(units) - len(missing)} of {len(units)} units")
    if refusals is not None:
        for cause, count in counted.items():
            refusals[cause] = refusals.get(cause, 0) + int(count)
    return {unit: list(by_unit[unit]) for unit in units}


def _searched(plan: SfePlan, *, fine_pass: bool, refusals: "dict[str, int] | None", stages: "dict[str, str] | None",
              runner: "SfeRunner | None") -> list[SupportFootprintCandidate]:
    """The search of ``plan``: its coarse units; the fine and rolled ones where the coarse grid found fewer than
    ``_FINE_BELOW`` grasps and ``fine_pass`` does not leave them for later; each merged in unit order, then ranked."""
    found = _merged(_units_run(plan, plan.units(COARSE), refusals, runner))
    if len(found) < _FINE_BELOW and not fine_pass:
        # Few grasps on the coarse grid, and the caller holds the fine search back: the coarse grid's grasps, said so.
        if stages is not None:
            stages["fine"] = FINE_SEARCH_DEFERRED
    elif len(found) < _FINE_BELOW:
        # Few grasps on the coarse grid: the finer one, every 7.5 degrees of tilt, each face's closing axis turned
        # inside the friction cone, a round part's fan every 15 degrees (the owner, 2026-10-06: "die Orientierung
        # feiner"); and each face's closing axis tilted out of the horizontal, one finger higher than the other, on the
        # coarse ladder of tilts (the owner, 2026-10-06: "die geneigte Schliessachse"). Their refusals are not counted,
        # so a count stays the coarse grid's, whichever part it is asked of.
        found += _merged(_units_run(plan, plan.units(FINE) + plan.units(ROLLED), None, runner))
    return rank_found(plan, found)


def generate_support_footprint_grasps(
    cloud_base_mm: np.ndarray,
    *,
    support_height_mm: float = 0.0,
    jaw: SupportFootprintJaw | None = None,
    obstacle_points_base_mm: np.ndarray | None = None,
    rigid_obstacle_points_base_mm: np.ndarray | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    height_offsets_mm: tuple[float, ...] = (4.0, 14.0),
    inflate_mm: float = 0.0,
    palm_aware: bool = False,
    score_weights: tuple[float, float, float, float, float] | None = None,
    floor_margin_mm: float = DEFAULT_FLOOR_MARGIN_MM,
    refusals: dict[str, int] | None = None,
    side_approaches: bool = False,
    corridor_seen: CorridorSeen | None = None,
    seen_envelope: Any = None,
    hand_floor: HandFloor | None = None,
    fine_pass: bool = True,
    stages: dict[str, str] | None = None,
    runner: "SfeRunner | None" = None,
    batched: bool = False,
) -> list[SupportFootprintCandidate]:
    """Ranked candidates in BASE, from a masked target cloud and the rest of the scene as obstacles.

    ``obstacle_points_base_mm`` is observed geometry, the neighbours' depth points, and is dilated
    to cover the gaps between samples. ``rigid_obstacle_points_base_mm`` is declared geometry, today
    the container walls, and is not: it is already exact, and dilating it would forbid a band of the
    bin that a gripper can legally reach. See :class:`_ObstacleSet`.

    ``floor_margin_mm`` is how far above the support a target point still counts as the support; see
    ``DEFAULT_FLOOR_MARGIN_MM``. Low fragments the reconstruction leaves out of the footprint join the
    undilated obstacles, so a finger is still kept off everything the camera saw above the floor.
    Undilated, because they were sampled at the target's own density rather than being the sparse view
    of a neighbour whose gaps the dilation covers; on the 45-degree ray-cast behind
    :func:`reconstruct_support_prism` either grid gave the same candidates, 2026-09-23.

    ``refusals``, where given, is filled with one count per refused build under its cause, every
    cause of :data:`REFUSAL_CAUSES` present and counts added to what the dict already holds. An
    obstacle refusal counts once under each set it met, so one build that meets a seen and a declared
    point counts under both. ``None`` counts nothing and is byte-identical to the generator before the
    counters.

    ``side_approaches`` offers every tilt the hand fits at instead of the first, and ranks by
    :data:`SIDE_APPROACH_SCORE_WEIGHTS` (upright weighs nothing; clearance is the room the grasp keeps,
    to the support and to every obstacle point), sorted by score and each run within 0.001 of its best by tilt, so
    vertical wins a tie. A candidate tilted 15 degrees or more whose open corridor passes a point
    ``corridor_seen`` says no depth ray reached is refused (``unseen_corridor``). Without
    ``corridor_seen`` no tilt past the first that fits is offered: the candidates are the ones the
    generator offers with side approaches off, ranked as above. Off, the default, is byte-identical
    to the generator before side approaches.

    ``hand_floor`` (:class:`HandFloor`) holds the solids the guard holds under the hand: no point of the hand comes
    nearer their top than the guard's distance (``table``), and the height solve starts from it, so a grasp sits as low
    as the guard admits and no lower. ``None`` holds the support clearance over the reading alone, as before.

    ``fine_pass`` off leaves the fine search for later: where the coarse grid finds fewer than ``_FINE_BELOW`` grasps,
    the coarse grid's grasps are returned as they are and ``stages``, where given, says so
    (``stages["fine"] = FINE_SEARCH_DEFERRED``). The fine search is most of what a part with few grasps costs (88 to
    93 % of the time on a cylinder boxed in by four cubes; it is what made one part on the owner's cell take 24 s,
    2026-10-08), and a pick that already holds another part's full result need not wait for it. On, the default, the
    search is the one it always was, and ``stages`` is left as it was given.

    The search is one function of independent units (:data:`SfeUnit`): it is planned (:func:`plan_support_footprint`),
    its units run (:func:`run_units`), merged in unit order and ranked (:func:`rank_found`). ``runner``
    (:data:`SfeRunner`, ``src.robot.grasping.workers.SfeWorkers``) runs them in other processes; ``None``, the default,
    runs every unit here, in order. The coarse units go first and the fine and rolled ones only where the merged coarse
    grasps ask for them, so a runner gives the same grasps, the same counts and the same ``stages`` as this process. A
    runner that cannot answer (:class:`SfeRunnerFailed`) has the part searched here from the start, and nothing it
    answered is kept.

    ``batched`` makes each closing line's builds at once, in one numpy pass per check (:func:`_build_many`), wherever
    the units run, a runner's processes included: the same grasps, counts and ``stages`` to the bit, and a build whose
    deciding number lies within :data:`_BATCH_SURE` of its threshold made by ``_build`` alone. The recorded boxed-in
    Zollstock's full search took 0.22 instead of 2.2 s on the desk, a boxed-in cube's 0.24 instead of 3.7 s
    (2026-10-09). Off, the default, makes every build one at a time, as before.

    Returns an empty list when the cloud is too sparse to reconstruct or admits no legal grasp,
    which is a real answer and not a failure. Refusing beats proposing a grasp that goes under the
    support surface.
    """
    if refusals is not None:
        for cause in REFUSAL_CAUSES:
            refusals.setdefault(cause, 0)
    plan = plan_support_footprint(SfeInputs(
        cloud_base_mm, support_height_mm=support_height_mm, jaw=jaw, obstacle_points_base_mm=obstacle_points_base_mm,
        rigid_obstacle_points_base_mm=rigid_obstacle_points_base_mm, max_candidates=max_candidates,
        height_offsets_mm=height_offsets_mm, inflate_mm=inflate_mm, palm_aware=palm_aware, score_weights=score_weights,
        floor_margin_mm=floor_margin_mm, counting=refusals is not None, side_approaches=side_approaches,
        corridor_seen=corridor_seen, seen_envelope=seen_envelope, hand_floor=hand_floor, batched=bool(batched)))
    if plan is None:
        return []
    if runner is not None:
        counted: dict[str, int] = {}
        said: dict[str, str] = {}
        try:
            kept = _searched(plan, fine_pass=fine_pass, refusals=counted if refusals is not None else None,
                             stages=said, runner=runner)
        except SfeRunnerFailed:
            pass   # the runner said why: the part is searched here, from the start, and nothing it answered is kept
        else:
            if refusals is not None:
                for cause, count in counted.items():
                    refusals[cause] = refusals.get(cause, 0) + count
            if stages is not None:
                stages.update(said)
            return kept
    return _searched(plan, fine_pass=fine_pass, refusals=refusals, stages=stages, runner=None)
