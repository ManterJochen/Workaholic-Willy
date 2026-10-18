"""The hand the support-footprint stage plans with: the jaw, the floor and the room it keeps, the grasp it offers.

What the part's own body (``body_grasps``) and its bores (``inner_grasps``) are searched with:

* :class:`SupportFootprintJaw`, the hand as the stage plans it: its fingers from the cell's hand file or its collision
  envelope, its stroke and its clearance over the support, with :data:`JAW_SIDE_ROOM_MM` beside each jaw;
* :class:`HandFloor`, the solids the guard holds under the hand, and how low the hand may come over them;
* the obstacle grids (``_Obstacles``, ``_ObstacleSet``) and the room the open fingers keep (``_Room``);
* the finger and palm geometry of a build and the hand rolled about its binormal (``_finger_points``,
  ``_palm_corners``, ``_palm_low_mm``, ``_rolled``);
* the ladders the hand leans on (``_TILTS_DEG``, ``_FINE_TILTS_DEG``), the scores a grasp is ranked by
  (``_SCORE_WEIGHTS``, :data:`SIDE_APPROACH_SCORE_WEIGHTS`) and the tie a vertical wins (``_upright_first``);
* :class:`SupportFootprintCandidate`, the grasp the stage offers, in BASE.

They were the support-footprint enumeration's (SFE), whose own search planned on a convex prism of the part and left
on 2026-10-16 (the owner's cleanup), once the part's body had been the only grasps the stage offers since 2026-10-14.
They moved here unchanged, every number and every check as it was.

The gripper dimensions are taken from :class:`ParallelJawGripperModel` rather than restated here. A restated copy
drifts: an independent one is about 8 mm too lenient toward the table. There is one measurement and every consumer
reads it.

Pure and deterministic: numpy, and SciPy's KD-tree where present for the room the open fingers keep (a NumPy scan
otherwise, the same answer). BASE millimetres throughout, Z up along the support normal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Final, Protocol, TypeVar

from dataclasses import dataclass

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.collision import ParallelJawGripperModel
from src.robot.grasping.collision.support_resolver import TABLE_CLEARANCE_MM
from src.robot.grasping.geometry.grasp_frame import pose_from_grasp_axes

_EPS = 1e-9

#: How many candidates ``Scene.grasps`` returns at most when the caller does not say, best first: a closing axis it is
#: asked for is chosen before the cap, not after it.
DEFAULT_MAX_CANDIDATES: Final[int] = 12

#: The cell of the grid that keeps the hand off the neighbours' points where the boxes the camera world builds of them
#: decide (``seen_envelope``), millimetres, grown by one cell: 2 to 4 mm off a point, under the guard's distance from a
#: fine box its margin grew round the same point (``fine_world``, 2 mm and the guard's 2 mm). At 4 mm, 4 to 8 mm off a
#: point, the grid refused grasps the fine world admits.
SEEN_POINTS_CELL_MM: Final[float] = 2.0

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
#: How near a solid's top face, or the edge of the face it stands over, a point may lie before :meth:`HandFloor.under_many`,
#: which asks a batch of grasps at once, leaves that grasp to :meth:`HandFloor.under` alone, millimetres: a product over
#: many points rounds its last bit, about 1e-13 mm here, as it likes.
_BATCH_SURE: Final[float] = 1e-6

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
    (``band_mm``, ``allowance_mm``), the reading and its excess, as the guard holds it in every cell's camera world
    (the owner's "wir müssen tiefer gehen" of 2026-10-05); every other
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
        over many points than over one build's (the part's body asks a batch of its grasps at once,
        ``body_grasps._kept``), and the band covers that.
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
#: (``body_grasps``) and be ranked first. The owner, 2026-10-09: 3 to 5 mm is what a real cell gets; the two looks of
#: the one recorded pick seen from two poses (2026-10-07) lay 5.4 mm apart at the median, about 3.8 mm each.
ROBUST_ERROR_MM: Final[float] = 4.0
#: How much room a grasp leaves each jaw beside the part as the open hand comes down, millimetres: a part is closed
#: across only where the stroke spans it with this much to spare on either side. The owner's 2 mm (2026-10-13), a little
#: over the hand-eye calibration's own spread (1.6 mm RMS over its 20 stations, 2.3 mm at most), so a jaw that comes down
#: as far off as the cell's error puts it still clears the part's edge. Until then the silhouette path offered spans up
#: to the stroke itself: at 15:14 that day a 48.9 mm metal part under the 50 mm Hand-E, half a millimetre a side, and
#: the hand closed on nothing. Every source of candidates reads this number: SFE twice over as its ``width_safety_mm``,
#: the silhouette and antipodal paths as the widest span they offer, and the pick loop's lazy ranking.
JAW_SIDE_ROOM_MM: Final[float] = 2.0
#: How far beside each jaw a hand that opens only as needed (lever G, a Robotiq hand told a width) opens past the part
#: before it comes down, millimetres: the stroke's spare and 5 mm more, 1 cm more in all. Opened 2 mm beside each jaw,
#: the owner's Robotiq came down onto the parts it was to close on and many grasps broke (2026-10-14: "eine Marge von
#: 1cm mehr"). A span this leaves no room for is grasped with the hand fully open.
NARROW_ROOM_MM: Final[float] = JAW_SIDE_ROOM_MM + 5.0
#: A candidate tilted this far off vertical or more is offered, with side approaches, only through space a depth ray saw.
_SEEN_FROM_TILT_DEG: Final[float] = 15.0
#: Scores this near the best of a run are equal by geometry, and the run is ranked vertical first. A face's normal read
#: off a noisy footprint moves the contact angle by hundredths of a degree from one anchor to the next, 0.0008 of score
#: on the straight-down cube of ``tests/robot/grasping/generation/test_a_tilted_view_grasps_on_the_part.py``; rounded to 0.001 the two sat either
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
    #: The stroke a span leaves spare, both sides together (:data:`JAW_SIDE_ROOM_MM` on either side).
    width_safety_mm: float = 2.0 * JAW_SIDE_ROOM_MM
    #: The palm. It sits behind the fingers, so on a straight-down grasp it is nowhere near the
    #: table; it is 70 mm wide against the fingers' 27, so as the approach tilts and the binormal
    #: swings toward vertical the palm becomes the lowest part of the gripper. The collision
    #: envelope always carries it, and the part's body keeps it over the support at every lean
    #: (``_palm_low_mm``), as SFE did only under its ``palm_aware``.
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
        width_safety_mm: float = 2.0 * JAW_SIDE_ROOM_MM,
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

        The fingers come from the named hand's registry file through the same collision envelope the cell hands
        the calculator (``gripper_model.jaw_envelope``), field for field. The stroke comes from
        ``robot.gripper``, which the cell hands the calculator as its grip widths, and the table
        clearance, the guard's 1 mm over the support (``support_resolver.TABLE_CLEARANCE_MM``), which the pick loop
        hands it per call.

        Until 2026-09-23 ``Scene.from_robot_config`` built its jaw from the stroke alone, so every
        finger number fell back to ``ParallelJawGripperModel()``, which is a 2F-85. MEASURED on the
        ``hande`` profile: a vertical approach then needs the anchor 62.4 mm above the support instead
        of 25.9 mm, so a 40 mm cube got only 90-degree side approaches 36 mm up, where the 75 mm Hand-E
        housing reaches below the table, and a 30 mm part got no grasp at all.
        """
        # Where the guard holds the jaw: past the TCP by as much as the registry and the plates put it
        # (``planning.hand.hand_past_the_tcp_mm``), as the cell's calculator plans it.
        from src.robot.grasping.collision.gripper_model import jaw_envelope  # noqa: PLC0415
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm  # noqa: PLC0415 (config side only)

        model = jaw_envelope(robot_config, past_the_tcp_mm=hand_past_the_tcp_mm(robot_config))
        return cls.from_model(
            model,
            aperture_mm=float(robot_config.gripper.max_width_mm),
            min_width_mm=float(robot_config.gripper.min_width_mm),
            table_clearance_mm=TABLE_CLEARANCE_MM,
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
    #: Whether the grasp still holds with the hand :data:`ROBUST_ERROR_MM` off it across the table: set where the body's
    #: search ranks it (``body_grasps``), which puts these first; ``False`` until then.
    robust: bool = False
    #: How wide the jaws open before they close, millimetres, where the grasp was planned for jaws that open only as
    #: needed (``body_grasps``, lever G); ``None`` for the whole stroke, as every grasp of SFE's.
    open_width_mm: float | None = None

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


# --------------------------------------------------------------------------- obstacles


#: The most cells an obstacle grid is dilated on as a whole grid of its extent (:class:`_Obstacles`), a byte each;
#: a larger one lists each cell's neighbours instead.
_DENSE_CELLS_MOST: Final[int] = 1 << 25


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
        cells = np.floor((p - self.origin) / cell_mm).astype(np.int64)
        self.dims = cells.max(axis=0) + r + 2
        if int(np.prod(self.dims)) <= _DENSE_CELLS_MOST:
            # On a grid of the whole extent, dilated axis by axis: the same cells as listing every cell's cube of
            # neighbours (every one of them lies inside ``dims``), in the order of their keys. Listed, the 2 mm grid of
            # a crate's 160,000 points took 0.2 s a build (2026-10-14).
            grid = np.zeros(tuple(int(v) for v in self.dims), dtype=bool)
            grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
            for axis in range(3):
                for _ in range(r):
                    grown = grid.copy()
                    lead = [slice(None)] * 3
                    lag = [slice(None)] * 3
                    lead[axis], lag[axis] = slice(1, None), slice(None, -1)
                    grown[tuple(lead)] |= grid[tuple(lag)]
                    grown[tuple(lag)] |= grid[tuple(lead)]
                    grid = grown
            self.keys = np.flatnonzero(grid)
            return
        k = np.unique(cells, axis=0)
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


class _Ranked(Protocol):
    """What :func:`_upright_first` ranks by: SFE's candidates and the body search's grasps alike."""

    @property
    def score(self) -> float: ...

    @property
    def approach(self) -> np.ndarray: ...


_RankedT = TypeVar("_RankedT", bound=_Ranked)


def _upright_first(found: list[_RankedT]) -> list[_RankedT]:
    """``found`` by score, each run within :data:`_TIE_SCORE` of its best ranked vertical first: the owner's approaches
    equal by geometry (2026-10-01), so vertical wins a tie. ``approach[2]`` is -cos(tilt) and rises with it; rounded, so
    float noise cannot reorder one rung of the ladder."""
    by_score = sorted(found, key=lambda c: -c.score)
    ranked: list[_RankedT] = []
    start = 0
    while start < len(by_score):
        end = start
        while end < len(by_score) and by_score[end].score >= by_score[start].score - _TIE_SCORE:
            end += 1
        ranked.extend(sorted(by_score[start:end], key=lambda c: round(float(c.approach[2]), 9)))
        start = end
    return ranked


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
