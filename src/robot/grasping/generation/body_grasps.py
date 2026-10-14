"""Grasps on the part's own body: what the cameras measured of it, not its convex footprint (lever A, 2026-10-13).

SFE (``support_footprint``) plans on a convex vertical prism: the part's footprint is its convex hull, every side a
vertical wall, the top one height. On the owner's Tuesday picks (2026-10-13) 91 to 99 % of the parts asked in a pile, a
crate and among the industrial parts got no grasp, and three of SFE's causes are the prism's own: a ring, a star or an
L is wider as a hull than the Hand-E opens (``aperture``, 400,000 builds of the industrial parts alone), a lying cylinder
has no vertical wall where the hull says it has one (``cone``, and a grasp 15 mm over its equator the cell saw slip), and
a bore or a gap between two lobes, where a finger fits, is solid in a hull.

This stage plans on the part as the cameras saw it. Its points stand on a 2 mm grid in BASE x and y
(:data:`BODY_CELL_MM`), each cell as high as the highest of them, the solid under it carried down to the support as the
camera world carries a neighbour down. For every closing direction (every :data:`BODY_YAW_STEP_DEG`), every place across
the part (every cell) and every height the pads can stand at (every :data:`BODY_HEIGHT_STEP_MM`, from
:data:`HOLD_OVERLAP_MM` into the part's top, or as deep as the support lets the fingertips go on a lower part, down to
the fingertips the support's clearance over it):

* the pads meet the body where it stands at least :data:`BODY_MIN_OVERLAP_MM` into the pad's height, over the finger's
  width; the pads close on one run of it along the closing axis, or on two or three runs and what lies between them, so
  one wall of a ring, one lobe of a star or one leg of an L is a grasp of its own (:func:`_spans`);
* the span fits the stroke with :data:`support_footprint.JAW_SIDE_ROOM_MM` either side, and no part of the body stands
  over the fingertips where the open fingers come down or between them and the span while they close;
* both contacts hold inside the friction cone, by the body's own flanks: how steeply it rises at the contact (a lying
  cylinder over its equator rises too gently) and how its outline turns across the pad there, the face the pad comes
  to lie on as the squeeze turns the part (a cube across its diagonal is a corner);
* the closed and the open hand keep what SFE keeps (``support_footprint._build``): the support's clearance, the solids
  the guard holds under the hand (``HandFloor``), the neighbours' points and the guard's boxes of them
  (``scene_obstacles.SeenEnvelope``), the declared bodies and the part's own low fragments;
* an open finger over the part's own outline, its convex hull in the plane, comes down only through space a ray of the
  frame reached (``scene_obstacles.SeenInTheFrame``, SFE's corridor test): a bore, a gap between two lobes or the space
  under one is free only where the camera saw it free, for the guard does not hold the part it grasps;
* where the scene refuses a grasp straight down (a wall, a neighbour over the fingers' way, the housing at either), the
  same contacts are asked with the hand leaning about the closing axis, :data:`BODY_TILTS_DEG` either way, the least
  tilt that keeps the scene taken: its fingers clear of the part's own body, its whole way in seen (as SFE asks a side
  approach), scored upright by the cosine of its tilt as SFE scores one.

What it offers is a :class:`support_footprint.SupportFootprintCandidate`, scored as SFE scored a side approach (its
weights: the friction cone's slack, the stroke left, how the pads sit on the part (:data:`_HOLD_SHARE` of that third)
and the room the open fingers keep from every obstacle point (the rest of it), upright, the closing line across the
part's centre of mass), robust first. Ranked beside SFE's own
until 2026-10-14 (:func:`merged_ranking`, the owner's "Beide, gemeinsam gerankt" of 2026-10-13); since then the stage
offers these alone (``_support_footprint_stage``).

Pure: numpy and SciPy's image tools. BASE millimetres throughout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np

from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
from src.robot.grasping.generation.support_footprint import (
    _CLEARANCE_FULL_MM,
    _SCORE_WEIGHTS,
    NARROW_ROOM_MM,
    SEEN_POINTS_CELL_MM,
    HandFloor,
    SupportFootprintCandidate,
    SupportFootprintJaw,
    _finger_points,
    _ObstacleSet,
    _Obstacles,
    _palm_corners,
    _Room,
    _trim_low_fragments,
)

__all__ = [
    "BODY_CELL_MM",
    "BODY_HEIGHT_STEP_MM",
    "BODY_MIN_OVERLAP_MM",
    "HOLD_OVERLAP_MM",
    "BODY_TILTS_DEG",
    "BODY_YAW_STEP_DEG",
    "BodyMap",
    "body_grasps",
    "body_least_height_mm",
    "body_map",
    "merged_ranking",
]

#: The body's grid in BASE x and y, millimetres: about the D415's own reading of a part at 0.5 to 0.8 m.
BODY_CELL_MM: Final[float] = 2.0
#: The closing directions, every this many degrees over half a turn.
BODY_YAW_STEP_DEG: Final[float] = 5.0
#: The heights a grasp is anchored at, this far apart, millimetres; at most :data:`_MOST_HEIGHTS` of them.
BODY_HEIGHT_STEP_MM: Final[float] = 2.0
#: How far into the pad's height the body has to stand for the pad to hold on it, millimetres.
BODY_MIN_OVERLAP_MM: Final[float] = 3.0
#: How far below a part's top the pads hold it at the least where it stands tall enough, millimetres: about half the
#: Hand-E's pad (the owner's cell of 2026-10-14, "sehr knapp an der Oberseite" and "Zylinder nicht in Backen" where
#: the grasps came down 6 to 15 mm into the part's top). A lower part is held as deep as its support lets the
#: fingertips go, and never by less than :data:`BODY_MIN_OVERLAP_MM`.
HOLD_OVERLAP_MM: Final[float] = 10.0
#: How much of the score's third that weighs how a grasp holds the part is how the pads sit on it; the rest is the room
#: the open fingers keep. Half and half, a grasp over the neighbours' tops outranked one deep in the part by its room.
_HOLD_SHARE: Final[float] = 0.6
#: How far the hand leans about the closing axis, either way, where the scene refuses a grasp straight down, degrees,
#: the least first: SFE's ladder to 45 (``support_footprint._TILTS_DEG``), past which a finger's corner reaches under a
#: part this hand holds.
BODY_TILTS_DEG: Final[tuple[float, ...]] = (15.0, 30.0, 45.0)
#: How many of a part's grasps the scene refused straight down are asked at each tilt, the best of each closing axis and
#: place across it (:data:`_LEAN_BAND_MM`) before the second of any, of those whose fingers clear the support leaning so
#: far: on the deep crate of 2026-10-12, every refused grasp of every part leaning took 25 s of a pick's 80.
_TILTED_MOST: Final[int] = 48
#: How wide a band across the closing axis holds one place for the leaning grasps, millimetres: under a beam over half
#: a part the grasps a lean mends stand under it, and the best refused ones beside it.
_LEAN_BAND_MM: Final[float] = 8.0
#: What the scene refuses a grasp straight down for that a lean can mend: a neighbour or a wall on the open fingers' way
#: in, at the closed fingers or the hand (the guard's boxes), or a solid under the hand. A finger over the part's outline
#: where no ray reached is not: leaning, the whole way in has to be seen.
_A_LEAN_MENDS: Final[frozenset[str]] = frozenset({"seen_corridor", "seen_fingers", "table"})
#: The most runs of the body along one closing line the pads close on together: a ring's two walls and its bore.
_MOST_RUNS: Final[int] = 3
#: The most anchor heights a direction is asked at.
_MOST_HEIGHTS: Final[int] = 16
#: How many grasps that hold on the part are checked against the rest of the scene at a time, best first, until
#: :data:`_ENOUGH` places pass of each opening the hand is offered at (:func:`_place`: a place a few grasps a cell apart
#: pass at fills none of the others; a narrow grasp's place none of the fully open ones, which rank after them).
_BATCH: Final[int] = 32
_ENOUGH: Final[int] = 24
#: How far up the pad from where it meets the body the body's outline is read again, for how it leans, millimetres.
_LEAN_RISE_MM: Final[float] = 4.0
#: How far the body's outline is read either side of a contact, along the pad, for how it turns there, in cells.
_OUTLINE_CELLS: Final[int] = 3
#: How far off its grasp across the table the hand may stand for the grasp to be robust, millimetres
#: (``support_footprint.ROBUST_ERROR_MM``).
_ROBUST_MM: Final[float] = 4.0
_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class BodyMap:
    """The part's body on a grid square with BASE: each cell as high as the highest point in it, ``-inf`` where none.

    ``origin_mm`` is the low corner of cell ``(0, 0)``; ``top[i, j]`` the cell ``i`` along x and ``j`` along y.
    """

    origin_mm: tuple[float, float]
    cell_mm: float
    top: np.ndarray
    support_mm: float
    #: The part's height, the 98th percentile of its points: where the pads sit best (``support_footprint._seated``).
    top_mm: float
    #: The part's low fragments the body left out, ``(N, 3)``: kept as obstacles, as SFE keeps them.
    trimmed: np.ndarray
    #: The body's points, ``(N, 3)``, and the top of the cell each stands in: where a pad meets them exactly.
    points: np.ndarray
    point_tops: np.ndarray

    def centres(self) -> "tuple[np.ndarray, np.ndarray]":
        """The BASE x and y of every seen cell's centre and its top, ``((N, 2), (N,))``."""
        index = np.argwhere(np.isfinite(self.top))
        xy = (index + 0.5) * self.cell_mm + np.asarray(self.origin_mm)
        return xy, self.top[index[:, 0], index[:, 1]]


def body_map(cloud_base_mm: np.ndarray, support_height_mm: float, *, floor_margin_mm: float,
             cell_mm: float = BODY_CELL_MM) -> BodyMap | None:
    """The part's body of ``cloud_base_mm``: its points over the support and the floor margin, its low fragments that
    stand apart from it left out (``support_footprint._trim_low_fragments``), on a grid of ``cell_mm``; one-cell holes a
    3 x 3 closing takes in filled as high as the cells about them. ``None`` where too few points stand over the support."""
    points = np.asarray(cloud_base_mm, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 2] > float(support_height_mm) + float(floor_margin_mm)]
    if points.shape[0] < 20:
        return None
    points, trimmed = _trim_low_fragments(points, float(support_height_mm))
    if points.shape[0] < 12:
        return None
    cell = float(cell_mm)
    ix = np.floor(points[:, 0] / cell).astype(np.int64)
    iy = np.floor(points[:, 1] / cell).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    top = np.full((int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1), -np.inf)
    np.maximum.at(top, (ix - x0, iy - y0), points[:, 2])
    top = _filled(top)
    return BodyMap(origin_mm=(x0 * cell, y0 * cell), cell_mm=cell, top=top, support_mm=float(support_height_mm),
                   top_mm=float(np.percentile(points[:, 2], 98.0)), trimmed=trimmed, points=points,
                   point_tops=top[ix - x0, iy - y0])


@dataclass(frozen=True, slots=True)
class _Found:
    """One grasp that holds on the part, before the rest of the scene is asked: its BASE anchor and closing axis."""

    anchor: np.ndarray
    axis: np.ndarray
    width_mm: float
    angle_rad: float
    score: float
    clearance_mm: float
    robust: bool
    #: The pad's contact height, BASE z millimetres, where the body has to stand into it.
    contact_z: float = 0.0
    #: How wide the jaws open before they close, millimetres; ``None`` for the whole stroke.
    open_mm: float | None = None
    #: Where the hand comes from, BASE: straight down, or leaning about the closing axis.
    approach: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))


def _filled(top: np.ndarray) -> np.ndarray:
    """``top`` with its one-cell holes, those a 3 x 3 closing takes in, filled as high as the cells about them."""
    from scipy import ndimage  # noqa: PLC0415 (only the stage pays for it)

    seen = np.isfinite(top)
    closed = ndimage.binary_closing(np.pad(seen, 1), structure=np.ones((3, 3), dtype=bool))[1:-1, 1:-1]
    new = closed & ~seen
    if not new.any():
        return top
    around = ndimage.maximum_filter(top, size=3, mode="constant", cval=-np.inf)
    return np.where(new, around, top)


def _rotated(body: BodyMap, yaw: float) -> "tuple[np.ndarray, float, float]":
    """The body laid on a grid square with the closing direction ``yaw``: ``top[iu, iv]`` along the closing axis and
    across it, each cell as high as the highest of the body's points in it, its one-cell holes filled as :func:`body_map`
    fills its own, a cell of nothing on every side, and the grid's low corner in the turned frame.

    Binned from the body's points, not from its map's cells: each cell turned as four samples stood in the turned grid
    up to a cell past the points on every side, so a 40 mm cube turned 45 degrees to the map spanned more than the
    stroke, and its straight faces turned into steps the friction cone refused. On the owner's picks of 2026-10-13 at
    06:20 and 07:10 the body found no grasp at all; binned from its points, 2 and 10 the guard admits (2026-10-14).
    """
    cell = body.cell_mm
    c, s = math.cos(yaw), math.sin(yaw)
    points = body.points
    u = c * points[:, 0] + s * points[:, 1]
    v = -s * points[:, 0] + c * points[:, 1]
    u0, v0 = float(u.min()) - cell, float(v.min()) - cell
    iu = np.floor((u - u0) / cell).astype(np.int64)
    iv = np.floor((v - v0) / cell).astype(np.int64)
    grid = np.full((int(iu.max()) + 2, int(iv.max()) + 2), -np.inf)
    np.maximum.at(grid, (iu, iv), points[:, 2])
    return _filled(grid), u0, v0


def _spans(stands: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Every span the pads close on, of ``stands`` (``(nu, nv, nz)``: the body stands into the pad at that cell along the
    closing axis, in the band about that row, at that height): each run along the closing axis, and each run with the
    next one or two of its line and what lies between them. ``(band, height, first, last)`` cell indices, inclusive."""
    padded = np.pad(stands, ((1, 1), (0, 0), (0, 0)))
    edge = np.diff(padded.astype(np.int8), axis=0)
    starts = np.argwhere(edge == 1)
    ends = np.argwhere(edge == -1)
    # Listed line by line (band, height) and along the axis in each: the n-th start and the n-th end are one run.
    starts = starts[np.lexsort((starts[:, 0], starts[:, 2], starts[:, 1]))]
    ends = ends[np.lexsort((ends[:, 0], ends[:, 2], ends[:, 1]))]
    band, level, first, last = starts[:, 1], starts[:, 2], starts[:, 0], ends[:, 0] - 1
    out = [(band, level, first, last)]
    for more in range(1, _MOST_RUNS):
        if band.size <= more:
            break
        same = (band[more:] == band[:-more]) & (level[more:] == level[:-more])
        out.append((band[:-more][same], level[:-more][same], first[:-more][same], last[more:][same]))
    return tuple(np.concatenate(parts) for parts in zip(*out))  # type: ignore[return-value]


def _next_at_or_after(solid: np.ndarray) -> np.ndarray:
    """Along axis 0 of ``solid`` (``(nu, ...)``), the first index at or after each cell where it is set; ``nu`` where
    none is."""
    nu = solid.shape[0]
    index = np.where(solid, np.arange(nu).reshape((nu,) + (1,) * (solid.ndim - 1)), nu)
    return np.minimum.accumulate(index[::-1], axis=0)[::-1]


def _last_at_or_before(solid: np.ndarray) -> np.ndarray:
    """Along axis 0 of ``solid``, the last index at or before each cell where it is set; -1 where none is."""
    nu = solid.shape[0]
    index = np.where(solid, np.arange(nu).reshape((nu,) + (1,) * (solid.ndim - 1)), -1)
    return np.maximum.accumulate(index, axis=0)


def _turn_at(rows: np.ndarray, contact: np.ndarray, *, forward: bool) -> "tuple[np.ndarray, np.ndarray]":
    """How far the body's outline turns from square to the closing axis where a pad meets it, radians, and where along
    the pad it meets it, in rows from the band's middle.

    ``rows`` holds, for each grasp and each row of its band, the first cell along the closing axis (``forward``) or the
    last (not ``forward``) where the body stands into the pad, -1 where a row has none; ``contact`` the band's own.
    Where the rows that meet the pad lie in both outer thirds of the band, the pad lies against the body across itself,
    a face square to the axis or a hollow it bridges (a ring's bore from inside), and pushes along its own normal: no
    turn, met at the middle of those rows. Else it meets one place, read from the middle of the rows that meet it
    :data:`_OUTLINE_CELLS` rows out either way, each side the least-squares line through the contact, the gentler side:
    the face the pad comes to lie on as the squeeze turns the part. A box a few degrees off the closing axis meets each
    pad at a corner, its face on one side and the next face on the other; read by the steeper side, the friction cone
    refused it there, and the owner's cubes of 2026-10-13 at 06:28 and 09:37 and of 2026-10-12 at 08:05 and 08:06 got no
    grasp of the body (15 the guard admits now, 1 to 7 a pick). A corner both of whose sides turn steeply (a cube across
    its diagonal) or a sliver is refused still, and a round part's extreme turns as little as its curvature over those
    rows. Read as the steepest of the single rows, a round part's outline crossing a cell's edge one row out turned 45
    degrees, and the friction cone refused the owner's lying cylinder of 2026-10-13 at 06:54 every grasp.
    """
    k, r = rows.shape
    seen = rows >= 0
    depth = np.where(seen, (rows - contact[:, None]) * (1.0 if forward else -1.0), np.nan)
    meets = seen & (np.abs(depth) < 0.5)
    count = meets.sum(axis=1)
    order = np.cumsum(meets, axis=1)
    # The middle row of those that meet the pad.
    middle = np.argmax(order >= np.maximum((count + 1) // 2, 1)[:, None], axis=1)
    third = max(1, r // 3)
    across = meets[:, :third].any(axis=1) & meets[:, r - third:].any(axis=1)
    gentlest = np.full(k, np.inf)
    for side in (-1, 1):
        moment = np.zeros(k)
        spread = np.zeros(k)
        for step in range(1, _OUTLINE_CELLS + 1):
            at = middle + side * step
            inside = np.nonzero((at >= 0) & (at < r))[0]
            out = depth[inside, at[inside]]
            read = np.isfinite(out)
            moment[inside[read]] += step * out[read]
            spread[inside[read]] += step * step
        # A side with no outline read (the band ends there, or the body does) says nothing of the face.
        slope = np.arctan(np.maximum(moment, 0.0) / np.maximum(spread, 1.0))
        gentlest = np.where(spread > 0.0, np.minimum(gentlest, slope), gentlest)
    turn = np.where(count > 0, np.where(across, 0.0, np.where(np.isfinite(gentlest), gentlest, 0.0)), math.pi / 2.0)
    first_meet = np.argmax(meets, axis=1)
    last_meet = r - 1 - np.argmax(meets[:, ::-1], axis=1)
    where = np.where(across, (first_meet + last_meet) / 2.0, middle.astype(np.float64)) - (r - 1) / 2.0
    return turn, where


def _across_extent(grid: np.ndarray, levels: np.ndarray, pad_low: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """For each grasp's height, the first and the last row across the closing axis where the body stands into the pad,
    in cells: the part's extent across at that height."""
    stands = (grid[:, :, None] >= (pad_low + BODY_MIN_OVERLAP_MM)[None, None, :]).any(axis=0)       # (nv, nz)
    rows = np.arange(stands.shape[0])[:, None]
    first = np.where(stands, rows, stands.shape[0]).min(axis=0)
    last = np.where(stands, rows, -1).max(axis=0)
    return first[levels].astype(np.float64), last[levels].astype(np.float64)


def body_grasps(
    cloud_base_mm: np.ndarray,
    *,
    support_height_mm: float,
    jaw: SupportFootprintJaw,
    obstacle_points_base_mm: np.ndarray | None = None,
    rigid_obstacle_points_base_mm: np.ndarray | None = None,
    seen_envelope: Any = None,
    hand_floor: HandFloor | None = None,
    corridor_seen: Any = None,
    floor_margin_mm: float = 2.0,
    open_width_mm: float | None = None,
    opens_as_needed: bool = False,
    max_candidates: int = 12,
    refusals: "dict[str, int] | None" = None,
    score_weights: "tuple[float, float, float, float, float] | None" = None,
) -> list[SupportFootprintCandidate]:
    """The grasps straight down on the part's measured body (module docstring), robust first, each by score,
    ``max_candidates`` at most.

    ``open_width_mm`` is how wide the hand opens before it closes, millimetres: the whole stroke where ``None``, a hand
    that opens fully (the toggle). ``opens_as_needed`` opens it, grasp by grasp, as wide as the span and
    :data:`support_footprint.NARROW_ROOM_MM` either side (a Robotiq hand that is told a width, lever G), and offers the
    same place fully open as well where the open fingers are clear there too: the narrow fingers stand where the D415
    smears a part's far edge, which the guard holds as the part's own fringe (2026-10-13: three of the owner's dense
    Tuesday picks lost every narrow grasp there and kept the fully open ones). Each grasp carries its opening
    (``SupportFootprintCandidate.open_width_mm``), the narrow one first. ``refusals``, where given, counts why the spans the pads close on were not offered,
    under SFE's causes (``support_footprint.REFUSAL_CAUSES``): ``aperture``, ``prism`` (the open fingers on the part
    itself) and ``cone`` for the part; ``table``, ``seen_corridor`` and ``seen_fingers`` for the rest of the scene, and
    ``unseen_corridor`` for an open finger over the part's outline where no ray of the frame reached.

    Args:
        cloud_base_mm: The part's points, ``(N, 3)`` BASE millimetres.
        support_height_mm: What the part stands on, BASE z millimetres.
        jaw: The hand, as SFE plans it.
        obstacle_points_base_mm: The neighbours' points the camera saw, as SFE keeps the hand off them.
        rigid_obstacle_points_base_mm: The declared bodies' points, kept off undilated.
        seen_envelope: The guard's boxes of the neighbours (``scene_obstacles.SeenEnvelope``), or ``None``.
        hand_floor: The solids the guard holds under the hand (``support_footprint.HandFloor``), or ``None``.
        corridor_seen: Whether a ray of the frame reached a point (``scene_obstacles.SeenInTheFrame``), or ``None``,
            where an open finger over the part's outline is not asked.
        floor_margin_mm: How far over the support a point still counts as the support.
        open_width_mm: How wide the hand opens before it closes; ``None`` for the whole stroke.
        opens_as_needed: Whether the hand opens only as wide as each span and the room beside each jaw.
        max_candidates: The most grasps returned.
        refusals: Counted into where given, as above.
        score_weights: SFE's five weights, ``None`` for its own (``support_footprint._SCORE_WEIGHTS``).

    Returns:
        list[SupportFootprintCandidate]: The grasps, BASE millimetres; empty where the part has none.
    """
    from scipy import ndimage  # noqa: PLC0415

    body = body_map(cloud_base_mm, support_height_mm, floor_margin_mm=floor_margin_mm)
    if body is None:
        return []
    counted: dict[str, int] = {} if refusals is None else refusals

    def count(cause: str, n: int) -> None:
        if n:
            counted[cause] = counted.get(cause, 0) + int(n)

    cell = body.cell_mm
    support = body.support_mm
    opening = float(jaw.aperture_mm if open_width_mm is None else min(float(open_width_mm), jaw.aperture_mm))
    widest = jaw.aperture_mm - jaw.width_safety_mm
    band_cells = max(1, int(round(jaw.finger_width_mm / cell)))
    band_cells += 1 - band_cells % 2
    half_band = band_cells // 2
    cone = jaw.cone_rad
    # The anchor heights: the fingertips the support's clearance over it at the lowest; at the highest the pads
    # HOLD_OVERLAP_MM into the part's top, or as deep as that lowest lets them on a lower part, BODY_MIN_OVERLAP_MM at the
    # least.
    lowest = support + jaw.table_clearance_mm + jaw.finger_ahead_mm
    top = float(body.top[np.isfinite(body.top)].max())
    hold = max(BODY_MIN_OVERLAP_MM, min(HOLD_OVERLAP_MM, top - (lowest - jaw.pad_ahead_mm)))
    highest = top - hold + jaw.pad_ahead_mm
    if highest < lowest or body.top_mm - support < body_least_height_mm(jaw):
        # Lower than the pads hold it with the fingertips the clearance over the support: the support refuses every
        # grasp, and the result says the part is too short for this hand (``scene_obstacles.why_no_grasp``).
        count("table", 1)
        return []
    heights = np.arange(highest, lowest - 1e-9, -BODY_HEIGHT_STEP_MM)
    if heights.size > _MOST_HEIGHTS:
        heights = heights[np.linspace(0, heights.size - 1, _MOST_HEIGHTS).round().astype(int)]
    pad_low = heights - jaw.pad_ahead_mm
    tips = heights - jaw.finger_ahead_mm
    weights = score_weights or _SCORE_WEIGHTS
    half_part = (body.top_mm - support) / 2.0
    shift = int(round(_ROBUST_MM / cell))
    finger_cells = jaw.finger_thickness_mm / cell
    # The part's centre of mass in the plane, each column weighed by its height over the support: a grasp whose closing
    # line runs through it lifts the part without turning it about that line.
    cells_xy, cells_top = body.centres()
    weighed = np.maximum(cells_top - support, 0.0)
    com_xy = (cells_xy * weighed[:, None]).sum(axis=0) / max(float(weighed.sum()), _EPS)

    found: list[_Found] = []
    for yaw in np.radians(np.arange(0.0, 180.0, BODY_YAW_STEP_DEG)):
        grid, u0, v0 = _rotated(body, float(yaw))
        nu, nv = grid.shape
        # The band each pad covers: the highest of the body over the finger's width, centred on each row.
        band = ndimage.maximum_filter1d(grid, size=band_cells, axis=1, mode="constant", cval=-np.inf)
        stands = band[:, :, None] >= (pad_low + BODY_MIN_OVERLAP_MM)[None, None, :]
        rows_stand = grid[:, :, None] >= (pad_low + BODY_MIN_OVERLAP_MM)[None, None, :]
        over_tips = band[:, :, None] >= tips[None, None, :]
        bands, levels, first, last = _spans(stands)
        if not bands.size:
            continue
        span = (last - first + 1) * cell
        fits = (span <= widest) & (span >= jaw.min_width_mm)
        count("aperture", int(np.count_nonzero(~fits)))
        bands, levels, first, last, span = (a[fits] for a in (bands, levels, first, last, span))
        if not bands.size:
            continue
        # The open fingers: their inner faces half the opening off the span's middle; no part of the body over their tips
        # where they come down, nor between them and the span while they close. A hand that opens as needed is asked
        # narrow and fully open, each place kept where either is clear.
        middle_u = (first + last + 1) / 2.0
        after = _next_at_or_after(over_tips)
        before = _last_at_or_before(over_tips)

        def clear_at(open_mm: np.ndarray) -> np.ndarray:
            outer_left = np.floor(middle_u - open_mm / (2.0 * cell) - finger_cells).astype(np.int64)
            outer_right = np.ceil(middle_u + open_mm / (2.0 * cell) + finger_cells).astype(np.int64) - 1
            left_from = np.clip(outer_left, 0, nu - 1)
            right_from = np.clip(outer_right, 0, nu - 1)
            return (after[left_from, bands, levels] >= first) & (before[right_from, bands, levels] <= last)

        narrow = np.minimum(span + 2.0 * NARROW_ROOM_MM, jaw.aperture_mm)
        full_clear = clear_at(np.full(span.shape, opening))
        # A span the narrow room leaves no stroke past is the fully open grasp alone.
        narrow_clear = (clear_at(narrow) & (narrow < opening) if opens_as_needed
                        else np.zeros(span.shape, dtype=bool))
        clear = full_clear | narrow_clear
        count("prism", int(np.count_nonzero(~clear)))
        bands, levels, first, last, span, middle_u, full_clear, narrow_clear = (
            a[clear] for a in (bands, levels, first, last, span, middle_u, full_clear, narrow_clear))
        if not bands.size:
            continue
        # How the body leans where a pad meets it: on a grid of columns it is widest at the pad's foot, so the pad meets
        # it there, and how far its outline steps in over the next :data:`_LEAN_RISE_MM` up the pad is how far the
        # surface leans off the vertical: a wall not at all, a lying cylinder over its equator by its slope there.
        higher = band[:, :, None] >= (pad_low + BODY_MIN_OVERLAP_MM + _LEAN_RISE_MM)[None, None, :]
        up_after, up_before = _next_at_or_after(higher), _last_at_or_before(higher)
        step_l = (up_after[first, bands, levels] - first).astype(np.float64)
        step_r = (last - up_before[last, bands, levels]).astype(np.float64)
        # Where the body never stands that high within the span the pad holds a sliver: leaning as far as the span.
        step_l = np.where(up_after[first, bands, levels] > last, last - first + 1.0, step_l)
        step_r = np.where(up_before[last, bands, levels] < first, last - first + 1.0, step_r)
        lean_l = np.arctan2(step_l * cell, _LEAN_RISE_MM)
        lean_r = np.arctan2(step_r * cell, _LEAN_RISE_MM)
        # How the outline turns across the pad at each contact: the rows of the band, their first and last cells.
        rows = np.clip(bands[:, None] + np.arange(-half_band, half_band + 1)[None, :], 0, nv - 1)
        meets = np.transpose(rows_stand[:, rows, levels[:, None]], (1, 0, 2))       # (K, nu, R)
        index = np.arange(nu)[None, :, None]
        firsts = np.where(meets & (index >= first[:, None, None]), index, nu).min(axis=1)
        lasts = np.where(meets & (index <= last[:, None, None]), index, -1).max(axis=1)
        firsts = np.where(firsts < nu, firsts, -1)
        turn_l, at_l = _turn_at(firsts, first, forward=True)
        turn_r, at_r = _turn_at(lasts, last, forward=False)
        # And the line between the two places the pads meet the body: off the closing axis, the squeeze turns the part.
        line = np.arctan2(np.abs(at_l - at_r) * cell, np.maximum(span, _EPS))
        angle = np.maximum.reduce([np.arccos(np.clip(np.cos(lean_l) * np.cos(turn_l), -1.0, 1.0)),
                                   np.arccos(np.clip(np.cos(lean_r) * np.cos(turn_r), -1.0, 1.0)), line])
        holds = angle <= cone
        count("cone", int(np.count_nonzero(~holds)))
        if not holds.any():
            continue
        bands, levels, first, last, span, middle_u, angle, full_clear, narrow_clear = (
            a[holds] for a in (bands, levels, first, last, span, middle_u, angle, full_clear, narrow_clear))
        # Robust: the open fingers stand off the part by the error, and the band the error off either way across the
        # table still meets the span and runs no wider than a cell past it.
        cumulative = np.concatenate([np.zeros((1, nv, heights.size), dtype=np.int64),
                                     np.cumsum(stands, axis=0)], axis=0)
        # The band the error off either way across the table; fully open, the stroke left beside the part as well. A hand
        # that opens only as needed stands its fingers the room beside the part, not the error, off it: it is robust
        # across the closing axis by the band alone.
        band_holds = np.ones(span.shape, dtype=bool)
        for side in (-shift, shift):
            other = bands + side
            inside = (other >= 0) & (other < nv)
            other = np.clip(other, 0, nv - 1)
            meets_there = (cumulative[last + 1, other, levels] - cumulative[first, other, levels]) > 0
            low_out = np.clip(first - 2, 0, nu - 1)
            high_out = np.clip(last + 2, 0, nu - 1)
            wider = stands[low_out, other, levels] | stands[high_out, other, levels]
            band_holds &= inside & meets_there & ~wider
        robust = band_holds & ((opening - span) / 2.0 >= _ROBUST_MM)
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        axis = np.array([c, s, 0.0])
        u_mm = u0 + middle_u * cell
        v_mm = v0 + (bands + 0.5) * cell
        xs, ys = c * u_mm - s * v_mm, s * u_mm + c * v_mm
        z = heights[levels]
        # As SFE scores (``support_footprint._build``): the cone's slack, the stroke left, how the pads sit on the part,
        # upright, centred (an anchor between its contacts is).
        covered = np.clip(np.minimum(z + jaw.pad_behind_mm, body.top_mm) - np.maximum(z - jaw.pad_ahead_mm, support),
                          0.0, None) / max(jaw.pad_ahead_mm + jaw.pad_behind_mm, _EPS)
        middle = (1.0 - np.minimum(1.0, np.abs(z - (support + half_part)) / half_part)
                  if half_part > _EPS else np.zeros_like(z))
        # Centred: between the contacts it is by construction; across, how near the closing line runs to the part's
        # centre of mass, against half the part's extent across at that height.
        across_lo, across_hi = _across_extent(grid, levels, pad_low)
        reach = np.maximum((across_hi - across_lo) / 2.0, 1.0) * cell
        com_v = -math.sin(float(yaw)) * float(com_xy[0]) + math.cos(float(yaw)) * float(com_xy[1])
        centred = 1.0 - np.minimum(1.0, np.abs(v0 + (bands + 0.5) * cell - com_v) / reach)
        # The rest of the third term is the room the open fingers keep, which the scene's checks measure (``_kept``).
        score = (weights[0] * (1.0 - angle / cone) + weights[1] * (1.0 - span / jaw.aperture_mm)
                 + weights[2] * _HOLD_SHARE * (0.5 * covered + 0.5 * middle) + weights[3] + weights[4] * centred)
        for n in range(bands.size):
            fully = _Found(anchor=np.array([xs[n], ys[n], z[n]]), axis=axis, width_mm=float(span[n]),
                           angle_rad=float(angle[n]), score=float(score[n]),
                           clearance_mm=float(z[n] - jaw.finger_ahead_mm - support), robust=bool(robust[n]),
                           contact_z=float(pad_low[levels[n]] + BODY_MIN_OVERLAP_MM))
            # The narrow opening first, robust by the band alone; fully open, by the stroke left as well, and on a hand
            # that opens as needed scored down by the stroke it opens past the narrow opening, as the stroke left scores:
            # the narrow grasps of other places are tried before it (the cell tries GRASPS_TRIED_PER_ATTEMPT).
            if narrow_clear[n]:
                found.append(replace(fully, robust=bool(band_holds[n]), open_mm=float(narrow[n])))
            if full_clear[n]:
                found.append(replace(fully, score=float(score[n]) - weights[1] * (opening - float(narrow[n]))
                                     / jaw.aperture_mm) if opens_as_needed else fully)
    if not found:
        return []
    # The rest of the scene, best first, a batch at a time, until enough pass.
    found.sort(key=lambda f: (not f.robust, -f.score))
    obstacles = _ObstacleSet(
        _Obstacles(obstacle_points_base_mm) if seen_envelope is None
        else _Obstacles(obstacle_points_base_mm, cell_mm=SEEN_POINTS_CELL_MM, margin_mm=SEEN_POINTS_CELL_MM),
        _Obstacles(_joined(rigid_obstacle_points_base_mm, body.trimmed), cell_mm=4.0, margin_mm=0.0),
    )
    outline = _outline(body) if corridor_seen is not None else None
    room = _Room(obstacle_points_base_mm, rigid_obstacle_points_base_mm, body.trimmed)
    checks = dict(jaw=jaw, opening=opening, support=support, obstacles=obstacles, seen_envelope=seen_envelope,
                  hand_floor=hand_floor, seen=corridor_seen, outline=outline, body=body, room=room,
                  room_weight=weights[2] * (1.0 - _HOLD_SHARE), count=count)
    passed: list[_Found] = []
    mendable: list[_Found] = []
    places: dict[bool, set[tuple[int, ...]]] = {narrow: set() for narrow in ((True, False) if opens_as_needed
                                                                            else (False,))}
    for start in range(0, len(found), _BATCH):
        batch = _refined(found[start:start + _BATCH], body, jaw, widest=widest, count=count, centre_xy=com_xy)
        verdicts = _kept(batch, **checks)
        passed.extend(v for v in verdicts if not isinstance(v, str))
        for v in verdicts:
            if not isinstance(v, str):
                places[v.open_mm is not None].add(_place(v))
        mendable.extend(f for f, v in zip(batch, verdicts) if isinstance(v, str) and v in _A_LEAN_MENDS)
        if all(len(each) >= _ENOUGH for each in places.values()):
            break
    # The grasps the scene refused straight down, leaning, only where a ray of the frame can vouch for the way in, as SFE
    # leans (``_SEEN_FROM_TILT_DEG``).
    if corridor_seen is not None and mendable:
        passed.extend(_least_tilts(mendable, weights[3], checks))
    passed.sort(key=lambda f: (not f.robust, -f.score))
    # ``max_candidates`` of each opening: the narrow ones, and on a hand that opens as needed the fully open ones too.
    out: list[SupportFootprintCandidate] = []
    taken = {True: 0, False: 0}
    for f in passed:
        narrow = f.open_mm is not None
        if taken[narrow] >= int(max_candidates):
            continue
        candidate = SupportFootprintCandidate(
            score=f.score, position_mm=f.anchor, approach=f.approach.copy(), closing_axis=f.axis,
            grip_width_mm=f.width_mm + 2.0, contact_angle_rad=f.angle_rad, clearance_mm=f.clearance_mm,
            robust=f.robust, open_width_mm=None if f.open_mm is None else min(
                jaw.aperture_mm, f.width_mm + 2.0 * NARROW_ROOM_MM))
        if not _near(candidate, out):
            out.append(candidate)
            taken[narrow] += 1
    return out


def _refined(batch: "list[_Found]", body: BodyMap, jaw: SupportFootprintJaw, *, widest: float,
             count: Callable[[str, int], None], centre_xy: "np.ndarray | None" = None) -> "list[_Found]":
    """Each grasp of ``batch`` with its contacts where the body's own points put them, not where its cells do: of the
    points in the band of the pad, whose cells stand into the pad, within two cells of the span the cells found, the
    first and the last along the closing axis; the anchor moved to their middle, the width theirs. The cells of the
    grid stand up to a cell wider than the points in them; a span the points make wider than the stroke allows is
    refused for the aperture. A closing line within a cell of the part's centre of mass (``centre_xy``) is moved onto
    it first: the grid's rows stand a cell apart, and on a cylinder between two cubes 16 mm off its sides the line
    0.9 mm off its centre came nearer the cubes than the guard keeps, where the line through it kept the distance."""
    out: list[_Found] = []
    half_band = jaw.finger_width_mm / 2.0
    points, tops = body.points, body.point_tops
    margin = 2.0 * body.cell_mm
    for f in batch:
        axis = f.axis
        across = np.array([-axis[1], axis[0], 0.0])
        if centre_xy is not None:
            off = float((np.asarray(centre_xy, dtype=np.float64) - f.anchor[:2]) @ across[:2])
            if abs(off) <= body.cell_mm:
                f = replace(f, anchor=f.anchor + off * across)
        rel = points - f.anchor
        u, v = rel @ axis, rel @ across
        near = (np.abs(v) <= half_band) & (tops >= f.contact_z) & (np.abs(u) <= f.width_mm / 2.0 + margin)
        if not near.any():
            out.append(f)
            continue
        low, high = float(u[near].min()), float(u[near].max())
        width = high - low
        if width > widest:
            count("aperture", 1)
            continue
        out.append(replace(f, anchor=f.anchor + ((low + high) / 2.0) * axis, width_mm=width))
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` x ``b`` for two 3-vectors, without numpy's broadcasting machinery: a pick asks it a hundred thousand times."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _leaning(f: _Found, degrees: float, upright_weight: float) -> _Found:
    """``f`` with the hand leaning ``degrees`` about its closing axis (the sign the side), scored upright by the
    cosine of the tilt as SFE scores a side approach (``support_footprint._build``)."""
    angle = math.radians(degrees)
    down = np.array([0.0, 0.0, -1.0])
    k = f.axis / max(float(np.linalg.norm(f.axis)), _EPS)
    approach = down * math.cos(angle) + _cross(k, down) * math.sin(angle) + k * float(k @ down) * (1.0 - math.cos(angle))
    return replace(f, approach=approach / max(float(np.linalg.norm(approach)), _EPS),
                   score=f.score - upright_weight * (1.0 - math.cos(angle)))


def _least_tilts(refused: "list[_Found]", upright_weight: float, checks: "dict[str, Any]") -> "list[_Found]":
    """Of the grasps the scene refused straight down, the least tilt either way that keeps the scene, or none: at each
    tilt, of those still without one whose closed fingers' lowest corner stays the clearance over the support leaning
    so far, :data:`_TILTED_MOST` asked at once (:func:`_each_place_first`)."""
    jaw, floor = checks["jaw"], float(checks["support"]) + float(checks["jaw"].table_clearance_mm)
    out: list[_Found] = []
    left = refused
    for degrees in BODY_TILTS_DEG:
        angle = math.radians(degrees)
        drop = jaw.finger_ahead_mm * math.cos(angle) + jaw.finger_width_mm / 2.0 * math.sin(angle)
        asked = _each_place_first([f for f in left if float(f.anchor[2]) - drop >= floor], _TILTED_MOST)
        if not asked:
            continue
        verdicts = _kept([_leaning(f, sign * degrees, upright_weight) for f in asked for sign in (1.0, -1.0)], **checks)
        done = set()
        for n, f in enumerate(asked):
            kept = [v for v in verdicts[2 * n:2 * n + 2] if not isinstance(v, str)]
            if kept:
                out.append(max(kept, key=lambda k: k.score))
                done.add(id(f))
        left = [f for f in left if id(f) not in done]
    return out


def _yaw_step(f: _Found) -> int:
    """Which of the yaw steps (:data:`BODY_YAW_STEP_DEG`) ``f`` closes along."""
    return int(round(math.degrees(math.atan2(float(f.axis[1]), float(f.axis[0]))) % 180.0 / BODY_YAW_STEP_DEG))


def _place(f: _Found) -> tuple[int, ...]:
    """Where ``f`` grasps, to the 4 mm cube (:func:`_near`'s distance), its yaw step and whether it opens narrow."""
    return (*(int(v) for v in np.floor(f.anchor / 4.0)), _yaw_step(f), int(f.open_mm is None))


def _each_place_first(grasps: "list[_Found]", most: int) -> "list[_Found]":
    """At most ``most`` of ``grasps`` (best first), one of each closing axis and band across it (:data:`_LEAN_BAND_MM`)
    before the second of any, the places in the order of their best grasp: of each place its best grasp, then its highest
    anchor, then the rest best first. Leaning, the fingers come down beside the part, and the higher they stop the more of
    their way in a ray saw: at the place's best anchor alone, a pad's height in the part's middle, every lean on the
    owner's trays of 2026-10-13 came down through space no look saw past, where SFE's lean at the part's top did not;
    under a beam the best anchor is the one that fits."""
    places: dict[tuple[int, int], list[_Found]] = {}
    for f in grasps:
        yaw = math.atan2(float(f.axis[1]), float(f.axis[0]))
        across = -math.sin(yaw) * float(f.anchor[0]) + math.cos(yaw) * float(f.anchor[1])
        places.setdefault((_yaw_step(f), int(math.floor(across / _LEAN_BAND_MM))), []).append(f)
    ranked = []
    for found in places.values():
        highest = max(range(len(found)), key=lambda n: float(found[n].anchor[2]))
        ranked.append([found[0], *([found[highest]] if highest else []),
                       *(f for n, f in enumerate(found) if n not in (0, highest))])
    out: list[_Found] = []
    for rank in range(max((len(found) for found in ranked), default=0)):
        out.extend(found[rank] for found in ranked if rank < len(found))
        if len(out) >= most:
            break
    return out[:most]


def _inside_body(body: BodyMap, points: np.ndarray) -> np.ndarray:
    """Which of ``(N, 3)`` BASE points stand inside the part's body: in a cell of its map, at or under the cell's top
    (each cell a column down to the support)."""
    cell = body.cell_mm
    ix = np.floor((points[:, 0] - body.origin_mm[0]) / cell).astype(np.int64)
    iy = np.floor((points[:, 1] - body.origin_mm[1]) / cell).astype(np.int64)
    on = (ix >= 0) & (ix < body.top.shape[0]) & (iy >= 0) & (iy < body.top.shape[1])
    out = np.zeros(points.shape[0], dtype=bool)
    out[on] = body.top[ix[on], iy[on]] >= points[on, 2]
    return out


def _outline(body: BodyMap) -> "Callable[[np.ndarray], np.ndarray] | None":
    """Whether BASE points stand over the part's outline: its convex hull in the plane, a cell inside it, where a bore, a
    gap or the space under a lobe lies; ``None`` where the body has no hull."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    xy, _ = body.centres()
    if xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(xy)
    except Exception:  # noqa: BLE001 (QhullError: a body in a line has no outline to stand over)
        return None
    normals, offsets, margin = hull.equations[:, :2], hull.equations[:, 2], float(body.cell_mm)
    return lambda points: np.all(points[:, :2] @ normals.T + offsets <= -margin, axis=1)


def _kept(batch: "list[_Found]", *, jaw: SupportFootprintJaw, opening: float, support: float,
          obstacles: _ObstacleSet, seen_envelope: Any, hand_floor: HandFloor | None, seen: Any = None,
          outline: "Callable[[np.ndarray], np.ndarray] | None" = None, body: "BodyMap | None" = None,
          room: "_Room | None" = None, room_weight: float = 0.0,
          count: Callable[[str, int], None]) -> "list[_Found | str]":
    """The grasps of ``batch`` whose hand keeps what SFE keeps (``support_footprint._build``): the closed fingers the
    support's clearance and the held solid's distance, the palm the solid's, the open fingers on their way in the
    solid's; the neighbours' points and the declared bodies off the open and the closed fingers; the guard's boxes of the
    neighbours off the open hand (``SeenEnvelope``). A verdict SFE would ask again one build at a time (2) is a refusal
    here: the stage offers only what it is sure of. Where ``seen`` and ``outline`` are given, an open finger's way down
    over the part's outline is space a ray of the frame reached, or the grasp is refused (``unseen_corridor``). A hand
    leaning off vertical keeps its fingers out of the part's own ``body`` and has its whole way in seen. Each grasp kept
    scores ``room_weight`` more as the room its open fingers keep from ``room``'s points nears :data:`_CLEARANCE_FULL_MM`
    (SFE's side approaches, ``support_footprint._build``). One answer a grasp: it kept, or the cause it was refused for,
    counted. The guard's boxes, the dearest of the checks, are asked only of the grasps every other check passed: the
    first cause a grasp is refused for is the same, and in a crate whose every grasp a neighbour's points refuse they
    were most of a part's search (2026-10-14)."""
    reach = jaw.finger_reach_mm + 80.0
    verdicts: "list[_Found | str]" = []
    asked: "list[tuple[int, _Found, np.ndarray, float]]" = []
    built: "list[tuple[_Found, np.ndarray, np.ndarray, float, np.ndarray]]" = []
    for f in batch:
        axis, approach = f.axis, f.approach
        binormal = _cross(approach, axis)
        binormal /= max(float(np.linalg.norm(binormal)), _EPS)
        reach_out = f.width_mm / 2.0 + jaw.finger_thickness_mm / 2.0
        closed = np.concatenate([
            _finger_points(f.anchor + reach_out * axis, approach, binormal, jaw),
            _finger_points(f.anchor - reach_out * axis, approach, binormal, jaw)])
        wide = opening if f.open_mm is None else min(f.width_mm + 2.0 * NARROW_ROOM_MM, jaw.aperture_mm)
        swept = np.concatenate([
            _finger_points(f.anchor + (wide / 2.0) * axis, approach, binormal, jaw, reach, 8.0),
            _finger_points(f.anchor - (wide / 2.0) * axis, approach, binormal, jaw, reach, 8.0)])
        built.append((f, closed, swept[swept[:, 2] >= support], wide, binormal))
    # The solids under the hand judged for the whole batch at once, three questions a grasp (the closed fingers, the
    # palm, the open fingers' way in): the same answers as one grasp at a time, most of a part's search before.
    floored = np.zeros((len(built), 3), dtype=bool)
    if hand_floor is not None and built:
        floored = np.column_stack([
            _under_each(hand_floor, [closed for _, closed, _, _, _ in built], fingers=True),
            _under_each(hand_floor, [_palm_corners(f.anchor, f.axis, f.approach, binormal, jaw)
                                     for f, _, _, _, binormal in built]),
            _under_each(hand_floor, [swept for _, _, swept, _, _ in built], fingers=True)])
    for n, ((f, closed, swept, wide, _), under) in enumerate(zip(built, floored)):
        why = _refusal(f, closed, swept, jaw=jaw, support=support, obstacles=obstacles,
                       floored=(bool(under[0]), bool(under[1]), bool(under[2])), seen=seen, outline=outline, body=body)
        verdicts.append(why)
        if not why:
            asked.append((n, f, swept, wide))
    if seen_envelope is not None and asked:
        positions = np.stack([f.anchor for _, f, _, _ in asked])
        rotations = np.stack([np.column_stack([f.axis, _cross(f.approach, f.axis), f.approach])
                              for _, f, _, _ in asked])
        # A grasp that opens only as needed comes down with its fingers where the guard holds them: as wide as it opens
        # and the slack the guard is told (lever G); every other at the opening the envelope stands open at.
        openings = (None if seen_envelope.open_width_mm is None else np.array(
            [seen_envelope.open_width_mm if f.open_mm is None else min(wide + FINGER_OPENING_SLACK_MM, jaw.aperture_mm)
             for _, f, _, wide in asked]))
        envelope = np.asarray(seen_envelope.refuses_many(positions, rotations, openings), dtype=np.int8)
    else:
        envelope = np.zeros(len(asked), dtype=np.int8)
    for (n, f, swept, _), refused in zip(asked, envelope):
        if int(refused) != 0:
            verdicts[n] = "seen_fingers"
            continue
        free = room.least_mm(swept) if room is not None else math.inf
        verdicts[n] = replace(f, score=f.score + room_weight * min(1.0, free / _CLEARANCE_FULL_MM))
    for verdict in verdicts:
        if isinstance(verdict, str):
            count(verdict, 1)
    return verdicts


def _refusal(f: _Found, closed: np.ndarray, swept: np.ndarray, *, jaw: SupportFootprintJaw, support: float,
             obstacles: _ObstacleSet, floored: "tuple[bool, bool, bool]", seen: Any,
             outline: "Callable[[np.ndarray], np.ndarray] | None", body: "BodyMap | None") -> str:
    """What the scene refuses grasp ``f`` for (:func:`_kept`, SFE's causes), its closed fingers' points ``closed`` and its
    open fingers' way in ``swept``, the guard's boxes aside (:func:`_kept` asks them last); ``floored`` says whether the
    closed fingers, the palm and the open fingers stand under the solids the guard holds under the hand (``HandFloor``,
    asked of the batch at once). ``""`` where it refuses nothing."""
    closed_under, palm_under, swept_under = floored
    if float(closed[:, 2].min()) < support + jaw.table_clearance_mm:
        return "table"
    if closed_under or palm_under:
        return "table"
    if float(f.approach[2]) > -1.0 + 1e-9:
        if seen is None or not bool(np.all(seen(swept))):
            return "unseen_corridor"
        if body is not None and (_inside_body(body, swept).any() or _inside_body(body, closed).any()):
            return "prism"
    elif seen is not None and outline is not None:
        over = swept[outline(swept)]
        if over.shape[0] and not bool(np.all(seen(over))):
            return "unseen_corridor"
    if swept_under:
        return "table"
    if obstacles.hits(swept):
        return "seen_corridor"
    if obstacles.hits(closed):
        return "seen_fingers"
    return ""


def _under_each(floor: HandFloor, clouds: "list[np.ndarray]", *, fingers: bool = False) -> np.ndarray:
    """``floor.under`` of each of ``clouds``, asked of them all at once (``HandFloor.under_many``) and of a cloud that pass
    is unsure of alone, by :meth:`HandFloor.under` itself: the same answers."""
    most = max((cloud.shape[0] for cloud in clouds), default=0)
    points = np.zeros((len(clouds), most, 3))
    asked = np.zeros((len(clouds), most), dtype=bool)
    for n, cloud in enumerate(clouds):
        points[n, :cloud.shape[0]] = cloud
        asked[n, :cloud.shape[0]] = True
    verdict = floor.under_many(points, asked, fingers=fingers)
    under = verdict == 1
    for unsure in np.nonzero(verdict == 2)[0]:
        under[unsure] = floor.under(clouds[int(unsure)], fingers=fingers)
    return under


def _joined(*clouds: "np.ndarray | None") -> "np.ndarray | None":
    """The clouds given as one ``(N, 3)`` array, or ``None`` where none holds a point."""
    parts = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in clouds if c is not None and np.size(c)]
    return np.vstack(parts) if parts else None


def _near(candidate: SupportFootprintCandidate, kept: "list[SupportFootprintCandidate]") -> bool:
    """Whether ``candidate`` is a grasp near one of ``kept``: within 4 mm, its closing axis the same within 5 degrees,
    the jaws opened alike (one narrow and one fully open are two grasps to the guard)."""
    return any(float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
               and abs(float(candidate.closing_axis @ k.closing_axis)) > math.cos(math.radians(5.0))
               and (candidate.open_width_mm is None) == (k.open_width_mm is None)
               for k in kept)


def merged_ranking(first: "list[SupportFootprintCandidate]", second: "list[SupportFootprintCandidate]", *,
                   most: int) -> list[SupportFootprintCandidate]:
    """Two stages' grasps ranked together as each ranks its own (``support_footprint.rank_found``): the robust ones
    first, each by score. Every grasp of ``first`` stays as its stage offered it, so ``second`` only adds: on the owner's
    Tuesday picks a body's grasp ranked in an SFE grasp's place had taken the one grasp the guard admitted from a pick.
    A grasp of ``second`` within 4 mm of one of ``first``, its axes alike and its jaws opened alike, is left out: one
    that opens narrow (lever G) is another grasp to the guard.

    Args:
        first: SFE's grasps, all of which stay.
        second: The body's.
        most: How many of ``second`` at most.

    Returns:
        list[SupportFootprintCandidate]: The grasps of both, ranked.
    """
    added = [candidate for candidate in second
             if not any(float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
                        and abs(float(candidate.closing_axis @ k.closing_axis)) > 0.995
                        and float(candidate.approach @ k.approach) > 0.995
                        and (candidate.open_width_mm is None) == (k.open_width_mm is None) for k in first)][:int(most)]
    return sorted([*first, *added], key=lambda c: (not c.robust, -float(c.score)))


def body_least_height_mm(jaw: SupportFootprintJaw) -> float:
    """How far over its support a part has to stand for the body's pads to hold on it, millimetres: the fingertips the
    table clearance over the support, the pad :data:`BODY_MIN_OVERLAP_MM` into the part from its lowest point, and the
    part's outline read :data:`_LEAN_RISE_MM` further up for how it leans: 8 mm on the Hand-E, whose pads reach its
    fingertips.

    Args:
        jaw: The hand.

    Returns:
        float: The least part height, millimetres.
    """
    return (float(jaw.table_clearance_mm) + max(0.0, float(jaw.finger_ahead_mm) - float(jaw.pad_ahead_mm))
            + BODY_MIN_OVERLAP_MM + _LEAN_RISE_MM)


def body_telemetry(found: "list[SupportFootprintCandidate]", refused: Mapping[str, int]) -> dict[str, Any]:
    """What a result says of this stage: how many grasps it offered, and why it offered no more."""
    return {"body_grasps": len(found), "body_refused": dict(refused)}
