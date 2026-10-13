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
the part (every cell) and every height the pads can stand at (every :data:`BODY_HEIGHT_STEP_MM`):

* the pads meet the body where it stands at least :data:`BODY_MIN_OVERLAP_MM` into the pad's height, over the finger's
  width; the pads close on one run of it along the closing axis, or on two or three runs and what lies between them, so
  one wall of a ring, one lobe of a star or one leg of an L is a grasp of its own (:func:`_spans`);
* the span fits the stroke with :data:`support_footprint.JAW_SIDE_ROOM_MM` either side, and no part of the body stands
  over the fingertips where the open fingers come down or between them and the span while they close;
* both contacts hold inside the friction cone, by the body's own flanks: how steeply it rises at the contact (a lying
  cylinder over its equator rises too gently) and how its outline turns across the pad there (a cube's corner);
* the closed and the open hand keep what SFE keeps (``support_footprint._build``): the support's clearance, the solids
  the guard holds under the hand (``HandFloor``), the neighbours' points and the guard's boxes of them
  (``scene_obstacles.SeenEnvelope``), the declared bodies and the part's own low fragments.

What it offers is a :class:`support_footprint.SupportFootprintCandidate`, scored as SFE scores (its weights: the friction
cone's slack, the stroke left, how the pads sit on the part, upright, centred), and ranked with SFE's own, robust first
(:func:`merged_ranking`): both stages offer, the better grasp first (the owner, 2026-10-13: "Beide, gemeinsam
gerankt"), and every grasp SFE offers stays. Straight down only; SFE still offers the tilted grasps.

Pure: numpy and SciPy's image tools. BASE millimetres throughout.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.robot.grasping.generation.support_footprint import (
    _SCORE_WEIGHTS,
    JAW_SIDE_ROOM_MM,
    SEEN_POINTS_CELL_MM,
    HandFloor,
    SupportFootprintCandidate,
    SupportFootprintJaw,
    _finger_points,
    _ObstacleSet,
    _Obstacles,
    _palm_corners,
    _trim_low_fragments,
)

__all__ = [
    "BODY_CELL_MM",
    "BODY_HEIGHT_STEP_MM",
    "BODY_MIN_OVERLAP_MM",
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
#: The most runs of the body along one closing line the pads close on together: a ring's two walls and its bore.
_MOST_RUNS: Final[int] = 3
#: The most anchor heights a direction is asked at.
_MOST_HEIGHTS: Final[int] = 16
#: How many grasps that hold on the part are checked against the rest of the scene at a time, best first, until
#: :data:`_ENOUGH` of them pass.
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
    from scipy import ndimage  # noqa: PLC0415 (only the stage pays for it)

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
    seen = np.isfinite(top)
    closed = ndimage.binary_closing(np.pad(seen, 1), structure=np.ones((3, 3), dtype=bool))[1:-1, 1:-1]
    new = closed & ~seen
    if new.any():
        around = ndimage.maximum_filter(top, size=3, mode="constant", cval=-np.inf)
        top = np.where(new, around, top)
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


def _rotated(body: BodyMap, yaw: float) -> "tuple[np.ndarray, float, float]":
    """The body laid on a grid square with the closing direction ``yaw``: ``top[iu, iv]`` along the closing axis and
    across it, each cell as high as the highest cell of the body that reaches into it (four samples a cell), a cell of
    nothing on every side, and the grid's low corner in the turned frame."""
    xy, height = body.centres()
    cell = body.cell_mm
    c, s = math.cos(yaw), math.sin(yaw)
    quarter = cell / 4.0
    offsets = np.array([[-quarter, -quarter], [quarter, -quarter], [-quarter, quarter], [quarter, quarter]])
    samples = (xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    tops = np.repeat(height, offsets.shape[0])
    u = c * samples[:, 0] + s * samples[:, 1]
    v = -s * samples[:, 0] + c * samples[:, 1]
    u0, v0 = float(u.min()) - cell, float(v.min()) - cell
    iu = np.floor((u - u0) / cell).astype(np.int64)
    iv = np.floor((v - v0) / cell).astype(np.int64)
    grid = np.full((int(iu.max()) + 2, int(iv.max()) + 2), -np.inf)
    np.maximum.at(grid, (iu, iv), tops)
    return grid, u0, v0


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
    :data:`_OUTLINE_CELLS` rows out either way as straight lines, the steeper side: a corner turns as far as its sides
    do, a round part's extreme as little as its curvature over those rows.
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
    worst = np.zeros(k)
    for side in (-1, 1):
        for step in range(1, _OUTLINE_CELLS + 1):
            at = middle + side * step
            inside = (at >= 0) & (at < r)
            rise = np.full(k, np.nan)
            rise[inside] = depth[np.nonzero(inside)[0], at[inside]] / step
            worst = np.maximum(worst, np.where(np.isfinite(rise), np.arctan(np.maximum(rise, 0.0)), 0.0))
    turn = np.where(count > 0, np.where(across, 0.0, worst), math.pi / 2.0)
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
    :data:`support_footprint.JAW_SIDE_ROOM_MM` either side (a Robotiq hand that is told a width, lever G). ``refusals``, where given, counts why the spans the pads close on were not offered,
    under SFE's causes (``support_footprint.REFUSAL_CAUSES``): ``aperture``, ``prism`` (the open fingers on the part
    itself) and ``cone`` for the part; ``table``, ``seen_corridor`` and ``seen_fingers`` for the rest of the scene.

    Args:
        cloud_base_mm: The part's points, ``(N, 3)`` BASE millimetres.
        support_height_mm: What the part stands on, BASE z millimetres.
        jaw: The hand, as SFE plans it.
        obstacle_points_base_mm: The neighbours' points the camera saw, as SFE keeps the hand off them.
        rigid_obstacle_points_base_mm: The declared bodies' points, kept off undilated.
        seen_envelope: The guard's boxes of the neighbours (``scene_obstacles.SeenEnvelope``), or ``None``.
        hand_floor: The solids the guard holds under the hand (``support_footprint.HandFloor``), or ``None``.
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
    # The anchor heights: the fingertips the support's clearance over it at the lowest, the pad holding the part's top by
    # the overlap at the highest.
    lowest = support + jaw.table_clearance_mm + jaw.finger_ahead_mm
    highest = float(body.top[np.isfinite(body.top)].max()) - BODY_MIN_OVERLAP_MM + jaw.pad_ahead_mm
    if highest < lowest:
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
        # where they come down, nor between them and the span while they close.
        middle_u = (first + last + 1) / 2.0
        open_mm = (np.minimum(span + 2.0 * JAW_SIDE_ROOM_MM, jaw.aperture_mm) if opens_as_needed
                   else np.full(span.shape, opening))
        outer_left = np.floor(middle_u - open_mm / (2.0 * cell) - finger_cells).astype(np.int64)
        outer_right = np.ceil(middle_u + open_mm / (2.0 * cell) + finger_cells).astype(np.int64) - 1
        after = _next_at_or_after(over_tips)
        before = _last_at_or_before(over_tips)
        left_from = np.clip(outer_left, 0, nu - 1)
        right_from = np.clip(outer_right, 0, nu - 1)
        clear = (after[left_from, bands, levels] >= first) & (before[right_from, bands, levels] <= last)
        count("prism", int(np.count_nonzero(~clear)))
        bands, levels, first, last, span, middle_u = (a[clear] for a in (bands, levels, first, last, span, middle_u))
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
        bands, levels, first, last, span, middle_u, angle = (
            a[holds] for a in (bands, levels, first, last, span, middle_u, angle))
        # Robust: the open fingers stand off the part by the error, and the band the error off either way across the
        # table still meets the span and runs no wider than a cell past it.
        cumulative = np.concatenate([np.zeros((1, nv, heights.size), dtype=np.int64),
                                     np.cumsum(stands, axis=0)], axis=0)
        open_kept = (np.minimum(span + 2.0 * JAW_SIDE_ROOM_MM, jaw.aperture_mm) if opens_as_needed
                     else np.full(span.shape, opening))
        # A hand that opens only as needed stands its fingers the room beside the part, not the error, off it: it is
        # robust across the closing axis by that room alone.
        robust = ((open_kept - span) / 2.0 >= _ROBUST_MM) | bool(opens_as_needed)
        for side in (-shift, shift):
            other = bands + side
            inside = (other >= 0) & (other < nv)
            other = np.clip(other, 0, nv - 1)
            meets_there = (cumulative[last + 1, other, levels] - cumulative[first, other, levels]) > 0
            low_out = np.clip(first - 2, 0, nu - 1)
            high_out = np.clip(last + 2, 0, nu - 1)
            wider = stands[low_out, other, levels] | stands[high_out, other, levels]
            robust &= inside & meets_there & ~wider
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
        # Centred: between the contacts it is by construction; across, how near the band stands to the part's middle,
        # where the pads hold the most of it.
        across_lo, across_hi = _across_extent(grid, levels, pad_low)
        reach = np.maximum((across_hi - across_lo) / 2.0, 1.0)
        centred = 1.0 - np.minimum(1.0, np.abs(bands - (across_lo + across_hi) / 2.0) / reach)
        score = (weights[0] * (1.0 - angle / cone) + weights[1] * (1.0 - span / jaw.aperture_mm)
                 + weights[2] * (0.5 * covered + 0.5 * middle) + weights[3] + weights[4] * centred)
        for n in range(bands.size):
            found.append(_Found(anchor=np.array([xs[n], ys[n], z[n]]), axis=axis, width_mm=float(span[n]),
                                angle_rad=float(angle[n]), score=float(score[n]),
                                clearance_mm=float(z[n] - jaw.finger_ahead_mm - support), robust=bool(robust[n]),
                                contact_z=float(pad_low[levels[n]] + BODY_MIN_OVERLAP_MM)))
    if not found:
        return []
    # The rest of the scene, best first, a batch at a time, until enough pass.
    found.sort(key=lambda f: (not f.robust, -f.score))
    obstacles = _ObstacleSet(
        _Obstacles(obstacle_points_base_mm) if seen_envelope is None
        else _Obstacles(obstacle_points_base_mm, cell_mm=SEEN_POINTS_CELL_MM, margin_mm=SEEN_POINTS_CELL_MM),
        _Obstacles(_joined(rigid_obstacle_points_base_mm, body.trimmed), cell_mm=4.0, margin_mm=0.0),
    )
    passed: list[_Found] = []
    for start in range(0, len(found), _BATCH):
        batch = _refined(found[start:start + _BATCH], body, jaw, widest=widest, count=count)
        passed.extend(_kept(batch, jaw=jaw, opening=opening, opens_as_needed=opens_as_needed, support=support,
                            obstacles=obstacles, seen_envelope=seen_envelope, hand_floor=hand_floor, count=count))
        if len(passed) >= _ENOUGH:
            break
    out: list[SupportFootprintCandidate] = []
    for f in passed:
        candidate = SupportFootprintCandidate(
            score=f.score, position_mm=f.anchor, approach=np.array([0.0, 0.0, -1.0]), closing_axis=f.axis,
            grip_width_mm=f.width_mm + 2.0, contact_angle_rad=f.angle_rad, clearance_mm=f.clearance_mm,
            robust=f.robust)
        if not _near(candidate, out):
            out.append(candidate)
        if len(out) >= int(max_candidates):
            break
    return out


def _refined(batch: "list[_Found]", body: BodyMap, jaw: SupportFootprintJaw, *, widest: float,
             count: Callable[[str, int], None]) -> "list[_Found]":
    """Each grasp of ``batch`` with its contacts where the body's own points put them, not where its cells do: of the
    points in the band of the pad, whose cells stand into the pad, within two cells of the span the cells found, the
    first and the last along the closing axis; the anchor moved to their middle, the width theirs. The cells of the
    grid stand up to a cell wider than the points in them; a span the points make wider than the stroke allows is
    refused for the aperture."""
    out: list[_Found] = []
    half_band = jaw.finger_width_mm / 2.0
    points, tops = body.points, body.point_tops
    margin = 2.0 * body.cell_mm
    for f in batch:
        axis = f.axis
        across = np.array([-axis[1], axis[0], 0.0])
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
        anchor = f.anchor + ((low + high) / 2.0) * axis
        out.append(_Found(anchor=anchor, axis=axis, width_mm=width, angle_rad=f.angle_rad, score=f.score,
                          clearance_mm=f.clearance_mm, robust=f.robust, contact_z=f.contact_z))
    return out


def _kept(batch: "list[_Found]", *, jaw: SupportFootprintJaw, opening: float, opens_as_needed: bool, support: float,
          obstacles: _ObstacleSet, seen_envelope: Any, hand_floor: HandFloor | None,
          count: Callable[[str, int], None]) -> "list[_Found]":
    """The grasps of ``batch`` whose hand keeps what SFE keeps (``support_footprint._build``): the closed fingers the
    support's clearance and the held solid's distance, the palm the solid's, the open fingers on their way in the
    solid's; the neighbours' points and the declared bodies off the open and the closed fingers; the guard's boxes of the
    neighbours off the open hand (``SeenEnvelope``). A verdict SFE would ask again one build at a time (2) is a refusal
    here: the stage offers only what it is sure of."""
    approach = np.array([0.0, 0.0, -1.0])
    reach = jaw.finger_reach_mm + 80.0
    if seen_envelope is not None and batch:
        positions = np.stack([f.anchor for f in batch])
        rotations = np.stack([np.column_stack([f.axis, np.cross(approach, f.axis), approach]) for f in batch])
        envelope = np.asarray(seen_envelope.refuses_many(positions, rotations), dtype=np.int8)
    else:
        envelope = np.zeros(len(batch), dtype=np.int8)
    kept: list[_Found] = []
    for n, f in enumerate(batch):
        axis = f.axis
        binormal = np.cross(approach, axis)
        binormal /= max(float(np.linalg.norm(binormal)), _EPS)
        reach_out = f.width_mm / 2.0 + jaw.finger_thickness_mm / 2.0
        closed = np.concatenate([
            _finger_points(f.anchor + reach_out * axis, approach, binormal, jaw),
            _finger_points(f.anchor - reach_out * axis, approach, binormal, jaw)])
        if float(closed[:, 2].min()) < support + jaw.table_clearance_mm:
            count("table", 1)
            continue
        if hand_floor is not None and (hand_floor.under(closed, fingers=True)
                                       or hand_floor.under(_palm_corners(f.anchor, axis, approach, binormal, jaw))):
            count("table", 1)
            continue
        wide = min(f.width_mm + 2.0 * JAW_SIDE_ROOM_MM, jaw.aperture_mm) if opens_as_needed else opening
        swept = np.concatenate([
            _finger_points(f.anchor + (wide / 2.0) * axis, approach, binormal, jaw, reach, 8.0),
            _finger_points(f.anchor - (wide / 2.0) * axis, approach, binormal, jaw, reach, 8.0)])
        swept = swept[swept[:, 2] >= support]
        if hand_floor is not None and hand_floor.under(swept, fingers=True):
            count("table", 1)
            continue
        if obstacles.hits(swept):
            count("seen_corridor", 1)
            continue
        if obstacles.hits(closed):
            count("seen_fingers", 1)
            continue
        if int(envelope[n]) != 0:
            count("seen_fingers", 1)
            continue
        kept.append(f)
    return kept


def _joined(*clouds: "np.ndarray | None") -> "np.ndarray | None":
    """The clouds given as one ``(N, 3)`` array, or ``None`` where none holds a point."""
    parts = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in clouds if c is not None and np.size(c)]
    return np.vstack(parts) if parts else None


def _near(candidate: SupportFootprintCandidate, kept: "list[SupportFootprintCandidate]") -> bool:
    """Whether ``candidate`` is a grasp near one of ``kept``: within 4 mm, its closing axis the same within 5 degrees."""
    return any(float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
               and abs(float(candidate.closing_axis @ k.closing_axis)) > math.cos(math.radians(5.0))
               for k in kept)


def merged_ranking(first: "list[SupportFootprintCandidate]", second: "list[SupportFootprintCandidate]", *,
                   most: int, apart: bool = False) -> list[SupportFootprintCandidate]:
    """Two stages' grasps ranked together as each ranks its own (``support_footprint.rank_found``): the robust ones
    first, each by score. Every grasp of ``first`` stays as its stage offered it, so ``second`` only adds: on the owner's
    Tuesday picks a body's grasp ranked in an SFE grasp's place had taken the one grasp the guard admitted from a pick.
    A grasp of ``second`` within 4 mm of one of ``first``, its axes alike, is left out, unless ``apart``: the two open the
    jaws differently (lever G), and the guard judges other fingers.

    Args:
        first: SFE's grasps, all of which stay.
        second: The body's.
        most: How many of ``second`` at most.
        apart: Whether a grasp of ``second`` opens the jaws otherwise than one of ``first`` at its place.

    Returns:
        list[SupportFootprintCandidate]: The grasps of both, ranked.
    """
    added = [candidate for candidate in second
             if apart or not any(float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
                                 and abs(float(candidate.closing_axis @ k.closing_axis)) > 0.995
                                 and float(candidate.approach @ k.approach) > 0.995 for k in first)][:int(most)]
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
