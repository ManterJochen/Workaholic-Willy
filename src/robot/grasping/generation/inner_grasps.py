"""Inner grasps: the hand comes down into a part's bore with its jaws narrow and opens them onto its wall
(every cell with a Robotiq hand, the owner, 2026-10-14: "Nur wo außen nichts geht").

A ring wider than the stroke cannot be closed across at all, and a ring, a tube or a coupling star's hub packed among
its neighbours leaves no room for a finger outside it; from inside it can be held. The fingers come down into the bore
as narrow as the bore leaves room for (a Robotiq told a width: their outer faces :data:`hand.NARROW_ROOM_MM`
inside the wall, lever G's room, closed where it leaves less), open at the close's force until the wall stops them, and
the part is lifted as a held part; at the place the jaws close and let it go (``grasping.motion.execution_policy``,
``execution.handling``). A toggle would need two changes of its output more a pick and says nothing of how it holds:
no hand but a Robotiq is offered one (``calculator_factory.inner_grasps``).

This stage plans on the part's measured body (``body_grasps.body_map``, the 2 mm grid of columns) beside the body's
outer search, which stays as it is, and its grasps rank after every outer grasp (``_support_footprint_stage``): they are
taken where no outer grasp is.

* A bore is a hollow of the body the body encloses in the plane: the cells no point stands in, or whose top stands at
  least :func:`body_grasps.body_least_height_mm` under the part's top, or that a hole of the part's colour mask covers
  (``footprint_hull.mask_hole_xy``: the depth inside a bore reads its wall's mixed pixels, the colour mask does not,
  the owner's L of 2026-10-15), joined to none at the grid's edge (:func:`bores`). What stands a cell inside such a hole
  is no part of the body. Its centre is the middle of the largest circle inside it.
* Through every bore's centre, every :data:`body_grasps.BODY_YAW_STEP_DEG` of closing direction and at every height the
  outer search asks (its pads :data:`body_grasps.HOLD_OVERLAP_MM` under the part's top down to the fingertips the
  support's clearance over it, and never so deep that the housing comes nearer the part's top than the room below), the
  closing line meets the bore's wall either side of the centre: the first of the body the fingers meet as they open,
  standing over their tips in the finger's band, read off the body's own points. There it stands into the pad by
  :data:`body_grasps.BODY_MIN_OVERLAP_MM`, or the fingers would open onto a ledge under the pad.
* The hand fits: the jaws as they come down keep the room each side from the wall along the closing
  axis and across it, the larger of :data:`hand.JAW_SIDE_ROOM_MM` and the guard's distance, with nothing
  of the body under their tips but the support's clearance below them; and the open stroke meets the wall with
  :data:`hand.JAW_SIDE_ROOM_MM` of it left a side.
* Both contacts hold inside the friction cone, read as the outer search reads them (``body_grasps._turn_at``): how the
  wall leans over the pad (a bore that widens upward leans away from it), how it turns across the pad (a pad that
  bridges the arc of a round bore meets it at both its edges and pushes along its own normal; off the bore's centre it
  meets the arc at one corner, turned), and the line between the two.
* The fingers' way down into the bore, and their way out to its wall, are space a ray of the frame reached
  (``scene_obstacles.SeenInTheFrame``, the body search's own seen-space rule): the guard does not hold the part it
  grasps, so a bore is free only where the camera saw it free, down to where the fingers go. Without a seen test no
  inner grasp is offered.
* The rest of the scene as the outer search judges it (``body_grasps._kept``): the neighbours' points and the declared
  bodies off the closed fingers on their way down and off the open fingers at the wall, the solids the guard holds
  under the hand, and the guard's boxes of the neighbours off the hand as the guard holds it from the way in to the
  lift: its fingers at the wall, the widest they stand, :data:`FINGER_OPENING_SLACK_MM` wider
  (``types.grasp_point.fingers_held_at``).

What it offers is an :class:`InnerGrasp`: where the hand goes and its closing axis (a ``SupportFootprintCandidate``,
whose ``grip_width_mm`` is the jaws' opening where their outer faces meet the wall and whose ``open_width_mm`` is the
narrow opening the hand comes down at), how wide the jaws stand on the way in, the bore across, how deep the fingertips
stand under the part's top, and the part's extent about the grasp, which the pick records. Scored as the
outer search scores (the friction cone's slack, the stroke left, how the pads sit on the wall, the room from every
obstacle point, the closing line across the part's centre of mass), robust first.

What none of this can judge: the guard keeps the part it grasps out of its world, so the bore's wall, anything lying in
the bore that is no part of a neighbour's mask, and the part's own top under the housing are this stage's alone to keep
the hand from, by the room above and the seen-space rule. Untried on the cell and in URSim until the owner's test.

Pure: numpy and SciPy's image tools. BASE millimetres throughout.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from src.robot.core.gripper import FINGER_OPENING_SLACK_MM
from src.robot.grasping.generation.body_grasps import (
    _BATCH,
    _HOLD_SHARE,
    _LEAN_RISE_MM,
    _MOST_HEIGHTS,
    BODY_HEIGHT_STEP_MM,
    BODY_MIN_OVERLAP_MM,
    BODY_YAW_STEP_DEG,
    HOLD_OVERLAP_MM,
    BodyMap,
    _cross,
    _joined,
    _last_at_or_before,
    _next_at_or_after,
    _rotated,
    _turn_at,
    _under_each,
    body_least_height_mm,
    body_map,
)
from src.robot.grasping.generation.hand import (
    _CLEARANCE_FULL_MM,
    _SCORE_WEIGHTS,
    JAW_SIDE_ROOM_MM,
    NARROW_ROOM_MM,
    ROBUST_ERROR_MM,
    SEEN_POINTS_CELL_MM,
    HandFloor,
    SupportFootprintCandidate,
    SupportFootprintJaw,
    _ObstacleSet,
    _Obstacles,
    _palm_corners,
    _Room,
)

__all__ = [
    "INNER_SEEN_STEP_MM",
    "Bore",
    "InnerGrasp",
    "bores",
    "hole_cells",
    "inner_grasps",
]

#: How far apart the points stand that sample the fingers' way into a bore and out to its wall, millimetres: under the
#: seen test's own tolerance (``scene_obstacles.SEEN_TOLERANCE_MM``), so no unseen gap between two of them is missed.
INNER_SEEN_STEP_MM: Final[float] = 3.0
#: The fewest cells a hollow holds to be a bore: four 2 mm cells, a hollow no finger goes into, are a speck.
_FEWEST_CELLS: Final[int] = 4
#: How far short of the wall the fingers' way out to it is sampled for the seen test, millimetres: the wall itself is
#: the part's surface, where a ray stops.
_SHORT_OF_THE_WALL_MM: Final[float] = 1.0
_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class Bore:
    """A hollow of a part's body the body encloses in the plane, where closed fingers may come down.

    Attributes:
        centre_mm (np.ndarray): Its centre, BASE x and y millimetres: the middle of the largest circle inside it.
        diameter_mm (float): The diameter of that circle, millimetres.
    """

    centre_mm: np.ndarray
    diameter_mm: float


@dataclass(frozen=True, slots=True)
class InnerGrasp:
    """One grasp from inside a bore, BASE millimetres.

    Attributes:
        candidate (SupportFootprintCandidate): Where the hand goes and how it closes: its ``grip_width_mm`` is the
            jaws' opening where their outer faces meet the wall, its ``open_width_mm`` the narrow opening the hand
            comes down at.
        line_in_width_mm (float): How far apart the jaws stand on the way into the bore, between their inner faces.
        bore_mm (float): The bore across, wall to wall, along the closing axis in the finger's band, at the pads.
        depth_mm (float): How far under the part's top the fingertips stand at the grasp.
        part_width_mm (float): The part's extent about the grasp: the diameter of the circle about the hand's axis that
            holds it in the plane.
    """

    candidate: SupportFootprintCandidate
    line_in_width_mm: float
    bore_mm: float
    depth_mm: float
    part_width_mm: float


@dataclass(frozen=True, slots=True)
class _Found:
    """One inner grasp that holds on the part, before the rest of the scene is asked."""

    anchor: np.ndarray
    axis: np.ndarray
    bore_mm: float
    line_in_mm: float
    angle_rad: float
    score: float
    robust: bool
    clearance_mm: float
    depth_mm: float
    part_width_mm: float


def bores(body: BodyMap, *, deeper_than_mm: float, hollow: "np.ndarray | None" = None) -> list[Bore]:
    """The bores of ``body``: each hollow the body encloses in the plane, its cells those no point stands in or whose
    top stands ``deeper_than_mm`` or more under the part's top (``body.top_mm``), or that ``hollow`` holds (a grid the
    shape of ``body.top``: the colour mask's holes, :func:`hole_cells`), joined edge to edge to no cell at the grid's
    edge; a hollow of fewer than :data:`_FEWEST_CELLS` cells is a speck.

    A through bore holds no point (its floor is the support, under the body's floor margin); a blind one holds its floor
    deeper than the pads need.

    Args:
        body: The part's body (``body_grasps.body_map``).
        deeper_than_mm: How far under the part's top a cell's top has to stand to be hollow, millimetres.
        hollow: Cells hollow whatever stands in them, ``None`` for none.

    Returns:
        list[Bore]: The bores, largest first; empty where the body encloses none.
    """
    from scipy import ndimage  # noqa: PLC0415 (only the stage pays for it)

    top = body.top
    empty = ~np.isfinite(top) | (top < float(body.top_mm) - float(deeper_than_mm))
    if hollow is not None:
        empty = empty | np.asarray(hollow, dtype=bool)
    labels, count = ndimage.label(np.pad(empty, 1, constant_values=True))
    outside = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    labels = labels[1:-1, 1:-1]
    found: list[Bore] = []
    for label in range(1, int(count) + 1):
        if label in outside:
            continue
        cells = labels == label
        if int(cells.sum()) < _FEWEST_CELLS:
            continue
        # How far each of its cells stands from the nearest cell of the body, centre to centre, in cells: the largest
        # circle inside it is centred where that is greatest, and reaches half a cell short of the body's cell.
        distance = ndimage.distance_transform_edt(np.pad(cells, 1))[1:-1, 1:-1]
        widest = float(distance.max())
        middle = np.argwhere(distance >= widest - 0.5)
        centre = (middle.mean(axis=0) + 0.5) * body.cell_mm + np.asarray(body.origin_mm, dtype=np.float64)
        found.append(Bore(centre_mm=centre, diameter_mm=2.0 * max(widest - 0.5, 0.0) * body.cell_mm))
    return sorted(found, key=lambda bore: -bore.diameter_mm)


def hole_cells(body: BodyMap, hole_xy_mm: np.ndarray) -> np.ndarray:
    """The cells of ``body``'s grid a hole of the part's colour mask covers (``footprint_hull.mask_hole_xy``, BASE x and
    y): a grid the shape of ``body.top``.

    Args:
        body: The part's body (``body_grasps.body_map``).
        hole_xy_mm: The mask's holes met at the part's top, ``(N, 2)`` BASE millimetres.

    Returns:
        np.ndarray: ``True`` where a hole point falls in the cell.
    """
    rows, cols = body.top.shape
    cells = np.zeros((rows, cols), dtype=bool)
    xy = np.asarray(hole_xy_mm, dtype=np.float64).reshape(-1, 2)
    index = np.floor((xy - np.asarray(body.origin_mm, dtype=np.float64)) / body.cell_mm).astype(np.int64)
    inside = (index >= 0).all(axis=1) & (index[:, 0] < rows) & (index[:, 1] < cols)
    cells[index[inside, 0], index[inside, 1]] = True
    return cells


def _without_what_stands_in_a_hole(cloud: np.ndarray, body: BodyMap, hole_xy_mm: np.ndarray) -> np.ndarray:
    """``cloud`` less its points that stand a cell inside a hole of the part's colour mask: the mixed pixels the depth
    reads inside a bore, no part of its wall, which stands at the hole's rim."""
    from scipy import ndimage  # noqa: PLC0415 (only the stage pays for it)

    inside = ndimage.binary_erosion(hole_cells(body, hole_xy_mm))
    if not inside.any():
        return cloud
    index = np.floor((cloud[:, :2] - np.asarray(body.origin_mm, dtype=np.float64)) / body.cell_mm).astype(np.int64)
    on = (index >= 0).all(axis=1) & (index[:, 0] < inside.shape[0]) & (index[:, 1] < inside.shape[1])
    held = np.zeros(cloud.shape[0], dtype=bool)
    held[on] = inside[index[on, 0], index[on, 1]]
    return cloud[~held]


def _guard_distance_mm(seen_envelope: Any, hand_floor: "HandFloor | None") -> float:
    """The distance the guard keeps from what it holds, millimetres, as the stage was handed it: the guard's boxes' own,
    else the floor's; 0 where neither was handed."""
    for holder in (seen_envelope, hand_floor):
        value = getattr(holder, "distance_mm", None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, float(value))
    return 0.0


def _nearest(along: np.ndarray, height: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """For each of ``levels``, the least of ``along`` (``(N,)``, positive) among the points standing at or over it
    (``height`` at least the level); ``inf`` where none does."""
    if not along.size:
        return np.full(levels.shape, math.inf)
    order = np.argsort(along, kind="stable")
    distance, standing = along[order], np.maximum.accumulate(height[order])
    first = np.searchsorted(standing, levels, side="left")
    return np.where(first < distance.size, distance[np.minimum(first, distance.size - 1)], math.inf)


def inner_grasps(
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
    max_candidates: int = 12,
    refusals: "dict[str, int] | None" = None,
    score_weights: "tuple[float, float, float, float, float] | None" = None,
    bores_found: "list[Bore] | None" = None,
    hole_xy_mm: "np.ndarray | None" = None,
) -> list[InnerGrasp]:
    """The grasps from inside the bores of the part's measured body (module docstring), robust first, each by score,
    ``max_candidates`` at most.

    The hand is told a width (a Robotiq, lever G): it comes down as narrow as the bore leaves
    :data:`hand.NARROW_ROOM_MM` beside each jaw, and closed where it leaves less. ``refusals``, where
    given, counts why the inner grasps asked were not offered, under the body search's causes:
    ``span`` (no wall either side of the bore's centre), ``aperture`` (the bore narrower than the closed jaws and the
    room, or wider than the stroke reaches), ``prism`` (the part's own body under the fingers' tips, a ledge under the
    pad), ``cone``, ``table``, ``unseen_corridor`` (the fingers' way into the bore or out to its wall where no ray of the
    frame reached, every bore of a part handed no seen test), ``seen_corridor`` and ``seen_fingers``.

    Args:
        cloud_base_mm: The part's points, ``(N, 3)`` BASE millimetres.
        support_height_mm: What the part stands on, BASE z millimetres.
        jaw: The hand, as the support-footprint stage plans it.
        obstacle_points_base_mm: The neighbours' points the camera saw.
        rigid_obstacle_points_base_mm: The declared bodies' points, kept off undilated.
        seen_envelope: The guard's boxes of the neighbours (``scene_obstacles.SeenEnvelope``), or ``None``.
        hand_floor: The solids the guard holds under the hand (``hand.HandFloor``), or ``None``.
        corridor_seen: Whether a ray of the frame reached a point (``scene_obstacles.SeenInTheFrame``), or ``None``,
            where no inner grasp is offered.
        floor_margin_mm: How far over the support a point still counts as the support.
        max_candidates: The most grasps returned.
        refusals: Counted into where given, as above.
        score_weights: The stage's five weights, ``None`` for its own (``hand._SCORE_WEIGHTS``).
        bores_found: Where given, the bores the part's body holds are added to it, a grasp found in them or not.
        hole_xy_mm: The holes of the part's colour mask met at its top (``footprint_hull.mask_hole_xy``), BASE x and y,
            ``None`` for none: their cells are hollow, and what stands a cell inside them is no part of the body.

    Returns:
        list[InnerGrasp]: The grasps; empty where the part has no bore a grasp from inside holds in.
    """
    counted: dict[str, int] = {} if refusals is None else refusals

    def count(cause: str, n: int = 1) -> None:
        if n:
            counted[cause] = counted.get(cause, 0) + int(n)

    body = body_map(cloud_base_mm, support_height_mm, floor_margin_mm=floor_margin_mm)
    if body is None:
        return []
    hollow = None
    if hole_xy_mm is not None and np.asarray(hole_xy_mm).size:
        body = body_map(_without_what_stands_in_a_hole(np.asarray(cloud_base_mm, dtype=np.float64).reshape(-1, 3),
                                                       body, hole_xy_mm),
                        support_height_mm, floor_margin_mm=floor_margin_mm)
        if body is None:
            return []
        hollow = hole_cells(body, hole_xy_mm)
    found_bores = bores(body, deeper_than_mm=body_least_height_mm(jaw), hollow=hollow)
    if bores_found is not None:
        bores_found.extend(found_bores)
    if not found_bores:
        return []
    if corridor_seen is None:
        # The guard does not hold the part it grasps: a bore nobody saw free is free to nobody.
        count("unseen_corridor", len(found_bores))
        return []
    cell = body.cell_mm
    support = body.support_mm
    thickness, width = float(jaw.finger_thickness_mm), float(jaw.finger_width_mm)
    room = max(JAW_SIDE_ROOM_MM, _guard_distance_mm(seen_envelope, hand_floor))
    closed = float(jaw.min_width_mm)
    widest = float(jaw.aperture_mm - jaw.width_safety_mm)
    top = float(body.top[np.isfinite(body.top)].max())
    # The anchor heights as the outer search asks them, and never so deep that the housing, finger_behind over the
    # anchor, comes nearer the part's top than the room: the guard holds no part it grasps under the hand.
    lowest = max(support + jaw.table_clearance_mm + jaw.finger_ahead_mm, top + room - jaw.finger_behind_mm)
    hold = max(BODY_MIN_OVERLAP_MM, min(HOLD_OVERLAP_MM, top - (lowest - jaw.pad_ahead_mm)))
    highest = top - hold + jaw.pad_ahead_mm
    if highest < lowest:
        count("table", len(found_bores))
        return []
    heights = np.arange(highest, lowest - 1e-9, -BODY_HEIGHT_STEP_MM)
    if heights.size > _MOST_HEIGHTS:
        heights = heights[np.linspace(0, heights.size - 1, _MOST_HEIGHTS).round().astype(int)]
    tips = heights - jaw.finger_ahead_mm
    pad_low = heights - jaw.pad_ahead_mm
    weights = score_weights or _SCORE_WEIGHTS
    half_part = (body.top_mm - support) / 2.0
    band_cells = max(1, int(round(width / cell)))
    band_cells += 1 - band_cells % 2
    half_band = band_cells // 2
    cells_xy, cells_top = body.centres()
    weighed = np.maximum(cells_top - support, 0.0)
    com_xy = (cells_xy * weighed[:, None]).sum(axis=0) / max(float(weighed.sum()), _EPS)
    xy, z_points = body.points[:, :2], body.points[:, 2]
    yaws = np.radians(np.arange(0.0, 180.0, BODY_YAW_STEP_DEG))
    turned = [_rotated(body, float(yaw)) for yaw in yaws]

    found: list[_Found] = []
    for bore in found_bores:
        if bore.diameter_mm < min(closed + 2.0 * thickness, width) + 2.0 * room - 2.0 * cell:
            # Narrower than the closed jaws or the finger's width, and the room, whichever way it is turned.
            count("aperture", yaws.size * heights.size)
            continue
        centre = np.asarray(bore.centre_mm, dtype=np.float64)
        reach = float(np.sqrt(((xy - centre) ** 2).sum(axis=1)).max())
        for yaw, (grid, u0, v0) in zip(yaws, turned):
            found.extend(_through_the_centre(
                grid, u0, v0, float(yaw), centre, reach, jaw=jaw, cell=cell, support=support, top=top,
                heights=heights, tips=tips, pad_low=pad_low, half_band=half_band, room=room, closed=closed,
                widest=widest, xy=xy, z_points=z_points, half_part=half_part,
                com_xy=com_xy, weights=weights, count=count))
    if not found:
        return []
    found.sort(key=lambda f: (not f.robust, -f.score))
    obstacles = _ObstacleSet(
        _Obstacles(obstacle_points_base_mm) if seen_envelope is None
        else _Obstacles(obstacle_points_base_mm, cell_mm=SEEN_POINTS_CELL_MM, margin_mm=SEEN_POINTS_CELL_MM),
        _Obstacles(_joined(rigid_obstacle_points_base_mm, body.trimmed), cell_mm=4.0, margin_mm=0.0),
    )
    room_from = _Room(obstacle_points_base_mm, rigid_obstacle_points_base_mm, body.trimmed)
    out: list[InnerGrasp] = []
    for start in range(0, len(found), _BATCH):
        for kept in _kept(found[start:start + _BATCH], jaw=jaw, support=support, top=top, obstacles=obstacles,
                          seen_envelope=seen_envelope, hand_floor=hand_floor, seen=corridor_seen, room=room_from,
                          room_weight=weights[2] * (1.0 - _HOLD_SHARE), count=count):
            grasp = _offered(kept, jaw)
            if not any(_near(grasp.candidate, other.candidate) for other in out):
                out.append(grasp)
        if len(out) >= int(max_candidates):
            break
    out.sort(key=lambda g: (not g.candidate.robust, -float(g.candidate.score)))
    return out[:int(max_candidates)]


def _through_the_centre(
    grid: np.ndarray, u0: float, v0: float, yaw: float, centre: np.ndarray, reach: float, *, jaw: SupportFootprintJaw,
    cell: float, support: float, top: float, heights: np.ndarray, tips: np.ndarray, pad_low: np.ndarray,
    half_band: int, room: float, closed: float, widest: float, xy: np.ndarray,
    z_points: np.ndarray, half_part: float, com_xy: np.ndarray,
    weights: "tuple[float, float, float, float, float]", count: Callable[[str, int], None],
) -> list[_Found]:
    """The grasps that hold on the part from inside a bore whose centre is ``centre``, closing along ``yaw``, at each of
    ``heights``; ``grid``, ``u0`` and ``v0`` are the body laid along the closing axis (``body_grasps._rotated``) and
    ``reach`` how far the body's farthest point stands from the centre in the plane."""
    thickness, width = float(jaw.finger_thickness_mm), float(jaw.finger_width_mm)
    levels = heights.size
    c, s = math.cos(yaw), math.sin(yaw)
    axis = np.array([c, s, 0.0])
    across = np.array([-s, c, 0.0])
    nu, nv = grid.shape
    iu = int(math.floor((c * float(centre[0]) + s * float(centre[1]) - u0) / cell))
    iv = int(math.floor((-s * float(centre[0]) + c * float(centre[1]) - v0) / cell))
    if not (0 < iu < nu - 1 and 0 < iv < nv - 1):
        count("span", levels)
        return []
    rows = np.clip(iv + np.arange(-half_band, half_band + 1), 0, nv - 1)
    column = grid[:, rows].max(axis=1)                                            # (nu,) over the finger's width
    meets = column[:, None] >= tips[None, :]                                      # the opening finger meets it
    stands = column[:, None] >= (pad_low + BODY_MIN_OVERLAP_MM)[None, :]          # it stands into the pad
    higher = column[:, None] >= (pad_low + BODY_MIN_OVERLAP_MM + _LEAN_RISE_MM)[None, :]
    level = np.arange(levels)
    right = _next_at_or_after(meets)[iu]
    left = _last_at_or_before(meets)[iu]
    # The fingers open from the bore's centre: the body standing there over their tips leaves them no way down.
    walled = (right > iu) & (left < iu) & (right < nu) & (left >= 0)
    blocked = meets[iu]
    count("prism", int(np.count_nonzero(blocked)))
    count("span", int(np.count_nonzero(~walled & ~blocked)))
    if not walled.any():
        return []
    right_c, left_c = np.clip(right, 0, nu - 1), np.clip(left, 0, nu - 1)
    # The first of the body each finger meets as it opens has to stand into its pad: else it opens onto a ledge.
    faced = walled & stands[right_c, level] & stands[left_c, level]
    count("prism", int(np.count_nonzero(walled & ~faced)))
    # How the wall leans over the pad: how far its outline steps back over the next _LEAN_RISE_MM up the pad.
    up_right = _next_at_or_after(higher)[right_c, level]
    up_left = _last_at_or_before(higher)[left_c, level]
    step_right = np.where(up_right < nu, up_right - right_c, nu).astype(np.float64)
    step_left = np.where(up_left >= 0, left_c - up_left, nu).astype(np.float64)
    lean_right = np.arctan2(step_right * cell, _LEAN_RISE_MM)
    lean_left = np.arctan2(step_left * cell, _LEAN_RISE_MM)
    # How the wall turns across the pad where each finger meets it, as the outer search reads a pad's contact.
    in_rows = grid[:, rows][:, :, None] >= (pad_low + BODY_MIN_OVERLAP_MM)[None, None, :]      # (nu, R, K)
    index = np.arange(nu)[:, None, None]
    firsts = np.where(in_rows & (index > iu), index, nu).min(axis=0)
    firsts = np.where(firsts < nu, firsts, -1).T                                            # (K, R)
    lasts = np.where(in_rows & (index < iu), index, -1).max(axis=0).T
    turn_right, at_right = _turn_at(firsts, right_c, forward=True)
    turn_left, at_left = _turn_at(lasts, left_c, forward=False)
    # The walls as the body's own points put them, in the finger's band through the centre: each finger meets the
    # nearest point standing over its tips; the cells of the grid stand up to a cell nearer than the points in them.
    rel = xy - centre[:2]
    along, off = rel @ axis[:2], rel @ across[:2]
    band = np.abs(off) <= width / 2.0
    to_right = _nearest(along[band & (along > 0.0)], z_points[band & (along > 0.0)], tips)
    to_left = _nearest(-along[band & (along < 0.0)], z_points[band & (along < 0.0)], tips)
    bore = np.minimum(to_right + to_left, (right_c - left_c + 1) * cell)
    middle = (to_right - to_left) / 2.0
    line = np.arctan2(np.abs(at_left - at_right) * cell, np.maximum(bore, _EPS))
    angle = np.maximum.reduce([np.arccos(np.clip(np.cos(lean_left) * np.cos(turn_left), -1.0, 1.0)),
                               np.arccos(np.clip(np.cos(lean_right) * np.cos(turn_right), -1.0, 1.0)), line])
    contact = bore - 2.0 * thickness
    line_in = np.maximum(closed, contact - 2.0 * NARROW_ROOM_MM)
    side_room = (bore - line_in - 2.0 * thickness) / 2.0
    fits = faced & np.isfinite(bore) & (side_room >= room - 1e-9) & (contact <= widest + 1e-9)
    count("aperture", int(np.count_nonzero(faced & ~fits)))
    holds = fits & (angle <= jaw.cone_rad)
    count("cone", int(np.count_nonzero(fits & ~holds)))
    out: list[_Found] = []
    com_off = float((com_xy - centre[:2]) @ across[:2])
    for k in np.nonzero(holds)[0]:
        anchor_uv = float(middle[k])
        outer = float(line_in[k]) / 2.0 + thickness
        # The jaws as they come down, the room about them along the closing axis and across it, stand over nothing of
        # the body but the support's clearance under their tips: no ledge, no floor of a blind bore, no smeared rim.
        if not _clear_under(grid, u0, v0, c, s, centre, anchor_uv, outer + room, width / 2.0 + room, cell=cell,
                            below=float(tips[k]) - jaw.table_clearance_mm):
            count("prism", 1)
            continue
        robust = bool(side_room[k] >= ROBUST_ERROR_MM) and _clear_under(
            grid, u0, v0, c, s, centre, anchor_uv, outer + max(room, ROBUST_ERROR_MM),
            width / 2.0 + max(room, ROBUST_ERROR_MM), cell=cell, below=float(tips[k]) - jaw.table_clearance_mm)
        z = float(heights[k])
        covered = float(np.clip(min(z + jaw.pad_behind_mm, top) - max(z - jaw.pad_ahead_mm, support), 0.0, None)
                        / max(jaw.pad_ahead_mm + jaw.pad_behind_mm, _EPS))
        mid = 1.0 - min(1.0, abs(z - (support + half_part)) / half_part) if half_part > _EPS else 0.0
        centred = 1.0 - min(1.0, abs(com_off) / max(reach, cell))
        score = (weights[0] * (1.0 - float(angle[k]) / jaw.cone_rad)
                 + weights[1] * (1.0 - float(contact[k]) / jaw.aperture_mm)
                 + weights[2] * _HOLD_SHARE * (0.5 * covered + 0.5 * mid) + weights[3] + weights[4] * centred)
        anchor = np.array([float(centre[0]), float(centre[1]), z]) + anchor_uv * axis
        out.append(_Found(anchor=anchor, axis=axis, bore_mm=float(bore[k]), line_in_mm=float(line_in[k]),
                          angle_rad=float(angle[k]), score=score, robust=robust,
                          clearance_mm=float(tips[k]) - support, depth_mm=top - float(tips[k]),
                          part_width_mm=2.0 * (reach + abs(anchor_uv))))
    return out


def _clear_under(grid: np.ndarray, u0: float, v0: float, c: float, s: float, centre: np.ndarray, middle: float,
                 half_along: float, half_across: float, *, cell: float, below: float) -> bool:
    """Whether every cell of ``grid`` (the body laid along the closing axis) that a rectangle ``half_along`` either
    side of ``middle`` along the closing axis and ``half_across`` either side of the centre across it reaches stands
    lower than ``below``."""
    u_mid = c * float(centre[0]) + s * float(centre[1]) + middle
    v_mid = -s * float(centre[0]) + c * float(centre[1])
    nu, nv = grid.shape
    lo_u = max(0, int(math.floor((u_mid - half_along - u0) / cell)))
    hi_u = min(nu - 1, int(math.floor((u_mid + half_along - u0) / cell)))
    lo_v = max(0, int(math.floor((v_mid - half_across - v0) / cell)))
    hi_v = min(nv - 1, int(math.floor((v_mid + half_across - v0) / cell)))
    if lo_u > hi_u or lo_v > hi_v:
        return True
    return bool((grid[lo_u:hi_u + 1, lo_v:hi_v + 1] < below).all())


def _box_points(anchor: np.ndarray, axis: np.ndarray, binormal: np.ndarray, approach: np.ndarray, *,
                along: "tuple[float, float]", half_across: float, tip: float, length: float,
                step: float = INNER_SEEN_STEP_MM) -> np.ndarray:
    """Points ``step`` apart through a box of the hand's frame: from ``along[0]`` to ``along[1]`` along the closing
    axis, ``half_across`` either side across it, and from ``tip`` past the anchor along the approach back up by
    ``length``; every face of it sampled."""
    def samples(lo: float, hi: float) -> np.ndarray:
        n = max(2, int(math.ceil(abs(hi - lo) / step)) + 1)
        return np.linspace(lo, hi, n)

    u, v, w = np.meshgrid(samples(*along), samples(-half_across, half_across), samples(0.0, length), indexing="ij")
    base = anchor + tip * approach
    points = (base[None, :] + u.reshape(-1, 1) * axis[None, :] + v.reshape(-1, 1) * binormal[None, :]
              - w.reshape(-1, 1) * approach[None, :])
    return np.asarray(points, dtype=np.float64)


def _kept(batch: "list[_Found]", *, jaw: SupportFootprintJaw, support: float, top: float, obstacles: _ObstacleSet,
          seen_envelope: Any, hand_floor: HandFloor | None, seen: Any, room: _Room, room_weight: float,
          count: Callable[[str, int], None]) -> "list[_Found]":
    """The grasps of ``batch`` the rest of the scene keeps, each scored ``room_weight`` more as the room its fingers keep
    from ``room``'s points nears :data:`hand._CLEARANCE_FULL_MM`; the cause each other one was refused for
    is counted. The fingers' way down into the bore (the closed jaws from their tips up the finger's reach and 80 mm
    more, as the outer search sweeps an open finger's way in), their way out to the wall and the open fingers at it have
    to be seen; the closed fingers and the palm stand over the solids the guard holds under the hand; the neighbours'
    points stay off the way down and off the open fingers; and last, the dearest, the guard's boxes off the hand as the
    guard holds it on the way in."""
    thickness, width = float(jaw.finger_thickness_mm), float(jaw.finger_width_mm)
    approach = np.array([0.0, 0.0, -1.0])
    way_down = jaw.finger_reach_mm + 80.0
    asked: list[tuple[_Found, np.ndarray]] = []
    for f in batch:
        binormal = _cross(approach, f.axis)
        binormal /= max(float(np.linalg.norm(binormal)), _EPS)
        inner_face, outer_face = f.line_in_mm / 2.0, f.line_in_mm / 2.0 + thickness
        wall = f.bore_mm / 2.0
        height = top - (float(f.anchor[2]) - jaw.finger_ahead_mm) + 2.0
        down, out_to_the_wall, at_the_wall = [], [], []
        for side in (1.0, -1.0):
            down.append(_box_points(f.anchor, side * f.axis, binormal, approach, along=(inner_face, outer_face),
                                    half_across=width / 2.0, tip=jaw.finger_ahead_mm, length=way_down))
            if wall - _SHORT_OF_THE_WALL_MM > outer_face:
                out_to_the_wall.append(_box_points(
                    f.anchor, side * f.axis, binormal, approach, along=(outer_face, wall - _SHORT_OF_THE_WALL_MM),
                    half_across=width / 2.0, tip=jaw.finger_ahead_mm, length=max(height, 0.0)))
            at_the_wall.append(_box_points(f.anchor, side * f.axis, binormal, approach,
                                           along=(wall - thickness, wall - _SHORT_OF_THE_WALL_MM),
                                           half_across=width / 2.0, tip=jaw.finger_ahead_mm,
                                           length=jaw.finger_reach_mm))
        closed_way = np.vstack(down)
        opening = np.vstack(out_to_the_wall) if out_to_the_wall else np.zeros((0, 3))
        open_fingers = np.vstack(at_the_wall)
        why = _refusal(closed_way, opening, open_fingers, f, binormal, approach, jaw=jaw, support=support,
                       obstacles=obstacles, hand_floor=hand_floor, seen=seen)
        if why:
            count(why, 1)
            continue
        asked.append((f, np.vstack([closed_way, open_fingers])))
    if seen_envelope is not None and asked:
        positions = np.stack([f.anchor for f, _ in asked])
        rotations = np.stack([np.column_stack([f.axis, _cross(approach, f.axis), approach]) for f, _ in asked])
        # The hand as the guard holds it from the way in to the lift: the fingers at the wall, the widest they stand,
        # and the slack the guard is told (``types.grasp_point.fingers_held_at``).
        openings = (None if seen_envelope.open_width_mm is None else np.array(
            [min(_contact_mm(f, jaw) + FINGER_OPENING_SLACK_MM, jaw.aperture_mm) for f, _ in asked]))
        refused = np.asarray(seen_envelope.refuses_many(positions, rotations, openings), dtype=np.int8)
    else:
        refused = np.zeros(len(asked), dtype=np.int8)
    kept: list[_Found] = []
    for (f, hand), verdict in zip(asked, refused):
        if int(verdict) != 0:
            count("seen_fingers", 1)
            continue
        free = room.least_mm(hand)
        kept.append(replace(f, score=f.score + room_weight * min(1.0, free / _CLEARANCE_FULL_MM)))
    return kept


def _refusal(closed_way: np.ndarray, opening: np.ndarray, open_fingers: np.ndarray, f: _Found, binormal: np.ndarray,
             approach: np.ndarray, *, jaw: SupportFootprintJaw, support: float, obstacles: _ObstacleSet,
             hand_floor: HandFloor | None, seen: Any) -> str:
    """What the scene refuses inner grasp ``f`` for, the guard's boxes aside (:func:`_kept` asks them last); ``""``
    where it refuses nothing."""
    if float(closed_way[:, 2].min()) < support + jaw.table_clearance_mm - 1e-9:
        return "table"
    if hand_floor is not None:
        palm = _palm_corners(f.anchor, f.axis, approach, binormal, jaw)
        under = _under_each(hand_floor, [closed_way, open_fingers], fingers=True)
        if bool(under.any()) or bool(_under_each(hand_floor, [palm]).any()):
            return "table"
    # Over the part's own outline every point of the fingers' way has to be seen: the guard holds no part it grasps.
    for points in (closed_way, opening, open_fingers):
        if points.shape[0] and not bool(np.all(seen(points))):
            return "unseen_corridor"
    if obstacles.hits(closed_way):
        return "seen_corridor"
    if obstacles.hits(open_fingers):
        return "seen_fingers"
    return ""


def _contact_mm(f: _Found, jaw: SupportFootprintJaw) -> float:
    """How far apart the jaws stand where their outer faces meet the wall of ``f``'s bore, millimetres."""
    return max(f.line_in_mm, f.bore_mm - 2.0 * float(jaw.finger_thickness_mm))


def _offered(f: _Found, jaw: SupportFootprintJaw) -> InnerGrasp:
    """``f`` as the stage offers it."""
    candidate = SupportFootprintCandidate(
        score=f.score, position_mm=f.anchor, approach=np.array([0.0, 0.0, -1.0]), closing_axis=f.axis,
        grip_width_mm=_contact_mm(f, jaw), contact_angle_rad=f.angle_rad, clearance_mm=f.clearance_mm, robust=f.robust,
        open_width_mm=f.line_in_mm)
    return InnerGrasp(candidate=candidate, line_in_width_mm=f.line_in_mm, bore_mm=f.bore_mm, depth_mm=f.depth_mm,
                      part_width_mm=f.part_width_mm)


def _near(candidate: SupportFootprintCandidate, other: SupportFootprintCandidate) -> bool:
    """Whether two inner grasps are one to the cell: within 4 mm, their closing axes the same within 5 degrees."""
    return (float(np.linalg.norm(candidate.position_mm - other.position_mm)) < 4.0
            and abs(float(candidate.closing_axis @ other.closing_axis)) > math.cos(math.radians(5.0)))
