"""What a camera saw of one object, as upright columns from the floor to the highest point seen over them.

A depth camera measures surfaces. What stands under a surface it saw was not measured and is not free, so every
column reaches down to the floor the object stands on: the bench, or the top of a keep-out it rests over. What stands
beside a surface it saw is another matter. One box around a whole cluster, which is what the camera world fitted
until 2026-09-30, covered all of it: an open bin came back as a block over its inside, a low part beside a tall one
took the tall one's height, and a square part turned on the bench was boxed square with BASE, 1.4 times as wide. The
owner's bins beside the robot's base were refused by exactly those boxes.

So a cluster is laid on a grid in the bench plane, turned the way the cluster lies (:func:`turn_of`) and cut into
cells of at least the clustering grid, 25 mm. Each cell keeps the highest point seen in it. Neighbouring cells of about one height, within
the cloud's own resolution of each other, whose points reach as far across the rectangle as its own do, within that
resolution too, merge into rectangles, and each rectangle is one box: its points' extent along the turned axes grown
by the margin, from the floor up to its highest point and the margin. A cell with no point seen in it is in no box, as
unseen space is nowhere else in the world. An open bin is its walls and a free inside, each wall as thick as it was
seen and each corner a box of its own (a corner cell holds the other wall's points, and a wall that took it was boxed
a cell thick: the grasp bench, 2026-10-05), two parts stay two boxes, and a turned part is a turned box.

Except where the robot itself hid it. A camera over a bin beside the base cannot see the stretch of rim under the
UR10's shoulder housing, because the housing is in the way; one box per cluster covered that stretch, and a height map
of what the camera saw left it free, so a rim 6 mm under the housing was accepted (review of 2026-09-30). So where the
caller says what the robot hides (``hidden``), every cell of the grid whose column the robot hides from the cameras,
and no camera showed free past it, stands as high as the cells seen in and beside that hidden stretch, and a cell seen
lower than that whose space above the robot hides is raised to it: never higher than the object was seen, never wider
than its grid, which is the one box it would have been. Past the grid only two things are filled. A row of cells that
runs into the grid's edge, a wall seen up to where the Hand-E's padded spheres took the rest of it for the hand, runs
on while the robot hides the next cell and the self filter took points in it no higher than the row, as far as the
reach at most; an object is never widened across its own row. And what the robot hid between two objects its shadow
cut apart is bridged, no further than they reach (:func:`bridge_columns`). What the scene hides from itself, the floor
behind a wall, stays free, and so does the robot's mere shadow beside an object.

The one promise every box keeps is the one the guard's distance to a seen box rests on: every point it was fitted to
lies inside it, the margin from each of its four sides and from its top, and the box reaches down to the floor. A
merge (:func:`coarsen`) only ever replaces two boxes with the box that holds both, so it keeps the promise too.

Pure: points in, boxes out, millimetres throughout. The turned frame is BASE turned about Z by the cluster's yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["Column", "bridge_columns", "coarsen", "height_map_columns", "turn_of"]

#: Where in a cell its column is asked whether the robot hides it, as shares of the cell along and across: its centre
#: and four points a fifth in from its corners. A choice: a cell no point was seen in is hidden whole or lies beside
#: what hid it, and five columns find either.
_SAMPLES_IN_A_CELL = ((0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8))
#: The most heights a hidden column is asked at, spread evenly from the floor to the object's top: a step apart up to
#: 120 mm tall, further apart above. A choice, which bounds the cost of a tall object.
_MOST_LEVELS = 12

#: How much smaller than the rectangle square with BASE the smallest turned rectangle about a cluster has to be before
#: the cluster is turned: 5 %. A choice, not a measurement. A round part or a noisy square one is about as small either
#: way, and a box turned by noise differs from one frame to the next, so the planner would be handed a different world
#: for a scene that did not move. A rectangle turned by a degree and a half already gains more than this.
_TURN_GAIN = 0.05


def turn_of(points_xy: np.ndarray) -> float:
    """How far to turn a box about BASE Z to sit closest around ``points_xy``, radians in (-pi/2, pi/2].

    The turn of the smallest rectangle about the points, along its longer side: one of its sides always lies along an
    edge of their convex hull, so every hull edge is tried. It is exact for a rectangle seen whole, and it holds for an
    object seen in part, where the principal direction does not: two walls of a bin meeting at a corner have their
    principal direction running between them, and their smallest rectangle along them. Where the rectangle square
    with BASE is not :data:`_TURN_GAIN` larger, the turn is 0.
    """
    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 2:
        return 0.0
    hull = _hull(xy)
    edges = np.roll(hull, -1, axis=0) - hull
    edges = edges[np.hypot(edges[:, 0], edges[:, 1]) > 1e-9]
    if edges.shape[0] == 0:
        return 0.0
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    along = cos * hull[None, :, 0] + sin * hull[None, :, 1]
    across = -sin * hull[None, :, 0] + cos * hull[None, :, 1]
    long_side = along.max(axis=1) - along.min(axis=1)
    short_side = across.max(axis=1) - across.min(axis=1)
    area = long_side * short_side
    best = int(np.argmin(area))
    square = float(np.ptp(hull[:, 0]) * np.ptp(hull[:, 1]))
    if float(area[best]) >= (1.0 - _TURN_GAIN) * square:
        return 0.0
    yaw = float(angles[best]) + (math.pi / 2.0 if short_side[best] > long_side[best] else 0.0)
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    if yaw <= -math.pi / 2.0:
        yaw += math.pi
    elif yaw > math.pi / 2.0:
        yaw -= math.pi
    return yaw


def _hull(xy: np.ndarray) -> np.ndarray:
    """The convex hull of ``xy``; for points on one line, its two ends; for one point repeated, that point."""
    from scipy.spatial import ConvexHull, QhullError  # noqa: PLC0415 (kept out of import time)

    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        centred = xy - xy.mean(axis=0)
        if not bool(np.any(np.abs(centred) > 1e-12)):
            return xy[:1]
        _, vectors = np.linalg.eigh(centred.T @ centred)
        line = centred @ vectors[:, -1]
        return xy[[int(np.argmin(line)), int(np.argmax(line))]]


@dataclass(eq=False)
class Column:
    """One box of a height map, in its cluster's turned frame: the points it holds and its two corners.

    ``low`` and ``high`` are ``(along, across, z)`` millimetres, the margin already in them. ``low[2]`` is the floor
    the box stands on where one is given, and otherwise its lowest point less the margin.
    """

    #: Which points of the cluster it was fitted to.
    members: np.ndarray
    low: np.ndarray
    high: np.ndarray
    #: How many of its cells stand where the robot hid the cell from the cameras, filled or raised to the height seen
    #: beside them; 0 for a box of what was seen alone.
    hidden_cells: int = 0

    @property
    def volume_mm3(self) -> float:
        return float(np.prod(np.maximum(self.high - self.low, 0.0)))

    def placed(self, yaw_rad: float) -> "tuple[tuple[float, float, float], tuple[float, float, float]]":
        """Its centre in BASE and its full sizes along the turned axes, millimetres."""
        middle = (self.low + self.high) / 2.0
        cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
        centre = (float(cos * middle[0] - sin * middle[1]), float(sin * middle[0] + cos * middle[1]), float(middle[2]))
        size = self.high - self.low
        return centre, (float(size[0]), float(size[1]), float(size[2]))

    def holds(self, other: "Column") -> bool:
        return bool(np.all(self.low <= other.low) and np.all(self.high >= other.high))


def height_map_columns(
    points_mm: np.ndarray,
    *,
    yaw_rad: float,
    floor_mm: float | None,
    margin_mm: float,
    cell_mm: float,
    step_mm: float,
    hidden: "Callable[[np.ndarray], np.ndarray] | None" = None,
    reach_mm: float = 0.0,
    taken_mm: "np.ndarray | None" = None,
    robot_on: "Callable[[np.ndarray], np.ndarray] | None" = None,
) -> list[Column]:
    """The boxes of one cluster's height map, turned by ``yaw_rad``.

    The points are laid on a grid in the turned bench plane, their span along each axis cut into whole cells of at
    least ``cell_mm`` (:func:`_cells`). Each cell
    keeps the highest point in it, and, with no floor, the lowest. Cells merge into rectangles in a fixed order, row by
    row, a rectangle growing along a row and then row by row while every cell it takes is seen, not yet taken, keeps
    its tops within ``step_mm`` of each other (its bottoms too, with no floor), and has its points reach as far across
    the rectangle as the rectangle's own do, within ``step_mm`` (:func:`_rectangles`). Each rectangle is one box grown
    by ``margin_mm``: from ``floor_mm`` where given, else its lowest point less the margin, up to its highest point
    plus the margin. The same points give the same boxes.

    ``hidden`` answers, for ``(N, 3)`` BASE points, which of them the robot's own body hides from the cameras and no
    camera showed free: a cell whose column it hides stands as high as the cells seen in and beside that hidden stretch
    (:func:`_fill_hidden`), and its box covers the whole cell. Past the grid's own cells, the extent of what was seen, a
    row that runs into its edge runs on while the robot hides the next cell and the self filter took points there no
    higher than the row, ``taken_mm``, ``reach_mm`` at most (:func:`_run_on`). ``None`` fills nothing.

    ``robot_on`` answers which BASE points lie on the robot's own body as it stands now (``SelfBody.on_itself``): no
    stretch is filled through it for its neighbours alone, its box, the margin grown round it, kept a margin under the
    robot (:func:`_under_the_robot`). Where the robot stands nothing else is: the Hand-E down in a bin hid the floor
    beside the part from the wrist camera, the floor was filled as high as the wall beside it, and the guard held a
    column 212 mm tall through the hand and refused every way out (the grasp bench, 2026-10-05). A row runs on as wide
    as the row (:func:`_run_on_extents`), and not through the robot standing in the run itself
    (:func:`_run_on_under_the_robot`): a part's row ran on a whole cell into the cell the fingers stood in, on one point
    the filter had taken there, the fingers 10 mm inside its box, and every way out was refused (the grasp bench,
    2026-10-06). A wall the robot stands beside, the margin of its box round the hand, still runs on: there the wall was
    seen up to the hand. A rim under the shoulder housing keeps its fill, the housing being over it.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return []
    cos, sin = math.cos(yaw_rad), math.sin(yaw_rad)
    along = cos * points[:, 0] + sin * points[:, 1]
    across = -sin * points[:, 0] + cos * points[:, 1]
    height = points[:, 2]
    col, cols, along_low, along_cell = _cells(along, float(cell_mm))
    row, rows, across_low, across_cell = _cells(across, float(cell_mm))
    # Room past the edge for a row to run on into, as whole cells: none where nothing is filled, and none along an axis
    # one cell deep, which no row runs along (and whose one cell may be as narrow as the points are).
    runs = hidden is not None and float(reach_mm) > 0.0
    wide = int(math.ceil(float(reach_mm) / along_cell)) if runs and cols > 1 else 0
    deep = int(math.ceil(float(reach_mm) / across_cell)) if runs and rows > 1 else 0
    col, cols, along_low = col + wide, cols + 2 * wide, along_low - wide * along_cell
    row, rows, across_low = row + deep, rows + 2 * deep, across_low - deep * across_cell
    top = np.full((rows, cols), -np.inf)
    bottom = np.full((rows, cols), np.inf)
    np.maximum.at(top, (row, col), height)
    np.minimum.at(bottom, (row, col), height)
    filled = np.zeros((rows, cols), dtype=bool)
    if hidden is not None:
        grid = (float(yaw_rad), along_low, along_cell, across_low, across_cell)
        inner = np.zeros((rows, cols), dtype=bool)
        inner[deep:rows - deep, wide:cols - wide] = True
        # The lowest point the self filter took for the robot in each cell, where a row may run on into.
        taken_low = np.full((rows, cols), np.inf)
        taken = np.zeros((0, 3)) if taken_mm is None else np.asarray(taken_mm, dtype=np.float64).reshape(-1, 3)
        if taken.shape[0] and (wide or deep):
            taken_col = np.floor((cos * taken[:, 0] + sin * taken[:, 1] - along_low) / along_cell).astype(np.int64)
            taken_row = np.floor((-sin * taken[:, 0] + cos * taken[:, 1] - across_low) / across_cell).astype(np.int64)
            on_grid = (taken_col >= 0) & (taken_col < cols) & (taken_row >= 0) & (taken_row < rows)
            np.minimum.at(taken_low, (taken_row[on_grid], taken_col[on_grid]), taken[on_grid, 2])
        seen_top, seen_bottom = top.copy(), bottom.copy()
        top, bottom, filled = _fill_hidden(top, bottom, grid, hidden, floor_mm=floor_mm, step_mm=float(step_mm),
                                           within=inner)
        if robot_on is not None and bool(filled.any()):
            # A stretch filled for its neighbours alone, never one run on where the filter took the wall for the hand.
            top, bottom, filled = _under_the_robot(top, bottom, filled, seen_top, seen_bottom, grid, robot_on,
                                                   float(margin_mm), floor_mm)
        top, bottom, ran, sources = _run_on(top, bottom, grid, hidden, inner=(deep, rows - deep, wide, cols - wide),
                                            taken_low=taken_low, floor_mm=floor_mm, step_mm=float(step_mm))
    else:
        seen_top, seen_bottom = top, bottom
        ran, sources = np.zeros((rows, cols), dtype=bool), {}
    # How far each cell's points reach along and across, the whole cell where the robot hid it, as wide as its row where
    # a row ran on into it: a rectangle takes a cell only where the two reach as far across it, within the cloud's
    # resolution (:func:`_rectangles`).
    cells = (along_low, along_cell, across_low, across_cell)
    extents = _cell_extents(along, across, row, col, (rows, cols), filled, cells)
    if bool(ran.any()):
        extents = _run_on_extents(extents, ran, sources, cells)
        if robot_on is not None:
            # A row is never run on through the robot standing in the run itself (:func:`_run_on_under_the_robot`).
            top, bottom, ran = _run_on_under_the_robot(top, bottom, ran, extents, seen_top, seen_bottom,
                                                       float(yaw_rad), robot_on, float(margin_mm), floor_mm)
        filled |= ran
    rectangle = _rectangles(top, bottom, float(step_mm), floored=floor_mm is not None, extents=extents)

    label = rectangle[row, col]
    order = np.argsort(label, kind="stable")
    starts = np.flatnonzero(np.diff(label[order], prepend=-1))
    groups = {int(label[order[first]]): members for first, members in zip(starts, np.split(order, starts[1:]))}
    columns: list[Column] = []
    for number in range(int(rectangle.max()) + 1):
        members = groups.get(number, np.zeros(0, dtype=np.int64))
        # A cell the robot hid is unknown over all of it, so its box covers the whole cell, not a point in it.
        cover = np.nonzero((rectangle == number) & filled)
        near, far = [], []
        if members.size:
            near.append((float(along[members].min()), float(across[members].min())))
            far.append((float(along[members].max()), float(across[members].max())))
        if cover[0].size:
            # Over its extent: the whole cell where the robot hid it, its row's width where a row ran on into it.
            near.append((float(extents[0][cover].min()), float(extents[2][cover].min())))
            far.append((float(extents[1][cover].max()), float(extents[3][cover].max())))
        highest = max([float(height[members].max())] if members.size else [], default=-np.inf)
        if cover[0].size:
            highest = max(highest, float(top[cover].max()))
        lowest = min([float(height[members].min())] if members.size else [], default=np.inf)
        if cover[0].size and floor_mm is None:
            lowest = min(lowest, float(bottom[cover].min()))
        columns.append(Column(
            members=members,
            low=np.array([min(v[0] for v in near) - margin_mm, min(v[1] for v in near) - margin_mm,
                          (min(float(floor_mm), lowest) if members.size else float(floor_mm)) if floor_mm is not None
                          else lowest - margin_mm]),
            high=np.array([max(v[0] for v in far) + margin_mm, max(v[1] for v in far) + margin_mm,
                           highest + margin_mm]),
            hidden_cells=int(cover[0].size),
        ))
    return columns


def _cells(values: np.ndarray, cell_mm: float) -> "tuple[np.ndarray, int, float, float]":
    """Each value's cell along one axis, how many cells, where the first begins and how wide each is: the values' span
    cut into whole cells of at least ``cell_mm``, so the last cell is as full as the first. A span cut at exactly
    ``cell_mm`` from its low end leaves a sliver of a cell at its high end, which the points thin out across, and a
    sliver half seen splits the rectangles beside it for nothing. Values with no span are one cell ``cell_mm`` wide
    about them."""
    low = float(values.min())
    span = float(values.max() - low)
    if span <= 0.0:
        return np.zeros(values.shape[0], dtype=np.int64), 1, low - cell_mm / 2.0, float(cell_mm)
    count = max(1, int(math.floor(span / cell_mm)))
    index = np.floor((values - low) * (count / span)).astype(np.int64)
    return np.minimum(index, count - 1), count, low, span / count


def _fill_hidden(
    top: np.ndarray,
    bottom: np.ndarray,
    grid: "tuple[float, float, float, float, float]",
    hidden: "Callable[[np.ndarray], np.ndarray]",
    *,
    floor_mm: float | None,
    step_mm: float,
    within: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The height map with what the robot hid filled: its tops, its bottoms, and which cells were filled or raised.

    Each cell's column is asked of ``hidden`` at :data:`_SAMPLES_IN_A_CELL`, at heights about ``step_mm`` apart from
    the floor up to the object's highest point (:data:`_MOST_LEVELS` at most), those over the cell's own seen top where
    one was seen. A cell with any of those hidden is one the robot hid. Each stretch of hidden cells, touching at a side
    or a corner, stands as high as the highest cell seen in it or beside it, and a hidden stretch nothing was seen in or
    beside stays as it was: nothing says there is anything there but the robot. With no floor it reaches down to the
    lowest of them.

    ``grid`` is ``(yaw, along_low, along_cell, across_low, across_cell)``: the turn and where the cells lie. ``within``
    is the cells of what was seen, the rest of the grid room to run on into (:func:`_run_on`); ``None`` is all of it.
    """
    from scipy import ndimage  # noqa: PLC0415 (kept out of import time)

    seen = np.isfinite(top)
    base = float(floor_mm) if floor_mm is not None else float(bottom[seen].min())
    shaded = _shaded(top, grid, hidden, base_mm=base, step_mm=step_mm, asked_cells=within)
    filled = np.zeros_like(shaded)
    if not bool(shaded.any()):
        return top, bottom, filled
    top, bottom = top.copy(), bottom.copy()
    stretches, count = ndimage.label(shaded, structure=_TOUCHING)
    for number in range(1, count + 1):
        stretch = stretches == number
        beside = ndimage.binary_dilation(stretch, structure=_TOUCHING) & seen
        if not bool(beside.any()):
            continue
        height = float(top[beside].max())
        lower = stretch & (top < height)
        top[lower] = height
        bottom[stretch] = np.minimum(bottom[stretch], float(bottom[beside].min()))
        filled |= lower
    return top, bottom, filled


#: Two cells touch at a side or at a corner.
_TOUCHING = np.ones((3, 3), dtype=bool)


def _run_on(
    top: np.ndarray,
    bottom: np.ndarray,
    grid: "tuple[float, float, float, float, float]",
    hidden: "Callable[[np.ndarray], np.ndarray]",
    *,
    inner: "tuple[int, int, int, int]",
    taken_low: np.ndarray,
    floor_mm: float | None,
    step_mm: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], tuple[int, int, bool]]]":
    """Every row and column of the height map that runs into the edge of what was seen, run on past it where the self
    filter took the rest of it for the robot: its tops, its bottoms, which cells it ran into, and for each of those the
    edge cell it continues and whether it runs along the grid's columns (:func:`_run_on_extents`).

    The self filter takes what a camera sees within the padding of the robot's body for the robot, and about the hand
    that is up to 27 mm of its mesh: a wall 6 mm from the Hand-E's fingers is seen up to that far from them, and a
    camera looking past the hand sees nothing of the rest of it (review of 2026-09-30). So a row runs into an edge where
    its last cell and the one inside it both stand (seen, or filled by :func:`_fill_hidden`) within ``step_mm`` of one
    height, one wall, and it runs on, as high as those two, one cell at a time into the room past the edge, while
    ``hidden`` hides that cell's column (asked as
    :func:`_shaded` asks, no higher than the row) and the self filter took a point in it no higher than half a step
    over the row (``taken_low``, the lowest point it took in each cell): stopping at the first cell that is not so, or
    at the end of the room. Only where the filter took something: past what a camera saw of a bin's wall, the robot's
    mere shadow is no sign the wall goes on, and the housing over it stands higher than the wall. A row one cell deep
    does not run across itself, so an object is never widened, only continued. Two cells of two heights are two things
    and no wall: a tray's wall and a small cylinder beside it, one cluster of two cells each, ran on past the cylinder as
    high as the wall, into the cell where the hand stood at the part, on the cube's own top the filter had taken for the
    hand (the grasp bench, 2026-10-06). ``inner`` is ``(first_row, end_row, first_col, end_col)`` of what was seen, the
    rest being the room.
    """
    first_row, end_row, first_col, end_col = inner
    rows, cols = top.shape
    stands = np.isfinite(top)
    # (row, col, step row, step col, height) of every row and column that runs into an edge, and its first cell out.
    # (first cell out, step, height, the edge cell it continues)
    runs: list[tuple[int, int, int, int, float, int, int]] = []

    def one_wall(a: float, b: float) -> bool:
        return abs(a - b) <= float(step_mm)

    for r in range(first_row, end_row):
        for edge, inward, out in ((end_col - 1, end_col - 2, 1), (first_col, first_col + 1, -1)):
            if (first_col <= inward < end_col and stands[r, edge] and stands[r, inward]
                    and one_wall(float(top[r, edge]), float(top[r, inward]))):
                runs.append((r, edge + out, 0, out, float(max(top[r, edge], top[r, inward])), r, edge))
    for c in range(first_col, end_col):
        for edge, inward, out in ((end_row - 1, end_row - 2, 1), (first_row, first_row + 1, -1)):
            if (first_row <= inward < end_row and stands[edge, c] and stands[inward, c]
                    and one_wall(float(top[edge, c]), float(top[inward, c]))):
                runs.append((edge + out, c, out, 0, float(max(top[edge, c], top[inward, c])), edge, c))
    ran = np.zeros(top.shape, dtype=bool)
    sources: dict[tuple[int, int], tuple[int, int, bool]] = {}
    if not runs:
        return top, bottom, ran, sources
    room = np.zeros(top.shape, dtype=bool)
    # A cell is asked up to the height of the highest row that would run into it: what stands over that is not the row.
    cap = np.full(top.shape, -np.inf)
    for r, c, dr, dc, height, _r0, _c0 in runs:
        while 0 <= r < rows and 0 <= c < cols and not (first_row <= r < end_row and first_col <= c < end_col):
            room[r, c] = True
            cap[r, c] = max(float(cap[r, c]), height)
            r, c = r + dr, c + dc
    seen = np.isfinite(top)
    base = float(floor_mm) if floor_mm is not None else float(bottom[seen].min())
    shaded = _shaded(top, grid, hidden, base_mm=base, step_mm=step_mm, asked_cells=room, cap=cap)
    top, bottom = top.copy(), bottom.copy()
    lowest = float(bottom[seen].min())
    for r, c, dr, dc, height, r0, c0 in runs:
        while (0 <= r < rows and 0 <= c < cols and room[r, c] and shaded[r, c]
               and taken_low[r, c] <= height + float(step_mm) / 2.0):
            if top[r, c] < height:
                top[r, c] = height
                ran[r, c] = True
                sources[(r, c)] = (r0, c0, dc != 0)
            bottom[r, c] = min(float(bottom[r, c]), lowest)
            r, c = r + dr, c + dc
    return top, bottom, ran, sources


def _run_on_extents(
    extents: "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]", ran: np.ndarray,
    sources: "dict[tuple[int, int], tuple[int, int, bool]]", grid: "tuple[float, float, float, float]",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """The cells' extents with each cell a row ran on into (:func:`_run_on`) as long as the cell along the run and as
    wide across it as the edge cell it continues: a wall 5 mm thick runs on 5 mm thick, not as thick as the cell. Until
    2026-10-06 a cell run on was boxed over its whole width, and a part's row run on into the hand's own cell was one box
    through the fingers (the grasp bench)."""
    along_low, along_cell, across_low, across_cell = grid
    along_lo, along_hi, across_lo, across_hi = (np.array(values, copy=True) for values in extents)
    for (r, c), (r0, c0, along_run) in sources.items():
        if not ran[r, c]:
            continue
        if along_run:
            along_lo[r, c], along_hi[r, c] = along_low + c * along_cell, along_low + (c + 1) * along_cell
            across_lo[r, c], across_hi[r, c] = extents[2][r0, c0], extents[3][r0, c0]
        else:
            across_lo[r, c], across_hi[r, c] = across_low + r * across_cell, across_low + (r + 1) * across_cell
            along_lo[r, c], along_hi[r, c] = extents[0][r0, c0], extents[1][r0, c0]
    return along_lo, along_hi, across_lo, across_hi


def _run_on_under_the_robot(
    top: np.ndarray,
    bottom: np.ndarray,
    ran: np.ndarray,
    extents: "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
    seen_top: np.ndarray,
    seen_bottom: np.ndarray,
    yaw: float,
    robot_on: "Callable[[np.ndarray], np.ndarray]",
    margin_mm: float,
    floor_mm: float | None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The cells a row ran on into kept out of where the robot stands in the run itself: its tops, its bottoms, and
    which cells stay run on.

    Each such cell is asked over its own extent (:func:`_run_on_extents`), not grown, from its floor up to twice the
    margin over its top, every :data:`_ROBOT_SAMPLE_MM`: where the robot stands in the run, the row did not go on
    through it, and the cell is lowered to keep its box a margin under the robot's lowest point there, or is no longer
    run on where that is under its floor. Where the robot stands only beside the run, within the margin its box grows
    by, the run stands: a wall 6 mm from the Hand-E's fingers is the wall the self filter took for the hand (review of
    2026-09-30). A part's row run on a whole cell into the cell the fingers stood in put them 10 mm inside its box, and
    every way out was refused (the grasp bench, 2026-10-06).
    """
    along_lo, along_hi, across_lo, across_hi = extents
    cos, sin = math.cos(yaw), math.sin(yaw)
    grow = 2.0 * float(margin_mm)
    top, bottom, ran = top.copy(), bottom.copy(), ran.copy()
    for r, c in zip(*np.nonzero(ran)):
        fill = float(top[r, c])
        floor = float(floor_mm) if floor_mm is not None else float(bottom[r, c])
        lo_a, hi_a = float(along_lo[r, c]), float(along_hi[r, c])
        lo_b, hi_b = float(across_lo[r, c]), float(across_hi[r, c])
        if not all(math.isfinite(v) for v in (fill, floor, lo_a, hi_a, lo_b, hi_b)) or fill <= floor:
            continue
        alongs = np.linspace(lo_a, hi_a, max(2, int(math.ceil((hi_a - lo_a) / _ROBOT_SAMPLE_MM)) + 1))
        acrosses = np.linspace(lo_b, hi_b, max(2, int(math.ceil((hi_b - lo_b) / _ROBOT_SAMPLE_MM)) + 1))
        heights = np.arange(floor, fill + grow + 1e-9, _ROBOT_SAMPLE_MM)
        grid_a, grid_b, grid_z = np.meshgrid(alongs, acrosses, heights, indexing="ij")
        samples = np.column_stack((cos * grid_a.ravel() - sin * grid_b.ravel(),
                                   sin * grid_a.ravel() + cos * grid_b.ravel(), grid_z.ravel()))
        on = np.asarray(robot_on(samples), dtype=bool).reshape(-1)
        if not bool(on.any()):
            continue
        cap = float(samples[on, 2].min()) - grow
        if cap <= max(float(seen_top[r, c]), floor):
            top[r, c], bottom[r, c] = float(seen_top[r, c]), float(seen_bottom[r, c])
            ran[r, c] = False
        else:
            top[r, c] = cap
    return top, bottom, ran


#: How far apart the samples stand that ask a filled cell's column whether the robot's body passes through it, along
#: each axis and up, millimetres: a hand's finger, 10.8 mm thick, is not missed between two of them.
_ROBOT_SAMPLE_MM = 5.0


def _under_the_robot(
    top: np.ndarray,
    bottom: np.ndarray,
    filled: np.ndarray,
    seen_top: np.ndarray,
    seen_bottom: np.ndarray,
    grid: "tuple[float, float, float, float, float]",
    robot_on: "Callable[[np.ndarray], np.ndarray]",
    margin_mm: float,
    floor_mm: float | None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The filled cells kept out of where the robot stands now: its tops, its bottoms, and which cells stay filled.

    ``robot_on`` answers which of ``(N, 3)`` BASE points lie on the robot's own body (``SelfBody.on_itself``). Each
    filled cell's column, grown by twice the margin (the margin its box grows by, and as much again to keep), is asked
    from its floor up to twice the margin over its fill, as high as its box and the margin past it reach, every
    :data:`_ROBOT_SAMPLE_MM`: the lowest point on the robot is where the robot stands down to there. A fill over that put
    the column's box through the robot itself, which nothing else can be in: it is lowered to keep its box a margin under
    that point, never under what was seen in the cell. A fill the robot stands higher over is kept as it is, so a rim the
    shoulder housing hides keeps its height under the housing. A cell lowered to what was seen in it, or under the floor
    of what it stands on, is no longer filled. Asked up to the fill alone, a hand hanging less than the margin over a
    fill stood inside the box grown round it: the Hand-E's housing 0.9 mm deep in a tray's wall filled across to the
    part, and the retreat was refused (the grasp bench, 2026-10-06).
    """
    rows_i, cols_i = np.nonzero(filled)
    if rows_i.size == 0:
        return top, bottom, filled
    yaw, along_low, along_cell, across_low, across_cell = grid
    cos, sin = math.cos(yaw), math.sin(yaw)
    grow = 2.0 * float(margin_mm)
    step = _ROBOT_SAMPLE_MM
    top, bottom, filled = top.copy(), bottom.copy(), filled.copy()
    for r, c in zip(rows_i, cols_i):
        fill = float(top[r, c])
        seen = float(seen_top[r, c])
        # The floor of what it stands on: the given floor, or where the fill reached down to.
        floor = float(floor_mm) if floor_mm is not None else float(bottom[r, c])
        if not (math.isfinite(fill) and math.isfinite(floor)) or fill <= floor:
            continue
        a0, b0 = along_low + c * along_cell - grow, across_low + r * across_cell - grow
        alongs = np.arange(a0, a0 + along_cell + 2.0 * grow + 1e-9, step)
        acrosses = np.arange(b0, b0 + across_cell + 2.0 * grow + 1e-9, step)
        heights = np.arange(floor, fill + grow + 1e-9, step)
        grid_a, grid_b, grid_z = np.meshgrid(alongs, acrosses, heights, indexing="ij")
        samples = np.column_stack((cos * grid_a.ravel() - sin * grid_b.ravel(),
                                   sin * grid_a.ravel() + cos * grid_b.ravel(), grid_z.ravel()))
        on = np.asarray(robot_on(samples), dtype=bool).reshape(-1)
        if not bool(on.any()):
            continue
        cap = float(samples[on, 2].min()) - grow
        if cap <= max(seen, floor):
            # Lowered to what was seen in it, or under its floor: it stands as it was seen, unfilled.
            top[r, c], bottom[r, c] = seen, float(seen_bottom[r, c])
            filled[r, c] = False
        else:
            top[r, c] = cap
    return top, bottom, filled


def _shaded(
    top: np.ndarray,
    grid: "tuple[float, float, float, float, float]",
    hidden: "Callable[[np.ndarray], np.ndarray]",
    *,
    base_mm: float,
    step_mm: float,
    asked_cells: "np.ndarray | None" = None,
    cap: "np.ndarray | None" = None,
) -> np.ndarray:
    """Which cells' columns ``hidden`` hides somewhere, ``(rows, cols)`` boolean.

    Each cell is asked at :data:`_SAMPLES_IN_A_CELL`, at heights about ``step_mm`` apart from ``base_mm`` up to the
    highest cell (:data:`_MOST_LEVELS` at most), those over the cell's own top where it has one and, with ``cap``, none
    over its cap (the lowest height stays asked). ``asked_cells`` leaves the rest out. ``grid`` is ``(yaw, along_low,
    along_cell, across_low, across_cell)``: the turn and where the cells lie.
    """
    seen = np.isfinite(top)
    rows, cols = top.shape
    ceiling = float(top[seen].max())
    rise = ceiling - float(base_mm)
    if rise > 0.0:
        count = min(_MOST_LEVELS, max(1, int(math.ceil(rise / float(step_mm)))))
        levels = float(base_mm) + (np.arange(count) + 0.5) * (rise / count)
    else:
        levels = np.array([ceiling])
    yaw, along_low, along_cell, across_low, across_cell = grid
    shares = np.asarray(_SAMPLES_IN_A_CELL, dtype=np.float64)
    cell_rows, cell_cols = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # (rows, cols, share, level): where each sample stands and whether it is above what the cell's own points showed.
    along = along_low + (cell_cols[:, :, None, None] + shares[None, None, :, 0, None]) * along_cell
    across = across_low + (cell_rows[:, :, None, None] + shares[None, None, :, 1, None]) * across_cell
    level = np.broadcast_to(levels[None, None, None, :], (rows, cols, shares.shape[0], levels.size))
    asked = level > np.where(seen, top, -np.inf)[:, :, None, None]
    if cap is not None:
        asked = asked & ((level <= cap[:, :, None, None]) | (level == levels[0]))
    if asked_cells is not None:
        asked = asked & asked_cells[:, :, None, None]
    along, across = np.broadcast_to(along, level.shape)[asked], np.broadcast_to(across, level.shape)[asked]
    cos, sin = math.cos(yaw), math.sin(yaw)
    samples = np.column_stack((cos * along - sin * across, sin * along + cos * across, level[asked]))
    hid = np.zeros(level.shape, dtype=bool)
    if samples.shape[0]:
        hid[asked] = np.asarray(hidden(samples), dtype=bool).reshape(-1)
    return np.asarray(hid.any(axis=(2, 3)))


def bridge_columns(
    parts: "Sequence[np.ndarray]",
    *,
    floor_mm: float | None,
    margin_mm: float,
    cell_mm: float,
    step_mm: float,
    hidden: "Callable[[np.ndarray], np.ndarray]",
    reach_mm: float,
    robot_on: "Callable[[np.ndarray], np.ndarray] | None" = None,
) -> list[Column]:
    """Boxes over what the robot hid between two objects or more, square with BASE, none of them holding a point.

    A camera looking past the UR10's shoulder housing at a bin beside the base sees the bin in two, the housing's
    shadow across it, and the two halves are two clusters (review of 2026-09-30): a height map of each fills what the
    robot hid inside its own extent (:func:`height_map_columns`), and the stretch between them is in neither. So the
    ``parts``, each an ``(N, 3)`` cloud in BASE, are laid on one grid square with BASE in cells of ``cell_mm``, and
    every cell no part was seen in, within ``reach_mm`` of one that was, is asked of ``hidden`` as a height map asks
    its own (:func:`_shaded`). A stretch of hidden cells, touching at a side or a corner, that lies beside two parts or
    more stands as high as the highest cell seen beside it, over the cells of it within the extent of the parts beside
    it; a stretch beside one part is that part's own shadow and stays free, as unseen space beside an object is
    everywhere else. Each rectangle of such cells, of about one height, is one box: the cells, cut back to the extent of
    the points of the parts beside them, grown by ``margin_mm``, from ``floor_mm`` where given, else the lowest of the
    parts less the margin. So no bridge reaches further than one box around the parts it joins would have.

    ``robot_on`` keeps a bridge out of where the robot stands now as it keeps a height map's fill
    (:func:`_under_the_robot`): what the robot hides between two parts is often where its own hand stands. The Hand-E at
    a cube in a tray hid the floor between the cube and the wall from the wrist camera, the bridge stood as high as the
    wall through the fingers, and every way out was refused (the grasp bench, 2026-10-06).
    """
    from scipy import ndimage  # noqa: PLC0415 (kept out of import time)

    clouds = [np.asarray(part, dtype=np.float64).reshape(-1, 3) for part in parts]
    clouds = [cloud for cloud in clouds if cloud.shape[0]]
    if len(clouds) < 2:
        return []
    everything = np.concatenate(clouds)
    cell = float(cell_mm)
    low = everything[:, :2].min(axis=0)
    cols, rows = (np.floor((everything[:, :2].max(axis=0) - low) / cell).astype(np.int64) + 1).tolist()
    top = np.full((rows, cols), -np.inf)
    bottom = np.full((rows, cols), np.inf)
    owner = np.full((rows, cols), -1, dtype=np.int64)
    for number, cloud in enumerate(clouds):
        index = np.minimum(np.floor((cloud[:, :2] - low) / cell).astype(np.int64), [cols - 1, rows - 1])
        np.maximum.at(top, (index[:, 1], index[:, 0]), cloud[:, 2])
        np.minimum.at(bottom, (index[:, 1], index[:, 0]), cloud[:, 2])
        mine = np.zeros((rows, cols), dtype=bool)
        mine[index[:, 1], index[:, 0]] = True
        # A cell two parts were seen in is where they meet: it borders neither alone.
        owner[mine] = np.where(owner[mine] == -1, number, -2)
    seen = np.isfinite(top)
    near = ndimage.binary_dilation(seen, structure=_TOUCHING, iterations=max(1, int(math.ceil(float(reach_mm) / cell))))
    base = float(floor_mm) if floor_mm is not None else float(bottom[seen].min())
    shaded = _shaded(top, (0.0, float(low[0]), cell, float(low[1]), cell), hidden, base_mm=base, step_mm=step_mm,
                     asked_cells=near & ~seen)
    if not bool(shaded.any()):
        return []
    lowest = float(bottom[seen].min())
    columns: list[Column] = []
    stretches, count = ndimage.label(shaded, structure=_TOUCHING)
    for number in range(1, count + 1):
        stretch = stretches == number
        beside = ndimage.binary_dilation(stretch, structure=_TOUCHING) & seen
        owners = set(owner[beside].tolist())
        parts_beside = sorted(owners - {-1, -2})
        if len(parts_beside) < 2 and -2 not in owners:
            continue
        # Over the extent of the parts beside it and no further: the shadow past them is beside one part at most.
        joined = np.concatenate([clouds[index] for index in parts_beside]) if parts_beside else everything
        near_xy, far_xy = joined[:, :2].min(axis=0), joined[:, :2].max(axis=0)
        filled_top = np.full((rows, cols), -np.inf)
        filled_top[stretch] = float(top[beside].max())
        filled_bottom = np.where(stretch, lowest, np.inf)
        if robot_on is not None:
            # Nothing was seen in a bridge's cells: one the robot stands down to under the floor stands no more.
            filled_top, filled_bottom, kept = _under_the_robot(
                filled_top, filled_bottom, stretch, np.full((rows, cols), -np.inf), np.full((rows, cols), np.inf),
                (0.0, float(low[0]), cell, float(low[1]), cell), robot_on, float(margin_mm), floor_mm)
            if not bool(kept.any()):
                continue
            filled_top = np.where(kept, filled_top, -np.inf)
            filled_bottom = np.where(kept, filled_bottom, np.inf)
        rectangle = _rectangles(filled_top, filled_bottom, float(step_mm), floored=True)
        for piece in range(int(rectangle.max()) + 1):
            cells_rows, cells_cols = np.nonzero(rectangle == piece)
            first = np.maximum([float(low[0]) + cells_cols.min() * cell, float(low[1]) + cells_rows.min() * cell],
                               near_xy)
            last = np.minimum([float(low[0]) + (cells_cols.max() + 1) * cell, float(low[1]) + (cells_rows.max() + 1) * cell],
                              far_xy)
            if bool(np.any(last < first)):
                continue
            columns.append(Column(
                members=np.zeros(0, dtype=np.int64),
                low=np.array([first[0] - margin_mm, first[1] - margin_mm,
                              float(floor_mm) if floor_mm is not None else lowest - margin_mm]),
                high=np.array([last[0] + margin_mm, last[1] + margin_mm,
                               float(filled_top[cells_rows, cells_cols].max()) + margin_mm]),
                hidden_cells=int(cells_rows.size),
            ))
    return columns


def _cell_extents(
    along: np.ndarray, across: np.ndarray, row: np.ndarray, col: np.ndarray, shape: "tuple[int, int]",
    filled: np.ndarray, grid: "tuple[float, float, float, float]",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """How far each cell's points reach, as four grids of ``shape``: the least and the most along, the least and the
    most across. A cell the robot hid (``filled``) reaches over the whole cell, as its box does; a cell with nothing in
    it reaches nowhere (``inf`` and ``-inf``). ``grid`` is ``(along_low, along_cell, across_low, across_cell)``."""
    along_low, along_cell, across_low, across_cell = grid
    along_lo, along_hi = np.full(shape, np.inf), np.full(shape, -np.inf)
    across_lo, across_hi = np.full(shape, np.inf), np.full(shape, -np.inf)
    np.minimum.at(along_lo, (row, col), along)
    np.maximum.at(along_hi, (row, col), along)
    np.minimum.at(across_lo, (row, col), across)
    np.maximum.at(across_hi, (row, col), across)
    hid_rows, hid_cols = np.nonzero(filled)
    along_lo[hid_rows, hid_cols] = along_low + hid_cols * along_cell
    along_hi[hid_rows, hid_cols] = along_low + (hid_cols + 1) * along_cell
    across_lo[hid_rows, hid_cols] = across_low + hid_rows * across_cell
    across_hi[hid_rows, hid_cols] = across_low + (hid_rows + 1) * across_cell
    return along_lo, along_hi, across_lo, across_hi


def _rectangles(
    top: np.ndarray, bottom: np.ndarray, step_mm: float, *, floored: bool,
    extents: "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None" = None,
) -> np.ndarray:
    """Every seen cell's rectangle number, -1 for a cell with nothing seen in it.

    With ``extents`` (:func:`_cell_extents`) a rectangle grows along its row only into a cell whose points reach as far
    across as the rectangle's do, and down a row only where that row's reach as far along, each within ``step_mm``, the
    cloud's own resolution. A bin's corner cell holds the other wall's points too, and a wall that took it was boxed as
    thick as the cell over its whole length: 12 mm into the bin on the grasp bench (2026-10-05), along every wall, where
    the fingers of a part beside it go. The corner is a box of its own instead.
    """
    rows, cols = top.shape
    seen = np.isfinite(top)
    rectangle = np.full(top.shape, -1, dtype=np.int64)
    count = 0

    def fits(tops: np.ndarray, bottoms: np.ndarray, bounds: list[float]) -> "list[float] | None":
        high, low = max(bounds[0], float(tops.max())), min(bounds[1], float(tops.min()))
        deepest, shallowest = max(bounds[2], float(bottoms.max())), min(bounds[3], float(bottoms.min()))
        if high - low > step_mm or (not floored and deepest - shallowest > step_mm):
            return None
        return [high, low, deepest, shallowest]

    def reaches(now: "list[float]", lo: float, hi: float) -> bool:
        return abs(lo - now[0]) <= step_mm and abs(hi - now[1]) <= step_mm

    for first_row in range(rows):
        for first_col in range(cols):
            if not seen[first_row, first_col] or rectangle[first_row, first_col] >= 0:
                continue
            bounds = [float(top[first_row, first_col])] * 2 + [float(bottom[first_row, first_col])] * 2
            along_now: list[float] = []
            across_now: list[float] = []
            if extents is not None:
                along_now = [float(extents[0][first_row, first_col]), float(extents[1][first_row, first_col])]
                across_now = [float(extents[2][first_row, first_col]), float(extents[3][first_row, first_col])]
            last_col = first_col
            while last_col + 1 < cols and seen[first_row, last_col + 1] and rectangle[first_row, last_col + 1] < 0:
                grown = fits(top[first_row, last_col + 1:last_col + 2], bottom[first_row, last_col + 1:last_col + 2],
                             bounds)
                if grown is None:
                    break
                if extents is not None:
                    lo, hi = float(extents[2][first_row, last_col + 1]), float(extents[3][first_row, last_col + 1])
                    if not reaches(across_now, lo, hi):
                        break
                    across_now = [min(across_now[0], lo), max(across_now[1], hi)]
                    along_now = [along_now[0], max(along_now[1], float(extents[1][first_row, last_col + 1]))]
                bounds, last_col = grown, last_col + 1
            last_row = first_row
            while last_row + 1 < rows:
                span = slice(first_col, last_col + 1)
                if not bool(seen[last_row + 1, span].all()) or bool((rectangle[last_row + 1, span] >= 0).any()):
                    break
                grown = fits(top[last_row + 1, span], bottom[last_row + 1, span], bounds)
                if grown is None:
                    break
                if extents is not None:
                    lo, hi = float(extents[0][last_row + 1, span].min()), float(extents[1][last_row + 1, span].max())
                    if not reaches(along_now, lo, hi):
                        break
                    along_now = [min(along_now[0], lo), max(along_now[1], hi)]
                bounds, last_row = grown, last_row + 1
            rectangle[first_row:last_row + 1, first_col:last_col + 1] = count
            count += 1
    return rectangle


def coarsen(
    parts: Sequence[list[Column]], budget: int, *, near_mm: "np.ndarray | None" = None,
    yaws: "Sequence[float] | None" = None,
) -> int:
    """Merge boxes until no more than ``budget`` are left in ``parts`` together, or each part is one box. In place.

    Only two boxes of one part merge, into the box that holds both, so nothing any box held is ever let go. The merge
    that adds the least volume to what the two already held goes first, over every part: the pieces of one surface
    before two objects, and the space between two objects before the inside of a bin. A box the merged one holds whole
    goes with them. Returns how many boxes the merging took away; what still does not fit is the caller's to refuse.

    ``near_mm`` is the goal of the motion the world is built for, in BASE, and ``yaws`` each part's turn: the volume a
    merge adds counts the more the nearer its box comes to the goal (:func:`_merge_cost`), so the boxes far from the hand
    merge first and those it passes between stay as the cameras saw them. In a pile of twenty parts on the mat, 35 to 42
    boxes merged to fit 64 slots wherever the least volume was, and the boxes beside the part grew past what the
    calculator had planned the fingers against: grasp after grasp was refused 2.5 to 3 mm short of the guard's 3 mm on
    the line down (the grasp bench, 2026-10-06).

    What each merge would add is kept as merges go (:class:`_Merges`), so a cluster of a few hundred columns merges
    down in tens of milliseconds rather than seconds (measured 2026-09-30: 494 columns to 64 took 2.9 s asked afresh
    every time).
    """
    total = sum(len(part) for part in parts)
    if total <= int(budget):
        return 0
    turns = list(yaws) if yaws is not None else [0.0] * len(parts)
    if len(turns) != len(parts):
        raise ValueError(f"coarsen needs one yaw per part, got {len(turns)} for {len(parts)}")
    merges = [_Merges(part, near=None if near_mm is None else _in_turn(np.asarray(near_mm, dtype=np.float64), yaw))
              for part, yaw in zip(parts, turns)]
    removed = 0
    while total > int(budget):
        best: "tuple[float, int, int, int] | None" = None
        for index, part_merges in enumerate(merges):
            option = part_merges.cheapest()
            if option is not None and (best is None or option[0] < best[0]):
                best = (option[0], index, option[1], option[2])
        if best is None:
            break
        _, index, first, second = best
        taken = merges[index].merge(first, second)
        removed += taken
        total -= taken
    for part, part_merges in zip(parts, merges):
        part[:] = [column for column in part_merges.slots if column is not None]
    return removed


#: How near the goal a merge counts more, millimetres: the volume a merge adds counts ``1 + (_NEAR_GOAL_MM / d) ** 2``
#: times, ``d`` how near the box holding both comes to the goal, no nearer than :data:`_NEAREST_MM`. Twice at 100 mm,
#: five times at 50, a hundred and one at 10 and in it. A choice: about the hand's own reach round the goal.
_NEAR_GOAL_MM = 100.0
_NEAREST_MM = 10.0


def _in_turn(point_mm: np.ndarray, yaw: float) -> np.ndarray:
    """A BASE point in a part's own turn, as its columns lie: along, across and up."""
    cos, sin = math.cos(float(yaw)), math.sin(float(yaw))
    x, y, z = (float(v) for v in np.asarray(point_mm, dtype=np.float64).reshape(3))
    return np.array([cos * x + sin * y, -sin * x + cos * y, z])


def _merge_cost(low: np.ndarray, high: np.ndarray, other_low: np.ndarray, other_high: np.ndarray,
                near: "np.ndarray | None") -> np.ndarray:
    """Per pair, what a merge costs: the volume it adds (:func:`_added_volume`), weighed by how near the box holding
    both comes to ``near`` where one is given (:data:`_NEAR_GOAL_MM`)."""
    added = _added_volume(low, high, other_low, other_high)
    if near is None:
        return added
    joined_low = np.minimum(low[:, None, :], other_low[None, :, :])
    joined_high = np.maximum(high[:, None, :], other_high[None, :, :])
    outside = np.maximum(np.maximum(joined_low - near, near - joined_high), 0.0)
    distance = np.maximum(np.linalg.norm(outside, axis=2), _NEAREST_MM)
    return np.asarray(added * (1.0 + (_NEAR_GOAL_MM / distance) ** 2))


def _added_volume(low: np.ndarray, high: np.ndarray, other_low: np.ndarray, other_high: np.ndarray) -> np.ndarray:
    """Per pair of a box of ``low``/``high`` and one of ``other_low``/``other_high``, the volume the box holding both
    adds to what the two hold together: ``(len(low), len(other_low))`` cubic millimetres."""
    volume = np.prod(np.maximum(high - low, 0.0), axis=1)
    other_volume = np.prod(np.maximum(other_high - other_low, 0.0), axis=1)
    joined = np.prod(np.maximum(high[:, None, :], other_high[None, :, :])
                     - np.minimum(low[:, None, :], other_low[None, :, :]), axis=2)
    shared = np.prod(np.maximum(np.minimum(high[:, None, :], other_high[None, :, :])
                                - np.maximum(low[:, None, :], other_low[None, :, :]), 0.0), axis=2)
    return np.asarray(joined - (volume[:, None] + other_volume[None, :] - shared))


class _Merges:
    """One part's boxes, and what merging any two of them would add, kept up to date as they merge.

    ``slots`` keeps each box where it was, ``None`` where a merge took it away, so the order stays the one the height
    map laid them in. Each row keeps its cheapest partner, so a merge costs a row and a column rather than the square.
    The table itself is the square, ``count`` by ``count`` doubles: 1,421 boxes of one object took 45 MB at the peak
    and 0.65 s to merge down to 64 (review of 2026-09-30).
    """

    #: Rows of the cost table computed at a time, which bounds the temporaries of computing it, not the table.
    _ROWS = 256

    def __init__(self, part: Sequence[Column], near: "np.ndarray | None" = None) -> None:
        self.slots: list[Column | None] = list(part)
        #: The goal in the part's own turn, which a merge near it pays for (:func:`_merge_cost`); ``None`` for none.
        self.near = near
        count = len(part)
        self.low = np.stack([column.low for column in part]) if count else np.zeros((0, 3))
        self.high = np.stack([column.high for column in part]) if count else np.zeros((0, 3))
        self.alive = np.ones(count, dtype=bool)
        self.cost = np.full((count, count), np.inf)
        for first in range(0, count, self._ROWS):
            rows = slice(first, min(count, first + self._ROWS))
            self.cost[rows] = _merge_cost(self.low[rows], self.high[rows], self.low, self.high, self.near)
        np.fill_diagonal(self.cost, np.inf)
        self.row_min = self.cost.min(axis=1) if count else np.zeros(0)
        self.row_arg = self.cost.argmin(axis=1) if count else np.zeros(0, dtype=np.int64)

    def cheapest(self) -> "tuple[float, int, int] | None":
        """What the cheapest merge of this part adds and its two boxes, lower index first; ``None`` for one box."""
        if int(self.alive.sum()) < 2:
            return None
        row = int(np.argmin(self.row_min))
        value = float(self.row_min[row])
        if not math.isfinite(value):
            return None
        other = int(self.row_arg[row])
        return value, min(row, other), max(row, other)

    def merge(self, first: int, second: int) -> int:
        """Merge box ``second`` into box ``first``, with every box the two together hold whole; how many went."""
        kept, other = self.slots[first], self.slots[second]
        assert kept is not None and other is not None  # both alive: the cheapest merge names live boxes
        low, high = np.minimum(self.low[first], self.low[second]), np.maximum(self.high[first], self.high[second])
        held = self.alive & np.all(self.low >= low, axis=1) & np.all(self.high <= high, axis=1)
        held[[first, second]] = False
        gone = [second, *np.flatnonzero(held).tolist()]
        merged = [kept, other, *(column for column in (self.slots[index] for index in gone[1:]) if column is not None)]
        self.slots[first] = Column(members=np.concatenate([column.members for column in merged]), low=low, high=high,
                                   hidden_cells=sum(column.hidden_cells for column in merged))
        for index in gone:
            self.slots[index] = None
        self.alive[gone] = False
        self.low[first], self.high[first] = low, high
        self.cost[gone, :] = np.inf
        self.cost[:, gone] = np.inf
        row = _merge_cost(low[None, :], high[None, :], self.low, self.high, self.near)[0]
        row[~self.alive] = np.inf
        row[first] = np.inf
        self.cost[first, :] = row
        self.cost[:, first] = row
        self.row_min[gone] = np.inf
        stale = self.alive & np.isin(self.row_arg, [first, *gone])
        stale[first] = True
        for stale_row in np.flatnonzero(stale).tolist():
            self.row_arg[stale_row] = int(np.argmin(self.cost[stale_row]))
            self.row_min[stale_row] = self.cost[stale_row, self.row_arg[stale_row]]
        better = self.alive & ~stale & (row < self.row_min)
        self.row_min[better] = row[better]
        self.row_arg[better] = first
        return len(gone)
