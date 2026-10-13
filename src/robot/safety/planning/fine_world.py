"""The camera world near where a motion goes, at the resolution the cameras measured it.

The camera world (``perceived.build_perceived_boxes``) thins every frame to one point per 10 mm voxel, joins what it
keeps in 25 mm cells, lays each cluster on a height map of at least that cell and grows every box by the cell's margin
(6 mm on the owner's cell). That is the world the planner gets, and it has to be: the planner holds a fixed number of
boxes. It was the guard's world too, and near a grasp it is the reason grasps die. On the owner's Tuesday picks
(2026-10-13, the replay bank) 304 of the 330 grasps the guard refused were refused at a box the camera saw, and the part
of the hand refused stood a median 7.1 mm from every point the cameras measured there, 68 % of them 4 mm or more: the
box, not the neighbour, took the room. A thinned point can also stand up to a voxel inside the surface it stands for,
which no margin under the voxel covers.

So near the motion's goal the guard holds this world instead: every pixel the camera world reads there, of every view
it holds, laid on a 2 mm grid in BASE x and y (:data:`FINE_CELL_MM`), each cell as high as the highest of them, carried
down to the floor as the camera world carries its boxes down, cells of one height (within one cell) and one owner merged
into rectangles, each rectangle a box grown by :data:`FINE_MARGIN_MM`. Every pixel lies in a box, a margin inside its
sides and top, and no pixel is thinned away. A box of a part the detector named holds the fingers to its measured
surface as the camera world's do (``PerceivedBox.soft_mm``); a box of anything else holds every part of the hand to the
guard's distance from it.

What the guard holds here is never less than the camera world kept: ``perceived`` checks that every point the camera
world kept within :class:`FineRegion` lies inside one of these boxes, and holds the camera world's own box beside them
where one does not (``fine_uncovered``). The planner keeps the camera world as it was.

Pure: numpy. BASE millimetres throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Sequence

import numpy as np

__all__ = [
    "FINE_CELL_MM",
    "FINE_LABEL_MM",
    "FINE_MARGIN_MM",
    "FINE_NOISE_VOXEL_MM",
    "FINE_REACH_MM",
    "FineColumn",
    "FineRegion",
    "fine_columns",
    "fringe_of",
    "noise_clusters",
    "specks",
]

#: The fine world's cell in BASE x and y, and the height step its boxes hold, millimetres: about the D415's own
#: resolution where it looks at a part from 0.5 to 0.8 m (a pixel is 0.6 to 0.9 mm there, the depth noise 0.5 to 1 mm).
FINE_CELL_MM: Final[float] = 2.0
#: What every fine box is grown by on its sides and its top, millimetres. The looks of one pick read the same floor
#: 2.35 mm apart at the median, 4.95 mm at the 90th percentile (the owner's 30 multi-look picks of 2026-10-13), and the
#: hand-eye calibration spreads 1.6 mm RMS over its 20 stations: a box built of every look's pixels holds the spread
#: between them, and this holds what one look may read off the surface, the guard's distance kept from it on top.
FINE_MARGIN_MM: Final[float] = 2.0
#: How far from the motion's goal, in BASE x and y, the fine world reaches, millimetres: past the open Hand-E (50 mm
#: and two 10.8 mm fingers across, its 75 mm palm along) and the wrist camera 60 mm beside it, at the tilts SFE offers.
FINE_REACH_MM: Final[float] = 150.0
#: A pixel lies on a named part where it lies this near one of the part's points, millimetres: the part's points are
#: pixels of one look, and the looks read one surface up to 4.95 mm apart (the 90th percentile, :data:`FINE_MARGIN_MM`).
#: The camera world's own rule takes 12 mm. A finger fits nowhere in a gap this narrow, so what it names beside a part is
#: only ever met where the part itself is.
FINE_LABEL_MM: Final[float] = 5.0
#: The voxel the fine world counts a cluster's points in for the camera world's noise rule, millimetres: the camera
#: world's own thinning voxel, so a cluster is noise here where it would be noise there (``perceived.min_points``).
FINE_NOISE_VOXEL_MM: Final[float] = 10.0
#: How far past a held target's box, in BASE x and y, a pixel no part owns is the target's own fringe, millimetres: the
#: box already stands the camera world's margin round the target's points, and the D415 smears a part's far edge 6 to
#: 13 mm into the floor behind it at the owner's cell's 45 degrees (the replay bank, 2026-10-13: the fingers of every
#: grasp of three single-look picks met such pixels and nothing else).
FINE_TARGET_FRINGE_MM: Final[float] = 8.0
#: How few pixels joined in touching :data:`FINE_SPECK_MM` cells make a speck, a flying pixel or two, which the fine
#: world leaves out: fewer than a pin 3 mm across shows at the stride the camera world reads.
FINE_SPECK_PIXELS: Final[int] = 4
FINE_SPECK_MM: Final[float] = 4.0
#: Two cells merge into one box where their heights lie within this many steps of each other: the box stands as high
#: as the higher, no more than this many steps over the lower. A part's top read at 0.5 to 1 mm of noise, a pixel or
#: two a cell, splits at one step into boxes of a cell or two (1628 boxes about one part of the owner's, 2026-10-13).
_MERGE_STEPS: Final[int] = 2
#: A cell no named part owns.
_UNNAMED: Final[int] = -1
#: The share of a cell's points one named part has to hold for the cell to be that part's: two in three. A pixel of a
#: part seen from another look lies up to the looks' spread off the part's points (:data:`FINE_LABEL_MM`), and every
#: such pixel in a part's top left the whole cell unnamed.
_OWNED_SHARE: Final[float] = 2.0 / 3.0


@dataclass(frozen=True, slots=True)
class FineRegion:
    """Where the fine world stands: a square about the motion's goal in BASE x and y, at every height."""

    centre_xy: tuple[float, float]
    half_mm: float

    @classmethod
    def about(cls, goal_mm: "Sequence[float] | np.ndarray", reach_mm: float = FINE_REACH_MM) -> "FineRegion":
        """The square ``reach_mm`` either side of ``goal_mm`` in BASE x and y."""
        goal = np.asarray(goal_mm, dtype=np.float64).reshape(-1)
        return cls(centre_xy=(float(goal[0]), float(goal[1])), half_mm=float(reach_mm))

    def contains(self, points_mm: np.ndarray) -> np.ndarray:
        """Which of ``(N, 3)`` (or ``(N, 2)``) BASE points stand over the square."""
        pts = np.asarray(points_mm, dtype=np.float64)
        if pts.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        return ((np.abs(pts[:, 0] - self.centre_xy[0]) <= self.half_mm)
                & (np.abs(pts[:, 1] - self.centre_xy[1]) <= self.half_mm))

    def render(self) -> str:
        """One clause for a log line."""
        return (f"{2.0 * self.half_mm:.0f} mm square about ({self.centre_xy[0]:.0f}, {self.centre_xy[1]:.0f}) at "
                f"{FINE_CELL_MM:.0f} mm")


@dataclass(frozen=True, slots=True)
class FineColumn:
    """One box of the fine world: upright, square with BASE, from its floor up to its top, the margin in it."""

    low_mm: tuple[float, float, float]
    high_mm: tuple[float, float, float]
    #: How many cells and how many pixels it holds.
    cells: int
    pixels: int
    #: The named part every pixel of it lies on, or ``None``.
    owner: int | None

    @property
    def centre_mm(self) -> tuple[float, float, float]:
        return tuple(float((a + b) / 2.0) for a, b in zip(self.low_mm, self.high_mm))  # type: ignore[return-value]

    @property
    def dims_mm(self) -> tuple[float, float, float]:
        return tuple(float(b - a) for a, b in zip(self.low_mm, self.high_mm))  # type: ignore[return-value]


def fine_columns(
    points_mm: np.ndarray,
    owners: np.ndarray,
    *,
    floor_mm: float,
    floors: "Sequence[tuple[Callable[[np.ndarray], np.ndarray], float]]" = (),
    cell_mm: float = FINE_CELL_MM,
    margin_mm: float = FINE_MARGIN_MM,
) -> list[FineColumn]:
    """The fine world's boxes of ``points_mm``, ``(N, 3)`` BASE millimetres, each point's owner in ``owners`` (a named
    part's index, or -1 for none).

    The points are laid on a grid of ``cell_mm`` square with BASE whose lines lie on whole multiples of the cell, so the
    same point falls in the same cell in every build. Each cell stands from its floor up to its highest point, the top
    rounded up to a whole step of ``cell_mm`` over the floor; its floor is ``floor_mm``, or the highest of ``floors``
    under it, each who says which ``(M, 2)`` BASE points stand over a held box and that box's top, where the cell's
    lowest point stands over that top (a part resting on a held target stands on the target, not through it). A cell is
    a named part's where every one of its points is that part's. A cell no point fell in, between seen cells that close
    round it, is seen too, as high as the highest of the cells about it (:func:`_closed`): the camera reads a 2 mm cell
    with one or two pixels at the stride the world reads, and an empty cell inside a part's top is no gap a finger fits.
    Cells merge into rectangles row by row, a rectangle growing along its row and then down the rows while every cell it
    takes is seen, not yet taken, of one floor and one owner, and keeps its tops within :data:`_MERGE_STEPS` steps of
    each other; each rectangle is one box, its cells' whole extent grown by ``margin_mm`` on its four sides and over its
    highest top. The same points give the same boxes.
    """
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return []
    owner_of = np.asarray(owners, dtype=np.int64).reshape(-1)
    cell = float(cell_mm)
    ix = np.floor(points[:, 0] / cell).astype(np.int64)
    iy = np.floor(points[:, 1] / cell).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    cols, rows = int(ix.max()) - x0 + 1, int(iy.max()) - y0 + 1
    col, row = ix - x0, iy - y0
    flat = row * cols + col
    top = np.full(rows * cols, -np.inf)
    low = np.full(rows * cols, np.inf)
    np.maximum.at(top, flat, points[:, 2])
    np.minimum.at(low, flat, points[:, 2])
    count = np.bincount(flat, minlength=rows * cols)
    # One owner per cell: the named part :data:`_OWNED_SHARE` of its points lie on, unnamed otherwise (a cell where a
    # part meets what nobody named, or where two parts meet, holds the fingers to the guard's distance).
    owner = np.full(rows * cols, _UNNAMED, dtype=np.int64)
    named = np.nonzero(owner_of >= 0)[0]
    if named.size:
        pairs, votes = np.unique(np.column_stack([flat[named], owner_of[named]]), axis=0, return_counts=True)
        order = np.lexsort((-votes, pairs[:, 0]))
        pairs, votes = pairs[order], votes[order]
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
        best_cell, best_owner, best_votes = pairs[first, 0], pairs[first, 1], votes[first]
        owned = best_votes >= _OWNED_SHARE * count[best_cell]
        owner[best_cell[owned]] = best_owner[owned]
    seen = count > 0
    top_grid, low_grid, owner_grid, seen_grid = _closed(top.reshape(rows, cols), low.reshape(rows, cols),
                                                        owner.reshape(rows, cols), seen.reshape(rows, cols))
    owner_grid = _fringe_owned(top_grid, owner_grid, seen_grid)
    top, low = np.ravel(top_grid), np.ravel(low_grid)
    owner, seen = np.ravel(owner_grid), np.ravel(seen_grid)
    # Each cell's floor: the floor, or the highest top of the boxes given under it that its lowest point stands over.
    floor = np.full(rows * cols, float(floor_mm))
    if floors:
        where = np.nonzero(seen)[0]
        centres = np.column_stack([(x0 + where % cols + 0.5) * cell, (y0 + where // cols + 0.5) * cell])
        for stands_over, roof in floors:
            over = where[np.asarray(stands_over(centres), dtype=bool) & (low[where] >= float(roof))]
            floor[over] = np.maximum(floor[over], float(roof))
    level = np.where(seen, np.ceil(np.maximum(top - floor, 0.0) / cell).astype(np.int64), -1)
    level_at, owner_at, floor_at, count_at = (a.reshape(rows, cols) for a in (level, owner, floor, count))

    out: list[FineColumn] = []
    taken = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        c = 0
        while c < cols:
            if level_at[r, c] < 0 or taken[r, c]:
                c += 1
                continue
            mine, base = int(owner_at[r, c]), float(floor_at[r, c])
            lo = hi = int(level_at[r, c])
            end = c
            while end + 1 < cols and level_at[r, end + 1] >= 0 and not taken[r, end + 1] \
                    and int(owner_at[r, end + 1]) == mine and float(floor_at[r, end + 1]) == base:
                step = int(level_at[r, end + 1])
                if max(hi, step) - min(lo, step) > _MERGE_STEPS:
                    break
                lo, hi, end = min(lo, step), max(hi, step), end + 1
            last = r
            while last + 1 < rows:
                span = slice(c, end + 1)
                below = level_at[last + 1, span]
                if (bool((below < 0).any()) or bool(taken[last + 1, span].any())
                        or bool((owner_at[last + 1, span] != mine).any()) or bool((floor_at[last + 1, span] != base).any())):
                    break
                if max(hi, int(below.max())) - min(lo, int(below.min())) > _MERGE_STEPS:
                    break
                lo, hi, last = min(lo, int(below.min())), max(hi, int(below.max())), last + 1
            taken[r:last + 1, c:end + 1] = True
            out.append(FineColumn(
                low_mm=((x0 + c) * cell - margin_mm, (y0 + r) * cell - margin_mm, base),
                high_mm=((x0 + end + 1) * cell + margin_mm, (y0 + last + 1) * cell + margin_mm,
                         base + hi * cell + margin_mm),
                cells=int((last - r + 1) * (end - c + 1)),
                pixels=int(count_at[r:last + 1, c:end + 1].sum()),
                owner=mine if mine >= 0 else None,
            ))
            c = end + 1
    return out


def _closed(top: np.ndarray, low: np.ndarray, owner: np.ndarray,
            seen: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """The grid with its one-cell holes closed: a cell no point fell in that a 3 x 3 closing of the seen cells takes in
    is seen, as high as the highest cell about it, as low as the lowest, a named part's where every seen cell about it
    is that part's and nobody's else."""
    from scipy import ndimage  # noqa: PLC0415 (only the fine world pays for it)

    square = np.ones((3, 3), dtype=bool)
    padded = np.pad(seen, 1)
    closed = ndimage.binary_closing(padded, structure=square)[1:-1, 1:-1]
    new = closed & ~seen
    if not new.any():
        return top, low, owner, seen
    top, low, owner = top.copy(), low.copy(), owner.copy()
    big = np.iinfo(np.int64).max
    around_top = ndimage.maximum_filter(np.where(seen, top, -np.inf), footprint=square, mode="constant", cval=-np.inf)
    around_low = ndimage.minimum_filter(np.where(seen, low, np.inf), footprint=square, mode="constant", cval=np.inf)
    named = seen & (owner >= 0)
    first = ndimage.minimum_filter(np.where(named, owner, big), footprint=square, mode="constant", cval=big)
    last = ndimage.maximum_filter(np.where(named, owner, -1), footprint=square, mode="constant", cval=-1)
    others = ndimage.maximum_filter(seen & (owner < 0), footprint=square, mode="constant", cval=False)
    top[new], low[new] = around_top[new], around_low[new]
    owner[new] = np.where(others[new] | (first[new] != last[new]), _UNNAMED, last[new])
    return top, low, owner, closed


#: How far past a named part's own cells, in cells, a cell nobody named is the part's fringe: 6 mm, about as wide as the
#: D415 smears a part's edge into the floor behind it, and the side a tilted look sees past the mask the detector drew.
_FRINGE_CELLS: Final[int] = 3


def _fringe_owned(top: np.ndarray, owner: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """``owner`` with a part's fringe given to the part: a seen cell nobody named that lies within
    :data:`_FRINGE_CELLS` of a cell a part owns, the nearest such cell, and stands no higher than that part's highest
    cell. A part's edge read as a ramp, or its side seen past its mask by another look, is then the part's, as the camera
    world's own boxes take it (a box is a part's where most of its points are); what stands higher than the part beside
    it, a wall or a bin's rim, stays nobody's."""
    from scipy import ndimage  # noqa: PLC0415 (only the fine world pays for it)

    named = seen & (owner >= 0)
    if not named.any():
        return owner
    distance, nearest = ndimage.distance_transform_edt(~named, return_indices=True)
    fringe = seen & (owner < 0) & (distance <= _FRINGE_CELLS)
    if not fringe.any():
        return owner
    parts = owner[named]
    highest = np.full(int(parts.max()) + 1, -np.inf)
    np.maximum.at(highest, parts, top[named])
    claim = owner[nearest[0][fringe], nearest[1][fringe]]
    owned = top[fringe] <= highest[claim] + 1e-9
    out = owner.copy()
    rows, cols = np.nonzero(fringe)
    out[rows[owned], cols[owned]] = claim[owned]
    return out


def fringe_of(points_mm: np.ndarray, boxes: Sequence[object], *, reach_mm: float = FINE_TARGET_FRINGE_MM) -> np.ndarray:
    """Which of ``points_mm`` are a held target's fringe: within ``reach_mm`` of one of ``boxes`` (each a held box with
    ``box_to_base_mm`` and ``half_extents_mm``, ``core.keep_out.KeepOutBox``) in the box's own x and y, and no higher
    than its top."""
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(points.shape[0], dtype=bool)
    for box in boxes:
        matrix = np.asarray(getattr(box, "box_to_base_mm"), dtype=np.float64).reshape(4, 4)
        half = np.asarray(getattr(box, "half_extents_mm"), dtype=np.float64)
        local = (points - matrix[:3, 3]) @ matrix[:3, :3]
        beyond = np.maximum(np.abs(local[:, :2]) - half[:2], 0.0)
        out |= (np.hypot(beyond[:, 0], beyond[:, 1]) <= float(reach_mm)) & (local[:, 2] <= half[2])
    return out


def specks(points_mm: np.ndarray, *, joined_mm: float = FINE_SPECK_MM, fewest: int = FINE_SPECK_PIXELS) -> np.ndarray:
    """Which of ``points_mm`` stand in a speck: fewer than ``fewest`` pixels joined in touching ``joined_mm`` cells."""
    from scipy import ndimage  # noqa: PLC0415 (only the fine world pays for it)

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    cells = np.floor(points / float(joined_mm)).astype(np.int64)
    cells -= cells.min(axis=0)
    grid = np.zeros(tuple(int(v) for v in cells.max(axis=0) + 1), dtype=np.int32)
    np.add.at(grid, tuple(cells.T), 1)
    labels, count = ndimage.label(grid > 0, structure=np.ones((3, 3, 3), dtype=bool))
    pixels = ndimage.sum(grid, labels, index=np.arange(1, count + 1))
    small = np.zeros(count + 1, dtype=bool)
    small[1:] = np.asarray(pixels) < int(fewest)
    return small[labels[tuple(cells.T)]]


def noise_clusters(points_mm: np.ndarray, *, cluster_mm: float, min_points: int,
                   voxel_mm: float = FINE_NOISE_VOXEL_MM) -> "tuple[np.ndarray, np.ndarray]":
    """Which of ``points_mm`` stand in a cluster the camera world takes for sensor noise, and each point's cluster:
    points joined in touching ``cluster_mm`` cells, a cluster noise where it fills fewer than ``min_points`` of the
    camera world's own thinning voxels (:data:`FINE_NOISE_VOXEL_MM`), as few as the camera world would keep of it."""
    from scipy import ndimage  # noqa: PLC0415 (only the fine world pays for it)

    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    cells = np.floor(points / float(cluster_mm)).astype(np.int64)
    low = cells.min(axis=0)
    cells -= low
    grid = np.zeros(tuple(int(v) for v in cells.max(axis=0) + 1), dtype=bool)
    grid[tuple(cells.T)] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    cluster = labels[tuple(cells.T)]
    voxels = np.floor(points / float(voxel_mm)).astype(np.int64)
    voxels -= voxels.min(axis=0)
    span = voxels.max(axis=0) + 1
    keys = (voxels[:, 0] * span[1] + voxels[:, 1]) * span[2] + voxels[:, 2]
    distinct = np.unique(np.column_stack([cluster, keys]), axis=0)
    filled = np.bincount(distinct[:, 0], minlength=int(cluster.max()) + 1)
    return filled[cluster] < int(min_points), cluster
