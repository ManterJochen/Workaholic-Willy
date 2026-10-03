"""Clear the blocker: which neighbour may be gripped as a blocker, where it goes, and the pose that sets it down there.

The owner's recovery of 2026-10-02: "das ganze Ding soll die Szene so verändern, dass er diesen greifen kann. Also
entweder das Objekt verschieben oder ein anderes Objekt nehmen, was im Weg liegt und dann das eigentliche Objekt
aufheben." Where every grasp of a part meets a neighbour (``ALL_COLLIDED``), or every grasp of it was judged from the
look and refused there, the pick loop takes a neighbour that stands in the way and sets it aside: for critical parts in
place of the push, for the others where no push plans (the owner's switch of 2026-10-03, ``recovery.critical_parts``)
(:meth:`~src.robot.grasping.loop.pick_loop.BinPickingOrchestrator._clear_the_blockers`). This module is what that reads,
and nothing else:

* :func:`clusters_of`: the points the calculator saw beside the part (Track A's obstacle points), grouped into separate
  objects where they stand at least :data:`CLUSTER_VOXEL_MM` apart; a group under :data:`MIN_CLUSTER_POINTS` is a speck.
* :func:`not_a_blocker`: why a group may not be gripped as a blocker, ``""`` where it may. A blocker stands on what the
  part stands on (its foot within :data:`STANDS_ON_THE_SUPPORT_MM` of the support's reading under it, or, where no look
  saw its foot, the support seen round it), apart from the part (not its own unmasked rest), away from the robot's base,
  and no wider across its narrow side than the hand opens.
* :func:`support_seen_round`: whether a group whose foot no look saw stands on the support. A look from almost straight
  above sees a block's sides at a grazing angle and leaves them out (URSim, 2026-10-03). It stands there where the
  support's own pixels fill at least :data:`FOOT_RING_MIN_TABLE_SHARE` of the ring within :data:`FOOT_RING_MM` round its
  footprint; where the camera saw the support under it, it floats (a finger, a cable) and is never gripped.
* :func:`mask_of`: a group's pixels in the frame it was seen in, the blocker's mask for the calculator (the camera
  world's cluster, as the owner allowed).
* :func:`free_spot`: a spot on table the camera saw with nothing seen within the hand's reach and the guard's clearance
  of it (:data:`SPOT_CLEARANCE_MM`), at least :data:`SPOT_FROM_THE_PART_MM` from the part, inside the workspace; the one
  nearest where the blocker stands. ``None`` where there is none.
* :func:`release_pose`: the grasp's own pose moved to the spot, so the blocker comes down as it was taken.

Pure: numpy and SciPy's labelling, BASE millimetres throughout. Nothing here moves the arm, asks a person, reads config
or touches the hand: the pick loop grips the blocker with its own policy, sets it down with the place verb, and the
exact guard judges every sample of every path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Optional, Sequence

import numpy as np

__all__ = [
    "BLOCKER_NOT_REACHED",
    "CLUSTER_VOXEL_MM",
    "FREED_NOTHING",
    "MIN_CLUSTER_POINTS",
    "NO_BLOCKER_FREES_A_GRASP",
    "NO_BLOCKER_SEEN",
    "NO_FREE_SPOT",
    "NO_GRASPABLE_BLOCKER",
    "REFUSED_CANNOT_ASK_AGAIN",
    "REFUSED_NO_ATTEMPT_LEFT",
    "REFUSED_STOP_REQUESTED",
    "SET_ASIDE",
    "SET_DOWN_AIR_MM",
    "SPOT_CLEARANCE_MM",
    "SPOT_FROM_THE_PART_MM",
    "STANDS_ON_THE_SUPPORT_MM",
    "FOOT_RING_MM",
    "FOOT_RING_MIN_TABLE_SHARE",
    "STOPPED_WHERE_THE_ARM_STANDS",
    "TOO_LONG_TO_CARRY",
    "BlockerRecord",
    "Cluster",
    "FreeSpot",
    "clusters_of",
    "free_spot",
    "mask_of",
    "not_a_blocker",
    "support_seen_round",
    "release_pose",
    "taught_place",
]

#: Seen points this far apart, millimetres, belong to separate objects: a voxel of this size, its 26 neighbours joined.
#: Finer than the camera world's 25 mm, so two parts a finger's width apart stay two.
CLUSTER_VOXEL_MM: Final[float] = 8.0
#: A group of fewer seen points is a speck, no object (the calculator's points come every second pixel).
MIN_CLUSTER_POINTS: Final[int] = 20
#: A blocker's foot stands at most this far over the support's reading under it, millimetres: what the part stands on.
#: Further up it rests on something, or is no part at all (a cable, a finger in view).
STANDS_ON_THE_SUPPORT_MM: Final[float] = 15.0
#: A neighbour whose foot no look saw (a look from straight above sees its sides at a grazing angle, and they are left
#: out) stands on the support where the support's own pixels fill at least :data:`FOOT_RING_MIN_TABLE_SHARE` of the
#: cells within this ring round it (the owner, 2026-10-03: "Allgemein ... sofern nicht ein Abgrund"), the push's rule.
FOOT_RING_MM: Final[float] = 15.0
FOOT_RING_MIN_TABLE_SHARE: Final[float] = 0.25
#: The cells the ring is counted in, millimetres.
_RING_CELL_MM: Final[float] = 5.0
#: A group whose points lie within this of the part's own points, millimetres, by this share of them, is the part's own
#: rest the mask missed, never a blocker.
PART_OVERLAP_MM: Final[float] = 4.0
PART_OVERLAP_SHARE: Final[float] = 0.2
#: Narrower than the hand opens by at least this across its narrow side, millimetres, or the jaws do not close round it.
JAW_SLACK_MM: Final[float] = 2.0
#: A free spot is read on cells of this size, millimetres, and its centre tried every this many.
SPOT_CELL_MM: Final[float] = 5.0
SPOT_STEP_MM: Final[float] = 10.0
#: Nothing seen stands within the hand's reach plus this of a free spot, millimetres: the camera world's margin round a
#: box (15) and the guard's distance from it (5), so the place's line in meets nothing the guard holds.
SPOT_CLEARANCE_MM: Final[float] = 20.0
#: A blocker comes down at least this far from the part, millimetres, foot to foot: past the open hand's span and its
#: corridor, so the part's grasps do not meet it again.
SPOT_FROM_THE_PART_MM: Final[float] = 150.0
#: Every point of a spot's blocker stays this far inside the workspace box, millimetres: the push's own margin.
SPOT_WORKSPACE_MARGIN_MM: Final[float] = 20.0
#: The blocker is let go this far over where it stood relative to its support, millimetres: it is set down, never pressed.
SET_DOWN_AIR_MM: Final[float] = 3.0

# --- what clearing a blocker came to (BlockerRecord.code) ---------------------------------------------------------------

#: A blocker was gripped, set down and the arm went back to the look and looked again.
SET_ASIDE = "set_aside"
#: Nothing the calculator saw beside the part is a separate object that may be gripped (``not_a_blocker`` says why).
NO_BLOCKER_SEEN = "no_blocker_seen"
#: No neighbour's removal frees a grasp of the part or spares one of its refusals, asked of the calculator.
NO_BLOCKER_FREES_A_GRASP = "no_blocker_frees_a_grasp"
#: The neighbours whose removal would free the part: the calculator found no grasp on any of them.
NO_GRASPABLE_BLOCKER = "no_graspable_blocker"
#: A graspable blocker, and no free seen spot to set it down on.
NO_FREE_SPOT = "no_free_spot"
#: The graspable blockers reach further past the fingertips than the planner models a carried part
#: (``safety.planning_world.payload.length_mm``): carried, the rest of them would go where nobody judges it.
TOO_LONG_TO_CARRY = "too_long_to_carry"
#: Every graspable blocker's pick was refused before anything was sent.
BLOCKER_NOT_REACHED = "blocker_not_reached"
#: The last removal freed no grasp of the part and spared none of its refusals: the clearing stops (no budget, a rule).
FREED_NOTHING = "freed_nothing"
#: No attempt is left to pick the part with after a blocker is set aside.
REFUSED_NO_ATTEMPT_LEFT = "refused_no_attempt_left"
#: A stop was asked for before anything of the clearing moved.
REFUSED_STOP_REQUESTED = "refused_stop_requested"
#: The calculator cannot be asked again with a neighbour left out (it was not handed a frame, or keeps no call).
REFUSED_CANNOT_ASK_AGAIN = "refused_cannot_ask_again"
#: The clearing stopped once something moved: a person decides (the blocker may be in the hand).
STOPPED_WHERE_THE_ARM_STANDS = "stopped_where_the_arm_stands"


@dataclass(frozen=True, slots=True, eq=False)
class Cluster:
    """One separate object among the points seen beside the part, BASE millimetres."""

    points_base_mm: np.ndarray

    @property
    def centre_mm(self) -> np.ndarray:
        """The median of its points."""
        return np.median(self.points_base_mm, axis=0)

    @property
    def low_mm(self) -> float:
        """Where it rests: the 2nd percentile of its heights, so a stray low point does not decide it."""
        return float(np.percentile(self.points_base_mm[:, 2], 2.0))

    @property
    def top_mm(self) -> float:
        return float(np.percentile(self.points_base_mm[:, 2], 98.0))

    @property
    def footprint_radius_mm(self) -> float:
        """How far its points reach from its centre in BASE x and y."""
        xy = self.points_base_mm[:, :2] - self.centre_mm[:2]
        return float(np.max(np.hypot(xy[:, 0], xy[:, 1]))) if len(xy) else 0.0

    @property
    def widths_mm(self) -> tuple[float, float]:
        """Its extent in BASE x and y along the sides of the smallest rectangle about it (the camera world's own fit,
        ``height_map.turn_of``), narrow first."""
        from src.robot.safety.planning.height_map import turn_of  # noqa: PLC0415

        xy = self.points_base_mm[:, :2]
        if len(xy) < 2:
            return 0.0, 0.0
        yaw = turn_of(xy)
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)
        local = xy @ np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]).T
        spans = np.ptp(local, axis=0)
        return float(spans.min()), float(spans.max())

    def to_dict(self) -> dict[str, Any]:
        narrow, wide = self.widths_mm
        return {"centre_mm": [round(float(v), 1) for v in self.centre_mm], "points": int(len(self.points_base_mm)),
                "low_mm": round(self.low_mm, 1), "top_mm": round(self.top_mm, 1),
                "widths_mm": [round(narrow, 1), round(wide, 1)]}


@dataclass(frozen=True, slots=True)
class FreeSpot:
    """Where a blocker comes down: the centre of its foot there, BASE x and y, how far its foot stands from the part's,
    and how far its centre travels from where it stood, millimetres."""

    centre_xy: tuple[float, float]
    distance_to_part_mm: float
    travel_mm: float
    radius_mm: float

    def to_dict(self) -> dict[str, Any]:
        return {"centre_xy": [round(v, 1) for v in self.centre_xy],
                "distance_to_part_mm": round(self.distance_to_part_mm, 1), "travel_mm": round(self.travel_mm, 1),
                "radius_mm": round(self.radius_mm, 1)}


@dataclass(frozen=True, slots=True)
class BlockerRecord:
    """What one clearing of a blocker came to, as the pick loop keeps it (``BinPickingOrchestrator.blockers``).

    ``code`` is one of this module's codes and ``sentence`` says it in words. ``centre_mm`` and ``widths_mm`` say which
    neighbour, ``freed`` how many grasps of the part its removal was asked to free and ``fewer_refusals`` how many of the
    part's refusals by what the camera saw it was asked to spare. ``place`` is where it went: ``free spot (x, y)`` or the
    owner's named place. ``moved`` is whether the arm left the look for it.
    """

    code: str
    sentence: str
    centre_mm: Optional[tuple[float, float, float]] = None
    widths_mm: Optional[tuple[float, float]] = None
    freed: int = 0
    fewer_refusals: int = 0
    place: str = ""
    moved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "sentence": self.sentence,
                "centre_mm": None if self.centre_mm is None else [round(v, 1) for v in self.centre_mm],
                "widths_mm": None if self.widths_mm is None else [round(v, 1) for v in self.widths_mm],
                "freed": int(self.freed), "fewer_refusals": int(self.fewer_refusals), "place": self.place,
                "moved": bool(self.moved)}


def clusters_of(
    points_base_mm: Any, *, voxel_mm: float = CLUSTER_VOXEL_MM, min_points: int = MIN_CLUSTER_POINTS,
) -> tuple[Cluster, ...]:
    """The separate objects among ``points_base_mm``, the largest first: points whose ``voxel_mm`` voxels touch at a
    face, an edge or a corner are one object, and one of fewer than ``min_points`` points is a speck."""
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < int(min_points):
        return ()
    from scipy import ndimage  # noqa: PLC0415 (only a clearing pays for it)

    keys = np.floor(points / float(voxel_mm)).astype(np.int64)
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    low = cells.min(axis=0)
    grid = np.zeros(tuple(int(v) for v in cells.max(axis=0) - low + 1), dtype=bool)
    grid[tuple((cells - low).T)] = True
    labels, _count = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    of_point = labels[tuple((cells - low).T)][inverse]
    groups = [points[of_point == label] for label in np.unique(of_point)]
    kept = sorted((group for group in groups if group.shape[0] >= int(min_points)), key=lambda g: -g.shape[0])
    return tuple(Cluster(points_base_mm=group) for group in kept)


def support_seen_round(footprint_xy: Any, table_xy: Any, *, ring_mm: float = FOOT_RING_MM,
                       min_share: float = FOOT_RING_MIN_TABLE_SHARE) -> bool:
    """Whether the support's own pixels (``table_xy``, BASE x and y) fill at least ``min_share`` of the
    :data:`_RING_CELL_MM` cells within ``ring_mm`` round a neighbour's footprint (``footprint_xy``), the footprint's own
    cells left out, and none of the cells inside its footprint: where its foot was not seen, it stands on the support,
    unless the camera saw the support under it, where it stands over the support on something, or floats (a finger, a
    cable)."""
    foot = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    table = np.asarray(table_xy, dtype=np.float64).reshape(-1, 2)
    if not len(foot) or not len(table):
        return False
    cells = {tuple(c) for c in np.floor(foot / _RING_CELL_MM).astype(np.int64)}
    seen = {tuple(c) for c in np.floor(table / _RING_CELL_MM).astype(np.int64)}
    inside = {(i, j) for i, j in cells
              if all((i + di, j + dj) in cells for di in (-1, 0, 1) for dj in (-1, 0, 1))}
    if len(inside & seen) > max(2, int(0.1 * len(inside))):
        return False
    steps = int(np.ceil(float(ring_mm) / _RING_CELL_MM))
    ring = {(i + di, j + dj) for i, j in cells for di in range(-steps, steps + 1) for dj in range(-steps, steps + 1)}
    ring -= cells
    return bool(ring) and len(ring & seen) / len(ring) >= float(min_share)


def not_a_blocker(
    cluster: Cluster,
    *,
    support_mm: Optional[float],
    target_points_mm: Any,
    base_radius_mm: Optional[float],
    open_width_mm: float,
    foot_seen_round: bool = False,
) -> str:
    """Why ``cluster`` may not be gripped as a blocker, in words, or ``""`` where it may.

    ``support_mm`` is the support's reading under it (BASE z), ``None`` where none was read; ``target_points_mm`` the
    part's own points; ``base_radius_mm`` how far from the BASE axis the robot's base and its padding reach, ``None``
    where its shape is not known; ``open_width_mm`` how far the hand opens. ``foot_seen_round`` says the support was
    seen round it (:func:`support_seen_round`): a foot no look saw is then taken to stand on it.
    """
    if support_mm is None:
        return "no support was read under it, so nothing says it stands on what the part stands on"
    over = cluster.low_mm - float(support_mm)
    if over > STANDS_ON_THE_SUPPORT_MM and not foot_seen_round:
        return (f"its foot stands {over:.0f} mm over the support under it, more than {STANDS_ON_THE_SUPPORT_MM:g} mm: it "
                "does not stand on what the part stands on (it rests on something, or is no part)")
    centre = cluster.centre_mm
    if base_radius_mm is not None and float(np.hypot(centre[0], centre[1])) <= float(base_radius_mm):
        return "it stands at the robot's base"
    target = np.asarray(target_points_mm, dtype=np.float64).reshape(-1, 3)
    if target.shape[0]:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        near, _ = cKDTree(target).query(cluster.points_base_mm, k=1, distance_upper_bound=PART_OVERLAP_MM)
        share = float(np.mean(np.isfinite(near)))
        if share >= PART_OVERLAP_SHARE:
            return (f"{100.0 * share:.0f}% of its points lie within {PART_OVERLAP_MM:g} mm of the part's own: it is the "
                    "part's own rest the mask missed")
    narrow, _wide = cluster.widths_mm
    if narrow > float(open_width_mm) - JAW_SLACK_MM:
        return (f"it is {narrow:.0f} mm wide across its narrow side and the hand opens {float(open_width_mm):.0f} mm: "
                "too wide to grip as one part")
    return ""


def mask_of(points_base_mm: Any, *, shape: tuple[int, int], intrinsics: Any, camera_to_base: Any,
            grow_px: int = 2) -> np.ndarray:
    """The pixels of a frame of ``shape`` that ``points_base_mm`` were seen in, grown by ``grow_px``: the mask a group of
    the frame's own points stands for, placed by the frame's ``intrinsics`` and ``camera_to_base``."""
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    mask = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    if points.shape[0] == 0:
        return mask
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    to_camera = np.linalg.inv(np.asarray(camera_to_base, dtype=np.float64).reshape(4, 4))
    camera = points @ to_camera[:3, :3].T + to_camera[:3, 3]
    ahead = camera[:, 2] > 1e-6
    camera = camera[ahead]
    u = np.rint(k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2]).astype(np.int64)
    v = np.rint(k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    mask[v[inside], u[inside]] = True
    if grow_px > 0 and mask.any():
        from scipy import ndimage  # noqa: PLC0415

        yy, xx = np.mgrid[-grow_px:grow_px + 1, -grow_px:grow_px + 1]
        mask = ndimage.binary_dilation(mask, structure=(xx * xx + yy * yy) <= grow_px * grow_px)
    return mask


def free_spot(
    *,
    table_points_mm: Any,
    occupied_points_mm: Any,
    blocker_points_mm: Any,
    target_points_mm: Any,
    workspace_xy: tuple[tuple[float, float], tuple[float, float]],
    hand_reach_mm: float,
    clearance_mm: float = SPOT_CLEARANCE_MM,
    from_part_mm: float = SPOT_FROM_THE_PART_MM,
    cell_mm: float = SPOT_CELL_MM,
    step_mm: float = SPOT_STEP_MM,
    centre_xy: Optional[Sequence[float]] = None,
) -> Optional[FreeSpot]:
    """Where the blocker may come down, or ``None`` where it may nowhere.

    ``centre_xy`` is the point of the blocker that comes down on the spot's centre (its grasp's, BASE x and y), the median
    of its points where none is given. Its foot reaches ``r`` from there; the hand reaches ``hand_reach_mm``. A spot is a
    centre with a disc
    of ``max(r, hand_reach_mm) + clearance_mm`` round it that lies wholly on cells of table the camera saw
    (``table_points_mm``) and holds no cell of anything seen (``occupied_points_mm``: everything standing over the
    support, the part and the blocker where it stands included), whose blocker's foot stays ``from_part_mm`` from the
    part's points (``target_points_mm``) and :data:`SPOT_WORKSPACE_MARGIN_MM` inside ``workspace_xy``. The squares the
    discs are tested on hold the discs, so a spot is free by more than the disc. The spot nearest where the blocker stands
    is taken, the one farther from the part on a tie.
    """
    table = np.asarray(table_points_mm, dtype=np.float64).reshape(-1, 3)[:, :2]
    occupied = np.asarray(occupied_points_mm, dtype=np.float64).reshape(-1, 3)[:, :2]
    blocker = np.asarray(blocker_points_mm, dtype=np.float64).reshape(-1, 3)[:, :2]
    target = np.asarray(target_points_mm, dtype=np.float64).reshape(-1, 3)[:, :2]
    if table.shape[0] == 0 or blocker.shape[0] == 0:
        return None
    centre = np.median(blocker, axis=0) if centre_xy is None else np.asarray(centre_xy, dtype=np.float64).reshape(2)
    radius = float(np.max(np.hypot(*(blocker - centre).T)))
    reach = max(radius, float(hand_reach_mm)) + float(clearance_mm)
    cell = float(cell_mm)
    origin = np.floor((table.min(axis=0) - reach) / cell) * cell
    size = np.ceil((table.max(axis=0) + reach - origin) / cell).astype(np.int64) + 1
    seen = _cells(table, origin, cell, size)
    taken = _cells(occupied, origin, cell, size)
    seen_sum, taken_sum = _summed(seen), _summed(taken)
    (wx0, wy0), (wx1, wy1) = workspace_xy
    inset = SPOT_WORKSPACE_MARGIN_MM + radius
    lo = np.maximum(table.min(axis=0), (wx0 + inset, wy0 + inset))
    hi = np.minimum(table.max(axis=0), (wx1 - inset, wy1 - inset))
    if np.any(hi < lo):
        return None
    xs = np.arange(lo[0], hi[0] + 1e-9, float(step_mm))
    ys = np.arange(lo[1], hi[1] + 1e-9, float(step_mm))
    if xs.size == 0 or ys.size == 0:
        return None
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    spots = np.column_stack([gx.ravel(), gy.ravel()])
    i0 = np.floor((spots - reach - origin) / cell).astype(np.int64)
    i1 = np.floor((spots + reach - origin) / cell).astype(np.int64)
    inside = np.all(i0 >= 0, axis=1) & (i1[:, 0] < size[0]) & (i1[:, 1] < size[1])
    spots, i0, i1 = spots[inside], i0[inside], i1[inside]
    if spots.shape[0] == 0:
        return None
    windows = (i1[:, 0] - i0[:, 0] + 1) * (i1[:, 1] - i0[:, 1] + 1)
    all_seen = _window(seen_sum, i0, i1) == windows
    nothing_there = _window(taken_sum, i0, i1) == 0
    keep = all_seen & nothing_there
    spots = spots[keep]
    if spots.shape[0] == 0:
        return None
    if target.shape[0]:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        to_part, _ = cKDTree(target).query(spots, k=1)
        to_part = np.asarray(to_part, dtype=np.float64) - radius
    else:
        to_part = np.full(spots.shape[0], math.inf)
    far = to_part >= float(from_part_mm)
    spots, to_part = spots[far], to_part[far]
    if spots.shape[0] == 0:
        return None
    travel = np.hypot(spots[:, 0] - centre[0], spots[:, 1] - centre[1])
    best = int(np.lexsort((-to_part, np.round(travel, 3)))[0])
    return FreeSpot(centre_xy=(float(spots[best, 0]), float(spots[best, 1])),
                    distance_to_part_mm=float(to_part[best]), travel_mm=float(travel[best]), radius_mm=radius)


def _cells(points_xy: np.ndarray, origin: np.ndarray, cell: float, size: np.ndarray) -> np.ndarray:
    grid = np.zeros((int(size[0]), int(size[1])), dtype=np.int64)
    if points_xy.shape[0]:
        index = np.floor((points_xy - origin) / cell).astype(np.int64)
        inside = np.all(index >= 0, axis=1) & (index[:, 0] < size[0]) & (index[:, 1] < size[1])
        grid[index[inside, 0], index[inside, 1]] = 1
    return grid


def _summed(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=np.int64)
    out[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
    return out


def _window(summed: np.ndarray, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    return (summed[i1[:, 0] + 1, i1[:, 1] + 1] - summed[i0[:, 0], i1[:, 1] + 1]
            - summed[i1[:, 0] + 1, i0[:, 1]] + summed[i0[:, 0], i0[:, 1]])


def release_pose(grasp: Any, *, shift_mm: Sequence[float]) -> Any:
    """The TCP pose that sets the blocker down: the grasp's own pose, turned as the policy turns it (tool x the closing
    axis, z the approach), moved by ``shift_mm`` in BASE."""
    from src.geometry import Frame, Pose  # noqa: PLC0415
    from src.geometry.quaternion import from_rotation_matrix  # noqa: PLC0415

    closing = np.asarray(grasp.axis, dtype=np.float64).reshape(3)
    approach = np.asarray(grasp.approach, dtype=np.float64).reshape(3)
    closing = closing / float(np.linalg.norm(closing))
    approach = approach - float(approach @ closing) * closing
    approach = approach / float(np.linalg.norm(approach))
    binormal = np.cross(approach, closing)
    rotation = np.column_stack([closing, binormal / float(np.linalg.norm(binormal)), approach])
    position = np.asarray(grasp.position, dtype=np.float64).reshape(3) + np.asarray(shift_mm, dtype=np.float64).reshape(3)
    return Pose(position_mm=position, quaternion_xyzw=np.asarray(from_rotation_matrix(rotation), dtype=np.float64),
                frame=Frame.BASE, label="blocker_set_down")


def taught_place(name: str, path: Any = None) -> Any:
    """The TCP pose a person taught under ``name`` (``src.robot.execution.teach``), the place a blocker may go to:
    ``orchestrator.blocker_place = taught_place("Müll")``. ``KeyError`` names the poses the file holds where none has
    that name."""
    from src.robot.execution.teach import DEFAULT_STORE, read_taught_poses  # noqa: PLC0415

    poses = read_taught_poses(DEFAULT_STORE if path is None else path)
    found = [pose for pose in poses if pose.name == name]
    if not found:
        raise KeyError(f"no pose taught as {name!r}; the file holds {[pose.name for pose in poses]}")
    return found[-1].tcp
