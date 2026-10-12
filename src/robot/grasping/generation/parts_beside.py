"""The parts beside the ones a pick named, found in the depth alone (the owner, 2026-10-12: "Klassisch aus Tiefe").

A prompt grounds the kind of part it names, and every other part beside it was a box nobody named: the fingers kept the
guard's distance and the camera world's margin from it. On the owner's crate a gray cube 30 mm from a green one had no
grasp the guard admitted. Asking the detector for every object did not name them: Qwen3-VL boxed the whole mat, or the
whole group, for "all objects" and its rephrasings on half the cell's frames, and where it listed the parts one by one
it took about 0.6 s a part on the cell, 16 s for a crate of 26 (2026-10-12). So the parts beside are read off the
depth, as the camera world reads its boxes, with no model at all:

* every pixel that no support surface holds (``support_surfaces.SupportModel``: the crate floor, the mat, the bench)
  and that stands over the bench's band stands; a surface holds every pixel its own pixels reach across pixels within
  its band of its plane, so the crate floor between the parts is the floor where the model found a patch of it, and
  so does the ground round each named part, the plane its ring of pixels lies on, where the model found none of it;
  read every :data:`PIXEL_STRIDE` pixels: 63 ms a frame at the median over the cell's 37 named frames of 2026-10-12
  on the dev box, beside the 35 ms its supports take;
* the standing pixels outside the named masks, each grown by :data:`NAMED_RIM_MM` (a part's own rim is no neighbour),
  fall into connected pieces; a piece is a part beside where it is seen whole (it does not meet the image's edge), is
  no piece of something larger (no standing piece too large to be a part comes within :data:`APART_MM` of it),
  stands within :data:`NEAR_A_NAMED_PART_MM` of a named part, where the fingers may meet it, and has a part's shape
  (:func:`why_no_part_beside`).

What it finds is a mask of the frame, each a part's (``PerceptionFrame.parts_beside``): the calculator and the camera
world name their boxes by it, so the fingers may brush them as they brush a named part's (2026-10-05). It finds less,
never more: a part touching a wall, a part in a crate whose floor its neighbours hide, a part at the image's edge stay
boxes nobody named, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import cv2
import numpy as np
from scipy.spatial import cKDTree

__all__ = ["LABEL", "PartBeside", "parts_beside", "why_no_part_beside"]

#: The label a part beside the named ones carries where the camera world names a box by it.
LABEL: Final[str] = "part beside"
#: Every how many pixels of the frame the depth is read.
PIXEL_STRIDE: Final[int] = 4
#: How far round a named mask its own rim reaches, millimetres: a part's mixed edge and its sides are not a neighbour.
NAMED_RIM_MM: Final[float] = 8.0
#: How near a standing piece too large to be a part a piece may come and still be a part of its own, BASE millimetres:
#: a piece of a wall the depth broke off stays the wall's. In space, not over the bench: a part beside a wall comes no
#: nearer the wall's rim high over it than its own height below that.
APART_MM: Final[float] = 10.0
#: How far from the middle of a named part a part beside may stand, BASE millimetres: past it no finger meets it.
NEAR_A_NAMED_PART_MM: Final[float] = 120.0
#: The fewest read pixels a part beside holds.
LEAST_POINTS: Final[int] = 25
#: The widest a part beside is across and the tallest it stands, millimetres, between the 5th and 95th percentile: the
#: owner's parts stand under 80 mm, and a bin of 220 mm read whole with what lies in it is no part. Wider, a piece is
#: something larger, which a part beside keeps apart from (:data:`APART_MM`). A detector's part is held to the world's
#: own, larger bound (``perceived.PART_SIZED_MM``): the depth alone vouches for less.
LARGEST_MM: Final[tuple[float, float]] = (120.0, 100.0)
#: The narrowest a part is across, millimetres, between the 5th and 95th percentile of its footprint: a wall seen from
#: above is a few millimetres of rim.
LEAST_ACROSS_MM: Final[float] = 15.0
#: How many times longer than wide a part may lie: past it, a strip of something.
MOST_ELONGATED: Final[float] = 4.0
#: How much of its own convex footprint a part fills: a wall's corner fills little of its hull.
LEAST_FILLED: Final[float] = 0.45
#: How tall over what it stands on a part is at the least, millimetres.
LEAST_TALL_MM: Final[float] = 10.0
#: The cell a footprint's fill is counted in, millimetres.
_FILL_CELL_MM: Final[float] = 5.0
#: How far over the bench a pixel no support surface covers has to read to stand, millimetres, where the support
#: model gives no band.
_BAND_MM: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class PartBeside:
    """A part beside the named ones: its mask in the frame and the middle of its footprint, BASE x and y millimetres."""

    mask: np.ndarray
    centre_mm: tuple[float, float]


def why_no_part_beside(points_base_mm: "np.ndarray | None", *, bench_mm: "float | None",
                       support_model: Any = None, floor_mm: "float | None" = None) -> str:
    """Why a piece the depth shows beside the named parts is no part, or ``""`` where it is one: part-sized and standing
    where the parts stand (the owner, 2026-10-12).

    Read over its 5th to 95th percentile in BASE: at least :data:`LEAST_POINTS` points, no surface the parts lie on
    (``pick_loop.SURFACE_SPAN_MM`` wide and flatter than ``SURFACE_FLAT_MM``), no larger than a part beside
    (:data:`LARGEST_MM`), its lowest point no higher over the bench than a part may stand (``perceived.PART_SIZED_MM``),
    and a part's shape:
    :data:`LEAST_ACROSS_MM` across at the least, no longer than :data:`MOST_ELONGATED` times that, filling
    :data:`LEAST_FILLED` of its convex footprint, and :data:`LEAST_TALL_MM` over what it stands on.

    Args:
        points_base_mm (np.ndarray | None): Its surface in BASE, ``(N, 3)`` millimetres; ``None`` where none was read.
        bench_mm (float | None): The declared bench's top, BASE z millimetres; ``None`` where the cell declares none.
        support_model (Any): The supports the frame's parts stand on (``support_surfaces.SupportModel``), or ``None``.
        floor_mm (float | None): The ground read round it, BASE z millimetres, or ``None``: then the support under it,
            else the bench, is what it stands on.

    Returns:
        str: Why it is no part beside the named ones, or ``""`` where it is one.
    """
    from src.robot.grasping.loop.pick_loop import SURFACE_FLAT_MM, SURFACE_SPAN_MM  # noqa: PLC0415 (the pick's rule)
    from src.robot.safety.planning.perceived import PART_SIZED_MM  # noqa: PLC0415 (the world's rule)

    if points_base_mm is None or np.asarray(points_base_mm).reshape(-1, 3).shape[0] < LEAST_POINTS:
        return "too few of its points were measured"
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    low, high = np.percentile(points, 5.0, axis=0), np.percentile(points, 95.0, axis=0)
    span = high[:2] - low[:2]
    flat = float(np.percentile(points[:, 2], 90.0) - np.percentile(points[:, 2], 10.0))
    if float(np.linalg.norm(span)) > SURFACE_SPAN_MM and flat < SURFACE_FLAT_MM:
        return "a surface the parts lie on"
    across, tall = LARGEST_MM
    if float(span.max()) > across or float(high[2] - low[2]) > tall:
        return "larger than a part"
    if bench_mm is not None and float(low[2]) - float(bench_mm) > PART_SIZED_MM[1]:
        return "it hovers over the parts"
    xy = points[:, :2] - np.median(points[:, :2], axis=0)
    _, axes = np.linalg.eigh(np.cov(xy.T))
    along = xy @ axes
    extents = np.percentile(along, 95.0, axis=0) - np.percentile(along, 5.0, axis=0)
    narrow, long = float(extents.min()), float(extents.max())
    if narrow < LEAST_ACROSS_MM:
        return "thinner than a part"
    if long > MOST_ELONGATED * narrow:
        return "a strip of something, not a part"
    hull = float(cv2.contourArea(cv2.convexHull(points[:, :2].astype(np.float32))))
    cells = np.unique(np.floor(points[:, :2] / _FILL_CELL_MM).astype(np.int64), axis=0).shape[0]
    if hull > 0.0 and cells * _FILL_CELL_MM * _FILL_CELL_MM < LEAST_FILLED * hull:
        return "hollow, not a part"
    under = (floor_mm if floor_mm is not None
             else support_model.height_under(points) if support_model is not None else None)
    floor = float(under) if under is not None else (float(bench_mm) if bench_mm is not None else None)
    if floor is not None and float(high[2]) - floor < LEAST_TALL_MM:
        return "lower than a part"
    return ""


def parts_beside(
    depth_mm: np.ndarray, intrinsics: np.ndarray, camera_to_base: np.ndarray, named_masks: "list[np.ndarray]", *,
    support_model: Any, bench_mm: "float | None",
) -> "tuple[list[PartBeside], dict[str, int]]":
    """The parts beside ``named_masks`` in one frame, read off its depth (the module docstring), and how many pieces
    were left out, by why.

    Args:
        depth_mm (np.ndarray): The frame's measured depth, ``(H, W)`` millimetres, 0 or not finite where none was read.
        intrinsics (np.ndarray): Its camera's 3x3 matrix.
        camera_to_base (np.ndarray): Its CAMERA to BASE, a 4x4 in millimetres.
        named_masks (list[np.ndarray]): The masks of the parts its detector named, each ``(H, W)``.
        support_model (Any): What the frame's parts stand on (``support_surfaces.SupportModel``), or ``None``: then
            whatever reads over the bench's band stands.
        bench_mm (float | None): The declared bench's top, BASE z millimetres, or ``None``.

    Returns:
        tuple[list[PartBeside], dict[str, int]]: The parts beside, each a mask of the frame, and the pieces left out.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    shape = depth.shape[:2]
    rows, cols = np.mgrid[0:shape[0]:PIXEL_STRIDE, 0:shape[1]:PIXEL_STRIDE]
    z = depth[rows, cols]
    with np.errstate(invalid="ignore"):
        read = np.isfinite(z) & (z > 0.0)
    k = np.asarray(intrinsics, dtype=np.float64)
    matrix = np.asarray(camera_to_base, dtype=np.float64)
    camera = np.stack(((cols - k[0, 2]) * z / k[0, 0], (rows - k[1, 2]) * z / k[1, 1], z), axis=-1)
    base = camera @ matrix[:3, :3].T + matrix[:3, 3]
    named = [np.asarray(m).astype(bool) for m in named_masks if np.asarray(m).shape[:2] == shape]
    if not named or not read.any():
        return [], {}
    rim = max(1, int(round(NAMED_RIM_MM * k[0, 0] / float(np.median(z[read])) / PIXEL_STRIDE)))
    sampled = [mask[::PIXEL_STRIDE, ::PIXEL_STRIDE].astype(np.uint8) for mask in named]
    taken = cv2.dilate(np.bitwise_or.reduce(sampled), np.ones((2 * rim + 1, 2 * rim + 1), np.uint8))
    rings = [read & (cv2.dilate(mask, np.ones((2 * rim + 5, 2 * rim + 5), np.uint8)) > 0) & (taken == 0)
             for mask in sampled]
    standing, held = _standing(base, read, support_model, bench_mm, rings)
    middles = [np.median(base[m[::PIXEL_STRIDE, ::PIXEL_STRIDE] & read][:, :2], axis=0) for m in named
               if bool((m[::PIXEL_STRIDE, ::PIXEL_STRIDE] & read).any())]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((standing & (taken == 0)).astype(np.uint8),
                                                               connectivity=8)
    found: list[PartBeside] = []
    left: dict[str, int] = {}

    def leave(why: str) -> None:
        left[why] = left.get(why, 0) + 1

    pieces, larger = [], np.zeros(read.shape, dtype=bool)
    for piece in (labels == i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 2):
        low, high = np.percentile(base[piece], 5.0, axis=0), np.percentile(base[piece], 95.0, axis=0)
        if float(np.max(high[:2] - low[:2])) > LARGEST_MM[0]:
            larger |= piece
            leave("larger than a part")
        else:
            pieces.append(piece)
    structures = cKDTree(base[larger]) if larger.any() else None
    height, width = read.shape
    for piece in pieces:
        where = np.nonzero(piece)
        if (where[0].min() == 0 or where[1].min() == 0 or where[0].max() == height - 1
                or where[1].max() == width - 1):
            leave("seen only in part, at the image's edge")
            continue
        points = base[piece]
        if structures is not None and bool(np.isfinite(structures.query(points,
                                                                        distance_upper_bound=APART_MM)[0]).any()):
            leave("a piece of something larger")
            continue
        middle = np.median(points[:, :2], axis=0)
        if not any(float(np.linalg.norm(middle - other)) <= NEAR_A_NAMED_PART_MM for other in middles):
            leave("too far from the parts named")
            continue
        # What it stands on: the ground the supports hold round it, where they hold enough of it.
        ground = (cv2.dilate(piece.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & held & ~piece
        floor = float(np.median(base[ground][:, 2])) if int(np.count_nonzero(ground)) >= 5 else None
        why = why_no_part_beside(points, bench_mm=bench_mm, support_model=support_model, floor_mm=floor)
        if why:
            leave(why)
            continue
        mask = np.kron(piece, np.ones((PIXEL_STRIDE, PIXEL_STRIDE), dtype=bool))[:shape[0], :shape[1]]
        found.append(PartBeside(mask=mask, centre_mm=(float(middle[0]), float(middle[1]))))
    return found, left


def _standing(base: np.ndarray, read: np.ndarray, support_model: Any, bench_mm: "float | None",
              rings: "list[np.ndarray]") -> "tuple[np.ndarray, np.ndarray]":
    """Which read pixels of ``base`` (BASE, ``(h, w, 3)``) stand: no support surface holds them, as the calculator's
    obstacles read a frame (``scene_obstacles``), nor any pixel a surface's own pixels reach across its plane, nor the
    ground ``rings`` (each the pixels round a named part) reach across the plane they lie on (the module docstring),
    they are no support's solid, and where no surface covers them they read over the bench by its band; and which the
    supports hold."""
    band = float(getattr(support_model, "bench_band_mm", _BAND_MM) or _BAND_MM)
    floor = (float(bench_mm) if bench_mm is not None else -np.inf) + band
    points = base[read]
    covered = np.zeros(points.shape[0], dtype=bool)
    surface = np.zeros(points.shape[0], dtype=bool)
    if support_model is not None and points.shape[0]:
        planes = support_model.local_plane_under(points[:, :2])
        if planes is not None:
            planes = np.asarray(planes, dtype=np.float64)
            covered = np.isfinite(planes[:, 0])
            reading = np.full(points.shape[0], np.inf)
            nx, ny, nz, d = (planes[covered, i] for i in range(4))
            reading[covered] = (d - nx * points[covered, 0] - ny * points[covered, 1]) / nz
            own = np.asarray(support_model.is_support_point(points), dtype=bool)
            surface = covered & (own | (points[:, 2] <= reading))
        erased = getattr(support_model, "erased_by_support", None)
        if callable(erased):
            surface |= np.asarray(erased(points), dtype=bool).reshape(-1)
    held = np.zeros(read.shape, dtype=bool)
    held[read] = surface
    grounds = [(plane.height_at(points[:, :2]), max(float(plane.band_mm), band), held)
               for plane in getattr(support_model, "surfaces", ()) or ()]
    for ring in rings:
        # The ground a named part stands on: the plane through the lower half of the pixels round it.
        lying = base[ring]
        if lying.shape[0] < LEAST_POINTS:
            continue
        lying = lying[lying[:, 2] <= np.median(lying[:, 2]) + band]
        coef = np.linalg.lstsq(np.column_stack((np.ones(lying.shape[0]), lying[:, :2])), lying[:, 2], rcond=None)[0]
        grounds.append((coef[0] + points[:, :2] @ coef[1:], band, ring))
    for height, within, own in grounds:
        # The plane across the frame: the pixels within its band of it, joined to one of its own pixels on it.
        on = np.zeros(read.shape, dtype=bool)
        on[read] = np.abs(points[:, 2] - height) <= within
        seeds = on & own
        if not seeds.any():
            continue
        _, pieces = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
        reached = np.unique(pieces[seeds])
        held |= np.isin(pieces, reached[reached > 0])
    stands = np.zeros(read.shape, dtype=bool)
    stands[read] = ~held[read] & (covered | (points[:, 2] > floor))
    return stands, held
