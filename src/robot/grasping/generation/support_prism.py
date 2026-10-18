"""A part's footprint as a vertical prism on its support: :func:`reconstruct_support_prism`.

The part's points over the support, its low fragments trimmed off, give a convex footprint in the support's plane,
extruded from the support up to the part's top (:class:`SupportPrism`). The calculator reads the part's top off it for
the points it places in the part's visual hull (``footprint_hull``), and the pick loop the footprint's short side, to
try the parts the hand closes across before those wider than its stroke.

It was the reconstruction the support-footprint enumeration (SFE) planned its grasps on, until the part's own body
became the only grasps the stage offers (2026-10-14); SFE's search left on 2026-10-16 (the owner's cleanup), and the
prism moved here unchanged. The grasps are the body's (``body_grasps``), which plans on what the cameras measured of
the part rather than on its convex hull.

Pure and deterministic: numpy. BASE millimetres throughout, Z up along the support normal.
"""

from __future__ import annotations

from typing import Final

import numpy as np

#: How far above the support a point still counts as the support, millimetres, when the caller does
#: not say. The number SFE's D5 reference measurements were taken with, in a simulator whose
#: calibration is exact. A real cell's table reads as high as its hand-eye and depth error, which every
#: cell's camera world holds in its bench band (``perceived.CAMERA_BENCH_BAND_MM``); the config door of
#: ``Scene`` passes the larger of the two.
DEFAULT_FLOOR_MARGIN_MM: Final[float] = 2.0

#: How close two footprint points have to lie to count as one surface, millimetres, for the fragment
#: trim in :func:`reconstruct_support_prism`. It is the side of a grid cell and two points in touching
#: cells are joined, so points closer than 6 mm are always one surface and pieces more than
#: 6 * 2 * sqrt(2) = 17 mm apart are always two. The 3 mm voxel grid the cloud is thinned to leaves
#: neighbouring samples of one surface at most about 7 mm apart, and the table seen past a 40 mm part's
#: far edge at 45 degrees lies 40 to 46 mm behind it.
_FRAGMENT_LINK_MM: Final[float] = 6.0
#: A footprint whose area fills less of its rectangle than this is round: its closing axes are a fan, and its lines are
#: placed from its centroid (``SupportPrism.middle``). A disc fills pi/4, 0.785; a square 1.
_ROUND_BELOW: Final[float] = 0.88


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


# --------------------------------------------------------------------------- the reconstruction


class SupportPrism:
    """The visible footprint extruded onto the support plane: a convex vertical prism.

    It answers the two questions a grasp needs, where a line enters and leaves the solid and which
    way the surface faces there, in closed form, on an estimate built only from a depth image and
    the calibrated support height.
    """

    __slots__ = ("hull", "normals", "offsets", "z0", "z1", "centre", "u", "v",
                 "ext_u", "ext_v", "roundness", "trimmed", "middle")

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
) -> SupportPrism | None:
    """The support-extruded prism from a masked target cloud in BASE. The whole perception step.

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
    return prism
