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

Pure and deterministic: numpy only. BASE millimetres throughout, Z up along the support normal.
"""

from __future__ import annotations

from typing import Final

from dataclasses import dataclass

import numpy as np

from src.robot.grasping.collision import ParallelJawGripperModel

__all__ = [
    "SupportFootprintCandidate",
    "SupportFootprintJaw",
    "SupportPrism",
    "generate_support_footprint_grasps",
    "reconstruct_support_prism",
]

_EPS = 1e-9

#: Approach tilts away from straight-down, in preference order: the first that yields a candidate for
#: an (axis, anchor) wins, so within one anchor a reachable vertical grasp is never traded for a
#: tilted one. Across anchors the final ``found.sort`` by blended score decides, and ``upright`` is
#: only 0.15 of that blend, so a tilted candidate from another anchor can still rank first.
_TILTS_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
#: Where along the perpendicular extent to place the closing line: the middle and both thirds.
_FRACS = (0.0, -0.35, 0.35)


@dataclass(frozen=True, slots=True)
class SupportFootprintJaw:
    """The gripper, as SFE needs to reason about it. Build it with :meth:`from_model`.

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

    @property
    def finger_reach_mm(self) -> float:
        return self.finger_ahead_mm + self.finger_behind_mm

    @property
    def cone_rad(self) -> float:
        return float(np.arctan(self.friction))


@dataclass(frozen=True, slots=True)
class SupportFootprintCandidate:
    """One enumerated grasp, in BASE millimetres."""

    score: float
    position_mm: np.ndarray
    approach: np.ndarray
    closing_axis: np.ndarray
    grip_width_mm: float
    #: Worst contact-normal deviation from the closing axis, radians. Inside the friction cone.
    contact_angle_rad: float
    #: Lowest point of the closed gripper above the support, millimetres.
    clearance_mm: float


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


def _poly_area(hull: np.ndarray) -> float:
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / max(float(np.linalg.norm(axis)), _EPS)
    return (vector * np.cos(angle) + np.cross(k, vector) * np.sin(angle)
            + k * (k @ vector) * (1.0 - np.cos(angle)))


# --------------------------------------------------------------------------- the reconstruction


class SupportPrism:
    """The visible footprint extruded onto the support plane: a convex vertical prism.

    It answers the two questions a grasp needs, where a line enters and leaves the solid and which
    way the surface faces there, in closed form, on an estimate built only from a depth image and
    the calibrated support height.
    """

    __slots__ = ("hull", "normals", "offsets", "z0", "z1", "centre", "u", "v",
                 "ext_u", "ext_v", "roundness", "inflate")

    def __init__(self, hull2d: np.ndarray, z0: float, z1: float) -> None:
        self.hull = hull2d
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


def reconstruct_support_prism(
    cloud_base_mm: np.ndarray,
    support_height_mm: float,
    *,
    voxel_mm: float = 3.0,
    top_percentile: float = 98.0,
    floor_margin_mm: float = 2.0,
    inflate_mm: float = 0.0,
) -> SupportPrism | None:
    """The support-extruded prism from a masked target cloud in BASE. The whole perception step.

    ``inflate_mm`` pushes every face outward. A depth image loses the grazing rim of a curved
    surface, so the measured footprint is systematically smaller than the object and every
    consequence of that error is one-sided (an under-estimated span, a finger that clips a flank on
    the way in). Inflating biases the estimate in the safe direction instead of leaving it unbiased.
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
    top_z = float(np.percentile(p[:, 2], top_percentile))
    if top_z <= support_height_mm + 3.0:
        return None
    hull = _hull2d(p[:, :2])
    if hull.shape[0] < 3:
        return None
    prism = SupportPrism(hull, support_height_mm, top_z)
    if inflate_mm:
        prism.offsets = prism.offsets + float(inflate_mm)
        prism.inflate = float(inflate_mm)
    return prism


def closing_axes(prism: SupportPrism, *, radial: int = 6) -> list[np.ndarray]:
    """The closing directions the reconstruction admits, enumerated rather than searched.

    A vertical prism's antipodal side-face pairs are the min-area rectangle's two axes; a round
    footprint admits every radial direction, so it gets a fan of ``radial`` directions on top of
    those two.
    """
    axes = [np.array([prism.u[0], prism.u[1], 0.0]), np.array([prism.v[0], prism.v[1], 0.0])]
    if prism.roundness < 0.88:
        for k in range(radial):
            angle = np.pi * k / radial
            d = np.cos(angle) * prism.u + np.sin(angle) * prism.v
            axes.append(np.array([d[0], d[1], 0.0]))
    return axes


# --------------------------------------------------------------------------- obstacles


class _Obstacles:
    """Occupancy grid over the non-target points, dilated once by one 12 mm cell.

    Built once per object; a query is one raveled ``searchsorted``. The calculator already builds
    this same cloud today and only hands it to a post-hoc filter.
    """

    __slots__ = ("cell", "keys", "origin", "dims")

    def __init__(self, points: np.ndarray | None,
                 cell_mm: float = 12.0, margin_mm: float = 12.0) -> None:
        self.cell = cell_mm
        self.keys: np.ndarray | None = None
        if points is None or np.size(points) == 0:
            return
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        p = p[np.isfinite(p).all(axis=1)]
        if p.shape[0] == 0:
            return
        r = int(np.ceil(margin_mm / cell_mm))
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
    """

    __slots__ = ("sparse", "rigid")

    def __init__(self, sparse: _Obstacles, rigid: _Obstacles) -> None:
        self.sparse = sparse
        self.rigid = rigid

    def hits(self, points: np.ndarray) -> bool:
        return self.sparse.hits(points) or self.rigid.hits(points)


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


def _palm_low_mm(anchor: np.ndarray, approach: np.ndarray, binormal: np.ndarray,
                 jaw: SupportFootprintJaw) -> float:
    """Lowest point of the palm box, anchored at the grasp centre exactly as the envelope anchors it.

    Mirrors ``ParallelJawGripperModel.collision_boxes``' palm: it spans ``-finger_length`` to
    ``-(finger_length + palm_depth)`` along the approach and ``+/- palm_width/2`` along the binormal.
    The closing extent is omitted because SFE's closing axis is horizontal by construction
    (``axis[2] == 0``), so it contributes nothing to a height.
    """
    back = float(approach[2])
    return (float(anchor[2])
            + min(back * -jaw.finger_behind_mm, back * -(jaw.finger_behind_mm + jaw.palm_depth_mm))
            - (jaw.palm_width_mm / 2.0) * abs(float(binormal[2])))


def _build(prism: SupportPrism, anchor: np.ndarray, axis: np.ndarray, approach: np.ndarray,
           jaw: SupportFootprintJaw, obstacles: "_ObstacleSet",
           support_height_mm: float, *, palm_aware: bool = False,
           score_weights: tuple[float, float, float, float, float] | None = None,
           ) -> SupportFootprintCandidate | None:
    binormal = np.cross(approach, axis)
    binormal /= max(float(np.linalg.norm(binormal)), _EPS)
    contacts = _pad_contacts(prism, anchor, axis, approach, binormal, jaw)
    if contacts is None:
        return None
    t_enter, t_exit, contact_b, contact_a, low_pad_z = contacts
    span = t_exit - t_enter
    if not (t_enter - 2.0 <= 0.0 <= t_exit + 2.0):
        return None
    if span > jaw.aperture_mm - jaw.width_safety_mm or span < jaw.min_width_mm:
        return None

    cone = jaw.cone_rad
    worst = max(float(np.arccos(np.clip(prism.normal_at(contact_a) @ axis, -1.0, 1.0))),
                float(np.arccos(np.clip(prism.normal_at(contact_b) @ -axis, -1.0, 1.0))))
    if worst > cone:
        return None

    half_thickness = jaw.finger_thickness_mm / 2.0
    closed = np.concatenate([
        _finger_points(contact_a + half_thickness * axis, approach, binormal, jaw),
        _finger_points(contact_b - half_thickness * axis, approach, binormal, jaw)])
    # Deliberately the worst case: the pad may touch anywhere on its face, so the fingertip can end
    # up a full ``finger_ahead`` below the lowest pad point that reaches the object, and the finger's
    # own width tips further down when the approach is tilted.
    low = min(float(closed[:, 2].min()),
              low_pad_z - jaw.finger_ahead_mm * max(0.0, -float(approach[2]))
              - (jaw.finger_width_mm / 2.0) * abs(float(binormal[2])))
    if palm_aware:
        # Taken as a minimum against the finger reasoning above, never as a replacement for it: the
        # finger term is deliberately worst-case (the pad may touch anywhere on its face) and is the
        # stricter of the two on a straight-down grasp. Adding the palm can only lower ``low``, so
        # this can refuse a candidate and can never admit one that was refused before.
        low = min(low, _palm_low_mm(anchor, approach, binormal, jaw))
    if low < support_height_mm + jaw.table_clearance_mm:
        return None
    if obstacles.hits(closed):
        return None
    if prism.contains(closed, margin_mm=-1.0 - prism.inflate).any():
        return None

    # A real jaw arrives open and closes at the end, so the corridor is checked at the aperture, not
    # at the final width. That also removes the reconstruction's one-sided thinness from the
    # question: fingers held wider than the object cannot clip a flank the depth image missed.
    open_half = jaw.aperture_mm / 2.0
    reach = jaw.finger_reach_mm + 80.0
    swept = np.concatenate([
        _finger_points(anchor + open_half * axis, approach, binormal, jaw, reach, 8.0),
        _finger_points(anchor - open_half * axis, approach, binormal, jaw, reach, 8.0)])
    swept = swept[swept[:, 2] >= support_height_mm]
    if obstacles.hits(swept):
        return None
    if prism.contains(swept, margin_mm=-1.0 - prism.inflate).any():
        return None

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
            return None

    # Rank on measured margins only. The weights and their measurement are at ``_SCORE_WEIGHTS``.
    cone_slack = 1.0 - worst / cone                                 # depth inside the friction cone
    width_margin = 1.0 - span / jaw.aperture_mm                     # aperture left over
    table_margin = min(1.0, (low - support_height_mm) / 20.0)       # fingertip off the table
    upright = max(0.0, float(-approach @ np.array([0.0, 0.0, 1.0])))
    centred = 1.0 - min(1.0, abs(t_enter + t_exit) / max(span, 1.0))
    w = _SCORE_WEIGHTS if score_weights is None else score_weights
    score = (w[0] * cone_slack + w[1] * width_margin + w[2] * table_margin
             + w[3] * upright + w[4] * centred)
    return SupportFootprintCandidate(
        score=float(score), position_mm=anchor.copy(), approach=approach.copy(),
        closing_axis=axis.copy(), grip_width_mm=float(span + 2.0),
        contact_angle_rad=float(worst), clearance_mm=float(low - support_height_mm))


def generate_support_footprint_grasps(
    cloud_base_mm: np.ndarray,
    *,
    support_height_mm: float = 0.0,
    jaw: SupportFootprintJaw | None = None,
    obstacle_points_base_mm: np.ndarray | None = None,
    rigid_obstacle_points_base_mm: np.ndarray | None = None,
    max_candidates: int = 12,
    height_offsets_mm: tuple[float, ...] = (4.0, 14.0),
    inflate_mm: float = 0.0,
    palm_aware: bool = False,
    score_weights: tuple[float, float, float, float, float] | None = None,
) -> list[SupportFootprintCandidate]:
    """Ranked candidates in BASE, from a masked target cloud and the rest of the scene as obstacles.

    ``obstacle_points_base_mm`` is observed geometry, the neighbours' depth points, and is dilated
    to cover the gaps between samples. ``rigid_obstacle_points_base_mm`` is declared geometry, today
    the container walls, and is not: it is already exact, and dilating it would forbid a band of the
    bin that a gripper can legally reach. See :class:`_ObstacleSet`.

    Returns an empty list when the cloud is too sparse to reconstruct or admits no legal grasp,
    which is a real answer and not a failure. Refusing beats proposing a grasp that goes under the
    support surface.
    """
    jaw = jaw or SupportFootprintJaw.from_model()
    prism = reconstruct_support_prism(cloud_base_mm, support_height_mm, inflate_mm=inflate_mm)
    if prism is None:
        return []
    obstacles = _ObstacleSet(
        _Obstacles(obstacle_points_base_mm),
        _Obstacles(rigid_obstacle_points_base_mm, cell_mm=4.0, margin_mm=0.0),
    )
    found: list[SupportFootprintCandidate] = []

    for raw_axis in closing_axes(prism):
        a2 = raw_axis[:2] / max(float(np.linalg.norm(raw_axis[:2])), _EPS)
        axis = np.array([a2[0], a2[1], 0.0])
        perp = np.array([-a2[1], a2[0]])
        ext_perp = float((prism.hull @ perp).max() - (prism.hull @ perp).min())
        for frac in _FRACS:
            base_xy = prism.centre + frac * (ext_perp / 2.0) * perp
            mid_z = (prism.z0 + prism.z1) / 2.0
            span_xy = prism.line_span(np.array([base_xy[0], base_xy[1], mid_z]), axis)
            if span_xy is None:
                continue
            mid = base_xy + ((span_xy[0] + span_xy[1]) / 2.0) * a2
            heights = [prism.z1 - dz for dz in height_offsets_mm]
            # Solve for the height at which the gripper still clears the support, per tilt. This is
            # the step the silhouette generator has no equivalent of: it anchors first and is
            # filtered afterwards, and a filter cannot move a grasp somewhere legal.
            for tilt in _TILTS_DEG:
                c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
                # The solve has to know what the check knows: teaching ``_build`` about the palm
                # without teaching this would only make SFE lose candidates, and the stage exists
                # because it can move a grasp somewhere legal where a filter can only refuse. The
                # palm's own requirement is (palm_width/2)*sin - finger_behind*cos: it is far below
                # the finger term when the approach is vertical (the palm sits behind the fingers)
                # and overtakes it as the grasp tilts. At 90 deg the two are 35.0 and 27.0 mm, the
                # 8 mm the envelope rejects candidates by.
                need_mm = 2.0 * jaw.finger_ahead_mm * c + jaw.finger_width_mm * s
                if palm_aware:
                    need_mm = max(need_mm, (jaw.palm_width_mm / 2.0) * s - jaw.finger_behind_mm * c)
                need = support_height_mm + jaw.table_clearance_mm + need_mm
                if support_height_mm + 2.0 < need <= prism.z1 - 2.0:
                    heights.append(need + 0.5)
            # A height is only worth trying once; the solve often lands on the same millimetre.
            heights = sorted({round(z, 1) for z in heights
                              if support_height_mm + 2.0 < z < prism.z1}, reverse=True)[:4]
            for z in heights:
                anchor = np.array([mid[0], mid[1], z])
                for tilt in _TILTS_DEG:
                    signs = (1.0,) if tilt == 0.0 else (1.0, -1.0)
                    hit = False
                    for sign in signs:
                        approach = _rodrigues(np.array([0.0, 0.0, -1.0]), axis,
                                              sign * np.radians(tilt))
                        approach /= float(np.linalg.norm(approach))
                        candidate = _build(prism, anchor, axis, approach, jaw, obstacles,
                                           support_height_mm, palm_aware=palm_aware,
                                           score_weights=score_weights)
                        if candidate is not None:
                            found.append(candidate)
                            hit = True
                    if hit:
                        break   # tilts are in preference order: the first that works is the one

    found.sort(key=lambda c: -c.score)
    kept: list[SupportFootprintCandidate] = []
    for candidate in found:
        duplicate = any(
            float(np.linalg.norm(candidate.position_mm - k.position_mm)) < 4.0
            and abs(float(candidate.closing_axis @ k.closing_axis)) > 0.995
            and float(candidate.approach @ k.approach) > 0.995
            for k in kept
        )
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= max_candidates:
            break
    return kept
