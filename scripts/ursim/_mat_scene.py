"""What the wrist camera of a URSim push probe sees: the owner's recorded mat, with parts rendered in.

URSim has no camera and moves no part, so ``probe_push_on_the_mat.py`` stands in for both, and for nothing else. The
camera is a D415 double behind the cell's own camera owner (``Camera``), so the pick perception and the live planner
world read one depth, placed by the arm's own TCP and the rig's own calibration:

* :class:`StaticScene` is one recorded look of the owner's cell (Track K's crop of P1, depth only), as BASE points: the
  mat as the camera read it, about 1 deg tilted, the bare table, the yellow bin. The pile of parts the Zollstock lay in
  is cut out of it and the mat filled in under it, a plane fitted to the mat around the pile with the mat's own noise,
  so the parts placed there have one neighbour each, the one the scene names.
* :class:`Part` is a cylinder or a box standing on the mat's reading (its foot on the local reading, its axis along the
  mat's normal), ray cast into every frame.
* :class:`Scene` holds the parts and what happens to them: a part the jaws close on is carried with the TCP and set
  down where they open (:meth:`Scene.jaws_changed`); a push moves the part where its plan put it
  (:meth:`Scene.pushed`). Nothing else moves a part.
* :class:`StandInD415` renders the scene from where the camera stands at every grab, as a D415 at 1280x720 reads it:
  nothing closer than :data:`MIN_Z_MM`, whole millimetres. :class:`StandInBackend` grounds the prompt on the part the
  scene names as the target, with the mask the same render cut, so the detector and the segmenter are the only
  models replaced.
* With ``StaticScene.whole_bench`` (2026-10-03), the mat, its four sides, the bench and the floor round it are ray cast
  from wherever the camera stands, the mat's top on the plane P1 reads, so a look from any pose sees the whole table
  and not only what P1 saw; the recorded look is kept for the yellow bin alone (:data:`RECORDED_FROM_X_MM` on).
* With ``Scene(depth_model="real")`` (2026-10-07), the clean render is read the way the cell's D415 reads the cell
  (:func:`d415_depth`, its numbers in :class:`D415`, measured on the cell's own frames of that day): an error growing
  with the depth on every surface, most of it the same at every grab from one pose; the band at the left no second
  imager sees and the shadow left of every nearer surface; a small step blurred and a large one kept; holes at edges,
  on grazing and shiny surfaces; whole millimetres. A fix the bench judges then meets the depth the cell hands it, not
  a drawing. ``clean`` keeps the render as it was (:data:`DEPTH_MODEL`).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

#: The stand-in reads no depth closer than this. Intel gives a D415 at 1280x720 about 45 cm; the owner's own D415 at that
#: resolution read depth on about half the frame with the tool 10 mm over a grasp 40 mm over the mat (robot.log of
#: 2026-10-01, "no depth on 50 %" before the line down to the grasp), where 45 cm leaves this render nothing at all for
#: half the wrist's turns and 30 cm 1 to 30 %. 20 cm reads 40 to 60 % there, as the cell did (2026-10-03): what a wrist
#: camera sees at a standoff is still largely holes, and never nothing.
MIN_Z_MM = 200.0
#: Nothing farther than this is read either; the cell's frames hold nothing beyond it.
MAX_Z_MM = 3000.0
#: Depth noise of a rendered part, one sigma, millimetres (a D415 at about 0.75 m). The recorded mat keeps its own.
PART_NOISE_MM = 0.8
#: The pile around the Zollstock in P1, BASE millimetres (x0, x1, y0, y1): cut out and filled with mat.
PILE_XY = (-290.0, -25.0, -830.0, -595.0)
#: The mat as P1 reads it, BASE millimetres (x0, x1, y0, y1): where the plane under the pile is fitted.
MAT_XY = (-420.0, 60.0, -880.0, -540.0)
#: Grid spacing of the mat filled in under the pile, millimetres.
FILL_STEP_MM = 1.5
#: The owner's mat as P1 shows it, BASE millimetres (x0, x1, y0, y1): what the whole bench casts, its top on the plane P1
#: reads, its sides down to the bench. E3's x from -411.3; its far end hides under the yellow bin, whose wall stands at
#: x 117 in P1 (the mat reads 55 mm up to there, the bin's wall and floor beyond), so the cast mat ends there too.
MAT_OUTLINE = (-411.3, 117.0, -865.0, -511.0)
#: The bench under it, at z = 0, as far as P1 read it.
BENCH_OUTLINE = (-430.0, 615.0, -935.0, -190.0)
#: The room's floor under and round the bench, BASE z millimetres: what a camera reads past the bench's edge.
FLOOR_Z_MM = -750.0
#: From this x on the whole bench keeps the recorded look: the yellow bin, from its wall on.
RECORDED_FROM_X_MM = 117.0
#: Depth noise of the cast mat and bench, one sigma, millimetres.
BENCH_NOISE_MM = 1.0
#: The depths a :class:`Scene` renders, by name. ``clean``: the render as it stood until 2026-10-07, the parts with
#: :data:`PART_NOISE_MM` and the cast bench with :data:`BENCH_NOISE_MM` of noise drawn afresh per pixel, no hole but out
#: of range. ``real``: the clean render read through :func:`d415_depth`, the cell's D415 as its own frames show it.
DEPTH_MODELS = ("clean", "real", "harsh")
#: The depth a scene renders unless it names one: ``clean``, so the URSim probe and every render a test pins keep theirs.
#: The bench renders ``real`` (``scripts/bench/run_bench.py --noise``).
DEPTH_MODEL = "clean"
#: What a pixel of a render shows, for :func:`d415_depth`: nothing, the room's floor, the bare bench, the mat's top, a side
#: of the mat, a point of the recorded look (the yellow bin), a part, a fixed part (a bin's or a tray's wall or floor).
_NOTHING, _FLOOR, _BENCH, _MAT, _MAT_SIDE, _RECORDED, _PART, _FIXED = range(8)


@dataclass(frozen=True)
class D415:
    """The cell's D415 as its own frames read it, for :func:`d415_depth`: 1280x720 aligned to colour, the spatial and the
    temporal filter on, no hole filling (the cell's ``cam.yaml``). Measured on the 30 looks the cell recorded on
    2026-10-07 (``D:/helper_cell/logs/robot/views``: four look poses, the mat 550 to 1000 mm away), each number with what
    it was fitted to. Distances in pixels are at :attr:`fx_px` and scale with the focal length."""

    #: The focal length the pixel numbers were measured at.
    fx_px: float = 913.7
    #: The depth's 1/z, as ``fx * baseline_mm`` pixel millimetres: the band at the left the second imager never sees is
    #: ``fx * baseline_mm / z + band_offset_px`` wide (fitted over 5 494 rows: 105 px at 600-700 mm, 95 px at 700-800,
    #: 89 px at 800-1000, 58 px past a metre), and a nearer surface hides what lies that much left of it from it.
    baseline_mm: float = 57.0
    band_offset_px: float = 23.0
    #: How much of a shadow reads no depth; the rest reads what lies there, blurred like any edge. Left of the parts on
    #: the mat a third to two fifths of the pixels next to the part read nothing, falling to 6 % five pixels out.
    shadow_lost: float = 0.4
    #: How coherent a shadow's holes are, pixels: the scale of the field they are cut from.
    shadow_hole_px: float = 3.0
    #: The spatial filter, as a domain transform in disparity (``cv2.ximgproc.dtFilter``): how far it reaches along a
    #: surface and how large a disparity step it reads as an edge (pixels), and how often it runs; a step is eased in full
    #: up to ``keep_from_px`` of disparity and not at all from ``keep_px`` on, by the largest within ``smooth_reach_px``.
    #: At a 25 to 45 mm part's hidden edge on the mat (about 4 px) the depth eases from its top to the mat over some eight
    #: pixels each side of the half-height line: 33 % of the step one pixel out, 12 % four out, 3 % eight out, and 29 %,
    #: 14 % and 4.5 % in; beside a 45 to 60 mm part two thirds of that, beside a taller one a third, and beside a bin's
    #: 110 mm wall (16 px) and the mat's own edge (8 px) on the bench next to nothing.
    smooth_px: float = 4.0
    smooth_disparity_px: float = 20.0
    smooth_iterations: int = 3
    smooth_reach_px: float = 12.0
    keep_from_px: float = 5.0
    keep_px: float = 11.0
    #: The depth error, one sigma at :attr:`noise_at_mm`, growing as the depth to ``noise_power`` (a stereo camera's
    #: disparity error grows as its square; the cell's mat between 600 and 900 mm a little slower). On the mat 25 px
    #: from any part, edge or hole: ``temporal_mm`` drawn afresh at every grab (the frames of one look differ by 0.64 mm
    #: at 600-700 mm, 0.81 at 700-800, 0.88 at 800-900, correlated over about ten pixels), and two parts that stay the
    #: same at every grab from one pose, as the cell's do: ``static_mm`` over about ``static_px`` (a frame less its own
    #: 31 px mean: 0.70 mm at 600-700 mm, 1.04 at 800-900) and ``warp_mm`` over about ``warp_px`` (a frame about one
    #: plane: 1.9 mm at 600-700 mm, 1.6 to 2.0 by look, 2.0 at 700-800; P1's mat of 2026-10-01 2.3 mm).
    noise_at_mm: float = 650.0
    noise_power: float = 1.5
    temporal_mm: float = 0.58
    temporal_px: float = 5.0
    static_mm: float = 0.98
    static_px: float = 8.0
    warp_mm: float = 1.85
    warp_px: float = 60.0
    #: Holes away from edges, the share of a surface's pixels: the mat 0.00 %, a part's top 0.03 %, the bare aluminium
    #: bench 2.4 %, the plastic bins 0.6 % (by :data:`_NOTHING` .. :data:`_FIXED`; the recorded look keeps its own holes
    #: and its own error). Holes come in blobs (on the bench a median 5 px, 86 % of what is lost in holes of 20 px and
    #: more): cut from a field of about ``hole_px``.
    surface_lost: tuple[float, ...] = (0.0, 0.01, 0.024, 0.0, 0.0, 0.0, 0.0003, 0.006)
    hole_px: float = 2.0
    #: Holes on a surface the camera sees at a grazing angle: the share lost at an incidence angle (degrees), over what
    #: any surface loses (the cell's frames: 0.9 % below 30 deg, 3.8 % at 60-70, 6.1 % at 70-75, 7.5 % at 75-80, 11.9 %
    #: at 80-85, 20.6 % past 85).
    grazing_deg: tuple[float, ...] = (55.0, 65.0, 72.5, 77.5, 82.5, 87.5, 90.0)
    grazing_lost: tuple[float, ...] = (0.0, 0.029, 0.052, 0.066, 0.11, 0.2, 0.2)
    #: Holes on the far side of a step, other than its shadow: ``edge_lost + edge_lost_per_px * step`` (disparity
    #: pixels), at most ``edge_lost_max``, next to the step, falling over ``edge_lost_px``. Below a part on the mat 20 %
    #: one pixel out, 6 % three out; right of it 3 %; beside a bin's 110 mm wall 16 to 24 % one to three pixels out, 10 %
    #: at five, 3 % at ten. A step is a disparity jump of more than ``edge_px`` between two neighbouring pixels.
    edge_px: float = 0.6
    edge_lost: float = 0.02
    edge_lost_per_px: float = 0.012
    edge_lost_max: float = 0.25
    edge_lost_px: float = 2.5
    #: How far past a step its holes reach, pixels.
    edge_reach_px: int = 12


#: A harsher D415 than the cell's own frames show, to stress the stack (``depth_model="harsh"``, the owner, 2026-10-07:
#: "die Tiefe bisschen schlechter ... sodass wir wirklich richtig viel abdecken"): twice the error at every scale, three
#: times the holes on every surface (the bare bench a tenth lost, as the cell's worst look read it), a stereo shadow
#: three fifths lost, twice the holes past a step and half as many again on a grazing surface.
HARSH = D415(temporal_mm=1.16, static_mm=1.96, warp_mm=2.8, shadow_lost=0.6,
             surface_lost=(0.0, 0.03, 0.10, 0.005, 0.005, 0.0, 0.001, 0.018),
             grazing_lost=(0.0, 0.044, 0.078, 0.099, 0.165, 0.3, 0.3),
             edge_lost=0.04, edge_lost_per_px=0.024, edge_lost_max=0.4)


def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return a / max(float(np.linalg.norm(a)), 1e-12)


def standing_frame(foot_mm: Any, normal: Any, yaw_rad: float) -> np.ndarray:
    """A part's frame in BASE: its origin at the centre of its foot, z along ``normal``, x along ``yaw_rad`` in plan."""
    z = _unit(normal)
    x = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    x = _unit(x - z * float(x @ z))
    y = np.cross(z, x)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = x, y, z
    frame[:3, 3] = np.asarray(foot_mm, dtype=np.float64).reshape(3)
    return frame


@dataclass
class Part:
    """A cylinder (``dims`` radius, radius, height) or a box (``dims`` x, y, z sizes) standing in its own frame."""

    name: str
    shape: str
    dims: tuple[float, float, float]
    frame: np.ndarray
    #: Its frame relative to the TCP while the jaws hold it, else ``None``.
    held: Optional[np.ndarray] = None
    #: A part of the cell, not of the scene's work: a bin's wall or floor, seen by the camera, never gripped or pushed.
    fixed: bool = False

    @property
    def height_mm(self) -> float:
        return float(self.dims[2])

    def placed(self, tcp: Optional[np.ndarray]) -> np.ndarray:
        """Where it stands now: its own frame, or carried with ``tcp`` while held."""
        if self.held is not None and tcp is not None:
            return np.asarray(tcp, dtype=np.float64) @ self.held
        return self.frame

    def corners(self) -> np.ndarray:
        """The 8 corners of its bounding box in its own frame."""
        if self.shape == "cylinder":
            hx = hy = float(self.dims[0])
        else:
            hx, hy = float(self.dims[0]) / 2.0, float(self.dims[1]) / 2.0
        h = self.height_mm
        return np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)])

    def contains(self, point_local: np.ndarray, grow_mm: float) -> bool:
        x, y, z = (float(v) for v in point_local[:3])
        if not -grow_mm <= z <= self.height_mm + grow_mm:
            return False
        if self.shape == "cylinder":
            return math.hypot(x, y) <= float(self.dims[0]) + grow_mm
        return abs(x) <= self.dims[0] / 2.0 + grow_mm and abs(y) <= self.dims[1] / 2.0 + grow_mm

    def surface_points(self, step_mm: float = 2.0) -> np.ndarray:
        """Its whole surface but the foot face, sampled every ``step_mm``, in BASE: what cameras all round would see."""
        h = self.height_mm
        local: list[tuple[float, float, float]] = []
        if self.shape == "cylinder":
            r = float(self.dims[0])
            for rr in np.arange(0.0, r + 1e-9, step_mm):
                n = max(1, int(2.0 * math.pi * rr / step_mm))
                local += [(rr * math.cos(a), rr * math.sin(a), h) for a in np.linspace(0.0, 2.0 * math.pi, n, False)]
            n = max(8, int(2.0 * math.pi * r / step_mm))
            for z in np.arange(0.5, h, step_mm):
                local += [(r * math.cos(a), r * math.sin(a), z) for a in np.linspace(0.0, 2.0 * math.pi, n, False)]
        else:
            hx, hy = self.dims[0] / 2.0, self.dims[1] / 2.0
            xs, ys = np.arange(-hx, hx + 1e-9, step_mm), np.arange(-hy, hy + 1e-9, step_mm)
            local += [(x, y, h) for x in xs for y in ys]
            for z in np.arange(0.5, h, step_mm):
                local += [(x, y, z) for x in xs for y in (-hy, hy)] + [(x, y, z) for y in ys for x in (-hx, hx)]
        pts = np.asarray(local, dtype=np.float64)
        return pts @ self.frame[:3, :3].T + self.frame[:3, 3]

    def hit(self, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """The ray parameter of the first hit of ``origin + s * dirs`` (local frame), ``inf`` where none."""
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        d = np.asarray(dirs, dtype=np.float64)
        h = self.height_mm
        out = np.full(d.shape[0], np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.shape == "cylinder":
                r = float(self.dims[0])
                a = d[:, 0] ** 2 + d[:, 1] ** 2
                b = 2.0 * (o[0] * d[:, 0] + o[1] * d[:, 1])
                c = o[0] ** 2 + o[1] ** 2 - r * r
                disc = b * b - 4.0 * a * c
                ok = (disc >= 0.0) & (a > 1e-12)
                s_side = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * a), np.inf)
                z_side = o[2] + s_side * d[:, 2]
                side = ok & (s_side > 0.0) & (z_side >= 0.0) & (z_side <= h)
                out = np.where(side, s_side, out)
                s_top = (h - o[2]) / d[:, 2]
                xt, yt = o[0] + s_top * d[:, 0], o[1] + s_top * d[:, 1]
                top = (s_top > 0.0) & (xt * xt + yt * yt <= r * r)
                out = np.where(top & (s_top < out), s_top, out)
            else:
                lo = np.array([-self.dims[0] / 2.0, -self.dims[1] / 2.0, 0.0])
                hi = np.array([self.dims[0] / 2.0, self.dims[1] / 2.0, h])
                t1 = (lo[None, :] - o[None, :]) / d
                t2 = (hi[None, :] - o[None, :]) / d
                t_near = np.nanmax(np.minimum(t1, t2), axis=1)
                t_far = np.nanmin(np.maximum(t1, t2), axis=1)
                box = (t_far >= t_near) & (t_near > 0.0)
                out = np.where(box, t_near, out)
        return out


def cylinder(name: str, foot_mm: Sequence[float], normal: Sequence[float], *, radius_mm: float,
             height_mm: float) -> Part:
    return Part(name, "cylinder", (radius_mm, radius_mm, height_mm), standing_frame(foot_mm, normal, 0.0))


def box(name: str, foot_mm: Sequence[float], normal: Sequence[float], *, size_mm: Sequence[float],
        yaw_rad: float = 0.0) -> Part:
    sx, sy, sz = (float(v) for v in size_mm)
    return Part(name, "box", (sx, sy, sz), standing_frame(foot_mm, normal, yaw_rad))


@dataclass
class StaticScene:
    """One recorded look as BASE points, the pile cut out and the mat filled in under it."""

    points: np.ndarray
    #: The mat's plane ``z = a + b x + c y`` and how far its readings lie from it (rms, mm).
    mat_plane: np.ndarray
    mat_rms_mm: float
    intrinsics: np.ndarray
    shape: tuple[int, int]
    source: str = ""
    pile_points_cut: int = 0
    #: Cast the mat, its sides and the bench from wherever the camera stands (:meth:`cast`), the recorded look kept
    #: for the bin alone; ``False`` projects the recorded look, as before.
    whole_bench: bool = False

    @classmethod
    def from_crop(cls, path: "str | Path", *, pile_xy: tuple[float, float, float, float] = PILE_XY,
                  mat_xy: tuple[float, float, float, float] = MAT_XY, seed: int = 20261005) -> "StaticScene":
        """Track K's depth crop of a recorded look (``tests/data/cell_2026_10_01``): its depth at every second pixel,
        its intrinsics and CAMERA to BASE. Every point it read is kept, the far ones too."""
        data = np.load(str(path))
        stride = int(data["stride"])
        depth = np.asarray(data["surface_depth_strided_mm"], dtype=np.float64)
        k = np.asarray(data["intrinsics"], dtype=np.float64)
        m = np.asarray(data["camera_to_base"], dtype=np.float64)
        rows, cols = np.nonzero(depth > 0)
        z = depth[rows, cols]
        u, v = cols * stride, rows * stride
        cam = np.column_stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
        base = cam @ m[:3, :3].T + m[:3, 3]
        px0, px1, py0, py1 = pile_xy
        in_pile = (base[:, 0] > px0) & (base[:, 0] < px1) & (base[:, 1] > py0) & (base[:, 1] < py1)
        mx0, mx1, my0, my1 = mat_xy
        ring = ((base[:, 0] > mx0) & (base[:, 0] < mx1) & (base[:, 1] > my0) & (base[:, 1] < my1) & ~in_pile
                & (base[:, 2] > 40.0) & (base[:, 2] < 75.0))
        a = np.column_stack([np.ones(int(ring.sum())), base[ring, 0], base[ring, 1]])
        coef, *_ = np.linalg.lstsq(a, base[ring, 2], rcond=None)
        for _ in range(2):  # drop what stands off the plane (an edge, a stray part) and fit again
            resid = base[ring, 2] - a @ coef
            inl = np.abs(resid) < 3.0 * max(float(np.std(resid)), 0.5)
            coef, *_ = np.linalg.lstsq(a[inl], base[ring, 2][inl], rcond=None)
        resid = base[ring, 2] - a @ coef
        rms = float(np.sqrt(np.mean(resid[np.abs(resid) < 6.0] ** 2)))
        rng = np.random.default_rng(seed)
        gx, gy = np.meshgrid(np.arange(px0, px1, FILL_STEP_MM), np.arange(py0, py1, FILL_STEP_MM))
        gx, gy = gx.ravel(), gy.ravel()
        gz = coef[0] + coef[1] * gx + coef[2] * gy + rng.normal(0.0, min(rms, 1.5), gx.size)
        fill = np.column_stack([gx, gy, gz])
        points = np.vstack([base[~in_pile], fill])
        h, w = (int(v) for v in data["frame_shape"])
        return cls(points=points, mat_plane=np.asarray(coef, dtype=np.float64), mat_rms_mm=rms, intrinsics=k,
                   shape=(h, w), source=str(data["source"]), pile_points_cut=int(in_pile.sum()))

    @property
    def mat_normal(self) -> np.ndarray:
        return _unit([-self.mat_plane[1], -self.mat_plane[2], 1.0])

    @property
    def mat_tilt_deg(self) -> float:
        return math.degrees(math.acos(float(self.mat_normal[2])))

    def mat_at(self, xy: Sequence[float]) -> float:
        return float(self.mat_plane[0] + self.mat_plane[1] * xy[0] + self.mat_plane[2] * xy[1])

    def support_at(self, xy: Sequence[float]) -> tuple[float, np.ndarray]:
        """What a part set down at ``xy`` stands on: the mat's plane where the mat is read, else the recorded surface."""
        mx0, mx1, my0, my1 = MAT_OUTLINE if self.whole_bench else MAT_XY
        if mx0 < xy[0] < mx1 and my0 < xy[1] < my1:
            return self.mat_at(xy), self.mat_normal
        if self.whole_bench:
            return 0.0, np.array([0.0, 0.0, 1.0])
        near = np.hypot(self.points[:, 0] - xy[0], self.points[:, 1] - xy[1]) < 20.0
        z = float(np.percentile(self.points[near, 2], 30.0)) if near.any() else 0.0
        return z, np.array([0.0, 0.0, 1.0])


    def cast(self, cam_to_base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The whole bench's depth from ``cam_to_base``, CAMERA z millimetres per pixel (``inf`` where nothing is hit): the
        mat's top on its plane and its four sides down to the bench within :data:`MAT_OUTLINE`, the bench at z = 0
        within :data:`BENCH_OUTLINE`, the floor at :data:`FLOOR_Z_MM` round it, each with :data:`BENCH_NOISE_MM` of
        noise, and the recorded look from :data:`RECORDED_FROM_X_MM` on."""
        best, _ = self._cast_clean(cam_to_base)
        seen = np.isfinite(best)
        best[seen] += rng.normal(0.0, BENCH_NOISE_MM, int(seen.sum()))
        recorded = self.points[self.points[:, 0] >= RECORDED_FROM_X_MM]
        return np.minimum(best, _project_static(recorded, cam_to_base, self.intrinsics, self.shape))

    def surfaces(self, cam_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What the camera at ``cam_to_base`` sees of the cell with no noise at all, CAMERA z millimetres per pixel
        (``inf`` where nothing is hit), and which surface each pixel shows (:data:`_FLOOR` .. :data:`_RECORDED`): the
        whole bench as :meth:`cast` casts it, the recorded look from :data:`RECORDED_FROM_X_MM` on; or, off the whole
        bench, the recorded look alone. What ``depth_model="real"`` reads through :func:`d415_depth`."""
        if not self.whole_bench:
            depth = _project_static(self.points, cam_to_base, self.intrinsics, self.shape)
            return depth, np.where(np.isfinite(depth), _RECORDED, _NOTHING).astype(np.int8)
        best, surface = self._cast_clean(cam_to_base)
        recorded = _project_static(self.points[self.points[:, 0] >= RECORDED_FROM_X_MM], cam_to_base, self.intrinsics,
                                   self.shape)
        nearer = recorded < best
        return np.where(nearer, recorded, best), np.where(nearer, _RECORDED, surface).astype(np.int8)

    def _cast_clean(self, cam_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`cast` before its noise and the recorded look: the depth, and the surface each pixel shows."""
        h, w = self.shape
        k = self.intrinsics
        r, origin = cam_to_base[:3, :3], cam_to_base[:3, 3]
        vv, uu = np.mgrid[0:h, 0:w]
        rays = np.stack([(uu - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1], np.ones((h, w))], axis=-1).reshape(-1, 3)
        d = rays @ r.T  # BASE directions, the camera's z component 1: the ray parameter is the camera depth
        ox, oy, oz = (float(v) for v in origin)
        a, b, c = (float(v) for v in self.mat_plane)
        best = np.full(d.shape[0], np.inf)
        surface = np.full(d.shape[0], _NOTHING, dtype=np.int8)
        mx0, mx1, my0, my1 = MAT_OUTLINE
        bx0, bx1, by0, by1 = BENCH_OUTLINE
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (FLOOR_Z_MM - oz) / d[:, 2]
            x, y = ox + s * d[:, 0], oy + s * d[:, 1]
            hit = (s > 0) & ~((x > bx0) & (x < bx1) & (y > by0) & (y < by1))
            surface[hit & (s < best)] = _FLOOR
            best = np.where(hit, np.minimum(best, s), best)
            s = (0.0 - oz) / d[:, 2]
            x, y = ox + s * d[:, 0], oy + s * d[:, 1]
            hit = (s > 0) & (x > bx0) & (x < bx1) & (y > by0) & (y < by1) & ~((x > mx0) & (x < mx1) & (y > my0)
                                                                               & (y < my1))
            surface[hit & (s < best)] = _BENCH
            best = np.where(hit, np.minimum(best, s), best)
            s = (a + b * ox + c * oy - oz) / (d[:, 2] - b * d[:, 0] - c * d[:, 1])
            x, y = ox + s * d[:, 0], oy + s * d[:, 1]
            hit = (s > 0) & (x > mx0) & (x < mx1) & (y > my0) & (y < my1)
            surface[hit & (s < best)] = _MAT
            best = np.where(hit, np.minimum(best, s), best)
            for axis, at in ((0, mx0), (0, mx1), (1, my0), (1, my1)):
                s = (at - (ox, oy)[axis]) / d[:, axis]
                x, y, z = ox + s * d[:, 0], oy + s * d[:, 1], oz + s * d[:, 2]
                along = y if axis == 0 else x
                low, high = (my0, my1) if axis == 0 else (mx0, mx1)
                hit = (s > 0) & (along > low) & (along < high) & (z > 0.0) & (z < a + b * x + c * y)
                surface[hit & (s < best)] = _MAT_SIDE
                best = np.where(hit, np.minimum(best, s), best)
        return best.reshape(h, w), surface.reshape(h, w)


def _project_static(points: np.ndarray, cam_to_base: np.ndarray, k: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r, t = cam_to_base[:3, :3], cam_to_base[:3, 3]
    pc = (points - t) @ r
    z = pc[:, 2]
    ok = z > 1.0
    pc, z = pc[ok], z[ok]
    u = np.rint(k[0, 0] * pc[:, 0] / z + k[0, 2]).astype(np.int64)
    v = np.rint(k[1, 1] * pc[:, 1] / z + k[1, 2]).astype(np.int64)
    zbuf = np.full(h * w, np.inf)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = u + du, v + dv
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(zbuf, vv[inside] * w + uu[inside], z[inside])
    return zbuf.reshape(h, w)


def d415_depth(depth_mm: np.ndarray, surface: np.ndarray, intrinsics: np.ndarray, *, static_rng: np.random.Generator,
               temporal_rng: Optional[np.random.Generator], model: D415 = D415()) -> np.ndarray:
    """``depth_mm``, a clean render (CAMERA z millimetres, ``inf`` where nothing is hit), as the cell's D415 reads it:
    millimetres, ``nan`` where it reads nothing, before the range and the whole millimetres :meth:`Scene.render` keeps.

    In the camera's own order: what the second imager cannot see, the band at the left and the shadow a nearer surface
    throws to its left, part of that shadow lost; the spatial filter, which eases a small step in disparity and keeps a
    large one; the error, one sigma growing with the depth, mostly the pose's (``static_rng``: the same at every grab
    from one pose) and partly the grab's (``temporal_rng``); and the holes, in blobs, by the surface each pixel shows
    (``surface``, :data:`_NOTHING` .. :data:`_FIXED`), on a grazing surface and just past a step. Every random number
    is drawn as a whole frame, so one seed and one pose give one frame, whatever the scene holds. ``temporal_rng`` None
    leaves the grab's own error out, for a caller that adds it at every grab (:func:`d415_temporal`)."""
    import cv2  # noqa: PLC0415 (the clean depth needs neither)
    from scipy import ndimage, special  # noqa: PLC0415

    h, w = depth_mm.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    scale = float(k[0, 0]) / model.fx_px
    kb = float(k[0, 0]) * model.baseline_mm
    seen = np.isfinite(depth_mm) & (depth_mm > 0.0)
    disp = np.where(seen, kb / np.where(seen, depth_mm, 1.0), 0.0)
    cols = np.arange(w, dtype=np.float64)[None, :]
    # The second imager sees a pixel at its column less its disparity: nothing left of its frame (the band), nothing a
    # nearer pixel right of it lands over (the shadow).
    band = seen & (cols < disp + model.band_offset_px * scale)
    lands = cols - disp
    over = np.full((h, w), np.inf)
    over[:, :-1] = np.minimum.accumulate(lands[:, ::-1], axis=1)[:, ::-1][:, 1:]
    shadow = seen & ~band & (over < lands - 0.25)
    lost = ~seen | band | (shadow & (_field(static_rng, (h, w), model.shadow_hole_px * scale)
                                     < special.ndtri(model.shadow_lost)))
    # The spatial filter, in disparity, guided by the scene's own steps and normalised over what is read, so it eases
    # toward what was read and never toward a hole's zero; a step in full up to ``keep_from_px``, not at all from
    # ``keep_px`` on, by the largest step within its reach.
    guide = disp.astype(np.float32)
    read = np.where(lost, 0.0, disp).astype(np.float32)
    weight = (~lost).astype(np.float32)
    filtered = [cv2.ximgproc.dtFilter(guide, src, model.smooth_px * scale, model.smooth_disparity_px,
                                      mode=cv2.ximgproc.DTF_RF, numIters=model.smooth_iterations).astype(np.float64)
                for src in (read, weight)]
    eased = filtered[0] / np.maximum(filtered[1], 1e-6)
    jump = np.zeros((h, w))
    for first, second in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                          ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        step = np.where(~lost[first] & ~lost[second], np.abs(disp[second] - disp[first]), 0.0)
        jump[first] = np.maximum(jump[first], step)
        jump[second] = np.maximum(jump[second], step)
    reach = 2 * int(math.ceil(model.smooth_reach_px * scale)) + 1
    largest = ndimage.maximum_filter(jump, size=reach)
    ease = np.clip((model.keep_px - largest) / (model.keep_px - model.keep_from_px), 0.0, 1.0)
    ease = ndimage.gaussian_filter(ease, reach / 8.0)  # no square seam where the largest step changes
    eased = np.where(lost, 0.0, disp + ease * (eased - disp))
    ok = ~lost & (eased > 0.0)
    out = np.where(ok, kb / np.where(ok, eased, 1.0), np.nan)
    grow = np.where(surface == _RECORDED, 0.0,
                    (np.where(ok, out, model.noise_at_mm) / model.noise_at_mm) ** model.noise_power)
    if temporal_rng is not None:
        out += grow * model.temporal_mm * _field(temporal_rng, (h, w), model.temporal_px * scale)
    out += grow * (model.static_mm * _field(static_rng, (h, w), model.static_px * scale)
                   + model.warp_mm * _field(static_rng, (h, w), model.warp_px * scale))
    # Holes: the surface's own share, a grazing surface's, and the far side of a step's.
    share = np.asarray(model.surface_lost, dtype=np.float64)[np.clip(surface, 0, len(model.surface_lost) - 1)]
    nearer = np.zeros((h, w), dtype=bool)  # the near side of a step: a read neighbour lies more than a step farther
    for first, second in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                          ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        both = seen[first] & seen[second]
        rise = disp[second] - disp[first]
        nearer[first] |= both & (rise < -model.edge_px)
        nearer[second] |= both & (rise > model.edge_px)
    on_step = nearer.copy()
    if nearer.any():
        far, (ir, ic) = ndimage.distance_transform_edt(~nearer, return_indices=True)
        rise = disp[ir, ic] - disp
        beside = seen & ~nearer & (far <= model.edge_reach_px * scale) & (rise > model.edge_px)
        share += np.where(beside, np.minimum(model.edge_lost + model.edge_lost_per_px * rise, model.edge_lost_max)
                          * np.exp(-(far - 1.0) / (model.edge_lost_px * scale)), 0.0)
        on_step |= beside & (far <= 1.5)
    # A grazing surface away from the steps (at a step the depth's own neighbours say nothing of the surface).
    flat = seen & ~ndimage.binary_dilation(on_step)
    grazing = np.interp(-_incidence_cos(depth_mm, k), -np.cos(np.radians(model.grazing_deg)), model.grazing_lost)
    share += np.where(flat, grazing, 0.0)
    holes = _field(static_rng, (h, w), model.hole_px * scale)
    some = share > 0.0
    out[some] = np.where(holes[some] < special.ndtri(np.minimum(share[some], 1.0)), np.nan, out[some])
    return out


def d415_temporal(depth_mm: np.ndarray, surface: np.ndarray, intrinsics: np.ndarray, rng: np.random.Generator,
                  model: D415 = D415()) -> np.ndarray:
    """The part of the D415's error a new grab draws afresh, over a frame :func:`d415_depth` read without it: one sigma
    ``temporal_mm`` at :attr:`D415.noise_at_mm`, growing with the depth as the rest, none on the recorded look."""
    h, w = depth_mm.shape
    scale = float(np.asarray(intrinsics, dtype=np.float64)[0, 0]) / model.fx_px
    ok = np.isfinite(depth_mm)
    grow = np.where(surface == _RECORDED, 0.0,
                    (np.where(ok, depth_mm, model.noise_at_mm) / model.noise_at_mm) ** model.noise_power)
    return grow * model.temporal_mm * _field(rng, (h, w), model.temporal_px * scale)


def _incidence_cos(depth_mm: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The cosine of the angle between each pixel's ray and the surface it meets, from the depth's own neighbours (1 where
    nothing is hit). For a pinhole camera the normal at ``z(u, v)`` is ``(-fx z_u, -fy z_v, fx z_u x + fy z_v y + z)``,
    ``x, y`` the ray's own slopes, and its dot product with the ray ``(x, y, 1)`` is ``z``."""
    h, w = depth_mm.shape
    z = np.where(np.isfinite(depth_mm) & (depth_mm > 0.0), depth_mm, np.nan)
    z_v, z_u = np.gradient(z)
    x = ((np.arange(w) - k[0, 2]) / k[0, 0])[None, :]
    y = ((np.arange(h) - k[1, 2]) / k[1, 1])[:, None]
    a, b = k[0, 0] * z_u, k[1, 1] * z_v
    c = a * x + b * y + z
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = z / (np.sqrt(a * a + b * b + c * c) * np.sqrt(x * x + y * y + 1.0))
    return np.where(np.isfinite(cos), cos, 1.0)


def _field(rng: np.random.Generator, shape: tuple[int, int], scale_px: float) -> np.ndarray:
    """A smooth field over ``shape``, standard normal at every pixel: white noise blurred by a Gaussian of ``scale_px``,
    drawn on a grid 2.5 pixels of scale apart where the scale allows it and brought to the frame bilinearly. Its size
    depends on ``shape`` and ``scale_px`` alone, so the numbers drawn never depend on what a frame shows."""
    from scipy import ndimage  # noqa: PLC0415

    h, w = shape
    step = max(1, int(scale_px // 2.5))
    grid = rng.standard_normal((h // step + 4, w // step + 4), dtype=np.float32)
    smooth = ndimage.gaussian_filter(grid, max(scale_px / step, 0.5), mode="wrap")
    if step > 1:
        smooth = ndimage.zoom(smooth, step, order=1)
    smooth = smooth[:h, :w]
    return smooth / max(float(np.std(smooth)), 1e-12)


@dataclass
class Render:
    depth_mm: np.ndarray
    masks: dict[str, np.ndarray]
    cam_to_base: np.ndarray
    seconds: float
    #: A D415 render read without the grab's own error: the depth before the range and the whole millimetres, and what
    #: each pixel shows, for the camera to add that error at every grab (:func:`d415_temporal`). ``None`` otherwise.
    unrounded_mm: Optional[np.ndarray] = None
    surface: Optional[np.ndarray] = None
    #: The masks of such a render before the range clipped them to what reads, for each grab to clip again.
    masks_before: Optional[dict[str, np.ndarray]] = None


class Scene:
    """The parts on the recorded mat, and the only things that move them: the jaws and a push.

    ``depth_model`` says how a render reads its depth (:data:`DEPTH_MODELS`); ``seed`` seeds every number it draws, so
    one seed and one sequence of renders give one sequence of frames."""

    def __init__(self, static: StaticScene, *, target: str = "", seed: int = 7, depth_model: str = DEPTH_MODEL) -> None:
        if depth_model not in DEPTH_MODELS:
            raise ValueError(f"depth_model is one of {DEPTH_MODELS}, got {depth_model!r}")
        self.static = static
        self.parts: dict[str, Part] = {}
        self.target = target
        self.version = 0
        self.events: list[dict[str, Any]] = []
        self.depth_model = depth_model
        #: The D415's numbers a ``real`` render reads with, a ``harsh`` one with :data:`HARSH`.
        self.d415 = HARSH if depth_model == "harsh" else D415()
        self._lock = threading.RLock()
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def set_parts(self, parts: Sequence[Part], *, target: str) -> None:
        with self._lock:
            self.parts = {part.name: part for part in parts}
            self.target = target
            self.version += 1
            self.events.append({"t": time.time(), "event": "scene", "parts": [p.name for p in parts]})

    def foot_on_mat(self, xy: Any) -> tuple[np.ndarray, np.ndarray]:
        z, normal = self.static.support_at(xy)
        return np.array([float(xy[0]), float(xy[1]), z]), normal

    def mat_points(self, step_mm: float = 2.5) -> np.ndarray:
        """The mat's reading on a grid over :data:`MAT_XY` (:data:`MAT_OUTLINE` on the whole bench), less where a part
        stands: the table cameras all round would see, BASE millimetres."""
        x0, x1, y0, y1 = MAT_OUTLINE if self.static.whole_bench else MAT_XY
        gx, gy = np.meshgrid(np.arange(x0, x1, step_mm), np.arange(y0, y1, step_mm))
        gx, gy = gx.ravel(), gy.ravel()
        plane = self.static.mat_plane
        pts = np.column_stack([gx, gy, plane[0] + plane[1] * gx + plane[2] * gy])
        keep = np.ones(pts.shape[0], dtype=bool)
        with self._lock:
            for part in self.parts.values():
                local = (pts - part.frame[:3, 3]) @ part.frame[:3, :3]
                if part.shape == "cylinder":
                    keep &= np.hypot(local[:, 0], local[:, 1]) > float(part.dims[0]) + 2.0
                else:
                    keep &= ~((np.abs(local[:, 0]) <= part.dims[0] / 2.0 + 2.0)
                              & (np.abs(local[:, 1]) <= part.dims[1] / 2.0 + 2.0))
        return pts[keep]

    def held(self) -> Optional[Part]:
        with self._lock:
            return next((p for p in self.parts.values() if p.held is not None), None)

    def jaws_changed(self, closed: bool, tcp: np.ndarray) -> None:
        """The jaws moved at ``tcp`` (4x4 BASE mm): closing grips the part the TCP stands in, opening sets it down."""
        with self._lock:
            if closed:
                for part in self.parts.values():
                    if part.fixed:
                        continue
                    local = np.linalg.inv(part.frame) @ np.append(tcp[:3, 3], 1.0)
                    if part.held is None and part.contains(local, grow_mm=4.0):
                        part.held = np.linalg.inv(tcp) @ part.frame
                        self.version += 1
                        self.events.append({"t": time.time(), "event": "gripped", "part": part.name,
                                            "tcp_mm": [round(float(v), 1) for v in tcp[:3, 3]]})
                        return
                self.events.append({"t": time.time(), "event": "closed_on_nothing",
                                    "tcp_mm": [round(float(v), 1) for v in tcp[:3, 3]]})
                return
            carried_part = next((p for p in self.parts.values() if p.held is not None), None)
            if carried_part is None:
                self.events.append({"t": time.time(), "event": "opened",
                                    "tcp_mm": [round(float(v), 1) for v in tcp[:3, 3]]})
                return
            carried = tcp @ carried_part.held
            foot, normal = self.foot_on_mat(carried[:2, 3])
            yaw = math.atan2(float(carried[1, 0]), float(carried[0, 0]))
            carried_part.frame, carried_part.held = standing_frame(foot, normal, yaw), None
            self.version += 1
            self.events.append({"t": time.time(), "event": "set_down", "part": carried_part.name,
                                "at_mm": [round(float(v), 1) for v in foot],
                                "dropped_mm": round(float(carried[2, 3] - foot[2]), 1)})

    def pushed(self, plan: Any, legs_done: Sequence[str], tcp_after: Optional[np.ndarray]) -> None:
        """A push ran: the part whose foot is nearest the plan's centre goes where the plan put it, or, stopped on its
        push leg, as far as the finger went past the contact gap."""
        with self._lock:
            if not self.parts:
                return
            centre = np.asarray(plan.target_centre_mm, dtype=np.float64)
            movable = [p for p in self.parts.values() if not p.fixed]
            if not movable:
                return
            part = min(movable, key=lambda p: float(np.hypot(*(p.frame[:2, 3] - centre[:2]))))
            direction = np.asarray(plan.direction, dtype=np.float64)[:2]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            if "push" in legs_done:
                moved = float(plan.push_distance_mm)
            elif tcp_after is not None:
                start = np.asarray(plan.contact_start_tcp_mm, dtype=np.float64)[:2]
                travelled = float((np.asarray(tcp_after)[:2, 3] - start) @ direction)
                moved = max(0.0, min(float(plan.push_distance_mm), travelled - 10.0))
            else:
                moved = 0.0
            if moved <= 0.0:
                return
            xy = part.frame[:2, 3] + moved * direction
            if getattr(self, "keep_height_on_push", False):
                # On a floor that is not the mat (a bin's), the part slides at the height it stood at.
                part.frame = part.frame.copy()
                part.frame[:2, 3] = xy
                foot = part.frame[:3, 3]
            else:
                foot, normal = self.foot_on_mat(xy)
                yaw = math.atan2(float(part.frame[1, 0]), float(part.frame[0, 0]))
                part.frame = standing_frame(foot, normal, yaw)
            self.version += 1
            self.events.append({"t": time.time(), "event": "pushed", "part": part.name, "moved_mm": round(moved, 1),
                                "to_mm": [round(float(v), 1) for v in foot]})

    def render(self, cam_to_base: np.ndarray, tcp: Optional[np.ndarray], *, temporal: bool = True) -> Render:
        """The scene from ``cam_to_base``. A D415 render (``real``, ``harsh``) with ``temporal`` false leaves the grab's
        own error out and keeps what :func:`d415_temporal` needs to add it (:attr:`Render.unrounded_mm`)."""
        started = time.perf_counter()
        st = self.static
        k, (h, w) = st.intrinsics, st.shape
        real = self.depth_model in ("real", "harsh")
        if real:
            depth, surface = st.surfaces(cam_to_base)
        else:
            depth = (st.cast(cam_to_base, self._rng) if st.whole_bench
                     else _project_static(st.points, cam_to_base, k, (h, w)))
        r, t = cam_to_base[:3, :3], cam_to_base[:3, 3]
        masks: dict[str, np.ndarray] = {}
        hits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        with self._lock:
            parts = [(p.name, p, p.placed(tcp)) for p in self.parts.values()]
        for name, part, frame in parts:
            corners = (frame[:3, :3] @ part.corners().T).T + frame[:3, 3]
            pc = (corners - t) @ r
            if np.any(pc[:, 2] <= 1.0):
                continue
            u = k[0, 0] * pc[:, 0] / pc[:, 2] + k[0, 2]
            v = k[1, 1] * pc[:, 1] / pc[:, 2] + k[1, 2]
            u0, u1 = max(0, int(math.floor(u.min()))), min(w - 1, int(math.ceil(u.max())))
            v0, v1 = max(0, int(math.floor(v.min()))), min(h - 1, int(math.ceil(v.max())))
            if u0 > u1 or v0 > v1:
                continue
            vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
            dirs_cam = np.column_stack([(uu.ravel() - k[0, 2]) / k[0, 0], (vv.ravel() - k[1, 2]) / k[1, 1],
                                        np.ones(uu.size)])
            inv = np.linalg.inv(frame)
            origin = inv[:3, :3] @ t + inv[:3, 3]
            dirs = (inv[:3, :3] @ (r @ dirs_cam.T)).T
            s = part.hit(origin, dirs)
            hits[name] = (vv.ravel(), uu.ravel(), s)
        nearest = depth.copy()
        for vv, uu, s in hits.values():
            np.minimum.at(nearest, (vv, uu), s)
        for name, (vv, uu, s) in hits.items():
            mine = np.isfinite(s) & (s <= nearest[vv, uu] + 1e-9)
            mask = np.zeros((h, w), dtype=bool)
            mask[vv[mine], uu[mine]] = True
            masks[name] = mask
        if real:
            fixed = {name for name, part, _ in parts if part.fixed}
            for name, mask in masks.items():
                surface[mask] = _FIXED if name in fixed else _PART
            noisy = d415_depth(nearest, surface, k, static_rng=self._pose_rng(cam_to_base),
                               temporal_rng=self._rng if temporal else None, model=self.d415)
        else:
            noisy = nearest.copy()
            if masks:
                on_parts = np.any(np.stack(list(masks.values())), axis=0)
                noisy[on_parts] += self._rng.normal(0.0, PART_NOISE_MM, int(on_parts.sum()))
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(noisy) & (noisy >= MIN_Z_MM) & (noisy <= MAX_Z_MM)
        out = np.where(valid, np.rint(noisy), 0.0).astype(np.uint16)
        kept = {name: mask & valid for name, mask in masks.items()}
        if real and not temporal:
            return Render(out, kept, cam_to_base, time.perf_counter() - started, unrounded_mm=noisy,
                          surface=surface, masks_before=masks)
        return Render(out, kept, cam_to_base, time.perf_counter() - started)

    def with_grab_error(self, rendered: Render) -> Render:
        """``rendered`` (read without the grab's own error) with a fresh draw of it: what a new grab from the same pose
        reads, its fixed error and its holes kept, as the cell's camera does."""
        assert rendered.unrounded_mm is not None and rendered.surface is not None
        noisy = rendered.unrounded_mm + d415_temporal(rendered.unrounded_mm, rendered.surface,
                                                      self.static.intrinsics, self._rng, self.d415)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(noisy) & (noisy >= MIN_Z_MM) & (noisy <= MAX_Z_MM)
        out = np.where(valid, np.rint(noisy), 0.0).astype(np.uint16)
        before = rendered.masks_before if rendered.masks_before is not None else rendered.masks
        return Render(out, {name: mask & valid for name, mask in before.items()}, rendered.cam_to_base,
                      rendered.seconds)

    def _pose_rng(self, cam_to_base: np.ndarray) -> np.random.Generator:
        """What the camera's error keeps at a pose (:func:`d415_depth`'s ``static_rng``): seeded by the scene's seed and
        the pose to a thousandth, so every grab from one pose meets the same static error and holes, as the cell's
        camera does, and another pose others."""
        pose = np.round(np.asarray(cam_to_base, dtype=np.float64)[:3, :4], 3) + 0.0  # + 0.0: a -0.0 seeds as 0.0
        digest = hashlib.blake2b(pose.tobytes(), digest_size=8).digest()
        return np.random.default_rng([self._seed, int.from_bytes(digest, "little")])


def colour_of(depth: np.ndarray, masks: dict[str, np.ndarray], target: str) -> np.ndarray:
    """A BGR picture of a render for a person or a debug overlay: grey by depth, the target green, the rest red."""
    d = depth.astype(np.float64)
    shade = np.where(d > 0, 255.0 - np.clip((d - MIN_Z_MM) / 4.0, 0.0, 200.0), 0.0).astype(np.uint8)
    bgr = np.repeat(shade[:, :, None], 3, axis=2)
    for name, mask in masks.items():
        bgr[mask] = (40, 200, 40) if name == target else (40, 40, 200)
    return bgr


class StandInD415:
    """A streamer the cell's ``Camera`` owner opens like a RealSense: every grab is the scene rendered from where the
    camera stands now (the TCP read off the controller, composed with the rig's CAMERA to TOOL)."""

    def __init__(self, rig: Any, scene: Scene, *, camera_to_tool: np.ndarray,
                 tcp: Callable[[], Optional[np.ndarray]]) -> None:
        self.rig = rig
        self.rig_id = str(getattr(rig, "rig_id", "camera"))
        self.scene = scene
        self.camera_to_tool = np.asarray(camera_to_tool, dtype=np.float64)
        self._tcp = tcp
        self._open = False
        self._cache: Optional[tuple[np.ndarray, int, Any, Render]] = None
        self.last: Optional[Render] = None
        self.grabs = 0
        self.renders = 0
        self.render_s = 0.0

    def open(self) -> None:
        self._open = True

    def release(self) -> None:
        self._open = False

    def get_intrinsics(self) -> np.ndarray:
        return self.scene.static.intrinsics.copy()

    def get_distortion(self) -> np.ndarray:
        return np.zeros(5)

    def camera_moved(self) -> None:
        return None

    def grab(self) -> Any:
        from src.camera.setup.image_taking.frames import RGBDFrame  # noqa: PLC0415

        if not self._open:
            raise RuntimeError(f"stand-in camera {self.rig_id!r} is not open")
        tcp = self._tcp()
        if tcp is None:
            raise RuntimeError("the stand-in camera could not read where the TCP stands")
        held = self.scene.held()
        key = (np.round(tcp, 3), self.scene.version, None if held is None else held.name)
        cached = self._cache
        # A D415 render is cached without the grab's own error, which every grab draws afresh, as on the cell (two grabs
        # from one pose differ by about 0.6 mm there); a clean one is handed back as it was.
        d415 = self.scene.depth_model in ("real", "harsh")
        if (cached is not None and np.array_equal(cached[0], key[0]) and cached[1] == key[1] and cached[2] == key[2]):
            rendered = cached[3]
        else:
            rendered = self.scene.render(tcp @ self.camera_to_tool, tcp, temporal=not d415)
            self._cache = (key[0], key[1], key[2], rendered)
            self.renders += 1
            self.render_s += rendered.seconds
        if d415:
            rendered = self.scene.with_grab_error(rendered)
        self.last = rendered
        self.grabs += 1
        colour = colour_of(rendered.depth_mm, rendered.masks, self.scene.target)
        return RGBDFrame(color=colour, depth=rendered.depth_mm.copy())


class StandInBackend:
    """The detector and segmenter: the prompt grounds the scene's target, cut by the render the last grab made."""

    def __init__(self, camera: StandInD415, *, min_pixels: int = 150) -> None:
        self.camera = camera
        self.min_pixels = int(min_pixels)
        self.calls: list[dict[str, Any]] = []
        #: Ground every part the render shows (a bin to clear, "alle Objekte"), not the scene's target alone. Fixed parts
        #: (a bin's walls and floor) and the part in the jaws are never grounded.
        self.all_parts = False

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
        from src.models.detection.types import Detection  # noqa: PLC0415
        from src.models.perception_backend import PerceivedObject  # noqa: PLC0415
        from src.models.segmentation.types import SegmentationResult  # noqa: PLC0415

        rendered = self.camera.last
        scene = self.camera.scene
        target = scene.target
        if self.all_parts and rendered is not None:
            with scene._lock:  # noqa: SLF001 (the scene's own lock, read once)
                names = [n for n, p in scene.parts.items() if not p.fixed and p.held is None]
        else:
            names = [target]
        label = str(prompt).strip() or target
        out = []
        for name in names:
            mask = None if rendered is None else rendered.masks.get(name)
            area = 0 if mask is None else int(mask.sum())
            if mask is None or area < self.min_pixels:
                continue
            rows, cols = np.nonzero(mask)
            x0, x1, y0, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1
            det = Detection(box=[float(x0), float(y0), float(x1), float(y1)], x_center=(x0 + x1) / 2.0,
                            y_center=(y0 + y1) / 2.0, label=label, score=0.9)
            seg = SegmentationResult(
                label=label, score=0.9, bbox_xyxy=(x0, y0, x1, y1), mask=mask.astype(np.uint8), mask_area_px=area,
                centroid_xy=(float(cols.mean()), float(rows.mean())), derived_bbox_xyxy=(x0, y0, x1, y1),
                inference_time_s=0.0, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                frame_id=None, metadata={"stand_in": True, "part": name},
            )
            out.append(PerceivedObject(detection=det, segmentation=seg))
        self.calls.append({"t": time.time(), "prompt": prompt, "target": target, "grounded": len(out)})
        return tuple(out)


@dataclass
class StandInSpec:
    """What ``PerceptionSpec.from_config`` hands the cell build: a spec whose ``build`` is the stand-in backend."""

    backend: StandInBackend
    built: list[int] = field(default_factory=list)

    def build(self) -> StandInBackend:
        self.built.append(1)
        return self.backend
