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
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

#: A D415 reads no depth closer than this at 1280x720 (Intel's Min-Z at the maximum resolution, about 45 cm): what a
#: wrist camera sees of the mat at P0 or a standoff is mostly holes, as on the cell.
MIN_Z_MM = 450.0
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
        mx0, mx1, my0, my1 = MAT_XY
        if mx0 < xy[0] < mx1 and my0 < xy[1] < my1:
            return self.mat_at(xy), self.mat_normal
        near = np.hypot(self.points[:, 0] - xy[0], self.points[:, 1] - xy[1]) < 20.0
        z = float(np.percentile(self.points[near, 2], 30.0)) if near.any() else 0.0
        return z, np.array([0.0, 0.0, 1.0])


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


@dataclass
class Render:
    depth_mm: np.ndarray
    masks: dict[str, np.ndarray]
    cam_to_base: np.ndarray
    seconds: float


class Scene:
    """The parts on the recorded mat, and the only things that move them: the jaws and a push."""

    def __init__(self, static: StaticScene, *, target: str = "", seed: int = 7) -> None:
        self.static = static
        self.parts: dict[str, Part] = {}
        self.target = target
        self.version = 0
        self.events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
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
        """The mat's reading on a grid over :data:`MAT_XY`, less where a part stands: the table cameras all round would
        see, BASE millimetres."""
        x0, x1, y0, y1 = MAT_XY
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
            part = min(self.parts.values(), key=lambda p: float(np.hypot(*(p.frame[:2, 3] - centre[:2]))))
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
            foot, normal = self.foot_on_mat(xy)
            yaw = math.atan2(float(part.frame[1, 0]), float(part.frame[0, 0]))
            part.frame = standing_frame(foot, normal, yaw)
            self.version += 1
            self.events.append({"t": time.time(), "event": "pushed", "part": part.name, "moved_mm": round(moved, 1),
                                "to_mm": [round(float(v), 1) for v in foot]})

    def render(self, cam_to_base: np.ndarray, tcp: Optional[np.ndarray]) -> Render:
        started = time.perf_counter()
        st = self.static
        k, (h, w) = st.intrinsics, st.shape
        depth = _project_static(st.points, cam_to_base, k, (h, w))
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
        noisy = nearest.copy()
        if masks:
            on_parts = np.any(np.stack(list(masks.values())), axis=0)
            noisy[on_parts] += self._rng.normal(0.0, PART_NOISE_MM, int(on_parts.sum()))
        valid = np.isfinite(noisy) & (noisy >= MIN_Z_MM) & (noisy <= MAX_Z_MM)
        out = np.where(valid, np.rint(noisy), 0.0).astype(np.uint16)
        masks = {name: mask & valid for name, mask in masks.items()}
        return Render(out, masks, cam_to_base, time.perf_counter() - started)


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
        if (cached is not None and np.array_equal(cached[0], key[0]) and cached[1] == key[1] and cached[2] == key[2]):
            rendered = cached[3]
        else:
            rendered = self.scene.render(tcp @ self.camera_to_tool, tcp)
            self._cache = (key[0], key[1], key[2], rendered)
            self.renders += 1
            self.render_s += rendered.seconds
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

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[Any, ...]:
        from src.models.detection.types import Detection  # noqa: PLC0415
        from src.models.perception_backend import PerceivedObject  # noqa: PLC0415
        from src.models.segmentation.types import SegmentationResult  # noqa: PLC0415

        rendered = self.camera.last
        target = self.camera.scene.target
        mask = None if rendered is None else rendered.masks.get(target)
        area = 0 if mask is None else int(mask.sum())
        self.calls.append({"t": time.time(), "prompt": prompt, "target": target, "pixels": area})
        if mask is None or area < self.min_pixels:
            return ()
        rows, cols = np.nonzero(mask)
        x0, x1, y0, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1
        label = str(prompt).strip() or target
        det = Detection(box=[float(x0), float(y0), float(x1), float(y1)], x_center=(x0 + x1) / 2.0,
                        y_center=(y0 + y1) / 2.0, label=label, score=0.9)
        seg = SegmentationResult(
            label=label, score=0.9, bbox_xyxy=(x0, y0, x1, y1), mask=mask.astype(np.uint8), mask_area_px=area,
            centroid_xy=(float(cols.mean()), float(rows.mean())), derived_bbox_xyxy=(x0, y0, x1, y1),
            inference_time_s=0.0, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            frame_id=None, metadata={"stand_in": True},
        )
        return (PerceivedObject(detection=det, segmentation=seg),)


@dataclass
class StandInSpec:
    """What ``PerceptionSpec.from_config`` hands the cell build: a spec whose ``build`` is the stand-in backend."""

    backend: StandInBackend
    built: list[int] = field(default_factory=list)

    def build(self) -> StandInBackend:
        self.built.append(1)
        return self.backend
