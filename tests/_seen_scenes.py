"""Scenes a camera over the bench sees, ray cast through a pinhole, for the tests of the camera world's boxes.

Every solid is an upright box turned about BASE Z, so a bin is its four walls and its floor and a part lies where it
fell. The camera hangs over the bench looking straight down: camera +x is BASE +x, camera +y is BASE -y, camera +z is
BASE -z. A pixel's depth is the distance along the camera's axis to the first solid its ray meets, or to the bench at
z = 0, which is the support plane of every world built here. Nothing in this file decides anything; it draws.

A real camera also sees the robot, and the robot hides what stands behind it: ``render(..., robot=...)`` draws the
arm's committed collision meshes, and the hand's, into the same frame (:func:`robot_triangles`). A frame drawn
without them is a camera that sees through the arm, which no camera does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.safety.planning.perceived import PerceivedBox, WorldBuildLimits

#: A 640 x 480 pinhole with a 500 px focal length: 1.8 mm a pixel at 900 mm.
WIDTH, HEIGHT, FOCAL = 640, 480, 500.0
K = np.array([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
#: The UR10's reach about its base, the bench at z = 0.
LIMITS = WorldBuildLimits(x_mm=(-760.0, 760.0), y_mm=(-760.0, 760.0), z_mm=(-50.0, 748.0), support_plane_top_mm=0.0)


@dataclass(frozen=True)
class Solid:
    """An upright box turned about BASE Z: its centre, its half sizes along its own axes, and its turn."""

    centre: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw_deg: float = 0.0

    def local(self, points_mm: np.ndarray) -> np.ndarray:
        """``(N, 3)`` BASE points in this solid's own frame."""
        points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) - np.asarray(self.centre)
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1],
                                points[:, 2]))

    def distance_mm(self, points_mm: np.ndarray) -> np.ndarray:
        """How far each point lies from this solid, 0 inside."""
        outside = np.maximum(np.abs(self.local(points_mm)) - np.asarray(self.half), 0.0)
        return np.linalg.norm(outside, axis=1)


def looking_down(x_mm: float, y_mm: float, height_mm: float = 900.0) -> np.ndarray:
    """CAMERA to BASE for a camera at ``height_mm`` over (x, y), looking straight down."""
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = (x_mm, y_mm, height_mm)
    return transform


def looking_at(eye_mm: "tuple[float, float, float]", target_mm: "tuple[float, float, float]") -> np.ndarray:
    """CAMERA to BASE for a camera at ``eye_mm`` looking at ``target_mm``, its image rows running down the view."""
    eye, target = np.asarray(eye_mm, dtype=np.float64), np.asarray(target_mm, dtype=np.float64)
    forward = (target - eye) / np.linalg.norm(target - eye)
    up = np.array([0.0, 0.0, 1.0]) if abs(float(forward[2])) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    transform[:3, 3] = eye
    return transform


@lru_cache(maxsize=32)
def _robot_triangles(model: str, joints_deg: "tuple[float, ...]", hand: Any) -> np.ndarray:
    from src.robot.safety.planning import self_envelope as envelope
    from src.robot.safety.planning.environment import collision_mesh_bundle, hand_mesh_bundle

    frames = envelope.yawed_link_transforms_mm(model, np.radians(np.asarray(joints_deg, dtype=np.float64)), 0.0)
    assert frames is not None
    placed: list[np.ndarray] = []
    with np.load(collision_mesh_bundle(model)) as bundle:
        for name, frame in envelope._ARM_LINKS:  # noqa: SLF001 (the links the bundle bakes, in their DH frames)
            vertices = bundle[f"{name}__v"].astype(np.float64) @ frames[frame][:3, :3].T + frames[frame][:3, 3]
            placed.append(vertices[bundle[f"{name}__f"].astype(np.int64)])
    if hand is not None:
        # As the guard places the hand: one plate out along the model's approach where the bundle starts at the
        # mounting face, turned by the declared tool frame's rotation, on the flange.
        with np.load(str(hand_mesh_bundle(hand.model))) as bundle:
            origin = str(np.asarray(bundle[envelope._ORIGIN_KEY]).reshape(-1)[0]) if envelope._ORIGIN_KEY in bundle.files else ""  # noqa: SLF001
            shift = np.asarray(envelope._MODEL_APPROACH, dtype=np.float64) * float(hand.coupling_mm)  # noqa: SLF001
            flange = frames[envelope._FLANGE_FRAME]  # noqa: SLF001
            for part in envelope._HAND_PARTS:  # noqa: SLF001
                vertices = bundle[f"{part}__v"].astype(np.float64) + (shift if origin == envelope._MOUNTING_FACE else 0.0)  # noqa: SLF001
                vertices = vertices @ np.asarray(hand.placement.rotation, dtype=np.float64).T
                vertices = vertices @ flange[:3, :3].T + flange[:3, 3]
                placed.append(vertices[bundle[f"{part}__f"].astype(np.int64)])
    return np.concatenate(placed)


def robot_triangles(joints_deg: "tuple[float, ...]", *, model: str = "ur10", hand: Any = None) -> np.ndarray:
    """``(T, 3, 3)`` BASE millimetres: the arm's committed collision meshes at ``joints_deg``, placed as the guard places
    them, and the hand's on its flange where a ``PlannerHand`` is given."""
    return _robot_triangles(model, tuple(float(v) for v in joints_deg), hand)


def robot_depth(camera_to_base: np.ndarray, triangles: np.ndarray, *, chunk: int = 400_000) -> np.ndarray:
    """Depth along each pixel's ray to the nearest of ``triangles``, ``inf`` where none: every triangle in front of the
    camera, each against the pixels of its own bounding box (Moller-Trumbore). Checked on 2026-09-30 against Open3D's
    ray caster on the UR10 at the tests' pose: the same 53,931 pixels, within 0.004 mm."""
    camera = np.asarray(camera_to_base, dtype=np.float64)
    local = (np.asarray(triangles, dtype=np.float64).reshape(-1, 3) - camera[:3, 3]) @ camera[:3, :3]
    tri = local.reshape(-1, 3, 3)
    tri = tri[np.all(tri[:, :, 2] > 1.0, axis=1)]
    u = FOCAL * tri[:, :, 0] / tri[:, :, 2] + K[0, 2]
    v = FOCAL * tri[:, :, 1] / tri[:, :, 2] + K[1, 2]
    c0 = np.maximum(np.ceil(u.min(axis=1)), 0).astype(np.int64)
    c1 = np.minimum(np.floor(u.max(axis=1)), WIDTH - 1).astype(np.int64)
    r0 = np.maximum(np.ceil(v.min(axis=1)), 0).astype(np.int64)
    r1 = np.minimum(np.floor(v.max(axis=1)), HEIGHT - 1).astype(np.int64)
    keep = (c1 >= c0) & (r1 >= r0)
    tri, c0, r0 = tri[keep], c0[keep], r0[keep]
    widths = c1[keep] - c0 + 1
    counts = widths * (r1[keep] - r0 + 1)
    ends = np.cumsum(counts)
    depth = np.full((HEIGHT, WIDTH), np.inf)
    start = 0
    while start < tri.shape[0]:
        stop = max(start + 1, int(np.searchsorted(ends, (ends[start - 1] if start else 0) + chunk, side="right")))
        ids = np.arange(start, stop)
        n = counts[ids]
        idx = np.repeat(ids, n)
        offset = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        col, row = c0[idx] + offset % widths[idx], r0[idx] + offset // widths[idx]
        ray = np.column_stack(((col - K[0, 2]) / FOCAL, (row - K[1, 2]) / FOCAL, np.ones(col.shape[0])))
        a, e1, e2 = tri[idx, 0], tri[idx, 1] - tri[idx, 0], tri[idx, 2] - tri[idx, 0]
        p = np.cross(ray, e2)
        det = np.einsum("ij,ij->i", e1, p)
        usable = np.abs(det) > 1e-12
        inverse = np.where(usable, 1.0 / np.where(usable, det, 1.0), 0.0)
        bu = np.einsum("ij,ij->i", -a, p) * inverse
        q = np.cross(-a, e1)
        bv = np.einsum("ij,ij->i", ray, q) * inverse
        t = np.einsum("ij,ij->i", e2, q) * inverse
        hit = usable & (bu >= 0.0) & (bv >= 0.0) & (bu + bv <= 1.0) & (t > 0.0)
        np.minimum.at(depth, (row[hit], col[hit]), t[hit])
        start = stop
    return depth


def render(camera_to_base: np.ndarray, solids: "list[Solid] | tuple[Solid, ...]", *,
           robot: "np.ndarray | None" = None) -> np.ndarray:
    """Depth along each ray to the nearest solid or the bench (the slab method in each solid's own frame), and to the
    robot's ``(T, 3, 3)`` triangles where given (:func:`robot_triangles`), which hide what stands behind them."""
    depth = _render_solids(camera_to_base, solids)
    if robot is None:
        return depth
    arm = robot_depth(camera_to_base, robot)
    return np.where(arm < np.where(depth > 0.0, depth, np.inf), arm, depth)


def _render_solids(camera_to_base: np.ndarray, solids: "list[Solid] | tuple[Solid, ...]") -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(cols - K[0, 2]) / FOCAL, (rows - K[1, 2]) / FOCAL, np.ones_like(cols)], axis=-1)
    rays = rays @ np.asarray(camera_to_base, dtype=np.float64)[:3, :3].T
    origin = np.asarray(camera_to_base, dtype=np.float64)[:3, 3]
    down = rays[..., 2] < 0.0
    depth = np.where(down, -origin[2] / np.where(down, rays[..., 2], -1.0), np.inf)
    for solid in solids:
        c, s = math.cos(math.radians(solid.yaw_deg)), math.sin(math.radians(solid.yaw_deg))
        start = origin - np.asarray(solid.centre, dtype=np.float64)
        start = np.array([c * start[0] + s * start[1], -s * start[0] + c * start[1], start[2]])
        way = np.stack([c * rays[..., 0] + s * rays[..., 1], -s * rays[..., 0] + c * rays[..., 1], rays[..., 2]],
                       axis=-1)
        half = np.asarray(solid.half, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_low = (-half - start) / way
            t_high = (half - start) / way
        near = np.nanmax(np.minimum(t_low, t_high), axis=-1)
        far = np.nanmin(np.maximum(t_low, t_high), axis=-1)
        depth = np.where((far >= near) & (near > 0.0) & (near < depth), near, depth)
    return np.where(np.isfinite(depth), depth, 0.0)


def open_bin(centre_xy: tuple[float, float], size_xy: tuple[float, float], rim_mm: float, *, yaw_deg: float = 0.0,
             wall_mm: float = 5.0, floor_mm: float = 3.0) -> list[Solid]:
    """A bin with its walls ``wall_mm`` thick up to ``rim_mm``, and its floor ``floor_mm`` thick, as five solids."""
    half_x, half_y = size_xy[0] / 2.0, size_xy[1] / 2.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def at(u: float, v: float, z: float) -> tuple[float, float, float]:
        return (centre_xy[0] + c * u - s * v, centre_xy[1] + s * u + c * v, z)

    wall = wall_mm / 2.0
    rim = rim_mm / 2.0
    return [
        Solid(at(-half_x + wall, 0.0, rim), (wall, half_y, rim), yaw_deg),
        Solid(at(half_x - wall, 0.0, rim), (wall, half_y, rim), yaw_deg),
        Solid(at(0.0, -half_y + wall, rim), (half_x, wall, rim), yaw_deg),
        Solid(at(0.0, half_y - wall, rim), (half_x, wall, rim), yaw_deg),
        Solid(at(0.0, 0.0, floor_mm / 2.0), (half_x, half_y, floor_mm / 2.0), yaw_deg),
    ]


def seen_points(camera_to_base: np.ndarray, depth: np.ndarray, *, above_mm: float = 5.0) -> np.ndarray:
    """Every pixel's surface point in BASE that stands above the bench band, unthinned."""
    rows, cols = np.nonzero(depth > 0.0)
    z = depth[rows, cols]
    camera = np.column_stack(((cols - K[0, 2]) * z / FOCAL, (rows - K[1, 2]) * z / FOCAL, z))
    base = camera @ np.asarray(camera_to_base)[:3, :3].T + np.asarray(camera_to_base)[:3, 3]
    return base[base[:, 2] > above_mm]


def box_local(box: PerceivedBox, points_mm: np.ndarray) -> np.ndarray:
    """``(N, 3)`` BASE points in a perceived box's own frame, its centre the origin."""
    points = np.asarray(points_mm, dtype=np.float64).reshape(-1, 3) - np.asarray(box.center_mm)
    c, s = math.cos(box.yaw_rad), math.sin(box.yaw_rad)
    return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1], points[:, 2]))


def clearance_inside(box: PerceivedBox, points_mm: np.ndarray) -> np.ndarray:
    """How deep each point lies inside the turned box the planner receives, the least over its six faces; below 0 is
    outside."""
    return np.min(np.asarray(box.dims_mm) / 2.0 - np.abs(box_local(box, points_mm)), axis=1)


def covered_by(boxes: "tuple[PerceivedBox, ...]", points_mm: np.ndarray) -> list[str]:
    """The boxes any of ``points_mm`` lies inside."""
    return [box.name for box in boxes if bool((clearance_inside(box, points_mm) > 0.0).any())]


def deepest(boxes: "tuple[PerceivedBox, ...]", points_mm: np.ndarray) -> np.ndarray:
    """Per point, how deep it lies inside the box that holds it deepest; below 0 where no box holds it."""
    if not boxes:
        return np.full(np.asarray(points_mm).reshape(-1, 3).shape[0], -np.inf)
    return np.max(np.stack([clearance_inside(box, points_mm) for box in boxes]), axis=0)
