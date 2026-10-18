"""Point clouds of parts and walls in BASE millimetres, and the hands they are grasped with, for the grasp tests.

The scene builders the support-footprint tests shared (taken from ``test_sfe_says_why_it_refused.py`` when SFE's own
search left on 2026-10-16, the owner's cleanup): a part's sampled surfaces as a camera sees them, a wall's face, the
owner's Hand-E as the cell builds it and the library's own jaw.
"""

from __future__ import annotations

import functools

import numpy as np

from src.robot.grasping.collision import ParallelJawGripperModel
from src.robot.grasping.generation.hand import SupportFootprintJaw

#: The mat the owner's parts stand on, as the camera reads it: about 55 mm over the declared bench.
MAT_MM = 55.0


@functools.lru_cache(maxsize=None)
def hande_jaw() -> SupportFootprintJaw:
    """The shipped ``hande`` profile's jaw, the owner's Hand-E, built the way the cell builds it."""
    from src.config.loader import load_robot_section

    return SupportFootprintJaw.from_robot_config(load_robot_section(profile="hande"))


def default_jaw() -> SupportFootprintJaw:
    """The library's own jaw, as ``test_support_footprint_stage.py`` builds it."""
    return SupportFootprintJaw.from_model(ParallelJawGripperModel(), table_clearance_mm=5.0)


def cylinder_cloud(diameter_mm: float, height_mm: float, *, centre_xy: tuple[float, float] = (0.0, 0.0),
                   support_mm: float = 0.0, step_mm: float = 1.5) -> np.ndarray:
    """A standing cylinder's top disc and side, BASE mm."""
    r = diameter_mm / 2.0
    cx, cy = centre_xy
    top = [(cx + x, cy + y, support_mm + height_mm) for x in np.arange(-r, r + 0.1, step_mm)
           for y in np.arange(-r, r + 0.1, step_mm) if x * x + y * y <= r * r]
    side = [(cx + r * np.cos(a), cy + r * np.sin(a), z) for a in np.arange(0.0, 2.0 * np.pi, 1.0 / r)
            for z in np.arange(support_mm + 1.0, support_mm + height_mm, step_mm)]
    return np.asarray(top + side, dtype=np.float64)


def box_cloud(lo: tuple[float, float, float], hi: tuple[float, float, float], *, step_mm: float = 2.0) -> np.ndarray:
    """An axis-aligned box's top and four sides, BASE mm (no bottom: it stands on something)."""
    xs = np.arange(lo[0], hi[0] + 1e-9, step_mm)
    ys = np.arange(lo[1], hi[1] + 1e-9, step_mm)
    zs = np.arange(lo[2], hi[2] + 1e-9, step_mm)
    faces = [np.array([(x, y, hi[2]) for x in xs for y in ys])]
    for x in (lo[0], hi[0]):
        faces.append(np.array([(x, y, z) for y in ys for z in zs]))
    for y in (lo[1], hi[1]):
        faces.append(np.array([(x, y, z) for x in xs for z in zs]))
    return np.vstack(faces).astype(np.float64)


def wall_cloud(y_mm: float, *, x_from: float = -80.0, x_to: float = 80.0, support_mm: float = 0.0,
               height_mm: float = 120.0, step_mm: float = 2.0) -> np.ndarray:
    """A thin vertical wall in the plane ``y = y_mm``, as a camera samples its face."""
    xs = np.arange(x_from, x_to + 0.1, step_mm)
    zs = np.arange(support_mm, support_mm + height_mm + 0.1, step_mm)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack([xx.ravel(), np.full(xx.size, y_mm), zz.ravel()]).astype(np.float64)


def top_face_cloud(height_mm: float, *, step: float = 2.0, extent: float = 15.0) -> np.ndarray:
    """``test_support_footprint_stage.py``'s box top face in BASE mm, as a top-down camera sees it."""
    xs = np.arange(450.0 - extent, 450.0 + extent + 1e-9, step)
    ys = np.arange(-extent, extent + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, height_mm)])


def cylinder_beside_a_wall(gap_mm: float | None = 12.0) -> tuple[np.ndarray, np.ndarray | None]:
    """A cylinder 40 mm across and 60 mm tall, a 120 mm wall ``gap_mm`` off it on +y."""
    cloud = cylinder_cloud(40.0, 60.0)
    return cloud, (None if gap_mm is None else wall_cloud(20.0 + gap_mm))
