"""A part on the bench as the calculator is handed it, for the tests of the calculator's own contract.

The calculator plans every grasp in BASE over the support the part stands on (``GraspCalculator.compute``), so a test
of what it carries along, its units, its frames or its filters hands it a frame it can grasp in: a box on the bench at
z = 0, ray cast from straight above through the pinhole of ``tests/_seen_scenes.py``. Nothing here decides anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import SupportPlane
from tests._seen_scenes import K, Solid, looking_down, render


@dataclass(frozen=True, eq=False)
class BenchScene:
    """One frame of a box on the bench: its mask, the depth along the camera's axis (millimetres, 0 where the ray met
    nothing), the lens, where the camera stood and the bench the box stands on, a BASE plane."""

    mask: np.ndarray
    depth_mm: np.ndarray
    camera_matrix: np.ndarray
    camera_to_base: np.ndarray
    support_plane: SupportPlane


@lru_cache(maxsize=None)
def box_on_the_bench(size_mm: "tuple[float, float, float]" = (40.0, 40.0, 40.0), *,
                     at_xy: "tuple[float, float]" = (450.0, 0.0), camera_height_mm: float = 600.0) -> BenchScene:
    """A box ``size_mm`` large standing on the bench at ``at_xy``, seen from ``camera_height_mm`` straight above it."""
    camera = looking_down(at_xy[0], at_xy[1], camera_height_mm)
    half = (size_mm[0] / 2.0, size_mm[1] / 2.0, size_mm[2] / 2.0)
    depth = render(camera, [Solid(centre=(at_xy[0], at_xy[1], half[2]), half=half)])
    bench = render(camera, [])
    mask = (depth > 0.0) & (depth < bench - 1.0)
    return BenchScene(mask=mask, depth_mm=depth, camera_matrix=K.copy(), camera_to_base=camera,
                      support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE))
