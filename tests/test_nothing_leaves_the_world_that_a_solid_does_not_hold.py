"""Nothing leaves the world that a solid the guard holds does not hold (fix plan Track S, "why 5 mm is safe", test 8).

The rule the support surfaces rest on: a point leaves the obstacles only where a solid the guard and the planner hold
contains every pixel it stands for, the bench's band included (F4; attack H3: per look 130-383 real pixels up to 13 mm
over the bench stood in no box once the band dropped thinned points by their own height). A point with pixels on both
sides stays, standing at its first pixel outside, so the box fitted to it covers what stands outside. So every pixel the
world reads lies in some box the guard holds: a seen box or a solid.

Random scenes of slabs, plates, cubes and pins on a bench, through a D415 at 1280 x 720 straight down; and the owner's
recorded looks.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.core.keep_out import KeepOutBox
from src.robot.safety.planning.perceived import (
    DepthView,
    WorldBuildLimits,
    _base_points,
    _pixels_in_base,
    build_perceived_boxes,
    target_keep_out_box,
)
from tests import _cell_2026_10_01 as cell
from tests.test_support_surfaces_are_found import _K, _down, _render, _turn

_LIMITS = WorldBuildLimits(x_mm=(-450.0, 450.0), y_mm=(-280.0, 280.0), z_mm=(-50.0, 800.0), support_plane_top_mm=0.0)
_TUNING = cell.owner_tuning(support_surfaces=True, support_allowance_mm=2.0)


def _held(boxes: Any, points: np.ndarray) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    for box in boxes:
        local = (points - np.asarray(box.center_mm)) @ box.rotation_matrix
        inside |= np.all(np.abs(local) <= np.asarray(box.dims_mm) / 2.0 + 1e-6, axis=1)
    return inside


def _read(view: DepthView) -> "tuple[np.ndarray, np.ndarray]":
    """Every pixel the world reads of a view, BASE, and which thinned point stands for each."""
    depth = np.asarray(view.surface_depth_mm, dtype=np.float64)
    _, _, _, (rows, cols, owner) = _base_points(depth, view.intrinsics, view.camera_to_base, np.ones(depth.shape, bool),
                                                float(_TUNING.voxel_size_mm), stride=int(_TUNING.pixel_stride))
    points, _ = _pixels_in_base(view, int(_TUNING.pixel_stride), rows, cols)
    return points, owner


def _scene(seed: int) -> list:
    """Slabs up to 3 degrees off level, and plates, cubes and pins on them and on the bench beside them."""
    rng = np.random.default_rng(seed)
    solids = []
    slabs = []
    for _ in range(int(rng.integers(1, 3))):
        size = (float(rng.uniform(240.0, 380.0)), float(rng.uniform(170.0, 230.0)), float(rng.uniform(15.0, 60.0)))
        centre = (float(rng.uniform(-200.0, 200.0)), float(rng.uniform(-60.0, 60.0)), size[2] / 2.0)
        turn = _turn(float(rng.uniform(0.0, 3.0)), float(rng.uniform(-90.0, 90.0)))
        solids.append((centre, size, turn))
        slabs.append((centre, size, turn))
    for _ in range(int(rng.integers(4, 8))):
        on_slab = bool(slabs) and rng.random() < 0.6
        x, y = float(rng.uniform(-380.0, 380.0)), float(rng.uniform(-220.0, 220.0))
        base = 0.0
        if on_slab:
            centre, size, turn = slabs[int(rng.integers(len(slabs)))]
            u, v = rng.uniform(-0.45, 0.45, 2) * np.asarray(size[:2])
            top = np.asarray(centre) + turn @ np.array([u, v, size[2] / 2.0])
            x, y, base = float(top[0]), float(top[1]), float(top[2]) - 1.0
        kind = rng.integers(3)
        if kind == 0:
            side = float(rng.uniform(15.0, 40.0))
            height = float(rng.uniform(2.0, 6.0))
        elif kind == 1:
            side = height = float(rng.uniform(8.0, 40.0))
        else:
            side, height = float(rng.uniform(3.0, 6.0)), float(rng.uniform(10.0, 30.0))
        solids.append(((x, y, base + height / 2.0), (side, side, height), _turn(0.0, float(rng.uniform(0.0, 90.0)))))
    return solids


class EveryPixelLiesInAHeldBoxTests(unittest.TestCase):
    def test_random_scenes_on_slabs_and_on_the_bench(self) -> None:
        camera = _down()
        mixed = 0
        for seed in range(8):
            with self.subTest(seed=seed):
                view = DepthView(surface_depth_mm=_render(camera, boxes=_scene(seed), seed=seed), intrinsics=_K,
                                 camera_to_base=camera, name="camera")
                world = build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING)
                self.assertEqual(world.dropped_obstacle_count, 0)
                points, owner = _read(view)
                inside = ((points[:, 0] >= _LIMITS.x_mm[0]) & (points[:, 0] <= _LIMITS.x_mm[1])
                          & (points[:, 1] >= _LIMITS.y_mm[0]) & (points[:, 1] <= _LIMITS.y_mm[1]))
                free = inside & ~_held(world.boxes, points)
                self.assertEqual(int(free.sum()), 0, f"pixels in no held box, e.g. {points[free][:3].round(1)}")
                # A point with pixels both inside a support's solid and over it stands at its first pixel outside: the
                # box fitted to it holds those outside pixels; the solid does not.
                model = world.supports
                assert model is not None
                swallowed = model.erased_by_support(points) | (points[:, 2] <= model.bench_top_mm + model.bench_band_mm)
                straddles = np.unique(owner[swallowed])
                straddles = np.intersect1d(straddles, np.unique(owner[~swallowed]))
                outside = np.isin(owner, straddles) & ~swallowed & inside
                seen = [box for box in world.boxes if box.kind == "seen"]
                self.assertTrue(bool(_held(seen, points[outside]).all()))
                mixed += int(straddles.size)
        self.assertGreater(mixed, 0, "no scene put a point on both sides of a solid's top")

    def test_the_owners_looks_leave_no_pixel_over_a_top_in_no_box(self) -> None:
        owner_cell = cell.owner_cell()
        for name in cell.LOOKS:
            with self.subTest(look=name):
                look = cell.look(name)
                world = owner_cell.world(look)
                world.offer_segmentation(camera="EIH_Cam", target_points_base_mm=look.target_points_base_mm,
                                         target_label="Einen Zollstock", hold=True)
                snapshot = world.world_for(self_envelope=owner_cell.envelope(np.radians(cell.LOOK_0_DEG)),
                                           near_point_mm=(-165.0, -679.0, 300.0))
                perceived = snapshot.perceived
                assert perceived is not None and perceived.supports is not None
                view = cell.depth_view(look, camera_to_base=look.tool_to_base @ look.camera_to_tool)
                points, _ = _read(view)
                limits = cell.OWNER_LIMITS
                target: KeepOutBox | None = target_keep_out_box(look.target_points_base_mm, name="target",
                                                                limits=limits, tuning=cell.owner_tuning())
                assert target is not None
                over = points[:, 2] > perceived.supports.erasure_top_under(points)
                # The bench's solid stands over the workspace box grown by the world's margin, and a pixel past that is
                # outside the world: asked as far out as the margin reaches, 15 mm until the owner's cell went to 8.
                grace = float(world.tuning.margin_mm)
                inside = ((points[:, 0] >= limits.x_mm[0] - grace) & (points[:, 0] <= limits.x_mm[1] + grace)
                          & (points[:, 1] >= limits.y_mm[0] - grace) & (points[:, 1] <= limits.y_mm[1] + grace))
                ask = over & inside & ~target.contains(points)
                self.assertGreater(int(ask.sum()), 20000)
                free = ask & ~_held(perceived.boxes, points)
                self.assertEqual(int(free.sum()), 0, f"{name}: pixels over a solid's top in no box")


if __name__ == "__main__":
    unittest.main()
