"""The stack finds what the parts stand on, and nothing else (fix plan Track S, test 2).

A support is a large, nearly level surface the camera sees flat: the owner's foam mat, a bin's floor, the bare bench
read higher than it is declared. At least 100 x 100 mm and 75 mm wide, at most 5 degrees off level. Not a small block, a
hand, the top of the part being picked, nor the robot's own base, whose top as a support would put the shoulder in a
solid. Where two looks disagree, the higher reading wins; where only a low-reading look sees a region, that region is
raised by how low that look reads where both see (F6). A frame's window is never covered by a solid (H8). Off, nothing
of it is in the world.

The scenes are drawn through a D415 at 1280 x 720 (the owner's intrinsics) 800 mm over the bench, and the owner's look P1.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.robot.core.keep_out import KeepOutBox
from src.robot.safety.planning.perceived import DepthView, WorldBuildLimits, build_perceived_boxes
from src.robot.safety.planning.self_envelope import ROBOT_BASES
from src.robot.safety.planning.support_surfaces import MAX_SUPPORT_TILT_DEG, find_supports
from tests import _cell_2026_10_01 as cell

_W, _H = 1280, 720
_K = np.array([[913.7, 0.0, 641.4], [0.0, 912.0, 354.0], [0.0, 0.0, 1.0]])
#: Room about the scenes, the bench at 0 with its 5 mm band.
_LIMITS = WorldBuildLimits(x_mm=(-500.0, 500.0), y_mm=(-500.0, 500.0), z_mm=(-50.0, 800.0), support_plane_top_mm=0.0)
_TUNING = cell.owner_tuning(support_surfaces=True, support_allowance_mm=2.0)


def _down(x: float = 0.0, y: float = 0.0, height: float = 800.0) -> np.ndarray:
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera[:3, 3] = (x, y, height)
    return camera


def _turn(tilt_x_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    a, b = math.radians(tilt_x_deg), math.radians(yaw_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx


def _render(camera: np.ndarray, *, bench_mm: float = 0.0, boxes: Any = (), cylinders: Any = (), noise_mm: float = 0.8,
            seed: int = 0) -> np.ndarray:
    """Depth along each ray to the nearest of ``boxes`` ``(centre, size, rotation)``, upright ``cylinders``
    ``(x, y, radius, z_low, z_high)`` and the bench at ``bench_mm``, with Gaussian noise."""
    v, u = np.mgrid[0:_H, 0:_W]
    rays = np.stack([(u - _K[0, 2]) / _K[0, 0], (v - _K[1, 2]) / _K[1, 1], np.ones(u.shape)], -1).reshape(-1, 3)
    d = rays @ camera[:3, :3].T
    o = camera[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        best = np.where(d[:, 2] < 0.0, (bench_mm - o[2]) / d[:, 2], np.inf)
    for centre, size, turn in boxes:
        rotation = np.eye(3) if turn is None else turn
        lo = (o - np.asarray(centre)) @ rotation
        ld = d @ rotation
        half = np.asarray(size) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - lo) / ld, (half - lo) / ld
        near, far = np.nanmax(np.minimum(t1, t2), axis=1), np.nanmin(np.maximum(t1, t2), axis=1)
        best = np.minimum(best, np.where((far >= near) & (near > 0.0), near, np.inf))
    for x, y, radius, z0, z1 in cylinders:
        ox, oy = o[0] - x, o[1] - y
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2.0 * (ox * d[:, 0] + oy * d[:, 1])
        disc = b * b - 4.0 * a * (ox * ox + oy * oy - radius * radius)
        ok = (disc >= 0.0) & (a > 1e-12)
        t0 = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * np.where(a > 1e-12, a, 1.0))
        z = o[2] + t0 * d[:, 2]
        best = np.minimum(best, np.where(ok & (t0 > 0.0) & (z >= z0) & (z <= z1), t0, np.inf))
        with np.errstate(divide="ignore", invalid="ignore"):
            lid = (z1 - o[2]) / d[:, 2]
        on = (lid > 0.0) & ((ox + lid * d[:, 0]) ** 2 + (oy + lid * d[:, 1]) ** 2 <= radius ** 2)
        best = np.minimum(best, np.where(on, lid, np.inf))
    depth = best.reshape(_H, _W)
    noisy = depth + np.random.default_rng(seed).normal(0.0, noise_mm, depth.shape)
    return np.where(np.isfinite(depth), noisy, 0.0)


def _view(camera: np.ndarray, depth: np.ndarray, **changes: Any) -> DepthView:
    return DepthView(surface_depth_mm=depth, intrinsics=_K, camera_to_base=camera, name="camera", **changes)


def _found(views: Any, *, limits: WorldBuildLimits = _LIMITS, **keywords: Any) -> Any:
    return find_supports(views, limits=limits, tuning=_TUNING, **keywords)


def _over(model: Any, above_mm: float = 10.0) -> list[Any]:
    """The surfaces standing clear of the bench: one read where the bench is declared may or may not be a support, as
    the noise has it, and either is safe (its solid is the bench's height)."""
    return [surface for surface in model.surfaces if surface.z_range_mm[1] > above_mm]


def _one(model: Any, above_mm: float = 10.0) -> Any:
    found = _over(model, above_mm)
    assert len(found) == 1, model.render()
    return found[0]


class TheSupportsAreFoundTests(unittest.TestCase):
    def test_the_owners_mat_is_found_as_the_camera_reads_it(self) -> None:
        for name in cell.LOOKS:
            with self.subTest(look=name):
                model = find_supports([cell.depth_view(cell.look(name))], limits=cell.OWNER_LIMITS, tuning=_TUNING,
                                      base=ROBOT_BASES["ur10"])
                mats = [s for s in model.surfaces if s.z_range_mm[0] > 40.0]
                self.assertEqual(len(mats), 1, model.render())
                self.assertGreaterEqual(mats[0].z_range_mm[0], 50.0)
                self.assertLessEqual(mats[0].z_range_mm[1], 63.0)
                self.assertGreaterEqual(mats[0].tilt_deg, 1.0)
                self.assertLessEqual(mats[0].tilt_deg, 1.2)
                self.assertLessEqual(len(mats[0].solids), 4)

    def test_tilts_of_0_1_and_3_degrees_are_found_as_they_tilt(self) -> None:
        camera = _down()
        for tilt in (0.0, 1.0, 3.0):
            with self.subTest(tilt_deg=tilt):
                depth = _render(camera, boxes=[((0.0, 0.0, 25.0), (400.0, 300.0, 50.0), _turn(tilt, 30.0))], seed=1)
                surface = _one(_found([_view(camera, depth)]))
                self.assertAlmostEqual(surface.tilt_deg, tilt, delta=0.3)
                self.assertLessEqual(surface.tilt_deg, MAX_SUPPORT_TILT_DEG)
                self.assertAlmostEqual(float(surface.height_at([[0.0, 0.0]])[0]), 50.0, delta=1.5)

    def test_a_surface_steeper_than_5_degrees_is_no_support(self) -> None:
        camera = _down()
        depth = _render(camera, boxes=[((0.0, 0.0, 25.0), (400.0, 300.0, 50.0), _turn(8.0))], seed=2)
        self.assertEqual(_over(_found([_view(camera, depth)])), [])

    def test_the_bare_bench_read_8_mm_high_is_a_support(self) -> None:
        camera = _down()
        surface = _one(_found([_view(camera, _render(camera, bench_mm=8.0, seed=3))]), above_mm=5.0)
        self.assertAlmostEqual(float(surface.height_at([[0.0, 0.0]])[0]), 8.0, delta=1.0)

    def test_the_bare_bench_read_low_is_the_benchs(self) -> None:
        camera = _down()
        model = _found([_view(camera, _render(camera, bench_mm=-4.0, seed=4))])
        self.assertEqual(model.surfaces, ())
        self.assertIsNotNone(model.bench_solid)

    def test_a_bins_floor_is_a_support_and_its_walls_stay_obstacles(self) -> None:
        camera = _down()
        mat = ((0.0, 0.0, 27.5), (700.0, 500.0, 55.0), None)
        walls = [((-97.5, 0.0, 105.0), (5.0, 250.0, 100.0), None), ((97.5, 0.0, 105.0), (5.0, 250.0, 100.0), None),
                 ((0.0, -122.5, 105.0), (200.0, 5.0, 100.0), None), ((0.0, 122.5, 105.0), (200.0, 5.0, 100.0), None)]
        floor = ((0.0, 0.0, 56.5), (200.0, 250.0, 3.0), None)
        depth = _render(camera, boxes=[mat, floor, *walls], seed=5)
        view = _view(camera, depth)
        model = _found([view])
        floors = [s for s in model.surfaces if 56.0 < float(s.height_at([[0.0, 0.0]])[0]) < 61.0
                  and s.area_mm2 < 0.06e6]
        self.assertEqual(len(floors), 1, model.render())
        world = build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING)
        seen = [box for box in world.boxes if box.kind == "seen"]
        for centre, size, _ in walls:
            top = np.array([[centre[0], centre[1], centre[2] + size[2] / 2.0 - 1.0]])
            held = [box.name for box in seen if _inside(box, top)]
            self.assertTrue(held, f"the wall at {centre} left the world")


def _inside(box: Any, points: np.ndarray) -> bool:
    local = (points - np.asarray(box.center_mm)) @ box.rotation_matrix
    return bool(np.all(np.abs(local) <= np.asarray(box.dims_mm) / 2.0 + 1e-9, axis=1).any())


class WhatIsNoSupportTests(unittest.TestCase):
    def test_a_block_under_100_by_100_mm_is_no_support(self) -> None:
        camera = _down()
        depth = _render(camera, boxes=[((0.0, 0.0, 40.0), (90.0, 90.0, 80.0), None)], seed=6)
        self.assertEqual(_over(_found([_view(camera, depth)])), [])

    def test_a_hand_sized_patch_is_no_support(self) -> None:
        camera = _down()
        depth = _render(camera, boxes=[((0.0, 0.0, 30.0), (100.0, 85.0, 60.0), None)], seed=7)
        self.assertEqual(_over(_found([_view(camera, depth)])), [])

    def test_the_top_of_the_part_being_picked_is_no_support(self) -> None:
        camera = _down()
        part = ((0.0, 0.0, 30.0), (150.0, 150.0, 60.0), None)
        view = _view(camera, _render(camera, boxes=[part], seed=8))
        self.assertEqual(len(_over(_found([view]))), 1, "the scene does not show what the rule refuses")
        held = KeepOutBox.from_matrix("target", np.eye(4) + _at(0.0, 0.0, 30.0), (90.0, 90.0, 45.0))
        self.assertEqual(_over(_found([view], keep_out=(held,))), [])

    def test_the_robots_base_is_never_a_support(self) -> None:
        """A full disc 95 mm about the base axis at 38 mm (attack S4): found where nothing says it is the base."""
        camera = _down(0.0, -150.0)
        depth = _render(camera, cylinders=[(0.0, 0.0, 95.05, 0.0, 38.0)], seed=9)
        limits = WorldBuildLimits(x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 800.0),
                                  support_plane_top_mm=0.0)
        self.assertEqual(len(_over(_found([_view(camera, depth)], limits=limits))), 1)
        self.assertEqual(_over(_found([_view(camera, depth)], limits=limits, base=ROBOT_BASES["ur10"])), [])


def _at(x: float, y: float, z: float) -> np.ndarray:
    shift = np.zeros((4, 4))
    shift[:3, 3] = (x, y, z)
    return shift


class TwoLooksTests(unittest.TestCase):
    def test_of_two_looks_4_mm_apart_the_higher_reading_wins(self) -> None:
        camera = _down()
        low = camera.copy()
        low[2, 3] -= 4.0  # placed 4 mm too low: everything it sees reads 4 mm low
        slab = ((0.0, 0.0, 25.0), (400.0, 300.0, 50.0), _turn(1.0))
        views = [_view(camera, _render(camera, boxes=[slab], seed=10)), _view(low, _render(camera, boxes=[slab], seed=11))]
        surface = _one(_found(views))
        self.assertAlmostEqual(float(surface.height_at([[0.0, 0.0]])[0]), 50.0, delta=1.5)
        alone = _one(_found(views[1:]))
        self.assertAlmostEqual(float(alone.height_at([[0.0, 0.0]])[0]), 46.0, delta=1.5)

    def test_a_region_only_the_low_look_sees_is_raised_by_how_low_it_reads(self) -> None:
        """F6 (attack H6): the higher look sees the left of the slab, the low look all of it."""
        camera = _down()
        low = camera.copy()
        low[2, 3] -= 4.0
        slab = ((0.0, 0.0, 25.0), (500.0, 300.0, 50.0), None)
        high_depth = _render(camera, boxes=[slab], seed=12)
        v, u = np.mgrid[0:_H, 0:_W]
        right = (u - _K[0, 2]) / _K[0, 0] * high_depth > 40.0  # what lies at x > 40 mm in the higher look
        views = [_view(camera, high_depth, exclude_masks=(right,)), _view(low, _render(camera, boxes=[slab], seed=13))]
        model = _found(views)
        self.assertGreater(model.stats["view_offsets_mm"][1], 3.0)
        solids = [s for s in model.support_solids if s.covers_xy([[200.0, 0.0]])[0]]
        self.assertTrue(solids)
        self.assertAlmostEqual(max(float(s.reading_at([[200.0, 0.0]])[0]) for s in solids), 50.0, delta=1.5)


class AWindowIsNeverCoveredTests(unittest.TestCase):
    def test_a_frames_window_is_under_no_solid_and_what_stands_in_it_stays(self) -> None:
        """H8: a 200 x 250 mm frame with a 50 mm border, 30 mm tall, a 20 mm cube in its window."""
        camera = _down()
        border = [((0.0, -100.0, 15.0), (200.0, 50.0, 30.0), None), ((0.0, 100.0, 15.0), (200.0, 50.0, 30.0), None),
                  ((-75.0, 0.0, 15.0), (50.0, 150.0, 30.0), None), ((75.0, 0.0, 15.0), (50.0, 150.0, 30.0), None)]
        cube = ((0.0, 0.0, 10.0), (20.0, 20.0, 20.0), None)
        view = _view(camera, _render(camera, boxes=[*border, cube], seed=14))
        model = _found([view])
        window = np.array([[0.0, 0.0], [0.0, 30.0], [-30.0, 0.0]])
        # One solid would hold the frame (35 000 mm2), and it would cover the window: so it is not built, and the frame
        # stays obstacles as before, said in the model.
        self.assertTrue(any("window" in why for why in model.left_out), model.render())
        for solid in model.support_solids:
            # The bench under the window may be a support of its own; no solid at the frame's height covers it.
            covers = solid.covers_xy(window) & (solid.reading_at(window) > 10.0)
            self.assertFalse(bool(covers.any()), f"{solid.name} covers the frame's window")
        world = build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING)
        top = np.array([[0.0, 0.0, 19.0]])
        self.assertTrue([box.name for box in world.boxes if box.kind == "seen" and _inside(box, top)])


class OffIsTheWorldOfBeforeTests(unittest.TestCase):
    def test_off_nothing_of_the_supports_is_in_the_world(self) -> None:
        look = cell.look("P1")
        off = build_perceived_boxes(views=[cell.depth_view(look)], limits=cell.OWNER_LIMITS,
                                    tuning=cell.owner_tuning(support_surfaces=False, support_allowance_mm=2.0))
        default = build_perceived_boxes(views=[cell.depth_view(look)], limits=cell.OWNER_LIMITS,
                                        tuning=cell.owner_tuning())
        self.assertIsNone(off.supports)
        self.assertNotIn("supports", off.to_dict())
        self.assertNotIn("support", off.dropped_points)
        self.assertTrue(all(box.kind == "seen" and box.rotation is None for box in off.boxes))
        self.assertEqual(off.to_dict(), default.to_dict())


if __name__ == "__main__":
    unittest.main()
