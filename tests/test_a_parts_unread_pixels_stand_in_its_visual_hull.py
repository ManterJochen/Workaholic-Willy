"""A part's mask pixels whose depth the support-footprint stage does not read stand in its visual hull (2026-10-09).

With the rim the cell cuts (``robot.grasping.geometry.footprint_rim_mm``, 2.0 by default since the owner's evening of
2026-10-09), every mask pixel SFE does not read, the rim, the depth steps and the holes, gets a point where its ray
meets the part's top, or just over its support, wherever that point's vertical projects into the mask
(``generation/footprint_hull.py``). On the cell's 23 recorded cubes that left the centre 0.58 mm off at the median and
1.79 at the 90th percentile against the rim with ``inflate_mm`` 1.25's 0.77 and 2.26, and 0.59 and 1.87 with a third
of the depth gone (0.87 and 2.28).

The part is a 40 mm cube ray-cast exactly, mask and depth, by the owner's D415 colour lens (fx 913.7, 1280 x 720)
680 mm away, tilted 36 or 45 degrees, as ``tests/test_a_parts_rim_is_cut_off_its_footprint_alone.py`` casts it
without its ramp. Holes are discs knocked into its depth, 3 to 8 px across, from a fixed seed.
"""

from __future__ import annotations

import functools
import logging
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cv2
import numpy as np

from src.geometry import Frame
from src.robot.grasping.collision import SupportPlane
from src.robot.grasping.generation import calculator as calculator_module
from src.robot.grasping.generation.footprint_hull import HULL_LIFT_MM, hull_points, unread_pixels
from src.robot.grasping.generation.support_footprint import reconstruct_support_prism

_W, _H, _F = 1280, 720, 913.7
_K = np.array([[_F, 0.0, 640.0], [0.0, _F, 360.0], [0.0, 0.0, 1.0]])
_CENTRE = np.array([500.0, -300.0])
_SIZE = 40.0
_FLOOR = 2.0


@functools.lru_cache(maxsize=None)
def _hande() -> Any:
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


def _camera_to_base(tilt_deg: float, distance_mm: float = 680.0) -> np.ndarray:
    """A camera ``distance_mm`` from the cube's centre, ``tilt_deg`` off straight down, looking from -y toward +y."""
    tilt = np.radians(tilt_deg)
    optical = np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([right, np.cross(optical, right), optical])
    matrix[:3, 3] = np.array([_CENTRE[0], _CENTRE[1], _SIZE / 2.0]) - distance_mm * optical
    return matrix


def _ray_cast(camera_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cube's exact mask and the depth of every pixel, cube or table, millimetres along the optical axis."""
    rotation, origin = camera_to_base[:3, :3], camera_to_base[:3, 3]
    u, v = np.meshgrid(np.arange(_W), np.arange(_H))
    rays = np.stack([(u - _K[0, 2]) / _F, (v - _K[1, 2]) / _F, np.ones(u.shape)], axis=-1) @ rotation.T
    low = np.array([_CENTRE[0] - _SIZE / 2, _CENTRE[1] - _SIZE / 2, 0.0])
    high = np.array([_CENTRE[0] + _SIZE / 2, _CENTRE[1] + _SIZE / 2, _SIZE])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low, t_high = (low - origin) / rays, (high - origin) / rays
    enter = np.nanmax(np.minimum(t_low, t_high), axis=-1)
    leave = np.nanmin(np.maximum(t_low, t_high), axis=-1)
    cube = (leave >= enter) & (enter > 0.0)
    t = np.where(cube, enter, -origin[2] / rays[..., 2])
    return cube, t * (rays @ rotation[:, 2])


def _holed(mask: np.ndarray, depth: np.ndarray, share: float, seed: int = 7) -> np.ndarray:
    """``depth`` with discs 3 to 8 px across knocked out of the part until ``share`` of its pixels hold none."""
    rng = np.random.default_rng(seed)
    out = depth.copy()
    rows, cols = np.nonzero(mask)
    while np.count_nonzero(mask & (out <= 0.0)) < share * rows.size:
        index = int(rng.integers(0, rows.size))
        disc = np.zeros(mask.shape, np.uint8)
        cv2.circle(disc, (int(cols[index]), int(rows[index])), int(rng.integers(2, 5)), 1, -1)
        out[disc.astype(bool) & mask] = 0.0
    return out


def _inside_the_cube_by(points: np.ndarray) -> np.ndarray:
    """How far each of ``points`` stands outside the cube's footprint, millimetres (0 inside)."""
    dx = np.maximum(np.abs(points[:, 0] - _CENTRE[0]) - _SIZE / 2, 0.0)
    dy = np.maximum(np.abs(points[:, 1] - _CENTRE[1]) - _SIZE / 2, 0.0)
    return np.hypot(dx, dy)


def _calculator(**keywords: Any) -> Any:
    cfg = _hande()
    return calculator_module.GraspCalculator(
        camera_matrix=_K, max_grip_width_mm=cfg.gripper.max_width_mm, min_grip_width_mm=cfg.gripper.min_width_mm,
        **keywords)


def _compute(calculator: Any, mask: np.ndarray, depth: np.ndarray, camera_to_base: np.ndarray,
             **keywords: Any) -> tuple[dict, np.ndarray]:
    """``(telemetry, the cloud SFE was handed)`` of one compute on the bench at z = 0."""
    handed: list[np.ndarray] = []
    real = calculator_module.support_footprint_breakdowns

    def spy(cloud: Any, **arguments: Any) -> Any:
        handed.append(np.array(cloud, dtype=np.float64))
        return real(cloud, **arguments)

    with mock.patch.object(calculator_module, "support_footprint_breakdowns", spy):
        calculator.compute(
            SimpleNamespace(mask=mask), depth, camera_to_base=camera_to_base,
            support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE), min_table_clearance_mm=1.0,
            **keywords)
    (cloud,) = handed
    return dict(calculator.last_telemetry), cloud


def _sides(cloud: np.ndarray, inflate_mm: float = 0.0) -> tuple[float, float, float]:
    """``(short side, long side, centre off the cube's)`` of the prism SFE builds from ``cloud``, millimetres."""
    prism = reconstruct_support_prism(cloud, 0.0, inflate_mm=inflate_mm)
    assert prism is not None
    short = min(prism.ext_u, prism.ext_v) + 2 * inflate_mm
    long_ = max(prism.ext_u, prism.ext_v) + 2 * inflate_mm
    return short, long_, float(np.linalg.norm(prism.centre - _CENTRE))


def _no_hull() -> Any:
    return mock.patch.object(calculator_module.GraspCalculator, "_with_the_hull",
                             lambda self, target_base, *arguments, **keywords: target_base)


# --------------------------------------------------------------------------- the hull itself


class TheHullPlacesWhatTheMaskShowsTests(unittest.TestCase):
    def test_every_pixel_of_the_part_stands_on_its_footprint(self) -> None:
        for tilt in (36.0, 45.0):
            camera_to_base = _camera_to_base(tilt)
            mask, _depth = _ray_cast(camera_to_base)
            placed = hull_points(mask, mask, _K, camera_to_base, 0.0, _SIZE, floor_margin_mm=_FLOOR)
            with self.subTest(tilt=tilt):
                self.assertGreater(placed.shape[0], 0.9 * np.count_nonzero(mask), "nearly every pixel has a place")
                self.assertLess(float(_inside_the_cube_by(placed).max()), 1.5, "none past the part by a pixel")
                heights = np.round(placed[:, 2], 6)
                low = round(_FLOOR + HULL_LIFT_MM, 6)
                self.assertTrue(set(np.unique(heights)) <= {low, _SIZE}, "at the top, or just over the floor")
                # The placed points span the footprint: each side within a pixel and a half.
                for axis, centre in ((0, _CENTRE[0]), (1, _CENTRE[1])):
                    self.assertLess(abs(float(placed[:, axis].min()) - (centre - _SIZE / 2)), 1.5)
                    self.assertLess(abs(float(placed[:, axis].max()) - (centre + _SIZE / 2)), 1.5)

    def test_a_pixel_off_the_mask_is_never_placed_past_the_part(self) -> None:
        """The table seen past the far edge, offered to be placed: its verticals leave the mask, and nothing grows."""
        camera_to_base = _camera_to_base(45.0)
        mask, _depth = _ray_cast(camera_to_base)
        band = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~mask
        placed = hull_points(mask | band, mask, _K, camera_to_base, 0.0, _SIZE, floor_margin_mm=_FLOOR)
        self.assertLess(float(_inside_the_cube_by(placed).max()), 1.5)
        alone = hull_points(band, mask, _K, camera_to_base, 0.0, _SIZE, floor_margin_mm=_FLOOR)
        self.assertLess(alone.shape[0], 0.05 * np.count_nonzero(band), "the band's own rays stand nowhere on the part")
        self.assertLess(float(_inside_the_cube_by(alone).max()) if alone.shape[0] else 0.0, 1.5)

    def test_unread_pixels_are_the_mask_less_what_was_read(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        read = np.zeros_like(mask)
        read[3:7, 3:7] = True
        np.testing.assert_array_equal(mask & ~read, unread_pixels(mask, read))
        with self.assertRaises(ValueError):
            unread_pixels(mask, read[:5])

    def test_nothing_to_place_or_no_height_to_test_places_nothing(self) -> None:
        camera_to_base = _camera_to_base(45.0)
        mask, _depth = _ray_cast(camera_to_base)
        self.assertEqual((0, 3), hull_points(np.zeros_like(mask), mask, _K, camera_to_base, 0.0, _SIZE,
                                             floor_margin_mm=_FLOOR).shape)
        self.assertEqual((0, 3), hull_points(mask, mask, _K, camera_to_base, 0.0, _FLOOR + 2 * HULL_LIFT_MM,
                                             floor_margin_mm=_FLOOR).shape, "a part no taller than the test")

    def test_images_of_two_sizes_or_a_transform_that_is_none_are_refused(self) -> None:
        camera_to_base = _camera_to_base(45.0)
        mask, _depth = _ray_cast(camera_to_base)
        with self.assertRaises(ValueError):
            hull_points(mask[:100], mask, _K, camera_to_base, 0.0, _SIZE, floor_margin_mm=_FLOOR)
        with self.assertRaises(ValueError):
            hull_points(mask, mask, _K, camera_to_base[:3], 0.0, _SIZE, floor_margin_mm=_FLOOR)


# --------------------------------------------------------------------------- in the calculator


class TheCalculatorPlacesItsUnreadPixelsTests(unittest.TestCase):
    def test_the_footprint_is_the_cube_with_or_without_a_third_of_its_depth(self) -> None:
        for tilt in (36.0, 45.0):
            camera_to_base = _camera_to_base(tilt)
            mask, depth = _ray_cast(camera_to_base)
            for share in (0.0, 0.35):
                holed = depth if share == 0.0 else _holed(mask, depth, share)
                telemetry, cloud = _compute(_calculator(support_footprint_rim_mm=2.0), mask, holed, camera_to_base)
                short, long_, centre = _sides(cloud)
                with self.subTest(tilt=tilt, share=share):
                    self.assertGreater(telemetry["support_footprint_hull_points"], 0)
                    self.assertLess(abs(short - _SIZE), 1.5)
                    self.assertLess(abs(long_ - _SIZE), 1.5)
                    self.assertLess(centre, 1.0)

    def test_without_the_hull_the_holes_shrink_the_footprint_it_keeps(self) -> None:
        """The rim with inflate_mm 1.25 as it was measured before the hull, on the same holes: the short side comes out
        short by more than the hull leaves it."""
        camera_to_base = _camera_to_base(45.0)
        mask, depth = _ray_cast(camera_to_base)
        holed = _holed(mask, depth, 0.35)
        _t, with_hull = _compute(_calculator(support_footprint_rim_mm=2.0), mask, holed, camera_to_base)
        with _no_hull():
            _t, without = _compute(_calculator(support_footprint_rim_mm=2.0, support_footprint_inflate_mm=1.25), mask,
                                   holed, camera_to_base)
        short_hull = _sides(with_hull)[0]
        short_rim = _sides(without, 1.25)[0]
        self.assertLess(abs(short_hull - _SIZE), abs(short_rim - _SIZE))

    def test_the_points_placed_are_stamped_and_said_every_compute(self) -> None:
        camera_to_base = _camera_to_base(45.0)
        mask, depth = _ray_cast(camera_to_base)
        calculator = _calculator(support_footprint_rim_mm=2.0)
        with self.assertLogs(calculator.logger, level=logging.INFO) as said:
            telemetry, cloud = _compute(calculator, mask, _holed(mask, depth, 0.2), camera_to_base)
        with _no_hull():
            _t, read = _compute(_calculator(support_footprint_rim_mm=2.0), mask, _holed(mask, depth, 0.2),
                                camera_to_base)
        self.assertEqual(cloud.shape[0] - read.shape[0], telemetry["support_footprint_hull_points"])
        np.testing.assert_array_equal(read, cloud[:read.shape[0]], "the points read come first, unchanged")
        self.assertTrue(any("footprint hull:" in line and "visual hull" in line for line in said.output), said.output)

    def test_no_rim_places_nothing(self) -> None:
        camera_to_base = _camera_to_base(45.0)
        mask, depth = _ray_cast(camera_to_base)
        telemetry, _cloud = _compute(_calculator(), mask, _holed(mask, depth, 0.2), camera_to_base)
        self.assertNotIn("support_footprint_hull_points", telemetry)

    def test_the_hull_reaches_sfes_input_and_nothing_else(self) -> None:
        """The target's own cloud, the one the jaw faces and the planner world read, is the measured surface alone."""
        camera_to_base = _camera_to_base(45.0)
        mask, depth = _ray_cast(camera_to_base)
        holed = _holed(mask, depth, 0.35)
        with_hull, _cloud = _compute(_calculator(support_footprint_rim_mm=2.0), mask, holed, camera_to_base)
        with _no_hull():
            without, _read = _compute(_calculator(support_footprint_rim_mm=2.0), mask, holed, camera_to_base)
        for key in ("target_cloud_z_min_mm", "target_cloud_z_max_mm", "mask_pixels"):
            with self.subTest(key=key):
                self.assertEqual(without[key], with_hull[key])

    def test_a_part_that_gives_no_prism_gets_no_point(self) -> None:
        """A part with almost no depth left: the hull has no top to place against, and places nothing."""
        camera_to_base = _camera_to_base(45.0)
        mask, depth = _ray_cast(camera_to_base)
        bare = depth.copy()
        bare[mask] = 0.0
        rows, cols = np.nonzero(mask)
        bare[rows[:5], cols[:5]] = depth[rows[:5], cols[:5]]
        calculator = _calculator(support_footprint_rim_mm=2.0)
        with mock.patch.object(calculator_module, "hull_points") as placer:
            calculator.compute(SimpleNamespace(mask=mask), bare, camera_to_base=camera_to_base,
                               support_plane=SupportPlane(np.array([0.0, 0.0, 1.0]), 0.0, Frame.BASE),
                               min_table_clearance_mm=1.0)
        placer.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
