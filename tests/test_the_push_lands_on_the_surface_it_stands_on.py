"""The push lands on the surface the part stands on, and its finger keeps clear of that surface's solid (fix plan
Track S, test 4).

Before, the push's table was every point within 5 mm of a level plane through the part's foot, from every look, so on
the owner's look its automatic landing box ran from the mat across the bench into the yellow bin beside it (x -400 to
333, the mat ends at 144; design 5.4). Now the table is the surface the part stands on, its own pixels within the band
of their local reading (F5, attack H7: a quarter of the mat's cells read more than 2 mm under the mat's plane, so a thing
8 mm over such a cell sat within 5 mm of the plane and passed for table). The camera world holds that surface as a solid
the push's own path is judged against, so the finger rides at least at a floor the pick sets from the solid's top
(``plan_push(finger_floor_mm=...)``).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.grasping.recovery.push_gate import PushCell
from src.robot.grasping.recovery.push_planner import (
    FINGER_HEIGHT_MIN_MM,
    MAX_SUPPORT_TILT_DEG,
    TABLE_REGION_PERCENTILE,
    PushPlan,
    plan_push,
    table_points_from_cloud,
)
from src.robot.safety.planning import support_surfaces
from src.robot.safety.planning.self_envelope import ROBOT_BASES
from tests import _cell_2026_10_01 as cell

#: The mat's outline in the owner's looks (E3's measure of the scene), BASE millimetres.
_MAT_X = (-411.3, 143.7)


def _model(look: cell.CellLook) -> support_surfaces.SupportModel:
    return support_surfaces.find_supports([cell.depth_view(look)], limits=cell.OWNER_LIMITS,
                                          tuning=cell.owner_tuning(support_surfaces=True, support_allowance_mm=2.0),
                                          base=ROBOT_BASES["ur10"])


def _box_x(table: np.ndarray, normal: np.ndarray, offset: float) -> "tuple[float, float]":
    """The x span of the push's automatic box over ``table``, as ``plan_push`` takes it, inside the workspace."""
    kept = table_points_from_cloud(table, support_normal=normal, support_offset_mm=offset)
    low = float(np.percentile(kept[:, 0], TABLE_REGION_PERCENTILE))
    high = float(np.percentile(kept[:, 0], 100.0 - TABLE_REGION_PERCENTILE))
    return max(low, cell.OWNER_LIMITS.x_mm[0]), min(high, cell.OWNER_LIMITS.x_mm[1])


def _all_points(look: cell.CellLook) -> np.ndarray:
    rows, cols = np.mgrid[0:720:2, 0:1280:2]
    z = look.surface_depth_mm[::2, ::2]
    ok = z > 0.0
    k, t = look.intrinsics, look.camera_to_base
    camera = np.column_stack(((cols[ok] - k[0, 2]) * z[ok] / k[0, 0], (rows[ok] - k[1, 2]) * z[ok] / k[1, 1], z[ok]))
    return camera @ t[:3, :3].T + t[:3, 3]


class ThePushsTableIsTheSurfaceTests(unittest.TestCase):
    def test_the_landing_box_is_the_mat_and_no_longer_reaches_into_the_bin(self) -> None:
        look = cell.look("P1")
        foot = look.target_points_base_mm
        model = _model(look)
        surface = model.surface_under(foot[:, :2])
        assert surface is not None and surface.offset_mm is not None
        self.assertLessEqual(surface.tilt_deg, MAX_SUPPORT_TILT_DEG)
        self.assertGreater(surface.tilt_deg, 0.8)
        table = model.table_points([cell.depth_view(look)], foot[:, :2])
        low, high = _box_x(table, surface.normal, surface.offset_mm)
        self.assertGreater(low, _MAT_X[0] - 20.0)
        self.assertLess(high, _MAT_X[1] + 20.0)
        # Before: within 5 mm of a level plane through the part's foot, from the whole look.
        foot_z = float(np.percentile(foot[:, 2], 2.0))
        before_low, before_high = _box_x(_all_points(look), np.array([0.0, 0.0, 1.0]), foot_z)
        self.assertGreater(before_high, 250.0, "the scene no longer shows the box reaching into the bin")

    def test_a_thing_8_mm_over_a_low_reading_cell_is_no_table(self) -> None:
        """F5: within the band of the mat's plane, but 8 mm over the mat's local reading there."""
        model = _model(cell.look("P1"))
        mat = model.surfaces[0]
        grid = np.stack(np.meshgrid(np.arange(-400.0, 140.0, 10.0), np.arange(-870.0, -510.0, 10.0)), -1).reshape(-1, 2)
        lowest = None
        for solid in model.support_solids:
            if solid.surface != mat.index:
                continue
            covered = grid[solid.covers_xy(grid)]
            under = mat.height_at(covered) - solid.reading_at(covered)
            if covered.size and (lowest is None or float(under.max()) > lowest[0]):
                index = int(np.argmax(under))
                lowest = (float(under[index]), covered[index], float(solid.reading_at(covered[index:index + 1])[0]))
        assert lowest is not None
        under, xy, reading = lowest
        self.assertGreater(under, 3.0, "no cell of the mat reads far enough under its plane for this scene")
        thing = np.array([[xy[0], xy[1], reading + 8.0]])
        self.assertLessEqual(abs(float(thing[0, 2]) - float(mat.height_at(xy[None, :])[0])), 5.0)
        self.assertFalse(bool(model.is_support_point(thing)[0]))
        self.assertTrue(bool(model.is_support_point(np.array([[xy[0], xy[1], reading + 2.0]]))[0]))


def _cylinder(x: float, y: float, base: float, radius: float, height: float) -> np.ndarray:
    points = []
    for ring in np.arange(0.0, radius + 0.1, 2.0):
        count = max(1, int(2.0 * np.pi * ring / 2.0))
        for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
            points.append((x + ring * np.cos(angle), y + ring * np.sin(angle), base + height))
    for z in np.arange(base + 1.0, base + height, 2.0):
        for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
            points.append((x + radius * np.cos(angle), y + radius * np.sin(angle), z))
    return np.asarray(points)


def _cube(cx: float, cy: float, size: float, base: float) -> np.ndarray:
    xs = np.arange(cx - size / 2.0, cx + size / 2.0 + 1e-9, 2.0)
    ys = np.arange(cy - size / 2.0, cy + size / 2.0 + 1e-9, 2.0)
    top = [(x, y, base + size) for x in xs for y in ys]
    sides = [(x, y, z) for z in np.arange(base + 1.0, base + size, 2.0) for x in xs for y in (ys[0], ys[-1])]
    sides += [(x, y, z) for z in np.arange(base + 1.0, base + size, 2.0) for y in ys for x in (xs[0], xs[-1])]
    return np.asarray(top + sides)


class TheFingerRidesOverTheSolidTests(unittest.TestCase):
    """A 40 mm cylinder 50 mm tall on a level 55 mm mat, a 40 mm cube 15 mm off its -x side: the push the probe found
    (PLAN push_scene_probe: 50 mm along +y, the finger at 20 mm over the mat)."""

    def _plan(self, finger_floor_mm: float = FINGER_HEIGHT_MIN_MM) -> PushPlan:
        cell_cfg = PushCell.from_robot_config(RobotConfig.model_validate(cell.owner_robot()))
        assert isinstance(cell_cfg, PushCell)
        mat = 55.0
        xs, ys = np.meshgrid(np.arange(-450.0, 150.0, 2.5), np.arange(-950.0, -350.0, 2.5))
        table = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, mat)))
        plan = plan_push(target_points_mm=_cylinder(-150.0, -650.0, mat, 20.0, 50.0),
                         neighbour_points_mm=_cube(-150.0 - 20.0 - 15.0 - 20.0, -650.0, 40.0, mat),
                         support_normal=(0.0, 0.0, 1.0), support_offset_mm=mat, workspace=cell_cfg.workspace,
                         hand=cell_cfg.hand, table_points_mm=table, push_distance_mm=50.0, hand_clearance_mm=20.0,
                         finger_floor_mm=finger_floor_mm)
        assert isinstance(plan, PushPlan), plan
        return plan

    def test_the_finger_rides_at_the_floor_where_the_floor_is_higher(self) -> None:
        self.assertAlmostEqual(self._plan().finger_height_mm, 20.0)
        self.assertAlmostEqual(self._plan(finger_floor_mm=23.0).finger_height_mm, 23.0)

    def test_a_floor_under_the_owners_10_mm_changes_nothing(self) -> None:
        self.assertAlmostEqual(self._plan(finger_floor_mm=2.0).finger_height_mm, self._plan().finger_height_mm)
        self.assertEqual(FINGER_HEIGHT_MIN_MM, 10.0)

    def test_a_floor_that_is_no_number_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._plan(finger_floor_mm=float("nan"))

    def test_the_floor_comes_from_the_held_top_of_the_surface_under_the_path(self) -> None:
        """What the pick sets the floor from: the highest top a held solid stands at over the swept path."""
        model = _model(cell.look("P1"))
        path = np.array([[-200.0, -650.0], [-150.0, -650.0], [-100.0, -650.0]])
        tops = [float(s.top_at(path[s.covers_xy(path)]).max()) for s in model.solids if s.covers_xy(path).any()]
        self.assertEqual(model.solid_top_over(path), max(tops))
        mat = [s for s in model.support_solids if s.covers_xy(path).any()]
        self.assertTrue(mat)
        for solid in mat:
            held = solid.reading_at(path) + solid.excess_mm + solid.band_mm + solid.allowance_mm
            covered = solid.covers_xy(path)
            np.testing.assert_allclose(solid.top_at(path)[covered], held[covered])
        bare = np.array([[300.0, -400.0]])
        bench = model.bench_solid
        assert bench is not None
        if not any(s.covers_xy(bare).any() for s in model.support_solids):
            self.assertAlmostEqual(model.solid_top_over(bare) or 0.0, bench.band_mm + bench.allowance_mm)


if __name__ == "__main__":
    unittest.main()
