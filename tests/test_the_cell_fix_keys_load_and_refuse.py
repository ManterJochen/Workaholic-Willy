"""The cell fixes' keys load with their defaults, a declared fixture keeps 10 mm, and the owner's looks ship as depth.

The owner's decisions of 2026-10-01 and 2026-10-02 (fix plan revision 3, Track K):

* four keys, each on by default where the owner chose it: ``robot.grasping.scene_obstacles`` (the calculator sees the
  parts beside the one it grasps), ``robot.grasping.side_approaches`` (tilted grasps join the vertical ones),
  ``robot.safety.planning_world.perceived.support_surfaces`` (the stack finds what it stands on) and
  ``...perceived.support_allowance_mm``, 2 mm, from 0 to 10;
* ``self_collision.min_distance_mm`` below 10 is refused at load while a fixture is declared and the planning world
  includes it: what a camera sees within 5 mm of a declared fixture leaves the world as the fixture's and nothing
  holds that band as solid (S8). The owner's cell declares none, and its 5 loads;
* two of the owner's recorded looks, depth only: the owner allowed the depth to ship and not the colour images.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import yaml
from pydantic import ValidationError

from src.config import load_robot_config
from src.config.loader import available_profiles
from src.config.schema.robot import RobotConfig
from src.config.schema.robot import safety_schema
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.config.schema.robot.safety_schema import PerceivedWorldConfig, RobotSafetyConfig
from src.robot.safety.planning.perceived import DECLARED_SURFACE_MM
from tests import _cell_2026_10_01 as cell

_OWNER_TREE = Path("D:/helper_cell/robominds_config/config/robot/robot.yaml")
_DATA = Path(__file__).resolve().parent / "data" / "cell_2026_10_01"
_LOOK_0_DEG = (-87.50, -64.51, -85.79, -129.82, 90.27, -108.84)


def _safety(min_distance_mm: float, fixtures: list[dict[str, object]], *, include: bool = True) -> RobotSafetyConfig:
    return RobotSafetyConfig.model_validate({
        "self_collision": {"min_distance_mm": min_distance_mm, "fixtures": fixtures},
        "planning_world": {"include_fixtures": include},
    })


_WALL: dict[str, object] = {"name": "bin_wall", "center_mm": [0.0, -600.0, 50.0],
                            "half_extents_mm": [100.0, 5.0, 50.0]}


class TheFourKeysLoadWithTheirDefaultsTests(unittest.TestCase):
    def test_every_key_is_on_where_the_owner_chose_it(self) -> None:
        robot = load_robot_config(profile=None)
        perceived = robot.safety.planning_world.perceived
        self.assertIs(robot.grasping.scene_obstacles, True)
        self.assertIs(robot.grasping.side_approaches, True)
        self.assertIs(perceived.support_surfaces, True)
        self.assertEqual(perceived.support_allowance_mm, 2.0)

    def test_each_key_can_be_switched(self) -> None:
        grasping = RobotGraspingConfig.model_validate({"scene_obstacles": False, "side_approaches": False})
        self.assertIs(grasping.scene_obstacles, False)
        self.assertIs(grasping.side_approaches, False)
        perceived = PerceivedWorldConfig.model_validate({"support_surfaces": False, "support_allowance_mm": 0.0})
        self.assertIs(perceived.support_surfaces, False)
        self.assertEqual(perceived.support_allowance_mm, 0.0)
        self.assertEqual(PerceivedWorldConfig.model_validate({"support_allowance_mm": 10.0}).support_allowance_mm, 10.0)

    def test_an_allowance_outside_0_to_10_is_refused(self) -> None:
        for bound in (0.0, 10.0):
            self.assertEqual(PerceivedWorldConfig.model_validate({"support_allowance_mm": bound}).support_allowance_mm,
                             bound)
        for refused in (-0.1, 10.5, 50.0):
            with self.subTest(allowance=refused), self.assertRaises(ValidationError) as caught:
                PerceivedWorldConfig.model_validate({"support_allowance_mm": refused})
            self.assertIn("support_allowance_mm", str(caught.exception))


class ADeclaredFixtureKeepsTenMillimetresTests(unittest.TestCase):
    def test_5_with_a_fixture_the_planning_world_includes_is_refused_naming_both_keys(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _safety(5.0, [_WALL])
        said = str(caught.exception)
        self.assertIn("safety.self_collision.min_distance_mm (5)", said)
        self.assertIn("safety.planning_world.include_fixtures", said)
        self.assertIn("bin_wall", said)

    def test_just_under_10_is_refused_and_10_loads(self) -> None:
        with self.assertRaises(ValidationError):
            _safety(9.9, [_WALL])
        self.assertEqual(_safety(10.0, [_WALL]).self_collision.min_distance_mm, 10.0)

    def test_the_owners_cell_declares_no_fixture_and_its_5_loads(self) -> None:
        self.assertEqual(_safety(5.0, []).self_collision.min_distance_mm, 5.0)

    def test_a_fixture_kept_out_of_the_planning_world_does_not_refuse(self) -> None:
        self.assertEqual(_safety(5.0, [_WALL], include=False).self_collision.min_distance_mm, 5.0)

    def test_a_fixture_with_a_zero_extent_declares_nothing(self) -> None:
        off: dict[str, object] = dict(_WALL, half_extents_mm=[0.0, 0.0, 0.0])
        self.assertEqual(_safety(5.0, [off]).self_collision.min_distance_mm, 5.0)

    def test_the_least_is_the_declared_band_plus_the_5_mm_the_guard_keeps(self) -> None:
        self.assertEqual(safety_schema._DECLARED_FIXTURE_LEAST_MM, DECLARED_SURFACE_MM + 5.0)


class EveryShippedProfileStillLoadsTests(unittest.TestCase):
    def test_every_profile_layer_loads(self) -> None:
        for layer in (None, *available_profiles()):
            with self.subTest(profile=layer):
                robot = load_robot_config(profile=layer)
                self.assertIs(robot.safety.planning_world.perceived.support_surfaces, True)

    def test_the_owners_robot_profile_loads(self) -> None:
        if not _OWNER_TREE.is_file():
            raise unittest.SkipTest("the owner's tree is not on this machine")
        data = yaml.safe_load(_OWNER_TREE.read_text(encoding="utf-8"))["robot"]
        robot = RobotConfig.model_validate(data)
        # 3 mm for everything since the owner's decision of 2026-10-05; 5 mm before.
        self.assertEqual(robot.safety.self_collision.min_distance_mm, 3.0)
        self.assertIs(robot.safety.planning_world.perceived.support_surfaces, True)
        self.assertIs(robot.grasping.side_approaches, True)


class TheOwnersLooksShipAsDepthOnlyTests(unittest.TestCase):
    def test_each_file_is_depth_and_poses_under_150_kb(self) -> None:
        files = sorted(_DATA.glob("*.npz"))
        self.assertEqual(len(files), 2)
        for path in files:
            with self.subTest(file=path.name):
                self.assertLess(path.stat().st_size, 150 * 1024)
                with np.load(path) as data:
                    for key in data.files:
                        self.assertNotIn("rgb", key)
                        array = data[key]
                        colour = array.ndim == 3 and array.shape[-1] in (3, 4) and array.dtype == np.uint8
                        self.assertFalse(colour, f"{path.name}: {key} looks like a colour image")

    def test_each_look_loads_with_finite_intrinsics_and_rigid_transforms(self) -> None:
        for name in cell.LOOKS:
            with self.subTest(look=name):
                look = cell.look(name)
                self.assertEqual(look.surface_depth_mm.shape, (720, 1280))
                self.assertTrue(np.all(np.isfinite(look.surface_depth_mm)))
                self.assertGreater(float(np.mean(look.surface_depth_mm > 0.0)), 0.8)
                self.assertTrue(np.all(np.isfinite(look.intrinsics)))
                self.assertGreater(look.intrinsics[0, 0], 0.0)
                self.assertGreater(look.intrinsics[1, 1], 0.0)
                for transform in (look.camera_to_base, look.tool_to_base):
                    self.assertTrue(np.all(np.isfinite(transform)))
                    np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-6)
                    self.assertAlmostEqual(float(np.linalg.det(transform[:3, :3])), 1.0, places=6)
                    np.testing.assert_allclose(transform[3], (0.0, 0.0, 0.0, 1.0))
                for got, want in zip(look.joints_deg, _LOOK_0_DEG):
                    self.assertTrue(math.isclose(got, want, abs_tol=0.06))
                self.assertGreater(look.target_points_base_mm.shape[0], 8000)
                self.assertTrue(np.all(np.isfinite(look.target_points_base_mm)))
                self.assertEqual(look.target_mask.shape, look.surface_depth_mm.shape)
                self.assertIs(look.depth_mm, look.surface_depth_mm)

    def test_the_world_reads_the_recorded_pixels(self) -> None:
        """At stride 2 every pixel read is the camera's, and the three beside it repeat it."""
        look = cell.look("P1")
        with np.load(_DATA / "p1_pick-63e1b3dac29f.npz") as data:
            stored = data["surface_depth_strided_mm"].astype(np.float64)
        np.testing.assert_array_equal(look.surface_depth_mm[::2, ::2], stored)
        np.testing.assert_array_equal(look.surface_depth_mm[1::2, 1::2], stored)


if __name__ == "__main__":
    unittest.main()
