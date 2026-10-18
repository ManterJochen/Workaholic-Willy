"""The cell fixes' keys are every cell's, a declared fixture is the guard's, and the owner's looks ship as depth.

The owner's decisions of 2026-10-01 and 2026-10-02 (fix plan revision 3, Track K):

* four keys the owner chose on, refused at load since 2026-10-17 at the values every cell runs since:
  ``robot.grasping.scene_obstacles`` (the calculator sees the parts beside the one it grasps) and
  ``robot.grasping.side_approaches`` (tilted grasps join the vertical ones), with ten more of the cell's grasping
  switches, and ``robot.safety.planning_world.perceived.support_surfaces`` and ``...perceived.support_allowance_mm``:
  the stack finds what it stands on and holds it 2 mm over its band on every cell;
* ``self_collision.min_distance_mm`` below 10 was refused at load while a fixture was declared and the planning world
  included it: what a camera saw within 5 mm of a declared fixture left the world as the fixture's and nothing held
  that band as solid (S8). A fixture stays the guard's on every cell since 2026-10-17 (the camera world drops nothing
  as one), so the rule went with the key, and a fixture loads beside any distance the guard keeps;
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
from src.config.schema._removed import removed_key_sentence
from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.config.schema.robot.safety_schema import RobotSafetyConfig
from src.robot.safety.planning.perceived import SUPPORT_ALLOWANCE_MM
from tests import _cell_2026_10_01 as cell

_OWNER_TREE = Path("D:/helper_cell/robominds_config/config/robot/robot.yaml")
_DATA = Path(__file__).resolve().parents[3] / "data" / "cell_2026_10_01"
_LOOK_0_DEG = (-87.50, -64.51, -85.79, -129.82, 90.27, -108.84)


_WALL: dict[str, object] = {"name": "bin_wall", "center_mm": [0.0, -600.0, 50.0],
                            "half_extents_mm": [100.0, 5.0, 50.0]}


class TheSupportsAreEveryCellsTests(unittest.TestCase):
    def test_every_cell_holds_its_supports_2_mm_over_their_band(self) -> None:
        # Every cell's camera world finds its supports and holds them 2 mm over their band (the owner's cleanup).
        self.assertEqual(SUPPORT_ALLOWANCE_MM, 2.0)

    def test_a_tree_that_still_tunes_the_supports_is_refused(self) -> None:
        for key, value in (("support_surfaces", False), ("support_allowance_mm", 10.0)):
            with self.subTest(key=key), self.assertRaises(ValidationError) as caught:
                RobotSafetyConfig.model_validate({"planning_world": {"perceived": {key: value}}})
            self.assertIn("perceived", str(caught.exception))


class TheCellsGraspingSwitchesAreRefusedTests(unittest.TestCase):
    """The twelve grasping switches that left on 2026-10-17 at the values the owner's cell ran (the owner's cleanup)."""

    GONE = ("max_attempts", "scene_obstacles", "side_approaches", "inner_grasps", "first_good_part", "good_part_score",
            "colour_check", "colour_check_clipped", "name_neighbours", "weak_look_trigger", "hide_own_places",
            "follow_parts")

    def test_each_is_refused_at_load_with_what_every_cell_does(self) -> None:
        for key in self.GONE:
            with self.subTest(key=key):
                self.assertNotIn(key, RobotGraspingConfig.model_fields)
                with self.assertRaises(ValidationError):
                    RobotGraspingConfig.model_validate({key: True})
                self.assertIsNotNone(removed_key_sentence(f"robot.grasping.{key}"))

    def test_a_tree_that_writes_one_does_not_load_and_says_why(self) -> None:
        from src.config.tree import ConfigTree

        refused = ConfigTree.from_directory(profile=None).load().with_values({"robot.grasping.max_attempts": 5})
        self.assertFalse(refused.ok)
        self.assertIn("removed on purpose", refused.error)
        self.assertIn("MAX_ATTEMPTS", refused.error)


class ADeclaredFixtureIsTheGuardsTests(unittest.TestCase):
    def test_a_fixture_loads_beside_any_distance_the_guard_keeps(self) -> None:
        """No planning world takes a declared fixture since 2026-10-17, so no band about one leaves the camera world."""
        for distance in (5.0, 9.9, 10.0):
            with self.subTest(distance=distance):
                safety = RobotSafetyConfig.model_validate({"self_collision": {
                    "min_distance_mm": distance, "perceived_min_distance_mm": 4.0, "fixtures": [_WALL]}})
                self.assertEqual(distance, safety.self_collision.min_distance_mm)

    def test_the_old_switch_is_refused_at_load(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            RobotSafetyConfig.model_validate({"planning_world": {"include_fixtures": True}})
        self.assertIn("include_fixtures", str(caught.exception))


class EveryShippedProfileStillLoadsTests(unittest.TestCase):
    def test_every_profile_layer_loads(self) -> None:
        for layer in (None, *available_profiles()):
            with self.subTest(profile=layer):
                load_robot_config(profile=layer)

    def test_the_owners_robot_profile_loads(self) -> None:
        if not _OWNER_TREE.is_file():
            raise unittest.SkipTest("the owner's tree is not on this machine")
        data = yaml.safe_load(_OWNER_TREE.read_text(encoding="utf-8"))["robot"]
        robot = RobotConfig.model_validate(data)
        # 3 mm for everything since the owner's decision of 2026-10-05; 5 mm before.
        self.assertEqual(robot.safety.self_collision.min_distance_mm, 3.0)


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
