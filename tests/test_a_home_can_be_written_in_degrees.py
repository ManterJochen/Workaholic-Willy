"""A cell's home can be written in degrees, as the pendant shows it, and every consumer still reads radians.

The owner, 2026-09-24: ``home_joint_positions`` may also be given in degrees. The pendant and
``python -m src.robot.drivers.ur --where`` show degrees, and the UR driver's home verb, the joint window centred on the
home and the sim arm all read radians. So ``robot.home_joint_positions_deg`` (and its sim twin) is read once, at load,
into ``home_joint_positions`` in radians, and both at once are refused rather than one silently winning.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.config.loader import ConfigError
from src.config.schema.robot import RobotConfig
from src.config.tree import ConfigTree, load_tree

_CONFIG = Path(__file__).resolve().parents[1] / "config"
_HOME_DEG = (-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)
_HOME_RAD = tuple(math.radians(v) for v in _HOME_DEG)


class AHomeInDegreesTests(unittest.TestCase):
    def test_degrees_load_as_radians_and_the_degrees_key_reads_none(self) -> None:
        robot = RobotConfig.model_validate({"home_joint_positions_deg": list(_HOME_DEG)})

        self.assertEqual(_HOME_RAD, robot.home_joint_positions)
        self.assertIsNone(robot.home_joint_positions_deg)
        self.assertIn("home_joint_positions", robot.model_fields_set)
        self.assertNotIn("home_joint_positions_deg", robot.model_fields_set)

    def test_radians_load_as_they_always_did(self) -> None:
        robot = RobotConfig.model_validate({"home_joint_positions": list(_HOME_RAD)})

        self.assertEqual(_HOME_RAD, robot.home_joint_positions)
        self.assertIsNone(RobotConfig().home_joint_positions)

    def test_both_at_once_are_refused_naming_both_keys(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            RobotConfig.model_validate({"home_joint_positions": list(_HOME_RAD),
                                        "home_joint_positions_deg": list(_HOME_DEG)})
        self.assertIn("robot.home_joint_positions and robot.home_joint_positions_deg are both set", str(caught.exception))

    def test_a_loaded_config_validates_again_as_it_dumps(self) -> None:
        """The degrees are gone from the loaded config, so a dump read back is not two homes."""
        robot = RobotConfig.model_validate({"home_joint_positions_deg": list(_HOME_DEG)})

        self.assertEqual(robot, RobotConfig.model_validate(robot.model_dump()))

    def test_the_sim_twin_takes_degrees_too(self) -> None:
        robot = RobotConfig.model_validate({"sim": {"home_joint_positions_deg": [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]}})

        self.assertEqual([0.0, -math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0], robot.sim.home_joint_positions)
        self.assertIsNone(robot.sim.home_joint_positions_deg)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"sim": {"home_joint_positions": [0.0] * 6,
                                                "home_joint_positions_deg": [0.0] * 6}})

    def test_a_cell_profile_over_one_with_a_home_in_radians_resets_it_to_give_degrees(self) -> None:
        """The ur5e profile writes its home in radians; a cell layered over it says so, or resets that key."""
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "config"
            shutil.copytree(_CONFIG, root)
            cell = root / "robot" / "robot.mycell.yaml"
            cell.write_text(f"robot:\n  home_joint_positions_deg: {list(_HOME_DEG)}\n", encoding="utf-8")
            refused = ConfigTree.from_directory(root=root, profile="ur5e,mycell").load()
            with self.assertRaises(ConfigError) as caught:
                _ = refused.robot
            self.assertIn('resets that key in its own file with "__null__"', str(caught.exception))

            cell.write_text(f'robot:\n  home_joint_positions: "__null__"\n  home_joint_positions_deg: {list(_HOME_DEG)}\n',
                            encoding="utf-8")
            tree = ConfigTree.from_directory(root=root, profile="ur5e,mycell").load()
            self.assertEqual(_HOME_RAD, tree.robot.home_joint_positions)

    def test_a_tree_given_the_home_in_degrees_hands_every_consumer_radians(self) -> None:
        tree = load_tree(None).with_values({"robot.home_joint_positions_deg": list(_HOME_DEG)})

        self.assertTrue(tree.ok, tree.error)
        self.assertEqual(_HOME_RAD, tree.robot.home_joint_positions)
        self.assertIsNone(tree.robot.home_joint_positions_deg)


if __name__ == "__main__":
    unittest.main()
