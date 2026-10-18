"""A cell's home can be written in degrees, as the pendant shows it, and every consumer still reads radians.

The owner, 2026-09-24: the home may be given in degrees. The pendant and
``python -m src.robot.execution.io_bench --where`` show degrees, and the UR driver's home verb, the joint window centred on the
home and the sim arm all read radians. Since 2026-10-17 (the owner's cleanup) the home is written in degrees alone, as
``robot.motion.home_deg``, and read in radians through ``robot.motion.home_rad``; the radians key
``robot.home_joint_positions`` is refused at load with the sentence that says so, and the old degrees key with the
sentence that names where it went. The sim keeps its own twin, ``robot.sim.home_joint_positions`` in radians or
``robot.sim.home_joint_positions_deg`` in degrees, read once at load into radians, both at once refused.
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

_CONFIG = Path(__file__).resolve().parents[4] / "config"
_HOME_DEG = (-90.0, -100.0, -110.0, -60.0, 90.0, 0.0)
_HOME_RAD = tuple(math.radians(v) for v in _HOME_DEG)


def _cell_error(layer: str) -> str:
    """The message a copy of the shipped tree is refused with, ``layer`` written as its profile ``mycell``."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "config"
        shutil.copytree(_CONFIG, root)
        (root / "robot" / "robot.mycell.yaml").write_text(layer, encoding="utf-8")
        refused = ConfigTree.from_directory(root=root, profile="mycell").load()
        try:
            _ = refused.robot
        except ConfigError as caught:
            return str(caught)
    raise AssertionError(f"the tree loaded with {layer!r}")


class AHomeInDegreesTests(unittest.TestCase):
    def test_degrees_load_and_are_read_in_radians(self) -> None:
        robot = RobotConfig.model_validate({"motion": {"home_deg": list(_HOME_DEG)}})

        self.assertEqual(_HOME_DEG, robot.motion.home_deg)
        self.assertEqual(_HOME_RAD, robot.motion.home_rad)
        self.assertIsNone(RobotConfig().motion.home_deg)
        self.assertIsNone(RobotConfig().motion.home_rad)

    def test_the_home_in_radians_is_refused_at_load_with_its_sentence(self) -> None:
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"home_joint_positions": list(_HOME_RAD)})
        said = _cell_error(f"robot:\n  home_joint_positions: {list(_HOME_RAD)}\n")
        self.assertIn("robot.home_joint_positions", said)
        self.assertIn("the home is written in degrees alone since 2026-10-17", said)
        self.assertIn("robot.motion.home_deg", said)

    def test_the_old_degrees_key_is_refused_naming_where_it_went(self) -> None:
        said = _cell_error(f"robot:\n  home_joint_positions_deg: {list(_HOME_DEG)}\n")
        self.assertIn("robot.home_joint_positions_deg", said)
        self.assertIn("this one is now robot.motion.home_deg", said)

    def test_a_loaded_config_validates_again_as_it_dumps(self) -> None:
        """The dump holds the home in degrees alone (``home_rad`` is read off it, never stored), so a dump read back is
        the same config."""
        robot = RobotConfig.model_validate({"motion": {"home_deg": list(_HOME_DEG)}})

        self.assertNotIn("home_rad", robot.model_dump()["motion"])
        self.assertEqual(robot, RobotConfig.model_validate(robot.model_dump()))

    def test_the_sim_twin_takes_degrees_too(self) -> None:
        robot = RobotConfig.model_validate({"sim": {"home_joint_positions_deg": [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]}})

        self.assertEqual([0.0, -math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0], robot.sim.home_joint_positions)
        self.assertIsNone(robot.sim.home_joint_positions_deg)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"sim": {"home_joint_positions": [0.0] * 6,
                                                "home_joint_positions_deg": [0.0] * 6}})

    def test_a_cell_profile_over_one_with_a_home_in_radians_resets_it_to_give_degrees(self) -> None:
        """The sim profile writes the sim's home in radians; a cell layered over it says so, or resets that key."""
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "config"
            shutil.copytree(_CONFIG, root)
            cell = root / "robot" / "robot.mycell.yaml"
            cell.write_text(f"robot:\n  sim:\n    home_joint_positions_deg: {list(_HOME_DEG)}\n", encoding="utf-8")
            refused = ConfigTree.from_directory(root=root, profile="sim,mycell").load()
            with self.assertRaises(ConfigError) as caught:
                _ = refused.robot
            self.assertIn('resets that key in its own file with "__null__"', str(caught.exception))

            cell.write_text(f'robot:\n  sim:\n    home_joint_positions: "__null__"\n'
                            f'    home_joint_positions_deg: {list(_HOME_DEG)}\n', encoding="utf-8")
            tree = ConfigTree.from_directory(root=root, profile="sim,mycell").load()
            self.assertEqual(list(_HOME_RAD), tree.robot.sim.home_joint_positions)

    def test_a_tree_given_the_home_in_degrees_hands_every_consumer_radians(self) -> None:
        tree = load_tree(None).with_values({"robot.motion.home_deg": list(_HOME_DEG)})

        self.assertTrue(tree.ok, tree.error)
        self.assertEqual(_HOME_RAD, tree.robot.motion.home_rad)
        self.assertEqual(_HOME_DEG, tree.robot.motion.home_deg)


if __name__ == "__main__":
    unittest.main()
