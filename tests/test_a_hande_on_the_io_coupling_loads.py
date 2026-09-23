"""A Hand-E on the Robotiq I/O Coupling is a digital-I/O hand, and a tree naming it with jaw_io loads.

The registry listed only the Robotiq socket driver for the Hand-E, so a UR10 cell driving its Hand-E through the
Robotiq I/O Coupling (one tool output, ``jaw_io`` ``single_toggle``) was refused at load, calibration included, and
leaving the hand unnamed made the planner and the guard model a 2F-85 (audit 3, gripper lens, 2026-09-23). The
registry now lists ``jaw_io`` beside ``robotiq`` for the Hand-E, and only for it.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "config"

#: The owner's cell as written on 2026-09-23: the Hand-E on the I/O Coupling, toggled on tool DO0, no feedback wired.
_OWNER_LAYER = """\
robot:
  gripper:
    model: robotiq_hande
    vendor: jaw_io
    jaw_io:
      actuation: single_toggle
      close_output_pin: 0
      io_port: tool
      close_settle_s: 2.0
"""


class _Tree:
    """A copy of the repository's config tree with one extra robot layer, ``robot.trial.yaml``."""

    def __init__(self, case: unittest.TestCase, layer: str) -> None:
        from src.config.loader import reload_config

        self.root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_DATA, self.root)
        (self.root / "robot" / "robot.trial.yaml").write_text(layer, encoding="utf-8")
        reload_config()
        case.addCleanup(reload_config)

    def load(self, profile: str):
        from src.config.loader import load_robot_config

        return load_robot_config(self.root, profile=profile)


class AHandEOnTheIoCouplingLoadsTests(unittest.TestCase):
    def test_ur10_with_the_hande_driven_as_jaw_io_single_toggle_loads(self) -> None:
        """Red before the registry listed jaw_io: ConfigError naming config/grippers/robotiq_hande.yaml."""
        robot = _Tree(self, _OWNER_LAYER).load("ur10,hande,trial")
        self.assertEqual(str(robot.ur.model), "ur10")
        self.assertEqual(str(robot.gripper.model), "robotiq_hande")
        self.assertEqual(str(robot.gripper.vendor), "jaw_io")
        self.assertEqual(str(robot.gripper.jaw_io.actuation), "single_toggle")
        self.assertEqual(robot.gripper.jaw_io.close_output_pin, 0)
        self.assertEqual(str(robot.gripper.jaw_io.io_port), "tool")
        self.assertEqual(robot.gripper.jaw_io.close_settle_s, 2.0)
        # The Hand-E's own widths, not the 2F-85's the tree falls back to when the hand is left unnamed.
        self.assertEqual((robot.gripper.max_width_mm, robot.gripper.min_width_mm), (49.99, 5.0))

    def test_the_hande_lists_jaw_io_and_the_2f85_does_not(self) -> None:
        from src.config.grippers import load_gripper

        self.assertEqual(load_gripper("robotiq_hande", aliases=False).drivers, ("robotiq", "jaw_io"))
        self.assertEqual(load_gripper("robotiq_2f85", aliases=False).drivers, ("robotiq",))

    def test_a_driver_the_hand_does_not_list_is_still_refused_at_load(self) -> None:
        """⭐ THE CONTROL: the list still refuses. The 2F-85 has no coupling route in the registry."""
        from src.config.loader import ConfigError

        layer = "robot:\n  gripper:\n    model: robotiq_2f85\n    vendor: jaw_io\n"
        with self.assertRaises(ConfigError) as caught:
            _Tree(self, layer).load("ur10,trial")
        for part in ("robotiq_2f85", "jaw_io", "config/grippers/robotiq_2f85.yaml"):
            self.assertIn(part, str(caught.exception))

    def test_the_desk_names_the_jaw_io_driver_for_the_hande(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        robot = _Tree(self, _OWNER_LAYER).load("ur10,hande,trial")
        rows = {check.name: check for check in
                run_config_preflight(robot, curobo_available=True, collision_engine="coal").checks}
        row = rows["gripper driver"]
        self.assertEqual(str(row.status), "ok", row.detail)
        self.assertIn("jaw_io", row.detail)
        self.assertIn("robotiq_hande", row.detail)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
