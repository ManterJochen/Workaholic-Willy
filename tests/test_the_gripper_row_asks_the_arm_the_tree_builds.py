"""The desk's gripper driver row decides on the arm the tree builds, as the build decides on the arm in hand (P0i i1).

The row asked the gripper registry with fixed facts, ``vendor "ur"`` and ``switches digital I/O``, whatever arm the tree
selected. It now builds that arm, never connecting it, and reads it through ``robot_parts.arm_facts``, the function the
build reads the handle in hand with: an arm that switches no pins has a solenoid jaw refused at the desk as it is at the
build, and an arm that does not build at all is said to, instead of the row promising a driver for it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from src.config.schema.robot import RobotConfig
from src.robot.core import RobotCapabilities
from src.robot.execution.real_cell.preflight import CheckStatus, PreflightCheck, run_config_preflight

_IP = "198.51.100.31"
_JAW_IO = {"vendor": "jaw_io", "model": "robotiq_2f85", "jaw_io": {"close_output_pin": 4, "io_port": "standard"}}


def _row(config: RobotConfig) -> PreflightCheck:
    report = run_config_preflight(config, curobo_available=True, collision_engine="coal")
    return next(check for check in report.checks if check.name == "gripper driver")


def _arm_without_io(_config: Any) -> Any:
    """A UR at this cell's address that switches no pins: its identity and nothing else."""
    return SimpleNamespace(capabilities=RobotCapabilities(
        vendor="ur", model="ur5e", dof=6, controller=_IP, owns_a_controller=True))


class TheGripperRowAsksTheArmTests(unittest.TestCase):
    def test_a_jaw_on_an_arm_that_switches_no_pins_is_refused_at_the_desk(self) -> None:
        config = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}, "gripper": _JAW_IO})
        with mock.patch("src.robot.execution.robot_parts.construct_arm", _arm_without_io):
            row = _row(config)
        self.assertIs(row.status, CheckStatus.BLOCK)
        self.assertIn("SupportsDigitalIO", row.detail)

    def test_the_same_jaw_on_the_arm_the_tree_builds_is_its_driver(self) -> None:
        """The control: the UR driver switches pins, so the row names the jaw_io driver."""
        row = _row(RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}, "gripper": _JAW_IO}))
        self.assertIs(row.status, CheckStatus.OK)
        self.assertIn("builds the jaw_io driver", row.detail)

    def test_an_arm_that_does_not_build_is_said_not_to(self) -> None:
        """Changes behaviour on purpose: the row said OK, "builds the robotiq driver", on an arm the build refuses."""
        config = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}, "gripper": {"vendor": "robotiq"}})
        row = _row(config)
        self.assertIs(row.status, CheckStatus.WARN)
        self.assertIn("that arm does not build: ConfigError: robot.gripper.model is unset", row.detail)

    def test_a_vendor_with_no_driver_is_blocked_even_where_the_arm_does_not_build(self) -> None:
        """A second cause the row states alone: which driver a named vendor has is the tree's to say, not the arm's.
        Found reviewing i1, where an arm that did not build turned this BLOCK into the arm's WARN."""
        config = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}, "gripper": {"vendor": "schunk"}})
        row = _row(config)
        self.assertIs(row.status, CheckStatus.BLOCK)
        self.assertIn("gripper.vendor='schunk' is a recognised name with no driver in this repo", row.detail)


if __name__ == "__main__":
    unittest.main()
