"""The checklist's planner rows follow the arm the tree selects, not the spelling of its vendor (P0i i1).

A carried part and a camera world are the planner's to model, and the desk asked for them only where ``robot.vendor``
read ``"ur"``. The selection already says the fact those rows need (``RobotConfig.arm``: a vendor whose driver reads
``robot.motion.planner`` names a planner, and one that drives a controller of its own owns one), so a vendor that joins
the UR in reading the key gets the rows by that fact alone. The Franka stands in for it here: its driver is not
written yet, so each test adds it to the vendors whose driver reads the key, as P1 will.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.config.schema.robot import RobotConfig, robot_schema
from src.robot.execution.real_cell.preflight import run_config_preflight

_PLANNER_ROWS = ("carried part", "camera world")


def _rows(config: RobotConfig) -> set[str]:
    report = run_config_preflight(config, curobo_available=True, collision_engine="coal")
    return {check.name for check in report.checks}


class TheChecklistAsksTheSelectionTests(unittest.TestCase):
    def test_a_vendor_whose_driver_reads_the_planner_gets_the_planner_rows(self) -> None:
        with mock.patch.object(robot_schema, "_HONOURS_THE_PLANNER_KEY", frozenset({"ur", "franka"})):
            config = RobotConfig.model_validate({"vendor": "franka"})
            self.assertEqual(config.arm.planner, "curobo")
            rows = _rows(config)
        for name in _PLANNER_ROWS:
            self.assertIn(name, rows)

    def test_a_vendor_whose_driver_reads_no_planner_gets_none_of_them(self) -> None:
        """The other half, a control: the old gate on the name "ur" gave the same answer here."""
        config = RobotConfig.model_validate({"vendor": "franka"})
        self.assertEqual(config.arm.planner, "")
        rows = _rows(config)
        for name in _PLANNER_ROWS:
            self.assertNotIn(name, rows)

    def test_a_ur_cell_keeps_them_and_a_desk_dummy_has_none(self) -> None:
        """The control: the rows are there on the UR, as before, and a dummy, which reads no planner, has none."""
        self.assertTrue(set(_PLANNER_ROWS) <= _rows(RobotConfig.model_validate({"vendor": "ur"})))
        self.assertFalse(set(_PLANNER_ROWS) & _rows(RobotConfig.model_validate({"vendor": "dummy"})))

    def test_the_vendors_said_to_read_the_planner_key_are_those_whose_arm_is_a_planned_arm(self) -> None:
        """The set the selection reads (``robot_schema._HONOURS_THE_PLANNER_KEY``) is kept by hand, and a planned arm reads
        ``robot.motion.planner`` whatever its vendor. A registered vendor whose driver builds one and is missing from the
        set would lose the planner rows in silence, and pass the desk (review of i1). So every registered vendor's arm
        is built from a shipped tree and the two answers are compared; a vendor registered without a tree here fails."""
        from src.config import ConfigTree
        from src.robot.drivers import available_vendors
        from src.robot.drivers.planned.arm import PlannedArm
        from src.robot.execution.robot_parts import construct_arm

        trees = {"ur": "ursim", "sim": "sim", "dummy": "console_dummy"}
        planned = []
        for vendor in available_vendors():
            with self.subTest(vendor=vendor):
                self.assertIn(vendor, trees, "a registered vendor with no tree in this test: add one, so its arm is "
                                             "built and compared")
                arm = construct_arm(ConfigTree.from_directory(profile=trees[vendor]).load().robot)
                self.assertEqual(vendor in robot_schema._HONOURS_THE_PLANNER_KEY, isinstance(arm, PlannedArm))
                if isinstance(arm, PlannedArm):
                    planned.append(vendor)
        self.assertTrue(planned, "no registered vendor builds a planned arm, so the comparison held for nothing")


if __name__ == "__main__":
    unittest.main()
