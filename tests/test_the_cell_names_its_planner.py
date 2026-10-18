"""How a cell plans its motions is the cell's key, and a tree that still writes the vendor's is refused.

``robot.ur.motion_planner`` chose the planner until 2026-10-13. It sat in the UR block, and every vendor block is a
``default_factory``, so a tree naming any other vendor materialised it too and carried ``curobo`` without a UR in the
cell. The desk checklist runs every real vendor's rows (``is_real = vendor not in ("sim", "dummy")``), so a tree for
another vendor was judged on the UR block's default: its ``cuRobo environment`` row blocked a cell whose driver never
heard of cuRobo.

The key is ``robot.motion.planner`` now, beside the vendor blocks rather than inside one, and a reader asks
``RobotConfig.arm`` (``test_the_tree_names_one_arm.py``), which answers ``""`` for a vendor whose driver does not read
the key. The Isaac driver is one such vendor: its planner lives in its own driver config.

The old key is refused, not ignored (ported from aurora, 2026-10-13): a tree that set ``ik`` on purpose on a box with
no cuRobo would otherwise take the ``curobo`` default and refuse every motion it was asked for, the same symptom as a
broken install.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError, reload_config
from src.config.schema.robot import RobotConfig
from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight

_DATA = Path(__file__).resolve().parents[1] / "config"


#: A vendor this layer has no block for. No RobotVendor names one since kuka, franka and ros2 were removed
#: (2026-10-17), so its tree is set past validation, as a driver package not written yet would name its own.
_FOREIGN = "acme"


def _tree(vendor: str, planner: str = "curobo") -> RobotConfig:
    if vendor == _FOREIGN:
        return _tree("ur", planner).model_copy(update={"vendor": vendor})
    return RobotConfig.model_validate({"vendor": vendor, "motion": {"planner": planner}})


class TheOldKeyIsRefusedNamingTheNewOneTests(unittest.TestCase):
    """A copy of the shipped tree with one extra layer, so the key that left can be written back."""

    def setUp(self) -> None:
        from src.config.loader import set_active_profile

        self._tmp = tempfile.TemporaryDirectory(prefix="willy_planner_key_")
        self.root = Path(self._tmp.name) / "config"
        shutil.copytree(_DATA, self.root)
        set_active_profile(None)
        reload_config()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(reload_config)
        self.addCleanup(set_active_profile, None)

    def _load_error(self, layer: str, text: str) -> str:
        from src.config import load_config

        (self.root / "robot" / f"robot.{layer}.yaml").write_text(text, encoding="utf-8")
        reload_config()
        with self.assertRaises(ConfigError) as caught:
            load_config(self.root, profile=layer)
        return str(caught.exception)

    def test_a_tree_that_still_writes_the_ur_key_is_refused_and_told_where_it_went(self) -> None:
        message = self._load_error("plannercheck", 'robot:\n  ur:\n    motion_planner: "ik"\n')

        self.assertIn("robot.ur.motion_planner", message)
        self.assertIn("robot.motion.planner", message, "the refusal must name the key that replaced it")
        self.assertIn("robot.plannercheck.yaml", message, "an operator has to be told which file to edit")
        self.assertIn("plannercheck", message)

    def test_the_refusal_says_why_ignoring_it_would_be_worse(self) -> None:
        """The reason is the decision: a dropped ``ik`` becomes a cell that refuses every motion."""
        message = self._load_error("plannerwhy", 'robot:\n  ur:\n    motion_planner: "ik"\n')

        self.assertIn("curobo", message)
        self.assertIn("refuse", message)

    def test_the_schema_refuses_the_key_at_its_own_door(self) -> None:
        """Not only the loader: a tree built in code gets the same answer, from ``extra='forbid'``."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"vendor": "ur", "ur": {"motion_planner": "ik"}})


class TheSelectionAnswersForTheSelectedDriverTests(unittest.TestCase):
    def test_a_ur_tree_answers_the_key_it_was_given(self) -> None:
        self.assertEqual("curobo", _tree("ur").arm.planner)
        self.assertEqual("ik", _tree("ur", "ik").arm.planner)

    def test_a_driver_that_does_not_read_the_key_states_nothing_although_the_key_says_curobo(self) -> None:
        """The key is loadable on every tree; what changes is that the selection does not quote it back."""
        for vendor in ("sim", "dummy", _FOREIGN):
            with self.subTest(vendor=vendor):
                tree = _tree(vendor)
                self.assertEqual("curobo", tree.motion.planner, "the key itself is unchanged and still loadable")
                self.assertEqual("", tree.arm.planner, "a cell whose driver never reads the key quoted it anyway")

    def test_the_pick_service_asks_the_selection_whether_this_cell_plans(self) -> None:
        """A cuRobo cell needs a live camera world or a decline for every motion, and no other cell does."""
        from src.robot.execution.autonomous_grasp.cells import _plans_with_curobo

        self.assertTrue(_plans_with_curobo(_tree("ur")))
        self.assertFalse(_plans_with_curobo(_tree("ur", "ik")))
        for vendor in ("sim", "dummy", _FOREIGN):
            with self.subTest(vendor=vendor):
                self.assertFalse(_plans_with_curobo(_tree(vendor)))

    def test_the_ur_driver_plans_as_the_cell_says(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        def arm(planner: str) -> URRobotArm:
            return URRobotArm(RobotConfig.model_validate(
                {"vendor": "ur", "motion": {"planner": planner}, "gripper": {"model": "robotiq_2f85"}}))

        self.assertTrue(arm("curobo").plans_paths)
        self.assertFalse(arm("ik").plans_paths)


class TheDeskStopsJudgingOneVendorByAnothersBlockTests(unittest.TestCase):
    """A real non-UR cell was judged on the UR block's ``curobo`` default.

    A vendor with no block of its own is the case the checklist got wrong (``franka`` was one until 2026-10-17): it is
    real, so every ``is_real`` row ran, and the only planner key belonged to a vendor it does not use.
    """

    @staticmethod
    def _rows(tree: RobotConfig) -> dict[str, CheckStatus]:
        report = run_config_preflight(tree, curobo_available=False, collision_engine="coal")
        return {check.name: check.status for check in report.checks}

    def test_a_missing_curobo_environment_does_not_block_a_cell_that_never_plans_with_it(self) -> None:
        rows = self._rows(_tree(_FOREIGN))

        self.assertIs(CheckStatus.WARN, rows["cuRobo environment"])
        self.assertIs(CheckStatus.BLOCK, self._rows(_tree("ur"))["cuRobo environment"],
                      "the control: a UR cell on the same box still blocks, so this is not a row that stopped firing")

    def test_a_planner_margin_is_not_demanded_of_a_cell_with_no_planner(self) -> None:
        rows = self._rows(_tree(_FOREIGN))

        self.assertIs(CheckStatus.WARN, rows["planner margin"])
        self.assertIs(CheckStatus.BLOCK, self._rows(_tree("ur"))["planner margin"],
                      "the control: a UR cell that declares no margin still blocks")

    def test_the_row_names_the_driver_rather_than_quoting_an_empty_key(self) -> None:
        """'robot.motion.planner is None' reads as a broken tree; the truth is that this driver does not read it."""
        report = run_config_preflight(_tree(_FOREIGN), curobo_available=False, collision_engine="coal")
        detail = next(check.detail for check in report.checks if check.name == "cuRobo environment")

        self.assertIn(_FOREIGN, detail)
        self.assertNotIn("None", detail)


if __name__ == "__main__":
    unittest.main()
