"""An undeclared planner margin is refused on every cell that plans with cuRobo on a controller of its own (P0e E12).

``planner_margin_refusal`` asked ``vendor != "ur"`` first, so for every vendor after the UR the refusal of an
undeclared margin would have switched itself off: the planner keeping no clearance while the guard demands its own,
and paths proposed that the guard refuses just before the arm moves. The rule reads the tree's selection now, whatever
vendor it names. The simulator and the desk dummy drive no controller and are not asked; the sim declares its margin in
its own layer.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.config.schema.robot import RobotConfig
from src.robot.safety.planning.margin import planner_margin_refusal


def _cell(vendor: str, *, planner: str = "curobo", owns: bool = True, margin: float | None = None) -> SimpleNamespace:
    """A tree as the rule reads it: its selection and its self-collision block, for a vendor the schema has no block for."""
    return SimpleNamespace(
        vendor=vendor,
        arm=SimpleNamespace(vendor=vendor, model="fr3", controller="172.16.0.2", planner=planner,
                            owns_a_controller=owns),
        safety=SimpleNamespace(self_collision=SimpleNamespace(planner_margin_mm=margin, min_distance_mm=10.0)),
    )


class EveryVendorIsAskedTests(unittest.TestCase):
    def test_a_franka_cell_on_curobo_with_no_margin_is_refused(self) -> None:
        refusal = planner_margin_refusal(_cell("franka"))  # type: ignore[arg-type]
        assert refusal is not None
        self.assertIn("planner_margin_mm is undeclared", refusal[0])

    def test_a_declared_margin_passes_on_any_vendor(self) -> None:
        self.assertIsNone(planner_margin_refusal(_cell("franka", margin=4.0)))  # type: ignore[arg-type]

    def test_a_cell_that_plans_with_no_curobo_is_not_asked(self) -> None:
        self.assertIsNone(planner_margin_refusal(_cell("franka", planner="")))  # type: ignore[arg-type]

    def test_a_cell_that_drives_no_controller_is_not_asked(self) -> None:
        self.assertIsNone(planner_margin_refusal(_cell("sim", owns=False)))  # type: ignore[arg-type]

    def test_the_ur_cell_reads_as_before(self) -> None:
        tree = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.1.2.3"}, "motion": {"planner": "curobo"}})
        self.assertIsNotNone(planner_margin_refusal(tree))
        declared = RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.1.2.3"}, "motion": {"planner": "curobo"},
                                               "safety": {"self_collision": {"planner_margin_mm": 4.0}}})
        self.assertIsNone(planner_margin_refusal(declared))


if __name__ == "__main__":
    unittest.main()
