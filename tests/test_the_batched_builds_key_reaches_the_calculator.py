"""``robot.grasping.batched_builds`` loads off, reaches the cell's calculator and ``Scene``, and off is SFE as before.

The key makes each of SFE's closing lines build its grasps at once (``support_footprint._build_many``): the same grasps
to the bit, a tenth of the time on a boxed-in part. Off by default, every build is made one at a time, as before:
nothing of the batched path is reached. On, ``build_calculator`` asks this machine's numpy whether its stacked products
answer to the bit (``batched_builds_hold``) and says the answer in the calculator's log; the calculator hands the switch
to the stage with every ranking (``sfe_batched``); ``Scene.from_robot_config`` takes it too. Where the numpy would round
otherwise the calculator builds one at a time and says so.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from src.config import load_robot_config
from src.config.loader import available_profiles
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.grasping.generation import calculator as calculator_module
from src.robot.grasping.generation import support_footprint as sf
from tests.test_a_lines_builds_made_at_once_answer_to_the_bit import said, zollstock


def _hande(batched: bool) -> Any:
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    cfg = hande_cell()
    return cfg.model_copy(update={"grasping": cfg.grasping.model_copy(update={"batched_builds": batched})})


def _calculator(cfg: Any) -> Any:
    from src.robot.grasping.calculator_factory import build_calculator

    return build_calculator(cfg, camera_matrix=None, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True)


class TheKeyLoadsOffTests(unittest.TestCase):
    def test_every_shipped_profile_builds_one_at_a_time(self) -> None:
        for layer in (None, *available_profiles()):
            with self.subTest(profile=layer):
                self.assertIs(load_robot_config(profile=layer).grasping.batched_builds, False)

    def test_the_key_can_be_switched_on(self) -> None:
        self.assertIs(RobotGraspingConfig.model_validate({"batched_builds": True}).batched_builds, True)
        self.assertIs(RobotGraspingConfig.model_validate({}).batched_builds, False)


class TheKeyReachesTheCalculatorTests(unittest.TestCase):
    def test_on_the_calculator_builds_at_once_and_says_so(self) -> None:
        with self.assertLogs("GraspCalculator", "INFO") as told:
            calculator = _calculator(_hande(True))
        self.assertIs(calculator.sfe_batched, True)
        self.assertTrue(any("at once" in line and "batched_builds" in line for line in told.output), told.output)

    def test_off_the_calculator_builds_one_at_a_time_and_says_nothing_of_it(self) -> None:
        with self.assertNoLogs("GraspCalculator", "INFO"):
            calculator = _calculator(_hande(False))
        self.assertIs(calculator.sfe_batched, False)

    def test_where_this_machines_numpy_would_round_otherwise_it_builds_one_at_a_time_and_says_why(self) -> None:
        with mock.patch.object(sf, "batched_builds_hold", return_value="numpy's stacked product rounds otherwise"), \
                self.assertLogs("GraspCalculator", "WARNING") as told:
            calculator = _calculator(_hande(True))
        self.assertIs(calculator.sfe_batched, False)
        self.assertIn("rounds otherwise", told.output[0])
        self.assertIn("one at a time", told.output[0])

    def test_the_calculator_hands_its_switch_to_the_stage_with_every_ranking(self) -> None:
        handed: list[bool] = []
        stage = calculator_module.support_footprint_breakdowns

        def spy(*args: Any, **keywords: Any) -> Any:
            handed.append(keywords["batched"])
            return stage(*args, **keywords)

        for batched in (False, True):
            with self.subTest(batched=batched), \
                    mock.patch.object(calculator_module, "support_footprint_breakdowns", side_effect=spy):
                result, calculator = zollstock("P1", batched=batched)
                self.assertIs(calculator.sfe_batched, batched)
                self.assertEqual(batched, handed[-1])

    def test_off_nothing_of_the_batched_path_is_reached(self) -> None:
        def reached(*args: Any, **keywords: Any) -> Any:
            raise AssertionError("the batched path was reached with the key off")

        with mock.patch.object(sf, "_run_units_many", side_effect=reached), \
                mock.patch.object(sf, "_build_many", side_effect=reached):
            result, calculator = zollstock("P5", batched=False)
        self.assertIn("support_footprint_refused", result.telemetry)


class TheKeyReachesTheSceneTests(unittest.TestCase):
    def test_a_scene_from_the_cells_tree_builds_as_the_key_says_with_the_same_grasps(self) -> None:
        from src.robot.grasping.scene import Scene
        from tests.test_sfe_says_why_it_refused import box_cloud

        cloud = box_cloud((-20.0, -20.0, 0.0), (20.0, 20.0, 40.0))
        grasps: dict[bool, Any] = {}
        for batched in (False, True):
            scene = Scene.from_robot_config(_hande(batched), cloud)
            asked: list[bool] = []
            real = sf.generate_support_footprint_grasps

            def spy(*args: Any, _asked: list[bool] = asked, **keywords: Any) -> Any:
                _asked.append(bool(keywords.get("batched", False)))
                return real(*args, **keywords)

            with mock.patch("src.robot.grasping.scene.generate_support_footprint_grasps", side_effect=spy):
                grasps[batched] = scene.grasps()
            self.assertEqual([batched], asked)
        self.assertTrue(grasps[False].candidates)
        self.assertEqual(said(list(grasps[False].candidates)), said(list(grasps[True].candidates)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
