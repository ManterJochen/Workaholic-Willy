"""The speed keys load on, reach the pick loop, and a loop built by hand ranks as before (the owner, 2026-10-08).

Three keys beside ``side_approaches``, each on where the owner chose speed first:

* ``robot.grasping.first_good_part`` (true): a look's parts are computed one at a time and the first good one is taken
  (``BinPickingOrchestrator.lazy_objects``);
* ``robot.grasping.good_part_score`` (0.75, 0 to 1): the score that part's best grasp needs (``accept_score``);
* ``robot.grasping.fine_pass_waits`` (true): SFE's fine search waits while another part may have a full result
  (``fine_search_waits``).

``RuntimePickService.from_robot_config`` sets them on the loop, beside ``natural_closing_axis``. The loop's own defaults
stay off, so every pick built by hand, and every test that builds one, ranks every part in full as before.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config import load_robot_config
from src.config.loader import available_profiles
from src.config.schema.robot import RobotConfig
from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from tests._helpers import _FakeArm, _FakePerception, _perception_frame, _ScriptedCalculator

_OWNER_TREE = Path("D:/helper_cell/robominds_config/config/robot/robot.yaml")


def _switches(orchestrator: BinPickingOrchestrator) -> tuple[bool, float, bool]:
    return orchestrator.lazy_objects, orchestrator.accept_score, orchestrator.fine_search_waits


class TheKeysLoadOnTests(unittest.TestCase):
    def test_every_shipped_profile_takes_the_first_good_part_and_lets_the_fine_search_wait(self) -> None:
        for layer in (None, *available_profiles()):
            with self.subTest(profile=layer):
                grasping = load_robot_config(profile=layer).grasping
                self.assertEqual((True, 0.75, True),
                                 (grasping.first_good_part, grasping.good_part_score, grasping.fine_pass_waits))

    def test_the_owners_tree_takes_the_first_good_part(self) -> None:
        if not _OWNER_TREE.is_file():
            raise unittest.SkipTest("the owner's tree is not on this machine")
        robot = RobotConfig.model_validate(yaml.safe_load(_OWNER_TREE.read_text(encoding="utf-8"))["robot"])
        self.assertIs(robot.grasping.first_good_part, True)
        self.assertIs(robot.grasping.fine_pass_waits, True)

    def test_each_key_can_be_switched(self) -> None:
        grasping = RobotGraspingConfig.model_validate(
            {"first_good_part": False, "good_part_score": 0.9, "fine_pass_waits": False})
        self.assertEqual((False, 0.9, False),
                         (grasping.first_good_part, grasping.good_part_score, grasping.fine_pass_waits))

    def test_a_score_outside_0_to_1_is_refused(self) -> None:
        for bound in (0.0, 1.0):
            self.assertEqual(bound, RobotGraspingConfig.model_validate({"good_part_score": bound}).good_part_score)
        for refused in (-0.01, 1.01, 75.0):
            with self.subTest(score=refused), self.assertRaises(ValidationError):
                RobotGraspingConfig.model_validate({"good_part_score": refused})


class TheKeysReachThePickLoopTests(unittest.TestCase):
    @staticmethod
    def _service(**grasping: object) -> RuntimePickService:
        config = RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping=grasping)
        return RuntimePickService.from_robot_config(
            config, calculator=_ScriptedCalculator([]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]))

    def test_a_cell_built_from_its_tree_ranks_as_the_keys_say(self) -> None:
        self.assertEqual((True, 0.75, True), _switches(self._service().orchestrator))
        switched = self._service(first_good_part=False, good_part_score=0.9, fine_pass_waits=False)
        self.assertEqual((False, 0.9, False), _switches(switched.orchestrator))

    def test_a_loop_built_by_hand_ranks_every_part_in_full_as_before(self) -> None:
        orchestrator = BinPickingOrchestrator(arm=_FakeArm(), calculator=_ScriptedCalculator([]),  # type: ignore[arg-type]
                                              perception=_FakePerception([_perception_frame()]))
        self.assertEqual((False, 0.75, False), _switches(orchestrator))


if __name__ == "__main__":
    unittest.main()
