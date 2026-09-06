"""Grasps on a point cloud, with no robot, no cell and no configuration.

⭐ **THE SMALLEST USEFUL THING THIS PACKAGE CAN DO, AND IT HAD NO DOOR.** Everything needed was
already pure: `generate_support_footprint_grasps` takes a bare BASE-frame array, takes a fully
defaulted frozen jaw, and returns frozen candidates. What had no name was the QUESTION, so the only
way in was to build a `GraspCalculator` with a camera matrix and a config block for a cell that does
not exist.

⛔⛔ **AND THE OBVIOUS VERSION WOULD HAVE BEEN A SAFETY DEFECT.** A `Scene` that simply called the
analytic primitive lets a cell whose config says `grasping.calculator: deep` receive ANALYTIC
geometry under the learned generator's name. `calculator_factory` fails closed on exactly that, and
the guard protecting it structurally cannot see this file: `tests/test_calculator_seam.py` sweeps for
modules constructing `GraspCalculator` BY NAME.

The answer is a stamped result rather than a comment. A silent substitution is only dangerous while
it is silent, and `SceneGrasps.generator` is on every report and in every wire form.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from src.contracts import Rendered, Structured
from src.robot.grasping import Scene, SceneGrasps


def _block(points: int = 900, seed: int = 0) -> np.ndarray:
    """A 60 x 60 x 40 mm block resting on z = 0, sampled as a cloud, in BASE millimetres."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(400.0, 460.0, points),
        rng.uniform(-30.0, 30.0, points),
        rng.uniform(0.0, 40.0, points),
    ])


class TheGeneratorIsAlwaysNamedTests(unittest.TestCase):
    """⛔ THE FIELD THAT STOPS ANALYTIC OUTPUT BEING READ AS LEARNED OUTPUT. It is the one way this
    class could mislead, so it is on the report and in the wire form and cannot be omitted."""

    def test_every_result_names_its_generator(self) -> None:
        grasps = Scene.from_cloud(_block(), support_height_mm=0.0).grasps()
        self.assertEqual(grasps.generator, "geometric")
        self.assertIn("geometric", grasps.render())
        self.assertEqual(grasps.to_dict()["generator"], "geometric")

    def test_an_empty_result_still_names_it(self) -> None:
        """The case where a reader is most likely to go looking for a reason, so the least useful
        place to omit which generator produced the nothing."""
        grasps = Scene.from_cloud(np.array([[400.0, 0.0, 10.0]]), support_height_mm=0.0).grasps()
        self.assertEqual(grasps.candidates, ())
        self.assertIn("geometric", grasps.render())

    def test_the_word_matches_the_config_key(self) -> None:
        """⚠ DELIBERATELY THE SAME WORD `robot.grasping.calculator` USES, so a reader comparing a
        report against a YAML file is comparing like with like rather than translating."""
        from src.config.schema.robot import RobotConfig

        self.assertEqual(str(RobotConfig().grasping.calculator), "geometric")


class ItFindsGraspsOnARealShapeTests(unittest.TestCase):

    def test_a_block_admits_ranked_grasps(self) -> None:
        grasps = Scene.from_cloud(_block(), support_height_mm=0.0).grasps(max_candidates=4)
        self.assertTrue(grasps.candidates)
        self.assertLessEqual(len(grasps.candidates), 4)
        self.assertIsNotNone(grasps.best)

    def test_the_candidates_are_ranked_best_first(self) -> None:
        grasps = Scene.from_cloud(_block(), support_height_mm=0.0).grasps(max_candidates=6)
        scores = [c.score for c in grasps.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))
        assert grasps.best is not None
        self.assertEqual(grasps.best.score, scores[0])

    def test_the_grasps_sit_on_the_object_not_under_the_support(self) -> None:
        """⛔ THE PRIMITIVE REFUSES RATHER THAN PROPOSING A GRASP THAT GOES UNDER THE SUPPORT, and a
        wrapper that lost that would be worse than no wrapper."""
        grasps = Scene.from_cloud(_block(), support_height_mm=0.0).grasps(max_candidates=8)
        for candidate in grasps.candidates:
            with self.subTest(score=candidate.score):
                self.assertGreater(candidate.clearance_mm, 0.0)
                self.assertGreater(float(candidate.position_mm[2]), 0.0)

    def test_an_empty_result_is_an_answer_not_a_failure(self) -> None:
        """Two points cannot reconstruct a prism. That is a real answer, and the count is what
        separates 'nothing over 2 points' from 'nothing over 12,000'."""
        grasps = Scene.from_cloud(np.array([[400.0, 0.0, 10.0], [401.0, 0.0, 10.0]]),
                                  support_height_mm=0.0).grasps()
        self.assertEqual(grasps.candidates, ())
        self.assertEqual(grasps.points, 2)
        self.assertIn("too sparse", grasps.render())
        self.assertIsNone(grasps.best)


class TheTwoDoorsTests(unittest.TestCase):
    """⭐ THE ONE-BUILDER RULE: `from_robot_config` resolves config into arguments and CALLS
    `from_cloud`, so the two cannot drift into two constructions."""

    def test_the_config_door_takes_its_geometry_from_the_config(self) -> None:
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig()
        scene = Scene.from_robot_config(cfg, _block())
        self.assertEqual(scene.support_height_mm, float(cfg.grasping.support.height_mm))
        self.assertIsNotNone(scene.jaw, "the jaw comes from the gripper block")

    def test_both_doors_produce_the_same_shape_of_answer(self) -> None:
        from src.config.schema.robot import RobotConfig

        cloud = _block()
        cfg = RobotConfig()
        bare = Scene.from_cloud(cloud, support_height_mm=float(cfg.grasping.support.height_mm))
        configured = Scene.from_robot_config(cfg, cloud)
        self.assertEqual(bare.grasps().generator, configured.grasps().generator)
        self.assertEqual(bare.support_height_mm, configured.support_height_mm)

    def test_a_list_of_lists_is_accepted_as_well_as_an_array(self) -> None:
        """A caller with plain Python data should not have to learn numpy to ask the question."""
        cloud = _block(points=400).tolist()
        self.assertTrue(Scene.from_cloud(cloud, support_height_mm=0.0).grasps().candidates)


class TheReportContractTests(unittest.TestCase):

    def test_it_is_rendered_and_structured(self) -> None:
        grasps = Scene.from_cloud(_block(), support_height_mm=0.0).grasps()
        self.assertIsInstance(grasps, Rendered)
        self.assertIsInstance(grasps, Structured)

    def test_render_obeys_the_contract_for_full_and_empty_results(self) -> None:
        for cloud in (_block(), np.array([[400.0, 0.0, 10.0]])):
            with self.subTest(points=len(cloud)):
                text = Scene.from_cloud(cloud, support_height_mm=0.0).grasps().render()
                self.assertTrue(text.isascii(), "printed to a cp1252 console")
                self.assertFalse(text.endswith("\n"), "the caller owns the line break")
                self.assertTrue(text.strip())

    def test_to_dict_survives_json_without_a_custom_encoder(self) -> None:
        """⛔ THE REASON `to_dict` EXISTS RATHER THAN `dataclasses.asdict`: every candidate carries
        numpy arrays, and asdict would hand a consumer an ndarray that fails on the first dumps."""
        payload = Scene.from_cloud(_block(), support_height_mm=0.0).grasps(max_candidates=3).to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertIsInstance(payload["candidates"][0]["position_mm"], list)

    def test_the_point_count_travels_with_the_result(self) -> None:
        grasps = Scene.from_cloud(_block(points=250), support_height_mm=0.0).grasps()
        self.assertEqual(grasps.points, 250)
        self.assertEqual(grasps.to_dict()["points"], 250)


class ItIsExportedTests(unittest.TestCase):
    """⚠ MEASURED BEFORE EXPORTING: 1 module and 0.7 ms on top of what `grasping` already imports,
    and nothing from `deep/`. This package's public surface must not be coupled to a tree being
    reorganised elsewhere."""

    def test_the_names_are_on_the_package(self) -> None:
        import src.robot.grasping as grasping

        self.assertIn("Scene", grasping.__all__)
        self.assertIn("SceneGrasps", grasping.__all__)
        self.assertIs(grasping.Scene, Scene)
        self.assertIs(grasping.SceneGrasps, SceneGrasps)

    def test_importing_the_package_pulls_nothing_from_deep(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, src.robot.grasping;"
             "print([m for m in sys.modules if '.deep' in m])"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]", "grasping now imports the deep tree")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
