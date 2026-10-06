"""The calculator plans the hand where the exact guard holds it, past the declared TCP by what the registry says.

The guard places the hand's meshes by the registry's grasp centre and the coupling plates; the arm puts the TCP where the
tool frame says; the calculator's jaw stood its grasp centre at the TCP. On the owner's cell the TCP is declared 157 mm
out and the guard's grasp centre stands at 23 + 136.2 = 159.2 mm, so its fingertips stood 2.2 mm further than the
calculator planned them, and grasp after grasp in a pile was refused 2.3 to 2.96 mm from a neighbour against the guard's
3 (the grasp bench, 2026-10-06). ``planning.hand.hand_past_the_tcp_mm`` says how far, and the calculator's jaw stands
there (``builders.build_gripper_geometry``, ``SupportFootprintJaw.from_robot_config``), never shorter than it was and
never by more than the tool frame's tolerance.
"""

from __future__ import annotations

import unittest
from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=None)
def _hande() -> Any:
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


def _with_frame(z_mm: float, *, source: str = "polyscope") -> Any:
    """The ``hande`` profile with its tool frame declared ``z_mm`` out along the flange's axis: its plates are 20 mm,
    its registry grasp centre 136.2, so the guard's grasp centre stands at 156.2 mm."""
    cfg = _hande()
    tool = cfg.gripper.tool_frame.model_copy(update={"source": source, "offset_mm": (0.0, 0.0, float(z_mm))})
    return cfg.model_copy(update={"gripper": cfg.gripper.model_copy(update={"tool_frame": tool})})


class HowFarPastTheTcpTests(unittest.TestCase):
    def test_a_tcp_short_of_the_grasp_centre_is_that_far(self) -> None:
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm

        self.assertAlmostEqual(2.2, hand_past_the_tcp_mm(_with_frame(154.0)), places=6)
        self.assertAlmostEqual(6.2, hand_past_the_tcp_mm(_with_frame(150.0)), places=6)

    def test_a_tcp_at_or_past_the_grasp_centre_is_nothing(self) -> None:
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm

        for z in (156.2, 160.0):
            with self.subTest(z=z):
                self.assertEqual(0.0, hand_past_the_tcp_mm(_with_frame(z)))

    def test_a_difference_past_the_tolerance_is_a_cell_to_measure_not_one_to_plan_around(self) -> None:
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm

        self.assertEqual(0.0, hand_past_the_tcp_mm(_with_frame(140.0)))   # 16.2 mm, over the 10 mm tolerance

    def test_an_undeclared_frame_is_nothing(self) -> None:
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm

        self.assertEqual(0.0, hand_past_the_tcp_mm(_hande()))


class TheJawStandsWhereTheGuardHoldsItTests(unittest.TestCase):
    def test_every_box_moves_along_the_approach_by_that_much(self) -> None:
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

        geometry = _hande().grasping.gripper_geometry
        plain = build_gripper_geometry(geometry)
        moved = build_gripper_geometry(geometry, past_the_tcp_mm=2.2)
        width = float(_hande().gripper.max_width_mm)
        for before, after in zip(plain.collision_boxes(width), moved.collision_boxes(width)):
            with self.subTest(box=before.label):
                np.testing.assert_allclose(np.asarray(after.min_corner_mm) - np.asarray(before.min_corner_mm),
                                           [0.0, 0.0, 2.2], atol=1e-9)
                np.testing.assert_allclose(np.asarray(after.max_corner_mm) - np.asarray(before.max_corner_mm),
                                           [0.0, 0.0, 2.2], atol=1e-9)
        self.assertAlmostEqual(float(moved.pad_ahead_mm) - float(plain.pad_ahead_mm), 2.2)

    def test_the_palm_is_as_wide_as_the_registrys_housing(self) -> None:
        """The Hand-E's housing is 75.0 mm across the closing axis, its open fingers' outer faces 71.6: the palm box is
        the housing's (the guard held it 1.7 mm past the box on either side, the grasp bench, 2026-10-06)."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.execution.autonomous_grasp.service import _registry_palm_thickness_mm

        cfg = _hande()
        self.assertEqual(75.0, _registry_palm_thickness_mm(cfg))
        width = float(cfg.gripper.max_width_mm)
        plain = {b.label: b for b in build_gripper_geometry(cfg.grasping.gripper_geometry).collision_boxes(width)}
        housed = {b.label: b for b in build_gripper_geometry(
            cfg.grasping.gripper_geometry, palm_thickness_mm=75.0).collision_boxes(width)}
        self.assertAlmostEqual(float(plain["palm"].max_corner_mm[0]), width / 2.0 + 10.8, places=6)
        self.assertAlmostEqual(float(housed["palm"].max_corner_mm[0]), 37.5, places=6)
        self.assertAlmostEqual(float(housed["palm"].min_corner_mm[0]), -37.5, places=6)
        for label in ("finger_negative_x", "finger_positive_x"):
            np.testing.assert_allclose(housed[label].min_corner_mm, plain[label].min_corner_mm)
            np.testing.assert_allclose(housed[label].max_corner_mm, plain[label].max_corner_mm)
        # Never narrower than the open fingers: a housing thinner than them leaves the palm as it was.
        thin = {b.label: b for b in build_gripper_geometry(
            cfg.grasping.gripper_geometry, palm_thickness_mm=40.0).collision_boxes(width)}
        np.testing.assert_allclose(thin["palm"].max_corner_mm, plain["palm"].max_corner_mm)

    def test_never_shorter_than_the_registry(self) -> None:
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

        geometry = _hande().grasping.gripper_geometry
        self.assertEqual(build_gripper_geometry(geometry), build_gripper_geometry(geometry, past_the_tcp_mm=-3.0))

    def test_sfe_plans_the_tree_jaw_the_calculator_plans(self) -> None:
        """``Scene`` and the locator build their jaw from the tree; it stands where the cell's calculator's does."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.generation.support_footprint import SupportFootprintJaw

        from src.robot.safety.planning.hand import hand_past_the_tcp_mm

        cfg = _with_frame(154.0)
        from_tree = SupportFootprintJaw.from_robot_config(cfg)
        from_cell = SupportFootprintJaw.from_model(
            build_gripper_geometry(cfg.grasping.gripper_geometry, past_the_tcp_mm=hand_past_the_tcp_mm(cfg)),
            aperture_mm=float(cfg.gripper.max_width_mm), min_width_mm=float(cfg.gripper.min_width_mm),
            table_clearance_mm=float(cfg.grasping.support.min_clearance_mm))
        self.assertEqual(from_cell, from_tree)
        self.assertAlmostEqual(float(from_tree.finger_ahead_mm), 10.45 + 2.2, places=6)
        self.assertEqual(SupportFootprintJaw.from_robot_config(_hande()).finger_ahead_mm, 10.45)


if __name__ == "__main__":
    unittest.main()
