"""The config door of ``Scene`` plans the hand, the support and the floor the cell's pick path plans (owner's cell, 2026-09-23).

``Located.scene(i, tree.robot)`` is how example 13 goes from a located part to ``robot.pick``, and it reaches
``Scene.from_robot_config``. That door built its jaw from the stroke alone, so every finger number fell back to
``ParallelJawGripperModel()``, a 2F-85, while the cell path (``Cell``, example 11) builds the jaw from ``grasping.gripper_geometry``.
On the owner's Robotiq Hand-E a vertical approach then needed the anchor 62.4 mm above the table instead of 25.9 mm:
a 40 mm cube got only 90-degree side approaches 36 mm up, where the 75 mm housing reaches below the table, and a
30 mm part got no grasp at all. The door also planned on the declared table where the pick loop raises it to the
part's own lowest point, and on a 2 mm floor where the cell counts everything within its plane clearance as bench.
"""

from __future__ import annotations

import functools
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.generation.support_footprint import SupportFootprintJaw
from src.robot.grasping.scene import Scene

_CENTRE = (500.0, -300.0)


@functools.lru_cache(maxsize=None)
def _hande() -> Any:
    """The shipped ``hande`` profile's robot section: the owner's hand, whose numbers the loader fills from the registry."""
    from src.config.loader import load_robot_section

    return load_robot_section(profile="hande")


def _replace(model: Any, path: str, value: Any) -> Any:
    """``model`` with the dotted field ``path`` set to ``value``, every level copied and nothing else touched."""
    head, _, rest = path.partition(".")
    if not rest:
        return model.model_copy(update={head: value})
    return model.model_copy(update={head: _replace(getattr(model, head), rest, value)})


def _cube(size_mm: float, *, base_mm: float = 0.0, step_mm: float = 2.0) -> np.ndarray:
    """A cube's top face and the face toward a camera at -y, as a wrist camera tilted 45 degrees sees them."""
    half = size_mm / 2.0
    across = np.arange(-half, half + 1e-9, step_mm)
    up = np.arange(0.0, size_mm + 1e-9, step_mm)
    top = np.array([[_CENTRE[0] + x, _CENTRE[1] + y, base_mm + size_mm] for x in across for y in across])
    side = np.array([[_CENTRE[0] + x, _CENTRE[1] - half, base_mm + z] for x in across for z in up])
    return np.vstack([top, side])


def _tilt_deg(candidate: Any) -> float:
    return float(np.degrees(np.arccos(np.clip(-float(candidate.approach[2]), -1.0, 1.0))))


class TheSceneBuildsTheCellsJawTests(unittest.TestCase):
    """The jaw is the tree's hand, the same jaw the pick path's calculator plans with."""

    def test_on_the_hande_profile_the_scene_and_the_pick_path_build_one_jaw(self) -> None:
        """The pick path's jaw, built the way ``compute`` builds it from what the cell hands it."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

        cfg = _hande()
        pick_path = SupportFootprintJaw.from_model(
            build_gripper_geometry(cfg.grasping.gripper_geometry),
            aperture_mm=float(cfg.gripper.max_width_mm),
            min_width_mm=float(cfg.gripper.min_width_mm),
            table_clearance_mm=float(cfg.grasping.support.min_clearance_mm),
        )
        self.assertEqual(pick_path, Scene.from_robot_config(cfg, _cube(40.0)).jaw)

    def test_the_calculator_plans_with_that_jaw(self) -> None:
        """Through the calculator itself, built by the factory as the real cell builds it."""
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
        from src.robot.grasping.calculator_factory import build_calculator

        cfg = _hande()
        calculator = build_calculator(
            cfg, camera_matrix=np.array([[925.0, 0.0, 640.0], [0.0, 925.0, 360.0], [0.0, 0.0, 1.0]]),
            max_grip_width_mm=cfg.gripper.max_width_mm, min_grip_width_mm=cfg.gripper.min_width_mm,
            support_footprint_geometry=cfg.grasping.geometry.stage == "support_footprint",
            support_footprint_inflate_mm=cfg.grasping.geometry.inflate_mm)
        jaw = calculator._support_footprint_jaw(  # noqa: SLF001
            build_gripper_geometry(cfg.grasping.gripper_geometry), cfg.grasping.support.min_clearance_mm)
        self.assertEqual(jaw, Scene.from_robot_config(cfg, _cube(40.0)).jaw)

    def test_the_jaw_carries_the_hande_fingers_and_stroke(self) -> None:
        cfg = _hande()
        jaw = Scene.from_robot_config(cfg, _cube(40.0)).jaw
        assert jaw is not None
        fingers = cfg.grasping.gripper_geometry.parallel_jaw
        self.assertEqual("robotiq_hande", cfg.gripper.model)
        self.assertEqual(float(fingers.fingertip_depth_mm), jaw.finger_ahead_mm)
        self.assertEqual(float(fingers.finger_thickness_mm), jaw.finger_thickness_mm)
        self.assertEqual(float(fingers.palm_width_mm), jaw.palm_width_mm)
        self.assertEqual(float(cfg.gripper.max_width_mm), jaw.aperture_mm)
        self.assertEqual(float(cfg.grasping.support.min_clearance_mm), jaw.table_clearance_mm)
        self.assertNotEqual(SupportFootprintJaw.from_model().finger_ahead_mm, jaw.finger_ahead_mm,
                            "the Hand-E must not plan with a 2F-85's fingertip")

    def test_the_default_tree_builds_the_same_jaw_both_ways_too(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

        cfg = RobotConfig()
        pick_path = SupportFootprintJaw.from_model(
            build_gripper_geometry(cfg.grasping.gripper_geometry),
            aperture_mm=float(cfg.gripper.max_width_mm), min_width_mm=float(cfg.gripper.min_width_mm),
            table_clearance_mm=float(cfg.grasping.support.min_clearance_mm))
        self.assertEqual(pick_path, Scene.from_robot_config(cfg, _cube(40.0)).jaw)

    def test_a_suction_hand_is_refused_rather_than_planned_as_a_jaw(self) -> None:
        from src.config.schema.robot import RobotConfig

        cfg = _replace(RobotConfig(), "grasping.gripper_geometry.kind", "suction")
        with self.assertRaises(ValueError) as caught:
            Scene.from_robot_config(cfg, _cube(40.0))
        self.assertIn("parallel jaw", str(caught.exception))


class AHandEGraspsSmallPartsFromAboveTests(unittest.TestCase):
    """What the 2F-85 fingers cost on the owner's hand, on the audit's clouds."""

    def test_a_40_mm_cube_gets_a_vertical_approach(self) -> None:
        grasps = Scene.from_robot_config(_hande(), _cube(40.0)).grasps()
        best = grasps.best
        assert best is not None, grasps.render()
        self.assertLess(_tilt_deg(best), 1.0, grasps.render())
        self.assertLessEqual(best.grip_width_mm, 44.0)

    def test_a_30_mm_cube_gets_a_grasp(self) -> None:
        grasps = Scene.from_robot_config(_hande(), _cube(30.0)).grasps()
        best = grasps.best
        assert best is not None, grasps.render()
        self.assertLess(_tilt_deg(best), 1.0, grasps.render())
        for candidate in grasps.candidates:
            self.assertGreaterEqual(candidate.clearance_mm, float(_hande().grasping.support.min_clearance_mm))


class TheSupportIsResolvedAsThePickLoopResolvesItTests(unittest.TestCase):
    """``resolve_support_plane`` over the target's own cloud, behind the pick loop's 25 mm extent rule."""

    def test_a_part_standing_on_a_block_is_planned_on_the_block(self) -> None:
        cloud = _cube(40.0, base_mm=30.0)
        scene = Scene.from_robot_config(_hande(), cloud)
        self.assertAlmostEqual(30.0, scene.support_height_mm, places=6)
        grasps = scene.grasps()
        self.assertTrue(grasps.candidates, grasps.render())
        for candidate in grasps.candidates:
            self.assertGreaterEqual(float(candidate.position_mm[2]), 30.0)

    def test_a_top_face_alone_leaves_the_declared_height(self) -> None:
        """One face seen head-on has not seen what the part stands on; its lowest point is the face."""
        top = _cube(40.0)
        top = top[top[:, 2] == 40.0]
        self.assertEqual(float(_hande().grasping.support.height_mm),
                         Scene.from_robot_config(_hande(), top).support_height_mm)

    def test_it_never_lowers_the_declared_height(self) -> None:
        cfg = _replace(_hande(), "grasping.support.height_mm", 10.0)
        self.assertEqual(10.0, Scene.from_robot_config(cfg, _cube(40.0, base_mm=2.0)).support_height_mm)

    def test_refine_from_target_off_keeps_the_declared_height(self) -> None:
        cfg = _replace(_hande(), "grasping.support.refine_from_target", False)
        self.assertEqual(0.0, Scene.from_robot_config(cfg, _cube(40.0, base_mm=30.0)).support_height_mm)

    def test_a_container_floor_replaces_the_workspace_height(self) -> None:
        cfg = _replace(_hande(), "grasping.support.container.floor_height_mm", 15.0)
        self.assertEqual(15.0, Scene.from_robot_config(cfg, _cube(40.0, base_mm=12.0)).support_height_mm)

    def test_the_extent_rule_is_the_pick_loops(self) -> None:
        from src.robot.grasping import scene as scene_module
        from src.robot.grasping.loop import pick_loop

        self.assertEqual(pick_loop._MIN_CLOUD_EXTENT_FOR_SUPPORT_MM,  # noqa: SLF001
                         scene_module._MIN_CLOUD_EXTENT_FOR_SUPPORT_MM)  # noqa: SLF001


class TheFloorIsTheSupportsOwnTests(unittest.TestCase):
    """The floor holds how high a real table reads, and not the slab's clearance (review of 2026-09-23).

    It was the larger of the stage default and ``plane_clearance_mm``, which robot.yaml tells a cell to raise by the
    sink of its slab: with the slab at -50 and the clearance at 55 the floor stood 55 mm over the table and a 40 mm
    cube had no grasp.
    """

    def test_the_floor_is_its_own_whatever_the_slabs_clearance(self) -> None:
        from src.robot.grasping.scene import SUPPORT_READ_ERROR_MM

        path = "safety.planning_world.perceived.plane_clearance_mm"
        for clearance in (None, 0.0, 8.0, 55.0):
            cfg = _hande() if clearance is None else _replace(_hande(), path, clearance)
            with self.subTest(plane_clearance_mm=clearance):
                self.assertEqual(SUPPORT_READ_ERROR_MM, Scene.from_robot_config(cfg, _cube(40.0)).floor_margin_mm)
        self.assertEqual(5.0, SUPPORT_READ_ERROR_MM)

    def test_a_40_mm_cube_on_a_cell_whose_slab_is_sunk_50_mm_is_grasped_from_above(self) -> None:
        from src.config.schema.robot.safety_schema import SupportPlaneConfig

        cfg = _replace(_hande(), "safety.planning_world.support_plane", SupportPlaneConfig(height_mm=-50.0))
        cfg = _replace(cfg, "safety.planning_world.perceived.plane_clearance_mm", 55.0)

        grasps = Scene.from_robot_config(cfg, _cube(40.0)).grasps()

        self.assertIsNotNone(grasps.best, grasps.render())
        self.assertTrue(any(_tilt_deg(c) < 1.0 for c in grasps.candidates), grasps.render())

    def test_the_inflation_is_the_one_the_cell_hands_its_calculator(self) -> None:
        cfg = _replace(_hande(), "grasping.geometry.inflate_mm", 3.0)
        self.assertEqual(3.0, Scene.from_robot_config(cfg, _cube(40.0)).inflate_mm)

    def test_a_table_read_4_mm_high_is_bench_and_not_part(self) -> None:
        """A hand-eye height error puts the bench around the part 4 mm up; the grasp spans the part, not the bench."""
        rng = np.random.default_rng(7)
        ring = np.column_stack([rng.uniform(_CENTRE[0] - 35.0, _CENTRE[0] + 35.0, 3000),
                                rng.uniform(_CENTRE[1] - 35.0, _CENTRE[1] + 35.0, 3000),
                                rng.normal(4.0, 0.2, 3000)])
        outside = (np.abs(ring[:, 0] - _CENTRE[0]) > 20.0) | (np.abs(ring[:, 1] - _CENTRE[1]) > 20.0)
        cloud = np.vstack([_cube(40.0), ring[outside]])
        grasps = Scene.from_robot_config(_hande(), cloud).grasps()
        best = grasps.best
        assert best is not None, grasps.render()
        self.assertLessEqual(best.grip_width_mm, 44.0, grasps.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
