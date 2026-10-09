"""The dense samples and the geometry-first contacts are not computed where SFE replaces them, and nothing else changes.

Where the support-footprint stage runs and its fallback is off (every cell's), its candidates replace every candidate
of the stages before it, so the calculator computed the dense samples and the geometry-first contacts and threw them
away: 0.84 s of every look on the owner's cell (2026-10-08), and the one branch of the calculator that read a clock (the
dense budget). They are not computed now, and the telemetry says so (``geometry_skipped``); the grasps, the reasons and
SFE's counts are the ones a calculator that computed them gives. Where SFE is not ready the dense path runs as before.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.geometry import Frame, Transform


def _look(side_mm: float = 40.0) -> tuple[Any, Any, Any]:
    """A cube ``side_mm`` across on the bench, ray-cast from 45 degrees over it: its mask, the depth and CAMERA to
    BASE."""
    from tests._wrist_views import Box, camera_looking_at, render

    half = side_mm / 2.0
    cube = Box((-half, -700.0 - half, 0.0), (half, -700.0 + half, side_mm), "part")
    looking = camera_looking_at((0.0, -700.0, 20.0), bearing_deg=90.0, elevation_deg=45.0, range_mm=520.0)
    depth, hit = render(looking, (cube,))
    return SimpleNamespace(mask=hit == 0, label="part", score=0.9), depth, looking


def _calculator(**extra: Any) -> Any:
    """The owner's calculator over the Hand-E: SFE with the scene and side approaches, as the cell builds it."""
    from src.robot.grasping.calculator_factory import build_calculator
    from tests._wrist_views import K
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell, owner_rules

    cfg = hande_cell()
    return build_calculator(cfg, camera_matrix=K, max_grip_width_mm=cfg.gripper.max_width_mm,
                            min_grip_width_mm=cfg.gripper.min_width_mm, support_footprint_geometry=True,
                            support_footprint_inflate_mm=0.0, side_approaches=True, scene_obstacles=owner_rules(),
                            **extra)


def _compute(calculator: Any, *, plane: bool = True, side_mm: float = 40.0) -> Any:
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.grasping.collision import resolve_support_plane
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    cfg = hande_cell()
    seg, depth, looking = _look(side_mm)
    keywords: dict[str, Any] = {}
    if plane:
        keywords["support_plane"] = resolve_support_plane(declared_height_mm=0.0, normal=(0.0, 0.0, 1.0),
                                                          refine_from_target=False).plane
        keywords["min_table_clearance_mm"] = float(cfg.grasping.support.min_clearance_mm)
    return calculator.compute_result(
        seg, depth, pixel_to_mm=None, dense_sampling=True, other_object_masks=[],
        camera_to_base=Transform.from_matrix(looking, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry), **keywords)


def _grasps(result: Any) -> str:
    return json.dumps([[c.position.tobytes().hex(), c.approach.tobytes().hex(), c.axis.tobytes().hex(),
                        np.float64(c.grip_width_mm).tobytes().hex(), np.float64(c.score).tobytes().hex()]
                       for c in result.candidates])


class WhereSfeReplacesThemTests(unittest.TestCase):
    def test_neither_the_dense_samples_nor_the_geometry_first_contacts_are_computed(self) -> None:
        calculator = _calculator()
        never = AssertionError("computed and thrown away")
        with mock.patch("src.robot.grasping.generation.calculator.dense_surface_samples", side_effect=never), \
                mock.patch.object(calculator._generator, "dense_geometry_poses", side_effect=never), \
                mock.patch.object(calculator._generator, "geometry_first_poses", side_effect=never):
            result = _compute(calculator)

        self.assertTrue(result.candidates, result.reasons)
        self.assertEqual("support_footprint", result.telemetry["geometry_stage"])
        self.assertEqual("replaced_by_support_footprint", result.telemetry["geometry_skipped"])
        self.assertEqual(0, result.telemetry["candidates_geometry"])
        self.assertNotIn("dense_runtime_ms", result.telemetry)

    def test_the_grasps_reasons_and_counts_are_those_of_a_calculator_that_computed_them(self) -> None:
        computing = _compute(_calculator(support_footprint_fallback=True))
        skipping = _compute(_calculator())

        self.assertIn("dense_runtime_ms", computing.telemetry, "the control computed the dense samples")
        self.assertNotIn("geometry_skipped", computing.telemetry)
        self.assertTrue(skipping.candidates)
        self.assertEqual(_grasps(computing), _grasps(skipping))
        self.assertEqual(computing.reasons, skipping.reasons)
        self.assertEqual(computing.telemetry["support_footprint_refused"],
                         skipping.telemetry["support_footprint_refused"])
        self.assertEqual(computing.telemetry["support_footprint_candidates"],
                         skipping.telemetry["support_footprint_candidates"])


class WhereSfeIsNotReadyTests(unittest.TestCase):
    def test_the_dense_path_runs_as_before(self) -> None:
        from src.robot.grasping.generation import calculator as module

        calculator = _calculator()
        with mock.patch.object(module, "dense_surface_samples", wraps=module.dense_surface_samples) as sampled:
            # A 24 mm cube, whose silhouette the Hand-E closes across, so the calculator reaches its geometry stages.
            result = _compute(calculator, plane=False, side_mm=24.0)

        sampled.assert_called_once()
        self.assertIn("dense_runtime_ms", result.telemetry)
        self.assertNotIn("geometry_skipped", result.telemetry)
        self.assertFalse(result.telemetry.get("support_footprint_ready"), result.telemetry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
