"""The support plane: a declared height, raised by what the cameras saw. Never lowered.

Every assertion here corresponds to a measured fact over 1073 reference objects, and the numbers are in
the module docstring. The one that matters most is the asymmetry: an estimate that is too HIGH refuses
grasps, an estimate that is too LOW drives a finger through a surface.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.collision import resolve_support_plane


def _cloud(*heights: float) -> np.ndarray:
    return np.array([[0.0, 0.0, z] for z in heights])


class ResolverTests(unittest.TestCase):
    def test_nothing_observed_gives_the_declared_height(self) -> None:
        """Byte-identical to every caller that predates this: no cameras, no refinement."""
        r = resolve_support_plane(declared_height_mm=0.0)
        self.assertEqual(r.height_mm, 0.0)
        self.assertEqual(r.source, "declared")
        self.assertIsNone(r.observed_mm)

    def test_a_part_standing_on_another_part_raises_the_plane(self) -> None:
        """The case a declared constant is too low for 100 % of the time."""
        r = resolve_support_plane(declared_height_mm=0.0,
                                  target_clouds_base_mm=[_cloud(40.0, 90.0)])
        self.assertAlmostEqual(r.height_mm, 40.0)
        self.assertEqual(r.source, "observed")

    def test_an_observation_below_the_declared_surface_is_ignored(self) -> None:
        """Depth noise looking through the table must never lower the floor."""
        r = resolve_support_plane(declared_height_mm=0.0,
                                  target_clouds_base_mm=[_cloud(-8.0, 50.0)])
        self.assertEqual(r.height_mm, 0.0)
        self.assertEqual(r.source, "declared")
        self.assertAlmostEqual(r.observed_mm or 0.0, -8.0,
                               msg="the rejected observation is still reported, not swallowed")

    def test_the_lowest_camera_wins_not_the_average(self) -> None:
        """A camera behind a bin wall reports the rim; averaging would spread its error instead of
        discarding it. Measured: p90 50.6 mm single-view against 13.3 mm fused, in the bin family."""
        blocked = _cloud(48.0, 90.0)        # sees the part over the wall
        clear = _cloud(2.0, 90.0)           # sees its base
        r = resolve_support_plane(declared_height_mm=0.0,
                                  target_clouds_base_mm=[blocked, clear])
        self.assertAlmostEqual(r.height_mm, 2.0)

    def test_a_container_floor_replaces_the_workspace_height(self) -> None:
        """A KLT on the table raises what its contents rest on by its own floor thickness."""
        r = resolve_support_plane(declared_height_mm=0.0, container_floor_mm=12.0)
        self.assertAlmostEqual(r.height_mm, 12.0)
        r = resolve_support_plane(declared_height_mm=0.0, container_floor_mm=12.0,
                                  target_clouds_base_mm=[_cloud(6.0, 80.0)])
        self.assertAlmostEqual(r.height_mm, 12.0,
                               msg="an observation below the declared floor must not lower it")

    def test_a_tilted_surface_measures_along_its_own_normal(self) -> None:
        """A tilted tray is a normal, not a special case."""
        normal = np.array([0.0, np.sin(np.radians(20.0)), np.cos(np.radians(20.0))])
        point = normal * 30.0
        r = resolve_support_plane(declared_height_mm=0.0, normal=normal,
                                  target_clouds_base_mm=[np.array([point, point + normal * 50.0])])
        self.assertAlmostEqual(r.height_mm, 30.0, places=6)
        np.testing.assert_allclose(r.plane.normal, normal / np.linalg.norm(normal), atol=1e-12)

    def test_an_empty_cloud_does_not_crash_or_invent_a_height(self) -> None:
        r = resolve_support_plane(declared_height_mm=5.0,
                                  target_clouds_base_mm=[np.zeros((0, 3))])
        self.assertEqual(r.height_mm, 5.0)
        self.assertEqual(r.source, "declared")

    def test_a_zero_normal_is_refused_rather_than_normalised_to_something(self) -> None:
        with self.assertRaises(ValueError):
            resolve_support_plane(normal=(0.0, 0.0, 0.0))

    def test_the_telemetry_carries_the_reason_not_just_the_number(self) -> None:
        """A clearance rejection nobody can trace to a height is how this check stayed off for years."""
        r = resolve_support_plane(declared_height_mm=0.0,
                                  target_clouds_base_mm=[_cloud(40.0, 90.0)])
        t = r.as_telemetry()
        self.assertEqual(
            set(t), {"support_height_mm", "support_declared_mm", "support_observed_mm",
                     "support_source"})
        self.assertEqual(t["support_source"], "observed")


class TheBenchIsThePlannerWorldsTests(unittest.TestCase):
    """The table a cell's parts stand on is its planner world's bench, 0 mm where none is declared (2026-10-17)."""

    def _on_a_bench(self, height_mm: float):  # noqa: ANN202
        from src.config.schema.robot import RobotConfig
        from src.config.schema.robot.safety_schema import SupportPlaneConfig

        bare = RobotConfig()
        world = bare.safety.planning_world.model_copy(update={"support_plane": SupportPlaneConfig(height_mm=height_mm)})
        return bare.model_copy(update={"safety": bare.safety.model_copy(update={"planning_world": world})})

    def test_the_bench_is_the_slabs_top_and_the_base_plate_where_none_is_declared(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.grasping.collision.support_resolver import bench_height_mm

        self.assertEqual(0.0, bench_height_mm(RobotConfig()))
        self.assertEqual(0.0, bench_height_mm(None))
        self.assertEqual(-12.5, bench_height_mm(self._on_a_bench(-12.5)))

    def test_the_declared_support_is_the_bench_and_the_bin_level_with_the_guards_floor(self) -> None:
        from src.robot.grasping.collision.support_resolver import DeclaredSupport

        declared = DeclaredSupport.of(self._on_a_bench(-12.5))
        self.assertEqual((-12.5, None, (0.0, 0.0, 1.0), 1.0),
                         (declared.height_mm, declared.floor_mm, declared.normal, declared.min_clearance_mm))
        self.assertEqual(-12.5, declared.resolve().height_mm, "the control: nothing seen, the bench stands")
        self.assertEqual(30.0, declared.resolve([_cloud(30.0, 70.0)]).height_mm, "a part's own foot raises it")

    def test_the_builder_hands_the_pick_loop_the_bench(self) -> None:
        from types import SimpleNamespace

        from src.robot.execution.autonomous_grasp.builders import apply_orchestrator_overlays
        from src.robot.execution.autonomous_grasp.config import GraspMode

        tree = self._on_a_bench(-12.5)
        runtime = SimpleNamespace(orchestrator=SimpleNamespace())
        apply_orchestrator_overlays(runtime, tree.grasping, resolved_mode=GraspMode.EASY,  # type: ignore[arg-type]
                                    robot_cfg=tree)
        self.assertEqual(-12.5, runtime.orchestrator.declared_support.height_mm)


class OrchestratorSeamTests(unittest.TestCase):
    def test_both_seams_default_to_none_so_the_path_is_unchanged(self) -> None:
        """The keys are only added when configured; nothing existing moves."""
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        fields = BinPickingOrchestrator.__dataclass_fields__
        self.assertIsNone(fields["gripper_model"].default)
        self.assertIsNone(fields["declared_support"].default)


if __name__ == "__main__":
    unittest.main()
