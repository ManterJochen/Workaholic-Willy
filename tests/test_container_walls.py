"""Container walls as collision geometry: the obstacle the candidate filter never had.

Measured motivation, not a hunch. Replaying the datagen reference verdict on exactly those rank-0
winners that lose the ranking gap on ``finger_collision`` put **60.9 %** of them against a bin wall
rather than an object -- and inside the ``bin`` family, 14 of 14. No perception change can reach
that, because no detector segments a wall as an instance, so the walls have to be declared.

These pin the geometry (a shell, not a surface; floor and rim respected), the fail-closed config,
and -- the part that matters most for a cell that already runs something else -- that walls are
ADDITIVE to the neighbour cloud rather than an alternative to it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import (
    FusionGeometryConfig,
    GraspingContainerConfig,
    GraspingSupportConfig,
)
from src.geometry import Frame, Transform
from src.robot.grasping.collision import container_wall_points_base_mm
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.perception import CameraObservation, PerceptionFrame

_LOGGER = "src.robot.grasping.loop.pick_loop"
_LOW = (300.0, -100.0, 0.0)
_HIGH = (600.0, 100.0, 147.0)      # the D5 reference bin: 300 x 200 x 147 mm interior


class WallGeometryTests(unittest.TestCase):
    def test_no_point_lands_in_the_interior(self) -> None:
        """A wall point inside the bin would reject every grasp the bin exists to hold."""

        points = container_wall_points_base_mm(_LOW, _HIGH, thickness_mm=5.0, sample_mm=8.0)

        low = np.asarray(_LOW)
        high = np.asarray(_HIGH)
        inside = np.all((points[:, :2] > low[:2] + 1e-9) & (points[:, :2] < high[:2] - 1e-9), axis=1)
        self.assertEqual(int(inside.sum()), 0, "wall points inside the container's own interior")

    def test_it_spans_floor_to_rim_and_no_further(self) -> None:
        points = container_wall_points_base_mm(_LOW, _HIGH, thickness_mm=5.0, sample_mm=8.0)

        self.assertAlmostEqual(float(points[:, 2].min()), _LOW[2])
        self.assertAlmostEqual(float(points[:, 2].max()), _HIGH[2])

    def test_all_four_walls_are_present(self) -> None:
        points = container_wall_points_base_mm(_LOW, _HIGH, thickness_mm=5.0, sample_mm=8.0)

        self.assertLess(float(points[:, 0].min()), _LOW[0] + 1e-9)
        self.assertGreater(float(points[:, 0].max()), _HIGH[0] - 1e-9)
        self.assertLess(float(points[:, 1].min()), _LOW[1] + 1e-9)
        self.assertGreater(float(points[:, 1].max()), _HIGH[1] - 1e-9)

    def test_it_is_a_shell_not_a_surface(self) -> None:
        """A finger standing INSIDE thick wall material has to hit something."""

        thin = container_wall_points_base_mm(_LOW, _HIGH, thickness_mm=1.0, sample_mm=8.0)
        thick = container_wall_points_base_mm(_LOW, _HIGH, thickness_mm=40.0, sample_mm=8.0)

        self.assertGreater(len(thick), len(thin))
        self.assertAlmostEqual(float(thick[:, 0].min()), _LOW[0] - 40.0)
        # Points exist strictly BETWEEN the two faces of the 40 mm wall, not only on them.
        mid = (thick[:, 0] > _LOW[0] - 40.0 + 1e-6) & (thick[:, 0] < _LOW[0] - 1e-6)
        self.assertTrue(bool(mid.any()), "the wall is hollow: only its two faces were sampled")

    def test_it_is_deterministic(self) -> None:
        first = container_wall_points_base_mm(_LOW, _HIGH)
        second = container_wall_points_base_mm(_LOW, _HIGH)

        np.testing.assert_array_equal(first, second)

    def test_a_degenerate_box_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            container_wall_points_base_mm((0.0, 0.0, 0.0), (0.0, 100.0, 100.0))


class ConfigIsFailClosedTests(unittest.TestCase):
    def test_walls_without_a_box_are_refused_at_load(self) -> None:
        """A cell that believes its walls are protected and is not is worse than one that knows."""

        with self.assertRaises(ValueError) as ctx:
            GraspingContainerConfig(wall_collision_enabled=True)

        self.assertIn("interior_min_mm", str(ctx.exception))

    def test_an_inverted_box_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            GraspingContainerConfig(
                wall_collision_enabled=True, interior_min_mm=_HIGH, interior_max_mm=_LOW
            )

    def test_declaring_the_box_alone_changes_nothing(self) -> None:
        """Describing the bin and changing grasping behaviour are two decisions."""

        config = GraspingContainerConfig(interior_min_mm=_LOW, interior_max_mm=_HIGH)

        self.assertFalse(config.wall_collision_enabled)


def _frame(*, boxes: tuple[tuple[int, int], ...] = ((10, 22),)) -> PerceptionFrame:
    segmentations = []
    for lo, hi in boxes:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[lo:hi, lo:hi] = 1
        segmentations.append(SimpleNamespace(mask=mask))
    return PerceptionFrame(
        depth_map=np.full((32, 32), 500.0, dtype=np.float64),
        intrinsics=np.array(
            [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
        segmentations=tuple(segmentations),
    )


class _Recording:
    render_debug_images = False

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def compute_result(self, *_args: object, **kwargs: object) -> GraspResult:
        self.kwargs.append(dict(kwargs))
        return GraspResult(candidates=(), reasons=(), telemetry={})


class _Resolver:
    def camera_to_base_for_frame(self, _frame_in, *, arm=None):  # noqa: ANN001, ANN202
        return Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)


def _support(**container) -> GraspingSupportConfig:
    return GraspingSupportConfig(container=GraspingContainerConfig(**container))


def _orchestrator(calculator, **kwargs) -> BinPickingOrchestrator:
    return BinPickingOrchestrator(
        arm=SimpleNamespace(),  # type: ignore[arg-type]
        calculator=calculator,  # type: ignore[arg-type]
        perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        frame_resolver=_Resolver(),  # type: ignore[arg-type]
        **kwargs,
    )


class PickPathTests(unittest.TestCase):
    def test_default_config_adds_nothing(self) -> None:
        calculator = _Recording()
        orch = _orchestrator(calculator, support_config=_support())

        orch._best_result_over_segmentations(_frame())

        self.assertNotIn("scene_points_mm", calculator.kwargs[0])

    def test_enabled_walls_reach_the_collision_filter(self) -> None:
        calculator = _Recording()
        orch = _orchestrator(
            calculator,
            support_config=_support(
                wall_collision_enabled=True, interior_min_mm=_LOW, interior_max_mm=_HIGH
            ),
        )

        orch._best_result_over_segmentations(_frame())

        walls = calculator.kwargs[0].get("rigid_obstacle_points_mm")
        self.assertIsNotNone(walls, "the declared walls never reached the candidate filter")
        assert walls is not None
        self.assertGreater(len(walls), 0)
        self.assertNotIn(
            "scene_points_mm", calculator.kwargs[0],
            "walls must not ride the OBSERVED cloud -- the generator dilates that one by 12 mm",
        )

    def test_walls_stack_on_top_of_the_fused_neighbour_cloud(self) -> None:
        """The regression this test exists for: enabling one obstacle must not drop the other."""

        class _Rig:
            def acquire_all(self):  # noqa: ANN202
                return (CameraObservation(camera_id="left", frame=_frame(boxes=((10, 22), (24, 30)))),)

        walls_only = _Recording()
        _orchestrator(
            walls_only,
            support_config=_support(
                wall_collision_enabled=True, interior_min_mm=_LOW, interior_max_mm=_HIGH
            ),
        )._best_result_over_segmentations(_frame(boxes=((10, 22),)))

        both = _Recording()
        _orchestrator(
            both,
            support_config=_support(
                wall_collision_enabled=True, interior_min_mm=_LOW, interior_max_mm=_HIGH
            ),
            multi_camera_perception=_Rig(),
            camera_frame_resolvers={"left": _Resolver()},
            fusion_geometry_config=FusionGeometryConfig(
                enabled=True, neighbour_scene_enabled=True
            ),
        )._best_result_over_segmentations(_frame(boxes=((10, 22),)))

        self.assertNotIn("scene_points_mm", walls_only.kwargs[0])
        self.assertIn(
            "scene_points_mm", both.kwargs[0],
            "turning on walls swallowed the fused neighbour cloud",
        )
        self.assertIn(
            "rigid_obstacle_points_mm", both.kwargs[0],
            "turning on fusion swallowed the declared walls",
        )
        np.testing.assert_array_equal(
            walls_only.kwargs[0]["rigid_obstacle_points_mm"],
            both.kwargs[0]["rigid_obstacle_points_mm"],
        )

    def test_no_transform_is_said_out_loud(self) -> None:
        calculator = _Recording()
        orch = BinPickingOrchestrator(
            arm=SimpleNamespace(),  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
            support_config=_support(
                wall_collision_enabled=True, interior_min_mm=_LOW, interior_max_mm=_HIGH
            ),
        )

        with self.assertLogs(_LOGGER, level="WARNING") as captured:
            orch._best_result_over_segmentations(_frame())

        self.assertTrue(any("wall-collision check is NOT running" in line
                            for line in captured.output))

    def test_the_walls_are_built_once(self) -> None:
        orch = _orchestrator(
            _Recording(),
            support_config=_support(
                wall_collision_enabled=True, interior_min_mm=_LOW, interior_max_mm=_HIGH
            ),
        )

        first = orch._container_wall_points_base_mm()
        second = orch._container_wall_points_base_mm()

        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
