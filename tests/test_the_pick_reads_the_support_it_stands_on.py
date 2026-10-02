"""The pick plans on what its part stands on, as the camera world found it (cell fixes Track C, contract 7).

On the owner's cell the declared bench is at 0 mm and the parts stand on a 55 mm foam mat. The camera world now finds
the mat itself (Track S); the pick loop reads the same model off the frames its looks took, with the live world's own
limits and tuning:

* the support a part's grasps are planned on is the higher of the declared one and the model's reading under the part's
  foot, and the model rides to the calculator beside the plane (``support_model``), which reads it for its obstacles,
  its SFE support and its envelope filter (Track A);
* the push plans on the plane of the surface under the part through its local reading, bounds its landing by that
  surface's own pixels, and rides its finger at least the guard's distance plus 1 mm over the highest solid the world
  holds under the hand's sweep;
* where no surface lies under the part, the push plans on today's table and plane; and a cell whose world reads no
  support surfaces (or has no live world) plans exactly as before.

The scene is ``tests/_wrist_views.py`` with a mat under the cube: the mat is rendered, never segmented.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingSupportConfig
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.push_budgets import PushBudgets
from src.robot.grasping.recovery.push_gate import PushCell, PushGate
from src.robot.grasping.recovery.push_planner import AxisBox
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.safety.planning.perceived import WorldBuildLimits, WorldBuildTuning
from tests._wrist_views import Box, LookCalculator, LookingArm, WristCamera, camera_to_tool
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import HAND_E, LOOK_PLUS_Y, _Policy, _poses, _World
from tests.test_the_push_sees_unprompted_neighbours import HAND_E as PUSH_HAND

MAT_MM = 55.0
#: A foam mat under the work, as the owner's: 55 mm thick, far wider than a part.
MAT = Box((-250.0, -950.0, 0.0), (250.0, -450.0, MAT_MM), "mat")
#: The cube standing on the mat, and a post beside it the push's evidence reads.
PART_ON_MAT = Box((-20.0, -720.0, MAT_MM), (20.0, -680.0, MAT_MM + 40.0), "part")
POST_ON_MAT = Box((-62.0, -722.0, MAT_MM), (-42.0, -714.0, MAT_MM + 30.0), "post")
PART_ON_BENCH = Box((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0), "part")
POST_ON_BENCH = Box((-62.0, -722.0, 0.0), (-42.0, -714.0, 30.0), "post")

LIMITS = WorldBuildLimits(x_mm=(-460.0, 440.0), y_mm=(-935.0, -235.0), z_mm=(-100.0, 600.0), support_plane_top_mm=0.0)
#: The world's tuning as the owner's cell builds it: support surfaces on, the 2 mm allowance, margin 15.
TUNING = WorldBuildTuning(pixel_stride=2, voxel_size_mm=10.0, cluster_voxel_mm=25.0, min_points=12, margin_mm=15.0,
                          max_boxes=64, floor_to_plane=True, support_surfaces=True, support_allowance_mm=2.0)
GUARD_MM = 5.0


class _SupportWorld(_World):
    def __init__(self, tuning: WorldBuildTuning = TUNING) -> None:
        super().__init__()
        self.limits = LIMITS
        self.tuning = tuning


class _MatCamera(WristCamera):
    """The wrist D415, whose detector never segments the mat it renders."""

    def acquire(self) -> Any:
        frame = super().acquire()
        kept = tuple(seg for seg in frame.segmentations if getattr(seg, "label", "") != "mat")
        return type(frame)(depth_map=frame.depth_map, intrinsics=frame.intrinsics, segmentations=kept, rgb=frame.rgb,
                           timestamp=frame.timestamp, tool_pose=frame.tool_pose)


class _Collides(LookCalculator):
    """Every grasp of the part collides (``ALL_COLLIDED``), as the cell's calculator says of a boxed-in part."""

    def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
        super().compute_result(seg, *args, **kwargs)
        return GraspResult(reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED))


class _Cell:
    def __init__(self, boxes: tuple[Box, ...], *, world: Any = None, collides: bool = False,
                 push: bool = False) -> None:
        self.arm = LookingArm(_poses())
        if world is not False:
            self.arm.live_planner_world = world if world is not None else _SupportWorld()  # type: ignore[attr-defined]
        self.camera = _MatCamera(self.arm, boxes=boxes)
        self.calculator = (_Collides if collides else LookCalculator)(self.camera)
        self.policy = _Policy(self.arm)
        self.orchestrator = BinPickingOrchestrator(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()), policy=self.policy,  # type: ignore[arg-type]
            max_attempts=2, primary_camera_id="wrist", gripper_model=HAND_E, looks=(LOOK_PLUS_Y,), target_label="part",
            support_config=GraspingSupportConfig(height_mm=0.0, refine_from_target=False),
        )
        if push:
            self.orchestrator.push_gate = PushGate(budgets=PushBudgets(), distance_mm=30.0, cell=PushCell(
                hand=PUSH_HAND, workspace=AxisBox((-460.0, -935.0, -100.0), (440.0, -235.0, 600.0)),
                hand_clearance_mm=20.0))

    def part_calls(self) -> list[dict[str, Any]]:
        return [kwargs for _number, label, kwargs in self.calculator.calls if label == "part"]


class TheGraspsArePlannedOnTheMatTests(unittest.TestCase):
    def test_the_support_a_part_is_planned_on_is_the_mat_s_reading_under_it(self) -> None:
        """Red before: the declared 0 mm stood, and the calculator was handed no model."""
        cell = _Cell((MAT, PART_ON_MAT))

        cell.orchestrator.run()

        (kwargs, *_) = cell.part_calls()
        self.assertAlmostEqual(MAT_MM, float(kwargs["support_plane"].offset_mm), delta=1.0)
        model = kwargs["support_model"]
        self.assertTrue(model.surfaces, model.render())
        self.assertAlmostEqual(MAT_MM, float(model.height_under(np.array([[0.0, -700.0]]))), delta=1.0)

    def test_on_the_bare_bench_the_declared_support_stands(self) -> None:
        """A bench read at its declared height may be held as the bench or as a surface at it (Track S): either way the
        part is planned on the declared 0 mm."""
        cell = _Cell((PART_ON_BENCH,))

        cell.orchestrator.run()

        (kwargs, *_) = cell.part_calls()
        self.assertAlmostEqual(0.0, float(kwargs["support_plane"].offset_mm), delta=1e-6)
        under = kwargs["support_model"].height_under(np.array([[0.0, -700.0]]))
        self.assertTrue(under is None or abs(float(under)) < 1.0, under)

    def test_a_world_that_reads_no_support_surfaces_hands_no_model(self) -> None:
        cell = _Cell((MAT, PART_ON_MAT), world=_SupportWorld(WorldBuildTuning(pixel_stride=2, margin_mm=15.0)))

        cell.orchestrator.run()

        (kwargs, *_) = cell.part_calls()
        self.assertNotIn("support_model", kwargs)
        self.assertAlmostEqual(0.0, float(kwargs["support_plane"].offset_mm), delta=1e-6)

    def test_an_arm_with_no_live_world_plans_as_before(self) -> None:
        cell = _Cell((MAT, PART_ON_MAT), world=False)

        cell.orchestrator.run()

        (kwargs, *_) = cell.part_calls()
        self.assertNotIn("support_model", kwargs)


class ThePushPlansOnTheSurfaceTests(unittest.TestCase):
    def _planned(self, boxes: tuple[Box, ...], **keywords: Any) -> dict[str, Any]:
        cell = _Cell(boxes, collides=True, push=True, **keywords)
        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", wraps=push_planner.plan_push) as planned:
            cell.orchestrator.run()
        self.assertTrue(planned.called, [push.sentence for push in cell.orchestrator.pushes])
        return dict(planned.call_args.kwargs)

    def test_the_push_reads_the_mat_s_plane_its_pixels_and_rides_over_its_solid(self) -> None:
        """Red before: the plane at the judged height, every view's pixels in a band of it, the finger at 10 mm."""
        kwargs = self._planned((MAT, PART_ON_MAT, POST_ON_MAT))

        normal = np.asarray(kwargs["support_normal"], dtype=np.float64)
        self.assertGreater(float(normal[2]), 0.999)
        self.assertAlmostEqual(MAT_MM, float(kwargs["support_offset_mm"]) / float(normal[2]), delta=1.0)
        table = np.asarray(kwargs["table_points_mm"], dtype=np.float64)
        self.assertGreater(len(table), 100)
        self.assertTrue(np.all(np.abs(table[:, 2] - MAT_MM) <= 5.0 + 1e-6), "a table point off the mat's band")
        self.assertTrue(np.all((table[:, 0] >= MAT.low[0] - 1.0) & (table[:, 0] <= MAT.high[0] + 1.0)))
        # The mat's solid stands its band and the allowance over its reading: the finger keeps the guard's 5 mm and 1.
        self.assertGreaterEqual(float(kwargs["finger_floor_mm"]), 5.0 + 2.0 + GUARD_MM + 1.0 - 0.5)
        self.assertLess(float(kwargs["finger_floor_mm"]), 20.0)

    def test_on_the_bare_bench_the_finger_rides_over_the_bench_solid(self) -> None:
        """Red before: the finger rode 10 mm over a bench the guard now holds to 7 mm, 3 mm from its solid."""
        kwargs = self._planned((PART_ON_BENCH, POST_ON_BENCH))

        self.assertAlmostEqual(0.0, float(kwargs["support_offset_mm"]), delta=0.5)
        np.testing.assert_allclose(kwargs["support_normal"], (0.0, 0.0, 1.0), atol=1e-6)
        # The bench solid is held to its band and the allowance (7 mm): the finger keeps the guard's distance over it.
        self.assertAlmostEqual(5.0 + 2.0 + GUARD_MM + 1.0, float(kwargs["finger_floor_mm"]), delta=0.5)

    def test_a_world_that_reads_no_support_surfaces_pushes_as_before(self) -> None:
        """No model: today's plane at the judged height, today's table, and the planner's own finger floor."""
        kwargs = self._planned((PART_ON_BENCH, POST_ON_BENCH),
                               world=_SupportWorld(WorldBuildTuning(pixel_stride=2, margin_mm=15.0)))

        self.assertAlmostEqual(0.0, float(kwargs["support_offset_mm"]), delta=1e-9)
        np.testing.assert_allclose(kwargs["support_normal"], (0.0, 0.0, 1.0))
        self.assertNotIn("finger_floor_mm", kwargs)
        table = np.asarray(kwargs["table_points_mm"], dtype=np.float64)
        self.assertTrue(np.all(np.abs(table[:, 2]) <= 5.0 + 1e-6))


if __name__ == "__main__":
    unittest.main()
