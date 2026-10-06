"""The push sees the neighbours no prompt named, and keeps the owner's 10 mm beside the part it pushes (cell fixes Track C).

Two things, both from the owner's cell. Every prompt there grounded one box, so the push knew no neighbour at all: the
segmented others were none, and "a neighbour within 25 mm" could never be evidenced (RC4). Now the push reads what the
calculator saw beside the part (the fix plan's contract 2): Track A's obstacle points, and every point the camera world's
rule keeps there with no voxel threshold, so a neighbour the voxel rule dropped still refuses the push's path.

And the hand's clearance. The open hand used to keep the camera world's margin plus the line clearance, 20 mm on the
owner's cell, from every neighbour point, so no one-neighbour scene that triggers a push also planned one (fix plan,
Track P). The owner decided 10 mm beside the pushed part (2026-10-02): a neighbour point within the margin the push's
keep-out grows round the part and its path, which the guard does not see during the push, is kept 10 mm from the hand;
a point beyond it keeps the line judge's clearance, so the planner plans no push the guard would predictably refuse
once a leg is under way.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from src.config.loader import load_robot_config, reload_config
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.push_gate import PushCell
from src.robot.grasping.recovery.push_planner import AxisBox, PushHand, PushPlan, PushRefusal, plan_push
from src.robot.grasping.types.feedback import GraspResult
from tests._wrist_views import CUBE
from tests.test_a_boxed_in_part_is_pushed_inside_the_pick import POST, _Bench, _Jammed, _PushCell

#: The Hand-E as the push sweeps it (config/grippers/robotiq_hande.yaml).
HAND_E = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45, finger_length_mm=36.53,
                  palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=75.0)
#: The owner's workspace, its floor well under the mat.
WORKSPACE = AxisBox((-460.0, -935.0, 0.0), (440.0, -235.0, 600.0))
#: The mat's top, flat here, and where the part stands on it.
MAT_MM = 55.0
CX, CY = -150.0, -650.0


def _box(cx: float, cy: float, sx: float, sy: float, h: float, *, z0: float = MAT_MM, step: float = 2.0) -> np.ndarray:
    """The top and the four sides of an upright box on the mat, every ``step`` mm: what the looks see of it."""
    xs = np.arange(cx - sx / 2.0, cx + sx / 2.0 + 1e-9, step)
    ys = np.arange(cy - sy / 2.0, cy + sy / 2.0 + 1e-9, step)
    zs = np.arange(z0 + 1.0, z0 + h + 1e-9, step)
    top = [(x, y, z0 + h) for x in xs for y in ys]
    sides = [(x, y, z) for z in zs for x in xs for y in (ys[0], ys[-1])]
    sides += [(x, y, z) for z in zs for y in ys for x in (xs[0], xs[-1])]
    return np.asarray(top + sides, dtype=np.float64)


def _cylinder(cx: float, cy: float, r: float, h: float, *, z0: float = MAT_MM, step: float = 2.0) -> np.ndarray:
    points = []
    for rr in np.arange(0.0, r + 0.1, step):
        n = max(1, int(2 * math.pi * rr / step))
        points += [(cx + rr * math.cos(a), cy + rr * math.sin(a), z0 + h) for a in np.linspace(0, 2 * math.pi, n, False)]
    for z in np.arange(z0 + 1.0, z0 + h, step):
        points += [(cx + r * math.cos(a), cy + r * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 64, False)]
    return np.asarray(points, dtype=np.float64)


def _table() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(CX - 300.0, CX + 300.0, 2.5), np.arange(CY - 300.0, CY + 300.0, 2.5))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, MAT_MM)])


def _plan(neighbours: np.ndarray, **clearance: Any) -> PushPlan | PushRefusal:
    return plan_push(target_points_mm=_cylinder(CX, CY, 20.0, 50.0), neighbour_points_mm=neighbours,
                     support_normal=(0.0, 0.0, 1.0), support_offset_mm=MAT_MM, workspace=WORKSPACE, hand=HAND_E,
                     table_points_mm=_table(), push_distance_mm=50.0, **clearance)


class TenMillimetresBesideThePartTests(unittest.TestCase):
    def test_a_cylinder_with_one_neighbour_8_mm_off_plans_at_10_mm_beside_it(self) -> None:
        """Track P's probe: at 20 mm everywhere no direction is free; at the owner's 10 mm beside the part one is."""
        cube = _box(CX - 20.0 - 8.0 - 20.0, CY, 40.0, 40.0, 40.0)

        everywhere = _plan(cube, hand_clearance_mm=20.0)
        beside = _plan(cube, hand_clearance_mm=20.0, beside_part_clearance_mm=10.0, beside_part_mm=15.0)

        assert isinstance(everywhere, PushRefusal), everywhere
        self.assertEqual("no_free_direction", everywhere.code)
        self.assertIsInstance(beside, PushPlan, getattr(beside, "sentence", ""))

    def test_a_point_beyond_the_keep_out_keeps_the_line_judge_s_clearance(self) -> None:
        """A post the guard sees during the push (farther than 15 mm from the part and its path) still keeps 20 mm from
        the hand; 10 mm everywhere would plan a push whose hand passes it at 15 mm, which the guard then refuses."""
        cube = _box(CX - 20.0 - 8.0 - 20.0, CY, 40.0, 40.0, 40.0)
        plan = _plan(cube, hand_clearance_mm=20.0, beside_part_clearance_mm=10.0, beside_part_mm=15.0)
        assert isinstance(plan, PushPlan), plan
        # A thin post standing beside the line the hand's trailing finger travels, 15 mm out from its outer face.
        direction = np.asarray(plan.direction[:2])
        across = np.array([-direction[1], direction[0]])
        start = np.asarray(plan.contact_start_tcp_mm[:2])
        lateral = 0.5 * HAND_E.finger_width_mm + 15.0 + 2.0
        post_xy = start - 35.0 * direction + lateral * across
        post = _box(float(post_xy[0]), float(post_xy[1]), 4.0, 4.0, 40.0)
        neighbours = np.vstack([cube, post])

        uniform = _plan(neighbours, hand_clearance_mm=10.0)
        split = _plan(neighbours, hand_clearance_mm=20.0, beside_part_clearance_mm=10.0, beside_part_mm=15.0)

        assert isinstance(uniform, PushPlan), uniform
        np.testing.assert_allclose(uniform.direction, plan.direction)
        if isinstance(split, PushPlan):
            self.assertFalse(np.allclose(split.direction, plan.direction), "the post was passed at 15 mm")
        else:
            self.assertEqual("no_free_direction", split.code)

    def test_a_clearance_under_the_planner_s_floor_is_raised_to_it(self) -> None:
        cube = _box(CX - 20.0 - 8.0 - 20.0, CY, 40.0, 40.0, 40.0)

        at_floor = _plan(cube, hand_clearance_mm=20.0, beside_part_clearance_mm=10.0, beside_part_mm=15.0)
        under = _plan(cube, hand_clearance_mm=20.0, beside_part_clearance_mm=2.0, beside_part_mm=15.0)

        assert isinstance(at_floor, PushPlan) and isinstance(under, PushPlan)
        np.testing.assert_allclose(at_floor.direction, under.direction)
        with self.assertRaises(ValueError):
            _plan(cube, hand_clearance_mm=20.0, beside_part_clearance_mm=float("nan"), beside_part_mm=15.0)

    def test_the_library_call_is_unchanged(self) -> None:
        """No beside clearance asked: every neighbour point keeps ``hand_clearance_mm``, as before."""
        cube = _box(CX - 20.0 - 8.0 - 20.0, CY, 40.0, 40.0, 40.0)

        self.assertEqual("no_free_direction", _plan(cube, hand_clearance_mm=20.0).code)  # type: ignore[union-attr]
        self.assertIsInstance(_plan(cube, hand_clearance_mm=10.0), PushPlan)

    def test_the_cell_s_push_keeps_10_mm_beside_the_part_and_the_margin_is_the_world_s(self) -> None:
        reload_config()
        self.addCleanup(reload_config)
        robot = load_robot_config(None, profile="ur10,hande")
        cell = PushCell.from_robot_config(robot)
        assert isinstance(cell, PushCell), cell

        self.assertEqual(10.0, cell.beside_part_clearance_mm)
        self.assertEqual(float(robot.safety.planning_world.perceived.margin_mm), cell.beside_part_mm)
        self.assertEqual(float(robot.safety.self_collision.perceived_min_distance_mm), cell.guard_distance_mm)
        # The line judge's clearance stays what it was for every point the guard sees.
        self.assertEqual(float(robot.safety.planning_world.perceived.margin_mm)
                         + float(robot.safety.planned_motion.line_clearance_mm), cell.hand_clearance_mm)

    def test_the_pick_hands_the_planner_the_clearance_beside_the_part(self) -> None:
        cell = _PushCell(self)
        with mock.patch("src.robot.grasping.recovery.push_planner.plan_push", wraps=push_planner.plan_push) as planned:
            cell.campaign()

        self.assertTrue(planned.called)
        self.assertEqual(10.0, planned.call_args.kwargs["beside_part_clearance_mm"])
        self.assertEqual(15.0, planned.call_args.kwargs["beside_part_mm"])
        self.assertEqual(25.0, planned.call_args.kwargs["hand_clearance_mm"])


class _Unsegmented(_Bench):
    """The wrist camera over the bench, whose detector grounds the part alone: the post is seen, never segmented."""

    def acquire(self) -> Any:
        frame = super().acquire()
        kept = tuple(seg for seg in frame.segmentations if getattr(seg, "label", "") == "part")
        return type(frame)(depth_map=frame.depth_map, intrinsics=frame.intrinsics, segmentations=kept, rgb=frame.rgb,
                           timestamp=frame.timestamp, tool_pose=frame.tool_pose)


class _JammedBySeen(_Jammed):
    """The jammed calculator, which says what it saw beside the part as the cell's calculator does (contract 2): the
    post's surface as Track A's obstacle points, and ``world_rule`` as the world rule's."""

    def __init__(self, camera: Any, *, world_rule: np.ndarray | None = None, **keywords: Any) -> None:
        super().__init__(camera, **keywords)
        self.world_rule = world_rule

    def compute_result(self, seg: Any, *args: Any, **kwargs: Any) -> GraspResult:
        result = super().compute_result(seg, *args, **kwargs)
        if result.is_success or str(getattr(seg, "label", "")) != "part":
            return result
        post = next((box for box in self.camera.boxes if box.label == "post"), None)
        seen = np.zeros((0, 3)) if post is None else _box((post.low[0] + post.high[0]) / 2.0,
                                                         (post.low[1] + post.high[1]) / 2.0, post.high[0] - post.low[0],
                                                         post.high[1] - post.low[1], post.high[2] - post.low[2], z0=0.0)
        rule = seen if self.world_rule is None else np.vstack([seen, self.world_rule])
        return GraspResult(reasons=result.reasons, telemetry=dict(result.telemetry),
                           metadata={"scene_obstacle_points_base_mm": seen, "scene_points_world_rule_base_mm": rule})


class ThePushSeesWhatNoPromptNamedTests(unittest.TestCase):
    def _cell(self, **keywords: Any) -> _PushCell:
        # The push alone: no rescan or next_target picks again after it, so the pick loop keeps what this pick's push
        # came to.
        cell = _PushCell(self, boxes=(CUBE, POST), allowed=("nudge_target",))
        camera = _Unsegmented(cell.arm, boxes=(CUBE, POST))
        cell.camera = camera
        cell.arm.camera = camera
        cell.calculator = _JammedBySeen(camera, **keywords)
        cell.orchestrator.perception = camera
        cell.orchestrator.calculator = cell.calculator  # type: ignore[assignment]
        cell.service.runtime.orchestrator.perception = camera
        return cell

    def test_a_neighbour_only_the_calculator_saw_is_the_push_s_evidence(self) -> None:
        """Red before: no segmented neighbour, so 'no neighbour stands within 25 mm' and nothing was pushed."""
        cell = self._cell()

        run = cell.campaign()

        pushes = cell.orchestrator.pushes
        self.assertTrue(pushes, run.render())
        self.assertNotEqual("no_blocking_neighbour", pushes[0].code, pushes[0].sentence)
        self.assertEqual("pushed", pushes[0].code, pushes[0].sentence)

    def test_a_neighbour_only_the_world_rule_holds_refuses_the_push_s_path(self) -> None:
        """A neighbour the voxel rule dropped (a thin pin, no 25 mm voxel of 12 points) still stands in the push's way:
        pins round the part on every side the hand comes down, held by the world rule's points alone."""
        pins = np.vstack([_box(x, y, 3.0, 3.0, 30.0, z0=0.0) for x in (-70.0, 70.0) for y in (-750.0, -700.0, -650.0)]
                         + [_box(x, y, 3.0, 3.0, 30.0, z0=0.0) for x in (-30.0, 0.0, 30.0) for y in (-770.0, -630.0)])
        cell = self._cell(world_rule=pins)

        cell.campaign()

        pushes = cell.orchestrator.pushes
        self.assertTrue(pushes)
        self.assertEqual("no_free_direction", pushes[0].code, pushes[0].sentence)

    def test_the_neighbour_points_join_the_segmented_and_the_fused_ones(self) -> None:
        part = _box(0.0, -700.0, 40.0, 40.0, 40.0, z0=0.0)
        seen = _box(-50.0, -700.0, 20.0, 20.0, 30.0, z0=0.0)
        rule = _box(50.0, -700.0, 3.0, 3.0, 30.0, z0=0.0)
        judged = SimpleNamespace(look=SimpleNamespace(clouds=(part,)), fused_scene=None,
                                 result=SimpleNamespace(metadata={"scene_obstacle_points_base_mm": seen,
                                                                  "scene_points_world_rule_base_mm": rule}))

        neighbours, named = BinPickingOrchestrator._neighbour_points_of(judged, 0)  # type: ignore[arg-type]

        self.assertEqual(len(seen) + len(rule), len(neighbours))
        self.assertFalse(named.any(), "nobody named what the calculator saw beside the part")

    def test_the_points_on_a_named_neighbours_mask_are_named(self) -> None:
        """What the calculator found on a named neighbour's mask comes down beside as a part the detector named (the
        owner, 2026-10-06), the rest as before."""
        part = _box(0.0, -700.0, 40.0, 40.0, 40.0, z0=0.0)
        seen = _box(-50.0, -700.0, 20.0, 20.0, 30.0, z0=0.0)
        rule = _box(50.0, -700.0, 3.0, 3.0, 30.0, z0=0.0)
        on_seen = np.arange(len(seen)) % 2 == 0
        judged = SimpleNamespace(look=SimpleNamespace(clouds=(part,)), fused_scene=None,
                                 result=SimpleNamespace(metadata={"scene_obstacle_points_base_mm": seen,
                                                                  "scene_points_world_rule_base_mm": rule,
                                                                  "scene_obstacle_part_points": on_seen,
                                                                  "scene_world_rule_part_points": None}))

        neighbours, named = BinPickingOrchestrator._neighbour_points_of(judged, 0)  # type: ignore[arg-type]

        self.assertEqual(len(neighbours), len(named))
        np.testing.assert_array_equal(named[:len(seen)], on_seen)
        self.assertFalse(named[len(seen):].any())


if __name__ == "__main__":
    unittest.main()
