"""The looks of a wrist pick fuse a footprint less each mask's rim beside their surfaces, for SFE alone (2026-10-09).

Where the calculator cuts a rim off its own SFE input (``robot.grasping.geometry.footprint_rim_mm``,
``GraspCalculator.support_footprint_rim_mm``), the pick loop cuts the same rim off every look's surfaces
(``_LookView.footprints``, ``ObservedView.footprints``) and fuses them by the association the surfaces made
(``fuse_scene_geometry(primary_footprints=...)``, ``FusedSceneGeometry.footprint_clouds_base_mm``). The calculator is
handed that beside the fused cloud (``footprint_points_base_mm``) and reads it for SFE's footprint alone. The fused
cloud itself, the association, the jaw faces, the planner world's hold-out and the report's centre are what they were.
With no rim, nothing of this is built and every call is the call of before.

The scene is ``tests/_wrist_views.py``: a 40 mm cube on the bench ray-cast through the wrist D415 at half resolution
from each look; the calculator grasps it on the second look only, so the second look is ranked on both looks fused.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.core import JointPositions
from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.generation.footprint_rim import footprint_rim
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome, _LookView
from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.multiview.scene_geometry import ObservedView, fuse_scene_geometry, to_base_mm
from tests._wrist_views import (
    CUBE,
    K,
    LookCalculator,
    LookingArm,
    WristCamera,
    camera_looking_at,
    camera_to_tool,
    render,
    tool_for,
)

LOOK_PLUS_X = JointPositions.deg(0.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOK_MINUS_X = JointPositions.deg(180.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: The Hand-E's contact patch (config/grippers/robotiq_hande.yaml), the owner's hand.
HAND_E = ParallelJawGripperModel(finger_width_mm=29.24, pad_length_mm=20.91, pad_ahead_mm=10.45)
#: The rim the owner's cell is tested with on 2026-10-12.
RIM_MM = 2.0


class _RimCalculator(LookCalculator):
    """The looks' calculator of a cell that cuts a rim off SFE's footprint input."""

    support_footprint_rim_mm = RIM_MM


class _World:
    """A live planner world as the pick loop meets it: it holds a pick's frames and keeps every offer it is made."""

    def __init__(self) -> None:
        self.offers: list[dict[str, Any]] = []

    def hold_pick_views(self) -> bool:
        return True

    def forget_pick_views(self) -> None:
        pass

    def offer_segmentation(self, **offered: Any) -> None:
        self.offers.append(offered)

    def forget_segmentation(self) -> None:
        pass


class _Policy:
    def __init__(self) -> None:
        self.executed: list[Any] = []

    def execute(self, grasp: Any) -> PolicyReport:
        self.executed.append(grasp)
        return PolicyReport(outcome=PolicyOutcome.EXECUTED)


def _pick(calculator_type: type) -> SimpleNamespace:
    """A wrist pick from the cube's +x and -x sides, grasped on the second look, on a world that keeps its offers."""
    arm = LookingArm({LOOK_PLUS_X: tool_for(camera_looking_at(bearing_deg=0.0)),
                      LOOK_MINUS_X: tool_for(camera_looking_at(bearing_deg=180.0))})
    arm.live_planner_world = _World()
    camera = WristCamera(arm)
    camera.camera_name = "wrist"  # type: ignore[attr-defined]  # the camera the offered masks belong to
    calculator = calculator_type(camera, grasps_on=(1,))
    orchestrator = BinPickingOrchestrator(
        arm=arm, calculator=calculator, perception=camera,  # type: ignore[arg-type]
        frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()), policy=_Policy(),  # type: ignore[arg-type]
        max_attempts=3, primary_camera_id="wrist", gripper_model=HAND_E, looks=(LOOK_PLUS_X, LOOK_MINUS_X))
    report = orchestrator.run()
    return SimpleNamespace(report=report, calculator=calculator, looked=orchestrator.looked_around,
                           world=arm.live_planner_world)


def _rows(points: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(row) for row in np.round(np.asarray(points, dtype=np.float64), 6)}


class ThePickLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cut = _pick(_RimCalculator)
        cls.whole = _pick(LookCalculator)

    def test_both_picks_grasp_on_the_second_look_fused(self) -> None:
        for pick in (self.cut, self.whole):
            self.assertIs(PickOutcome.EXECUTED, pick.report.outcome)
            self.assertEqual(2, len(pick.looked.looks_fused))

    def test_the_second_look_hands_sfe_its_footprint_beside_the_fused_cloud(self) -> None:
        (second,) = self.cut.calculator.calls_on(1)
        fused = np.asarray(second["geometry_points_base_mm"])
        footprint = np.asarray(second["footprint_points_base_mm"])
        self.assertGreater(footprint.shape[0], 0)
        self.assertLess(footprint.shape[0], fused.shape[0], "the rim took nothing off the looks")
        self.assertLessEqual(_rows(footprint), _rows(fused), "the footprint holds a point the looks did not see")

    def test_the_first_look_ranked_alone_is_handed_no_footprint(self) -> None:
        """A look ranked alone has its rim cut by the calculator, on its own mask."""
        (first,) = self.cut.calculator.calls_on(0)
        self.assertNotIn("footprint_points_base_mm", first)
        self.assertNotIn("geometry_points_base_mm", first)

    def test_the_fused_cloud_and_everything_that_reads_it_are_what_they_were(self) -> None:
        """The jaw faces, the planner world's hold-out and the report's centre read the target cloud the looks judged:
        the fused cloud whole, the same with the rim cut as without."""
        (cut,) = self.cut.calculator.calls_on(1)
        (whole,) = self.whole.calculator.calls_on(1)
        np.testing.assert_array_equal(whole["geometry_points_base_mm"], cut["geometry_points_base_mm"])
        np.testing.assert_array_equal(cut["geometry_points_base_mm"], self.cut.looked.judged.target_cloud_base_mm)
        np.testing.assert_array_equal(self.whole.looked.judged.target_cloud_base_mm,
                                      self.cut.looked.judged.target_cloud_base_mm)
        np.testing.assert_array_equal(self.whole.world.offers[-1]["target_points_base_mm"],
                                      self.cut.world.offers[-1]["target_points_base_mm"])
        np.testing.assert_array_equal(cut["geometry_points_base_mm"], self.cut.world.offers[-1]["target_points_base_mm"])
        self.assertEqual(self.whole.report.target_centre_mm, self.cut.report.target_centre_mm)
        self.assertEqual(self.whole.looked.judged.members, self.cut.looked.judged.members, "the association changed")

    def test_the_planner_world_holds_out_the_frames_whole_mask(self) -> None:
        judged = self.cut.looked.judged
        target = np.asarray(judged.frame.segmentations[judged.target_index].mask).astype(bool)
        offered = self.cut.world.offers[-1]
        (held_out,) = offered["exclude_masks"]
        np.testing.assert_array_equal(target, np.asarray(held_out).astype(bool))
        self.assertEqual(len(self.whole.world.offers), len(self.cut.world.offers))
        for whole, cut in zip(self.whole.world.offers, self.cut.world.offers):
            self.assertTrue(cut["labelled_masks"], "the offer carried no masks to compare")
            for (left_label, left), (right_label, right) in zip(whole["labelled_masks"], cut["labelled_masks"]):
                self.assertEqual(left_label, right_label)
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(whole["exclude_masks"], cut["exclude_masks"]):
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_every_look_keeps_its_surfaces_and_a_footprint_inside_each(self) -> None:
        for whole, cut in zip(self.whole.looked.views, self.cut.looked.views):
            self.assertEqual((), whole.footprints)
            self.assertEqual(len(cut.surfaces), len(cut.footprints))
            for surface, kept, footprint in zip(whole.surfaces, cut.surfaces, cut.footprints):
                np.testing.assert_array_equal(surface, kept)
                self.assertFalse(np.any(footprint & ~kept), "a footprint pixel outside its surface")
                self.assertLess(int(np.count_nonzero(footprint)), int(np.count_nonzero(kept)))
            for left, right in zip(whole.clouds, cut.clouds):
                np.testing.assert_array_equal(left, right)

    def test_without_a_rim_no_call_carries_a_footprint(self) -> None:
        for _number, _label, keywords in self.whole.calculator.calls:
            self.assertNotIn("footprint_points_base_mm", keywords)

    def test_every_look_says_what_its_rim_took(self) -> None:
        with self.assertLogs("src.robot.grasping.loop.pick_loop", level="INFO") as said:
            _pick(_RimCalculator)
        lines = [line for line in said.output if "SFE's footprint is built from 1 part(s) less a" in line]
        self.assertEqual(2, len(lines), said.output)
        self.assertIn("every other reader keeps the whole mask", lines[0])


def _views(boxes: tuple[Any, ...] = (CUBE,)) -> tuple[ObservedView, ObservedView]:
    """The cube from its +x and its -x side, each with its masks and their footprints."""
    views = []
    for name, bearing in (("plus_x", 0.0), ("minus_x", 180.0)):
        placed = camera_looking_at(bearing_deg=bearing)
        depth, hit = render(placed, boxes)
        masks = tuple(hit == index for index in range(len(boxes)))
        views.append(ObservedView(name=name, masks=masks, depth_map=depth, intrinsics=K, camera_to_base=placed,
                                  footprints=tuple(footprint_rim(mask, depth, float(K[0, 0]), RIM_MM).mask
                                                   for mask in masks)))
    return views[0], views[1]


class TheFusionTests(unittest.TestCase):
    def test_footprints_change_no_cloud_no_association_and_no_neighbour(self) -> None:
        primary, other = _views()
        plain = fuse_scene_geometry(primary.masks, primary.depth_map, K, primary.camera_to_base, [other],
                                    with_neighbours=True)
        asked = fuse_scene_geometry(primary.masks, primary.depth_map, K, primary.camera_to_base, [other],
                                    with_neighbours=True, primary_footprints=primary.footprints)
        self.assertEqual((), plain.footprint_clouds_base_mm)
        self.assertIsNone(plain.footprint_for(0))
        np.testing.assert_array_equal(plain.cloud_for(0), asked.cloud_for(0))
        self.assertEqual(plain.associations, asked.associations)
        self.assertEqual(plain.views_used, asked.views_used)
        self.assertEqual(len(plain.neighbour_clouds_base_mm), len(asked.neighbour_clouds_base_mm))

    def test_the_footprint_is_each_views_footprint_by_the_clouds_association(self) -> None:
        primary, other = _views()
        fused = fuse_scene_geometry(primary.masks, primary.depth_map, K, primary.camera_to_base, [other],
                                    primary_footprints=primary.footprints)
        expected = np.vstack([to_base_mm(primary.footprints[0], primary.depth_map, K, primary.camera_to_base),
                              to_base_mm(other.footprints[0], other.depth_map, K, other.camera_to_base)])
        np.testing.assert_array_equal(expected, fused.footprint_for(0))
        self.assertLess(fused.footprint_for(0).shape[0], fused.cloud_for(0).shape[0])

    def test_a_view_with_no_footprints_gives_its_masks(self) -> None:
        primary, other = _views()
        bare = ObservedView(name=other.name, masks=other.masks, depth_map=other.depth_map, intrinsics=K,
                            camera_to_base=other.camera_to_base)
        fused = fuse_scene_geometry(primary.masks, primary.depth_map, K, primary.camera_to_base, [bare],
                                    primary_footprints=primary.footprints)
        expected = np.vstack([to_base_mm(primary.footprints[0], primary.depth_map, K, primary.camera_to_base),
                              to_base_mm(other.masks[0], other.depth_map, K, other.camera_to_base)])
        np.testing.assert_array_equal(expected, fused.footprint_for(0))

    def test_an_object_no_other_view_confirms_has_no_footprint_as_it_has_no_cloud(self) -> None:
        primary, _other = _views()
        elsewhere = camera_looking_at((0.0, -300.0, 0.0), bearing_deg=0.0)
        depth, _hit = render(elsewhere, (CUBE,))
        empty = ObservedView(name="elsewhere", masks=(np.zeros(depth.shape, dtype=bool),), depth_map=depth,
                             intrinsics=K, camera_to_base=elsewhere)
        fused = fuse_scene_geometry(primary.masks, primary.depth_map, K, primary.camera_to_base, [empty],
                                    primary_footprints=primary.footprints)
        self.assertIsNone(fused.cloud_for(0))
        self.assertIsNone(fused.footprint_for(0))


class APushedPartLeavesItsFootprintTooTests(unittest.TestCase):
    def test_the_footprints_stay_beside_their_masks(self) -> None:
        """A pushed part leaves every earlier look's view (``_without_the_pushed_part``): its footprint goes with its
        mask, so the view's footprints still stand beside its masks, one for one."""
        primary, _other = _views()
        second = np.zeros_like(primary.masks[0])
        second[:4, :4] = True
        view = ObservedView(name="wrist@a", masks=(primary.masks[0], second), depth_map=primary.depth_map,
                            intrinsics=K, camera_to_base=primary.camera_to_base, segmentations=("part", "bolt"),
                            footprints=(primary.footprints[0], second))
        look = _LookView(label="a", pose=None, name="wrist@a", frame=None,  # type: ignore[arg-type]
                         camera_to_base=primary.camera_to_base, surfaces=view.masks, view=view, blob_objects=(0, 1),
                         clouds=(np.ones((5, 3)), np.ones((5, 3))), footprints=view.footprints)
        orchestrator = BinPickingOrchestrator.__new__(BinPickingOrchestrator)
        orchestrator._look_views = [look]
        orchestrator._fixed_views = None

        orchestrator._without_the_pushed_part(SimpleNamespace(members={"wrist@a": 0}))

        (kept,) = orchestrator._look_views
        self.assertEqual(1, len(kept.view.masks))
        np.testing.assert_array_equal(second, kept.view.footprints[0])
        self.assertIsNone(kept.footprints[0])
        np.testing.assert_array_equal(second, kept.footprints[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
