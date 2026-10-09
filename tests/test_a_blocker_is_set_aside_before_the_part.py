"""Clear the blocker: a neighbour that leaves the part no grasp is picked, set aside and the part picked after it (the
owner's recovery of 2026-10-02, cell fixes Track R).

The owner: "das ganze Ding soll die Szene so verändern, dass er diesen greifen kann. Also entweder das Objekt verschieben
oder ein anderes Objekt nehmen, was im Weg liegt und dann das eigentliche Objekt aufheben." When every grasp of the part
meets a neighbour (``ALL_COLLIDED``) and the pick may change the scene (its push gate, ``nudge_target``), the pick loop
first clears a blocker and pushes only where none can be cleared:

* the blocker is the neighbour whose removal frees the most grasps of the part, read by asking the calculator again with
  that neighbour's pixels left out of the frame; it stands on what the part stands on, apart from the part;
* it is gripped like any part (the calculator, the policy, the exact guard on every path), with the part back in the
  planner world and the blocker held out of it;
* it is set down on a free spot of the support the camera saw, away from the part and from everything else, or at the
  place the owner names (``blocker_place``, a taught pose such as "Müll"), through the existing place verb;
* the arm goes back to the look, looks again, and the next attempt grasps the part from that look;
* no numeric budget, but a removal that frees no grasp of the part, no free spot and no graspable blocker each stop it
  with a typed reason. A stop asked for once the blocker is held ends the pick where the arm stands: a person decides.

The scene: the cube and a post 7 mm off its -x face on the bench, seen from the cube's +y side by the wrist D415, which
grounds the cube alone. The calculator double grasps the cube only where the post's pixels are gone from the depth it is
handed, and grasps the post when asked for the blocker. The real ``GraspExecutionPolicy`` drives the looking arm and a
hand that closes and opens on whatever stands under it.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.schema.robot.grasping_schema import GraspingSupportConfig
from src.geometry import Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, PickOutcome
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver
from src.robot.grasping.recovery.push_budgets import PushBudgets
from src.robot.grasping.recovery.push_gate import PushCell, PushGate
from src.robot.grasping.recovery.push_planner import AxisBox
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.safety.planning.perceived import WorldBuildLimits, WorldBuildTuning
from tests._wrist_views import CUBE, K, Box, LookingArm, WristCamera, camera_to_tool, render
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import HAND_E, LOOK_PLUS_Y, _poses, _World
from tests.test_the_push_sees_unprompted_neighbours import HAND_E as PUSH_HAND

#: A post 7 mm off the cube's -x face: 28 mm across x, 30 along y, 40 tall. The hand's 50 mm opens over its x side.
POST = Box((-55.0, -715.0, 0.0), (-27.0, -685.0, 40.0), "post")
WORKSPACE = AxisBox((-460.0, -935.0, -100.0), (440.0, -235.0, 600.0))
LIMITS = WorldBuildLimits(x_mm=(-460.0, 440.0), y_mm=(-935.0, -235.0), z_mm=(-100.0, 600.0), support_plane_top_mm=0.0)
TUNING = WorldBuildTuning(pixel_stride=2, voxel_size_mm=10.0, cluster_voxel_mm=25.0, min_points=12, margin_mm=15.0,
                          max_boxes=64, floor_to_plane=True, support_surfaces=True, support_allowance_mm=2.0)


class _Scene:
    """The boxes on the bench, which the hand moves: a close takes the box under the TCP, an open sets it down there."""

    def __init__(self, boxes: tuple[Box, ...]) -> None:
        self.boxes = list(boxes)
        self.held: str | None = None

    def box(self, label: str) -> Box:
        return next(box for box in self.boxes if box.label == label)

    def centre(self, label: str) -> np.ndarray:
        box = self.box(label)
        return (np.asarray(box.low) + np.asarray(box.high)) / 2.0

    def close_at(self, tcp: np.ndarray) -> None:
        for box in self.boxes:
            centre = (np.asarray(box.low) + np.asarray(box.high)) / 2.0
            if float(np.hypot(*(centre[:2] - tcp[:2]))) <= 25.0:
                self.held = box.label
                return

    def open_at(self, tcp: np.ndarray) -> None:
        if self.held is None:
            return
        box = self.box(self.held)
        half = (np.asarray(box.high) - np.asarray(box.low)) / 2.0
        low = (float(tcp[0] - half[0]), float(tcp[1] - half[1]), 0.0)
        high = (float(tcp[0] + half[0]), float(tcp[1] + half[1]), float(box.high[2] - box.low[2]))
        self.boxes = [Box(low, high, box.label) if each.label == box.label else each for each in self.boxes]
        self.held = None


class _SceneCamera(WristCamera):
    """The wrist D415 over the scene as it stands now, whose detector grounds the part alone. ``last_hit`` says which
    box every pixel of the last frame met."""

    def __init__(self, arm: Any, scene: _Scene) -> None:
        super().__init__(arm, boxes=tuple(scene.boxes))
        self.scene = scene
        self.last_hit: np.ndarray | None = None
        self.last_boxes: tuple[Box, ...] = ()

    def acquire(self) -> Any:
        self.boxes = tuple(box for box in self.scene.boxes if box.label != self.scene.held)
        frame = super().acquire()
        _depth, self.last_hit = render(self.taken[-1].to_matrix() @ _mount(), self.boxes)
        self.last_boxes = self.boxes
        kept = tuple(seg for seg in frame.segmentations if getattr(seg, "label", "") == "part")
        return type(frame)(depth_map=frame.depth_map, intrinsics=frame.intrinsics, segmentations=kept, rgb=frame.rgb,
                           timestamp=frame.timestamp, tool_pose=frame.tool_pose)


def _mount() -> np.ndarray:
    from tests._wrist_views import mount

    return mount()


class _Hand:
    """A two-state hand that closes and opens on what stands under the TCP; ``commands`` says each, in order."""

    is_connected = True
    min_width_mm = 0.0
    max_width_mm = 50.0

    def __init__(self, arm: Any, scene: _Scene, *, on_close: Any = None) -> None:
        self.arm = arm
        self.scene = scene
        self.on_close = on_close
        self.commands: list[str] = []
        self.closed = False

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def activate(self) -> None:
        return None

    def set_closed(self, closed: bool) -> None:
        tcp = np.asarray(self.arm.get_tcp_pose().position_mm, dtype=np.float64)
        self.commands.append("close" if closed else "open")
        self.closed = bool(closed)
        if closed:
            self.scene.close_at(tcp)
            if self.on_close is not None:
                self.on_close()
        else:
            self.scene.open_at(tcp)

    def set_width_mm(self, width_mm: float, **_keywords: Any) -> None:
        raise AssertionError("a two-state hand is told open or close, never a width")

    def get_width_mm(self) -> float:
        return 0.0 if self.closed else self.max_width_mm


def _top_down(centre: np.ndarray, *, width_mm: float, label: str) -> GraspPoint:
    return GraspPoint(position=np.asarray(centre, dtype=np.float64), approach=np.array([0.0, 0.0, -1.0]),
                      axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=width_mm, score=0.9, frame=GraspFrame.BASE,
                      label=label)


class _Boxed:
    """The cell's calculator as the scene needs it: the part collides with the post while the post's pixels stand in
    the depth it is handed and the post stands beside it (``ALL_COLLIDED``, the post's surface as Track A's obstacle
    points, ``seen`` refusals by SFE's count); otherwise one grasp at the part's centre. Asked for the blocker, a grasp at
    the post's centre where ``graspable``. ``stuck``: the part stays boxed in once the post has gone, by nothing seen.
    ``calls`` keeps every label asked for, in order."""

    render_debug_images = False

    def __init__(self, camera: _SceneCamera, scene: _Scene, *, graspable: bool = True, stuck: bool = False,
                 seen: int = 6, named: bool = False) -> None:
        self.camera = camera
        self.scene = scene
        self.graspable = graspable
        #: Whether the post is a part the detector named: its points handed as such (``scene_obstacle_part_points``).
        self.named = named
        self.stuck = stuck
        self.seen = seen
        self.calls: list[str] = []
        self.camera_matrix = K.copy()
        #: How many grasps the blocker is offered, best first: the first closes along x, the next along y, then
        #: diagonally, all at the post's centre.
        self.blocker_grasps = 1

    def compute_result(self, seg: Any, depth: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        label = str(getattr(seg, "label", ""))
        self.calls.append(label)
        if label == "blocker":
            if not self.graspable:
                return GraspResult(reasons=(GraspFailureReason.NO_VALID_GRASP, GraspFailureReason.RESCAN_RECOMMENDED))
            axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7071068, 0.7071068, 0.0))[:max(1, self.blocker_grasps)]
            grasps = tuple(replace(_top_down(self.scene.centre("post"), width_mm=28.0, label="blocker"),
                                   axis=np.array(axis), score=0.9 - 0.1 * number)
                           for number, axis in enumerate(axes))
            return GraspResult(candidates=grasps, top_score=0.9)
        if label != "part":
            return GraspResult(reasons=(GraspFailureReason.NO_CANDIDATES_GENERATED,))
        handed = np.asarray(depth, dtype=np.float64)
        index = next((i for i, box in enumerate(self.camera.last_boxes) if box.label == "post"), None)
        post = (self.camera.last_hit == index) & (handed > 0.0) if index is not None else np.zeros(handed.shape, bool)
        beside = index is not None and float(np.hypot(*(self.scene.centre("post")[:2] - self.scene.centre("part")[:2]))) < 60.0
        if (post.any() and beside) or (self.stuck and not beside):
            points = self._base(post, handed, kwargs) if post.any() and beside else np.zeros((0, 3))
            return GraspResult(
                reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED),
                telemetry={"support_footprint_refused": {"seen_corridor": self.seen}},
                metadata={"scene_obstacle_points_base_mm": points, "scene_points_world_rule_base_mm": points,
                          **({"scene_obstacle_part_points": np.ones(len(points), dtype=bool)} if self.named else {})})
        grasp = _top_down(self.scene.centre("part"), width_mm=40.0, label="part")
        return GraspResult(candidates=(grasp,), top_score=0.9)

    @staticmethod
    def _base(pixels: np.ndarray, depth: np.ndarray, kwargs: dict[str, Any]) -> np.ndarray:
        rows, cols = np.nonzero(pixels)
        z = depth[rows, cols]
        camera = np.column_stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z])
        matrix = np.asarray(kwargs["camera_to_base"].to_matrix(), dtype=np.float64)
        return camera @ matrix[:3, :3].T + matrix[:3, 3]


#: A second post 7 mm off the cube's +x face, as the first stands off its -x face.
POST_PLUS_X = Box((27.0, -715.0, 0.0), (55.0, -685.0, 40.0), "post_plus_x")


class _TwoPosts(_Boxed):
    """Both posts box the part in: it collides while either post's pixels stand in the depth it is handed and that post
    stands beside it, by the same count of seen refusals however many do, and has one grasp once neither does. Asked
    for a blocker, a grasp at the centre of the post the blocker's mask lies on."""

    def compute_result(self, seg: Any, depth: Any, *_args: Any, **kwargs: Any) -> GraspResult:
        label = str(getattr(seg, "label", ""))
        if label == "blocker":
            self.calls.append(label)
            mask = np.asarray(getattr(seg, "mask"), dtype=bool)
            assert self.camera.last_hit is not None
            met = [self.camera.last_boxes[int(i)].label for i in np.unique(self.camera.last_hit[mask]) if int(i) >= 0]
            post = next((name for name in met if name.startswith("post")), None)
            if post is None:
                return GraspResult(reasons=(GraspFailureReason.NO_VALID_GRASP, GraspFailureReason.RESCAN_RECOMMENDED))
            return GraspResult(candidates=(_top_down(self.scene.centre(post), width_mm=28.0, label="blocker"),),
                               top_score=0.9)
        if label != "part":
            return super().compute_result(seg, depth, *_args, **kwargs)
        self.calls.append(label)
        handed = np.asarray(depth, dtype=np.float64)
        points = []
        for index, box in enumerate(self.camera.last_boxes):
            if not box.label.startswith("post"):
                continue
            pixels = (self.camera.last_hit == index) & (handed > 0.0)
            beside = float(np.hypot(*(self.scene.centre(box.label)[:2] - self.scene.centre("part")[:2]))) < 60.0
            if pixels.any() and beside:
                points.append(self._base(pixels, handed, kwargs))
        if points:
            seen = np.vstack(points)
            return GraspResult(
                reasons=(GraspFailureReason.ALL_COLLIDED, GraspFailureReason.RESCAN_RECOMMENDED),
                telemetry={"support_footprint_refused": {"seen_corridor": self.seen}},
                metadata={"scene_obstacle_points_base_mm": seen, "scene_points_world_rule_base_mm": seen})
        return GraspResult(candidates=(_top_down(self.scene.centre("part"), width_mm=40.0, label="part"),),
                           top_score=0.9)


class _HeldWorld(_World):
    """The live world double, with the cell's limits and tuning, which also notes the offer held at each close."""

    def __init__(self) -> None:
        super().__init__()
        self.limits = LIMITS
        self.tuning = TUNING
        self.held_offers: list[dict[str, Any]] = []
        self.forgets = 0

    def offer_segmentation(self, **offered: Any) -> None:
        super().offer_segmentation(**offered)
        self.held_offers.append(offered)

    def forget_segmentation(self) -> None:
        self.forgets += 1
        self.held_offers = []


class _Cell:
    def __init__(self, *, gate: bool = True, workspace: AxisBox = WORKSPACE, graspable: bool = True,
                 stuck: bool = False, on_close: Any = None, named: bool = False, **wiring: Any) -> None:
        self.scene = _Scene((CUBE, POST))
        self.arm = LookingArm(_poses())
        self.world = _HeldWorld()
        self.arm.live_planner_world = self.world  # type: ignore[attr-defined]
        self.camera = _SceneCamera(self.arm, self.scene)
        self.calculator = _Boxed(self.camera, self.scene, graspable=graspable, stuck=stuck, named=named)
        self.offers_at_close: list[list[str]] = []

        def closing() -> None:
            self.offers_at_close.append([str(offer.get("target_label", "")) for offer in self.world.held_offers])
            if on_close is not None:
                on_close()

        self.hand = _Hand(self.arm, self.scene, on_close=closing)
        self.policy = GraspExecutionPolicy(arm=self.arm, gripper=self.hand, standoff_mm=60.0)  # type: ignore[arg-type]
        self.orchestrator = BinPickingOrchestrator(
            arm=self.arm, calculator=self.calculator, perception=self.camera,  # type: ignore[arg-type]
            frame_resolver=EyeInHandFrameResolver(t_cam_to_tool=camera_to_tool()), policy=self.policy,
            gripper=self.hand, max_attempts=3, primary_camera_id="wrist", gripper_model=HAND_E,  # type: ignore[arg-type]
            looks=(LOOK_PLUS_Y,), target_label="part",
            support_config=GraspingSupportConfig(height_mm=0.0, refine_from_target=False), **wiring,
        )
        if gate:
            self.orchestrator.push_gate = PushGate(budgets=PushBudgets(), distance_mm=30.0, cell=PushCell(
                hand=PUSH_HAND, workspace=workspace, hand_clearance_mm=20.0))

    def run(self) -> Any:
        return self.orchestrator.run()


class ABlockerIsSetAsideTests(unittest.TestCase):
    def test_the_post_is_set_aside_and_then_the_part_is_picked(self) -> None:
        """Red before: the part was never picked; with no push planned, the pick ended without one."""
        cell = _Cell()

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome, [a.action for a in report.attempts])
        self.assertEqual(["clear_blocker", "executed"], [attempt.action for attempt in report.attempts])
        self.assertEqual("set_aside", report.attempts[0].blocker)
        # The hand: the post closed on and opened over a free spot, then the part closed on. One command each.
        self.assertEqual(["close", "open", "close"], cell.hand.commands)
        self.assertEqual("part", cell.scene.held)
        # The post stands well away from the part now, on the bench.
        gap = float(np.hypot(*(cell.scene.centre("post")[:2] - cell.scene.centre("part")[:2])))
        self.assertGreaterEqual(gap, 150.0)
        self.assertEqual(0.0, cell.scene.box("post").low[2])
        # While the post was gripped it was the one held out of the planner world, and the part was back in it.
        self.assertEqual(["blocker"], cell.offers_at_close[0])
        # The calculator was asked about the blocker before anything moved; the part was asked again after the look.
        self.assertIn("blocker", cell.calculator.calls)
        (record,) = cell.orchestrator.blockers
        self.assertEqual("set_aside", record.code)
        self.assertGreaterEqual(record.freed, 1)

    def test_the_arm_goes_back_to_the_look_and_looks_again_before_the_part(self) -> None:
        cell = _Cell()

        cell.run()

        frames_before = len(cell.camera.taken)
        self.assertGreaterEqual(frames_before, 2, "the look was not taken again after the post was set aside")
        joints = [key for kind, key in cell.arm.motions if kind == "joints"]
        self.assertGreaterEqual(len(joints), 2)
        self.assertEqual(joints[0], joints[-1], "the look taken again is not the look the part was judged on")

    def test_the_owner_s_named_place_takes_the_blocker(self) -> None:
        """A taught pose such as "Müll": the post is set down there, not on a free spot."""
        place = Pose.tool_down(200.0, -400.0, 20.0)
        cell = _Cell(blocker_place=place)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        np.testing.assert_allclose(cell.scene.centre("post")[:2], (200.0, -400.0), atol=1e-6)


class ABlockerIsThePickWhereTheTaskTakesEveryPartTests(unittest.TestCase):
    """The owner, 2026-10-06: "einmal das es direkt weggepackt wird und einmal nur umgelegt". Where the pick loop's
    ``blocker_is_the_pick`` is on (a task that takes every part to one place), the blocker it takes away is the part it
    picks: gripped and lifted, reported executed on the blocker's grasp with the blocker's cloud the judged one, and
    set down by the caller where its parts go. Nothing is set down here, and no free spot is asked for."""

    def test_the_post_is_the_part_the_pick_takes(self) -> None:
        cell = _Cell(blocker_is_the_pick=True, named=True)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome, [a.action for a in report.attempts])
        self.assertEqual(["blocker_picked"], [attempt.action for attempt in report.attempts])
        self.assertEqual("taken_as_the_pick", report.attempts[0].blocker)
        # One close, on the post, and no open: the caller sets it down where its parts go.
        self.assertEqual(["close"], cell.hand.commands)
        self.assertEqual("post", cell.scene.held)
        (record,) = cell.orchestrator.blockers
        self.assertEqual("taken_as_the_pick", record.code)
        self.assertEqual("where the parts go", record.place)
        # The grasp reported is the post's, and so is the cloud the caller's drop reads how far the part hangs off.
        assert report.executed_grasp is not None
        grasp = report.executed_grasp.candidates[0]
        self.assertLess(float(np.hypot(*(np.asarray(grasp.position)[:2] - cell.scene.centre("post")[:2]))), 20.0)
        judged = cell.orchestrator.looked_around.judged
        cloud = np.asarray(judged.target_cloud_base_mm)
        self.assertLess(float(np.hypot(*(np.median(cloud[:, :2], axis=0) - cell.scene.centre("post")[:2]))), 20.0)
        np.testing.assert_allclose(np.asarray(report.target_centre_mm)[:2], cell.scene.centre("post")[:2], atol=20.0)

    def test_no_free_spot_is_asked_for(self) -> None:
        """The workspace leaves no spot to set the post down on, and it is taken all the same."""
        tight = AxisBox((-70.0, -740.0, -100.0), (60.0, -640.0, 600.0))
        cell = _Cell(workspace=tight, blocker_is_the_pick=True, named=True)

        report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual("post", cell.scene.held)

    def test_only_a_part_the_detector_named_is_taken_as_the_pick(self) -> None:
        """A wall of the tray the part stood in passed every other test and was gripped as the pick (the grasp bench,
        2026-10-06): what nobody named is no blocker taken as the pick, and nothing moves for it."""
        cell = _Cell(blocker_is_the_pick=True, named=False)

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertEqual("no_blocker_seen", cell.orchestrator.blockers[0].code)
        self.assertIn("no part the detector named", cell.orchestrator.blockers[0].sentence)

    def test_nothing_in_a_region_the_task_keeps_out_is_a_blocker(self) -> None:
        from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion, ExclusionZones

        zones = ExclusionZones()
        zones.keep_out_region(ExclusionRegion.circle((-41.0, -700.0), 30.0, reason="the bin"))
        cell = _Cell(blocker_is_the_pick=True, named=True, exclusion_zones=zones)

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertIn("in a region the task keeps out", cell.orchestrator.blockers[0].sentence)

    def test_off_the_post_is_set_aside_as_before(self) -> None:
        cell = _Cell(blocker_is_the_pick=False)

        report = cell.run()

        self.assertEqual(["clear_blocker", "executed"], [attempt.action for attempt in report.attempts])
        self.assertEqual("part", cell.scene.held)

    def test_a_stop_asked_for_once_the_post_is_held_still_leaves_it_to_a_person(self) -> None:
        stops: list[bool] = []

        def held() -> bool:
            return bool(stops)

        cell = _Cell(blocker_is_the_pick=True, named=True, on_close=lambda: stops.append(True), should_cancel=held)

        report = cell.run()

        self.assertIsNot(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(["close"], cell.hand.commands)


class ClearingTheBlockerStopsWithAReasonTests(unittest.TestCase):
    def test_no_graspable_blocker_falls_to_the_push_with_nothing_moved(self) -> None:
        cell = _Cell(graspable=False)

        cell.run()

        self.assertEqual([], cell.hand.commands)
        records = cell.orchestrator.blockers
        self.assertTrue(records)
        self.assertEqual("no_graspable_blocker", records[0].code)
        self.assertTrue(cell.orchestrator.pushes, "the push was not considered after the blocker")
        self.assertLess(cell.calculator.calls.index("blocker"), len(cell.calculator.calls))

    def test_no_free_spot_falls_to_the_push_with_nothing_moved(self) -> None:
        tight = AxisBox((-70.0, -740.0, -100.0), (60.0, -640.0, 600.0))
        cell = _Cell(workspace=tight)

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertEqual("no_free_spot", cell.orchestrator.blockers[0].code)
        self.assertTrue(cell.orchestrator.pushes)

    def test_a_removal_that_frees_no_grasp_stops_the_clearing(self) -> None:
        cell = _Cell(stuck=True)

        report = cell.run()

        self.assertEqual(["close", "open"], cell.hand.commands, "more than the one blocker was moved")
        codes = [record.code for record in cell.orchestrator.blockers]
        self.assertEqual(["set_aside", "freed_nothing"], codes)
        self.assertNotIn("executed", [attempt.action for attempt in report.attempts])

    def test_two_posts_that_free_the_part_only_together_are_both_taken_away(self) -> None:
        """A post either side of the part: each alone frees nothing, and once the first is set aside the part's count of
        refusals is a new look's and need not fall. The second frees a grasp outright, so it is taken away too, and the
        part is picked (the grasp bench, 2026-10-06: two posts 10 mm off a cylinder, the clearing stopped after the
        first)."""
        cell = _Cell()
        cell.scene = _Scene((CUBE, POST, POST_PLUS_X))
        cell.camera.scene = cell.scene
        cell.calculator = _TwoPosts(cell.camera, cell.scene)
        cell.orchestrator.calculator = cell.calculator  # type: ignore[assignment]
        cell.hand.scene = cell.scene

        report = cell.run()

        codes = [record.code for record in cell.orchestrator.blockers]
        self.assertEqual(["set_aside", "set_aside"], codes[:2], codes)
        self.assertNotIn("freed_nothing", codes)
        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        # Each post closed on and set down, then the part closed on and held.
        self.assertEqual(["close", "open", "close", "open", "close"], cell.hand.commands)

    def test_without_the_push_gate_nothing_is_cleared(self) -> None:
        cell = _Cell(gate=False)

        report = cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertEqual((), cell.orchestrator.blockers)
        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)

    def test_a_stop_asked_for_once_the_blocker_is_held_moves_nothing_more(self) -> None:
        stop: list[bool] = []
        cell = _Cell(on_close=lambda: stop.append(True), should_cancel=lambda: bool(stop))

        report = cell.run()

        self.assertEqual(["close"], cell.hand.commands, "the blocker was set down after the stop")
        self.assertIs(PickOutcome.ABORTED, report.outcome)
        stopped = [push for push in cell.orchestrator.pushes if push.stopped]
        self.assertEqual(["clear_the_blocker"], [push.trigger for push in stopped])

    def test_a_camera_world_that_cannot_vouch_while_the_blocker_is_picked_leaves_it_to_a_person(self) -> None:
        """The pick raises, as every motion of a pick raises it, and is kept as one that stopped where the arm stands."""
        from src.robot.core.errors import CameraWorldUnavailable

        cell = _Cell()
        moved = cell.arm.move

        def move(pose: Any, **keywords: Any) -> Any:
            if cell.calculator.calls.count("blocker"):
                raise CameraWorldUnavailable(camera="wrist", verdict="blind", attempts=4,
                                             reason="the wrist camera could not vouch for the cell on the way")
            return moved(pose, **keywords)

        cell.arm.move = move  # type: ignore[method-assign]

        with self.assertRaises(CameraWorldUnavailable):
            cell.run()

        stopped = [push for push in cell.orchestrator.pushes if push.stopped]
        self.assertEqual(["clear_the_blocker"], [push.trigger for push in stopped])
        self.assertEqual([], cell.hand.commands)

    def test_no_attempt_left_to_pick_the_part_after_moves_nothing(self) -> None:
        cell = _Cell()
        cell.orchestrator.max_attempts = 1

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertEqual("refused_no_attempt_left", cell.orchestrator.blockers[0].code)


class TheBlockerIsNeverThePartOrTheRobotTests(unittest.TestCase):
    def test_a_cluster_that_does_not_stand_on_the_support_is_no_blocker(self) -> None:
        """A post held in the air beside the part (a finger, a cable) is seen, and never gripped as a blocker."""
        cell = _Cell()
        floating = replace(POST, low=(POST.low[0], POST.low[1], 30.0), high=(POST.high[0], POST.high[1], 70.0))
        cell.scene.boxes = [CUBE, floating]

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertTrue(cell.orchestrator.blockers)
        self.assertTrue(all(record.code != "set_aside" for record in cell.orchestrator.blockers))
        self.assertNotIn("blocker", cell.calculator.calls, "a cluster in the air was asked to be gripped")


#: A mat 55 mm tall under the part and the post, as on the owner's cell: no support model reads it, the bench reads under
#: the post, and the part's grasp is planned on what its own cloud reads (``refine_from_target``), the mat.
MAT_TOP_MM = 55.0
MAT = Box((-200.0, -800.0, 0.0), (120.0, -560.0, MAT_TOP_MM), "mat")


def _on_the_mat(box: Box, *, over_mm: float = 0.0) -> Box:
    lift = MAT_TOP_MM + over_mm
    return Box((box.low[0], box.low[1], box.low[2] + lift), (box.high[0], box.high[1], box.high[2] + lift), box.label)


def _mat_cell(**wiring: Any) -> "_Cell":
    cell = _Cell(**wiring)
    cell.scene.boxes = [_on_the_mat(CUBE), _on_the_mat(POST), MAT]
    # No support model reads the mat here, as none reads it under the parts inside a pile: the bench reads under the post.
    cell.world.tuning = replace(TUNING, support_surfaces=False)
    cell.orchestrator.support_config = GraspingSupportConfig(height_mm=0.0, refine_from_target=True)
    return cell


class ABlockerInsideAPileOnAMatIsTakenAwayTests(unittest.TestCase):
    """The owner's piles on the mat (the cell, 2026-10-07; built 2026-10-09: "Blocker mitten im Haufen"): inside a pile
    the camera sees no mat under the parts, and the bench read under a neighbour put its foot 55 mm and more over what it
    stands on, so it was never taken away. A neighbour whose foot stands within ``STANDS_ON_THE_SUPPORT_MM`` over what
    the part stands on stands on that, and its grasps are planned on it."""

    def test_a_post_on_the_mat_with_the_bench_under_it_is_set_aside(self) -> None:
        """Red before: the bench read under the post put its foot 55 mm over what it stands on: no blocker."""
        cell = _mat_cell(blocker_place=Pose.tool_down(200.0, -400.0, 20.0))

        report = cell.run()

        self.assertEqual("set_aside", report.attempts[0].blocker, [a.action for a in report.attempts])
        self.assertEqual(["close", "open", "close"], cell.hand.commands)

    def test_its_grasps_are_planned_on_the_mat_never_on_the_bench(self) -> None:
        cell = _mat_cell(blocker_place=Pose.tool_down(200.0, -400.0, 20.0))
        planes: list[Any] = []
        real = cell.calculator.compute_result

        def noting(seg: Any, depth: Any, *args: Any, **kwargs: Any) -> GraspResult:
            if str(getattr(seg, "label", "")) == "blocker":
                planes.append(kwargs.get("support_plane"))
            return real(seg, depth, *args, **kwargs)

        cell.calculator.compute_result = noting  # type: ignore[method-assign]
        cell.run()

        self.assertTrue(planes)
        self.assertTrue(all(plane is not None and float(plane.offset_mm) >= MAT_TOP_MM for plane in planes),
                        [None if plane is None else float(plane.offset_mm) for plane in planes])

    def test_a_post_over_the_mat_that_does_not_stand_on_it_is_still_no_blocker(self) -> None:
        cell = _mat_cell(blocker_place=Pose.tool_down(200.0, -400.0, 20.0))
        cell.scene.boxes = [_on_the_mat(CUBE), _on_the_mat(POST, over_mm=30.0), MAT]

        cell.run()

        self.assertNotIn("blocker", cell.calculator.calls, "a post 30 mm over the mat was asked to be gripped")

    def test_the_plane_a_blocker_s_grasps_are_planned_on_is_never_lower_than_what_it_stands_on(self) -> None:
        from src.geometry import Frame
        from src.robot.grasping.collision import SupportPlane
        from src.robot.grasping.loop.pick_loop import _plane_at_least

        bench = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=0.0, frame=Frame.BASE)
        raised = _plane_at_least(bench, MAT_TOP_MM, (10.0, -700.0))
        self.assertAlmostEqual(MAT_TOP_MM, float(raised.offset_mm))
        self.assertIs(Frame.BASE, raised.frame)
        higher = SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=70.0, frame=Frame.BASE)
        self.assertIs(higher, _plane_at_least(higher, MAT_TOP_MM, (10.0, -700.0)))
        self.assertIs(bench, _plane_at_least(bench, None, (10.0, -700.0)))


def _models_a_carried_part(arm: Any, length_mm: float) -> None:
    """The arm models a carried part ``length_mm`` past the fingertips, as a UR arm with ``planning_world.payload``."""
    arm.payload_declined_reason = lambda: None
    arm.config = SimpleNamespace(safety=SimpleNamespace(planning_world=SimpleNamespace(
        payload=SimpleNamespace(length_mm=length_mm))))


class ABlockerThePlannerCannotCarryIsNotTakenTests(unittest.TestCase):
    """Between the close and the release only the planner holds a part (the console's rule): it models one
    ``safety.planning_world.payload.length_mm`` past the fingertips. A blocker reaching further would be carried partly
    where nobody judges it, so it is not taken (the lead's review of 2026-10-02). The post, gripped at half its 40 mm,
    reaches about 10 mm past the Hand-E's fingertips."""

    def test_a_blocker_reaching_past_the_modelled_part_is_not_taken(self) -> None:
        cell = _Cell()
        _models_a_carried_part(cell.arm, length_mm=2.0)

        cell.run()

        self.assertEqual([], cell.hand.commands)
        self.assertEqual("too_long_to_carry", cell.orchestrator.blockers[0].code)
        self.assertIn("2 mm", cell.orchestrator.blockers[0].sentence)
        self.assertTrue(cell.orchestrator.pushes, "the push was not considered after the blocker")

    def test_a_blocker_within_the_modelled_part_is_set_aside(self) -> None:
        cell = _Cell()
        _models_a_carried_part(cell.arm, length_mm=60.0)

        report = cell.run()

        self.assertEqual("set_aside", report.attempts[0].blocker)
        self.assertEqual(["close", "open", "close"], cell.hand.commands)

    def test_an_arm_that_models_no_carried_part_takes_it_as_before(self) -> None:
        """A desk arm models nothing, as for every part it picks: nothing changes there."""
        report = _Cell().run()

        self.assertEqual("set_aside", report.attempts[0].blocker)


def _refusing_the_line_down_to(cell: "_Cell", label: str, *, status: MotionStatus) -> list[Any]:
    """``cell``'s arm refuses, before anything is sent, the line in to the grasp at the centre of ``label``; the calls
    it refused, in order."""
    real = cell.arm.move
    refused: list[Any] = []

    def move(pose: Any, **keywords: Any) -> MotionResult:
        at = np.asarray(pose.position_mm, dtype=np.float64)
        centre = cell.scene.centre(label)
        if keywords.get("linear") and float(np.hypot(*(at[:2] - centre[:2]))) < 1.0 and at[2] <= centre[2] + 1.0:
            refused.append(pose)
            return MotionResult.failed(status, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="the guard refused the line in")
        return real(pose, **keywords)

    cell.arm.move = move  # type: ignore[method-assign]
    return refused


class ABlockerGraspRefusedInTheAirGoesBackToTheLookTests(unittest.TestCase):
    """The URSim gate of 2026-10-02 met it three times in eight: the line in to the blocker refused once the arm stood at
    its standoff, or its carried lift refused before the close. The hand never closed, so the arm goes back to its look
    on a judged move, the blocker is not tried again, and the pick goes on to its next action, as after a blocker
    refused before anything was sent; no person is needed for a hand known empty and open. The hand here measures its width
    (``core.gripper.why_not_known_open`` reads it open at 50 mm, closed at 0), as the owner's toggle counts its own."""

    @staticmethod
    def _measuring(cell: "_Cell") -> "_Cell":
        cell.hand.width_is_measured = lambda: True  # type: ignore[attr-defined]
        return cell

    def test_a_line_in_refused_at_the_standoff_goes_back_to_the_look(self) -> None:
        cell = self._measuring(_Cell())
        refused = _refusing_the_line_down_to(cell, "post", status=MotionStatus.SELF_COLLISION_REJECTED)

        report = cell.run()

        self.assertTrue(refused, "the line in to the blocker was never asked")
        self.assertEqual([], cell.hand.commands, "the hand was commanded")
        record = cell.orchestrator.blockers[0]
        self.assertEqual("blocker_not_reached", record.code, record.sentence)
        self.assertTrue(record.moved)
        self.assertIn("went back to the look", record.sentence)
        joints = [key for kind, key in cell.arm.motions if kind == "joints"]
        self.assertGreaterEqual(len(joints), 2)
        self.assertEqual(joints[0], joints[-1], "the arm did not go back to the look it judged the part on")
        self.assertNotIn("stopped_where_the_arm_stands", [r.code for r in cell.orchestrator.blockers])
        self.assertTrue(cell.orchestrator.pushes, "the push was not considered after the blocker")
        self.assertNotIn(report.outcome, (PickOutcome.ABORTED, PickOutcome.GRIPPER_FAULT))

    def test_the_arm_leaves_the_blocker_in_the_world_it_came_down_in(self) -> None:
        """The URSim re-run of 2026-10-02: with the blocker back in the world, the open jaws round it stood inside its
        box, and the guard and the planner refused every move from there. The move back is judged with the blocker
        still held out, as the line down to it was, and the part is offered again once the arm is back."""
        cell = self._measuring(_Cell())
        refused = _refusing_the_line_down_to(cell, "post", status=MotionStatus.SELF_COLLISION_REJECTED)
        held_at: list[tuple[int, list[str]]] = []
        real = cell.arm.move_to_joints

        def move_to_joints(joints: Any, **keywords: Any) -> MotionResult:
            held_at.append((len(refused), [str(offer.get("target_label", "")) for offer in cell.world.held_offers]))
            return real(joints, **keywords)

        cell.arm.move_to_joints = move_to_joints  # type: ignore[method-assign]

        cell.run()

        self.assertEqual("blocker_not_reached", cell.orchestrator.blockers[0].code, cell.orchestrator.blockers[0].sentence)
        back = [labels for after, labels in held_at if after]
        self.assertTrue(back, "the arm never moved back after the refused line in")
        self.assertIn("blocker", back[0], "the move back was judged with the blocker in the world")
        self.assertNotIn("blocker", [str(offer.get("target_label", "")) for offer in cell.world.held_offers])

    def test_the_refused_blocker_is_not_tried_again(self) -> None:
        cell = self._measuring(_Cell(gate=True))
        _refusing_the_line_down_to(cell, "post", status=MotionStatus.SELF_COLLISION_REJECTED)

        cell.run()

        self.assertEqual(1, sum(1 for r in cell.orchestrator.blockers if r.code == "blocker_not_reached"))

    def test_a_close_that_ran_before_a_refusal_still_leaves_it_to_a_person(self) -> None:
        """A refusal once the jaws closed on the blocker: the hand holds it, so nothing goes back on its own."""
        cell = self._measuring(_Cell())
        real = cell.arm.move
        closes = {"n": 0}

        def move(pose: Any, **keywords: Any) -> MotionResult:
            if cell.hand.closed and keywords.get("linear"):
                closes["n"] += 1
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           target_pose=pose, message="the guard refused the lift")
            return real(pose, **keywords)

        cell.arm.move = move  # type: ignore[method-assign]

        cell.run()

        self.assertGreaterEqual(closes["n"], 1)
        self.assertEqual("stopped_where_the_arm_stands", cell.orchestrator.blockers[-1].code)


class ABlockerIsTriedOnItsNextGraspTests(unittest.TestCase):
    """``recovery.blocker_grasp_tries`` (the owner, 2026-10-03): a blocker's grasps are tried best first, each judged
    from the look before the arm leaves for it (``GraspExecutionPolicy.refusal_ahead``). A grasp refused there costs no
    motion; one refused once the arm stood over the blocker sends the arm back to the look first; the next grasp is
    tried, the count the gate allows at most."""

    @staticmethod
    def _executed(cell: "_Cell") -> list[tuple[float, ...]]:
        """Records the closing axis of every grasp the policy drives to, in order."""
        executed: list[tuple[float, ...]] = []
        real = cell.policy.execute

        def execute(grasp: Any) -> Any:
            executed.append(tuple(round(float(v), 3) for v in grasp.axis))
            return real(grasp)

        cell.policy.execute = execute  # type: ignore[method-assign]
        return executed

    @staticmethod
    def _judged(cell: "_Cell", refuse: Any) -> list[tuple[float, ...]]:
        """The policy's judgement ahead answers ``refuse(number)`` for the n-th grasp asked; the asked axes, in order."""
        asked: list[tuple[float, ...]] = []

        def ahead(grasp: Any) -> str:
            asked.append(tuple(round(float(v), 3) for v in grasp.axis))
            return refuse(len(asked))

        cell.policy.refusal_ahead = ahead  # type: ignore[attr-defined]
        return asked

    def test_a_grasp_judged_ahead_as_refused_moves_nothing_and_the_next_is_taken(self) -> None:
        cell = _Cell()
        cell.calculator.blocker_grasps = 3
        asked = self._judged(cell, lambda n: "the move to the standoff would be refused (test)" if n == 1 else "")
        executed = self._executed(cell)

        cell.run()

        self.assertEqual("set_aside", cell.orchestrator.blockers[0].code, cell.orchestrator.blockers[0].sentence)
        self.assertEqual([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], asked[:2])
        self.assertEqual((0.0, 1.0, 0.0), executed[0], "the grasp refused ahead was driven to")

    def test_the_tries_end_at_the_count_the_gate_allows(self) -> None:
        cell = _Cell()
        cell.calculator.blocker_grasps = 3
        assert cell.orchestrator.push_gate is not None
        cell.orchestrator.push_gate = replace(cell.orchestrator.push_gate, blocker_grasp_tries=2)
        asked = self._judged(cell, lambda n: f"the lift would be refused (test {n})")
        executed = self._executed(cell)

        cell.run()

        record = cell.orchestrator.blockers[0]
        self.assertEqual("blocker_not_reached", record.code, record.sentence)
        self.assertEqual(2, len(asked))
        self.assertIn("None of the blocker's 2 grasp(s)", record.sentence)
        self.assertFalse(record.moved)
        self.assertEqual([], executed, "a blocker grasp refused ahead was driven to")
        self.assertEqual([], cell.hand.commands, "the hand was commanded for a blocker nothing was driven to")

    def test_a_grasp_refused_in_the_air_goes_back_to_the_look_and_the_next_is_taken(self) -> None:
        cell = ABlockerGraspRefusedInTheAirGoesBackToTheLookTests._measuring(_Cell())
        cell.calculator.blocker_grasps = 2
        real = cell.arm.move
        refused: list[Any] = []

        def move(pose: Any, **keywords: Any) -> MotionResult:
            at = np.asarray(pose.position_mm, dtype=np.float64)
            centre = cell.scene.centre("post")
            if (keywords.get("linear") and not refused and float(np.hypot(*(at[:2] - centre[:2]))) < 1.0
                    and at[2] <= centre[2] + 1.0):
                refused.append(pose)
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           target_pose=pose, message="the guard refused the first line in")
            return real(pose, **keywords)

        cell.arm.move = move  # type: ignore[method-assign]
        executed = self._executed(cell)

        cell.run()

        self.assertEqual(1, len(refused))
        self.assertEqual([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], executed[:2])
        self.assertEqual("set_aside", cell.orchestrator.blockers[0].code, cell.orchestrator.blockers[0].sentence)
        joints = [key for kind, key in cell.arm.motions if kind == "joints"]
        self.assertIn(joints[0], joints[1:], "the arm did not go back to the look between the two grasps")


if __name__ == "__main__":
    unittest.main()
