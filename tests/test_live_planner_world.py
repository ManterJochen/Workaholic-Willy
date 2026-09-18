"""The thing that answers "what does the cell look like right now", and when it refuses to.

The failure this guards is not a wrong box. It is a world that looks fine and is out of date: a
planner handed the cell as it was ten seconds ago plans through whatever arrived since, and the
resulting plan is indistinguishable from a plan through empty space at every layer above.

So most of what is asserted here is refusal. No frame, an unstamped frame, a frame past its age, an
arm that cannot say where its own links are: each one has to come back as a snapshot that says it
cannot be planned against, while still carrying the declared world, because the declared world is
the one thing that is true whatever the camera did.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    WorldVerdict,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (
    LinkCapsule,
    PerceptionGeometryError,
    SelfEnvelope,
    WorldBuildLimits,
    WorldBuildTuning,
)
from src.robot.safety.planning.world import planner_cuboid

_FX = _FY = 500.0
_CX = _CY = 100.0
_SHAPE = (200, 200)
_CAMERA_HEIGHT_MM = 1000.0
_CAMERA_TO_BASE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, _CAMERA_HEIGHT_MM],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_INTRINSICS = np.array([[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
_LIMITS = WorldBuildLimits(
    x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0
)
#: A declared bench, exactly as `build_planner_cuboids` would emit it.
_DECLARED = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (1600.0, 1600.0, 50.0)),)
#: The arm, standing out of the way of the scene these tests build: one capsule of 135 mm, which with the
#: world's 15 mm padding is the 150 mm the single polyline segment carried before Step 4h.
_SELF = SelfEnvelope(
    frames_mm=(np.eye(4),),
    capsules=(LinkCapsule(frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, -350.0, 300.0), radius_mm=135.0),),
)


def _scene_with_a_block(*, at_mm: tuple[float, float] = (150.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A bench with one block on it, and the block's mask."""
    depth = np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)
    top_depth = _CAMERA_HEIGHT_MM - 80.0
    scale = top_depth / _FX
    rows, cols = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1]]
    mask = (
        (np.abs(cols - (_CX + at_mm[0] / scale)) <= 30.0 / scale)
        & (np.abs(rows - (_CY - at_mm[1] / scale)) <= 30.0 / scale)
    )
    depth[mask] = top_depth
    return depth, mask


class _Camera:
    """A depth source under the test's control, counting how often it was asked."""

    def __init__(self, snapshot: DepthSnapshot | None) -> None:
        self.snapshot = snapshot
        self.grabs = 0

    def grab_surface_depth(self) -> DepthSnapshot | None:
        self.grabs += 1
        return self.snapshot


def _world(camera: _Camera, **kwargs: object) -> LivePlannerWorld:
    settings: dict[str, object] = {
        "cameras": (
            CameraView(name="overhead", depth_source=camera, camera_to_base=_CAMERA_TO_BASE),
        ),
        "declared": _DECLARED,
        "limits": _LIMITS,
        "tuning": WorldBuildTuning(max_boxes=4),
        "max_age_ms": 500.0,
    }
    settings.update(kwargs)
    return LivePlannerWorld(**settings)  # type: ignore[arg-type]


class FreshWorldTests(unittest.TestCase):
    def test_a_fresh_frame_yields_the_declared_world_plus_what_was_seen(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(self_envelope=_SELF, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.FRESH)
        self.assertTrue(snapshot.usable)
        self.assertEqual(snapshot.declared_count, 1)
        self.assertEqual(snapshot.perceived_count, 1)
        names = [box["name"] for box in snapshot.cuboids]
        self.assertEqual(names[0], "support_plane", "the declared world must come first and survive")
        self.assertTrue(names[1].startswith("seen_"))
        self.assertAlmostEqual(snapshot.age_ms or 0.0, 100.0, delta=1.0)

    def test_a_second_ask_inside_the_age_reuses_the_reading(self) -> None:
        """A grab is I/O and a conversion is arithmetic. Two plans in one cycle share the reading."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        world.world_for(self_envelope=_SELF, now=100.1)
        world.world_for(self_envelope=_SELF, now=100.2)
        self.assertEqual(camera.grabs, 1)

        world.world_for(self_envelope=_SELF, now=101.0)
        self.assertEqual(camera.grabs, 2, "past the age limit the camera has to be asked again")

    @staticmethod
    def _moved(envelope: SelfEnvelope, by_mm: float) -> SelfEnvelope:
        """The same body, every frame of it carried ``by_mm`` along base X."""
        shift = np.eye(4)
        shift[0, 3] = by_mm
        return SelfEnvelope(frames_mm=tuple(shift @ frame for frame in envelope.frames_mm), capsules=envelope.capsules)

    def test_a_frame_taken_before_the_arm_moved_is_not_served_again(self) -> None:
        """Step 4h.3. The frame shows the arm where it stood; filtered with the arm where it stands now, the arm it
        shows would be registered as an obstacle beside the arm. So a body that moved asks the camera again."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        world.world_for(self_envelope=_SELF, now=100.1)
        world.world_for(self_envelope=self._moved(_SELF, 50.0), now=100.2)
        self.assertEqual(camera.grabs, 2, "a reading taken before the arm moved 50 mm was served again")

    def test_two_questions_with_the_arm_still_share_one_reading(self) -> None:
        """The control, green before and after: what the cache is for."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        world.world_for(self_envelope=_SELF, now=100.1)
        world.world_for(self_envelope=_SELF, now=100.2)
        self.assertEqual(camera.grabs, 1)

    def test_a_body_that_moved_less_than_a_millimetre_keeps_the_reading(self) -> None:
        """Bound, green before and after: joint encoder noise is not a motion."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        world.world_for(self_envelope=_SELF, now=100.1)
        world.world_for(self_envelope=self._moved(_SELF, 0.5), now=100.2)
        self.assertEqual(camera.grabs, 1)

    def test_the_arm_is_taken_out_of_what_the_camera_saw(self) -> None:
        """One scene, two arm poses. What changes is which points are the robot itself."""
        depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        frame = DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0)

        over_the_block = _world(_Camera(frame)).world_for(
            self_envelope=SelfEnvelope(
                frames_mm=(np.eye(4),),
                capsules=(
                    LinkCapsule(frame=0, start_mm=(150.0, 0.0, 0.0), end_mm=(150.0, 0.0, 400.0), radius_mm=135.0),
                ),
            ),
            now=100.1,
        )
        self.assertEqual(over_the_block.perceived_count, 0, over_the_block.render())

        elsewhere = _world(_Camera(frame)).world_for(self_envelope=_SELF, now=100.1)
        self.assertEqual(elsewhere.perceived_count, 1, elsewhere.render())


class RefusalTests(unittest.TestCase):
    def test_no_frame_is_not_planned_against_and_keeps_the_declared_world(self) -> None:
        snapshot = _world(_Camera(None)).world_for(self_envelope=_SELF, now=100.0)

        self.assertIs(snapshot.verdict, WorldVerdict.NO_FRAME)
        self.assertFalse(snapshot.usable)
        self.assertEqual([box["name"] for box in snapshot.cuboids], ["support_plane"])
        self.assertIn("no depth", snapshot.reason)
        self.assertIn("no_frame", snapshot.render())

    def test_an_unstamped_frame_counts_as_unknown_age_rather_than_as_fresh(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=None))

        snapshot = _world(camera).world_for(self_envelope=_SELF, now=100.0)

        self.assertIs(snapshot.verdict, WorldVerdict.STALE)
        self.assertIn("no capture time", snapshot.reason)

    def test_a_frame_past_the_age_limit_is_refused_and_says_by_how_much(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(self_envelope=_SELF, now=102.0)

        self.assertIs(snapshot.verdict, WorldVerdict.STALE)
        self.assertIn("2000 ms old", snapshot.reason)
        self.assertIn("500 ms", snapshot.reason)

    def test_an_arm_that_cannot_place_its_own_links_gets_no_perceived_world(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(self_envelope=None, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.UNUSABLE)
        self.assertIn("its own links", snapshot.reason)
        self.assertEqual([box["name"] for box in snapshot.cuboids], ["support_plane"])

    def test_geometry_that_cannot_be_built_is_reported_rather_than_raised(self) -> None:
        """A plan is about to happen. The caller needs a verdict it can refuse on, not a traceback."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        limits = WorldBuildLimits(
            x_mm=(-400.0, 400.0), y_mm=(-400.0, 400.0), z_mm=(-50.0, 900.0),
            support_plane_top_mm=None,
        )

        snapshot = _world(camera, limits=limits).world_for(self_envelope=_SELF, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.UNUSABLE)
        self.assertIn("support_plane", snapshot.reason)


class SegmentationTests(unittest.TestCase):
    def test_offered_masks_name_the_boxes_and_leave_the_target_out(self) -> None:
        depth, mask = _scene_with_a_block(at_mm=(150.0, 0.0))
        far_depth, far_mask = _scene_with_a_block(at_mm=(-150.0, 0.0))
        depth = np.minimum(depth, far_depth)
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)

        both = world.world_for(self_envelope=_SELF, now=100.1)
        self.assertEqual(both.perceived_count, 2, both.render())

        world.offer_segmentation(
            camera="overhead", labelled_masks=(("red cube", far_mask),), exclude_masks=(mask,), timestamp=100.0
        )
        one = world.world_for(self_envelope=_SELF, now=100.1)
        self.assertEqual(one.perceived_count, 1, one.render())
        self.assertEqual(one.cuboids[1]["name"], "seen_00_red_cube")

    def test_masks_older_than_the_age_limit_stop_being_used(self) -> None:
        """A target mask from half a second ago is a hole where the object is no longer standing."""
        depth, mask = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", exclude_masks=(mask,), timestamp=100.0)

        self.assertEqual(world.world_for(self_envelope=_SELF, now=100.1).perceived_count, 0)

        camera.snapshot = DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=102.0)
        self.assertEqual(world.world_for(self_envelope=_SELF, now=102.1).perceived_count, 1)

    def test_forgetting_the_segmentation_puts_the_object_back_in_the_world(self) -> None:
        depth, mask = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", exclude_masks=(mask,), timestamp=100.0)
        self.assertEqual(world.world_for(self_envelope=_SELF, now=100.1).perceived_count, 0)

        world.forget_segmentation()
        self.assertEqual(world.world_for(self_envelope=_SELF, now=100.1).perceived_count, 1)


def _block_top_points(at_mm: tuple[float, float] = (150.0, 0.0)) -> np.ndarray:
    """The top face of the 60 mm block ``_scene_with_a_block`` draws, 80 mm up, as BASE points."""
    xs, ys = np.meshgrid(np.arange(-30.0, 30.1, 5.0) + at_mm[0], np.arange(-30.0, 30.1, 5.0) + at_mm[1])
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 80.0)])


def _two_camera_world(depth: np.ndarray, *, stamp: float = 100.0, **kwargs: object) -> LivePlannerWorld:
    views = (
        CameraView(name="overhead", depth_source=_Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS,
                                                                        timestamp=stamp)),
                   camera_to_base=_CAMERA_TO_BASE),
        CameraView(name="side", depth_source=_Camera(DepthSnapshot(depth_mm=depth.copy(), intrinsics=_INTRINSICS,
                                                                    timestamp=stamp)),
                   camera_to_base=_CAMERA_TO_BASE),
    )
    return _world(_Camera(None), cameras=views, **kwargs)


class KeepOutTests(unittest.TestCase):
    """An offered target box leaves the target out of every view; masks stay with their own camera."""

    def test_a_target_box_leaves_the_target_out_of_every_camera(self) -> None:
        depth, _ = _scene_with_a_block()
        world = _two_camera_world(depth)
        self.assertEqual(1, world.world_for(self_envelope=_SELF, now=100.1).perceived_count)

        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0)
        snapshot = world.world_for(self_envelope=_SELF, now=100.1)

        self.assertEqual(0, snapshot.perceived_count, snapshot.render())
        assert snapshot.perceived is not None
        self.assertGreater(snapshot.perceived.dropped_points.get("keep_out", 0), 0)

    def test_a_wrist_view_loses_the_target_by_its_box_not_its_mask(self) -> None:
        # The offer came from the frame taken at the first tool pose; the refresh reads a frame taken 40 mm further.
        _, mask_then = _scene_with_a_block(at_mm=(150.0, 0.0))
        depth_now, _ = _scene_with_a_block(at_mm=(110.0, 0.0))
        tool_now = _CAMERA_TO_BASE.copy()
        tool_now[0, 3] += 40.0
        camera = _Camera(DepthSnapshot(depth_mm=depth_now, intrinsics=_INTRINSICS, timestamp=100.0,
                                       tool_to_base_mm=tool_now))
        world = _world(camera, cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=np.eye(4)),))
        world.offer_segmentation(camera="wrist", exclude_masks=(mask_then,),
                                 target_points_base_mm=_block_top_points(), timestamp=100.0)

        snapshot = world.world_for(self_envelope=_SELF, now=100.1)

        self.assertEqual(0, snapshot.perceived_count, snapshot.render())
        assert snapshot.perceived is not None
        self.assertEqual(0, snapshot.perceived.dropped_points.get("excluded", 0),
                         "a mask was applied to a camera that moved since it was taken")

    def test_an_offer_for_one_camera_does_not_redate_another(self) -> None:
        depth, mask = _scene_with_a_block()
        world = _two_camera_world(depth, stamp=102.0)
        world.offer_segmentation(camera="overhead", exclude_masks=(mask,), timestamp=100.0)
        world.offer_segmentation(camera="side", exclude_masks=(mask,), timestamp=102.0)

        snapshot = world.world_for(self_envelope=_SELF, now=102.1)

        self.assertEqual(1, snapshot.perceived_count,
                         "the overhead masks are two seconds old and the side offer made them fresh again")

    def test_a_held_offer_outlives_the_age_limit_and_forget_ends_it(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=105.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0,
                                 hold=True)
        self.assertEqual(0, world.world_for(self_envelope=_SELF, now=105.1).perceived_count)

        world.forget_segmentation()
        self.assertEqual(1, world.world_for(self_envelope=_SELF, now=105.1).perceived_count)

    def test_an_unheld_box_ages_like_its_frame(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=105.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0)
        self.assertEqual(1, world.world_for(self_envelope=_SELF, now=105.1).perceived_count)

    def test_forgetting_a_pick_drops_the_frames_taken_while_its_target_was_held(self) -> None:
        """A pick that ends has changed the cell: the frame cached while its target was held shows the part in the hand.

        Measured on the box on 2026-09-17: after a lift the M2 runner opens the jaw and parks without moving the arm
        first, the world served the frame from the lift, and the released cube between the fingers refused every park.
        """
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0,
                                 hold=True)
        world.world_for(self_envelope=_SELF, now=100.1)
        self.assertEqual(1, camera.grabs)

        world.forget_segmentation()
        world.world_for(self_envelope=_SELF, now=100.2)

        self.assertEqual(2, camera.grabs, "the first motion after the pick planned against a frame taken during it")

    def test_a_neighbour_outside_the_grown_box_stays(self) -> None:
        depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        far_depth, _ = _scene_with_a_block(at_mm=(-150.0, 0.0))
        camera = _Camera(DepthSnapshot(depth_mm=np.minimum(depth, far_depth), intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera)
        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0)

        snapshot = world.world_for(self_envelope=_SELF, now=100.1)

        self.assertEqual(1, snapshot.perceived_count, snapshot.render())
        assert snapshot.perceived is not None
        self.assertLess(snapshot.perceived.boxes[0].center_mm[0], 0.0, "the neighbour at x -150 stayed")

    def test_a_target_box_leaves_the_target_out_of_the_live_scene_field(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera, tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))
        before = world.world_for(self_envelope=_SELF, now=100.1)
        assert before.perceived is not None and before.perceived.voxels is not None
        self.assertGreater(before.perceived.voxels.occupied, 0)

        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0)
        after = world.world_for(self_envelope=_SELF, now=100.1)

        assert after.perceived is not None and after.perceived.voxels is not None
        self.assertEqual(0, after.perceived.voxels.occupied)

    def test_a_refresh_that_keeps_everything_out_still_replaces_the_field(self) -> None:
        """The planner keeps the last field until another replaces it: sending none would keep the target in it."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(camera, tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))
        planner = _ScenePlanner()
        refresh_planner_world(source=world, client=planner, self_envelope=_SELF, now=100.1)

        world.offer_segmentation(camera="overhead", target_points_base_mm=_block_top_points(), timestamp=100.0)
        refresh = refresh_planner_world(source=world, client=planner, self_envelope=_SELF, now=100.1)

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(2, len(planner.scene_calls), "the second refresh sent no field")
        values = np.load(planner.scene_calls[-1]["voxels"]["path"])
        self.assertGreater(float(values.min()), 0.0, "a field with nothing in it reads free everywhere")


class TwoCamerasThroughThePickLoopTests(unittest.TestCase):
    def test_a_two_camera_world_leaves_the_target_out_of_both_views_through_the_pick_loop(self) -> None:
        from types import SimpleNamespace

        from src.geometry import Frame, Transform
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator
        from src.robot.grasping.types.perception import PerceptionFrame

        depth, mask = _scene_with_a_block()
        world = _two_camera_world(depth)
        orchestrator = BinPickingOrchestrator(
            arm=SimpleNamespace(live_planner_world=world),  # type: ignore[arg-type]
            calculator=None,  # type: ignore[arg-type]
            perception=SimpleNamespace(streamer=SimpleNamespace(rig_id="overhead")),  # type: ignore[arg-type]
        )
        orchestrator._pending_camera_to_base = Transform.from_matrix(  # noqa: SLF001
            _CAMERA_TO_BASE, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        frame = PerceptionFrame(depth_map=depth, intrinsics=_INTRINSICS,
                                segmentations=(SimpleNamespace(label="red cube", mask=mask),), timestamp=100.0)
        orchestrator._offer_masks_to_planner_world(frame, 0)  # noqa: SLF001
        try:
            snapshot = world.world_for(self_envelope=_SELF, now=100.1)
        finally:
            orchestrator._keep_out.close()  # noqa: SLF001

        self.assertEqual(0, snapshot.perceived_count, snapshot.render())
        self.assertEqual(1, world.world_for(self_envelope=_SELF, now=100.1).perceived_count,
                         "closing the scope did not put the target back")

    def test_masks_without_a_camera_are_refused_and_a_box_alone_is_not(self) -> None:
        depth, mask = _scene_with_a_block()
        world = _two_camera_world(depth)
        with self.assertRaises(PerceptionGeometryError) as caught:
            world.offer_segmentation(exclude_masks=(mask,), timestamp=100.0)
        self.assertIn("names the camera", str(caught.exception))

        world.offer_segmentation(target_points_base_mm=_block_top_points(), timestamp=100.0)
        self.assertEqual(0, world.world_for(self_envelope=_SELF, now=100.1).perceived_count)


class _Planner:
    """A planner client under the test's control, recording what it was handed."""

    def __init__(self, *, confirm_all: bool = True, take_voxels: bool = True) -> None:
        self.worlds: list[list[dict]] = []
        self.meshes: list[list[dict]] = []
        self.voxel_calls: list[dict] = []
        self.confirm_all = confirm_all
        self.take_voxels = take_voxels

    def set_world(self, cuboids: list[dict], meshes: "list[dict] | None" = None) -> int:
        self.worlds.append(list(cuboids))
        self.meshes.append(list(meshes or ()))
        total = len(cuboids) + len(meshes or ())
        return total if self.confirm_all else max(0, total - 1)

    def set_voxels(self, path, *, dims_m, voxel_size_m, pose):  # noqa: ANN001, ANN201
        self.voxel_calls.append(
            {"path": path, "dims_m": list(dims_m), "voxel_size_m": voxel_size_m, "pose": list(pose)}
        )
        if not self.take_voxels:
            return None
        return int(np.load(path).shape[0])


class _OldPlanner:
    """A planner client from before the live scene existed, which must keep working."""

    def __init__(self) -> None:
        self.worlds: list[list[dict]] = []

    def set_world(self, cuboids: list[dict]) -> int:
        self.worlds.append(list(cuboids))
        return len(cuboids)


class RefreshTests(unittest.TestCase):
    """What a driver actually calls: build the world, register it, or refuse the motion."""

    def _source(self, **kwargs: object) -> LivePlannerWorld:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        return _world(camera, **kwargs)

    def test_a_good_refresh_registers_the_world_and_reports_its_price(self) -> None:
        planner = _Planner()
        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(len(planner.worlds), 1)
        self.assertEqual([box["name"] for box in planner.worlds[0]][0], "support_plane")
        self.assertGreater(refresh.build_ms, 0.0)
        self.assertEqual(refresh.registered, refresh.sent)
        self.assertIn("planner world refreshed", refresh.render())

    def test_the_live_scene_goes_as_a_file_and_the_declared_world_goes_with_it(self) -> None:
        planner = _Planner()
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(len(planner.voxel_calls), 1)
        call = planner.voxel_calls[0]
        self.assertTrue(str(call["path"]).endswith(".npy"))
        self.assertAlmostEqual(call["voxel_size_m"], 0.03)
        self.assertEqual(len(call["pose"]), 7, "a pose is position plus a WXYZ quaternion")
        self.assertGreater(refresh.voxels_registered or 0, 0)
        self.assertIn("live scene", refresh.render())
        # The boxes still went, because the guard cannot read a distance field and an operator
        # cannot read one either.
        self.assertGreaterEqual(len(planner.worlds[0]), 2)
        self.assertGreater(len(refresh.guard_boxes), 0)

    def test_a_planner_with_no_live_scene_channel_refuses_rather_than_pretending(self) -> None:
        """A cell that asked for a live scene and cannot have one must not plan as if it had one."""
        planner = _OldPlanner()
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("no live-scene channel", refresh.reason)
        self.assertIn("voxel reservation", refresh.reason)

    def test_a_cell_that_asked_for_no_live_scene_is_untouched_by_any_of_this(self) -> None:
        """The byte-identical path: no field configured, so nothing is sent and nothing refuses."""
        planner = _OldPlanner()

        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertIsNone(refresh.voxels_registered)
        self.assertEqual(len(planner.worlds), 1)

    def test_a_refused_live_scene_is_a_refused_motion(self) -> None:
        """Asking for a live scene and silently not getting one is the defect this exists to end."""
        planner = _Planner(take_voxels=False)
        source = self._source(tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))

        refresh = refresh_planner_world(
            source=source, client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("the planner refused it", refresh.reason)
        self.assertEqual(refresh.guard_boxes, (), "a refused refresh must not leave the guard armed")

    def test_a_declared_mesh_reaches_the_planner_beside_the_boxes(self) -> None:
        """A container is the case: as a box it is solid, as a mesh it keeps the hollow."""
        planner = _Planner()
        source = self._source(
            declared_meshes=(
                {
                    "name": "tote",
                    "file_path": "assets/tote.obj",
                    "pose": [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "scale": [0.001, 0.001, 0.001],
                },
            )
        )

        refresh = refresh_planner_world(
            source=source, client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual([m["name"] for m in planner.meshes[0]], ["tote"])
        self.assertEqual(
            refresh.sent, len(planner.worlds[0]) + 1, "a mesh counts as an obstacle that was sent"
        )

    def test_a_cell_with_no_mesh_never_names_the_parameter(self) -> None:
        """A planner client from before meshes existed takes one argument, and must keep working."""
        planner = _OldPlanner()

        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())

    def test_a_partial_registration_refuses_and_says_the_two_counts(self) -> None:
        planner = _Planner(confirm_all=False)
        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("confirmed", refresh.reason)
        self.assertEqual(refresh.guard_boxes, ())

    def test_a_stale_frame_never_reaches_the_planner_at_all(self) -> None:
        """A frame past the age limit is asked again and then raises, registering nothing.

        Until Step 4e this returned a refused refresh. It raises now, after the camera's fresh-frame
        attempts (owner, Step 4), and what it pinned still holds: nothing reached the planner.
        """
        from src.robot.core.errors import CameraWorldUnavailable

        planner = _Planner()
        with self.assertRaises(CameraWorldUnavailable) as caught:
            refresh_planner_world(
                source=self._source(), client=planner, self_envelope=_SELF, now=102.0
            )

        self.assertEqual(planner.worlds, [], "nothing is registered from a world nobody can vouch for")
        self.assertIs(caught.exception.verdict, WorldVerdict.STALE)
        self.assertIn("NOT refreshed", caught.exception.refresh.render())


class _ScenePlanner(_Planner):
    """A planner client that registers a whole scene in one request, as the sidecar does from Step 4 on.

    It keeps the two older methods from `_Planner`, so a refresh that still reaches for them shows up
    in `worlds` and `voxel_calls` rather than passing by accident.
    """

    def __init__(self, *, confirm_all: bool = True, voxels_reason: str = "") -> None:
        import tempfile
        from pathlib import Path

        super().__init__(confirm_all=confirm_all, take_voxels=not voxels_reason)
        self.scene_calls: list[dict] = []
        self.voxels_reason = voxels_reason
        self.live_scene_path = str(Path(tempfile.gettempdir()) / f"willy_test_scene_{id(self)}.npy")

    def set_scene(self, cuboids, meshes, voxels):  # noqa: ANN001, ANN201
        from types import SimpleNamespace

        self.scene_calls.append(
            {
                "cuboids": list(cuboids),
                "meshes": list(meshes or ()),
                "voxels": None if voxels is None else dict(voxels),
            }
        )
        total = len(cuboids) + len(meshes or ())
        registered = total if self.confirm_all else max(0, total - 1)
        if self.voxels_reason:
            return SimpleNamespace(world_set=registered, voxels_set=None, reason=self.voxels_reason)
        count = None if voxels is None else int(np.load(voxels["path"]).shape[0])
        return SimpleNamespace(world_set=registered, voxels_set=count, reason="")


class OneRegistrationTests(unittest.TestCase):
    """A field and the declared world reach the planner in ONE request, and its reason reaches the refusal.

    The defect behind the first test: the world went as one request and the field as a second, and the
    sidecar's field branch rebuilt its scene from the cuboids it remembered. A declared mesh therefore
    vanished the moment a camera produced a field, and nothing refused, because both requests reported
    success. The defect behind the other two: the sidecar says exactly why it would not take a field,
    and the operator was told "the planner refused it", or, when the counts also disagreed, only that.
    """

    _TOTE = {
        "name": "tote",
        "file_path": "assets/tote.obj",
        "pose": [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "scale": [0.001, 0.001, 0.001],
    }
    _NO_STORAGE = (
        "this planner was started with no voxel storage; set WILLY_CUROBO_VOXEL_GRID before it starts"
    )

    def _source(self, **kwargs: object) -> LivePlannerWorld:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        settings: dict[str, object] = {
            "tuning": WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0),
            "declared_meshes": (self._TOTE,),
        }
        settings.update(kwargs)
        return _world(camera, **settings)

    def test_a_field_and_the_declared_meshes_go_in_one_registration(self) -> None:
        planner = _ScenePlanner()

        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(
            len(planner.scene_calls), 1, "the field and the world must reach the planner in one request"
        )
        self.assertEqual(planner.worlds, [], "a separate world request leaves the field to replace it")
        self.assertEqual(planner.voxel_calls, [], "a separate field request drops the declared meshes")
        (call,) = planner.scene_calls
        self.assertEqual(call["cuboids"][0]["name"], "support_plane")
        self.assertEqual([m["name"] for m in call["meshes"]], ["tote"])
        self.assertEqual(
            call["voxels"]["path"], planner.live_scene_path,
            "the field goes to the file this client owns, so two arms cannot overwrite each other",
        )
        self.assertAlmostEqual(call["voxels"]["voxel_size_m"], 0.03)
        self.assertEqual(len(call["voxels"]["pose"]), 7)
        self.assertGreater(refresh.voxels_registered or 0, 0)

    def test_the_sidecar_reason_reaches_the_refusal(self) -> None:
        planner = _ScenePlanner(voxels_reason=self._NO_STORAGE)

        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("started with no voxel storage", refresh.reason)
        self.assertEqual(refresh.guard_boxes, (), "a refused refresh must not leave the guard armed")

    def test_a_count_refusal_does_not_hide_a_field_refusal(self) -> None:
        planner = _ScenePlanner(confirm_all=False, voxels_reason=self._NO_STORAGE)

        refresh = refresh_planner_world(
            source=self._source(), client=planner, self_envelope=_SELF, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIn("confirmed", refresh.reason)
        self.assertIn("started with no voxel storage", refresh.reason)

    def test_a_scene_with_no_field_still_goes_as_a_world(self) -> None:
        """The control, green before and after: no field configured, nothing changes on the wire."""
        planner = _ScenePlanner()

        refresh = refresh_planner_world(
            source=self._source(tuning=WorldBuildTuning(max_boxes=4)),
            client=planner, self_envelope=_SELF, now=100.1,
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(planner.scene_calls, [])
        self.assertEqual(len(planner.worlds), 1)
        self.assertIsNone(refresh.voxels_registered)


class TheFieldOnTheWireTests(unittest.TestCase):
    """The file the planner reads holds metres, negative inside an obstacle.

    Measured on this box on 2026-09-13 (tests/test_voxel_field_sign.py carries the numbers): the planner
    reads the field in metres and negative inside. The code wrote millimetres, positive inside, so every
    voxel that was not an obstacle read as hundreds of metres inside one.
    """

    def test_the_block_is_negative_and_the_values_are_metres(self) -> None:
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        source = _world(camera, tuning=WorldBuildTuning(max_boxes=4, voxel_field_mm=30.0))
        planner = _Planner()

        refresh = refresh_planner_world(source=source, client=planner, self_envelope=_SELF, now=100.1)

        self.assertTrue(refresh.ok, refresh.render())
        values = np.load(planner.voxel_calls[0]["path"]).astype(np.float64)
        shape = (27, 27, 30)  # 800 x 800 x 900 mm at 30 mm, as the grid is cut from _LIMITS
        self.assertEqual(values.size, int(np.prod(shape)))
        grid = values.reshape(shape)
        # The block stands at x 150, y 0 with its top 80 mm up; the grid is indexed from (-400, -400, 0).
        self.assertLess(grid[18, 13, 1], 0.0, "the block must read negative, as the planner reads inside")
        self.assertGreater(grid[1, 1, 28], 0.0, "far free space must read positive")
        self.assertLess(
            float(np.abs(values).max()), 2.0,
            "a field over a two metre cell in metres stays under two; millimetres read hundreds",
        )


def _blind() -> DepthSnapshot:
    return DepthSnapshot(
        depth_mm=np.zeros(_SHAPE, dtype=np.float64), intrinsics=_INTRINSICS, timestamp=100.0
    )


class BlindFrameTests(unittest.TestCase):
    """A camera that sees nothing is blind, and a blind camera is not an empty cell.

    A frame with no valid depth converted into an empty perceived world and came back FRESH, so a
    covered lens, a dead emitter or a frame of zeros read exactly like a bench with nothing on it.
    """

    def test_a_frame_with_no_valid_pixel_is_blind_and_names_the_camera(self) -> None:
        cases = {
            "zeros": np.zeros(_SHAPE, dtype=np.float64),
            "not a number": np.full(_SHAPE, np.nan, dtype=np.float64),
            "negative": np.full(_SHAPE, -1.0, dtype=np.float64),
        }
        for label, depth in cases.items():
            with self.subTest(label):
                camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

                snapshot = _world(camera).world_for(self_envelope=_SELF, now=100.1)

                self.assertIs(snapshot.verdict, WorldVerdict.BLIND)
                self.assertFalse(snapshot.usable)
                self.assertEqual(snapshot.camera, "overhead")
                self.assertIn("overhead", snapshot.reason)
                self.assertEqual([box["name"] for box in snapshot.cuboids], ["support_plane"])

    def test_one_blind_camera_beside_a_sighted_one_is_blind(self) -> None:
        """Fused, the sighted camera's points hid the blind one: the cloud was not empty."""
        depth, _ = _scene_with_a_block()
        sighted = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))
        world = _world(
            sighted,
            cameras=(
                CameraView(name="overhead", depth_source=sighted, camera_to_base=_CAMERA_TO_BASE),
                CameraView(name="side", depth_source=_Camera(_blind()), camera_to_base=_CAMERA_TO_BASE),
            ),
        )

        snapshot = world.world_for(self_envelope=_SELF, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.BLIND)
        self.assertEqual(snapshot.camera, "side")
        self.assertIn("side", snapshot.reason)

    def test_a_blind_frame_is_asked_again(self) -> None:
        """A young blind frame used to be cached like any other, so the camera was not asked again."""
        camera = _Camera(_blind())
        world = _world(camera)

        world.world_for(self_envelope=_SELF, now=100.1)
        world.world_for(self_envelope=_SELF, now=100.2)

        self.assertEqual(camera.grabs, 2)

    def test_a_bench_with_nothing_on_it_stays_fresh(self) -> None:
        """The control, green before and after: valid depth that shows no obstacle is a real answer."""
        bench = np.full(_SHAPE, _CAMERA_HEIGHT_MM, dtype=np.float64)
        camera = _Camera(DepthSnapshot(depth_mm=bench, intrinsics=_INTRINSICS, timestamp=100.0))

        snapshot = _world(camera).world_for(self_envelope=_SELF, now=100.1)

        self.assertIs(snapshot.verdict, WorldVerdict.FRESH)
        self.assertEqual(snapshot.perceived_count, 0)

    def test_the_sim_refusal_names_blindness(self) -> None:
        """A non-UR double: the sim refreshes through the same call and must say the same thing."""
        import time

        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        arm = IsaacRobotArm(SimRobotConfig(enabled=True, scene="placeholder.usd", robot_prim_path="/World/Robot"))
        arm._get_curobo_client = lambda: _Planner()  # type: ignore[method-assign]  # noqa: SLF001
        arm._self_envelope = lambda: _SELF  # type: ignore[method-assign]  # noqa: SLF001
        # Stamped now: the sim refreshes on the real clock, and a frame from t=100 would be refused as
        # stale before anybody looked at whether it holds any depth.
        blind_now = DepthSnapshot(
            depth_mm=np.zeros(_SHAPE, dtype=np.float64), intrinsics=_INTRINSICS, timestamp=time.time()
        )
        arm.set_live_planner_world(_world(_Camera(blind_now)))

        from src.robot.core.errors import CameraWorldUnavailable

        # A blind camera stays blind through its fresh-frame attempts, and from Step 4e that raises
        # rather than returning a reason (owner, Step 4). The sentence still names the blindness.
        with self.assertRaises(CameraWorldUnavailable) as caught:
            arm._refresh_planner_world()  # noqa: SLF001

        self.assertIn("blind", str(caught.exception))
        self.assertIn("overhead", str(caught.exception))


class SlotOverflowTests(unittest.TestCase):
    """An obstacle the cameras saw and the planner has no slot for is a hole in the world."""

    def test_an_obstacle_without_a_slot_refuses_the_refresh(self) -> None:
        near, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        far, _ = _scene_with_a_block(at_mm=(-150.0, 0.0))
        camera = _Camera(
            DepthSnapshot(depth_mm=np.minimum(near, far), intrinsics=_INTRINSICS, timestamp=100.0)
        )
        source = _world(camera, tuning=WorldBuildTuning(max_boxes=1))

        refresh = refresh_planner_world(source=source, client=_Planner(), self_envelope=_SELF, now=100.1)

        self.assertFalse(refresh.ok, refresh.render())
        self.assertIn("1 perceived obstacle(s) did not fit", refresh.reason)
        self.assertEqual(refresh.guard_boxes, (), "a refused refresh must not leave the guard armed")

    def test_a_world_that_fits_still_registers(self) -> None:
        """The control: the same scene with room for both boxes is an ordinary refresh."""
        near, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
        far, _ = _scene_with_a_block(at_mm=(-150.0, 0.0))
        camera = _Camera(
            DepthSnapshot(depth_mm=np.minimum(near, far), intrinsics=_INTRINSICS, timestamp=100.0)
        )

        refresh = refresh_planner_world(
            source=_world(camera), client=_Planner(), self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())


class _SequenceCamera:
    """A camera that answers each grab with the next reading, and with the last one from then on."""

    def __init__(self, *snapshots: "DepthSnapshot | None") -> None:
        self.snapshots = list(snapshots)
        self.grabs = 0

    def grab_surface_depth(self) -> "DepthSnapshot | None":
        index = min(self.grabs, len(self.snapshots) - 1)
        self.grabs += 1
        return self.snapshots[index]


class FreshFrameAttemptsTests(unittest.TestCase):
    """A camera that cannot vouch for the cell is asked again, and after that the motion raises.

    Owner, Step 4: a camera that stays missing, blind or stale after its fresh-frame attempts is not an
    ordinary refusal. The refresh raises `CameraWorldUnavailable`, every verb lets it out, and a pick
    campaign stops. Everything else a refresh can refuse for stays a refusal: a partial registration, a
    refused field, an obstacle without a slot, an arm that cannot place its own links.
    """

    @staticmethod
    def _fresh() -> DepthSnapshot:
        depth, _ = _scene_with_a_block()
        return DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0)

    def test_a_silent_camera_is_asked_n_more_times_then_the_refresh_raises(self) -> None:
        from src.robot.core.errors import CameraWorldUnavailable

        camera = _Camera(None)
        planner = _Planner()

        with self.assertRaises(CameraWorldUnavailable) as caught:
            refresh_planner_world(
                source=_world(camera, fresh_frame_attempts=3), client=planner,
                self_envelope=_SELF, now=100.1,
            )

        self.assertEqual(camera.grabs, 4, "the first reading and three more")
        self.assertEqual(planner.worlds, [], "nothing is registered from a camera nobody can vouch for")
        error = caught.exception
        self.assertEqual((error.camera, error.verdict, error.attempts), ("overhead", WorldVerdict.NO_FRAME, 4))
        self.assertIn("overhead", str(error))
        self.assertIn("no_frame", str(error))

    def test_a_stale_camera_that_recovers_plans(self) -> None:
        stale = DepthSnapshot(depth_mm=self._fresh().depth_mm, intrinsics=_INTRINSICS, timestamp=90.0)
        camera = _SequenceCamera(stale, self._fresh())

        refresh = refresh_planner_world(
            source=_world(camera), client=_Planner(), self_envelope=_SELF, now=100.1
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(camera.grabs, 2)

    def test_a_blind_camera_raises_after_its_attempts(self) -> None:
        from src.robot.core.errors import CameraWorldUnavailable

        camera = _Camera(_blind())

        with self.assertRaises(CameraWorldUnavailable) as caught:
            refresh_planner_world(
                source=_world(camera, fresh_frame_attempts=1), client=_Planner(),
                self_envelope=_SELF, now=100.1,
            )

        self.assertEqual(camera.grabs, 2)
        self.assertIs(caught.exception.verdict, WorldVerdict.BLIND)

    def test_zero_attempts_raises_on_the_first_failure(self) -> None:
        from src.robot.core.errors import CameraWorldUnavailable

        camera = _Camera(None)

        with self.assertRaises(CameraWorldUnavailable):
            refresh_planner_world(
                source=_world(camera, fresh_frame_attempts=0), client=_Planner(),
                self_envelope=_SELF, now=100.1,
            )

        self.assertEqual(camera.grabs, 1)

    def test_a_refusal_that_is_not_the_camera_stays_a_refusal(self) -> None:
        """The control, green before and after: an arm that cannot place its links asks nobody again."""
        camera = _Camera(self._fresh())

        refresh = refresh_planner_world(
            source=_world(camera), client=_Planner(), self_envelope=None, now=100.1
        )

        self.assertFalse(refresh.ok)
        self.assertIs(refresh.verdict, WorldVerdict.UNUSABLE)
        self.assertEqual(camera.grabs, 1)

    def test_the_key_reaches_the_world_source(self) -> None:
        from types import SimpleNamespace

        from src.config.schema.robot import RobotConfig
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig
        from src.geometry import Frame, Transform
        from src.robot.execution.camera_world_wiring import CameraWorldPlan, CameraWorldWiring

        self.assertEqual(PerceivedWorldConfig().fresh_frame_attempts, 3)
        cfg = RobotConfig.model_validate({
            "vendor": "ur",
            "safety": {
                "payload": {"enforce": False},
                "planning_world": {
                    "enabled": True,
                    "require_registration": False,
                    "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
                    "perceived": {"fresh_frame_attempts": 5},
                },
            },
        })
        # The wiring builds the world from a rig that declares its calibration and from its open owner.
        rig = SimpleNamespace(rig_id="overhead", enabled=True, source="rgbd",
                              extrinsics=SimpleNamespace(mounting_mode="eye_to_hand"))
        owner = SimpleNamespace(
            rig_id="overhead",
            handle=lambda: SimpleNamespace(grab=lambda: None, get_intrinsics=lambda: None),
            calibration=lambda: SimpleNamespace(
                mounting_mode="eye_to_hand",
                camera_to_base=lambda: Transform.from_matrix(
                    _CAMERA_TO_BASE, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
            ),
        )
        plan = CameraWorldPlan.from_config(cfg, [rig], primary_rig_id="overhead")

        world = CameraWorldWiring.from_cameras(cfg, plan=plan, cameras={"overhead": owner}).world

        assert world is not None
        self.assertEqual(world.fresh_frame_attempts, 5)
        self.assertFalse(world.require_registration)


class LoopHandsOverItsMasksTests(unittest.TestCase):
    """The pick loop is the only thing that knows what the attempt is reaching for.

    Without this the object being grasped is registered as an obstacle like everything else, and the
    planner refuses the one approach the attempt exists for: the goal sits inside a solid. The offer is
    built by ``segmentation_offer_from_frame`` and held by ``keeping_out`` for the pick.
    """

    @staticmethod
    def _frame(labels: "tuple[str, ...]") -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            segmentations=tuple(
                SimpleNamespace(label=label, mask=np.full((4, 4), index, dtype=np.uint8))
                for index, label in enumerate(labels)
            ),
            depth_map=np.full((4, 4), 500.0), surface_depth_map=None, intrinsics=np.eye(3),
            timestamp=100.0,
        )

    @staticmethod
    def _offer(frame: object, target: "int | None"):  # noqa: ANN205
        from src.robot.grasping.loop.pick_loop import segmentation_offer_from_frame

        return segmentation_offer_from_frame(frame, target, camera="overhead", camera_to_base_mm=None)

    def test_the_target_is_left_out_and_the_others_are_named(self) -> None:
        offer = self._offer(self._frame(("red cube", "blue box")), 1)

        assert offer is not None
        self.assertEqual("overhead", offer.camera)
        self.assertEqual([name for name, _ in offer.labelled_masks], ["red cube", "blue box"])
        self.assertEqual(len(offer.exclude_masks), 1)
        np.testing.assert_array_equal(offer.exclude_masks[0], np.full((4, 4), True))
        self.assertEqual(offer.captured_at_s, 100.0, "masks age with the frame, not with this call")

    def test_no_winner_excludes_nothing_and_still_names_what_was_seen(self) -> None:
        offer = self._offer(self._frame(("red cube",)), None)

        assert offer is not None
        self.assertEqual(offer.exclude_masks, ())
        self.assertEqual(len(offer.labelled_masks), 1)

    def test_an_unnamed_segmentation_still_gets_a_name(self) -> None:
        """A refusal that says `object_2` beats one that says nothing at all."""
        from types import SimpleNamespace

        frame = SimpleNamespace(
            segmentations=(SimpleNamespace(label="", mask=np.zeros((4, 4), dtype=np.uint8)),),
            depth_map=np.full((4, 4), 500.0), surface_depth_map=None, intrinsics=np.eye(3), timestamp=100.0,
        )
        offer = self._offer(frame, 0)

        assert offer is not None
        self.assertEqual(offer.labelled_masks[0][0], "object_0")

    def test_an_arm_with_no_live_world_costs_one_lookup(self) -> None:
        """The byte-identical path, which is every cell that did not ask for a live world."""
        from types import SimpleNamespace

        from src.robot.core.keep_out import keeping_out

        offer = self._offer(self._frame(("x",)), 0)
        assert offer is not None
        for arm in (SimpleNamespace(live_planner_world=None), SimpleNamespace()):
            with keeping_out(arm, offer) as scope:
                self.assertFalse(scope.world_wired)


#: A camera bolted to the tool and looking along its approach: the fixed camera's rotation, carried by the tool.
_CAMERA_TO_TOOL = np.diag([1.0, -1.0, -1.0, 1.0])


def _tool_at(x_mm: float) -> np.ndarray:
    """The TCP in BASE, at the fixed camera's height above the bench and ``x_mm`` along base X."""
    pose = np.eye(4)
    pose[0, 3] = x_mm
    pose[2, 3] = _CAMERA_HEIGHT_MM
    return pose


def _wrist_world(camera: _Camera) -> LivePlannerWorld:
    return _world(
        camera, cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_CAMERA_TO_TOOL),),
    )


class _NoPlanner:
    """A planner that must not be reached: a world nobody can place is refused before it is registered."""

    def set_world(self, *_args: object) -> int:
        raise AssertionError("a world with an unplaced wrist camera reached the planner")


class AWristCameraTests(unittest.TestCase):
    """A camera on the wrist is placed by where the tool was when its shutter opened, frame by frame."""

    def test_a_wrist_frame_is_placed_by_the_pose_it_was_taken_at(self) -> None:
        """One block, seen from two tool poses 92 mm apart, registers at one place in BASE.

        92 mm is 50 pixels at the block's top (920 mm deep at a 500 px focal length), so the two images are the
        same pixels shifted, and what is compared is the placement rather than the rasterisation.
        """
        centres = []
        for tool_x_mm in (0.0, 92.0):
            depth, _ = _scene_with_a_block(at_mm=(150.0 - tool_x_mm, 0.0))
            camera = _Camera(DepthSnapshot(
                depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0, tool_to_base_mm=_tool_at(tool_x_mm),
            ))
            snapshot = _wrist_world(camera).world_for(self_envelope=_SELF, now=100.1)
            self.assertIs(snapshot.verdict, WorldVerdict.FRESH, snapshot.render())
            self.assertEqual(snapshot.perceived_count, 1, snapshot.render())
            assert snapshot.perceived is not None
            centres.append(np.asarray(snapshot.perceived.boxes[0].center_mm, dtype=np.float64))
        np.testing.assert_allclose(centres[0], centres[1], atol=1.0)
        self.assertAlmostEqual(float(centres[0][0]), 150.0, delta=15.0)

    def test_a_view_needs_exactly_one_transform(self) -> None:
        camera = _Camera(None)
        with self.assertRaises(ValueError):
            CameraView(name="both", depth_source=camera, camera_to_base=_CAMERA_TO_BASE,
                       camera_to_tool=_CAMERA_TO_TOOL)
        with self.assertRaises(ValueError):
            CameraView(name="neither", depth_source=camera)

    def test_a_wrist_frame_without_a_pose_is_unusable_and_not_asked_again(self) -> None:
        """Where the tool stood is not something a second reading of the camera can supply."""
        depth, _ = _scene_with_a_block()
        camera = _Camera(DepthSnapshot(depth_mm=depth, intrinsics=_INTRINSICS, timestamp=100.0))

        refresh = refresh_planner_world(
            source=_wrist_world(camera), client=_NoPlanner(), self_envelope=_SELF, now=100.1,
        )

        self.assertIs(refresh.verdict, WorldVerdict.UNUSABLE)
        self.assertFalse(refresh.ok)
        self.assertIn("'wrist'", refresh.reason)
        self.assertIn("tool pose", refresh.reason)
        self.assertEqual(camera.grabs, 1)


if __name__ == "__main__":
    unittest.main()
