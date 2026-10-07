"""A frame is read once for every world built from it, and a question asked twice is answered with one world.

The camera world's build was made fast on 2026-10-07 (``test_a_faster_camera_world_builds_the_same_world``), and two
caches came with it, each only where the build is pure: the same inputs build the same world, bit for bit.

* ``perceived.FrameProjections`` keeps every frame the latest build read, back-projected and thinned, so the next world
  built from the same frame does not read it again: a pick's held frames of its earlier poses above all, which every
  world of the pick reads, each about 360 ms of a build on the cell's PC. Served only for the very same frame: the same
  depth array, the same shutter time, intrinsics, placement and masks, the same stride and voxel.
* ``LivePlannerWorld`` keeps the world it built last with everything it was built from (``live_world._BuildInputs``),
  and answers the same question of the same frames with it: the same frames served again from its frame cache, the
  same body to the bit, the same goal and goal region, the same offers in force, the same tuning. Its age and stamp
  are said afresh. A newer frame, a body moved by anything, another goal, an offer made or aged out, a tuning changed,
  frames dropped or a pick's hold started or ended: each builds afresh.

What either serves is held to what a build without it makes. And the refresh says how much of the time it logs as the
build was the camera's (``WorldRefresh.grab_ms``): a wrist camera throws five frames away before the one it keeps,
which no build can shorten.
"""

from __future__ import annotations

import dataclasses
import time
import unittest

import numpy as np

from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld, refresh_planner_world
from src.robot.safety.planning.perceived import (
    DepthView,
    FrameProjections,
    LinkCapsule,
    SelfEnvelope,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.core.keep_out import GoalKeepOut, KeepOutBox
from tests.test_live_planner_world import (
    _CAMERA_TO_BASE,
    _INTRINSICS,
    _LIMITS,
    _SELF,
    _Camera,
    _Planner,
    _scene_with_a_block,
    _world,
)


def _moved(envelope: SelfEnvelope, by_mm: float) -> SelfEnvelope:
    """The same body, every frame of it carried ``by_mm`` along base X."""
    shift = np.eye(4)
    shift[0, 3] = by_mm
    return SelfEnvelope(frames_mm=tuple(shift @ frame for frame in envelope.frames_mm), capsules=envelope.capsules)


def _two_blocks() -> np.ndarray:
    depth, _ = _scene_with_a_block(at_mm=(150.0, 0.0))
    other, _ = _scene_with_a_block(at_mm=(-150.0, 60.0))
    return np.minimum(depth, other)


def _goal_region(x_mm: float) -> GoalKeepOut:
    tcp = np.eye(4)
    tcp[:3, 3] = (x_mm, 0.0, 100.0)
    return GoalKeepOut(region=KeepOutBox.from_matrix("goal", tcp, (20.0, 10.0, 15.0)), reason="")


class TheSameQuestionIsAnsweredWithOneWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = _Camera(DepthSnapshot(depth_mm=_two_blocks(), intrinsics=_INTRINSICS, timestamp=100.0))
        self.world = _world(self.camera)

    def test_the_same_question_of_the_same_frame_is_answered_with_the_world_built_for_it(self) -> None:
        first = self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.1)
        again = self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.3)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served, self.camera.grabs), (1, 1, 1))
        self.assertIs(again.perceived, first.perceived)
        self.assertEqual(again.cuboids, first.cuboids)
        # The age is the frame's at this question, not at the one before.
        self.assertAlmostEqual(first.age_ms or 0.0, 100.0, delta=1.0)
        self.assertAlmostEqual(again.age_ms or 0.0, 300.0, delta=1.0)

    def test_the_world_served_is_the_world_a_fresh_build_makes(self) -> None:
        self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.1,
                             goal_keep_out=_goal_region(150.0))
        served = self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.2,
                                      goal_keep_out=_goal_region(150.0))
        fresh = _world(_Camera(self.camera.snapshot)).world_for(
            self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.2, goal_keep_out=_goal_region(150.0))
        self.assertEqual(self.world.worlds_served, 1)
        assert served.perceived is not None and fresh.perceived is not None
        self.assertEqual(served.perceived.to_dict(), fresh.perceived.to_dict())
        self.assertEqual(served.to_dict(), fresh.to_dict())

    def test_a_newer_frame_is_never_answered_with_the_world_of_an_older_one(self) -> None:
        self.world.world_for(self_envelope=_SELF, now=100.1)
        # The same depth, read again by the camera: a new frame, past the age the cached one may be served at.
        self.camera.snapshot = DepthSnapshot(depth_mm=self.camera.snapshot.depth_mm, intrinsics=_INTRINSICS,
                                             timestamp=100.7)
        self.world.world_for(self_envelope=_SELF, now=100.8)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served, self.camera.grabs), (2, 0, 2))

    def test_another_goal_or_goal_region_builds_its_own_world(self) -> None:
        self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.0), now=100.1)
        self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.5), now=100.15)
        self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.5), now=100.2,
                             goal_keep_out=_goal_region(150.0))
        self.world.world_for(self_envelope=_SELF, near_point_mm=(150.0, 0.0, 100.5), now=100.25,
                             goal_keep_out=_goal_region(-150.0))
        self.world.world_for(self_envelope=_SELF, near_point_mm=None, now=100.3)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served, self.camera.grabs), (5, 0, 1))

    def test_a_body_moved_by_less_than_the_frame_cache_allows_builds_its_own_world(self) -> None:
        """The frame is served again while the body stands within a millimetre of where it was taken; the world is
        built for the body as it stands, to the bit."""
        self.world.world_for(self_envelope=_SELF, now=100.1)
        self.world.world_for(self_envelope=_moved(_SELF, 0.5), now=100.2)
        self.world.world_for(self_envelope=_moved(_SELF, 1e-9), now=100.3)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served, self.camera.grabs), (3, 0, 1))

    def test_a_body_with_other_capsules_builds_its_own_world(self) -> None:
        self.world.world_for(self_envelope=_SELF, now=100.1)
        wider = SelfEnvelope(frames_mm=_SELF.frames_mm, capsules=(
            dataclasses.replace(_SELF.capsules[0], radius_mm=_SELF.capsules[0].radius_mm + 1.0),))
        self.world.world_for(self_envelope=wider, now=100.2)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served), (2, 0))

    def test_an_offer_and_its_age_build_a_new_world(self) -> None:
        _, mask = _scene_with_a_block(at_mm=(150.0, 0.0))
        self.world.world_for(self_envelope=_SELF, now=100.1)
        # Offered before the frame was taken, so it ages out while the frame is still served.
        self.world.offer_segmentation(camera="overhead", exclude_masks=(mask,), timestamp=99.75)
        with_offer = self.world.world_for(self_envelope=_SELF, now=100.15)
        aged_out = self.world.world_for(self_envelope=_SELF, now=100.3)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served, self.camera.grabs), (3, 0, 1))
        self.assertEqual(with_offer.perceived_count, 1, with_offer.render())
        self.assertEqual(aged_out.perceived_count, 2, aged_out.render())

    def test_named_parts_are_held_as_they_were_offered(self) -> None:
        points = np.array([[150.0, 0.0, 80.0], [160.0, 0.0, 80.0]])
        self.world.offer_segmentation(target_points_base_mm=None, named_points_base_mm=(("cube", points),),
                                      camera="overhead", timestamp=100.0, hold=True)
        first = self.world.world_for(self_envelope=_SELF, now=100.1)
        points[:] = 0.0  # the pick loop's own array, written into after the offer
        again = self.world.world_for(self_envelope=_SELF, now=100.2)
        self.assertEqual(self.world.worlds_served, 1)
        assert first.perceived is not None and again.perceived is not None
        self.assertEqual([box.label for box in again.perceived.boxes], [box.label for box in first.perceived.boxes])

    def test_dropping_the_frames_and_holding_a_pick_let_the_world_go(self) -> None:
        self.world.world_for(self_envelope=_SELF, now=100.1)
        self.world.drop_cached_frames()
        self.world.world_for(self_envelope=_SELF, now=100.15)
        self.world.hold_pick_views()
        self.world.world_for(self_envelope=_SELF, now=100.2)
        self.world.forget_pick_views()
        self.world.world_for(self_envelope=_SELF, now=100.25)
        self.world.forget_segmentation()
        self.world.world_for(self_envelope=_SELF, now=100.3)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served), (5, 0))

    def test_a_changed_tuning_builds_afresh(self) -> None:
        self.world.world_for(self_envelope=_SELF, now=100.1)
        self.world.tuning = dataclasses.replace(self.world.tuning, margin_mm=20.0)
        wider = self.world.world_for(self_envelope=_SELF, now=100.2)
        self.assertEqual((self.world.worlds_built, self.world.worlds_served), (2, 0))
        fresh = _world(_Camera(self.camera.snapshot), tuning=self.world.tuning).world_for(self_envelope=_SELF,
                                                                                           now=100.2)
        assert wider.perceived is not None and fresh.perceived is not None
        self.assertEqual(wider.perceived.to_dict(), fresh.perceived.to_dict())


class AFrameIsReadOncePerWorldTests(unittest.TestCase):
    _TUNING = WorldBuildTuning(max_boxes=8, support_surfaces=True, support_allowance_mm=2.0)

    @staticmethod
    def _views(depth: np.ndarray, *, timestamp: float = 100.0, shift_mm: float = 0.0,
               masks: tuple = ()) -> list[DepthView]:
        other = _CAMERA_TO_BASE.copy()
        other[0, 3] += 30.0 + shift_mm
        return [
            DepthView(surface_depth_mm=depth, intrinsics=_INTRINSICS, camera_to_base=_CAMERA_TO_BASE, name="a",
                      timestamp=timestamp, exclude_masks=masks),
            DepthView(surface_depth_mm=depth, intrinsics=_INTRINSICS, camera_to_base=other, name="b",
                      timestamp=timestamp),
        ]

    def _build(self, views: list[DepthView], projections: "FrameProjections | None") -> dict:
        return build_perceived_boxes(views=views, limits=_LIMITS, tuning=self._TUNING, near_point_mm=(0.0, 0.0, 50.0),
                                     projections=projections).to_dict()

    def test_a_frame_read_by_one_build_is_served_to_the_next_and_builds_the_same_world(self) -> None:
        depth = _two_blocks()
        projections = FrameProjections()
        first = self._build(self._views(depth), projections)
        self.assertEqual((projections.read, projections.served), (2, 0))
        again = self._build(self._views(depth), projections)
        self.assertEqual((projections.read, projections.served), (2, 2))
        self.assertEqual(first, again)
        self.assertEqual(again, self._build(self._views(depth), None))

    def test_another_array_shutter_placement_or_mask_is_read_afresh(self) -> None:
        depth = _two_blocks()
        _, mask = _scene_with_a_block(at_mm=(150.0, 0.0))
        projections = FrameProjections()
        self._build(self._views(depth), projections)
        for views in (self._views(depth.copy()), self._views(depth, timestamp=100.5),
                      self._views(depth, shift_mm=0.25), self._views(depth, masks=(mask,))):
            read = projections.read
            built = self._build(views, projections)
            self.assertEqual(built, self._build(views, None))
            self.assertGreater(projections.read, read)

    def test_only_the_latest_builds_frames_are_kept_and_none_can_be_written_into(self) -> None:
        depth, newer = _two_blocks(), _two_blocks()
        projections = FrameProjections()
        self._build(self._views(depth), projections)
        projections.retain_used()
        self.assertEqual(len(projections), 2)
        self._build(self._views(newer), projections)
        projections.retain_used()
        self.assertEqual(len(projections), 2)
        self._build(self._views(depth), projections)
        self.assertEqual(projections.served, 0, "a frame the latest build did not read was kept")
        kept = projections.of(self._views(depth)[0], depth, _CAMERA_TO_BASE, self._TUNING, "a")
        for array in (kept.points, kept.pixels, kept.ranges, *kept.read, kept.camera_to_base, *kept.in_base()):
            self.assertFalse(array.flags.writeable)
        projections.clear()
        self.assertEqual(len(projections), 0)

    def test_a_picks_held_frame_is_read_once_for_every_world_of_the_pick(self) -> None:
        """A wrist camera at three poses of a pick: from the third world on, the frames of the earlier poses are served
        and only the frame taken now is read. Each world is the one a world without the cache builds."""
        depth = _two_blocks()

        def body(at: float) -> SelfEnvelope:
            return SelfEnvelope(frames_mm=(np.eye(4),), capsules=(
                LinkCapsule(frame=0, start_mm=(at, 400.0, 600.0), end_mm=(at, 380.0, 600.0), radius_mm=30.0),))

        def frame(stamp: float, tool_x: float) -> DepthSnapshot:
            tool = _CAMERA_TO_BASE.copy()
            tool[0, 3] += tool_x
            return DepthSnapshot(depth_mm=depth.copy(), intrinsics=_INTRINSICS, timestamp=stamp, tool_to_base_mm=tool)

        def wrist_world(camera: _Camera) -> LivePlannerWorld:
            return LivePlannerWorld(cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=np.eye(4)),),
                                    declared=(), limits=_LIMITS, tuning=self._TUNING, max_age_ms=500.0)

        camera = _Camera(frame(100.0, 0.0))
        world = wrist_world(camera)
        world.hold_pick_views()
        poses = ((100.0, 0.0, -300.0), (101.0, 20.0, -200.0), (102.0, 40.0, -100.0), (103.0, 60.0, 0.0))
        for number, (stamp, tool_x, at) in enumerate(poses):
            camera.snapshot = frame(stamp, tool_x)
            served = world._projections.served  # noqa: SLF001
            snapshot = world.world_for(self_envelope=body(at), now=stamp + 0.05)
            self.assertEqual(snapshot.held, number, snapshot.render())
            # The frames held from the poses before the last are served; the last one held is read once, as held.
            self.assertEqual(world._projections.served - served, max(0, number - 1))  # noqa: SLF001
            replay = wrist_world(_Camera(camera.snapshot))
            replay.hold_pick_views()
            replay._pick_frames[:] = list(world._pick_frames or [])[:number]  # noqa: SLF001
            replay._frame_bodies.update(world._frame_bodies)  # noqa: SLF001
            fresh = replay.world_for(self_envelope=body(at), now=stamp + 0.05)
            assert snapshot.perceived is not None and fresh.perceived is not None
            self.assertEqual(snapshot.perceived.to_dict(), fresh.perceived.to_dict())


class TheRefreshSaysWhatTheCameraCostTests(unittest.TestCase):
    def test_the_cameras_share_of_the_build_is_said_and_a_frame_served_again_costs_none(self) -> None:
        class _Slow(_Camera):
            def grab_surface_depth(self) -> "DepthSnapshot | None":
                time.sleep(0.02)
                return super().grab_surface_depth()

        world = _world(_Slow(DepthSnapshot(depth_mm=_two_blocks(), intrinsics=_INTRINSICS, timestamp=100.0)))
        first = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.1)
        self.assertTrue(first.ok, first.render())
        self.assertGreaterEqual(first.grab_ms, 15.0)
        self.assertLess(first.grab_ms, first.build_ms)
        self.assertIn("ms of it reading the camera", first.render())
        self.assertEqual(first.to_dict()["grab_ms"], round(first.grab_ms, 3))
        again = refresh_planner_world(source=world, client=_Planner(), self_envelope=_SELF, now=100.2)
        self.assertEqual(again.grab_ms, 0.0)
        self.assertNotIn("reading the camera", again.render())


if __name__ == "__main__":
    unittest.main()
