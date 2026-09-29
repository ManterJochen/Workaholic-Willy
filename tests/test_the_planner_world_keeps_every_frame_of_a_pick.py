"""The planner world keeps every frame of a pick, each without the robot where it stood (plan step 3, 2026-09-29).

The owner's decision of 2026-09-28: a pick on the wrist D415 looks from every look its program declares, and from one
view it generates, and fuses what they saw. The grasp is ranked on the fused cloud, and the motions that carry it out
have to be planned against the same cell. Until now the planner world held one frame per camera, the one taken where
the arm stands, so everything the first look saw was gone from the world by the approach: a block beside the part that
only the first look saw was free space to the planner at the standoff.

So a pick holds its wrist frames (``LivePlannerWorld.hold_pick_views``). Each is placed by the tool pose it was stamped
with, and each loses the robot where the robot stood when it was taken as well as where it stands now. The first of
those two filters is the phantom the owner asked about: a frame from the first look that showed a link of the arm,
filtered only by the arm where it stands at the approach, keeps the link where it stood as an obstacle, and the planner
routes round a robot that is no longer there. The second is the start state: nothing else can be where the robot
stands now, and a planner whose start sits inside stale points has no plan.

The cell is the owner's (``test_the_camera_world_lets_the_owners_pick_finish``): the D415 60 mm beside the Hand-E and
130 mm behind its TCP, tilted 45 degrees outward, rendered through a pinhole with its field of view and cut at its
minimum range at 848 x 480. Two looks at a 40 mm cube from about 500 mm, the first from the +y side and the second from
the +x side, then the standoff 80 mm over the cube, where the camera looks outward at the bench beyond the workspace.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.robot.safety.planning import live_world
from src.robot.safety.planning.live_world import (
    CameraView,
    DepthSnapshot,
    LivePlannerWorld,
    PlannerWorldSnapshot,
    WorldVerdict,
    refresh_planner_world,
)
from src.robot.safety.planning.perceived import (
    DepthView,
    LinkCapsule,
    SelfBody,
    SelfEnvelope,
    WorldBuildTuning,
    build_perceived_boxes,
)
from tests.test_the_camera_world_lets_the_owners_pick_finish import (
    _BENCH,
    _CUBE,
    _GRASP,
    _K,
    _LIMITS,
    _STANDOFF,
    _covered,
    _look,
    _mount,
    _Planner,
    _render,
)

#: The D415's minimum range at 848 x 480, as the helper this module reuses states it.
_MIN_RANGE_MM = 310.0
#: The first look: the camera 500 mm from the cube, from the +y side, 45 degrees down.
_CAMERA_A = _look((0.0, -346.4, 353.6), (0.0, -1.0, -1.0))
#: The second look: the same from the +x side.
_CAMERA_B = _look((353.6, -700.0, 353.6), (-1.0, 0.0, -1.0))
_TOOL_A = _CAMERA_A @ np.linalg.inv(_mount())
_TOOL_B = _CAMERA_B @ np.linalg.inv(_mount())
_TOOL_STANDOFF = _STANDOFF.to_matrix()
_TOOL_GRASP = _GRASP.to_matrix()
#: Where the pick's motions go: the grasp centre.
_GOAL = tuple(float(v) for v in _GRASP.position_mm)
_CUBE_CENTRE = (0.0, -700.0, 20.0)
#: A block beside the part that only the first look sees: under the lower edge of the second look's view, and behind
#: the camera at the standoff.
_BESIDE = ((250.0, -740.0, 0.0), (300.0, -680.0, 80.0))
_BESIDE_CENTRE = (275.0, -710.0, 40.0)
#: The forearm where no camera of these poses sees it.
_ARM_AWAY = ((-600.0, 300.0, 600.0), (-450.0, 300.0, 600.0))
#: The forearm where the first look saw it: in the air beside the cube, 360 mm in front of the camera.
_ARM_AT_A = ((-200.0, -650.0, 150.0), (-60.0, -650.0, 150.0))
_ARM_AT_A_CENTRE = (-130.0, -650.0, 150.0)
#: Something on the bench the first look saw and that left while the arm moved on, and the forearm standing there now.
_GONE = ((-260.0, -620.0, 0.0), (-180.0, -560.0, 60.0))
_GONE_CENTRE = (-220.0, -590.0, 30.0)
_ARM_WHERE_IT_WAS = ((-220.0, -590.0, 10.0), (-220.0, -590.0, 50.0))
#: The capsule radius of the forearm before the world's 15 mm padding, and the half size of the box it is drawn as: the
#: box's corners lie 49.5 mm from its axis, inside the padded 55.
_FOREARM_RADIUS_MM = 40.0
_FOREARM_DRAWN_MM = 35.0
#: A goal the camera at the grasp looks at inside its minimum range: 200 mm along its axis, as in the helper module.
_CAMERA_GRASP = _TOOL_GRASP @ _mount()
_INSIDE_ITS_RANGE = tuple(float(v) for v in _CAMERA_GRASP[:3, 3] + 200.0 * _CAMERA_GRASP[:3, 2])


def _body(tool: np.ndarray, forearm: tuple = _ARM_AWAY) -> SelfEnvelope:
    """The robot at one pose: a forearm between two points in BASE, and the hand behind the TCP on the tool frame."""
    start, end = (np.asarray(point, dtype=np.float64) for point in forearm)
    link = np.eye(4)
    link[:3, 3] = start
    along = end - start
    return SelfEnvelope(
        frames_mm=(np.eye(4), link, np.asarray(tool, dtype=np.float64)),
        capsules=(
            LinkCapsule(frame=1, start_mm=(0.0, 0.0, 0.0), end_mm=(float(along[0]), float(along[1]), float(along[2])),
                        radius_mm=_FOREARM_RADIUS_MM),
            LinkCapsule(frame=2, start_mm=(0.0, 0.0, -200.0), end_mm=(0.0, 0.0, -40.0), radius_mm=45.0),
        ),
    )


def _drawn(forearm: tuple) -> tuple:
    """The box a camera sees of a forearm along base x: its length, and its drawn half size across it."""
    (x0, y, z), (x1, _, _) = forearm
    half = _FOREARM_DRAWN_MM
    return (min(x0, x1), y - half, z - half), (max(x0, x1), y + half, z + half)


def _self_body(envelope: SelfEnvelope) -> SelfBody:
    """The body as the world pads it (``WorldBuildTuning.margin_mm``, 15 mm)."""
    return SelfBody.from_frames(envelope.frames_mm, envelope.capsules, padding_mm=15.0)


class _Camera:
    """A D415 on the wrist of an arm the test stands at one pose after another, or fixed where ``fixed`` places it.

    Every grab renders the cell as it is now from where the camera stands: ``scene`` on the bench, and what it sees of
    the robot (``robot``), which the test sets for the pose. ``blank`` is a window of the image that holds no depth, a
    surface the camera cannot read.

    What a producer may get wrong is set here too. ``stamps_tool`` stamps the tool pose on a fixed camera's frames as
    well; ``no_tool_pose`` stamps none on a wrist frame; ``stamped_tool`` stamps that pose instead of where the tool
    stands, and ``placement_error_mm`` declares that error; ``reuses_buffers`` hands out the same depth and tool-pose
    arrays on every grab, written over in place.
    """

    def __init__(
        self, *scene: tuple, fixed: np.ndarray | None = None, stamps_tool: bool = False, reuses_buffers: bool = False,
    ) -> None:
        self.scene = list(scene)
        self.fixed = fixed
        self.tool = _TOOL_A
        self.robot: tuple = ()
        self.stamp = 100.0
        self.blank: tuple[slice, slice] | None = None
        self.grabs = 0
        self.stamps_tool = stamps_tool
        self.no_tool_pose = False
        self.stamped_tool: np.ndarray | None = None
        self.placement_error_mm = 0.0
        self.reuses_buffers = reuses_buffers
        self._buffers: tuple[np.ndarray, np.ndarray] | None = None

    def grab_surface_depth(self) -> DepthSnapshot:
        self.grabs += 1
        camera_to_base = self.fixed if self.fixed is not None else self.tool @ _mount()
        depth = _render(camera_to_base, _MIN_RANGE_MM, *self.scene, *self.robot)
        if self.blank is not None:
            depth[self.blank] = 0.0
        tool: np.ndarray | None = None
        if (self.fixed is None or self.stamps_tool) and not self.no_tool_pose:
            tool = (self.tool if self.stamped_tool is None else self.stamped_tool).copy()
        if self.reuses_buffers and tool is not None:
            if self._buffers is None:
                self._buffers = (depth, tool)
            else:
                self._buffers[0][...] = depth
                self._buffers[1][...] = tool
            depth, tool = self._buffers
        return DepthSnapshot(
            depth_mm=depth, intrinsics=_K, timestamp=self.stamp, tool_to_base_mm=tool,
            placement_error_mm=self.placement_error_mm,
        )


def _world(*scene: tuple) -> tuple[LivePlannerWorld, _Camera]:
    camera = _Camera(*(scene or (_CUBE, _BESIDE)))
    world = LivePlannerWorld(
        cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_mount()),),
        declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
    )
    return world, camera


def _ask(
    world: LivePlannerWorld, camera: _Camera, tool: np.ndarray, *, stamp: float, forearm: tuple = _ARM_AWAY,
    robot: tuple = (), goal: tuple | None = _GOAL,
) -> PlannerWorldSnapshot:
    """Stand the arm at ``tool``, with its forearm at ``forearm``, and ask the world about a motion to ``goal``."""
    camera.tool, camera.robot, camera.stamp = tool, robot, stamp
    return world.world_for(self_envelope=_body(tool, forearm), near_point_mm=goal, now=stamp + 0.05)


def _look_twice_and_stand_over_the_part(
    world: LivePlannerWorld, camera: _Camera,
) -> tuple[PlannerWorldSnapshot, PlannerWorldSnapshot, PlannerWorldSnapshot]:
    """The pick's poses: the first look, the second, and the standoff, one second apart."""
    first = _ask(world, camera, _TOOL_A, stamp=100.0)
    second = _ask(world, camera, _TOOL_B, stamp=101.0)
    standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)
    return first, second, standoff


def _boxes(snapshot: PlannerWorldSnapshot) -> tuple:
    assert snapshot.perceived is not None, snapshot.render()
    return snapshot.perceived.boxes


def _cube_top_points() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(-20.0, 20.1, 5.0), np.arange(-720.0, -679.9, 5.0))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 40.0)])


# ---------------------------------------------------------------------------------------------------
# What one look saw stays until the pick ends
# ---------------------------------------------------------------------------------------------------


class WhatOneLookSawStaysTests(unittest.TestCase):
    """The block beside the part is in the first look's frame and in no frame taken after it."""

    def test_an_obstacle_seen_only_from_the_first_look_stays_at_the_approach(self) -> None:
        world, camera = _world()
        world.hold_pick_views()

        _, _, standoff = _look_twice_and_stand_over_the_part(world, camera)

        self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
        self.assertTrue(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())
        self.assertEqual(2, standoff.held, "the two looks, each placed where it was taken")
        self.assertEqual(3, world.held_view_count, "the looks and the standoff, one frame per pose")

    def test_without_a_hold_the_approach_plans_against_the_frame_it_stands_in(self) -> None:
        """The control, green before and after: the scene is what the test says it is."""
        world, camera = _world()

        first, second, standoff = _look_twice_and_stand_over_the_part(world, camera)

        self.assertTrue(_covered(_boxes(first), _BESIDE_CENTRE), first.render())
        self.assertFalse(_covered(_boxes(second), _BESIDE_CENTRE), "the second look sees the block")
        self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
        self.assertFalse(_covered(_boxes(standoff), _BESIDE_CENTRE), "the standoff sees the block")
        self.assertEqual(0, standoff.held)
        self.assertEqual(0, world.held_view_count)


# ---------------------------------------------------------------------------------------------------
# Each frame loses the robot where it stood when it was taken, and where it stands now
# ---------------------------------------------------------------------------------------------------


class NoPhantomOfTheRobotTests(unittest.TestCase):
    """A held frame shows the robot where it stood; where it stood is not an obstacle at any later pose."""

    def test_the_arm_in_an_old_frame_leaves_no_phantom_where_it_stood(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        first = _ask(world, camera, _TOOL_A, stamp=100.0, forearm=_ARM_AT_A, robot=(_drawn(_ARM_AT_A),))
        self.assertFalse(_covered(_boxes(first), _ARM_AT_A_CENTRE), "where it stands now, the arm is the robot")

        _ask(world, camera, _TOOL_B, stamp=101.0)
        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)

        self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
        self.assertFalse(_covered(_boxes(standoff), _ARM_AT_A_CENTRE),
                         "the first look's frame keeps the forearm where it stood as an obstacle")
        self.assertTrue(_covered(_boxes(standoff), _BESIDE_CENTRE), "and the frame is in the world")

    def test_ignoring_the_capture_pose_would_leave_the_phantom(self) -> None:
        """The control: the first look's frame, filtered by the robot at the standoff alone, keeps the forearm."""
        camera = _Camera(_CUBE, _BESIDE)
        camera.tool, camera.robot = _TOOL_A, (_drawn(_ARM_AT_A),)
        frame = camera.grab_surface_depth()

        def built(then: SelfBody | None):  # noqa: ANN202
            return build_perceived_boxes(
                views=[DepthView(surface_depth_mm=frame.depth_mm, intrinsics=_K, camera_to_base=_CAMERA_A,
                                 name="wrist (held 1)", self_body=then)],
                limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
                self_body=_self_body(_body(_TOOL_STANDOFF)),
            )

        ignored = built(None)
        filtered = built(_self_body(_body(_TOOL_A, _ARM_AT_A)))

        self.assertTrue(_covered(ignored.boxes, _ARM_AT_A_CENTRE), ignored.render())
        self.assertFalse(_covered(filtered.boxes, _ARM_AT_A_CENTRE), filtered.render())
        self.assertGreater(filtered.dropped_points["self"], ignored.dropped_points["self"])

    def test_a_views_own_body_takes_nothing_out_of_another_view(self) -> None:
        """Where the robot stood for one frame says nothing about what another frame saw there."""
        camera = _Camera(_CUBE, _BESIDE)
        camera.tool, camera.robot = _TOOL_A, (_drawn(_ARM_AT_A),)
        depth = camera.grab_surface_depth().depth_mm
        then = _self_body(_body(_TOOL_A, _ARM_AT_A))

        world = build_perceived_boxes(
            views=[
                DepthView(surface_depth_mm=depth, intrinsics=_K, camera_to_base=_CAMERA_A, name="then", self_body=then),
                DepthView(surface_depth_mm=depth.copy(), intrinsics=_K, camera_to_base=_CAMERA_A, name="another"),
            ],
            limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8), self_body=_self_body(_body(_TOOL_STANDOFF)),
        )

        self.assertTrue(_covered(world.boxes, _ARM_AT_A_CENTRE), world.render())

    def test_the_robot_where_it_stands_now_is_taken_out_of_every_held_frame(self) -> None:
        """What the first look saw where the forearm stands now is gone: nothing else can be there."""
        for forearm, stays in ((_ARM_WHERE_IT_WAS, False), (_ARM_AWAY, True)):
            with self.subTest(forearm=forearm):
                world, camera = _world(_CUBE, _BESIDE, _GONE)
                world.hold_pick_views()
                first = _ask(world, camera, _TOOL_A, stamp=100.0)
                self.assertTrue(_covered(_boxes(first), _GONE_CENTRE), first.render())
                camera.scene.remove(_GONE)

                _ask(world, camera, _TOOL_B, stamp=101.0)
                standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0, forearm=forearm)

                self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
                self.assertIs(stays, bool(_covered(_boxes(standoff), _GONE_CENTRE)), standoff.render())
                self.assertTrue(_covered(_boxes(standoff), _BESIDE_CENTRE))


# ---------------------------------------------------------------------------------------------------
# A goal any frame of the pick measured has been seen
# ---------------------------------------------------------------------------------------------------


class AGoalSeenByAnyFrameOfThePickTests(unittest.TestCase):
    """At the grasp the camera stands inside its minimum range of the region below it; the first look measured it."""

    def test_a_goal_any_frame_of_the_pick_measured_is_not_unseen(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)

        snapshot = _ask(world, camera, _TOOL_GRASP, stamp=101.0, goal=_INSIDE_ITS_RANGE)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        self.assertEqual(("wrist (held 1)",), snapshot.goal_seen_by)

    def test_without_a_hold_the_same_goal_is_unseen(self) -> None:
        """The control, green before and after."""
        world, camera = _world()
        _ask(world, camera, _TOOL_A, stamp=100.0)

        snapshot = _ask(world, camera, _TOOL_GRASP, stamp=101.0, goal=_INSIDE_ITS_RANGE)

        self.assertIs(WorldVerdict.UNSEEN, snapshot.verdict, snapshot.render())

    def test_a_goal_no_frame_measured_is_still_refused(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        inverse = np.linalg.inv(_CAMERA_A)
        x, y, z = inverse[:3, :3] @ np.asarray(_INSIDE_ITS_RANGE) + inverse[:3, 3]
        u, v = int(_K[0, 0] * x / z + _K[0, 2]), int(_K[1, 1] * y / z + _K[1, 2])
        camera.blank = (slice(max(0, v - 45), v + 45), slice(max(0, u - 45), u + 45))
        _ask(world, camera, _TOOL_A, stamp=100.0, goal=None)
        camera.blank = None

        snapshot = _ask(world, camera, _TOOL_GRASP, stamp=101.0, goal=_INSIDE_ITS_RANGE)

        self.assertIs(WorldVerdict.UNSEEN, snapshot.verdict, snapshot.render())
        self.assertIn("'wrist' ", snapshot.reason, "the camera where the arm stands looks at it too")
        self.assertIn("'wrist (held 1)' 0%", snapshot.reason)

    def test_a_refusal_only_a_held_frame_looked_at_names_the_camera_to_go_to(self) -> None:
        """At the standoff the camera looks away from the grasp, so only the first look's frame is asked about it.

        The refusal names ``wrist``, the camera an operator can go to; the reason names the held frame whose share it
        was.
        """
        world, camera = _world()
        world.hold_pick_views()
        inverse = np.linalg.inv(_CAMERA_A)
        x, y, z = inverse[:3, :3] @ np.asarray(_GOAL) + inverse[:3, 3]
        u, v = int(_K[0, 0] * x / z + _K[0, 2]), int(_K[1, 1] * y / z + _K[1, 2])
        camera.blank = (slice(max(0, v - 50), v + 50), slice(max(0, u - 50), u + 50))
        _ask(world, camera, _TOOL_A, stamp=100.0, goal=None)
        camera.blank = None

        snapshot = _ask(world, camera, _TOOL_STANDOFF, stamp=101.0)

        self.assertIs(WorldVerdict.UNSEEN, snapshot.verdict, snapshot.render())
        self.assertIn("'wrist (held 1)' 0%", snapshot.reason)
        self.assertNotIn("'wrist' ", snapshot.reason, "the camera where the arm stands does not look at the grasp")
        self.assertEqual("wrist", snapshot.camera)


# ---------------------------------------------------------------------------------------------------
# How long a frame is held, and which
# ---------------------------------------------------------------------------------------------------


class WhatIsHeldAndForHowLongTests(unittest.TestCase):
    def test_held_frames_do_not_age_and_the_current_frame_must_still_be_fresh(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)

        later = _ask(world, camera, _TOOL_B, stamp=130.0)

        self.assertIs(WorldVerdict.FRESH, later.verdict, later.render())
        self.assertEqual(1, later.held)
        self.assertAlmostEqual(30050.0, later.held_oldest_age_ms or 0.0, delta=1.0)
        self.assertAlmostEqual(50.0, later.age_ms or 0.0, delta=1.0, msg="the age is the current frame's")
        self.assertTrue(_covered(_boxes(later), _BESIDE_CENTRE), later.render())

        camera.tool = _TOOL_STANDOFF
        stale = world.world_for(self_envelope=_body(_TOOL_STANDOFF), near_point_mm=_GOAL, now=131.0)

        self.assertIs(WorldVerdict.STALE, stale.verdict, stale.render())
        self.assertEqual(2, world.held_view_count, "a stale frame is not held")

    def test_frames_survive_forget_segmentation_and_go_when_the_pick_ends(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        _ask(world, camera, _TOOL_B, stamp=101.0)
        world.offer_segmentation(target_points_base_mm=_cube_top_points(), timestamp=101.0, hold=True)

        world.forget_segmentation()
        world.drop_cached_frames()
        world.drop_cached_frame("wrist")

        self.assertEqual(2, world.held_view_count, "every keep-out scope of a pick closing is not the pick ending")
        kept = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)
        self.assertTrue(_covered(_boxes(kept), _BESIDE_CENTRE), kept.render())

        world.forget_pick_views()

        self.assertEqual(0, world.held_view_count)
        gone = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)
        self.assertEqual(0, gone.held)
        self.assertFalse(_covered(_boxes(gone), _BESIDE_CENTRE), gone.render())

    def test_a_new_hold_starts_the_pick_afresh(self) -> None:
        """A hold does not nest: a second start drops what the first held, which belongs to a pick that ended."""
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        _ask(world, camera, _TOOL_B, stamp=101.0)

        world.hold_pick_views()

        self.assertEqual(0, world.held_view_count)
        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)
        self.assertFalse(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())
        self.assertEqual(1, world.held_view_count)

    def test_a_frame_cached_before_the_hold_is_held_when_it_is_served_again(self) -> None:
        """A pick that starts where a question was just asked keeps that pose's frame, and asks the camera no more."""
        world, camera = _world()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        world.hold_pick_views()

        _ask(world, camera, _TOOL_A, stamp=100.0)

        self.assertEqual(1, camera.grabs)
        self.assertEqual(1, world.held_view_count)
        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=101.0)
        self.assertTrue(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())

    def test_a_frame_taken_where_an_earlier_one_was_replaces_it(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        _ask(world, camera, _TOOL_A, stamp=100.0)
        self.assertEqual(1, world.held_view_count, "two questions at one pose share one frame")
        _ask(world, camera, _TOOL_B, stamp=101.0)
        camera.scene.remove(_BESIDE)

        back = _ask(world, camera, _TOOL_A, stamp=102.0)

        self.assertEqual(2, world.held_view_count, "the first look's frame was replaced, not kept beside the new one")
        self.assertFalse(_covered(_boxes(back), _BESIDE_CENTRE), back.render())
        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=103.0)
        self.assertEqual(2, standoff.held)
        self.assertFalse(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())

    def test_a_fixed_camera_holds_no_old_frames(self) -> None:
        overhead = _look((0.0, -650.0, 700.0), (0.0, 0.0, -1.0))
        camera = _Camera(_CUBE, _BESIDE, fixed=overhead)
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=overhead),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
        )
        world.hold_pick_views()
        first = _ask(world, camera, _TOOL_A, stamp=100.0)
        self.assertTrue(_covered(_boxes(first), _BESIDE_CENTRE), first.render())
        camera.scene.remove(_BESIDE)

        second = _ask(world, camera, _TOOL_B, stamp=101.0)

        self.assertEqual(2, camera.grabs)
        self.assertEqual(0, world.held_view_count)
        self.assertEqual(0, second.held)
        self.assertFalse(_covered(_boxes(second), _BESIDE_CENTRE), "the newest frame of a fixed camera supersedes it")
        assert second.perceived is not None
        self.assertEqual(["overhead"], list(second.perceived.depth_coverage))

    def test_a_frame_the_robot_cannot_be_taken_out_of_is_not_held(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        camera.tool = _TOOL_A

        snapshot = world.world_for(self_envelope=None, near_point_mm=_GOAL, now=100.05)

        self.assertIs(WorldVerdict.UNUSABLE, snapshot.verdict)
        self.assertEqual(0, world.held_view_count)

    def test_one_shutter_is_held_once_however_the_arm_creeps_about_it(self) -> None:
        """A frame cached before the hold and served again twice, the arm 0.9 mm to one side and then to the other.

        Each question stands within a millimetre of where the frame was taken, so the frame is served again both times;
        the two bodies are 1.8 mm apart. The frame is the same shutter both times and is held once, and a world is
        never built from it twice, once as the frame taken now and once as a held one: its points would count double
        toward a cluster's points and a held frame would be named that is not one.
        """
        world, camera = _world()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        world.hold_pick_views()

        for creep_mm in (0.9, -0.9):
            tool = _TOOL_A.copy()
            tool[0, 3] += creep_mm
            snapshot = _ask(world, camera, tool, stamp=100.0)
            with self.subTest(creep_mm=creep_mm):
                self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
                self.assertEqual((), snapshot.held_views, "the frame served now is not a held frame beside itself")

        self.assertEqual(1, camera.grabs)
        self.assertEqual(1, world.held_view_count)

    def test_a_frame_whose_world_cannot_be_built_is_not_held(self) -> None:
        """Held, a frame that makes its own world unusable would make every world of the pick unusable, at every pose."""
        cases = {"a tool pose that is not a pose": ("stamped_tool", np.full((4, 4), np.nan)),
                 "a placement error that is not a distance": ("placement_error_mm", float("nan"))}
        for case, (attribute, poison) in cases.items():
            with self.subTest(case):
                world, camera = _world()
                world.hold_pick_views()
                clean = getattr(camera, attribute)
                setattr(camera, attribute, poison)

                refused = _ask(world, camera, _TOOL_A, stamp=100.0)

                self.assertIs(WorldVerdict.UNUSABLE, refused.verdict, refused.render())
                self.assertEqual(0, world.held_view_count)
                setattr(camera, attribute, clean)
                second = _ask(world, camera, _TOOL_B, stamp=101.0)
                self.assertIs(WorldVerdict.FRESH, second.verdict, second.render())
                self.assertEqual(0, second.held)
                self.assertEqual(1, world.held_view_count)

    def test_a_wrist_frame_without_a_tool_pose_is_not_held_and_the_next_pose_is_fresh(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        camera.no_tool_pose = True

        refused = _ask(world, camera, _TOOL_A, stamp=100.0)

        self.assertIs(WorldVerdict.UNUSABLE, refused.verdict, refused.render())
        self.assertIn("carries no tool pose", refused.reason)
        self.assertEqual(0, world.held_view_count)
        camera.no_tool_pose = False
        second = _ask(world, camera, _TOOL_B, stamp=101.0)
        self.assertIs(WorldVerdict.FRESH, second.verdict, second.render())
        self.assertEqual(0, second.held)
        self.assertEqual(1, world.held_view_count)

    def test_a_fixed_camera_that_stamps_a_tool_pose_holds_nothing_either(self) -> None:
        """Whether a camera is on the wrist is the cell's to say, not the frame's: a fixed camera's frame never moves."""
        overhead = _look((0.0, -650.0, 700.0), (0.0, 0.0, -1.0))
        camera = _Camera(_CUBE, _BESIDE, fixed=overhead, stamps_tool=True)
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=camera, camera_to_base=overhead),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
        )
        world.hold_pick_views()

        _ask(world, camera, _TOOL_A, stamp=100.0)
        second = _ask(world, camera, _TOOL_B, stamp=101.0)

        self.assertIs(WorldVerdict.FRESH, second.verdict, second.render())
        self.assertEqual(0, world.held_view_count)
        self.assertEqual(0, second.held)

    def test_a_producer_that_writes_into_the_same_buffers_changes_nothing_held(self) -> None:
        """The first look's frame is held as it was taken, although its producer wrote the standoff's into its arrays."""
        camera = _Camera(_CUBE, _BESIDE, reuses_buffers=True)
        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_mount()),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
        )
        world.hold_pick_views()
        first = _ask(world, camera, _TOOL_A, stamp=100.0)
        self.assertTrue(_covered(_boxes(first), _BESIDE_CENTRE), first.render())

        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=101.0)

        self.assertEqual(2, camera.grabs)
        self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
        self.assertEqual(1, standoff.held)
        self.assertTrue(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())

    def test_a_pick_holds_no_more_frames_than_its_cap_and_says_so(self) -> None:
        """Past the cap a new pose is not held, and nothing held is let go: dropping a frame would drop what it saw.

        The cap is two here, so the standoff is the pose past it. A frame taken where a held one was still replaces it,
        which holds no more than before.
        """
        world, camera = _world()
        world.hold_pick_views()
        with mock.patch.object(live_world, "_MAX_HELD_FRAMES", 2):
            _ask(world, camera, _TOOL_A, stamp=100.0)
            _ask(world, camera, _TOOL_B, stamp=101.0)

            standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)

            self.assertIs(WorldVerdict.FRESH, standoff.verdict, standoff.render())
            self.assertEqual((2, 1), (standoff.held, standoff.held_not_kept),
                             "the standoff's own frame is in its world, and goes with the arm")
            self.assertEqual(2, world.held_view_count)
            grasp = _ask(world, camera, _TOOL_GRASP, stamp=103.0)
            self.assertEqual((2, 2), (grasp.held, grasp.held_not_kept), "neither the standoff's nor the grasp's is held")
            self.assertTrue(_covered(_boxes(grasp), _BESIDE_CENTRE), grasp.render())
            self.assertIn("2 more NOT kept", grasp.render())
            self.assertEqual(2, grasp.to_dict()["held_not_kept"])
            camera.tool, camera.stamp = _TOOL_A, 104.0
            refresh = refresh_planner_world(
                source=world, client=_Planner(), self_envelope=_body(_TOOL_A), near_point_mm=_GOAL, now=104.05,
            )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual((1, 2), (refresh.held_frames, refresh.held_not_kept), "back at the first look: replaced")
        self.assertIn("2 more NOT kept", refresh.render())
        self.assertEqual(2, refresh.to_dict()["held_not_kept"])
        self.assertEqual(2, world.held_view_count)
        world.hold_pick_views()
        self.assertEqual(0, _ask(world, camera, _TOOL_B, stamp=105.0).held_not_kept, "a new pick starts afresh")

    def test_a_held_frame_keeps_its_depth_in_single_precision(self) -> None:
        """About 3.7 MB a frame at 1280 x 720 rather than 7.4, for a rounding of 0.03 micrometre at a metre."""
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)

        held = world._pick_frames  # noqa: SLF001
        assert held is not None and len(held) == 1

        self.assertEqual(np.float32, held[0].frame.depth_mm.dtype)


# ---------------------------------------------------------------------------------------------------
# The target a pick holds out leaves every frame it holds
# ---------------------------------------------------------------------------------------------------


class TheHeldTargetTests(unittest.TestCase):
    def test_the_held_target_leaves_every_held_frame(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _, _, standoff = _look_twice_and_stand_over_the_part(world, camera)
        self.assertTrue(_covered(_boxes(standoff), _CUBE_CENTRE), "both looks saw the cube")

        world.offer_segmentation(target_points_base_mm=_cube_top_points(), target_label="cube", timestamp=101.0,
                                 hold=True)
        snapshot = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)

        self.assertIs(WorldVerdict.FRESH, snapshot.verdict, snapshot.render())
        self.assertFalse(_covered(_boxes(snapshot), _CUBE_CENTRE), snapshot.render())
        self.assertTrue(_covered(_boxes(snapshot), _BESIDE_CENTRE))
        assert snapshot.keep_out is not None
        ((_, _, points),) = snapshot.keep_out.held
        self.assertGreater(points, 0)


# ---------------------------------------------------------------------------------------------------
# No hold, no change
# ---------------------------------------------------------------------------------------------------


class NoHoldNoChangeTests(unittest.TestCase):
    def test_no_hold_is_byte_identical(self) -> None:
        """A world that never held and one whose hold ended answer alike; the existing live-world tests pass unchanged."""
        never, never_camera = _world()
        ended, ended_camera = _world()
        ended.hold_pick_views()
        for tool, stamp in ((_TOOL_A, 100.0), (_TOOL_B, 101.0)):
            _ask(never, never_camera, tool, stamp=stamp)
            _ask(ended, ended_camera, tool, stamp=stamp)
        ended.forget_pick_views()

        a = _ask(never, never_camera, _TOOL_STANDOFF, stamp=102.0)
        b = _ask(ended, ended_camera, _TOOL_STANDOFF, stamp=102.0)

        self.assertEqual(a.cuboids, b.cuboids)
        assert a.perceived is not None and b.perceived is not None
        self.assertEqual(a.perceived.to_dict(), b.perceived.to_dict())
        self.assertEqual((a.goal_seen_by, a.age_ms, a.captured_at_s), (b.goal_seen_by, b.age_ms, b.captured_at_s))
        self.assertEqual((0, None), (b.held, b.held_oldest_age_ms))


# ---------------------------------------------------------------------------------------------------
# What a refresh says of the frames it kept
# ---------------------------------------------------------------------------------------------------


class WhatTheRefreshSaysTests(unittest.TestCase):
    def test_the_refresh_says_how_many_frames_of_the_pick_it_kept(self) -> None:
        world, camera = _world()
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        _ask(world, camera, _TOOL_B, stamp=101.0)
        camera.tool, camera.stamp = _TOOL_STANDOFF, 102.0

        refresh = refresh_planner_world(
            source=world, client=_Planner(), self_envelope=_body(_TOOL_STANDOFF), near_point_mm=_GOAL, now=102.05,
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual(2, refresh.held_frames)
        self.assertAlmostEqual(2050.0, refresh.held_oldest_age_ms or 0.0, delta=1.0)
        self.assertIn("2 frame(s) of earlier poses of this pick", refresh.render())
        data = refresh.to_dict()
        self.assertEqual(2, data["held_frames"])
        self.assertAlmostEqual(2050.0, data["held_oldest_age_ms"], delta=1.0)
        # The stamp stays about the frames taken now.
        self.assertEqual(["wrist"], [name for name, _ in refresh.depth_coverage])
        stamp = refresh.camera_world()
        assert stamp is not None
        self.assertEqual(102.0, stamp.captured_at_s)
        self.assertEqual(("wrist (held 1)", "wrist (held 2)"), refresh.goal_seen_by)

        snapshot = world.world_for(self_envelope=_body(_TOOL_STANDOFF), near_point_mm=_GOAL, now=102.05)
        self.assertIn("2 frame(s) of earlier poses of this pick", snapshot.render())
        self.assertEqual(2, snapshot.to_dict()["held"])

    def test_a_refresh_refused_for_its_slots_says_the_held_frames_were_counted(self) -> None:
        """The cube and the block need two slots and the cell has one: part of the world came from earlier poses."""
        camera = _Camera(_CUBE, _BESIDE)
        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=camera, camera_to_tool=_mount()),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=1),
        )
        world.hold_pick_views()
        _ask(world, camera, _TOOL_A, stamp=100.0)
        _ask(world, camera, _TOOL_B, stamp=101.0)
        camera.tool, camera.stamp = _TOOL_STANDOFF, 102.0

        refresh = refresh_planner_world(
            source=world, client=_Planner(), self_envelope=_body(_TOOL_STANDOFF), near_point_mm=_GOAL, now=102.05,
        )

        self.assertFalse(refresh.ok, refresh.render())
        self.assertGreater(refresh.dropped_obstacles, 0)
        self.assertIn("2 frame(s) of earlier poses of this pick", refresh.reason)
        self.assertIn("2 frame(s) of earlier poses of this pick", refresh.render())

    def test_a_refresh_with_nothing_held_says_nothing_of_it(self) -> None:
        world, camera = _world()
        _ask(world, camera, _TOOL_A, stamp=100.0)

        refresh = refresh_planner_world(
            source=world, client=_Planner(), self_envelope=_body(_TOOL_A), near_point_mm=_GOAL, now=100.05,
        )

        self.assertTrue(refresh.ok, refresh.render())
        self.assertEqual((0, None), (refresh.held_frames, refresh.held_oldest_age_ms))
        self.assertNotIn("earlier poses", refresh.render())
        self.assertEqual((0, None), (refresh.to_dict()["held_frames"], refresh.to_dict()["held_oldest_age_ms"]))


# ---------------------------------------------------------------------------------------------------
# The scope an arm's caller holds a pick's frames with
# ---------------------------------------------------------------------------------------------------


class HoldingViewsTests(unittest.TestCase):
    def test_a_block_holds_every_wrist_frame_and_lets_them_go_after(self) -> None:
        from src.robot.core.keep_out import holding_views

        world, camera = _world()
        arm = SimpleNamespace(live_planner_world=world)

        with holding_views(arm) as holding:
            self.assertTrue(holding)
            _ask(world, camera, _TOOL_A, stamp=100.0)
            _ask(world, camera, _TOOL_B, stamp=101.0)
            self.assertEqual(2, world.held_view_count)

        self.assertEqual(0, world.held_view_count)
        standoff = _ask(world, camera, _TOOL_STANDOFF, stamp=102.0)
        self.assertFalse(_covered(_boxes(standoff), _BESIDE_CENTRE), standoff.render())

        with self.assertRaises(RuntimeError), holding_views(arm):
            _ask(world, camera, _TOOL_A, stamp=103.0)
            raise RuntimeError("the pick raised")
        self.assertEqual(0, world.held_view_count, "a block that raised let its frames go")

    def test_an_arm_whose_world_cannot_hold_opens_an_empty_scope(self) -> None:
        from src.robot.core.keep_out import holding_views

        for arm in (SimpleNamespace(), SimpleNamespace(live_planner_world=None),
                    SimpleNamespace(live_planner_world=object())):
            with self.subTest(arm=arm), holding_views(arm) as holding:
                self.assertFalse(holding)

    def test_a_world_with_no_camera_on_the_wrist_holds_nothing_and_says_so(self) -> None:
        from src.robot.core.keep_out import holding_views

        overhead = _look((0.0, -650.0, 700.0), (0.0, 0.0, -1.0))
        world = LivePlannerWorld(
            cameras=(CameraView(name="overhead", depth_source=_Camera(fixed=overhead), camera_to_base=overhead),),
            declared=_BENCH, limits=_LIMITS, tuning=WorldBuildTuning(max_boxes=8),
        )

        with holding_views(SimpleNamespace(live_planner_world=world)) as holding:
            self.assertFalse(holding)
        self.assertIs(False, world.hold_pick_views())
        self.assertIs(True, _world()[0].hold_pick_views())


if __name__ == "__main__":
    unittest.main()
