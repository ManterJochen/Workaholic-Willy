"""What the robot's own body hides from a camera is not free space: it stands as high as what was seen beside it.

The review of the height map (2026-09-30) drew the robot into the depth frames, which no test had: every frame before
was a camera that saw through the arm. A camera over the owner's bin beside the UR10's base cannot see the stretch of
rim under the shoulder housing, 52 mm over the base plate, because the housing is in the way. One box per cluster
covered that stretch, as it covered everything between what the camera saw of an object; a height map of what the
camera saw left it free, so a 47 mm rim 6 mm under the housing was accepted, and a joint-1 move drove the housing into a
post standing in the bin under it. The Hand-E's padded spheres take up to 27 mm of what a camera sees about the hand for
the hand, and a wall 6 mm from its fingers came back as boxes that stopped short of them.

Now, where the robot hides part of an object from every camera that could show it free (``height_map``):
  * a stretch inside what was seen of it stands as high as the cells seen beside it, from the floor;
  * a row of it that runs into where the self filter took the rest for the robot runs on, no higher than itself;
  * the robot's shadow between two objects it cut apart is bridged, no further than the two reach;
and a refusal says the robot hid part of the box. What the scene hides from itself, the floor behind a wall, stays free,
and so does a cell another camera saw free. The owner's bins that cleared the housing still pass with the arm in view.

Every frame here is drawn with the robot's committed meshes in it (``tests/_seen_scenes.render(robot=...)``), and the
guard is the exact one on the same meshes. They skip where no collision engine loads. Names new with this change are
imported inside the tests, so this file loads against the tree before it.
"""

from __future__ import annotations

import math
import re
import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.config.schema.robot.safety_schema import FixtureBoxConfig, SelfCollisionSafetyConfig
from src.robot.core import JointPositions, MotionCommand
from src.robot.safety.guard import SafetyContext
from src.robot.safety.path_samples import joint_path_samples
from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld, refresh_planner_world
from src.robot.safety.planning.perceived import SelfEnvelope, WorldBuildLimits, WorldBuildTuning
from src.robot.safety.planning.self_envelope import arm_capsules, hand_spheres, yawed_link_transforms_mm
from src.robot.safety.preflight import SafetyPreflight
from src.robot.safety.self_collision import SelfCollisionGuard
from tests._seen_scenes import (
    K,
    LIMITS,
    Solid,
    covered_by,
    looking_at,
    looking_down,
    open_bin,
    render,
    robot_depth,
    robot_triangles,
)
from tests.test_the_guard_holds_the_boxes_the_planner_holds import (
    _ARM,
    _Q_DEG,
    _SUPPORT,
    _guard,
    _needs_the_engine,
    _Recording,
    _ruler,
)

#: The investigation's cameras over the housing's ring, each looking at the bin beside the base (2026-09-30 review).
_CAMERAS = {
    "straight down over the bin": looking_down(0.0, -300.0),
    "straight down over the housing, 700 mm up": looking_down(0.0, -180.0, 700.0),
    "straight down 100 mm aside the housing": looking_down(100.0, -180.0),
    "from +x at 45 degrees": looking_at((500.0, -250.0, 550.0), (0.0, -250.0, 0.0)),
    "from -y at 45 degrees": looking_at((0.0, -800.0, 500.0), (0.0, -250.0, 0.0)),
    "from over the base at 60 degrees": looking_at((0.0, 300.0, 700.0), (0.0, -250.0, 0.0)),
}
#: Wide enough for a camera placed a metre from the base to see the bench it looks at.
_WIDE = WorldBuildLimits(x_mm=(-1500.0, 1500.0), y_mm=(-1500.0, 1500.0), z_mm=(-50.0, 1300.0), support_plane_top_mm=0.0)


def _envelope(joints_deg: "tuple[float, ...]" = _Q_DEG, extra: tuple = ()) -> SelfEnvelope:
    frames = yawed_link_transforms_mm("ur10", np.radians(joints_deg), 0.0)
    capsules = arm_capsules("ur10")
    assert frames is not None and capsules is not None
    return SelfEnvelope(frames_mm=tuple(frames), capsules=capsules + tuple(extra))


def _refresh(views: "list[tuple[np.ndarray, np.ndarray]]", *, envelope: "SelfEnvelope | None" = None,
             near_point_mm: "tuple[float, float, float]" = (600.0, 0.0, 300.0), limits: WorldBuildLimits = LIMITS,
             tuning: "WorldBuildTuning | None" = None) -> Any:
    """The live world fixed cameras build of what each ``(camera_to_base, depth)`` saw, refreshed into a recording
    planner."""

    def camera(depth: np.ndarray) -> Any:
        class _Camera:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=100.0)

        return _Camera()

    world = LivePlannerWorld(
        cameras=tuple(CameraView(name=f"camera {index}", depth_source=camera(depth), camera_to_base=placed)
                      for index, (placed, depth) in enumerate(views)),
        declared=_SUPPORT, limits=limits, tuning=tuning or WorldBuildTuning(pixel_stride=2), max_age_ms=5000.0,
    )
    return refresh_planner_world(source=world, client=_Recording(), self_envelope=envelope or _envelope(),
                                 near_point_mm=near_point_mm, now=100.1)


def _ctx(joints_deg: "tuple[float, ...]" = _Q_DEG) -> SafetyContext:
    return SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=_ARM,  # type: ignore[arg-type]
                         target_joints=JointPositions(tuple(math.radians(v) for v in joints_deg)))


def _judged(refresh: Any, guard: Any = None, joints_deg: "tuple[float, ...]" = _Q_DEG) -> Any:
    guard = guard or _guard(min_distance_mm=10.0)
    guard.set_perceived_fixtures(refresh.guard_boxes)
    return guard.evaluate(_ctx(joints_deg))


def _perceived(refresh: Any) -> Any:
    """The turned boxes of a refresh, in the shape the scene helpers read."""
    from src.robot.safety.planning.perceived import PerceivedBox

    return tuple(PerceivedBox(name=box.name, center_mm=tuple(box.center_mm),
                              dims_mm=tuple(2.0 * np.asarray(box.turned.half_extents_mm)),
                              yaw_rad=float(box.turned.yaw_rad), points=0, distance_mm=0.0)
                 for box in refresh.guard_boxes)


def _first_colliding_sample(start_deg: "tuple[float, ...]", end_deg: "tuple[float, ...]", solid: Solid,
                            **guard: Any) -> "tuple[Any, int]":
    """The joint path from ``start_deg`` to ``end_deg`` sampled as the gate samples it, and the first sample, 1-based, at
    which the arm meets ``solid`` (0 where it never does): the truth, ``solid`` as an exact fixture."""
    truth = SelfCollisionGuard(SelfCollisionSafetyConfig(
        kinematics_model="ur10", min_distance_mm=0.001,
        fixtures=[FixtureBoxConfig(name="truth", center_mm=solid.centre, half_extents_mm=solid.half)]), **guard)
    samples = joint_path_samples(tuple(math.radians(v) for v in start_deg), tuple(math.radians(v) for v in end_deg),
                                 reach_mm=SafetyPreflight([truth]).joint_radii_mm(_ARM), max_step_mm=10.0)
    for index, joints in enumerate(samples.configs):
        if truth.evaluate(SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=_ARM,  # type: ignore[arg-type]
                                        target_joints=JointPositions(tuple(joints)))).rejected:
            return samples, index + 1
    return samples, 0


def _refused_at(refusal: Any) -> int:
    """The 1-based sample a path gate's refusal names."""
    found = re.search(r"sample (\d+) of", refusal.message)
    assert found is not None, refusal.message
    return int(found.group(1))


class TheHousingHidesTheRimUnderItTests(unittest.TestCase):
    """Physics stays: a rim under the shoulder housing is refused though the housing hides it from the camera."""

    def setUp(self) -> None:
        _needs_the_engine(self, _guard(min_distance_mm=10.0))

    def test_a_47_mm_rim_the_housing_hides_from_the_camera_over_the_bin_is_refused(self) -> None:
        """⛔ Accepted before: the camera the investigation's test used, with the arm drawn in, sees no rim under the
        housing, and the height map left that stretch free."""
        tall = open_bin((0.0, -300.0), (300.0, 200.0), 47.0)
        self.assertLess(_ruler(tall), 8.0)
        camera = _CAMERAS["straight down over the bin"]
        depth = render(camera, tall, robot=robot_triangles(_Q_DEG))
        self.assertGreater(int(np.count_nonzero(depth < render(camera, tall) - 1e-6)), 20000,
                           "the control: the arm hides part of the frame")

        refresh = _refresh([(camera, depth)])
        self.assertTrue(refresh.ok, refresh.render())
        decision = _judged(refresh)
        self.assertTrue(decision.rejected, "a rim 6 mm under the housing was accepted")
        self.assertIn("fixture:seen_", decision.detail["pair"])
        self.assertIn("the robot's own body hid from the cameras", decision.message)
        self.assertIn("hid from the cameras", decision.detail["box_note"])

    def test_a_rim_the_housing_hides_is_refused_from_every_camera_over_the_ring(self) -> None:
        """⛔ Before, four of these six cameras let a 43 mm and a 47 mm rim under the housing pass. From over the base
        the housing's shadow cuts the bin in two, and the stretch between the halves is bridged."""
        arm = robot_triangles(_Q_DEG)
        for rim in (43.0, 47.0):
            scene = open_bin((0.0, -300.0), (300.0, 200.0), rim)
            self.assertLess(_ruler(scene), 11.0)
            for name, camera in _CAMERAS.items():
                with self.subTest(rim_mm=rim, camera=name):
                    refresh = _refresh([(camera, render(camera, scene, robot=arm))])
                    self.assertTrue(refresh.ok, refresh.render())
                    self.assertTrue(_judged(refresh).rejected, refresh.render())

    def test_a_post_the_housing_hides_refuses_the_joint_1_move_that_drives_the_housing_into_it(self) -> None:
        """⛔ Before, every sample of a joint-1 turn of 8 degrees passed, and from the 12th on the housing was in the
        post. A 10 x 10 mm post 62 mm tall stands in a 40 mm bin, under the housing's +x flank."""
        post = Solid((53.0, -215.0, 31.0), (5.0, 5.0, 31.0))
        scene = [*open_bin((0.0, -300.0), (300.0, 200.0), 40.0), post]
        samples, meets = _first_colliding_sample(_Q_DEG, (8.0, *_Q_DEG[1:]), post)
        self.assertGreater(meets, 1, "the control: the turn starts clear and meets the post")

        camera = _CAMERAS["straight down over the bin"]
        refresh = _refresh([(camera, render(camera, scene, robot=robot_triangles(_Q_DEG)))])
        self.assertTrue(refresh.ok, refresh.render())
        guard = _guard(min_distance_mm=10.0)
        guard.set_perceived_fixtures(refresh.guard_boxes)
        refused = SafetyPreflight([guard]).gate_joint_path(samples, arm=_ARM)  # type: ignore[arg-type]
        self.assertIsNotNone(refused, "a turn that drives the housing into the post passed every sample")
        self.assertLessEqual(_refused_at(refused), meets)


@lru_cache(maxsize=1)
def _hande() -> Any:
    from src.config.loader import load_robot_config
    from src.robot.safety.planning.hand import planner_hand

    return planner_hand(load_robot_config(profile="hande"))


def _hand_guard(**config: Any) -> SelfCollisionGuard:
    return SelfCollisionGuard(SelfCollisionSafetyConfig(kinematics_model="ur10", **config), hand=_hande())


class TheHandHidesNothingItStandsBesideTests(unittest.TestCase):
    """The Hand-E on the UR10's flange, drawn in with the arm. Its padded spheres take up to 27 mm of what the camera
    sees about it for the hand, and the hand hides what stands behind it."""

    _FOLDED = (0.0, -43.0, -127.0, 80.0, -90.0, 0.0)
    _DOWN = (0.0, -74.0, 124.0, -140.0, -90.0, 0.0)

    def setUp(self) -> None:
        _needs_the_engine(self, _hand_guard(min_distance_mm=10.0))

    def _hand_envelope(self, joints_deg: "tuple[float, ...]") -> SelfEnvelope:
        spheres = hand_spheres(_hande(), "ur10")
        assert spheres is not None
        return _envelope(joints_deg, spheres)

    def test_a_post_the_housing_hides_from_a_camera_beside_the_flange_refuses_the_turn(self) -> None:
        """⛔ Before, six of eight such cameras let the turn pass (review of 2026-09-30). The arm folded over the base,
        the flange 552 mm up pointing down, a camera beside it looking at the bin beside the base; the post of the
        test above stands in it under the housing, which the camera cannot see past. (A camera 80 mm along +x from the
        flange stands inside the hand, which the review's point-splat drawing let it see past.)"""
        post = Solid((53.0, -215.0, 31.0), (5.0, 5.0, 31.0))
        scene = [*open_bin((0.0, -300.0), (300.0, 200.0), 40.0), post]
        samples, meets = _first_colliding_sample(self._FOLDED, (8.0, *self._FOLDED[1:]), post, hand=_hande())
        self.assertGreater(meets, 1, "the control: the turn starts clear and meets the post")
        frames = yawed_link_transforms_mm("ur10", np.radians(self._FOLDED), 0.0)
        assert frames is not None
        flange = frames[6][:3, 3]
        robot = robot_triangles(self._FOLDED, hand=_hande())
        for offset in ((120.0, -40.0), (0.0, -80.0), (-80.0, 0.0), (60.0, -60.0)):
            with self.subTest(camera_beside_the_flange_mm=offset):
                camera = looking_at(tuple(flange + np.array([offset[0], offset[1], 0.0])), (0.0, -300.0, 40.0))
                self.assertGreater(int(np.count_nonzero(render(camera, scene) < robot_depth(camera, robot))), 100000,
                                   "the control: the camera sees the bin past the robot")
                refresh = _refresh([(camera, render(camera, scene, robot=robot))], limits=_WIDE,
                                   envelope=self._hand_envelope(self._FOLDED), near_point_mm=tuple(flange))
                self.assertTrue(refresh.ok, refresh.render())
                guard = _hand_guard(min_distance_mm=10.0)
                guard.set_perceived_fixtures(refresh.guard_boxes)
                refused = SafetyPreflight([guard]).gate_joint_path(samples, arm=_ARM)  # type: ignore[arg-type]
                self.assertIsNotNone(refused, "the turn into the hidden post passed every sample")
                self.assertLessEqual(_refused_at(refused), meets)

    def _wall(self, side: str, gap_mm: float, rim_over_tip_mm: float) -> "tuple[Solid, np.ndarray]":
        """A wall 5 mm thick and 300 mm long ``gap_mm`` from the hand's lowest 60 mm on ``side``, its rim
        ``rim_over_tip_mm`` over the fingertips, and a camera 450 mm over its rim, 300 mm to one side of it."""
        robot = robot_triangles(self._DOWN, hand=_hande())
        hand = robot[robot_triangles(self._DOWN).shape[0]:].reshape(-1, 3)
        tip = float(hand[:, 2].min())
        low = hand[hand[:, 2] < tip + 60.0]
        axis = 0 if side[1] == "x" else 1
        other = 1 - axis
        edge = low[:, axis].max() if side[0] == "+" else low[:, axis].min()
        rim = tip + rim_over_tip_mm
        centre, half = [0.0, 0.0, rim / 2.0], [0.0, 0.0, rim / 2.0]
        centre[axis] = edge + (gap_mm + 2.5) * (1.0 if side[0] == "+" else -1.0)
        centre[other] = float((low[:, other].max() + low[:, other].min()) / 2.0)
        half[axis], half[other] = 2.5, 150.0
        look = np.array([centre[0], centre[1], rim])
        eye = look + np.array([0.0, 0.0, 450.0])
        eye[other] -= 300.0
        return Solid(tuple(centre), tuple(half)), looking_at(tuple(eye), tuple(look))  # type: ignore[arg-type]

    def _judge_wall(self, side: str, gap_mm: float, rim_over_tip_mm: float) -> Any:
        wall, camera = self._wall(side, gap_mm, rim_over_tip_mm)
        truth = _hand_guard(min_distance_mm=0.001, fixtures=[FixtureBoxConfig(name="wall", center_mm=wall.centre,
                                                                               half_extents_mm=wall.half)])
        self.assertTrue(truth.evaluate(_ctx(self._DOWN)).accepted, "the control: the wall does not touch the hand")
        frames = yawed_link_transforms_mm("ur10", np.radians(self._DOWN), 0.0)
        assert frames is not None
        refresh = _refresh([(camera, render(camera, [wall], robot=robot_triangles(self._DOWN, hand=_hande())))],
                           limits=_WIDE, envelope=self._hand_envelope(self._DOWN),
                           near_point_mm=tuple(frames[6][:3, 3]))
        self.assertTrue(refresh.ok, refresh.render())
        return _judged(refresh, _hand_guard(min_distance_mm=10.0), self._DOWN)

    def test_a_wall_6_or_10_mm_from_the_fingers_is_refused(self) -> None:
        """⛔ Before, these three passed: the boxes of the wall stopped 6 to 12 mm short of the fingers, the spheres
        having taken the stretch in front of them, and past the hand the camera sees nothing of the wall."""
        for side, gap, rim in (("+x", 6.0, 30.0), ("+y", 6.0, 15.0), ("+y", 10.0, 15.0)):
            with self.subTest(side=side, gap_mm=gap, rim_over_tip_mm=rim):
                decision = self._judge_wall(side, gap, rim)
                self.assertTrue(decision.rejected, f"a wall {gap} mm from the fingers passed")

    def test_a_wall_that_keeps_its_distance_from_the_fingers_still_passes(self) -> None:
        """What the hand's spheres took is run on no higher than the wall and no further than they took it: a wall
        26 mm and more from the fingers, 20 mm past the box's margin and the guard's distance, passes as it did."""
        for side, gap, rim in (("+x", 26.0, 15.0), ("+x", 40.0, 30.0), ("+y", 26.0, 30.0), ("-y", 30.0, 15.0)):
            with self.subTest(side=side, gap_mm=gap, rim_over_tip_mm=rim):
                decision = self._judge_wall(side, gap, rim)
                self.assertTrue(decision.accepted, decision.message)


class TheBinsStillPassWithTheArmInViewTests(unittest.TestCase):
    """What the robot hides is filled from what was seen beside it, never past it: the bins that cleared the housing
    still pass, the arm drawn in, from every camera over the ring."""

    def setUp(self) -> None:
        _needs_the_engine(self, _guard(min_distance_mm=10.0))

    def test_a_bin_turned_30_degrees_48_mm_from_the_housing_passes(self) -> None:
        reach = 150.0 * math.sin(math.radians(30.0)) + 100.0 * math.cos(math.radians(30.0))
        centre = (0.0, -260.0 - reach)
        turned = open_bin(centre, (300.0, 200.0), 40.0, yaw_deg=30.0)
        self._passes_from_every_camera(centre, turned)

    def test_the_investigations_square_bin_passes_at_5_mm(self) -> None:
        centre = (0.0, -364.4)
        self._passes_from_every_camera(centre, open_bin(centre, (300.0, 200.0), 40.0))

    def _passes_from_every_camera(self, centre: "tuple[float, float]", solids: "list[Solid]") -> None:
        arm = robot_triangles(_Q_DEG)
        cameras = {
            "straight down over the bin": looking_down(*centre),
            "straight down over the housing, 700 mm up": looking_down(0.0, -180.0, 700.0),
            "from +x at 45 degrees": looking_at((500.0, centre[1], 550.0), (0.0, centre[1], 0.0)),
            "from -x at 45 degrees": looking_at((-500.0, centre[1], 550.0), (0.0, centre[1], 0.0)),
            "from -y at 45 degrees": looking_at((0.0, centre[1] - 500.0, 500.0), (0.0, centre[1], 0.0)),
            "from over the base at 60 degrees": looking_at((0.0, 300.0, 700.0), (0.0, centre[1], 0.0)),
        }
        for name, camera in cameras.items():
            with self.subTest(camera=name):
                refresh = _refresh([(camera, render(camera, solids, robot=arm))])
                self.assertTrue(refresh.ok, refresh.render())
                decision = _judged(refresh)
                self.assertTrue(decision.accepted, decision.message)

    def test_an_open_bin_by_the_base_keeps_its_inside_free(self) -> None:
        scene = open_bin((0.0, -420.0), (300.0, 200.0), 40.0)
        camera = looking_down(0.0, -420.0)
        refresh = _refresh([(camera, render(camera, scene, robot=robot_triangles(_Q_DEG)))])
        self.assertTrue(refresh.ok, refresh.render())
        inside = np.array([[x, y, z] for x in (-100.0, -50.0, 0.0, 50.0, 100.0) for y in (-470.0, -420.0, -370.0)
                           for z in (5.0, 20.0, 35.0)])
        self.assertEqual(covered_by(_perceived(refresh), inside), [])


class WhatAnotherCameraSawIsNotFilledTests(unittest.TestCase):
    def test_the_inside_the_housing_hides_from_above_is_free_once_a_camera_below_it_saw_the_floor(self) -> None:
        """The housing hides the bin's inside beside its wall from the camera over the bin; a camera from -y sees the
        floor there under the housing. Filled for the one, free for both. The probes stand a cell and the margin in
        from the wall, whose hidden stretch stays filled for both: the housing hides its top from both cameras."""
        scene = open_bin((0.0, -300.0), (300.0, 200.0), 30.0)
        arm = robot_triangles(_Q_DEG)
        above = _CAMERAS["straight down over the bin"]
        below = _CAMERAS["from -y at 45 degrees"]
        under_the_housing = np.array([[x, -245.0, 20.0] for x in (-20.0, 0.0, 20.0)])

        alone = _refresh([(above, render(above, scene, robot=arm))])
        both = _refresh([(above, render(above, scene, robot=arm)), (below, render(below, scene, robot=arm))])
        self.assertTrue(alone.ok and both.ok, alone.render() + both.render())
        self.assertTrue(covered_by(_perceived(alone), under_the_housing),
                        "what the housing hides from the camera above is free")
        self.assertEqual(covered_by(_perceived(both), under_the_housing), [])


class WhatAKeepOutLeftOutStaysOutTests(unittest.TestCase):
    def test_nothing_is_filled_within_the_margin_of_a_keep_out(self) -> None:
        """The target a pick closes on is left out of the world, and the hand that closes on it hides it: the fill
        never stands in it. A box held out over the stretch of the 47 mm rim the housing hides: without it the fill
        covers the stretch; with it, nothing reaches into the box past the margin every box is grown by."""
        from src.robot.core.keep_out import KeepOutBox
        from src.robot.safety.planning.perceived import DepthView, SelfBody, build_perceived_boxes

        tall = open_bin((0.0, -300.0), (300.0, 200.0), 47.0)
        camera = _CAMERAS["straight down over the bin"]
        frames = yawed_link_transforms_mm("ur10", np.radians(_Q_DEG), 0.0)
        capsules = arm_capsules("ur10")
        assert frames is not None and capsules is not None
        view = DepthView(surface_depth_mm=render(camera, tall, robot=robot_triangles(_Q_DEG)), intrinsics=K,
                         camera_to_base=camera, name="over the bin", timestamp=100.0)
        placed = np.eye(4)
        placed[:3, 3] = (0.0, -202.5, 30.0)
        target = KeepOutBox.from_matrix("target", placed, (40.0, 40.0, 35.0))
        deep_inside = np.array([[x, y, z] for x in (-20.0, 0.0, 20.0) for y in (-222.5, -202.5, -182.5)
                                for z in (10.0, 30.0, 50.0)])

        def world(keep_out: tuple) -> Any:
            return build_perceived_boxes(views=(view,), limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2),
                                         self_body=SelfBody.from_frames(frames, capsules, padding_mm=15.0),
                                         near_point_mm=(600.0, 0.0, 300.0), keep_out=keep_out, cut_around=keep_out)

        self.assertTrue(covered_by(world(()).boxes, deep_inside), "the control: the fill covers the hidden stretch")
        self.assertEqual(covered_by(world((target,)).boxes, deep_inside), [])


class ALoneBlockUnderTheHousingTests(unittest.TestCase):
    """A 30 x 30 mm block 62 mm tall under the housing's +x flank, its top 8 to 11 mm under it (review of 2026-09-30).
    Seen from the side it is refused. Seen only from above, the housing hides all of it: no camera saw anything of it
    and nothing beside it, so it is not in the world (the limitation the docs state)."""

    def setUp(self) -> None:
        _needs_the_engine(self, _guard(min_distance_mm=10.0))

    def test_a_block_under_the_housing_seen_from_the_side_is_refused(self) -> None:
        block = Solid((53.0, -215.0, 31.0), (15.0, 15.0, 31.0))
        arm = robot_triangles(_Q_DEG)
        for name, camera in (("from +x", looking_at((500.0, -215.0, 480.0), (53.0, -215.0, 31.0))),
                             ("from -y", looking_at((53.0, -700.0, 480.0), (53.0, -215.0, 31.0)))):
            with self.subTest(camera=name):
                refresh = _refresh([(camera, render(camera, [block], robot=arm))])
                self.assertTrue(refresh.ok, refresh.render())
                self.assertTrue(_judged(refresh).rejected, refresh.render())


class ARobotSeenOffItsModelIsNamedTests(unittest.TestCase):
    """A camera placed a degree off sees the arm's links 15 mm off at 0.85 m, past the self filter's padding, and the
    boxes of them stand where the arm does: the refusal says the box may be the robot itself (review of 2026-09-30).
    Half a degree off, nothing of the arm is left. The camera looks at the upper arm from 0.8 m, the arm alone in view."""

    _CAMERA = looking_at((300.0, -450.0, 900.0), (200.0, -150.0, 200.0))

    def setUp(self) -> None:
        _needs_the_engine(self, _guard(min_distance_mm=10.0))

    def _placed_off(self, axis: "tuple[float, float, float]", degrees: float) -> Any:
        turn = np.eye(4)
        unit = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        skew = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
        angle = math.radians(degrees)
        turn[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew
        depth = render(self._CAMERA, [], robot=robot_triangles(_Q_DEG))
        return _refresh([(self._CAMERA @ turn, depth)], limits=_WIDE)

    def test_half_a_degree_off_leaves_nothing_of_the_arm(self) -> None:
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            with self.subTest(axis=axis):
                refresh = self._placed_off(axis, 0.5)
                self.assertTrue(refresh.ok, refresh.render())
                self.assertTrue(_judged(refresh).accepted)

    def test_a_degree_off_the_refusal_names_the_robot_seen_off(self) -> None:
        refresh = self._placed_off((0.0, 1.0, 0.0), 1.0)
        self.assertTrue(refresh.ok, refresh.render())
        decision = _judged(refresh)
        self.assertTrue(decision.rejected)
        self.assertIn("may be the robot itself", decision.message)
        self.assertIn("hand-eye calibration", decision.detail["box_note"])


class TheGuardAndThePlannerHoldTheSameFilledBoxesTests(unittest.TestCase):
    def test_the_boxes_filled_where_the_robot_hid_the_rim_reach_both_authorities_alike(self) -> None:
        """The fill is part of the one list of turned boxes: the planner is handed exactly what the guard judges."""
        from tests.test_the_guard_holds_the_boxes_the_planner_holds import _as_wire

        tall = open_bin((0.0, -300.0), (300.0, 200.0), 47.0)
        client = _Recording()
        for name in ("straight down over the bin", "from over the base at 60 degrees"):
            camera = _CAMERAS[name]

            def grab(depth: np.ndarray = render(camera, tall, robot=robot_triangles(_Q_DEG))) -> Any:
                class _Camera:
                    def grab_surface_depth(self) -> DepthSnapshot:
                        return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=100.0)

                return _Camera()

            world = LivePlannerWorld(cameras=(CameraView(name="look", depth_source=grab(), camera_to_base=camera),),
                                     declared=_SUPPORT, limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2),
                                     max_age_ms=5000.0)
            with self.subTest(camera=name):
                refresh = refresh_planner_world(source=world, client=client, self_envelope=_envelope(),
                                                near_point_mm=(600.0, 0.0, 300.0), now=100.1)
                self.assertTrue(refresh.ok, refresh.render())
                self.assertTrue(any("hid from the cameras" in box.note for box in refresh.guard_boxes),
                                "the control: the robot hid part of the rim")
                sent = [box for box in client.worlds[-1] if box["name"].startswith("seen_")]
                self.assertEqual(sent, [_as_wire(box) for box in refresh.guard_boxes])


class TheWorldSaysWhatTheRobotHidTests(unittest.TestCase):
    def test_the_boxes_and_the_world_count_the_cells_the_robot_hid(self) -> None:
        from src.robot.safety.planning.perceived import DepthView, SelfBody, build_perceived_boxes

        tall = open_bin((0.0, -300.0), (300.0, 200.0), 47.0)
        camera = _CAMERAS["straight down over the bin"]
        frames = yawed_link_transforms_mm("ur10", np.radians(_Q_DEG), 0.0)
        capsules = arm_capsules("ur10")
        assert frames is not None and capsules is not None
        view = DepthView(surface_depth_mm=render(camera, tall, robot=robot_triangles(_Q_DEG)), intrinsics=K,
                         camera_to_base=camera, name="over the bin", timestamp=100.0)
        world = build_perceived_boxes(views=(view,), limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2),
                                      self_body=SelfBody.from_frames(frames, capsules, padding_mm=15.0),
                                      near_point_mm=(600.0, 0.0, 300.0))
        hid = [box for box in world.boxes if box.hidden_cells]  # type: ignore[attr-defined]
        self.assertTrue(hid)
        self.assertEqual(world.hidden_cells, sum(box.hidden_cells for box in world.boxes))  # type: ignore[attr-defined]
        self.assertEqual(world.to_dict()["hidden_cells"], world.hidden_cells)  # type: ignore[attr-defined]
        self.assertEqual(world.to_dict()["boxes"][world.boxes.index(hid[0])]["hidden_cells"], hid[0].hidden_cells)
        self.assertIn("the robot hid from the cameras", world.render())
        self.assertIn(f"{hid[0].hidden_cells} of its cells", hid[0].note())  # type: ignore[attr-defined]

        # The same frame drawn without the arm: nothing hidden, nothing said.
        clear = DepthView(surface_depth_mm=render(camera, tall), intrinsics=K, camera_to_base=camera, name="over the bin",
                          timestamp=100.0)
        seen = build_perceived_boxes(views=(clear,), limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2),
                                     self_body=SelfBody.from_frames(frames, capsules, padding_mm=15.0),
                                     near_point_mm=(600.0, 0.0, 300.0))
        self.assertEqual(seen.hidden_cells, 0)  # type: ignore[attr-defined]
        self.assertNotIn("hid from the cameras", seen.render())


if __name__ == "__main__":
    unittest.main()
