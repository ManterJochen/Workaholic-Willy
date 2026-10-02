"""The robot's base is the robot: the camera world takes it out as it takes the arm out (fix plan RC3, Track D).

Twice on 2026-10-01 the camera's view of the UR10's base became a box the shoulder stood in: ``shoulder|seen_54`` at
-3.5 mm with a part in the hand at R6's place (robot.log:425), and ``shoulder|seen_32`` at -17.4 mm at R7's start
(robot.log:448), every look of that program after it. The self filter's links began at the shoulder; nothing stood for
the base. Now the base is a cylinder on the base frame, 95.05 mm about the axis and 38 mm up (``ur10_base.obj``), with
its surface laid like the links': a point is the base only within the padding, ``perceived.margin_mm`` (15 mm), of that
surface. The guard's own parts are unchanged (the owner, 2026-09-30).

What the filter cannot do is said here too. The logged boxes stood 54-55 mm high over the base, 16-17 mm over its top,
past the padding: if what the camera saw there stood that high, it stays an obstacle, and the box of it still meets the
shoulder. The source points were not recorded, so RC3 is closed only for a reading within the padding; Monday records
the views and one photo of the base.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.robot.core import JointPositions, MotionCommand
from src.robot.safety.guard import SafetyContext
from src.robot.safety.planning.environment import project_root
from src.robot.safety.planning.live_world import _guard_boxes
from src.robot.safety.planning.perceived import (
    DepthView,
    PerceivedWorld,
    SelfBody,
    SelfEnvelope,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
)
from src.robot.safety.planning.self_envelope import (
    ROBOT_BASES,
    arm_capsules,
    base_capsule,
    self_envelope,
    yawed_link_transforms_mm,
)
from tests._seen_scenes import FOCAL, HEIGHT, K, WIDTH, Solid, covered_by, looking_at, looking_down, render
from tests.test_the_guard_holds_the_boxes_the_planner_holds import _ARM, _guard, _needs_the_engine

#: R6's place, the part in the hand (robot.log:425), and R7's start, LOOK[0] (robot.log:448).
_R6_PLACE_DEG = (-46.3, -88.2, -106.4, -75.4, 90.0, -282.6)
_R7_START_DEG = (-87.5, -64.5, -85.8, -129.8, 90.3, -108.8)
_PADDING_MM = 15.0
_BASE = ROBOT_BASES["ur10"]
#: Where the camera world builds here: room about the base, the bench at 0.
_LIMITS = WorldBuildLimits(x_mm=(-800.0, 800.0), y_mm=(-1200.0, 600.0), z_mm=(-50.0, 900.0), support_plane_top_mm=0.0)
_TUNING = WorldBuildTuning(pixel_stride=1)
#: Two cameras over the base, from either side of the shoulder.
_CAMERAS = {
    "over the base from +x": looking_at((350.0, -350.0, 900.0), (40.0, -20.0, 20.0)),
    "over the base from -x": looking_at((-350.0, -400.0, 900.0), (-10.0, -60.0, 20.0)),
}
#: The logged boxes' sources as the log describes them: the box less the 15 mm margin, up to its raw top.
_SEEN_54 = Solid((89.15, 5.7, 54.9 / 2.0), ((115.3 - 63.0) / 2.0, (19.3 + 7.9) / 2.0, 54.9 / 2.0))
_SEEN_32 = Solid((-9.0, -61.6, 54.2 / 2.0), (54.9 / 2.0, 84.3 / 2.0, 54.2 / 2.0), 61.1)
#: seen_54's footprint over the base, read 12 mm over its top: within the padding.
_SEEN_54_LOW = Solid((79.0, 5.7, 25.0), ((95.0 - 63.0) / 2.0, (19.3 + 7.9) / 2.0, 25.0))
_BASE_MESH = (project_root() / "ext_deps" / "curobo" / "curobo" / "content" / "assets" / "robot" / "ur_description"
              / "meshes" / "ur10" / "ur10_base.obj")


def _cylinder_depth(camera_to_base: np.ndarray, radius_mm: float, top_mm: float) -> np.ndarray:
    """Camera depth to an upright cylinder about the base axis, from the bench up to ``top_mm``; ``inf`` where missed."""
    cols, rows = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(cols - K[0, 2]) / FOCAL, (rows - K[1, 2]) / FOCAL, np.ones_like(cols)], -1).reshape(-1, 3)
    d = rays @ camera_to_base[:3, :3].T
    o = camera_to_base[:3, 3]
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2.0 * (o[0] * d[:, 0] + o[1] * d[:, 1])
    c = o[0] ** 2 + o[1] ** 2 - radius_mm ** 2
    disc = b * b - 4.0 * a * c
    hit = (disc >= 0.0) & (a > 1e-12)
    t = (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * np.where(a > 1e-12, a, 1.0))
    z = o[2] + t * d[:, 2]
    side = np.where(hit & (t > 0.0) & (z >= 0.0) & (z <= top_mm), t, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        lid = (top_mm - o[2]) / d[:, 2]
    on_lid = (lid > 0.0) & ((o[0] + lid * d[:, 0]) ** 2 + (o[1] + lid * d[:, 1]) ** 2 <= radius_mm ** 2)
    return np.minimum(side, np.where(on_lid, lid, np.inf)).reshape(HEIGHT, WIDTH)


def _frame(camera: np.ndarray, solids: "list[Solid]") -> np.ndarray:
    """What the camera sees: the bench, ``solids`` and the base itself."""
    depth = render(camera, solids)
    base = _cylinder_depth(camera, _BASE.radius_mm, _BASE.top_mm)
    return np.where(base < np.where(depth > 0.0, depth, np.inf), base, depth)


def _envelope(joints_deg: "tuple[float, ...]", *, base: bool) -> SelfEnvelope:
    frames = yawed_link_transforms_mm("ur10", np.radians(joints_deg), 0.0)
    links = arm_capsules("ur10")
    capsule = base_capsule("ur10")
    assert frames is not None and links is not None and capsule is not None
    return SelfEnvelope(frames_mm=tuple(frames), capsules=links + ((capsule,) if base else ()))


def _world(camera: np.ndarray, depth: np.ndarray, joints_deg: "tuple[float, ...]", *, base: bool) -> PerceivedWorld:
    envelope = _envelope(joints_deg, base=base)
    return build_perceived_boxes(
        views=[DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=camera, name="camera")],
        limits=_LIMITS, tuning=_TUNING, near_point_mm=(0.0, 0.0, 50.0),
        self_body=SelfBody.from_frames(envelope.frames_mm, envelope.capsules, padding_mm=_PADDING_MM),
        reach=envelope.reach(padding_mm=_PADDING_MM),
    )


def _shoulder_refusal(world: PerceivedWorld, joints_deg: "tuple[float, ...]") -> str:
    """What the exact guard at 5 mm says of the shoulder against the world's boxes: empty when it accepts."""
    guard = _guard(min_distance_mm=5.0)
    guard.set_perceived_fixtures(_guard_boxes(world))
    decision = guard.evaluate(SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=_ARM,  # type: ignore[arg-type]
                                            target_joints=JointPositions(tuple(math.radians(v) for v in joints_deg))))
    return decision.message if decision.rejected else ""


def _near_the_base(world: PerceivedWorld) -> list[str]:
    """The boxes that reach into the base's padded cylinder."""
    reach = _BASE.radius_mm + _PADDING_MM
    out = []
    for box in world.boxes:
        half = np.asarray(box.enclosing_half_extents_mm)
        nearest = np.clip(np.zeros(2), np.asarray(box.center_mm[:2]) - half[:2], np.asarray(box.center_mm[:2]) + half[:2])
        if float(np.hypot(*nearest)) < reach and box.center_mm[2] - half[2] < _BASE.top_mm + _PADDING_MM:
            out.append(box.name)
    return out


def _seen_points(camera: np.ndarray, depth: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.0))
    z = depth[rows, cols]
    local = np.column_stack(((cols - K[0, 2]) * z / FOCAL, (rows - K[1, 2]) * z / FOCAL, z))
    return local @ camera[:3, :3].T + camera[:3, 3]


def _cylinder_distance(points: np.ndarray) -> np.ndarray:
    """How far each point lies from the base's cylinder (its top disc and its side; 0 inside)."""
    r = np.hypot(points[:, 0], points[:, 1])
    over = np.maximum(r - _BASE.radius_mm, 0.0)
    above = np.maximum(points[:, 2] - _BASE.top_mm, 0.0)
    below = np.maximum(-points[:, 2], 0.0)
    return np.hypot(over, np.maximum(above, below))


class TheBaseIsMeasuredAndCarriedTests(unittest.TestCase):
    def test_the_ur10_base_is_the_measured_cylinder_on_the_base_frame(self) -> None:
        self.assertEqual((_BASE.radius_mm, _BASE.top_mm), (95.05, 38.0))
        self.assertIn("ur10_base.obj", _BASE.provenance)
        capsule = base_capsule("ur10")
        assert capsule is not None and capsule.surface is not None
        self.assertEqual(capsule.frame, 0)
        self.assertEqual((capsule.start_mm, capsule.end_mm), ((0.0, 0.0, 0.0), (0.0, 0.0, 38.0)))
        self.assertAlmostEqual(capsule.radius_mm, 95.05, places=3)
        # Laid no coarser than the padding, so the filter refines to the surface rather than taking the capsule whole.
        self.assertLessEqual(capsule.surface.spacing_mm, _PADDING_MM)

    def test_a_model_whose_base_was_not_measured_keeps_it_in_the_world(self) -> None:
        self.assertIsNone(base_capsule("ur5e"))
        self.assertNotIn("ur5e", ROBOT_BASES)

    def test_the_owners_cell_carries_its_base_in_the_envelope(self) -> None:
        from tests.test_the_exact_guard_decides_the_planners_self_pairs import owner_like_arm

        arm: Any = owner_like_arm()
        envelope = self_envelope(arm._preflight, arm, [math.radians(v) for v in _R7_START_DEG])
        if envelope is None:
            self.skipTest("no committed arm bundle or hand map for the self filter on this tree")
        self.assertIn(base_capsule("ur10"), envelope.capsules)
        # The shape rides on its own too: the support detection leaves the disc round the base out.
        self.assertIs(envelope.base, ROBOT_BASES["ur10"])

    def test_the_base_mesh_lies_inside_the_cylinder(self) -> None:
        if not _BASE_MESH.is_file():
            raise unittest.SkipTest("cuRobo's ur_description is not installed under ext_deps on this checkout")
        rows = [line.split()[1:4] for line in _BASE_MESH.read_text(encoding="utf-8").splitlines()
                if line.startswith("v ")]
        vertices = np.asarray(rows, dtype=np.float64) * 1000.0
        radius = np.hypot(vertices[:, 0], vertices[:, 1])
        self.assertAlmostEqual(float(radius.max()), _BASE.radius_mm, delta=0.01)
        self.assertAlmostEqual(float(vertices[:, 2].max()), _BASE.top_mm, delta=0.01)
        self.assertGreater(float(vertices[:, 2].min()), -0.01)

    def test_a_dome_over_the_top_is_taken_only_within_the_padding(self) -> None:
        """Every point of a dome 80 mm about the top's centre that lies further than 15 mm from the base stays."""
        capsule = base_capsule("ur10")
        assert capsule is not None and capsule.surface is not None
        body = SelfBody.from_frames([np.eye(4)], (capsule,), padding_mm=_PADDING_MM)
        theta, phi = np.meshgrid(np.linspace(0.0, math.pi / 2.0, 60), np.linspace(0.0, 2.0 * math.pi, 120))
        for radius in (20.0, 60.0, 80.0, 120.0):
            with self.subTest(dome_mm=radius):
                dome = np.column_stack((radius * np.sin(theta.ravel()) * np.cos(phi.ravel()),
                                        radius * np.sin(theta.ravel()) * np.sin(phi.ravel()),
                                        _BASE.top_mm + radius * np.cos(theta.ravel())))
                taken = body.contains(dome)
                distance = _cylinder_distance(dome)
                self.assertFalse(bool(np.any(taken & (distance > _PADDING_MM))), "a point past the padding was taken")
                # The laid surface errs long by at most its spacing, never short.
                within = distance <= _PADDING_MM - capsule.surface.spacing_mm
                self.assertTrue(bool(np.all(taken[within])), "a point well within the padding was left")


class TheLoggedBoxesAtTheBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        _needs_the_engine(self, _guard(min_distance_mm=5.0))

    def test_the_base_as_it_stands_was_a_box_the_shoulder_stood_in_and_is_now_the_robot(self) -> None:
        for name, camera in _CAMERAS.items():
            depth = _frame(camera, [])
            for joints in (_R6_PLACE_DEG, _R7_START_DEG):
                with self.subTest(camera=name, joints=joints):
                    before = _world(camera, depth, joints, base=False)
                    self.assertTrue(_near_the_base(before))
                    self.assertIn("shoulder|fixture:seen_", _shoulder_refusal(before, joints))
                    after = _world(camera, depth, joints, base=True)
                    self.assertEqual(_near_the_base(after), [])
                    self.assertEqual(_shoulder_refusal(after, joints), "")

    def test_what_stood_over_the_base_within_the_padding_leaves_with_it(self) -> None:
        for name, camera in _CAMERAS.items():
            depth = _frame(camera, [_SEEN_54_LOW])
            for joints in (_R6_PLACE_DEG, _R7_START_DEG):
                with self.subTest(camera=name, joints=joints):
                    self.assertIn("shoulder|fixture:seen_", _shoulder_refusal(
                        _world(camera, depth, joints, base=False), joints))
                    after = _world(camera, depth, joints, base=True)
                    self.assertEqual(_near_the_base(after), [])
                    self.assertEqual(_shoulder_refusal(after, joints), "")

    def test_the_logged_boxes_read_as_high_as_logged_stay_and_the_shoulder_still_meets_them(self) -> None:
        """RC3 stays open for such a reading: the filter takes what lies within 15 mm of the base and nothing more."""
        capsule = base_capsule("ur10")
        assert capsule is not None
        base_alone = SelfBody.from_frames([np.eye(4)], (capsule,), padding_mm=_PADDING_MM)
        for source in (_SEEN_54, _SEEN_32):
            for name, camera in _CAMERAS.items():
                depth = _frame(camera, [source])
                points = _seen_points(camera, depth)
                points = points[np.hypot(points[:, 0], points[:, 1]) < 250.0]
                taken = base_alone.contains(points)
                with self.subTest(source=source.centre, camera=name):
                    # What stays: everything above 53 mm or beyond 110 mm from the axis, everything past the padding.
                    above = points[:, 2] > _BASE.top_mm + _PADDING_MM
                    beyond = np.hypot(points[:, 0], points[:, 1]) > _BASE.radius_mm + _PADDING_MM
                    self.assertFalse(bool(np.any(taken & (above | beyond))))
                    self.assertFalse(bool(np.any(taken & (_cylinder_distance(points) > _PADDING_MM))))
                    self.assertTrue(bool(np.any(above & ~taken)))
                    for joints in (_R6_PLACE_DEG, _R7_START_DEG):
                        after = _world(camera, depth, joints, base=True)
                        self.assertIn("shoulder|fixture:seen_", _shoulder_refusal(after, joints),
                                      "the logged reading is past the padding and its box still meets the shoulder")


class TheBandsEdgeKeepsWhatStandsBesideTheBaseTests(unittest.TestCase):
    """What stands beside the base past the padding stays an obstacle."""

    def _kept(self, solid: Solid, joints_deg: "tuple[float, ...]" = _R7_START_DEG) -> None:
        camera = looking_down(float(solid.centre[0]), float(solid.centre[1]), 900.0)
        depth = _frame(camera, [solid])
        world = _world(camera, depth, joints_deg, base=True)
        top = np.array([[solid.centre[0], solid.centre[1], solid.centre[2] + solid.half[2] - 1.0]])
        self.assertTrue(covered_by(world.boxes, top), f"{solid} left the world")

    def test_blocks_20_to_30_mm_from_the_base_stay(self) -> None:
        for gap in (20.0, 25.0, 30.0):
            for angle in (0.0, 90.0, 200.0):
                with self.subTest(gap_mm=gap, angle_deg=angle):
                    r = _BASE.radius_mm + gap + 15.0
                    a = math.radians(angle)
                    self._kept(Solid((r * math.cos(a), r * math.sin(a), 20.0), (15.0, 15.0, 20.0), angle))

    def test_a_part_beside_the_rim_stays(self) -> None:
        r = _BASE.radius_mm + 17.0 + 15.0
        self._kept(Solid((0.0, -r, 30.0), (15.0, 15.0, 30.0)))

    def test_a_40_mm_cube_160_mm_from_the_axis_stays(self) -> None:
        self._kept(Solid((160.0, 0.0, 20.0), (20.0, 20.0, 20.0)))
        self._kept(Solid((0.0, -160.0, 20.0), (20.0, 20.0, 20.0)))

    def test_ws_bin_30_mm_from_the_housing_stays_its_walls(self) -> None:
        from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import OFF_DEG, _bin

        _needs_the_engine(self, _guard(min_distance_mm=5.0))
        _, solids = _bin(30.0)
        camera = looking_down(0.0, -300.0, 900.0)
        depth = _frame(camera, list(solids))
        world = _world(camera, depth, OFF_DEG, base=True)
        for wall in solids[:4]:
            with self.subTest(wall=wall.centre):
                top = np.array([[wall.centre[0], wall.centre[1], 2.0 * wall.half[2] - 1.0]])
                self.assertTrue(covered_by(world.boxes, top), f"a wall of W's bin left the world: {wall}")


if __name__ == "__main__":
    unittest.main()
