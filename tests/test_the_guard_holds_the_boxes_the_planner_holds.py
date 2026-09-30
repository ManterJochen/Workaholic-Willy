"""The exact guard holds the same turned boxes cuRobo holds, and keeps its own distance from what a camera saw.

The owner's cell refused the upper arm beside the base (2026-09-30): ``upper_arm|fixture:seen_00``. Three things
stacked. cuRobo was handed each seen box turned, and the guard the axis-aligned box around it, up to 1.41 times as wide
per side. The guard asked the same 10 mm of a seen box as of a measured fixture, although a seen box is already grown
15 mm, so a camera-seen object needed 25 mm of ruler clearance square to BASE and more turned. And nothing in a refusal
said where the box was or where the arm stood.

Now the guard judges the turned box itself (the capsule fallback keeps the enclosure, the safe side), a seen box at
``robot.safety.self_collision.perceived_min_distance_mm`` (5 mm: 20 mm to what the camera saw off a face, about 26 mm
across an edge, where the 15 mm on two faces of the box reaches 21), declared fixtures and the arm against itself at
``min_distance_mm`` as before, and a load refuses a config whose seen-box distance and box margin together fall short
of that 10 mm: it is the step every path is sampled at, so the real surface stays clear between two samples. A refusal
names the box's corners, turn and size, and the joints in degrees.

The UR10 cases are the investigation's own (scratchpad selfcol_seen_answer.md), rerun on the real pipeline: a bin ray
cast straight down, the live world, the refresh, the exact guard on the committed UR10 meshes at joints
(0, -60, 80, -110, -90, 0) degrees, where the shoulder housing hangs 52 mm over the base plate along base -Y. They skip
where no collision engine loads. Names new with this change are imported inside the tests.
"""

from __future__ import annotations

import logging
import math
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic import ValidationError

from src.config.schema.robot.safety_schema import FixtureBoxConfig, RobotSafetyConfig, SelfCollisionSafetyConfig
from src.robot.core import JointPositions, MotionCommand
from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety.guard import SafetyContext
from src.robot.safety.planning.environment import collision_mesh_bundle
from src.robot.safety.planning.live_world import CameraView, DepthSnapshot, LivePlannerWorld, refresh_planner_world
from src.robot.safety.planning.perceived import SelfEnvelope, WorldBuildTuning
from src.robot.safety.planning.self_envelope import arm_capsules, yawed_link_transforms_mm
from src.robot.safety.planning.world import planner_cuboid
from src.robot.safety.preflight import SafetyPreflight
from src.robot.safety.self_collision import SelfCollisionGuard
from tests._seen_scenes import K, LIMITS, Solid, looking_down, open_bin, render

_ARM = SimpleNamespace(capabilities=SimpleNamespace(vendor="ur", model="ur10"))
#: The investigation's pose: the upper arm's shoulder housing hangs along base -Y, 86 to 263 mm out, 52 mm up.
_Q_DEG = (0.0, -60.0, 80.0, -110.0, -90.0, 0.0)
_Q = tuple(math.radians(v) for v in _Q_DEG)
_SUPPORT = (planner_cuboid("support_plane", (0.0, 0.0, -25.0), (2000.0, 2000.0, 50.0)),)


def _upper_arm_vertices() -> np.ndarray:
    frames = yawed_link_transforms_mm("ur10", np.asarray(_Q), 0.0)
    assert frames is not None
    with np.load(collision_mesh_bundle("ur10")) as bundle:
        vertices = bundle["upper_arm__v"].astype(np.float64)
    return vertices @ frames[2][:3, :3].T + frames[2][:3, 3]


def _ruler(solids: "list[Solid]") -> float:
    """What a ruler says: the upper arm's nearest vertex to any of the solids, millimetres."""
    vertices = _upper_arm_vertices()
    return float(min(solid.distance_mm(vertices).min() for solid in solids))


class _Recording:
    """A planner client that keeps every world it was handed and confirms all of it."""

    def __init__(self) -> None:
        self.worlds: list[list[dict[str, Any]]] = []

    def set_world(self, boxes: list, meshes: "list | None" = None) -> int:
        self.worlds.append([dict(box) for box in boxes])
        return len(boxes) + len(meshes or ())


def _refresh(solids: "list[Solid]", camera_xy: "tuple[float, float]", *, client: "_Recording | None" = None) -> Any:
    """The live world a camera 900 mm over ``camera_xy`` builds of ``solids``, refreshed into ``client``."""
    camera = looking_down(*camera_xy)
    depth = render(camera, solids)

    class _Camera:
        def grab_surface_depth(self) -> DepthSnapshot:
            return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=100.0)

    world = LivePlannerWorld(
        cameras=(CameraView(name="look", depth_source=_Camera(), camera_to_base=camera),), declared=_SUPPORT,
        limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2), max_age_ms=5000.0,
    )
    frames = yawed_link_transforms_mm("ur10", np.asarray(_Q), 0.0)
    capsules = arm_capsules("ur10")
    assert frames is not None and capsules is not None
    envelope = SelfEnvelope(frames_mm=tuple(frames), capsules=capsules)
    return refresh_planner_world(source=world, client=client or _Recording(), self_envelope=envelope,
                                 near_point_mm=(600.0, 0.0, 300.0), now=100.1)


def _ctx() -> SafetyContext:
    return SafetyContext(command=MotionCommand.MOVE_JOINTS, target_joints=JointPositions(_Q), arm=_ARM)  # type: ignore[arg-type]


def _guard(**config: Any) -> SelfCollisionGuard:
    return SelfCollisionGuard(SelfCollisionSafetyConfig(kinematics_model="ur10", **config))


def _needs_the_engine(test: unittest.TestCase, guard: SelfCollisionGuard) -> None:
    if guard.exact_mesh_engine(_ARM) is None:
        test.skipTest("no exact mesh engine or UR10 bundle on this box")


def _as_wire(box: AxisAlignedBox) -> dict[str, Any]:
    """A guard box in the planner's wire format, from the turned box it holds."""
    turned = box.turned  # type: ignore[attr-defined]
    return planner_cuboid(box.name, box.center_mm, 2.0 * np.asarray(turned.half_extents_mm), yaw_rad=turned.yaw_rad)


# ---------------------------------------------------------------------------------------------------------------------
# One list of boxes, for both
# ---------------------------------------------------------------------------------------------------------------------


class TheGuardAndThePlannerHoldOneListTests(unittest.TestCase):
    _SCENE = [*open_bin((0.0, -430.0), (300.0, 200.0), 40.0, yaw_deg=30.0),
              Solid((260.0, -380.0, 30.0), (20.0, 50.0, 30.0), yaw_deg=-15.0)]

    def test_the_refresh_hands_the_guard_the_turned_boxes_it_hands_the_planner(self) -> None:
        """⛔ Before, the guard got the axis-aligned box around each turned one the planner got."""
        client = _Recording()
        refresh = _refresh(self._SCENE, (0.0, -430.0), client=client)
        self.assertTrue(refresh.ok, refresh.render())

        sent = [box for box in client.worlds[-1] if box["name"].startswith("seen_")]
        self.assertGreater(len(sent), 1)
        self.assertEqual(sent, [_as_wire(box) for box in refresh.guard_boxes])
        for box in refresh.guard_boxes:
            with self.subTest(box=box.name):
                # The enclosure a guard with no turned boxes holds: never smaller than the turned box.
                turned = box.turned  # type: ignore[attr-defined]
                c, s = abs(math.cos(turned.yaw_rad)), abs(math.sin(turned.yaw_rad))
                hx, hy, hz = turned.half_extents_mm
                np.testing.assert_allclose(box.half_extents_mm, (hx * c + hy * s, hx * s + hy * c, hz), atol=1e-9)

    def test_both_get_a_bin_by_the_base_as_its_walls_and_a_low_part_at_its_own_height(self) -> None:
        """Through the live world: the inside of an open bin beside the base, and the space over a part 20 mm from a
        taller one, are in no box the planner is sent and in no box the guard holds."""
        scene = [*open_bin((0.0, -420.0), (300.0, 200.0), 40.0),
                 Solid((260.0, -400.0, 20.0), (30.0, 30.0, 20.0)), Solid((340.0, -400.0, 45.0), (30.0, 30.0, 45.0))]
        client = _Recording()
        refresh = _refresh(scene, (100.0, -420.0), client=client)
        self.assertTrue(refresh.ok, refresh.render())

        free = np.array([[x, y, z] for x in (-80.0, 0.0, 80.0) for y in (-460.0, -420.0, -380.0) for z in (10.0, 35.0)]
                        + [[260.0, -400.0, 70.0], [245.0, -385.0, 70.0]])
        sent = [box for box in client.worlds[-1] if box["name"].startswith("seen_")]
        for wire in sent:
            with self.subTest(planner_box=wire["name"]):
                centre = np.asarray(wire["pose"][:3]) * 1000.0
                qw, qz = wire["pose"][3], wire["pose"][6]
                yaw = 2.0 * math.atan2(qz, qw)
                local = free - centre
                c, s = math.cos(yaw), math.sin(yaw)
                u, v = c * local[:, 0] + s * local[:, 1], -s * local[:, 0] + c * local[:, 1]
                half = np.asarray(wire["dims_m"]) * 500.0
                inside = (np.abs(u) < half[0]) & (np.abs(v) < half[1]) & (np.abs(local[:, 2]) < half[2])
                self.assertFalse(bool(inside.any()), free[inside])
        for box in refresh.guard_boxes:
            with self.subTest(guard_box=box.name):
                turned = box.turned  # type: ignore[attr-defined]
                local = free - np.asarray(box.center_mm)
                c, s = math.cos(turned.yaw_rad), math.sin(turned.yaw_rad)
                u, v = c * local[:, 0] + s * local[:, 1], -s * local[:, 0] + c * local[:, 1]
                inside = ((np.abs(u) < turned.half_extents_mm[0]) & (np.abs(v) < turned.half_extents_mm[1])
                          & (np.abs(local[:, 2]) < turned.half_extents_mm[2]))
                self.assertFalse(bool(inside.any()), free[inside])

    def test_the_ur_planner_hands_the_same_boxes_to_the_sidecar_and_to_the_path_guard(self) -> None:
        """Through the UR glue: the sidecar gets them in its own base, half a turn about Z, and the guard in the
        controller's."""
        from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES, CuroboUrPlanner
        from src.robot.drivers.ur.planner_frame import planner_pose

        camera = looking_down(0.0, -430.0)
        depth = render(camera, self._SCENE)

        class _Camera:
            def grab_surface_depth(self) -> DepthSnapshot:
                return DepthSnapshot(depth_mm=depth, intrinsics=K, timestamp=__import__("time").time())

        class _Sidecar(_Recording):
            joint_names = list(UR_ARM_JOINT_NAMES)

            def start(self) -> None:
                return None

        class _Connection:
            is_connected = True

            def get_joint_positions(self) -> list[float]:
                return list(_Q)

        sidecar = _Sidecar()
        told: list = []
        world = LivePlannerWorld(
            cameras=(CameraView(name="look", depth_source=_Camera(), camera_to_base=camera),), declared=_SUPPORT,
            limits=LIMITS, tuning=WorldBuildTuning(pixel_stride=2), max_age_ms=5000.0,
        )
        frames = yawed_link_transforms_mm("ur10", np.asarray(_Q), 0.0)
        capsules = arm_capsules("ur10")
        assert frames is not None and capsules is not None
        planner = CuroboUrPlanner(
            _Connection(), client_factory=lambda: sidecar, live_world=world,  # type: ignore[arg-type]
            self_envelope=lambda: SelfEnvelope(frames_mm=tuple(frames), capsules=capsules),
            on_perceived_obstacles=told.append,
        )
        planner.refresh_world(near_point_mm=(600.0, 0.0, 300.0))

        (guard_boxes,) = told
        sent = [box for box in sidecar.worlds[-1] if box["name"].startswith("seen_")]
        self.assertGreater(len(sent), 1)
        expected = [dict(wire, pose=planner_pose(wire["pose"])) for wire in map(_as_wire, guard_boxes)]
        self.assertEqual(sent, expected)


# ---------------------------------------------------------------------------------------------------------------------
# The exact guard judges the turned box, at its own distance
# ---------------------------------------------------------------------------------------------------------------------


class _Backend:
    """An exact mesh backend that answers what it is told to and writes down every question."""

    engine = "fcl"

    def __init__(self, hits: "dict[int, tuple[str, float]] | None" = None) -> None:
        self.hits = hits or {}
        self.asked: list[dict[str, Any]] = []

    def evaluate(self, transforms: Any, yaw: float, fixtures: Any, min_distance_mm: float, broadphase: bool = False,
                 **keywords: Any) -> "tuple[str, float] | None":
        self.asked.append({"fixtures": tuple(fixtures), "min_distance_mm": float(min_distance_mm), **keywords})
        return self.hits.get(len(self.asked) - 1)


def _stubbed(backend: _Backend, **config: Any) -> SelfCollisionGuard:
    guard = _guard(**config)
    guard._fcl_backend = backend  # noqa: SLF001 (the backend stands in for the meshes)
    guard._fcl_backend_built = True  # noqa: SLF001
    guard._fcl_status = "ok"  # noqa: SLF001
    return guard


def _seen_box(name: str = "seen_00", *, centre: "tuple[float, float, float]" = (0.0, -300.0, 27.5),
              half: "tuple[float, float, float]" = (150.0, 20.0, 27.5), yaw_deg: float = 30.0) -> AxisAlignedBox:
    from src.robot.safety._capsule import TurnedBox

    yaw = math.radians(yaw_deg)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    enclosure = (half[0] * c + half[1] * s, half[0] * s + half[1] * c, half[2])
    return AxisAlignedBox(center_mm=np.asarray(centre), half_extents_mm=np.asarray(enclosure), name=name,
                          turned=TurnedBox(half_extents_mm=np.asarray(half), yaw_rad=yaw))  # type: ignore[call-arg]


class TheGuardKeepsItsOwnDistanceFromWhatACameraSawTests(unittest.TestCase):
    def test_a_seen_box_is_judged_at_5_mm_and_the_arm_and_the_declared_fixtures_at_10(self) -> None:
        """⛔ Before, one call at 10 mm judged the arm, the declared fixtures and the seen boxes alike."""
        backend = _Backend()
        declared = FixtureBoxConfig(name="bench_wall", center_mm=(500.0, 0.0, 100.0), half_extents_mm=(5.0, 300.0, 100.0))
        guard = _stubbed(backend, min_distance_mm=10.0, fixtures=[declared])
        guard.set_perceived_fixtures([_seen_box()])

        self.assertTrue(guard.evaluate(_ctx()).accepted)
        first, second = backend.asked
        self.assertEqual(first["min_distance_mm"], 10.0)
        self.assertEqual([box.name for box in first["fixtures"]], ["bench_wall"])
        self.assertTrue(first.get("arm_pairs", True), "the arm against itself was not judged")
        self.assertEqual(second["min_distance_mm"], 5.0)
        self.assertEqual([box.name for box in second["fixtures"]], ["seen_00"])
        self.assertFalse(second["arm_pairs"], "the arm against itself was judged at the seen-box distance")
        self.assertEqual(guard.perceived_min_distance_mm, 5.0)  # type: ignore[attr-defined]

    def test_without_a_seen_box_the_guard_asks_once_as_it_always_did(self) -> None:
        backend = _Backend()
        _stubbed(backend, min_distance_mm=10.0).evaluate(_ctx())
        self.assertEqual(len(backend.asked), 1)
        self.assertEqual(backend.asked[0]["min_distance_mm"], 10.0)
        self.assertNotIn("arm_pairs", backend.asked[0])

    def test_a_seen_box_is_kept_at_5_mm_and_the_same_box_declared_at_10_mm(self) -> None:
        """On the UR10's meshes: a slab 7 mm under the shoulder housing's lowest point."""
        guard = _guard(min_distance_mm=10.0)
        _needs_the_engine(self, guard)
        vertices = _upper_arm_vertices()
        lowest = vertices[int(np.argmin(vertices[:, 2]))]
        top = float(lowest[2]) - 7.0
        centre, half = (float(lowest[0]), float(lowest[1]), top / 2.0), (20.0, 20.0, top / 2.0)

        guard.set_perceived_fixtures([AxisAlignedBox(center_mm=np.asarray(centre), half_extents_mm=np.asarray(half),
                                                     name="seen_00")])
        self.assertTrue(guard.evaluate(_ctx()).accepted, "a seen box 7 mm off was refused at 10 mm")

        declared = _guard(min_distance_mm=10.0, fixtures=[FixtureBoxConfig(name="plate", center_mm=centre,
                                                                           half_extents_mm=half)])
        refused = declared.evaluate(_ctx())
        self.assertTrue(refused.rejected)
        self.assertEqual(refused.detail["pair"], "upper_arm|fixture:plate")
        self.assertAlmostEqual(float(refused.detail["signed_distance_mm"]), 7.0, delta=0.05)

    def test_the_exact_guard_judges_the_turned_box_and_not_the_box_around_it(self) -> None:
        """A bar turned 45 degrees beside the housing: its enclosure reaches under the housing, the bar does not."""
        guard = _guard(min_distance_mm=10.0)
        _needs_the_engine(self, guard)
        vertices = _upper_arm_vertices()
        lowest = vertices[int(np.argmin(vertices[:, 2]))]
        # The bar runs at 45 degrees through a point 80 mm out along x and in along -y from the lowest point, 20 mm
        # thick, 60 mm tall: its own faces stay 90 mm and more from the housing, while its enclosure, a 156 mm square
        # about its centre, reaches under the housing's lowest point.
        centre = (float(lowest[0]) + 80.0, float(lowest[1]) - 80.0, 30.0)
        bar = _seen_box(centre=centre, half=(100.0, 10.0, 30.0), yaw_deg=45.0)
        enclosure = AxisAlignedBox(center_mm=bar.center_mm, half_extents_mm=bar.half_extents_mm, name="seen_00")

        guard.set_perceived_fixtures([enclosure])
        self.assertTrue(guard.evaluate(_ctx()).rejected, "the control: the enclosure itself reaches the housing")
        guard.set_perceived_fixtures([bar])
        self.assertTrue(guard.evaluate(_ctx()).accepted, "the guard judged the enclosure of a turned box")

    def test_the_capsule_fallback_keeps_the_enclosure_at_the_guards_distance(self) -> None:
        """A guard that cannot turn a box holds the box around it, which is never smaller: the safe side."""
        guard = SelfCollisionGuard(SelfCollisionSafetyConfig(backend="capsule", kinematics_model="ur10",
                                                             min_distance_mm=10.0))
        # A bar turned 45 degrees 160 mm out along base -Y: its own faces stay 23 mm from the 80 mm base column, its
        # enclosure 2 mm.
        bar = _seen_box(centre=(0.0, -160.0, 50.0), half=(100.0, 10.0, 50.0), yaw_deg=45.0)
        guard.set_perceived_fixtures([bar])
        away = SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=_ARM,  # type: ignore[arg-type]
                             target_joints=JointPositions((math.pi, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0)))
        refused = guard.evaluate(away)
        self.assertTrue(refused.rejected)
        self.assertEqual(refused.detail["pair"], "base|fixture:seen_00")
        self.assertEqual(refused.detail["min_distance_mm"], f"{10.0:.6f}")

    def test_the_sphere_cull_never_changes_a_verdict_on_turned_boxes(self) -> None:
        from src.robot.safety._fcl_self_collision import make_backend
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        backend = make_backend("ur10")
        if backend is None:
            self.skipTest("no exact mesh engine or UR10 bundle on this box")
        rng = np.random.default_rng(19)
        for trial in range(40):
            joints = np.radians([rng.uniform(-180, 180), rng.uniform(-150, -30), rng.uniform(-150, 150),
                                 rng.uniform(-180, 0), rng.uniform(-120, 120), rng.uniform(-180, 180)])
            boxes = [_seen_box(f"seen_{k:02d}", centre=(rng.uniform(-900, 900), rng.uniform(-900, 900), 40.0),
                               half=(rng.uniform(10, 150), rng.uniform(10, 60), 40.0),
                               yaw_deg=float(rng.uniform(-90, 90))) for k in range(6)]
            transforms = ur_link_transforms_mm("ur10", joints)
            for limit in (5.0, 10.0):
                with self.subTest(trial=trial, limit=limit):
                    brute = backend.evaluate(transforms, 0.0, boxes, limit, broadphase=False, arm_pairs=False)  # type: ignore[call-arg]
                    culled = backend.evaluate(transforms, 0.0, boxes, limit, broadphase=True, arm_pairs=False)  # type: ignore[call-arg]
                    self.assertEqual(brute, culled)


# ---------------------------------------------------------------------------------------------------------------------
# The owner's bins beside the UR10's base
# ---------------------------------------------------------------------------------------------------------------------


class TheOwnersBinsBesideTheBaseTests(unittest.TestCase):
    """The investigation's table, rerun on an open bin 300 x 200 mm with 5 mm walls on the real pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.guard = _guard(min_distance_mm=10.0)

    def setUp(self) -> None:
        _needs_the_engine(self, self.guard)

    def _verdict(self, solids: "list[Solid]", centre_xy: "tuple[float, float]", guard: SelfCollisionGuard) -> Any:
        refresh = _refresh(solids, centre_xy)
        self.assertTrue(refresh.ok, refresh.render())
        guard.set_perceived_fixtures(refresh.guard_boxes)
        return guard.evaluate(_ctx())

    def test_a_bin_turned_30_degrees_48_mm_from_the_housing_passes(self) -> None:
        """⛔ Refused before at 4.973 mm < 10: the enclosure of the turned block reached under the housing."""
        reach = 150.0 * math.sin(math.radians(30.0)) + 100.0 * math.cos(math.radians(30.0))
        centre = (0.0, -260.0 - reach)
        turned = open_bin(centre, (300.0, 200.0), 40.0, yaw_deg=30.0)
        self.assertAlmostEqual(_ruler(turned), 48.3, delta=2.0)

        decision = self._verdict(turned, centre, self.guard)
        self.assertTrue(decision.accepted, decision.message)

    def test_the_investigations_square_bin_passes_at_5_mm_and_would_not_at_10(self) -> None:
        """⛔ Refused before at 5.980 mm < 10. The investigation's "25 mm" bin: its depth frame drew the rim at the
        bench plane's scale, 4.4 % inside the footprint it names, so the camera saw a rim 28.3 mm from the housing.
        Ray cast exactly here. Its rim meets the housing's round underside across the box's top edge, where 15 mm of
        margin on two faces reaches 21 mm: the guard reads 6.8 mm, and a rim 25.4 mm away reads 4.0 mm."""
        centre = (0.0, -364.4)
        square = open_bin(centre, (300.0, 200.0), 40.0)
        self.assertAlmostEqual(_ruler(square), 28.3, delta=0.5)

        self.assertTrue(self._verdict(square, centre, self.guard).accepted)
        stricter = _guard(min_distance_mm=10.0, perceived_min_distance_mm=10.0)
        self.assertTrue(self._verdict(square, centre, stricter).rejected, "the 5 mm is what admits it")

    def test_a_square_bin_25_mm_from_the_arm_is_refused_at_5_mm_and_one_26_mm_away_passes(self) -> None:
        """The plan's "a square bin with 25 mm real clearance passes at 5 mm", measured with the exact engine, mesh to
        solid (review of 2026-09-30): at 25.0 mm the guard reads 4.0 mm across the rim's top edge, where the 15 mm on
        two faces of the box reaches 21 mm, and refuses it; at 26.3 mm it passes. Kept at 5 mm, which keeps 20 mm off a
        face; about 3.5 mm would pass the 25 mm bin and keep 18.5 mm off a face, the owner's call."""
        from src.robot.safety._fcl_self_collision import make_backend
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        backend = make_backend("ur10")
        assert backend is not None
        transforms = ur_link_transforms_mm("ur10", np.asarray(_Q))

        def exact(solids: "list[Solid]") -> float:
            """The arm's exact distance to ``solids``: the least limit the engine finds a pair under."""
            boxes = [AxisAlignedBox(center_mm=np.asarray(s.centre), half_extents_mm=np.asarray(s.half), name=f"s{i}")
                     for i, s in enumerate(solids)]
            low, high = 0.0, 100.0
            for _ in range(30):
                middle = (low + high) / 2.0
                if backend.evaluate(transforms, 0.0, boxes, middle, arm_pairs=False) is None:  # type: ignore[call-arg]
                    low = middle
                else:
                    high = middle
            return (low + high) / 2.0

        for centre_y, clearance, passes in ((-360.05, 25.0, False), (-362.05, 26.3, True)):
            square = open_bin((0.0, centre_y), (300.0, 200.0), 40.0)
            with self.subTest(clearance_mm=clearance):
                self.assertAlmostEqual(exact(square), clearance, delta=0.1)
                decision = self._verdict(square, (0.0, centre_y), self.guard)
                self.assertEqual(decision.accepted, passes, decision.message)

    def test_a_47_mm_rim_under_the_housing_is_refused(self) -> None:
        """Physics stays: 6 mm of ruler under the housing is refused, whatever the camera world."""
        centre = (0.0, -300.0)
        tall = open_bin(centre, (300.0, 200.0), 47.0)
        self.assertLess(_ruler(tall), 8.0)

        decision = self._verdict(tall, centre, self.guard)
        self.assertTrue(decision.rejected)
        self.assertIn("fixture:seen_", decision.detail["pair"])


# ---------------------------------------------------------------------------------------------------------------------
# The invariant at load
# ---------------------------------------------------------------------------------------------------------------------


class TheRealSurfaceStaysClearBetweenSamplesTests(unittest.TestCase):
    """A path is sampled at ``min_distance_mm``; a seen box is its surface grown by the perceived margin. Between two
    samples the arm moves no more than the step, so the guard's distance to a seen box and the margin have to add up to
    the step, or the arm can touch what the camera saw between two samples that both passed."""

    def test_a_seen_box_distance_and_margin_short_of_the_step_is_refused_at_load(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            RobotSafetyConfig.model_validate({
                "self_collision": {"min_distance_mm": 10.0, "perceived_min_distance_mm": 5.0},
                "planning_world": {"perceived": {"margin_mm": 4.0}},
            })
        said = str(caught.exception)
        for key in ("perceived_min_distance_mm", "perceived.margin_mm", "min_distance_mm"):
            self.assertIn(key, said)

    def test_a_voxel_coarser_than_the_one_the_seen_distance_was_measured_at_is_refused_at_load(self) -> None:
        """At a 20 mm voxel a pixel the camera measured lay 5.6 mm outside every box, so a sample the guard passes at
        5 mm can touch it: while seen boxes are held nearer than the step, the thinning stays at the 10 it was measured
        at. Held at the step, the rule from before 2026-09-30 stands unchanged."""
        with self.assertRaises(ValidationError) as caught:
            RobotSafetyConfig.model_validate({"planning_world": {"perceived": {"voxel_size_mm": 20.0}}})
        said = str(caught.exception)
        for key in ("perceived.voxel_size_mm", "perceived_min_distance_mm", "min_distance_mm"):
            self.assertIn(key, said)
        for voxel in (10.0, 5.0):
            RobotSafetyConfig.model_validate({"planning_world": {"perceived": {"voxel_size_mm": voxel}}})
        RobotSafetyConfig.model_validate({"self_collision": {"perceived_min_distance_mm": 10.0},
                                          "planning_world": {"perceived": {"voxel_size_mm": 20.0}}})

    def test_the_shipped_numbers_add_up_to_the_step(self) -> None:
        config = RobotSafetyConfig()
        self.assertEqual(config.self_collision.perceived_min_distance_mm, 5.0)  # type: ignore[attr-defined]
        self.assertGreaterEqual(
            config.self_collision.perceived_min_distance_mm + config.planning_world.perceived.margin_mm,  # type: ignore[attr-defined]
            config.self_collision.min_distance_mm)
        RobotSafetyConfig.model_validate({"self_collision": {"min_distance_mm": 10.0, "perceived_min_distance_mm": 5.0},
                                          "planning_world": {"perceived": {"margin_mm": 5.0}}})
        preflight = SafetyPreflight.from_safety_config(
            RobotSafetyConfig.model_validate({"self_collision": {"backend": "capsule"}}),
            SimpleNamespace(x_min=-1000.0, x_max=1000.0, y_min=-1000.0, y_max=1000.0, z_min=-100.0, z_max=1000.0),  # type: ignore[arg-type]
        )
        self.assertEqual(preflight.path_step_mm, config.self_collision.min_distance_mm)

    def test_the_shipped_cell_leaves_room_for_the_height_map(self) -> None:
        from src.config.loader import load_robot_section
        from src.robot.safety.planning.reservation import PlannerReservation

        from src.config.schema.robot import RobotConfig

        robot = load_robot_section(None, profile="ur10")
        self.assertEqual(robot.safety.self_collision.perceived_min_distance_mm, 5.0)  # type: ignore[attr-defined]
        self.assertEqual(robot.safety.planning_world.perceived.max_boxes, 64)
        self.assertEqual(WorldBuildTuning().max_boxes, 64)
        # The planner reserves a slot for every box the world may send: the bench, and the 64.
        cell = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur10", "motion_planner": "curobo"}, "gripper": {"model": "robotiq_hande"},
            "safety": {"self_collision": {"planner_margin_mm": 4.0}, "planning_world": {
                "enabled": True, "support_plane": {"height_mm": 0.0}}},
        })
        self.assertEqual(PlannerReservation.from_config(robot_cfg=cell).cuboid_slots, 1 + 64)


# ---------------------------------------------------------------------------------------------------------------------
# A refusal says where the box is and where the arm stood
# ---------------------------------------------------------------------------------------------------------------------


class ASeenBoxRefusalSaysWhereTests(unittest.TestCase):
    def _refusing(self) -> SelfCollisionGuard:
        guard = _stubbed(_Backend(hits={1: ("upper_arm|fixture:seen_00", 2.5)}), min_distance_mm=10.0)
        guard.set_perceived_fixtures([_seen_box(centre=(0.0, -300.0, 27.5), half=(150.0, 20.0, 27.5), yaw_deg=30.0)])
        return guard

    def test_the_refusal_carries_the_box_and_the_joints(self) -> None:
        refused = self._refusing().evaluate(_ctx())

        self.assertTrue(refused.rejected)
        self.assertEqual(refused.message, "upper_arm|fixture:seen_00: mesh distance 2.500 mm < 5.000 mm")
        detail = refused.detail
        self.assertEqual(detail["min_distance_mm"], f"{5.0:.6f}")
        self.assertEqual(detail["fixture"], "seen")
        self.assertEqual(detail["joints_deg"], "(0.0, -60.0, 80.0, -110.0, -90.0, 0.0)")
        self.assertEqual(detail["box_yaw_deg"], "30.0")
        self.assertEqual(detail["box_size_mm"], "300.0 x 40.0 x 55.0")
        self.assertEqual(detail["box_centre_mm"], "(0.0, -300.0, 27.5)")
        c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        corners = [(u * c - v * s, -300.0 + u * s + v * c) for u, v in ((-150, -20), (150, -20), (150, 20), (-150, 20))]
        self.assertEqual(detail["box_corners_mm"],
                         " ".join(f"({x:.1f}, {y:.1f})" for x, y in corners) + ", z 0.0 to 55.0")

    def test_the_path_gate_and_the_joint_move_gate_log_it(self) -> None:
        from src.robot.safety.path_samples import PathSamples

        preflight = SafetyPreflight([self._refusing()])
        with self.assertLogs("SafetyPreflight", level=logging.WARNING) as said:
            self.assertIsNotNone(preflight.gate_joint_target(JointPositions(_Q), arm=_ARM))  # type: ignore[arg-type]
        self.assertIn("box_corners_mm", said.output[-1])
        self.assertIn("(0.0, -60.0, 80.0, -110.0, -90.0, 0.0)", said.output[-1])

        preflight = SafetyPreflight([self._refusing()])
        with self.assertLogs("SafetyPreflight", level=logging.WARNING) as said:
            self.assertIsNotNone(preflight.gate_joint_path(PathSamples(configs=(_Q,), step_bound_mm=10.0), arm=_ARM))  # type: ignore[arg-type]
        self.assertIn("sample 1 of 1", said.output[-1])
        self.assertIn("box_yaw_deg", said.output[-1])
        self.assertIn("(0.0, -60.0, 80.0, -110.0, -90.0, 0.0)", said.output[-1])


if __name__ == "__main__":
    unittest.main()
