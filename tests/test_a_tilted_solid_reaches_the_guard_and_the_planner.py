"""A support's solid tilts, and the planner and the exact guard hold it tilted (fix plan Track S, test 3).

The camera reads the owner's mat a degree off level, and an upright box round a tilted reading stood up to 14 mm over it
at its low corner. So a support surface's solid tilts with its local reading: the planner gets its whole turn as the
pose's quaternion, the exact guard builds it with that turn, and the capsule proxy, which cannot turn a box, judges the
axis-aligned box that encloses it. Every other box stays turned about base Z alone, byte for byte as before. The
guard's broadphase sphere is the enclosure's, so it holds the tilted box too and never culls one the exact query would
refuse (attack test 6).
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from src.robot.safety.planning.live_world import _guard_boxes
from src.robot.safety.planning.perceived import PerceivedBox, PerceivedWorld
from src.robot.safety.planning.world import PlanningWorldError, planner_cuboid
from src.robot.safety.self_collision import SelfCollisionGuard
from src.config.schema.robot.safety_schema import SelfCollisionSafetyConfig


def _turn(yaw_deg: float, tilt_x_deg: float, tilt_y_deg: float = 0.0) -> np.ndarray:
    yaw, tx, ty = (math.radians(v) for v in (yaw_deg, tilt_x_deg, tilt_y_deg))
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(tx), -math.sin(tx)], [0.0, math.sin(tx), math.cos(tx)]])
    ry = np.array([[math.cos(ty), 0.0, math.sin(ty)], [0.0, 1.0, 0.0], [-math.sin(ty), 0.0, math.cos(ty)]])
    return rz @ ry @ rx


def _matrix(wxyz: "list[float]") -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _solid(rotation: np.ndarray, centre: "tuple[float, float, float]" = (-150.0, -650.0, 30.0),
           dims: "tuple[float, float, float]" = (300.0, 200.0, 70.0)) -> PerceivedBox:
    return PerceivedBox(name="seen_s00_support0", center_mm=centre, dims_mm=dims,
                        yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]), points=0, distance_mm=0.0,
                        rotation=tuple(float(v) for v in rotation.reshape(-1)), kind="support")


def _backend(test: unittest.TestCase) -> Any:
    guard = SelfCollisionGuard(SelfCollisionSafetyConfig(kinematics_model="ur10"))
    backend = guard._exact_mesh_backend("ur10")  # noqa: SLF001
    if backend is None:
        test.skipTest("no exact mesh engine or UR10 bundle on this box")
    return backend


class ThePlannerGetsTheWholeTurnTests(unittest.TestCase):
    def test_the_quaternion_is_the_solids_turn(self) -> None:
        for yaw, tx, ty in ((0.0, 1.0, 0.0), (30.0, 3.0, -2.0), (-120.0, 4.9, 4.9), (179.0, -5.0, 1.0)):
            with self.subTest(yaw=yaw, tilt=(tx, ty)):
                turn = _turn(yaw, tx, ty)
                wire = planner_cuboid("seen_s00_support0", (100.0, -600.0, 30.0), (300.0, 200.0, 70.0),
                                      yaw_rad=math.radians(yaw), rotation=turn.reshape(-1).tolist())
                quaternion = wire["pose"][3:]
                self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)
                self.assertGreaterEqual(quaternion[0], 0.0)
                np.testing.assert_allclose(_matrix(quaternion), turn, atol=1e-12)
                self.assertEqual(wire["pose"][:3], [0.1, -0.6, 0.03])
                self.assertEqual(wire["dims_m"], [0.3, 0.2, 0.07])

    def test_a_box_turned_about_z_is_written_as_before(self) -> None:
        wire = planner_cuboid("seen_03", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), yaw_rad=0.3)
        self.assertEqual(wire["pose"], [0.001, 0.002, 0.003, math.cos(0.15), 0.0, 0.0, math.sin(0.15)])

    def test_what_is_no_rotation_is_refused(self) -> None:
        for wrong in (np.diag([1.0, 1.0, -1.0]), np.full((3, 3), 0.5), [1.0, 0.0, 0.0]):
            with self.subTest(wrong=wrong), self.assertRaises(PlanningWorldError):
                planner_cuboid("seen_s00_support0", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
                               rotation=np.asarray(wrong, dtype=np.float64).reshape(-1).tolist())

    def test_the_live_world_writes_the_solids_turn(self) -> None:
        from src.robot.safety.planning.live_world import planner_cuboid as wired  # what world_for writes with

        turn = _turn(20.0, 2.0)
        box = _solid(turn)
        wire = wired(box.name, box.center_mm, box.dims_mm, yaw_rad=box.yaw_rad, rotation=box.rotation)
        np.testing.assert_allclose(_matrix(wire["pose"][3:]), turn, atol=1e-12)


class TheGuardHoldsItTiltedTests(unittest.TestCase):
    def test_the_exact_engine_measures_a_tilted_box_as_the_analytic_one(self) -> None:
        engine = _backend(self)._a  # noqa: SLF001
        rng = np.random.default_rng(3)
        for index in range(40):
            turn = _turn(*rng.uniform((-180.0, -5.0, -5.0), (180.0, 5.0, 5.0)))
            centre = rng.uniform((-300.0, -800.0, 0.0), (300.0, -400.0, 60.0))
            half = rng.uniform((50.0, 50.0, 20.0), (200.0, 150.0, 40.0))
            solid = engine.box_object(half, centre, 0.0, turn)
            # Outside: past a corner, or past one face within the other two.
            side = rng.choice((-1.0, 1.0), 3)
            local = side * (half + rng.uniform(1.0, 30.0, 3))
            if index % 2:
                axis = int(rng.integers(3))
                inner = rng.uniform(-0.9, 0.9, 3) * half
                inner[axis] = local[axis]
                local = inner
            point = centre + turn @ local
            local = np.abs((point - centre) @ turn) - half
            analytic = float(np.linalg.norm(np.maximum(local, 0.0)) + min(float(local.max()), 0.0))
            probe = engine.box_object(np.full(3, 1e-4), point)
            self.assertAlmostEqual(engine.distance(solid, probe), analytic, delta=1e-3)

    def test_the_capsule_proxy_judges_the_box_that_encloses_it(self) -> None:
        turn = _turn(35.0, 4.0, -3.0)
        box = _solid(turn)
        world = PerceivedWorld(boxes=(box,), dropped_points={}, dropped_clusters={}, considered_points=0)
        (guard_box,) = _guard_boxes(world)
        assert guard_box.turned is not None and guard_box.turned.rotation is not None
        np.testing.assert_allclose(guard_box.turned.rotation, turn)
        half = np.asarray(box.dims_mm) / 2.0
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        placed = np.asarray(box.center_mm) + corners @ turn.T
        inside = np.abs(placed - guard_box.center_mm) <= np.asarray(guard_box.half_extents_mm) + 1e-9
        self.assertTrue(bool(inside.all()), "a corner of the tilted solid lies outside the box the proxy judges")
        np.testing.assert_allclose(guard_box.half_extents_mm, np.abs(turn) @ half)

    def test_the_broadphase_never_culls_a_tilted_solid_the_exact_query_refuses(self) -> None:
        backend = _backend(self)
        rng = np.random.default_rng(11)
        refused = 0
        joints = np.radians((-90.0, -100.0, -110.0, -60.0, 90.0, 0.0))
        transforms = ur_link_transforms_mm("ur10", joints)
        assert transforms is not None
        wrist = np.asarray(transforms[6])[:3, 3]
        for _ in range(60):
            turn = _turn(*rng.uniform((-180.0, -5.0, -5.0), (180.0, 5.0, 5.0)))
            half = rng.uniform((20.0, 20.0, 10.0), (220.0, 160.0, 40.0))
            centre = wrist + rng.uniform(-250.0, 250.0, 3)
            box = _solid(turn, (float(centre[0]), float(centre[1]), float(centre[2])),
                         (float(2.0 * half[0]), float(2.0 * half[1]), float(2.0 * half[2])))
            fixtures = _guard_boxes(PerceivedWorld(boxes=(box,), dropped_points={}, dropped_clusters={},
                                                   considered_points=0))
            for limit in (5.0, 10.0):
                culled = backend.evaluate(transforms, 0.0, fixtures, limit, broadphase=True, arm_pairs=False)
                brute = backend.evaluate(transforms, 0.0, fixtures, limit, broadphase=False, arm_pairs=False)
                self.assertEqual(culled is None, brute is None)
                refused += brute is not None
        self.assertGreater(refused, 10, "the sample never came near the arm")


if __name__ == "__main__":
    unittest.main()
