"""SFE's builds got cheaper and answer to the bit: the cross product of two 3-vectors and the floor the guard holds.

Profiled on the owner's boxed-in part (2026-10-08), numpy's general ``np.cross`` was a quarter of the search and the
floor read every solid again on every call another 29 %. ``support_footprint._cross3`` does the same three products and
differences in the same order, and ``HandFloor`` reads its solids once when it is made; both answer what they answered
before, every bit, a signed zero included. A floor that crosses into a worker process is read again there, as it is
read here.
"""

from __future__ import annotations

import math
import pickle
import unittest

import numpy as np

from src.robot.grasping.generation.support_footprint import HandFloor, _cross3
from src.robot.safety.planning.support_surfaces import SupportSolid


def _turn(rng: np.random.Generator, tilt_deg: float) -> np.ndarray:
    """A rotation about z, then tipped ``tilt_deg`` about x."""
    yaw = float(rng.uniform(-math.pi, math.pi))
    tip = math.radians(tilt_deg)
    about_z = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(tip), -math.sin(tip)], [0.0, math.sin(tip), math.cos(tip)]])
    return about_z @ about_x


def _solid(rng: np.random.Generator, *, tilt_deg: float = 0.0) -> SupportSolid:
    return SupportSolid(name="s", kind="support", surface=0,
                        centre_mm=rng.uniform([-60.0, -60.0, 20.0], [60.0, 60.0, 60.0]),
                        half_extents_mm=rng.uniform([20.0, 20.0, 5.0], [200.0, 200.0, 30.0]),
                        rotation=_turn(rng, tilt_deg), local_plane=np.array([50.0, 0.0, 0.0]),
                        excess_mm=float(rng.uniform(0.0, 3.0)), band_mm=float(rng.uniform(0.0, 4.0)),
                        allowance_mm=float(rng.uniform(0.0, 3.0)))


def _at_as_before(floor: HandFloor, xy: np.ndarray, *, fingers: bool = False) -> np.ndarray:
    """``HandFloor.at`` as it read its solids on every call, before they were read once."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.full(points.shape[0], -math.inf)
    for solid in floor.solids:
        centre = np.asarray(solid.centre_mm, dtype=np.float64).reshape(3)
        turn = np.asarray(solid.rotation, dtype=np.float64).reshape(3, 3)
        half = np.asarray(solid.half_extents_mm, dtype=np.float64).reshape(3)
        up = turn[:, 2]
        if up[2] <= 1e-6:
            continue
        offset = points - centre[:2]
        z = centre[2] + (half[2] - offset @ up[:2]) / up[2]
        local = np.column_stack([offset, z - centre[2]]) @ turn
        over = np.all(np.abs(local[:, :2]) <= half[:2] + 1e-9, axis=1)
        drop = floor._drop(solid, fingers) / up[2]
        out[over] = np.maximum(out[over], z[over] - drop)
    return out + float(floor.distance_mm)


def _highest_as_before(floor: HandFloor) -> float:
    tops = [float(np.asarray(solid.centre_mm, dtype=np.float64)[2]
                  + (np.abs(np.asarray(solid.rotation, dtype=np.float64).reshape(3, 3))
                     @ np.asarray(solid.half_extents_mm, dtype=np.float64))[2]) for solid in floor.solids]
    return max(tops) + float(floor.distance_mm) if tops else -math.inf


def _under_as_before(floor: HandFloor, points: np.ndarray, *, fingers: bool = False) -> bool:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[pts[:, 2] < _highest_as_before(floor)]
    return bool(pts.shape[0]) and bool((pts[:, 2] < _at_as_before(floor, pts[:, :2], fingers=fingers)).any())


class TheCrossProductTests(unittest.TestCase):
    def test_every_bit_is_numpys_signed_zeros_and_extremes_included(self) -> None:
        rng = np.random.default_rng(20261008)
        special = np.array([0.0, -0.0, 1.0, -1.0, 1e-308, -5e-324, 1e300, -1e300, 0.5, 3.0])
        pairs = [(rng.standard_normal(3), rng.standard_normal(3)) for _ in range(4000)]
        pairs += [(rng.choice(special, 3), rng.choice(special, 3)) for _ in range(4000)]
        pairs += [(np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])),
                  (np.array([-0.0, 0.0, -1.0]), np.array([0.0, -0.0, 0.0]))]
        for a, b in pairs:
            with np.errstate(over="ignore", invalid="ignore"):
                expected = np.cross(a, b)
            got = _cross3(a, b)
            self.assertEqual(expected.dtype, got.dtype)
            self.assertEqual(expected.shape, got.shape)
            self.assertEqual(expected.tobytes(), got.tobytes(), (a, b))

    def test_a_unit_axis_and_a_tilted_approach_as_sfe_builds_them(self) -> None:
        for angle in np.radians(np.arange(0.0, 360.0, 7.5)):
            axis = np.array([math.cos(angle), math.sin(angle), 0.0])
            for tilt in np.radians((0.0, 7.5, 15.0, 45.0, 90.0)):
                approach = np.array([math.sin(tilt) * -axis[1], math.sin(tilt) * axis[0], -math.cos(tilt)])
                self.assertEqual(np.cross(approach, axis).tobytes(), _cross3(approach, axis).tobytes())


class TheFloorReadOnceTests(unittest.TestCase):
    def _floors(self) -> list[HandFloor]:
        rng = np.random.default_rng(7)
        floors = [HandFloor(solids=(), distance_mm=3.0)]
        for fingers in (False, True):
            for drop in (0.0, 1.0, -2.0):
                solids = tuple(_solid(rng, tilt_deg=tilt) for tilt in (0.0, 2.0, 12.0, 89.9999999, 95.0))
                floors.append(HandFloor(solids=solids, distance_mm=float(rng.uniform(1.0, 5.0)),
                                        fingers_to_the_reading=fingers, finger_floor_drop_mm=drop))
        return floors

    def test_the_floor_and_the_highest_top_are_the_bits_of_before(self) -> None:
        rng = np.random.default_rng(11)
        for floor in self._floors():
            self.assertEqual(np.float64(_highest_as_before(floor)).tobytes(), np.float64(floor.highest_mm).tobytes())
            xy = rng.uniform(-250.0, 250.0, (500, 2))
            for fingers in (False, True):
                self.assertEqual(_at_as_before(floor, xy, fingers=fingers).tobytes(),
                                 floor.at(xy, fingers=fingers).tobytes())

    def test_under_says_what_it_said_before(self) -> None:
        rng = np.random.default_rng(13)
        for floor in self._floors():
            for _ in range(60):
                points = rng.uniform([-250.0, -250.0, 0.0], [250.0, 250.0, 120.0], (int(rng.integers(1, 40)), 3))
                for fingers in (False, True):
                    self.assertEqual(_under_as_before(floor, points, fingers=fingers),
                                     floor.under(points, fingers=fingers))

    def test_a_floor_pickled_into_another_process_is_read_again_and_answers_the_same_bits(self) -> None:
        rng = np.random.default_rng(17)
        xy = rng.uniform(-250.0, 250.0, (500, 2))
        for floor in self._floors():
            again = pickle.loads(pickle.dumps(floor))
            self.assertEqual(floor.solids.__len__(), again.solids.__len__())
            self.assertEqual(np.float64(floor.highest_mm).tobytes(), np.float64(again.highest_mm).tobytes())
            for fingers in (False, True):
                self.assertEqual(floor.at(xy, fingers=fingers).tobytes(), again.at(xy, fingers=fingers).tobytes())
            for (_, turn, _, up, _), (_, turn_again, _, up_again, _) in zip(floor._read, again._read):
                # Read again where it was unpickled: the up is a view of the turn there too, as here.
                self.assertIs(up.base, turn.base if turn.base is not None else turn)
                self.assertIs(up_again.base, turn_again.base if turn_again.base is not None else turn_again)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
