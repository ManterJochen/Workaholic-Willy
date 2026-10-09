"""The whole-path judge measures the arm against itself, and a part against a box, only where it could refuse.

On the owner's cell a route of 905 samples cost the exact guard 3,142 queries of the arm against itself, every one far
from the guard's 3 mm: forearm|wrist_2 never under 18.3 mm, wrist_2|camera never under 42.7. The boxes had been bounded
since a84f755: a part measured against a box keeps that distance less how far any point of it has moved since. The
whole-path judge (``MeshSelfCollisionBackend.first_suspect``) holds the arm's own pairs to the same rule, in one part's
frame, where only the other moves.

What this file pins:

* the bound itself on the owner's guard: over seeded pairs of configurations, every pair the guard checks keeps at the
  second at least its distance at the first less the bound (``_pair_motion``), and every part against a box less how
  far the part moved (``_moved_mm``), to 1e-9 mm;
* the walk (``_first_flagged``): the first sample its spheres do not clear is measured, a sample the bound clears is not,
  and the first under the limit is flagged, never a later one;
* the chain the judge places every sample with (``ur_link_transforms_mm_many``) is bit for bit the chain one sample is
  placed with, on 20,000 seeded configurations of the cell's UR10 and 2,000 of every other bundled model;
* on the cell's routes of 2026-10-07 the arm is measured against itself at most 30 times, and the boxes at most 30.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from typing import Any

import numpy as np

from src.robot.safety._fcl_self_collision import _first_flagged, _moved_mm
from src.robot.safety._ur_kinematics import UR_DH_TABLES_M, ur_link_transforms_mm, ur_link_transforms_mm_many
from src.robot.safety.path_samples import waypoint_path_samples
from tests import _owner_guard as owner
from tests.test_a_faster_camera_world_builds_the_same_world import frame

#: How far a bound may lie past the distance it bounds before it is no bound, millimetres: float rounding, nothing more.
_ROUNDING_MM = 1e-9


def _backend() -> Any:
    preflight, arm = owner.owner_cell(whole=True)
    guard = preflight._path_authority(arm)
    assert guard is not None
    return guard._exact_mesh_backend("ur10"), preflight, arm


def _distance(backend: Any, frames: np.ndarray, part: str, other: Any) -> float:
    placed = frames[backend._frame[part]]
    backend._a.set_transform(backend._models[part], placed[:3, :3], placed[:3, 3])
    return float(backend._a.distance(backend._models[part], other))


class TheBoundHoldsTests(unittest.TestCase):
    def setUp(self) -> None:
        owner.needs_the_engine()

    def test_no_pair_the_guard_checks_comes_nearer_than_its_distance_less_the_bound(self) -> None:
        backend, _, _ = _backend()
        rng = np.random.default_rng(11)
        checked = 0
        for _ in range(120):
            first = rng.uniform(-math.pi, math.pi, 6)
            second = first + rng.uniform(-0.3, 0.3, 6)
            frames = ur_link_transforms_mm_many("ur10", np.vstack([first, second]))
            assert frames is not None
            placed = backend._placements(frames, 0.0)
            for index, (i, j) in enumerate(backend._pairs):
                part_i, part_j = backend._names[i], backend._names[j]
                then = max(0.0, backend.distance_mm(list(frames[0]), 0.0, part_i, part_j))
                now = max(0.0, backend.distance_mm(list(frames[1]), 0.0, part_i, part_j))
                moved = float(backend._pair_motion(placed, i, j)(0, np.array([1]))[0])
                self.assertGreaterEqual(now, then - moved - _ROUNDING_MM, f"{part_i}|{part_j}")
                checked += 1
        self.assertGreater(checked, 120 * 20, "the premise: the owner's guard checks the arm's pairs")

    def test_no_part_comes_nearer_a_box_than_its_distance_less_how_far_it_moved(self) -> None:
        backend, _, _ = _backend()
        rng = np.random.default_rng(12)
        look = np.radians(owner.HOME_DEG)
        for _ in range(60):
            first = look + rng.uniform(-0.4, 0.4, 6)
            second = first + rng.uniform(-0.2, 0.2, 6)
            frames = ur_link_transforms_mm_many("ur10", np.vstack([first, second]))
            assert frames is not None
            placed = backend._placements(frames, 0.0)
            for row, part in enumerate(backend._names):
                centre = placed.centres[0, row] + rng.uniform(-150.0, 150.0, 3)
                box = backend._a.box_object(rng.uniform(5.0, 60.0, 3), centre, float(rng.uniform(-1.0, 1.0)))
                then = max(0.0, _distance(backend, frames[0], part, box))
                now = max(0.0, _distance(backend, frames[1], part, box))
                moved = float(_moved_mm(placed.centres[0, row], placed.turns[0, row], placed.centres[1:, row],
                                        placed.turns[1:, row], float(backend._radius_row[row]))[0])
                self.assertGreaterEqual(now, then - moved - _ROUNDING_MM, part)

    def test_the_bound_is_how_far_a_point_of_the_part_can_move(self) -> None:
        """A part turned half a turn about its sphere's centre: every point of it moves at most its diameter."""
        turn = np.diag([-1.0, -1.0, 1.0])
        moved = _moved_mm(np.zeros(3), np.eye(3), np.zeros((1, 3)), turn[None], 10.0)
        self.assertAlmostEqual(20.0, float(moved[0]))
        self.assertAlmostEqual(5.0, float(_moved_mm(np.zeros(3), np.eye(3), np.array([[3.0, 4.0, 0.0]]),
                                                    np.eye(3)[None], 10.0)[0]))
        for angle in (0.3, 1e-4, 1e-9):
            c, s = math.cos(angle), math.sin(angle)
            turned = np.array([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]])
            moved = float(_moved_mm(np.zeros(3), np.eye(3), np.zeros((1, 3)), turned, 382.0)[0])
            self.assertGreaterEqual(moved, 382.0 * 2.0 * math.sin(angle / 2.0) - 1e-15, angle)
            self.assertLess(moved, 382.0 * 2.0 * math.sin(angle / 2.0) + 1e-9, angle)

    def test_a_part_that_barely_turns_is_still_bounded(self) -> None:
        """1e-9 rad on the forearm's 382 mm sphere is 3.8e-7 mm: the trace of the turn rounds to 3 and would bound
        nothing; the chord from the difference keeps it."""
        c, s = math.cos(1e-9), math.sin(1e-9)
        turned = np.array([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]])
        self.assertEqual(3.0, float(np.trace(turned[0])), "the premise: the trace rounds to 3")
        self.assertGreater(float(_moved_mm(np.zeros(3), np.eye(3), np.zeros((1, 3)), turned, 382.0)[0]), 3.8e-7)


class TheWalkTests(unittest.TestCase):
    """``_first_flagged`` over a pair whose distance is known at every sample, 1 mm a sample nearer from 20.5 mm."""

    def _walk(self, near: np.ndarray, distances: np.ndarray, limit: float = 3.0,
              per_sample_mm: float = 1.0) -> "tuple[int, list[int]]":
        measured: list[int] = []

        def measure(sample: int) -> float:
            measured.append(sample)
            return float(distances[sample])

        def moved(sample: int, later: np.ndarray) -> np.ndarray:
            # The pair comes at most per_sample_mm a sample nearer.
            return per_sample_mm * np.abs(later - sample).astype(np.float64)

        return _first_flagged(near, len(distances), limit, measure, moved), measured

    def test_the_first_sample_under_the_limit_is_flagged_and_what_the_bound_clears_is_not_measured(self) -> None:
        distances = 20.5 - np.arange(30, dtype=np.float64)
        flagged, measured = self._walk(np.ones(30, dtype=bool), distances)
        self.assertEqual(18, flagged, "20.5 - 18 = 2.5 mm, the first under 3")
        self.assertEqual([0, 18], measured, "20.5 mm at 0 clears 1 to 17 (3.5 mm at 17); 2.5 at 18 flags")

    def test_a_long_path_is_walked_a_window_at_a_time_and_nothing_cleared_is_measured(self) -> None:
        distances = 2000.5 - np.arange(2100, dtype=np.float64)
        flagged, measured = self._walk(np.ones(2100, dtype=bool), distances)
        self.assertEqual(1998, flagged, "2000.5 - 1998 = 2.5 mm, the first under 3, four windows on")
        self.assertEqual([0, 1998], measured)

    def test_a_sample_the_spheres_clear_is_never_measured(self) -> None:
        distances = 20.5 - np.arange(30, dtype=np.float64)
        near = np.zeros(30, dtype=bool)
        flagged, measured = self._walk(near, distances)
        self.assertEqual(30, flagged)
        self.assertEqual([], measured)

    def test_a_distance_within_the_slack_of_the_limit_is_flagged(self) -> None:
        distances = np.array([10.0, 3.0 + 1e-7, 10.0])
        flagged, _ = self._walk(np.ones(3, dtype=bool), distances, per_sample_mm=10.0)
        self.assertEqual(1, flagged, "the gate decides a sample this near; the walk never passes it")

    def test_a_distance_the_engine_cannot_give_is_flagged(self) -> None:
        flagged, _ = self._walk(np.ones(3, dtype=bool), np.array([10.0, math.nan, 10.0]), per_sample_mm=10.0)
        self.assertEqual(1, flagged)


class TheChainOfManyIsTheChainOfEachTests(unittest.TestCase):
    def test_every_frame_of_20000_configurations_is_bit_for_bit_the_frame_of_each(self) -> None:
        rng = np.random.default_rng(20261009)
        for model in UR_DH_TABLES_M:
            count = 20_000 if model == "ur10" else 2_000
            configs = rng.uniform(-2.0 * math.pi, 2.0 * math.pi, (count, 6))
            many = ur_link_transforms_mm_many(model, configs)
            assert many is not None
            each = np.asarray([np.asarray(ur_link_transforms_mm(model, q)) for q in configs])
            self.assertTrue(np.array_equal(many, each), model)

    def test_a_model_or_a_table_the_chain_cannot_place_is_none(self) -> None:
        self.assertIsNone(ur_link_transforms_mm_many("kr6", np.zeros((3, 6))))
        self.assertIsNone(ur_link_transforms_mm_many("ur10", np.zeros((3, 7))))
        self.assertIsNone(ur_link_transforms_mm_many("ur10", np.zeros(6)))
        empty = ur_link_transforms_mm_many("ur10", np.zeros((0, 6)))
        assert empty is not None
        self.assertEqual((0, 7, 4, 4), empty.shape)


class TheCellsRoutesTests(unittest.TestCase):
    """The route from each cell frame's look to 100 mm over its target, as the guard judges it whole."""

    def setUp(self) -> None:
        owner.needs_the_engine()

    def test_the_arm_is_measured_against_itself_at_most_30_times_a_route(self) -> None:
        backend, preflight, arm = _backend()
        guard = preflight._path_authority(arm)
        assert guard is not None
        reach = preflight.joint_radii_mm(arm)
        assert reach is not None
        arm_parts = set(backend._names)
        asked: Counter[str] = Counter()
        measure = backend._a.distance

        def counting(a: Any, b: Any) -> float:
            held = {id(obj): name for name, obj in backend._models.items()}
            held.update({id(obj): name for name, obj in (backend._hulls or {}).items()})
            asked["arm" if held.get(id(a)) in arm_parts and held.get(id(b)) in arm_parts else "box"] += 1
            return float(measure(a, b))

        backend._a.distance = counting
        try:
            for name in ("F1", "F2", "F3"):
                recorded = frame(name)
                preflight.set_perceived_obstacles(owner.seen_world(name, "A"))
                standoff = owner.nearest_solution(np.asarray(recorded["goal"]), recorded["down_rad"])
                assert standoff is not None
                route = waypoint_path_samples([list(recorded["look_rad"]), standoff.tolist()], reach_mm=reach,
                                              max_step_mm=owner.GUARD_MM)
                asked.clear()
                configs = np.asarray(route.configs, dtype=np.float64)
                self.assertEqual(len(configs), guard.first_suspect(arm, configs, len(configs)), f"{name}: clear")
                self.assertGreater(len(configs), 700, "the premise: a route of the cell's length")
                self.assertLessEqual(asked["arm"], 30, name)
                self.assertGreater(asked["arm"], 0, f"{name}: the control, the wrist pairs are measured")
                self.assertLessEqual(asked["box"], 30, name)
        finally:
            del backend._a.distance


if __name__ == "__main__":
    unittest.main()
