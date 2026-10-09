"""SFE's units, run in any order and in any process, give the search its own answer, to the bit.

The owner's cell, 2026-10-08: one boxed-in cube took 24 of a look's 30 s, every grasp SFE tried built one after another.
A part's search is now planned once (``plan_support_footprint``), cut into units, one closing line at one height each
(``SfeUnit``: search, axis, place, height), run by any process (``run_units``), merged in unit order and ranked
(``rank_found``); the fine and the rolled searches run only where the merged coarse grasps ask for them, and the fine
pass waits where the caller lets it. ``generate_support_footprint_grasps`` runs every unit here, in order, with no
runner, and hands them to one where it is given (``runner``, the cell's worker pool). Here the units of five scenes run
shuffled, in random chunks, each chunk on a plan of its own as a worker plans it, and give the function's grasps, every
field bit for bit, and its counts; a runner that cannot answer, or answers only part, has the part searched here and
keeps nothing it answered.
"""

from __future__ import annotations

import random
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.generation.scene_obstacles import SeenEnvelope, corridor_seen_in
from src.robot.grasping.generation.support_footprint import (
    COARSE,
    FINE,
    FINE_SEARCH_DEFERRED,
    REFUSAL_CAUSES,
    ROLLED,
    HandFloor,
    SfeInputs,
    SfeRunnerFailed,
    generate_support_footprint_grasps,
    plan_support_footprint,
    rank_found,
    run_units,
)
from src.robot.safety.planning.perceived import SeenBox
from src.robot.safety.planning.support_surfaces import SupportSolid
from tests.test_a_side_grasp_never_goes_through_unseen_space import (
    CYLINDER,
    LOW_ON_MINUS_Y,
    NEIGHBOUR,
    WALL,
    camera,
)
from tests.test_sfe_says_why_it_refused import (
    MAT_MM,
    box_cloud,
    boxed_in_bar,
    cylinder_beside_a_wall,
    cylinder_cloud,
    hande_jaw,
    thin_bar_with_a_low_fragment,
)


def _mat() -> SupportSolid:
    """The owner's mat as the guard holds it: its top 7 mm over the reading at ``MAT_MM``."""
    return SupportSolid(name="seen_s00_support0", kind="support", surface=0,
                        centre_mm=np.array([0.0, 0.0, MAT_MM + 7.0 - 20.0]),
                        half_extents_mm=np.array([400.0, 400.0, 20.0]), rotation=np.eye(3),
                        local_plane=np.array([MAT_MM, 0.0, 0.0]), excess_mm=2.0, band_mm=3.0, allowance_mm=2.0)


def _seen_boxes(neighbours: np.ndarray) -> SeenEnvelope:
    """The boxes the camera world would hold for ``neighbours``: one box round each cylinder of the bar's pile."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    boxes = []
    for x in (-50.0, 0.0, 50.0):
        for side in (-1.0, 1.0):
            boxes.append(SeenBox(centre_mm=(x, side * 48.0, MAT_MM + 50.0), rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                                                  (0.0, 0.0, 1.0)),
                                 half_extents_mm=(33.0, 33.0, 53.0), soft_mm=2.0 if x == 0.0 else 0.0))
    cfg = hande_cell()
    envelope = SeenEnvelope.of(boxes, gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry),
                               open_width_mm=float(cfg.gripper.max_width_mm), distance_mm=3.0)
    assert envelope is not None
    return envelope


def scenes() -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Five parts, each with the keywords its search is run with: open, boxed in with the guard's boxes and the floor
    it holds, a cylinder beside a wall seen from a camera that cannot see one side (side approaches), a thin bar with
    its own low fragment and a declared wall, and a round footprint alone."""
    bar, pile = boxed_in_bar()
    cylinder, wall = cylinder_beside_a_wall(12.0)
    assert wall is not None
    low = camera(LOW_ON_MINUS_Y, CYLINDER, WALL, NEIGHBOUR)
    to_base = np.eye(4)
    to_base[:3, :3] = low.rotation
    to_base[:3, 3] = low.eye
    intrinsics = np.array([[low.focal_px, 0.0, low.width / 2.0], [0.0, low.focal_px, low.height / 2.0],
                           [0.0, 0.0, 1.0]])
    seen = corridor_seen_in(np.where(np.isfinite(low.depth), low.depth, 0.0), intrinsics, to_base)
    jaw = hande_jaw()
    return {
        "an open cube": (box_cloud((-20.0, -20.0, MAT_MM), (20.0, 20.0, MAT_MM + 40.0)),
                         dict(support_height_mm=MAT_MM, jaw=jaw)),
        "a bar boxed in": (bar, dict(support_height_mm=MAT_MM, jaw=jaw, obstacle_points_base_mm=pile,
                                     seen_envelope=_seen_boxes(pile),
                                     hand_floor=HandFloor(solids=(_mat(),), distance_mm=3.0,
                                                          fingers_to_the_reading=True, finger_floor_drop_mm=1.0))),
        "a cylinder beside a wall": (cylinder, dict(support_height_mm=0.0, jaw=jaw, obstacle_points_base_mm=wall,
                                                    side_approaches=True, corridor_seen=seen)),
        "a thin bar with a low fragment": (thin_bar_with_a_low_fragment(),
                                           dict(support_height_mm=0.0, jaw=jaw,
                                                rigid_obstacle_points_base_mm=box_cloud((40.0, -60.0, 0.0),
                                                                                        (44.0, 60.0, 90.0)))),
        "a round footprint": (cylinder_cloud(36.0, 50.0), dict(support_height_mm=0.0, jaw=jaw, side_approaches=True)),
    }


def _same(test: unittest.TestCase, expected: list[Any], got: list[Any], what: str) -> None:
    test.assertEqual(len(expected), len(got), what)
    for rank, (a, b) in enumerate(zip(expected, got)):
        for name in ("score", "grip_width_mm", "contact_angle_rad", "clearance_mm"):
            test.assertEqual(np.float64(getattr(a, name)).tobytes(), np.float64(getattr(b, name)).tobytes(),
                             f"{what}: grasp {rank} {name}")
        for name in ("position_mm", "approach", "closing_axis"):
            test.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), f"{what}: grasp {rank} {name}")
            test.assertEqual(getattr(a, name).tobytes(), getattr(b, name).tobytes(), f"{what}: grasp {rank} {name}")


def _counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for cause in REFUSAL_CAUSES:
        counts.setdefault(cause, 0)
    return counts


def _shuffled_elsewhere(inputs: SfeInputs, units: list[Any], counts: "dict[str, int] | None",
                        rng: random.Random) -> dict[Any, list[Any]]:
    """``units`` shuffled and cut into random chunks, each run on a plan of its own, as a worker plans it."""
    order = list(units)
    rng.shuffle(order)
    found: dict[Any, list[Any]] = {}
    start = 0
    while start < len(order):
        size = rng.randint(1, max(2, len(order) // 3))
        plan = plan_support_footprint(inputs)
        assert plan is not None
        found.update(run_units(plan, order[start:start + size], counts))
        start += size
    return found


def _merged(found: dict[Any, list[Any]]) -> list[Any]:
    return [candidate for unit in sorted(found) for candidate in found[unit]]


class _ShufflingRunner:
    """A runner as the worker pool is one: every unit it is handed, shuffled, on plans of its own; what it ran, kept."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.handed: list[list[Any]] = []

    def __call__(self, plan: Any, units: Any) -> tuple[dict[Any, list[Any]], dict[str, int]]:
        self.handed.append(list(units))
        counted: dict[str, int] = {}
        found = _shuffled_elsewhere(plan.inputs, list(units), counted if plan.inputs.counting else None, self.rng)
        return found, counted


class TheUnitsInAnyOrderAreTheSearchTests(unittest.TestCase):
    def test_shuffled_units_on_plans_of_their_own_give_the_functions_grasps_and_counts(self) -> None:
        rng = random.Random(20261008)
        for name, (cloud, keywords) in scenes().items():
            for counting in (True, False):
                with self.subTest(scene=name, counting=counting):
                    expected_counts: dict[str, int] | None = {} if counting else None
                    expected = generate_support_footprint_grasps(cloud, refusals=expected_counts, **keywords)
                    inputs = SfeInputs(cloud, counting=counting, **keywords)
                    plan = plan_support_footprint(inputs)
                    assert plan is not None
                    counts = _counts() if counting else None
                    found = _merged(_shuffled_elsewhere(inputs, plan.units(COARSE), counts, rng))
                    if len(found) < 3:
                        found += _merged(_shuffled_elsewhere(inputs, plan.units(FINE) + plan.units(ROLLED), None,
                                                             rng))
                    _same(self, expected, rank_found(plan, found), name)
                    self.assertEqual(expected_counts, counts)
                    if counting:
                        self.assertEqual(list(REFUSAL_CAUSES), list(counts or {}))

    def test_a_runner_that_runs_them_shuffled_gives_the_same_grasps_counts_and_stages(self) -> None:
        for name, (cloud, keywords) in scenes().items():
            for fine_pass in (True, False):
                with self.subTest(scene=name, fine_pass=fine_pass):
                    expected_counts: dict[str, int] = {}
                    expected_stages: dict[str, str] = {}
                    expected = generate_support_footprint_grasps(cloud, refusals=expected_counts,
                                                                 stages=expected_stages, fine_pass=fine_pass,
                                                                 **keywords)
                    counts: dict[str, int] = {}
                    stages: dict[str, str] = {}
                    got = generate_support_footprint_grasps(cloud, refusals=counts, stages=stages, fine_pass=fine_pass,
                                                            runner=_ShufflingRunner(7), **keywords)
                    _same(self, expected, got, name)
                    self.assertEqual(expected_counts, counts)
                    self.assertEqual(list(expected_counts), list(counts), "the counts' order")
                    self.assertEqual(expected_stages, stages)

    def test_the_fine_and_rolled_units_are_handed_on_only_where_the_coarse_grasps_ask(self) -> None:
        cases = scenes()
        open_cube, boxed = cases["an open cube"], cases["a bar boxed in"]
        runner = _ShufflingRunner(3)
        generate_support_footprint_grasps(open_cube[0], refusals={}, runner=runner, **open_cube[1])
        self.assertEqual(1, len(runner.handed), "an open cube's coarse grid finds enough")
        self.assertEqual({COARSE}, {unit[0] for unit in runner.handed[0]})

        runner = _ShufflingRunner(3)
        stages: dict[str, str] = {}
        found = generate_support_footprint_grasps(boxed[0], refusals={}, stages=stages, runner=runner, **boxed[1])
        self.assertEqual([], found)
        self.assertEqual(2, len(runner.handed))
        self.assertEqual({FINE, ROLLED}, {unit[0] for unit in runner.handed[1]})

        runner = _ShufflingRunner(3)
        generate_support_footprint_grasps(boxed[0], refusals={}, stages=stages, runner=runner, fine_pass=False,
                                          **boxed[1])
        self.assertEqual(1, len(runner.handed), "the fine pass left for later is handed to nobody")
        self.assertEqual(FINE_SEARCH_DEFERRED, stages["fine"])


class ARunnerThatCannotAnswerTests(unittest.TestCase):
    def test_a_runner_that_fails_in_the_fine_search_has_the_part_searched_here_and_keeps_nothing(self) -> None:
        cloud, keywords = scenes()["a bar boxed in"]
        expected_counts: dict[str, int] = {}
        expected = generate_support_footprint_grasps(cloud, refusals=expected_counts, **keywords)

        class _FailsLate(_ShufflingRunner):
            def __call__(self, plan: Any, units: Any) -> tuple[dict[Any, list[Any]], dict[str, int]]:
                if any(unit[0] != COARSE for unit in units):
                    raise SfeRunnerFailed("a worker died")
                found, counted = super().__call__(plan, units)
                return found, {cause: count + 1000 for cause, count in counted.items()}

        counts: dict[str, int] = {}
        got = generate_support_footprint_grasps(cloud, refusals=counts, runner=_FailsLate(5), **keywords)
        _same(self, expected, got, "searched here")
        self.assertEqual(expected_counts, counts, "a count the runner answered was kept")

    def test_a_runner_that_answers_part_of_its_units_is_not_believed(self) -> None:
        cloud, keywords = scenes()["an open cube"]
        expected = generate_support_footprint_grasps(cloud, refusals={}, **keywords)

        def half(plan: Any, units: Any) -> tuple[dict[Any, list[Any]], dict[str, int]]:
            planned = plan_support_footprint(plan.inputs)
            assert planned is not None
            ran = run_units(planned, list(units)[: len(units) // 2], {})
            return ran, {}

        _same(self, expected, generate_support_footprint_grasps(cloud, refusals={}, runner=half, **keywords), "half")

    def test_a_runner_that_raises_anything_else_is_a_fault_of_the_program(self) -> None:
        cloud, keywords = scenes()["an open cube"]

        def broken(plan: Any, units: Any) -> Any:
            raise TypeError("a runner with the wrong arguments")

        with self.assertRaises(TypeError):
            generate_support_footprint_grasps(cloud, runner=broken, **keywords)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
