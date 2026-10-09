"""SFE leaves its fine search for later where its caller asks, and says so; asked nothing, it searches as it always did.

Where SFE's coarse grid finds fewer than three grasps it searches again on a fine grid, the most of what such a part
costs: on the owner's cell (2026-10-08) one tilted cube took 24 s, and the part the pick took 3. A pick that already holds
another part's full result need not wait for it (the owner's "speed first"). So
``generate_support_footprint_grasps(..., fine_pass=False, stages=...)`` returns the coarse grid's grasps as they are
and says ``stages["fine"] = "deferred"``; the stage stamps ``support_footprint_fine: "deferred"``, and the calculator holds
the switch the pick loop turns (``GraspCalculator.sfe_fine_pass``). ``fine_pass=True``, the default, is byte-identical to
the search before, and stamps nothing.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.generation import support_footprint as sf
from src.robot.grasping.generation._support_footprint_stage import (
    FINE_SEARCH_DEFERRED,
    FINE_SEARCH_KEY,
    support_footprint_breakdowns,
)
from tests.test_a_grasp_keeps_the_guards_distance_from_the_support_it_holds import READING_MM, _box_cloud, _jaw
from tests.test_sfe_says_why_it_refused import (
    MAT_MM,
    box_cloud,
    boxed_in_bar,
    candidate_digest,
    cylinder_cloud,
    hande_jaw,
)


def _bar_between_declared_neighbours() -> "tuple[np.ndarray, dict[str, Any]]":
    """The Zollstock between six cylinders handed as declared: the coarse grid fits no grasp between them, the fine grid
    fits a few closing 20 degrees off the bar's axis (``test_sfe_says_why_it_refused``)."""
    bar, neighbours = boxed_in_bar()
    return bar, {"support_height_mm": MAT_MM, "jaw": hande_jaw(), "rigid_obstacle_points_base_mm": neighbours}


def _run(cloud: np.ndarray, kwargs: "dict[str, Any]", *, most: int = 12,
         **asked: Any) -> "tuple[list[Any], dict[str, str], dict[str, int]]":
    """SFE's grasps (``most`` at most), what it says it left out, and its counts."""
    stages: dict[str, str] = {}
    counts: dict[str, int] = {}
    found = sf.generate_support_footprint_grasps(cloud, max_candidates=most, stages=stages, refusals=counts,
                                                 **kwargs, **asked)
    return found, stages, counts


class HeldBackTests(unittest.TestCase):
    def test_where_the_coarse_grid_finds_few_it_returns_the_coarse_grids_grasps_and_says_so(self) -> None:
        """Every part finds few here (``_FINE_BELOW`` raised): held back, the grasps are the coarse grid's alone, as a
        search with no fine grid at all returns them, and the fine grid's turned and rolled grasps are not among them."""
        cloud, kwargs = _box_cloud(40.0), {"support_height_mm": READING_MM, "jaw": _jaw()}
        with mock.patch.object(sf, "_FINE_BELOW", 0):
            coarse, _, _ = _run(cloud, kwargs, most=400)
        with mock.patch.object(sf, "_FINE_BELOW", 10_000):
            held, stages, _ = _run(cloud, kwargs, most=400, fine_pass=False)
            fine, fine_stages, _ = _run(cloud, kwargs, most=400, fine_pass=True)
        self.assertTrue(coarse)
        self.assertEqual(candidate_digest(coarse), candidate_digest(held))
        self.assertEqual({"fine": FINE_SEARCH_DEFERRED}, stages)
        self.assertNotEqual(candidate_digest(coarse), candidate_digest(fine), "the fine grid added nothing to compare")
        self.assertEqual({}, fine_stages)

    def test_a_part_the_fine_grid_alone_grips_waits_with_no_grasp(self) -> None:
        cloud, kwargs = _bar_between_declared_neighbours()
        held, stages, _ = _run(cloud, kwargs, fine_pass=False)
        full, full_stages, _ = _run(cloud, kwargs)
        self.assertEqual([], held)
        self.assertEqual({"fine": FINE_SEARCH_DEFERRED}, stages)
        self.assertTrue(full, "the fine grid no longer grips the bar: the scene says nothing")
        self.assertEqual({}, full_stages)

    def test_the_counts_are_the_coarse_grids_held_back_or_not(self) -> None:
        """The fine search's refusals were never counted, so holding it back changes no count."""
        cloud, kwargs = _bar_between_declared_neighbours()
        _, _, held = _run(cloud, kwargs, fine_pass=False)
        _, _, full = _run(cloud, kwargs)
        self.assertEqual(full, held)

    def test_a_part_the_coarse_grid_grips_enough_is_the_same_held_back_or_not(self) -> None:
        cloud, kwargs = cylinder_cloud(40.0, 60.0), {"support_height_mm": 0.0, "jaw": hande_jaw()}
        held, stages, held_counts = _run(cloud, kwargs, fine_pass=False)
        full, _, counts = _run(cloud, kwargs)
        self.assertGreaterEqual(len(full), sf._FINE_BELOW)
        self.assertEqual(candidate_digest(full), candidate_digest(held))
        self.assertEqual(counts, held_counts)
        self.assertEqual({}, stages, "a part the coarse grid grips enough said it waited")


class AskedNothingTests(unittest.TestCase):
    def test_fine_pass_on_is_byte_identical_to_the_search_before(self) -> None:
        scenes = {
            "the bar between declared neighbours": _bar_between_declared_neighbours(),
            "a free cylinder": (cylinder_cloud(40.0, 60.0), {"support_height_mm": 0.0, "jaw": hande_jaw()}),
            "a 60 mm cube": (box_cloud((-30.0, -30.0, 0.0), (30.0, 30.0, 60.0)),
                             {"support_height_mm": 0.0, "jaw": hande_jaw()}),
        }
        for name, (cloud, kwargs) in scenes.items():
            with self.subTest(name):
                before = sf.generate_support_footprint_grasps(cloud, max_candidates=12, **kwargs)
                asked, stages, _ = _run(cloud, kwargs, fine_pass=True)
                self.assertEqual(candidate_digest(before), candidate_digest(asked))
                self.assertEqual({}, stages)


class TheStageSaysItTests(unittest.TestCase):
    def _stage(self, **asked: Any) -> "tuple[list[Any], dict[str, Any]]":
        cloud, kwargs = _bar_between_declared_neighbours()
        return support_footprint_breakdowns(
            cloud, camera_to_base=np.eye(4), support_height_mm=kwargs["support_height_mm"], jaw=kwargs["jaw"],
            obstacle_points_base_mm=None, rigid_obstacle_points_base_mm=kwargs["rigid_obstacle_points_base_mm"],
            max_candidates=12, **asked)

    def test_held_back_the_stage_stamps_support_footprint_fine_deferred(self) -> None:
        breakdowns, telemetry = self._stage(fine_pass=False)
        self.assertEqual([], breakdowns)
        self.assertEqual(FINE_SEARCH_DEFERRED, telemetry.get(FINE_SEARCH_KEY))

    def test_asked_nothing_the_stage_stamps_nothing_new(self) -> None:
        breakdowns, telemetry = self._stage()
        self.assertTrue(breakdowns)
        self.assertNotIn(FINE_SEARCH_KEY, telemetry)
        _, held = self._stage(fine_pass=False)
        self.assertEqual(telemetry["support_footprint_refused"], held["support_footprint_refused"])


class TheCalculatorHoldsTheSwitchTests(unittest.TestCase):
    def test_the_switch_is_on_until_the_pick_loop_turns_it_off(self) -> None:
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import calculator

        self.assertIs(True, calculator(scene=True).sfe_fine_pass)

    def test_turned_off_a_part_whose_coarse_grid_finds_few_says_it_waited_and_one_that_finds_enough_does_not(
            self) -> None:
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import BENCH, MAT, PART, Frame_, boxed_in_bar as bar
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import calculator, compute

        calc = calculator(scene=True)
        boxed_in = bar()
        self.assertNotIn(FINE_SEARCH_KEY, compute(calc, boxed_in).telemetry)
        calc.sfe_fine_pass = False
        self.assertEqual(FINE_SEARCH_DEFERRED, compute(calc, boxed_in).telemetry.get(FINE_SEARCH_KEY))
        alone = compute(calc, Frame_((BENCH, MAT), PART))
        self.assertTrue(alone.is_success)
        self.assertNotIn(FINE_SEARCH_KEY, alone.telemetry)


if __name__ == "__main__":
    unittest.main()
