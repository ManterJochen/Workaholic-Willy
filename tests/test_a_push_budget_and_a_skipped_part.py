"""The two memories that last across picks: how many pushes are left, and which parts to skip.

Both are small, held by the service for one campaign, and keyed by where a part stands in BASE. No part
identity survives a rescan, but a position does.

Push budgets (owner, 2026-09-29): 1 push per part, 2 per pick, 5 per campaign. A spent budget means no more
pushes, never a stop. A part that was pushed is known again anywhere within 30 mm plus how far it was pushed
of where it stood: a push that glanced off moved it about that far, in whatever direction.

Exclusion zones for ``next_target``, which is "rescan while skipping the failed part":

* same label only;
* BASE XY centre;
* radius max(30 mm, half the footprint diagonal);
* lasting this pick plus the next 2;
* when only excluded parts remain, that is detectable, so the campaign can stop with a clear message.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.robot.grasping.recovery.exclusion_zones import (
    EXCLUSION_EXTRA_PICKS,
    EXCLUSION_RADIUS_FLOOR_MM,
    ExclusionZones,
    exclusion_radius_mm,
    footprint_diagonal_mm,
)
from src.robot.grasping.recovery.push_budgets import (
    BUDGET_ALLOWED,
    BUDGET_CAMPAIGN_SPENT,
    BUDGET_PART_SPENT,
    BUDGET_PICK_SPENT,
    PUSHES_PER_CAMPAIGN,
    PUSHES_PER_PART,
    PUSHES_PER_PICK,
    SAME_PART_RADIUS_MM,
    PushBudgets,
)


class PushBudgetsTests(unittest.TestCase):
    def test_the_owners_numbers(self) -> None:
        self.assertEqual((PUSHES_PER_PART, PUSHES_PER_PICK, PUSHES_PER_CAMPAIGN), (1, 2, 5))
        self.assertEqual(SAME_PART_RADIUS_MM, 30.0)

    def test_one_push_per_part_known_by_its_centre_or_its_landing(self) -> None:
        budgets = PushBudgets()
        budgets.start_pick()
        self.assertTrue(budgets.check((0.0, 0.0, 15.0)).allowed)
        budgets.record(centre_mm=(0.0, 0.0, 15.0), landing_mm=(30.0, 0.0, 15.0))
        for seen_at in ((5.0, 5.0, 15.0), (28.0, 3.0, 15.0), (0.0, 0.0, 90.0)):
            with self.subTest(seen_at=seen_at):
                verdict = budgets.check(seen_at)
                self.assertFalse(verdict.allowed)
                self.assertEqual(verdict.code, BUDGET_PART_SPENT)
                self.assertTrue(verdict.sentence.endswith("."))
        self.assertTrue(budgets.check((100.0, 0.0, 15.0)).allowed)

    def test_a_part_pushed_askew_is_still_the_part_that_was_pushed(self) -> None:
        # A 50 mm push that glanced off: the part ended 50 mm from where it stood, 60 deg off the push line,
        # 50 mm from the predicted landing too. It must not be pushed again on the next pick.
        budgets = PushBudgets()
        budgets.start_pick()
        budgets.record(centre_mm=(0.0, 0.0, 15.0), landing_mm=(50.0, 0.0, 15.0))
        budgets.start_pick()
        turned = math.radians(60.0)
        verdict = budgets.check((50.0 * math.cos(turned), 50.0 * math.sin(turned), 15.0))
        self.assertEqual((verdict.allowed, verdict.code), (False, BUDGET_PART_SPENT))

    def test_a_landing_farther_than_thirty_millimetres_still_names_the_part(self) -> None:
        budgets = PushBudgets()
        budgets.start_pick()
        budgets.record(centre_mm=(0.0, 0.0, 15.0), landing_mm=(60.0, 0.0, 15.0))
        self.assertEqual(budgets.check((58.0, 3.0, 15.0)).code, BUDGET_PART_SPENT)

    def test_the_same_part_radius_is_thirty_millimetres_plus_how_far_it_was_pushed(self) -> None:
        budgets = PushBudgets(per_pick=10, per_campaign=10)
        budgets.start_pick()
        budgets.record(centre_mm=(0.0, 0.0, 0.0))
        self.assertEqual(budgets.check((29.9, 0.0, 0.0)).code, BUDGET_PART_SPENT)
        self.assertEqual(budgets.check((30.1, 0.0, 0.0)).code, BUDGET_ALLOWED)
        budgets.record(centre_mm=(500.0, 0.0, 0.0), landing_mm=(550.0, 0.0, 0.0))
        self.assertEqual(budgets.check((500.0, 79.9, 0.0)).code, BUDGET_PART_SPENT)
        self.assertEqual(budgets.check((500.0, 80.1, 0.0)).code, BUDGET_ALLOWED)

    def test_two_pushes_per_pick_then_a_fresh_pick_has_two_again(self) -> None:
        budgets = PushBudgets()
        budgets.start_pick()
        budgets.record(centre_mm=(0.0, 0.0, 0.0))
        budgets.record(centre_mm=(200.0, 0.0, 0.0))
        verdict = budgets.check((400.0, 0.0, 0.0))
        self.assertEqual((verdict.allowed, verdict.code), (False, BUDGET_PICK_SPENT))
        budgets.start_pick()
        self.assertEqual(budgets.pushes_this_pick, 0)
        self.assertEqual(budgets.check((400.0, 0.0, 0.0)).code, BUDGET_ALLOWED)

    def test_five_pushes_per_campaign_then_none_and_the_campaign_goes_on(self) -> None:
        budgets = PushBudgets()
        for pick in range(5):
            budgets.start_pick()
            self.assertTrue(budgets.check((pick * 100.0, 0.0, 0.0)).allowed)
            budgets.record(centre_mm=(pick * 100.0, 0.0, 0.0))
        budgets.start_pick()
        verdict = budgets.check((900.0, 0.0, 0.0))
        self.assertEqual((verdict.allowed, verdict.code), (False, BUDGET_CAMPAIGN_SPENT))
        self.assertIn("campaign", verdict.sentence)
        self.assertEqual(budgets.pushes_this_campaign, 5)
        # Spent is not a stop: the budget keeps answering, picks keep starting.
        budgets.start_pick()
        self.assertFalse(budgets.check((900.0, 0.0, 0.0)).allowed)

    def test_the_records_are_kept_for_telemetry(self) -> None:
        budgets = PushBudgets()
        budgets.start_pick()
        budgets.record(centre_mm=(1.0, 2.0, 3.0), landing_mm=(31.0, 2.0, 3.0))
        (record,) = budgets.records
        self.assertEqual((record.centre_mm, record.landing_mm, record.pick_index), ((1.0, 2.0, 3.0),
                                                                                     (31.0, 2.0, 3.0), 1))

    def test_other_budgets_can_be_given_and_bad_ones_are_refused(self) -> None:
        budgets = PushBudgets(per_part=2, per_pick=3, per_campaign=4)
        budgets.start_pick()
        budgets.record(centre_mm=(0.0, 0.0, 0.0))
        self.assertTrue(budgets.check((0.0, 0.0, 0.0)).allowed)
        budgets.record(centre_mm=(0.0, 0.0, 0.0))
        self.assertEqual(budgets.check((0.0, 0.0, 0.0)).code, BUDGET_PART_SPENT)
        self.assertEqual(PushBudgets(per_campaign=0).check((0.0, 0.0, 0.0)).code, BUDGET_CAMPAIGN_SPENT)
        for bad in (dict(per_part=-1), dict(per_pick=1.5), dict(same_part_radius_mm=0.0), dict(per_campaign=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PushBudgets(**bad)  # type: ignore[arg-type]

    def test_a_centre_that_is_not_three_finite_numbers_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            PushBudgets().check((0.0, math.nan, 0.0))


class ExclusionRadiusTests(unittest.TestCase):
    def test_radius_is_thirty_or_half_the_diagonal(self) -> None:
        self.assertEqual(EXCLUSION_RADIUS_FLOOR_MM, 30.0)
        self.assertEqual(exclusion_radius_mm(20.0), 30.0)
        self.assertEqual(exclusion_radius_mm(100.0), 50.0)
        self.assertEqual(exclusion_radius_mm(None), 30.0)
        self.assertEqual(exclusion_radius_mm(math.nan), 30.0)

    def test_footprint_diagonal_follows_the_part_not_the_axes(self) -> None:
        xs, ys = np.meshgrid(np.linspace(-40.0, 40.0, 41), np.linspace(-10.0, 10.0, 11))
        flat = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 5.0)])
        self.assertAlmostEqual(footprint_diagonal_mm(flat), math.hypot(80.0, 20.0), places=6)
        turn = math.radians(30.0)
        rotation = np.array([[math.cos(turn), -math.sin(turn), 0.0], [math.sin(turn), math.cos(turn), 0.0],
                             [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(footprint_diagonal_mm(flat @ rotation.T), math.hypot(80.0, 20.0), places=6)
        self.assertEqual(footprint_diagonal_mm(np.zeros((0, 3))), 0.0)

    def test_a_square_footprint_gets_its_own_diagonal(self) -> None:
        # A square's principal axes are arbitrary; the BASE-axis box keeps the estimate at the true diagonal.
        xs, ys = np.meshgrid(np.linspace(-25.0, 25.0, 26), np.linspace(-25.0, 25.0, 26))
        square = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
        self.assertAlmostEqual(footprint_diagonal_mm(square), math.hypot(50.0, 50.0), places=6)


class ExclusionZonesTests(unittest.TestCase):
    def test_a_zone_lasts_this_pick_and_the_next_two(self) -> None:
        self.assertEqual(EXCLUSION_EXTRA_PICKS, 2)
        zones = ExclusionZones()
        zones.start_pick()
        zones.exclude(label="bolt", centre_mm=(100.0, 0.0, 10.0), footprint_diagonal_mm=40.0)
        for pick in range(3):
            with self.subTest(pick=zones.pick_index):
                self.assertTrue(zones.excludes(label="bolt", centre_mm=(110.0, 5.0, 12.0)))
            zones.start_pick()
        self.assertFalse(zones.excludes(label="bolt", centre_mm=(110.0, 5.0, 12.0)))
        self.assertEqual(zones.zones(), ())

    def test_same_label_only(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.exclude(label="Bolt ", centre_mm=(0.0, 0.0, 0.0))
        self.assertTrue(zones.excludes(label="bolt", centre_mm=(0.0, 0.0, 0.0)))
        self.assertFalse(zones.excludes(label="nut", centre_mm=(0.0, 0.0, 0.0)))

    def test_the_radius_decides_what_is_the_same_part(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zone = zones.exclude(label="bolt", centre_mm=(0.0, 0.0, 0.0), footprint_diagonal_mm=100.0)
        self.assertEqual(zone.radius_mm, 50.0)
        self.assertTrue(zones.excludes(label="bolt", centre_mm=(49.0, 0.0, 0.0)))
        self.assertFalse(zones.excludes(label="bolt", centre_mm=(51.0, 0.0, 0.0)))
        # Height does not matter: the zone is a column over the table.
        self.assertTrue(zones.excludes(label="bolt", centre_mm=(0.0, 0.0, 300.0)))

    def test_remaining_and_only_excluded_parts_remain(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        zones.exclude(label="bolt", centre_mm=(0.0, 0.0, 0.0))
        zones.exclude(label="bolt", centre_mm=(200.0, 0.0, 0.0))
        seen = [(5.0, 0.0, 0.0), (205.0, 0.0, 0.0), (400.0, 0.0, 0.0)]
        self.assertEqual(zones.remaining(label="bolt", centres_mm=seen), (2,))
        self.assertFalse(zones.only_excluded_remain(label="bolt", centres_mm=seen))
        self.assertTrue(zones.only_excluded_remain(label="bolt", centres_mm=seen[:2]))
        self.assertFalse(zones.only_excluded_remain(label="bolt", centres_mm=[]))
        self.assertFalse(zones.only_excluded_remain(label="nut", centres_mm=seen[:2]))
        sentence = zones.only_excluded_sentence(label="bolt", count=2)
        self.assertIn("bolt", sentence)
        self.assertIn("2", sentence)
        self.assertTrue(sentence.endswith("."))

    def test_a_zone_added_later_in_the_campaign_lasts_from_its_own_pick(self) -> None:
        zones = ExclusionZones()
        for _ in range(4):
            zones.start_pick()
        zone = zones.exclude(label="bolt", centre_mm=(0.0, 0.0, 0.0))
        self.assertEqual((zone.added_pick, zone.last_pick), (4, 6))
        self.assertEqual(zones.pick_index, 4)

    def test_a_bad_label_or_centre_is_refused(self) -> None:
        zones = ExclusionZones()
        with self.assertRaises(ValueError):
            zones.exclude(label="  ", centre_mm=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            zones.exclude(label="bolt", centre_mm=(0.0, math.inf, 0.0))
        with self.assertRaises(ValueError):
            ExclusionZones(extra_picks=-1)


if __name__ == "__main__":
    unittest.main()
