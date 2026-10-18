"""How many pushes are left: the memory a campaign keeps across its picks.

Held by the service for one campaign and keyed by where a part stands in BASE: no part identity survives a
rescan, but a position does.

Push budgets (owner, 2026-09-29): 1 push per part, 2 per pick, 5 per campaign. A spent budget means no more
pushes, never a stop. A part that was pushed is known again anywhere within 30 mm plus how far it was pushed
of where it stood: a push that glanced off moved it about that far, in whatever direction.

The exclusion zones ``next_target`` laid over the part a pick failed on left on 2026-10-16 with ``next_target`` (the
owner's cleanup); the regions a task keeps out are pinned in ``tests/robot/execution/autonomous_grasp/test_a_task_keeps_its_bin_out_of_its_picks.py``.
"""

from __future__ import annotations

import math
import unittest


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


if __name__ == "__main__":
    unittest.main()
