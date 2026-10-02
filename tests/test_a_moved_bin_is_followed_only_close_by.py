"""Before every drop the task looks at its bin again, and follows it only where it moved a little (Q13 (b), 2026-09-30).

A task keeps the bin it found (``KeptTarget``) and checks it once per part, from the look it was first seen from
(``recheck``): the bin it sees there nearest the kept one is taken, and followed only when both hold:

* it moved at most min(100 mm, half the kept footprint's diagonal) in BASE XY;
* its footprint is within 20 % of the kept one's, side for side;
* its rim stands within 20 mm of the kept one's.

Anything else is the target lost: not seen at all (``not_seen``), moved further (``moved_too_far``), or another bin in
its place (``footprint_changed``: another size, its footprint or its height). A bin removed or swapped is never
mistaken for the target, and the part goes back where it was gripped (the task's half, held elsewhere). What was seen
is kept for the chat either way, so the operator reads how far it moved. The kept bin is the one the survey found: a
task checks every sighting against it, never against a later one it followed, so a bin that creeps is not followed
further and further (the task's half).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from tests._task_fakes import BIN_CENTRE, BIN_RIM_MM, BIN_SIZE, ScriptedLocator, bin_object, bin_points


def _kept() -> Any:
    from src.robot.execution.place_target import KeptTarget

    kept = KeptTarget.of(bin_object(), camera="wrist", look=None, seen_at=1.0)
    assert kept is not None
    return kept


def _seen(dx: float = 0.0, dy: float = 0.0, *, size: "tuple[float, float]" = BIN_SIZE, **keywords: Any) -> Any:
    return bin_object(bin_points((BIN_CENTRE[0] + dx, BIN_CENTRE[1] + dy), size), **keywords)


def _check(*seen: Any, kept: Any = None, **keywords: Any) -> Any:
    from src.robot.execution.place_target import recheck

    locator = ScriptedLocator({"blue bin": list(seen)})
    return recheck([locator], kept if kept is not None else _kept(), **keywords)


class AMovedBinTests(unittest.TestCase):
    def test_a_bin_that_moved_a_little_is_followed_where_it_stands_now(self) -> None:
        check = _check(_seen(30.0, -10.0))

        self.assertEqual("", check.why)
        assert check.kept is not None
        self.assertIs(check.kept, check.seen)
        np.testing.assert_allclose((BIN_CENTRE[0] + 30.0, BIN_CENTRE[1] - 10.0), check.kept.centre_xy_mm, atol=3.0)
        self.assertAlmostEqual(np.hypot(30.0, 10.0), check.moved_mm, delta=3.0)

    def test_a_bin_moved_150_mm_is_lost_and_what_was_seen_is_said(self) -> None:
        check = _check(_seen(150.0))

        self.assertEqual("moved_too_far", check.why)
        self.assertIsNone(check.kept)
        assert check.seen is not None
        self.assertAlmostEqual(150.0, check.moved_mm, delta=3.0)

    def test_the_bound_is_half_the_footprints_diagonal_where_that_is_less_than_100_mm(self) -> None:
        from src.robot.execution.place_target import KeptTarget

        small = (100.0, 80.0)
        kept = KeptTarget.of(bin_object(bin_points(BIN_CENTRE, small)), camera="wrist", look=None, seen_at=0.0)
        assert kept is not None
        bound = 0.5 * float(np.hypot(*small))

        near = _check(_seen(bound - 8.0, size=small), kept=kept)
        far = _check(_seen(bound + 8.0, size=small), kept=kept)

        self.assertEqual("", near.why)
        self.assertEqual("moved_too_far", far.why)

    def test_another_bin_in_its_place_is_lost_however_close_it_stands(self) -> None:
        for size in ((300.0 * 1.3, 200.0), (300.0, 200.0 * 0.7), (200.0, 120.0)):
            with self.subTest(size=size):
                self.assertEqual("footprint_changed", _check(_seen(size=size)).why)

    def test_a_bin_within_20_percent_of_its_size_is_the_same_bin(self) -> None:
        self.assertEqual("", _check(_seen(size=(300.0 * 1.12, 200.0 * 0.9))).why)

    def test_a_bin_of_its_footprint_whose_rim_stands_more_than_20_mm_off_is_another_bin(self) -> None:
        """Red before: the height was never compared, so a bin of the same footprint 60 mm lower in its place was
        followed, and the drop went down to the new rim."""
        for rim_off_mm, why in ((-60.0, "footprint_changed"), (25.0, "footprint_changed"), (-15.0, ""), (15.0, "")):
            with self.subTest(rim_off_mm=rim_off_mm):
                check = _check(bin_object(bin_points(BIN_CENTRE, rim_mm=BIN_RIM_MM + rim_off_mm)))
                self.assertEqual(why, check.why)
                self.assertEqual(why == "", check.followed)

    def test_the_rim_tolerance_is_the_callers_to_narrow(self) -> None:
        self.assertEqual("footprint_changed",
                         _check(bin_object(bin_points(BIN_CENTRE, rim_mm=BIN_RIM_MM - 15.0)), rim_tolerance_mm=10.0).why)

    def test_a_bin_gone_is_not_seen(self) -> None:
        check = _check()
        self.assertEqual("not_seen", check.why)
        self.assertIsNone(check.seen)
        self.assertIsNone(check.kept)
        self.assertIsNone(check.moved_mm)

    def test_a_bin_seen_by_too_few_points_is_not_seen(self) -> None:
        self.assertEqual("not_seen", _check(bin_object(bin_points()[:15])).why)

    def test_of_two_bins_the_one_nearest_the_kept_one_is_taken(self) -> None:
        check = _check(_seen(400.0, score=0.95), _seen(20.0, score=0.5))

        self.assertEqual("", check.why)
        assert check.kept is not None
        np.testing.assert_allclose((BIN_CENTRE[0] + 20.0, BIN_CENTRE[1]), check.kept.centre_xy_mm, atol=3.0)

    def test_the_bound_and_the_tolerance_are_the_callers_to_narrow(self) -> None:
        self.assertEqual("moved_too_far", _check(_seen(30.0), max_shift_mm=20.0).why)
        self.assertEqual("footprint_changed", _check(_seen(size=(300.0 * 1.12, 200.0)), footprint_tolerance=0.1).why)

    def test_the_check_reads_as_the_console_says_it(self) -> None:
        check = _check(_seen(150.0))
        said = check.to_dict()
        self.assertEqual({"moved_mm", "followed", "target", "why"}, set(said))
        self.assertIs(False, said["followed"])
        self.assertEqual(check.render(), str(check))
        self.assertIn("150", check.render())

    def test_the_kept_bins_camera_is_the_one_that_looks_again(self) -> None:
        from src.robot.execution.place_target import recheck

        wrist = ScriptedLocator({"blue bin": [_seen(10.0)]}, rig_id="wrist")
        other = ScriptedLocator({"blue bin": [_seen(300.0)]}, rig_id="side")

        check = recheck([other, wrist], _kept())

        self.assertEqual("", check.why)
        self.assertEqual(["blue bin"], wrist.asked)
        self.assertEqual([], other.asked, "a camera that did not see the bin was asked where it is")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
