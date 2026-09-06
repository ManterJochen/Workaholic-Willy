"""Proposals that are the same grasp eight times, and the number that makes it visible.

⛔⛔ MEASURED ON THE FIRST REAL PROPOSAL RUN, 320 grasps over 40 scenes from a trained artifact:

    distinct at   5 mm / 20 deg :  170  (53.1 %)
    distinct at  10 mm / 20 deg :  119  (37.2 %)
    distinct at  20 mm / 20 deg :   82  (25.6 %)
    distinct at  40 mm / 20 deg :   64  (20.0 %)

    distinct approach vectors, to three decimals:  11  of  320

Eleven approaches across 320 grasps. A referee handed that list judges roughly 1.6 distinct grasps per
scene and reports eight trials, so the primary metric would carry four times the sample size it
earned, out of trials that are not independent. The corpus says the same thing one level down: 96.3 %
of candidates there share a single approach vector.

⚠ THE SHARE IS REPORTED, THE FILTER IS OPT-IN. A number a run prints about itself cannot be forgotten;
a filter that ran by default would silently change every rate taken before it.
"""

from __future__ import annotations

import math
import unittest

from src.robot.grasping.deep.eval.propose import Proposal, distinct_share, suppress


def _p(rank: int, position, approach=(0.0, 0.0, -1.0), scene: str = "s1") -> Proposal:
    return Proposal(scene_id=scene, instance_id=1, position_mm=tuple(position),
                    approach=tuple(approach), closing_axis=(1.0, 0.0, 0.0), width_mm=50.0,
                    confidence=1.0 - rank * 0.01, seed_index=rank, slot=0, rank=rank)


class SuppressTests(unittest.TestCase):

    def test_a_near_twin_with_the_same_approach_is_dropped(self) -> None:
        kept = suppress([_p(0, (0, 0, 0)), _p(1, (5, 0, 0))], radius_mm=20.0)
        self.assertEqual([p.rank for p in kept], [0])

    def test_the_same_position_from_the_OTHER_SIDE_is_kept(self) -> None:
        """⛔ THE ANGLE IS NOT OPTIONAL. Two poses a few millimetres apart approaching from opposite
        sides are two grasps, and a radius alone would merge them, throwing away exactly the variety
        the head is being asked to produce."""
        kept = suppress([_p(0, (0, 0, 0), (0, 0, -1)), _p(1, (5, 0, 0), (0, 0, 1))],
                        radius_mm=20.0)
        self.assertEqual([p.rank for p in kept], [0, 1])

    def test_a_far_twin_with_the_same_approach_is_kept(self) -> None:
        kept = suppress([_p(0, (0, 0, 0)), _p(1, (100, 0, 0))], radius_mm=20.0)
        self.assertEqual(len(kept), 2)

    def test_it_keeps_the_HIGHER_RANKED_one_and_never_reorders(self) -> None:
        """⚠ The head's ranking is the product. Suppression may only remove what the head ranked
        lower; a filter that promoted a survivor would be answering a different question."""
        kept = suppress([_p(0, (0, 0, 0)), _p(1, (2, 0, 0)), _p(2, (60, 0, 0))], radius_mm=20.0)
        self.assertEqual([p.rank for p in kept], [0, 2])

    def test_scenes_do_not_suppress_each_other(self) -> None:
        """Two scenes can hold the same pose and they are two grasps. Nothing joins them."""
        kept = suppress([_p(0, (0, 0, 0), scene="a"), _p(1, (0, 0, 0), scene="b")], radius_mm=50.0)
        self.assertEqual(len(kept), 2)

    def test_a_shallow_angle_difference_still_counts_as_the_same_grasp(self) -> None:
        """Within the 20 degree window a cell would place the gripper the same way, so it is one
        trial and not two."""
        tilt = math.radians(5.0)
        kept = suppress([_p(0, (0, 0, 0), (0, 0, -1)),
                         _p(1, (3, 0, 0), (math.sin(tilt), 0, -math.cos(tilt)))], radius_mm=20.0)
        self.assertEqual(len(kept), 1)

    def test_an_empty_list_is_a_share_of_zero_and_not_a_crash(self) -> None:
        self.assertEqual(distinct_share([]), 0.0)

    def test_all_distinct_is_a_share_of_one(self) -> None:
        """The control. A share that is always low would flag a healthy run as a collapsed one."""
        spread = [_p(i, (i * 200.0, 0, 0)) for i in range(5)]
        self.assertEqual(distinct_share(spread), 1.0)

    def test_all_identical_is_a_share_of_one_over_n(self) -> None:
        self.assertAlmostEqual(distinct_share([_p(i, (0, 0, 0)) for i in range(8)]), 1 / 8)


class WiringTests(unittest.TestCase):

    def test_the_flag_exists_and_defaults_to_OFF(self) -> None:
        """⚠ Off by default. A filter that ran on its own would change every rate taken before it,
        and the whole reason the share is printed is so nobody has to trust a default."""
        from src.robot.grasping.deep.__main__ import build_parser

        args = build_parser().parse_args(
            ["propose", "--artifact", "w.pt", "--clouds", "c", "--out", "o.jsonl"])
        self.assertEqual(args.spread_mm, 0.0)
        args = build_parser().parse_args(
            ["propose", "--artifact", "w.pt", "--clouds", "c", "--out", "o.jsonl",
             "--spread-mm", "15"])
        self.assertEqual(args.spread_mm, 15.0)

    def test_it_reaches_the_function(self) -> None:
        from pathlib import Path

        source = Path("src/robot/grasping/deep/__main__.py").read_text(encoding="utf-8")
        self.assertIn("spread_mm=args.spread_mm", source)

    def test_the_prefix_is_taken_AFTER_the_suppression(self) -> None:
        """⚠ Order matters. Taking the top 8 and then removing duplicates leaves however many happen
        to survive, which is a different number per scene and therefore not a prefix of anything."""
        from pathlib import Path

        source = Path("src/robot/grasping/deep/eval/propose.py").read_text(encoding="utf-8")
        spread_at = source.index("grasps = _spread(")
        prefix_at = source.index("for rank, grasp in enumerate(grasps[:top])")
        self.assertLess(spread_at, prefix_at)


if __name__ == "__main__":
    unittest.main()
