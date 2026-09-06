"""Drawing seeds and collecting what each of them is supposed to answer.

⚠ TWO FAILURES HERE ARE SILENT AND BOTH INFLATE A METRIC.

A duplicated seed is counted twice by the loss and twice by the coverage number, so a head looks
better by exactly the duplication rate. And a truncation that keeps the FIRST labels rather than a
random subset drops whichever modes sort late: the labeller walks `axes x anchors x approaches` in
order, so "first" means "early axes", and the head would simply never be shown the rest.

Neither shows up as a wrong shape, so both are checked directly.
"""

from __future__ import annotations

import unittest

import torch

from src.robot.grasping.deep.net.set_targets import (
    GraspTable,
    SeedShares,
    gather_set_targets,
    sample_seeds,
)


def _table(count: int, seed: int = 0) -> GraspTable:
    generator = torch.Generator().manual_seed(seed)
    approach = torch.randn(count, 3, generator=generator)
    axis = torch.randn(count, 3, generator=generator)
    return GraspTable(
        position_m=torch.randn(count, 3, generator=generator) * 0.05,
        approach=approach / approach.norm(dim=-1, keepdim=True),
        axis=axis / axis.norm(dim=-1, keepdim=True),
        width_m=torch.rand(count, generator=generator) * 0.08)


class SeedDrawTests(unittest.TestCase):

    def _masks(self, points: int = 200, labelled: int = 40, objects: int = 120) -> tuple:
        has_label = torch.zeros(points, dtype=torch.bool)
        has_label[:labelled] = True
        candidate = torch.zeros(points, dtype=torch.bool)
        candidate[:objects] = True
        score = torch.linspace(0.0, 1.0, points)
        return has_label, score, candidate

    def test_the_seeds_are_distinct(self) -> None:
        """⭐ A duplicate is counted twice by the loss AND by the coverage metric."""
        has_label, score, candidate = self._masks()
        draw = sample_seeds(count=48, labelled=has_label, score=score, candidate=candidate,
                            generator=torch.Generator().manual_seed(1))
        self.assertEqual(draw.point_index.numel(), torch.unique(draw.point_index).numel())

    def test_every_seed_sits_on_a_candidate_point(self) -> None:
        """A seed on the table is a grasp proposal in mid-air. `candidate` is the object mask."""
        has_label, score, candidate = self._masks()
        draw = sample_seeds(count=48, labelled=has_label, score=score, candidate=candidate,
                            generator=torch.Generator().manual_seed(2))
        self.assertTrue(bool(candidate[draw.point_index].all()))

    def test_all_three_sources_contribute(self) -> None:
        has_label, score, candidate = self._masks()
        draw = sample_seeds(count=40, labelled=has_label, score=score, candidate=candidate,
                            generator=torch.Generator().manual_seed(3))
        self.assertGreater(draw.from_labelled, 0)
        self.assertGreater(draw.from_predicted, 0)
        self.assertGreater(draw.from_random, 0)
        self.assertEqual(draw.from_labelled + draw.from_predicted + draw.from_random,
                         draw.point_index.numel())

    def test_the_predicted_share_takes_the_highest_scores_available(self) -> None:
        """It stands in for what inference will hand the head. If it drew at random it would be a
        second random share and the train-to-inference gap would stay open."""
        has_label, score, candidate = self._masks()
        draw = sample_seeds(count=40, labelled=has_label, score=score, candidate=candidate,
                            shares=SeedShares(labelled=0.0, predicted=1.0, random=0.0),
                            generator=torch.Generator().manual_seed(4))
        best = score[candidate].topk(draw.point_index.numel()).values.min()
        self.assertGreaterEqual(float(score[draw.point_index].min()), float(best) - 1e-6)

    def test_a_scene_with_no_labels_still_produces_seeds(self) -> None:
        """v5 holds objects no grasp reaches. Those seeds train the confidence to say so."""
        _, score, candidate = self._masks()
        draw = sample_seeds(count=32, labelled=torch.zeros(200, dtype=torch.bool), score=score,
                            candidate=candidate, generator=torch.Generator().manual_seed(5))
        self.assertEqual(draw.point_index.numel(), 32)
        self.assertEqual(draw.from_labelled, 0)

    def test_fewer_candidates_than_the_budget_is_not_an_error(self) -> None:
        candidate = torch.zeros(200, dtype=torch.bool)
        candidate[:5] = True
        draw = sample_seeds(count=32, labelled=candidate.clone(), score=torch.rand(200),
                            candidate=candidate, generator=torch.Generator().manual_seed(6))
        self.assertEqual(draw.point_index.numel(), 5)

    def test_refusals(self) -> None:
        has_label, score, candidate = self._masks()
        with self.assertRaises(ValueError):
            sample_seeds(count=0, labelled=has_label, score=score, candidate=candidate)
        with self.assertRaises(ValueError):
            sample_seeds(count=4, labelled=score, score=score, candidate=candidate)
        with self.assertRaises(ValueError):
            sample_seeds(count=4, labelled=has_label, score=score[:10], candidate=candidate)


class GatherTests(unittest.TestCase):

    def _pairs(self) -> tuple:
        """Point 3 reaches grasps 0,1,2; point 7 reaches grasp 4; point 5 reaches nothing."""
        pair_point = torch.tensor([3, 3, 3, 7], dtype=torch.int64)
        pair_grasp = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        return pair_point, pair_grasp

    def test_each_seed_gets_exactly_the_grasps_it_reaches(self) -> None:
        pair_point, pair_grasp = self._pairs()
        table = _table(8)
        points = torch.randn(10, 3) * 0.1
        target, truncated = gather_set_targets(
            torch.tensor([3, 5, 7]), pair_point, pair_grasp, table, points, max_labels=4,
            generator=torch.Generator().manual_seed(7))
        self.assertEqual(truncated, 0)
        self.assertEqual([int(v) for v in target.valid.sum(dim=1)], [3, 0, 1])

    def test_the_offset_is_the_grasp_centre_minus_the_seed(self) -> None:
        """⛔ Both sides must be in ONE frame. They were not in the first version, and every offset
        was wrong by a per-scene constant, which is the shape a head can almost learn around."""
        pair_point, pair_grasp = self._pairs()
        table = _table(8)
        points = torch.randn(10, 3) * 0.1
        target, _ = gather_set_targets(torch.tensor([7]), pair_point, pair_grasp, table, points,
                                       max_labels=4)
        self.assertTrue(torch.allclose(target.offset_m[0, 0],
                                       table.position_m[4] - points[7], atol=1e-6))

    def test_a_seed_that_reaches_nothing_is_all_invalid(self) -> None:
        pair_point, pair_grasp = self._pairs()
        target, _ = gather_set_targets(torch.tensor([5]), pair_point, pair_grasp, _table(8),
                                       torch.randn(10, 3), max_labels=4)
        self.assertFalse(bool(target.valid.any()))

    def test_an_empty_pair_list_does_not_index_out_of_bounds(self) -> None:
        empty = torch.zeros(0, dtype=torch.int64)
        target, truncated = gather_set_targets(torch.tensor([0, 1]), empty, empty, _table(8),
                                               torch.randn(4, 3), max_labels=4)
        self.assertFalse(bool(target.valid.any()))
        self.assertEqual(truncated, 0)

    def test_truncation_is_reported_rather_than_hidden(self) -> None:
        pair_point = torch.zeros(20, dtype=torch.int64)
        pair_grasp = torch.arange(20)
        target, truncated = gather_set_targets(torch.tensor([0]), pair_point, pair_grasp,
                                               _table(20), torch.randn(4, 3), max_labels=8)
        self.assertEqual(int(target.valid.sum()), 8)
        self.assertEqual(truncated, 1)

    def test_truncation_is_random_rather_than_the_first_by_index(self) -> None:
        """⭐⭐ THE BIAS THIS FILE EXISTS FOR. The labeller walks axes then anchors then approaches, so
        keeping the lowest grasp indices means keeping the earliest AXES and never showing the head
        the modes that sort late. Read across seeds rather than within one: over many draws, a late
        index must appear about as often as an early one."""
        pair_point = torch.zeros(20, dtype=torch.int64)
        pair_grasp = torch.arange(20)
        table = _table(20)
        points = torch.randn(4, 3)
        seen = torch.zeros(20)
        for trial in range(200):
            target, _ = gather_set_targets(
                torch.tensor([0]), pair_point, pair_grasp, table, points, max_labels=4,
                generator=torch.Generator().manual_seed(trial))
            # Recover which rows were taken by matching the widths, which are distinct here.
            for width in target.width_m[0][target.valid[0]]:
                seen[(table.width_m - width).abs().argmin()] += 1
        early, late = float(seen[:10].sum()), float(seen[10:].sum())
        self.assertGreater(min(early, late) / max(early, late), 0.7,
                           f"the subsample favours one end of the table: {early} vs {late}")

    def test_a_scene_with_no_grasps_at_all_is_not_an_error(self) -> None:
        """⛔ FOUND BY THE FIRST SMOKE TEST AGAINST REAL CLOUDS, and missed by every fixture here.

        v5 holds objects no grasp reaches, so a scene whose objects are all of that kind has a grasp
        table of LENGTH ZERO. The pair list is empty too, so nothing would be supervised anyway; the
        defect was that the gather still indexed the empty table and raised. Every synthetic scene in
        this suite had grasps in it, which is why 117 tests did not see it.
        """
        empty_table = GraspTable(position_m=torch.zeros(0, 3), approach=torch.zeros(0, 3),
                                 axis=torch.zeros(0, 3), width_m=torch.zeros(0))
        empty = torch.zeros(0, dtype=torch.int64)
        target, truncated = gather_set_targets(torch.tensor([0, 1, 2]), empty, empty, empty_table,
                                               torch.randn(4, 3), max_labels=5)
        self.assertEqual(target.valid.shape, (3, 5))
        self.assertFalse(bool(target.valid.any()))
        self.assertEqual(truncated, 0)
        for field in (target.approach, target.axis, target.offset_m, target.width_m):
            self.assertTrue(torch.isfinite(field).all())

    def test_a_grasp_index_can_never_run_past_the_table(self) -> None:
        """The clamp is against the TABLE, not only against the pair list. An invalid slot reads a
        row it never uses, and reading a masked-away number is fine while indexing past the end is
        not."""
        pair_point = torch.tensor([0, 0, 0], dtype=torch.int64)
        pair_grasp = torch.tensor([0, 1, 1], dtype=torch.int64)
        table = _table(2)
        target, _ = gather_set_targets(torch.tensor([0, 1]), pair_point, pair_grasp, table,
                                       torch.randn(3, 3), max_labels=6)
        self.assertEqual(int(target.valid[0].sum()), 3)
        self.assertEqual(int(target.valid[1].sum()), 0)

    def test_refusals(self) -> None:
        empty = torch.zeros(0, dtype=torch.int64)
        with self.assertRaises(ValueError):
            gather_set_targets(torch.tensor([0]), empty, empty, _table(4), torch.randn(2, 3),
                               max_labels=0)


if __name__ == "__main__":
    unittest.main()
