"""Matching several predicted grasps to several labelled ones, one to one.

⚠ WHAT THIS FILE HAS TO CATCH is not a crash. It is a matching that quietly lets two slots claim the
same label, because then the head can fill every slot with the same easy grasp, score well on the
loss, and report a coverage it never achieved. That failure is invisible to a shape test, so every
structural property below runs against BOTH implementations rather than only the default.

The second thing it does is MEASURE, because measuring here already overturned one decision. The
module first shipped greedy matching on the stated grounds that exact assignment was too slow at our
batch sizes. That was an assumption written down as a fact, and it was wrong: the exact solver is
FASTER. The gap test that remains is about the approximation we keep as a fallback, and it carries a
second lesson: on uniform random costs greedy looks about 13 % off, on a cost shaped like the real
one about 2 %. The obvious benchmark is seven times too pessimistic.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable

import numpy as np
import torch

from src.robot.grasping.deep.net.assignment import (
    Matching,
    confidence_target,
    greedy_one_to_one,
    optimal_one_to_one,
)

#: Every structural property must hold for both. The default is first.
_MATCHERS: tuple[tuple[str, Callable[..., Matching]], ...] = (
    ("optimal", optimal_one_to_one),
    ("greedy", greedy_one_to_one),
)


def _valid(batch: int, labels: int, counts: list[int]) -> torch.Tensor:
    mask = torch.zeros(batch, labels, dtype=torch.bool)
    for row, count in enumerate(counts):
        mask[row, :count] = True
    return mask


def _distance_cost(rng: np.random.Generator, slots: int, labels: int) -> np.ndarray:
    """A cost shaped like the real one: slots and labels are grasps, the cost is how far apart."""
    left = rng.normal(size=(slots, 6))
    right = rng.normal(size=(labels, 6))
    return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)


class OneToOneTests(unittest.TestCase):

    def test_no_two_slots_claim_the_same_label(self) -> None:
        """⭐ THE PROPERTY THE MODULE EXISTS FOR. Without it the head collapses onto one mode and the
        coverage metric reports slots rather than modes."""
        cost = torch.rand(64, 6, 9, generator=torch.Generator().manual_seed(0))
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, _valid(64, 9, [9] * 64))
                for row in range(64):
                    claimed = matching.slot_label[row][matching.slot_matched[row]].tolist()
                    self.assertEqual(len(claimed), len(set(claimed)))

    def test_padding_labels_are_never_matched(self) -> None:
        cost = torch.rand(32, 4, 8, generator=torch.Generator().manual_seed(1))
        counts = [0, 1, 2, 3] * 8
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, _valid(32, 8, counts))
                for row, count in enumerate(counts):
                    self.assertFalse(bool(matching.label_matched[row, count:].any()))
                    self.assertEqual(int(matching.slot_matched[row].sum()), min(count, 4))

    def test_a_seed_with_no_labels_is_fully_unmatched(self) -> None:
        """A real and frequent case: v5 holds objects no grasp reaches, and a seed proposed there has
        nothing to be supervised against except its own confidence going to zero."""
        cost = torch.rand(8, 5, 6)
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, torch.zeros(8, 6, dtype=torch.bool))
                self.assertFalse(bool(matching.slot_matched.any()))
                self.assertTrue(torch.equal(matching.slot_label,
                                            torch.full((8, 5), -1, dtype=torch.long)))

    def test_more_labels_than_slots_fills_every_slot(self) -> None:
        cost = torch.rand(16, 3, 10, generator=torch.Generator().manual_seed(2))
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, _valid(16, 10, [10] * 16))
                self.assertTrue(bool(matching.slot_matched.all()))
                self.assertEqual(int(matching.label_matched.sum()), 16 * 3)

    def test_one_slot_reproduces_the_old_single_output_behaviour(self) -> None:
        """K = 1 must be exactly "supervise against the nearest label", so the sweep starts from what
        the repository already measured rather than from something new."""
        cost = torch.rand(24, 1, 7, generator=torch.Generator().manual_seed(3))
        nearest = cost.squeeze(1).argmin(-1)
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, _valid(24, 7, [7] * 24))
                self.assertTrue(torch.equal(matching.slot_label.squeeze(1), nearest))

    def test_the_cheapest_pair_is_always_taken(self) -> None:
        cost = torch.full((1, 3, 3), 5.0)
        cost[0, 2, 1] = 0.1
        for name, match in _MATCHERS:
            with self.subTest(name):
                self.assertEqual(int(match(cost, torch.ones(1, 3, dtype=torch.bool))
                                     .slot_label[0, 2]), 1)

    def test_refusals(self) -> None:
        for name, match in _MATCHERS:
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    match(torch.rand(4, 3), torch.ones(4, 3, dtype=torch.bool))
                with self.assertRaises(ValueError):
                    match(torch.rand(4, 3, 5), torch.ones(4, 4, dtype=torch.bool))

    def test_the_matching_carries_no_gradient(self) -> None:
        """⛔ The assignment is a discrete choice. Gradients reach the loss through the COST of the
        chosen pairs, never through the choosing, and a matcher that returned a differentiable tensor
        would make that silently untrue."""
        cost = torch.rand(4, 3, 5, requires_grad=True)
        for name, match in _MATCHERS:
            with self.subTest(name):
                matching = match(cost, torch.ones(4, 5, dtype=torch.bool))
                for field in matching:
                    self.assertFalse(field.requires_grad)


class ApproximationTests(unittest.TestCase):
    """What the kept fallback costs, and how badly the obvious benchmark misleads."""

    def test_greedy_is_never_better_than_optimal_and_is_close_on_real_costs(self) -> None:
        """⭐ MEASURED, NOT ASSUMED, and the two cost shapes disagree by a factor of seven.

        A cost matrix of uniform noise is what an unthinking benchmark generates and it is the worst
        case for a greedy rule. Ours is a distance between grasps, which has structure: a slot near one
        label is near it in every component.
        """
        rng = np.random.default_rng(0)
        shapes: dict[str, Callable[[int, int], np.ndarray]] = {
            "uniform noise": lambda rows, cols: rng.random((rows, cols)),
            "distance, like ours": lambda rows, cols: _distance_cost(rng, rows, cols),
        }
        measured: dict[str, float] = {}
        for name, maker in shapes.items():
            gaps: list[float] = []
            for _ in range(300):
                slots, labels = int(rng.integers(2, 9)), int(rng.integers(2, 13))
                matrix = maker(slots, labels)
                cost = torch.from_numpy(matrix).unsqueeze(0)
                mask = torch.ones(1, labels, dtype=torch.bool)
                totals = []
                for _, match in _MATCHERS:
                    matching = match(cost, mask)
                    taken = matching.slot_matched[0]
                    totals.append(float(
                        matrix[taken.numpy(), matching.slot_label[0][taken].numpy()].sum()))
                best, approximate = totals
                self.assertLessEqual(best, approximate + 1e-9,
                                     "greedy beat an optimal assignment, which is impossible")
                gaps.append((approximate - best) / max(best, 1e-9))
            measured[name] = float(np.mean(gaps))
            print(f"\n  greedy above optimal, {name:>20}: {measured[name]:.2%}")

        self.assertLess(measured["distance, like ours"], 0.05)
        self.assertGreater(measured["uniform noise"], measured["distance, like ours"],
                           "the unstructured benchmark should be the pessimistic one")


class ConfidenceTests(unittest.TestCase):

    def test_the_target_is_one_for_matched_and_zero_for_unmatched(self) -> None:
        matching = Matching(
            slot_label=torch.tensor([[0, -1, 2]]),
            slot_matched=torch.tensor([[True, False, True]]),
            label_matched=torch.tensor([[True, False, True]]))
        self.assertTrue(torch.equal(confidence_target(matching),
                                    torch.tensor([[1.0, 0.0, 1.0]])))

    def test_a_head_with_more_slots_than_modes_is_asked_to_say_so(self) -> None:
        """The honest half of set prediction: without this, extra slots are indistinguishable from
        real ones at inference and a consumer takes K candidates on faith."""
        cost = torch.rand(8, 6, 6, generator=torch.Generator().manual_seed(4))
        target = confidence_target(optimal_one_to_one(cost, _valid(8, 6, [2] * 8)))
        self.assertTrue(torch.equal(target.sum(dim=1), torch.full((8,), 2.0)))


if __name__ == "__main__":
    unittest.main()
