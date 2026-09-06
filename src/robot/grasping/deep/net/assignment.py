"""Matching a head's several outputs to a seed's several labels, one to one.

A seed usually carries several different labelled grasps. A head with one output regressed against
that set lands between the modes on the approach, the offset and the width alike, and it can never
report more than one hit against coverage, the second of the four metrics. `rotation.py` removes
the sign ambiguity of the closing axis and the seed-local offset frame removes it again; neither
touches this case.

The corpus hides it behind an assignment rule: `corpus.sample._assign_grasps` keeps only the
nearest contact.

The head emits `K` slots per seed. Each label is matched to at most one slot and each slot to at
most one label. Matched slots are supervised against their label; unmatched slots are pushed toward
low confidence, which is how a head reports that it found fewer modes than there were. `K = 1`
reproduces single-output behaviour exactly, so K is a sweep and not a commitment.

This is not the thing `rotation.py` refutes. Both take a minimum, in different places:

    refuted    mean over labels of (1 - |cos|), the minimum taken inside, over the two signs of
               one label. It settles on the bisector of two axes, 45 degrees from every correct
               answer.

    used here  min over labels of (loss), the minimum taken outside, over the assignment. It
               settles on a real mode, never between two.

The first averages across distinct modes and is minimised where nothing is true. The second picks
a mode and commits to it.

The known failure of outside-minimum assignment is mode dropping: every slot converges on the
easiest label, because nothing stops two slots claiming the same one. One-to-one matching is the
guard, and it is why this module exists rather than a single `cost.min(dim=-1)`.

The default is the exact solver, which is also the faster of the two here. One training step's
worth of 8x12 matrices:

    seeds     optimal (Hungarian)     greedy (cpu)     greedy (gpu)
     1024              0.9 ms            5.9 ms           8.0 ms
     4096              3.3 ms            4.8 ms           7.9 ms
     8192              6.6 ms            6.8 ms           8.5 ms

These matrices are tiny and the solver's inner loop is compiled, while the vectorised greedy pays
`min(K, L)` full-batch passes.

Greedy's gap to optimal is about 13 % on uniform random costs and about 2 % on a cost shaped like
the real one, a distance between grasps (300 matrices per shape, so those are estimates, not
constants). A benchmark on uniform random costs is therefore seven times too pessimistic about the
gap. `greedy_one_to_one` is kept and measured because it needs no scipy and runs on GPU, but it is
not what trains anything.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = ["Matching", "confidence_target", "greedy_one_to_one", "optimal_one_to_one"]

#: Stands in for infinity in the cost matrix. A real infinity works for `argmin` but produces NaN the
#: moment anything downstream multiplies it by a zero mask.
_BLOCKED = 1.0e30


class Matching(NamedTuple):
    """Which slot got which label, from both sides.

    Both directions are returned because both are read: the loss needs the slot's view to know what to
    supervise it against, and the coverage metric needs the label's view to know how many modes were
    found at all.
    """

    #: `(B, K)` label index per slot, `-1` where the slot was left unmatched.
    slot_label: torch.Tensor
    #: `(B, K)` bool, true where the slot has a label.
    slot_matched: torch.Tensor
    #: `(B, L)` bool, true where the label was claimed by some slot. The coverage numerator.
    label_matched: torch.Tensor


def _checked(cost: torch.Tensor, label_valid: torch.Tensor) -> tuple[int, int, int]:
    if cost.dim() != 3:
        raise ValueError(f"cost must be (B, K, L), got {tuple(cost.shape)}")
    batch, slots, labels = cost.shape
    if label_valid.shape != (batch, labels):
        raise ValueError(f"label_valid must be {(batch, labels)}, got {tuple(label_valid.shape)}")
    return batch, slots, labels


def optimal_one_to_one(cost: torch.Tensor, label_valid: torch.Tensor) -> Matching:
    """Match `(B, K, L)` costs one to one, exactly. The default, and measured faster than greedy.

    Never differentiated through: the assignment is a discrete choice about which label supervises
    which slot, and the gradient flows through the cost of the chosen pairs, not through the
    choosing. Detaching here is what makes the loss well defined.

    Invalid label columns are filled with a blocking cost rather than sliced out. Slicing would
    mean a differently shaped problem per sample and a slow Python loop; blocking keeps one
    rectangular solve, and any slot that lands on a blocked column is dropped. The assignment among
    real columns is unaffected, because every blocked cell carries the same cost and the solver
    avoids them whenever a real one is free.
    """
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    batch, slots, labels = _checked(cost, label_valid)
    device = cost.device
    blocked = cost.detach().masked_fill(~label_valid.unsqueeze(1), _BLOCKED)
    matrices = blocked.to(torch.float64).cpu().numpy()
    usable = label_valid.cpu().numpy()

    slot_label = torch.full((batch, slots), -1, dtype=torch.long)
    slot_matched = torch.zeros((batch, slots), dtype=torch.bool)
    label_matched = torch.zeros((batch, labels), dtype=torch.bool)
    for row in range(batch):
        if not usable[row].any():
            continue
        chosen_slots, chosen_labels = linear_sum_assignment(matrices[row])
        # Drop the pairs the solver was forced onto padding because there were more slots than labels.
        real = usable[row][chosen_labels]
        chosen_slots, chosen_labels = chosen_slots[real], chosen_labels[real]
        slot_label[row, chosen_slots] = torch.from_numpy(chosen_labels).to(torch.long)
        slot_matched[row, chosen_slots] = True
        label_matched[row, chosen_labels] = True
    return Matching(slot_label=slot_label.to(device), slot_matched=slot_matched.to(device),
                    label_matched=label_matched.to(device))


def greedy_one_to_one(cost: torch.Tensor, label_valid: torch.Tensor) -> Matching:
    """Match `(B, K, L)` costs one to one, cheapest pair first.

    `label_valid` is `(B, L)` and marks which label columns are real rather than padding. Seeds carry
    different numbers of labels, so padding is the normal case, not an edge one.

    A batch item whose labels are all padding comes back fully unmatched rather than raising. That
    is a real and frequent case: v5 has objects no grasp reaches, and a seed proposed there has
    nothing to be supervised against except its own confidence, which should go to zero.
    """
    batch, slots, labels = _checked(cost, label_valid)
    remaining = cost.masked_fill(~label_valid.unsqueeze(1), _BLOCKED).clone()
    slot_label = cost.new_full((batch, slots), -1, dtype=torch.long)
    slot_matched = torch.zeros((batch, slots), dtype=torch.bool, device=cost.device)
    label_matched = torch.zeros((batch, labels), dtype=torch.bool, device=cost.device)
    rows = torch.arange(batch, device=cost.device)

    for _ in range(min(slots, labels)):
        flat = remaining.reshape(batch, slots * labels)
        best = flat.argmin(dim=-1)
        # A batch item with nothing left to match has every entry blocked. It must not consume a
        # round, and it must not be assigned the arbitrary index `argmin` returns for a flat tensor.
        open_pair = flat.gather(1, best.unsqueeze(1)).squeeze(1) < _BLOCKED
        slot = torch.div(best, labels, rounding_mode="floor")
        label = best % labels

        slot_label[rows[open_pair], slot[open_pair]] = label[open_pair]
        slot_matched[rows[open_pair], slot[open_pair]] = True
        label_matched[rows[open_pair], label[open_pair]] = True
        # Strike out the row and the column, which is the whole difference between this and a plain
        # per-slot minimum: without the column, every slot may claim the same label.
        remaining[rows[open_pair], slot[open_pair], :] = _BLOCKED
        remaining[rows[open_pair], :, label[open_pair]] = _BLOCKED

    return Matching(slot_label=slot_label, slot_matched=slot_matched, label_matched=label_matched)


def confidence_target(matching: Matching) -> torch.Tensor:
    """`(B, K)` float target for the per-slot confidence: 1 where matched, 0 where not.

    The honest half of set prediction. Without it a head fills every slot with a plausible-looking
    grasp whether or not the seed admits that many, and the extra slots are indistinguishable from
    real ones at inference. With it the head is asked to say how many modes it actually found, and a
    consumer can threshold on that rather than take K candidates on faith.
    """
    return matching.slot_matched.to(torch.float32)
