"""Which points the head is asked about, and what the answer is at each of them.

Between the sampler and the loss. `corpus.sample.build_sample`, under `SampleSpec.grasp_set`, emits
a scene's grasp table plus a sparse list of `(point, grasp)` pairs; the head runs at a few dozen
seeds and the loss wants a padded block per seed. This turns one into the other, and it draws the
seeds.

The seeds are a mixture of three sources:

    labelled centres     the head sees real positives from the first step
    the model's top-k    what it will actually receive at inference
    random object points negatives with the right marginal

Trained only on labelled points, the head is never asked about a place its own graspability field
would propose. Trained only on its own top-k, it starts from noise and may never find a positive.
The same distribution argument gives the scorer a published +6.5 % AUC from training a discriminator
on the generator's own proposals rather than on an offline label set, because the offline set holds
only collision-free grasps and the generator produces slightly colliding ones. The shares are an
assumption: their order follows that argument, their values are a starting point and a sweep.

Truncation is a bias unless it is random. A seed can reach more labels than the block holds, and
keeping the first `L` by grasp index would not be harmless: the labeller walks
`axes x anchors x approaches` in order, so low indices are early axes and a head trained that way
would never be shown the modes that sort late. The subsample is random, without replacement.

At the default `max_labels = 16` some seeds are truncated, and `truncated` is returned so a run can
report how many rather than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from src.robot.grasping.deep.net.set_loss import GraspSetTarget

__all__ = ["GraspTable", "SeedDraw", "SeedShares", "gather_set_targets", "sample_seeds"]


@dataclass(frozen=True, slots=True)
class GraspTable:
    """One scene's labelled grasps. Positions are in the same frame as the point cloud."""

    position_m: torch.Tensor    # (G, 3)
    approach: torch.Tensor      # (G, 3) unit
    axis: torch.Tensor          # (G, 3) unit, sign meaningless
    width_m: torch.Tensor       # (G,)
    #: `(G,)` index into the part-role vocabulary, or None when nothing supplies one.
    #:
    #: Optional so every existing construction site is unchanged. A table without roles produces a
    #: target without roles, which produces a loss without the affordance term.
    part_index: "torch.Tensor | None" = None


@dataclass(frozen=True, slots=True)
class SeedShares:
    """How the seed budget is split. See the module docstring for why there are three sources."""

    labelled: float = 0.5
    predicted: float = 0.25
    random: float = 0.25


class SeedDraw(NamedTuple):
    """The chosen seeds and where each came from, so a run can report the mixture it actually got."""

    # Named `point_index`, not `index`: a NamedTuple field called `index` shadows `tuple.index`, and
    # the shadowing is legal and silent.
    point_index: torch.Tensor   # (S,) unique point indices
    from_labelled: int
    from_predicted: int
    from_random: int


def _take(pool: torch.Tensor, count: int, generator: torch.Generator | None) -> torch.Tensor:
    """`count` entries of `pool` without replacement, or all of it when it is smaller.

    The permutation is drawn on the CPU and moved, never drawn on the pool's device: a CUDA tensor
    with a CPU generator raises. Drawing on the CPU also makes the seed device-independent, so a
    seeded CPU run and a seeded GPU run pick the same seeds; a per-device generator would make those
    two runs quietly different.
    """
    if count <= 0 or not pool.numel():
        return pool[:0]
    if pool.numel() <= count:
        return pool
    order = torch.randperm(pool.numel(), generator=generator)
    return pool[order[:count].to(pool.device)]


def sample_seeds(*, count: int, labelled: torch.Tensor, score: torch.Tensor,
                 candidate: torch.Tensor, shares: SeedShares = SeedShares(),
                 generator: torch.Generator | None = None) -> SeedDraw:
    """Draw `count` distinct seeds from the three sources.

    `labelled` marks points that reach at least one grasp, `score` is the graspability logit at
    every point, `candidate` marks points a seed may sit on at all (the object, not the table).

    Distinct: a duplicated seed is counted twice by the loss and twice by the coverage metric, which
    inflates both by exactly the duplication rate. Sources are drawn in order and later ones only
    fill what earlier ones left, so a scene with plenty of labelled points does not waste half its
    budget re-drawing them.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    for name, mask in (("labelled", labelled), ("candidate", candidate)):
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a bool mask, got {mask.dtype}")
    if labelled.shape != score.shape or labelled.shape != candidate.shape:
        raise ValueError("labelled, score and candidate must describe the same points")

    chosen: list[torch.Tensor] = []
    used = torch.zeros_like(candidate)

    def remaining(mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask & candidate & ~used, as_tuple=False).flatten()

    picked_labelled = _take(remaining(labelled), int(round(count * shares.labelled)), generator)
    used[picked_labelled] = True
    chosen.append(picked_labelled)

    want = int(round(count * shares.predicted))
    pool = remaining(torch.ones_like(candidate))
    if want > 0 and pool.numel():
        # The top-k of the model's own field, which is what inference will hand the head. Taken from
        # what is left rather than from everything, so it does not simply re-draw the labelled points
        # the field is best at.
        top = pool[score[pool].argsort(descending=True)[:want]]
        used[top] = True
        chosen.append(top)
    else:
        chosen.append(pool[:0])

    picked_random = _take(remaining(torch.ones_like(candidate)),
                          count - sum(int(part.numel()) for part in chosen), generator)
    used[picked_random] = True
    chosen.append(picked_random)

    index = torch.cat(chosen)
    return SeedDraw(point_index=index, from_labelled=int(chosen[0].numel()),
                    from_predicted=int(chosen[1].numel()),
                    from_random=int(chosen[2].numel()))


def gather_set_targets(seed_index: torch.Tensor, pair_point: torch.Tensor,
                       pair_grasp: torch.Tensor, table: GraspTable, points_m: torch.Tensor,
                       *, max_labels: int = 16,
                       generator: torch.Generator | None = None
                       ) -> tuple[GraspSetTarget, int]:
    """The `(S, L, ...)` block the loss consumes, plus how many seeds were truncated.

    The offset is formed here, from `position - seed`, and both sides must be in one frame: the
    sampler emits the grasp table in the same local frame as the cloud so that this subtraction is
    meaningful. A frame mismatch makes every offset wrong by a per-scene constant, which is the
    shape a head can almost learn around.

    The random subsample is done by sorting the pairs on a random key and then stably by point, so
    each point's pairs arrive in a random order and taking the first `max_labels` is a draw without
    replacement. Two sorts rather than a loop over seeds.
    """
    if max_labels < 1:
        raise ValueError(f"max_labels must be >= 1, got {max_labels}")
    device = points_m.device
    seeds = int(seed_index.numel())

    # An empty grasp table is a normal scene, not an error. v5 holds objects no grasp reaches, and a
    # scene whose every object is one of those has a table of length zero; the pair list is then
    # empty too, so `valid` is already all false and only the gather would still index.
    if not table.approach.shape[0]:
        zeros = torch.zeros((seeds, max_labels, 3), dtype=points_m.dtype, device=device)
        return GraspSetTarget(
            approach=zeros, axis=zeros.clone(), offset_m=zeros.clone(),
            width_m=torch.zeros((seeds, max_labels), dtype=points_m.dtype, device=device),
            valid=torch.zeros((seeds, max_labels), dtype=torch.bool, device=device),
            # Shaped, not None: `valid` is already all false so nothing reads these, but a shape that
            # matches the populated case keeps the two branches interchangeable for a caller.
            part_index=(torch.zeros((seeds, max_labels), dtype=torch.long, device=device)
                        if table.part_index is not None else None)), 0

    if pair_point.numel():
        # Drawn on the CPU and moved, for the reason `_take` gives: a CUDA tensor with a CPU
        # generator raises, and drawing per device would make a seeded run mean two different things.
        shuffled = torch.argsort(
            torch.rand(pair_point.numel(), generator=generator).to(device))
        order = shuffled[torch.argsort(pair_point[shuffled], stable=True)]
        sorted_point, sorted_grasp = pair_point[order], pair_grasp[order]
    else:
        sorted_point = pair_point.new_zeros(0)
        sorted_grasp = pair_grasp.new_zeros(0)

    low = torch.searchsorted(sorted_point, seed_index.to(sorted_point.dtype), right=False)
    high = torch.searchsorted(sorted_point, seed_index.to(sorted_point.dtype), right=True)
    available = high - low
    take = available.clamp(max=max_labels)

    slot = torch.arange(max_labels, device=device).unsqueeze(0)
    valid = slot < take.unsqueeze(1)
    # Clamped so an invalid slot indexes something rather than out of bounds. Its value is masked
    # everywhere it is read; reading a wrong number that is thrown away is fine, indexing past the
    # end is not.
    flat = (low.unsqueeze(1) + slot).clamp(max=max(sorted_grasp.numel() - 1, 0))
    # Clamped against the table as well: an invalid slot reads a row it never uses, and reading
    # a wrong number that is masked away is fine while indexing past the end is not.
    grasp = (sorted_grasp[flat].clamp(max=table.approach.shape[0] - 1)
             if sorted_grasp.numel() else torch.zeros_like(flat))

    seed_points = points_m[seed_index].unsqueeze(1)
    target = GraspSetTarget(
        approach=table.approach[grasp],
        axis=table.axis[grasp],
        offset_m=table.position_m[grasp] - seed_points,
        width_m=table.width_m[grasp],
        valid=valid,
        # Through the same `grasp` index as everything above it, so a slot's role is the role of the
        # very label whose approach and width it is being graded on.
        part_index=(table.part_index[grasp] if table.part_index is not None else None),
    )
    return target, int((available > max_labels).sum())
