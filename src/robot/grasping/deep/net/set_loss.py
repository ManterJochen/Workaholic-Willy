"""Scoring K predicted grasps against the several a seed actually admits.

Where the pieces meet: `rotation.py` says how an orientation is written down, `assignment.py` says
which prediction answers which label, and this decides what a mismatch costs.

Two different weightings, and confusing them is the trap:

    the cost matrix   decides which label a slot is matched to
    the loss          decides how hard the slot is pushed toward it

They are computed from the same four terms and they do not have to agree, because they answer
different questions. The cost identifies a mode; the loss fits one.

The cost weights follow the shape of the labels (`datagen/eval/label_set_shape.py`): a supervised
point usually admits more than one grasp, and those grasps differ in their approach more often than
in their closing axis.

The approach is what tells two modes apart, by a wide margin, and the position is not: the several
grasps a point admits sit, by construction, at nearly the same place. A cost dominated by the offset
term would match slots by position and be blind to that distinction. The default weights therefore
let the approach dominate, and that ordering is the contract rather than the exact values.

The weights are a sweep, not a measurement. Their order follows the label shape above; their values
are a starting point, and anything read off a run has to be read knowing that.

Units. The four terms are not commensurable as they stand: two angles in radians, a distance in
metres and a width in metres. Lengths are divided by `_LENGTH_SCALE_M`, which is 50 mm, roughly the
scale at which a positional error stops being a refinement and becomes a different grasp. That puts
every term near 1 for a badly wrong prediction, so a weight of 1.0 means "as important as".

Unmatched slots. A slot with no label is supervised on its confidence alone and on nothing else.
Pushing its pose anywhere would invent a target, and pushing it toward the nearest label is the
mode-dropping failure `assignment.py` exists to prevent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Final

import torch
from torch.nn import functional as F  # noqa: N812 (the conventional alias)

from src.robot.grasping.deep.net.assignment import (
    Matching,
    confidence_target,
    optimal_one_to_one,
)
from src.robot.grasping.deep.net.rotation import (
    approach_loss,
    axis_loss,
    director_matrix,
    director_params,
)

__all__ = [
    "SetLossWeights",
    "GraspSetPrediction",
    "GraspSetTarget",
    "grasp_set_cost",
    "grasp_set_loss",
]

#: What a metre is divided by before it sits beside an angle in radians. 50 mm: below that a
#: positional error is a refinement, above it the fingers are somewhere else entirely, and the 2F-85's
#: half-aperture is 42.5 mm.
_LENGTH_SCALE_M: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class SetLossWeights:
    """How the four terms trade off, for matching and for fitting.

    The two sets are separate fields rather than one shared set, because they answer different
    questions and a future sweep will want to move them apart. They start equal, so a run made before
    anyone sweeps them is still one variable.
    """

    #: Matching. The approach leads, because it distinguishes two modes more often than the closing
    #: axis does, and the offset is deliberately not allowed to dominate: the several grasps at one
    #: point sit at nearly the same place, so a position-led cost cannot tell them apart at all.
    cost_approach: float = 1.0
    cost_axis: float = 0.5
    cost_offset: float = 0.5
    cost_width: float = 0.25

    #: Fitting. Same terms, and the offset weighs more here than in the cost: once a slot knows which
    #: grasp it is answering, hitting the position is most of the job. A position target that drops
    #: two of the three degrees of freedom breaks the round trip through the encoding.
    loss_approach: float = 1.0
    loss_axis: float = 1.0
    loss_offset: float = 1.0
    loss_width: float = 0.25
    #: The confidence term, over all slots rather than only matched ones. It is what makes an
    #: unmatched slot say so instead of emitting a plausible-looking grasp nobody asked for.
    loss_confidence: float = 0.5

    #: The affordance term. Zero when the head has no role output, which is the default, so a net
    #: without roles computes and adds nothing.
    #:
    #: 0.25 rather than 1.0, from what a full-weight role term did on the binned family: a 6-way
    #: cross-entropy there took a large share of the trunk's gradient for a head nothing read at
    #: inference. A role weighted at a quarter of the approach cannot starve the geometry.
    loss_part: float = 0.25

    def for_target(self, target: str) -> "SetLossWeights":
        """The same weights, with the approach removed when the run is not predicting it.

        Both the cost and the loss. Leaving the approach in the cost means the assignment still
        chooses which label a slot answers by a quantity the head is no longer trained on, so every
        other term is fitted against a label picked at random from the many a seed carries. Leaving
        it in the loss means the head is still spending capacity on a target measured to be
        unlearnable per seed.

        See `SetStepConfig.target` for the measurement behind this: at one seed the closing axis is
        effectively single-valued while the approach spreads widely.
        """
        if target == "approach":
            return self
        if target != "axis":
            raise ValueError(f"unknown target {target!r}; choose from approach, axis")
        return dataclasses.replace(self, cost_approach=0.0, loss_approach=0.0,
                                   cost_axis=1.0, loss_axis=1.0)


@dataclass(frozen=True, slots=True)
class GraspSetPrediction:
    """`K` grasps per seed, straight off the head. Raw outputs, not normalised or decoded.

    `axis_mode` travels with the prediction rather than being passed alongside it. The loss, the
    metrics and the decode all have to read the axis slots the same way, and a mode carried in three
    separate arguments is a mode three callers can disagree about. One of them would eventually
    supervise a director and score a vector, which is a defect that produces plausible numbers.
    """

    #: `(B, K, 3)` unnormalised approach direction.
    approach: torch.Tensor
    #: `(B, K, 6)` director parameters for the closing axis. See `rotation.director_params`.
    axis_params: torch.Tensor
    #: `(B, K, 3)` seed-to-centre offset in metres, in the seed-local cloud frame.
    offset_m: torch.Tensor
    #: `(B, K)` gripper opening in metres.
    width_m: torch.Tensor
    #: `(B, K)` logit, "this slot answers a real grasp".
    confidence: torch.Tensor
    #: How `axis_params` is to be read. Defaulted so every existing construction site is unchanged.
    axis_mode: str = "director"
    #: `(B, K, R)` logits over part roles, or None when the head has no affordance output.
    #:
    #: Optional, and `None` is the default. The roles come from a separate output layer rather than
    #: a wider `SLOT_OUTPUTS`, so `raw[..., 0:14]` keeps its meaning, every artifact already written
    #: keeps loading, and a net without roles has neither the tensor nor the parameters.
    part_logits: "torch.Tensor | None" = None


@dataclass(frozen=True, slots=True)
class GraspSetTarget:
    """The `L` grasps a seed admits, padded. `valid` marks the real ones."""

    approach: torch.Tensor      # (B, L, 3) unit
    axis: torch.Tensor          # (B, L, 3) unit, sign meaningless
    offset_m: torch.Tensor      # (B, L, 3)
    width_m: torch.Tensor       # (B, L)
    valid: torch.Tensor         # (B, L) bool
    #: `(B, L)` index into the role vocabulary, or None when the corpus carries no roles.
    #:
    #: Index 0 is the empty role, "this object has no parts to tell apart". It is a real class
    #: rather than a padding value: a box genuinely has no handle, and the head is trained to say
    #: so.
    part_index: "torch.Tensor | None" = None


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def grasp_set_cost(prediction: GraspSetPrediction, target: GraspSetTarget,
                   weights: SetLossWeights = SetLossWeights()) -> torch.Tensor:
    """`(B, K, L)` cost of answering label `l` with slot `k`.

    The axis term is antipodal here too. Matching on a signed axis would make a slot that is right
    about the closing line look maximally wrong whenever the corpus happened to store the other
    sign, which it does often. The matching would then be decided by bookkeeping.
    """
    approach = _unit(prediction.approach).unsqueeze(2)           # (B, K, 1, 3)
    label_approach = _unit(target.approach).unsqueeze(1)         # (B, 1, L, 3)
    approach_term = 1.0 - (approach * label_approach).sum(-1)

    # The director of the prediction against the director of the label. Frobenius distance in matrix
    # space is antipodal by construction, so no sign convention is needed anywhere.
    predicted_axis = director_params(prediction.axis_params).unsqueeze(2)
    label_axis = director_matrix(target.axis).unsqueeze(1)
    axis_term = (predicted_axis - label_axis).pow(2).flatten(start_dim=-2).sum(-1)

    offset_term = ((prediction.offset_m.unsqueeze(2) - target.offset_m.unsqueeze(1))
                   .norm(dim=-1) / _LENGTH_SCALE_M)
    width_term = ((prediction.width_m.unsqueeze(2) - target.width_m.unsqueeze(1)).abs()
                  / _LENGTH_SCALE_M)

    return (weights.cost_approach * approach_term
            + weights.cost_axis * axis_term
            + weights.cost_offset * offset_term
            + weights.cost_width * width_term)


def grasp_set_loss(prediction: GraspSetPrediction, target: GraspSetTarget,
                   weights: SetLossWeights = SetLossWeights(),
                   matching: Matching | None = None) -> dict[str, torch.Tensor]:
    """The four fitted terms plus the confidence, and the matching that produced them.

    Every term is returned separately as well as `total`, because a set head can improve its total
    while getting worse at the one thing a cell cares about, and a run logging only the sum cannot
    tell.

    `matching` may be passed in when a caller has already computed it, which the metrics do, so the
    loss and the coverage number are never computed from two different assignments.
    """
    cost = grasp_set_cost(prediction, target, weights)
    if matching is None:
        matching = optimal_one_to_one(cost, target.valid)

    matched = matching.slot_matched
    # `-1` would index the last label rather than none. Clamped, then masked: the values under
    # unmatched slots are read and thrown away, and reading garbage is fine while indexing out of
    # bounds is not.
    index = matching.slot_label.clamp_min(0)
    pairs = matched.sum().clamp_min(1).to(prediction.approach.dtype)

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        expanded = index.unsqueeze(-1).expand(*index.shape, tensor.shape[-1])
        return tensor.gather(1, expanded)

    chosen_approach = gather(target.approach)
    chosen_axis = gather(target.axis)
    chosen_offset = gather(target.offset_m)
    chosen_width = target.width_m.gather(1, index)

    mask = matched.to(prediction.approach.dtype)
    terms = {
        "approach": (approach_loss(prediction.approach, chosen_approach, reduction="none")
                     * mask).sum() / pairs,
        "axis": (axis_loss(prediction.axis_params, chosen_axis, mode=prediction.axis_mode,
                           reduction="none")
                 * mask).sum() / pairs,
        # The lengths are divided by the scale, not merely given it as a beta. `beta` moves
        # smooth-L1's transition point; it does not make a metre commensurable with a radian.
        # Passed as a beta instead, the two angular terms take nearly the whole total, the two
        # metric heads are left with almost none of the gradient budget, and the offset head never
        # leaves its initialisation: its error matches a random baseline's. `top1_hit` needs an
        # offset within 20 mm, so it stays near zero however good the approach becomes.
        "offset": ((F.smooth_l1_loss(prediction.offset_m / _LENGTH_SCALE_M,
                                     chosen_offset / _LENGTH_SCALE_M, reduction="none",
                                     beta=1.0).sum(-1)) * mask).sum() / pairs,
        "width": (F.smooth_l1_loss(prediction.width_m / _LENGTH_SCALE_M,
                                   chosen_width / _LENGTH_SCALE_M, reduction="none",
                                   beta=1.0) * mask).sum() / pairs,
        # Over every slot, matched or not. This is the only term an unmatched slot receives, and the
        # only one that can teach a head to leave a slot empty.
        "confidence": F.binary_cross_entropy_with_logits(
            prediction.confidence, confidence_target(matching).to(prediction.confidence.dtype)),
    }
    # The part role, on matched pairs only, and only when both sides carry one.
    #
    # Both sides. A head with roles trained on a corpus without them would fit noise, and a corpus
    # with roles under a head without them would silently drop supervision that is already in the
    # files. Neither is an error worth raising: an operator may legitimately train a role-less head
    # on a role-carrying corpus.
    #
    # Matched pairs only, exactly like the four geometric terms. An unmatched slot answers no
    # label, so it has no role to be right or wrong about, and averaging one in would grade the head
    # on a target that does not exist. `confidence` remains the only term over every slot.
    if prediction.part_logits is not None and target.part_index is not None:
        chosen_part = target.part_index.gather(1, index)
        per_slot = F.cross_entropy(
            prediction.part_logits.reshape(-1, prediction.part_logits.shape[-1]),
            chosen_part.reshape(-1), reduction="none").reshape(chosen_part.shape)
        terms["part"] = (per_slot * mask).sum() / pairs
    terms["total"] = (weights.loss_approach * terms["approach"]
                      + weights.loss_axis * terms["axis"]
                      + weights.loss_offset * terms["offset"]
                      + weights.loss_width * terms["width"]
                      + weights.loss_confidence * terms["confidence"])
    if "part" in terms:
        terms["total"] = terms["total"] + weights.loss_part * terms["part"]
    # Coverage is not computed here. It lives in `train.step._metrics`, where the hit thresholds
    # are, and counts only the labels whose matched slot is actually close.
    #
    # `labels matched / labels valid` is not a coverage metric: the matching is one-to-one and
    # unconstrained, so it always binds `min(K, L)` pairs however badly they fit, and the ratio is a
    # pure function of the label counts and K, identical for every model.
    return terms
