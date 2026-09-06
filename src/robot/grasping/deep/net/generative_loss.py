"""The denoising objective that makes stage 4b trainable, and the sampler that makes it comparable.

A supervised point usually reaches more than one label, and often admits several approaches far
enough apart to be distinct grasps. The deterministic head answers with K slots and a one-to-one
matching, which binds at most K of them, so its coverage cannot reach 1.0 wherever a seed carries
more.

So the loss trains on every label, not on a matched subset, and that is the whole difference.
Running an assignment here would reimpose the bound the arm exists to escape, and the comparison
would be decided by the wiring rather than by the architecture. A seed with fifteen labels
contributes fifteen training examples.

It is an arm, not a replacement. The plan's risk row states the falsifier: beating 4a on seen
assets and losing on unseen means memorisation, and a small asset bank against a large parameter
count is exactly that regime.
"""

from __future__ import annotations

from typing import Final

import torch

from src.robot.grasping.deep.net.generative_head import (
    POSE_DIM,
    GenerativeGraspHead,
    _draw,
)
from src.robot.grasping.deep.net.slot_head import SlotHeadConfig
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction, GraspSetTarget

__all__ = ["pose_from_target", "generative_loss", "decode_samples", "samples_as_prediction",
           "SEPARATE_SCHEDULES"]

#: Translation and rotation are noised independently, which outside work measured as better than one
#: schedule and which the units make obvious: an offset is metres and an axis is a direction. The
#: head is told both levels and must read both.
SEPARATE_SCHEDULES: Final[bool] = True

#: How the ten numbers are laid out. Named rather than sliced inline, because a decode that disagrees
#: with the encode by one column is a defect that produces plausible poses.
_APPROACH = slice(0, 3)
_AXIS = slice(3, 6)
_OFFSET = slice(6, 9)
_WIDTH = slice(9, 10)


def pose_from_target(target: GraspSetTarget) -> tuple[torch.Tensor, torch.Tensor]:
    """`(B, L, POSE_DIM)` poses and their `(B, L)` validity, straight from the label set.

    The axis keeps its sign here. The director representation exists to make a loss sign-invariant,
    and this head predicts noise: a sign it cannot see is a sign it cannot denoise. Sign invariance
    comes out of the sampling instead, because a distribution covering both signs is what a
    generative head is for. The corpus stores both signs, so both are represented.

    The width is centred and scaled by the deterministic head's own constants, imported from
    `SlotHeadConfig` rather than copied. In metres it sits two orders of magnitude below the unit
    directions, and one noise schedule over mixed scales denoises the small column at the wrong
    rate.

    The constants are the baseline's rather than better ones: a centre and scale fitted to the
    corpus width distribution would map it onto the same band as a unit direction and 70 / 50 does
    not. An arm that differs from 4a in two things measures neither, so the normalisation is its
    own sweep.
    """
    pose = torch.cat([
        target.approach,
        target.axis,
        target.offset_m,
        (target.width_m.unsqueeze(-1) - _WIDTH_CENTRE_M) / _WIDTH_SCALE_M,
    ], dim=-1)
    return pose, target.valid


#: Imported from the baseline, not restated. Two copies of a normalisation drift, and when they do
#: the arm is comparing two encodings rather than two architectures. `SlotHeadConfig` owns these.
_WIDTH_CENTRE_M: Final[float] = SlotHeadConfig().width_centre_m
_WIDTH_SCALE_M: Final[float] = SlotHeadConfig().width_scale_m


def generative_loss(head: GenerativeGraspHead, features: torch.Tensor, gripper: torch.Tensor,
                    target: GraspSetTarget, *,
                    generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Noise every valid label, ask the head for the noise, and score it.

    Every valid label, flattened. A seed with fifteen labels contributes fifteen examples. Taking
    one per seed would train a head that has seen one mode of the several a seed carries, and
    taking a matched subset would reimpose the K bound it exists to escape.

    The denominator is the label count, not the seed count and not the batch size. A batch whose
    seeds carry no label at all contributes nothing rather than a zero, the same denominator rule
    the deterministic metrics follow: counting such a batch as a zero reports a ceiling well below
    the truth.
    """
    pose, valid = pose_from_target(target)
    seeds, labels = valid.shape
    flat_valid = valid.reshape(-1)
    if not bool(flat_valid.any()):
        zero = features.sum() * 0.0
        return {"total": zero, "noise_mse_sum": zero,
                "noise_mse_count": torch.zeros((), device=features.device)}

    # One row per (seed, label), keeping only the real ones.
    wide_features = features.unsqueeze(1).expand(seeds, labels, features.shape[-1])
    wide_gripper = gripper.unsqueeze(1).expand(seeds, labels, gripper.shape[-1])
    rows = flat_valid.nonzero(as_tuple=True)[0]
    clean = pose.reshape(-1, POSE_DIM)[rows]
    take_features = wide_features.reshape(-1, features.shape[-1])[rows]
    take_gripper = wide_gripper.reshape(-1, gripper.shape[-1])[rows]

    steps = head.config.steps
    device = clean.device
    # Through `_draw`, because the training loop's generator is a CPU one and the tensors are on
    # CUDA. See its docstring: torch refuses the mismatch, and drawing on the generator's device also
    # makes a seeded run reproducible across machines rather than only against itself.
    level_translation = _draw((len(clean),), device=device, kind="randint", high=steps,
                              generator=generator)
    level_rotation = _draw((len(clean),), device=device, kind="randint", high=steps,
                           generator=generator)
    noise = _draw(tuple(clean.shape), device=device, generator=generator)

    # The two schedules are applied to their own columns. Applying one level to all ten numbers is
    # the defect the separate embeddings exist to prevent, and it would be invisible: the head would
    # still train, just at the wrong rate for six of the ten.
    noised = clean.clone()
    rotation_scale = _alpha(level_rotation, steps).unsqueeze(-1)
    translation_scale = _alpha(level_translation, steps).unsqueeze(-1)
    for span, scale in ((_APPROACH, rotation_scale), (_AXIS, rotation_scale),
                        (_OFFSET, translation_scale), (_WIDTH, translation_scale)):
        noised[:, span] = clean[:, span] * scale + noise[:, span] * (1.0 - scale)

    predicted = head(take_features, take_gripper, noised,
                     level_translation.to(torch.float32), level_rotation.to(torch.float32))
    per_row = (predicted - noise).pow(2).mean(dim=-1)
    return {
        "total": per_row.mean(),
        "noise_mse_sum": per_row.sum(),
        "noise_mse_count": torch.tensor(float(len(per_row)), device=device),
    }


def _alpha(level: torch.Tensor, steps: int) -> torch.Tensor:
    """How much of the clean pose survives at this noise level. Linear, and no claim is made for it.

    The schedule is the simplest one that can work. A cosine schedule is the usual choice and would
    be a second thing changed between this arm and its baseline; if the arm is worth keeping, the
    schedule becomes its own sweep.
    """
    return 1.0 - level.to(torch.float32) / float(steps)


def decode_samples(drawn: torch.Tensor) -> GraspSetTarget:
    """`(S, N, POSE_DIM)` samples as a label-shaped set. See `samples_as_prediction` for scoring."""
    if drawn.dim() != 3 or drawn.shape[-1] != POSE_DIM:
        raise ValueError(f"expected (S, N, {POSE_DIM}) samples, got {tuple(drawn.shape)}")
    approach = torch.nn.functional.normalize(drawn[..., _APPROACH], dim=-1)
    axis = torch.nn.functional.normalize(drawn[..., _AXIS], dim=-1)
    width = drawn[..., _WIDTH].squeeze(-1) * _WIDTH_SCALE_M + _WIDTH_CENTRE_M
    return GraspSetTarget(
        approach=approach, axis=axis, offset_m=drawn[..., _OFFSET], width_m=width,
        valid=torch.ones(drawn.shape[:2], dtype=torch.bool, device=drawn.device))


def samples_as_prediction(drawn: torch.Tensor) -> GraspSetPrediction:
    """The same samples, shaped so the deterministic metrics score them unchanged.

    The point is comparability. Coverage and top-1 are computed by `_metrics` over a `(B, K, ...)`
    prediction; handing it N samples where it expects K slots means both arms are scored by the same
    code at the same thresholds. An arm with its own metric is not an arm.

    The confidence is uniform, and that is an asymmetry rather than a detail. The deterministic
    head emits a per-slot confidence and `top1_hit` asks about the slot it is most confident in,
    because that is the one a cell would take first. A generative head has no such output: it draws
    samples, and nothing in it says which draw is better. So `top1_hit` for this arm means "the
    first sample drawn", which is still what a cell would take, and it is not the same question as
    the baseline's. Coverage is directly comparable; top-1 is comparable only in the sense that both
    describe the candidate a cell reaches for.

    The repair is stage 5: a scorer fitted on the physics referee would rank a generative head's
    samples the way confidence ranks the baseline's. Until that is fitted, this asymmetry travels
    with every number this arm produces.
    """
    decoded = decode_samples(drawn)
    return GraspSetPrediction(
        approach=decoded.approach,
        # `vector` mode reads the first three columns as the axis directly, no eigendecomposition, so
        # a sampled direction survives the decode exactly.
        axis_params=torch.cat(
            [decoded.axis, torch.zeros_like(decoded.axis)], dim=-1),
        offset_m=decoded.offset_m,
        width_m=decoded.width_m,
        confidence=torch.zeros(decoded.width_m.shape, device=drawn.device),
        axis_mode="vector")
