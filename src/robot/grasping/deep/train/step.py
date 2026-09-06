"""One training step of the set generator, and what it reports.

The wiring between `corpus.sample.build_sample` and everything under `net/`. A plain function
rather than a class: a step that owns state is a step whose result depends on how many times it has
been called.

A list of samples, not a collated batch. Every sample carries its own grasp table of its own length
and its own sparse `(point, grasp)` pair list. Collating those into padded blocks would mean a mask
threaded through the gather, the loss and both metrics. Encoding is batched, because the clouds are
all `points` long; everything downstream is a flat list of seeds. That is where the ragged part
belongs.

What is reported, and why more than the total:

    total, approach, axis, offset, width, confidence   the loss and its parts
    graspability                                       the where stage, kept separate
    coverage                                           metric two of four
    top1_hit                                           metric three, comparable to the old arms
    approach_error_deg, offset_error_mm                what the numbers mean in units a cell has
    truncated, seeds_*                                 what the step actually drew

A set head can improve its total while getting worse at the one thing a cell cares about, and a run
that logged only the sum could not tell: one term can move while another cancels it.

The referee success rate, metric one, is not here. It needs a physics verdict on the head's own
proposals, which is an evaluation-time number that arrives after a shake run; a training step
cannot compute it, and reporting a proxy for it under its name would be the worst of both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812 (the conventional alias)

from src.robot.grasping.deep.net.assignment import optimal_one_to_one
from src.robot.grasping.deep.net.rotation import angular_error_deg, axis_decode
from src.robot.grasping.deep.net.set_generator import SetGenerator
from src.robot.grasping.deep.net.set_loss import (
    SetLossWeights,
    grasp_set_cost,
    grasp_set_loss,
)
from src.robot.grasping.deep.net.set_targets import (
    GraspTable,
    SeedShares,
    gather_set_targets,
    sample_seeds,
)

__all__ = ["SetStepConfig", "set_training_step"]

_M_TO_MM = 1000.0


@dataclass(frozen=True, slots=True)
class SetStepConfig:
    """The knobs of a step that are not part of the network itself."""

    #: Seeds drawn per sample. Capped by how many candidate points a sample has, so a sparse scene
    #: yields fewer and that is reported rather than padded.
    seeds: int = 64
    #: Labels carried per seed. A seed can carry more labels than this, so the tail is truncated
    #: and `truncated` says how often.
    max_labels: int = 16
    shares: SeedShares = SeedShares()
    weights: SetLossWeights = SetLossWeights()
    #: How much the where stage weighs against the what stage. Reported separately either way, so a
    #: run can see which one moved.
    graspability_weight: float = 1.0
    #: What counts as a hit for `top1_hit` and for coverage. Fifteen degrees is the separation the
    #: corpus measurement used to call two approaches distinct, and 20 mm is under half a 2F-85
    #: half-aperture, so it is the physically meaningful criterion.
    hit_angle_deg: float = 15.0
    hit_offset_mm: float = 20.0

    #: Extra distance thresholds, reported beside the main one.
    #:
    #: The 20 mm criterion sits at the edge of what is achievable, so a head still on its way in
    #: from further out shows `top1_hit = 0` for an entire run, which is a metric that cannot
    #: display progress.
    #:
    #: The main threshold stays 20 mm because it is the one a cell cares about; the others exist so
    #: a curve can be read while it is still climbing.
    extra_hit_offsets_mm: tuple[float, ...] = (30.0, 40.0)

    #: Which direction the run is actually trying to predict, and it is the most consequential
    #: field in this file. Default `approach`.
    #:
    #: A seed that carries several labels carries many approaches around one shared closing axis.
    #: The approach is therefore not a function of the seed at all, and a per-seed head regressing
    #: it has an irreducible floor no matter how much data it is given. The offset is local geometry
    #: (a contact to a centre, half an opening away) and the approach is a free parameter the
    #: labeller sampled.
    #:
    #: Further observations say the same thing from other directions: the approach is perpendicular
    #: to the closing axis, the approaches at one seed sit on a rotational grid about it, and
    #: neither "the perpendicular closest to straight down" nor "the perpendicular with the freest
    #: palm corridor" predicts which one was chosen.
    #:
    #: `axis` moves the target to the quantity that is a function of the seed. The approach then
    #: belongs where it already lives: the collision, IK and reachability layer, which searches for a
    #: reachable approach anyway and can do it better than a head that would be guessing.
    #:
    #: It moves the hit criterion too. `top1_hit` gates on an angle, and under `approach` that
    #: angle is the approach angle. Leaving it there while training the axis reports a flat zero for
    #: a head doing exactly what was asked. Both angles are reported under both targets, so the two
    #: arms stay comparable.
    target: str = "approach"


def _table_of(sample: dict[str, Any], device: torch.device) -> GraspTable:
    def get(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(sample[key]), dtype=torch.float32, device=device)

    roles = sample.get("set_grasp_part_index")
    return GraspTable(position_m=get("set_grasp_position_m"), approach=get("set_grasp_approach"),
                      axis=get("set_grasp_axis"), width_m=get("set_grasp_width_m"),
                      # `.get`, not `[...]`: a sample built without this key is a sample
                      # that trains a role-less head.
                      part_index=(torch.as_tensor(np.asarray(roles), dtype=torch.long,
                                                  device=device)
                                  if roles is not None else None))



def _padded_batch(samples: list[dict[str, Any]],
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack samples of different point counts into one block, and say which rows are real.

    Short clouds are rare and fatal without this. `build_sample` returns
    `min(budget, points available)`, so a cloud holding fewer points than the budget is shorter than
    the rest of its batch. A plain `torch.stack` refuses those.

    The two arrays are padded differently, and that is the whole care in this function.

    `points` and `features` are repeated cyclically. A short cloud is padded with copies of its own
    points spread evenly through it, which the encoder sees as the same geometry slightly denser.
    Padding with the last point instead would pile hundreds of copies at one place and put a spike
    into the cloud that is not in the scene.

    `supervise` is padded with `False`, never cyclically. A duplicated supervised point would be a
    second seed candidate at the same position: `sample_seeds` would call the two distinct because
    their indices differ, and the loss and the coverage metric would both count that place twice.
    """
    counts = [int(len(s["points_m"])) for s in samples]
    width = max(counts)
    points, features, supervise = [], [], []
    for sample, count in zip(samples, counts, strict=True):
        take = np.arange(width) % count
        points.append(torch.as_tensor(np.asarray(sample["points_m"])[take], dtype=torch.float32,
                                      device=device))
        features.append(torch.as_tensor(np.asarray(sample["features"])[take], dtype=torch.float32,
                                        device=device))
        real = np.zeros(width, dtype=bool)
        real[:count] = np.asarray(sample["supervise"], dtype=bool)
        supervise.append(torch.as_tensor(real, dtype=torch.bool, device=device))
    return torch.stack(points), torch.stack(features), torch.stack(supervise)


def set_training_step(net: SetGenerator, samples: list[dict[str, Any]], gripper: torch.Tensor, *,
                      config: SetStepConfig | None = None,
                      generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Encode, draw seeds, propose, score. Returns every term and every metric, `total` among them.

    `gripper` is `(B, 14)`, one row per sample, and it is expanded to one row per seed here, which is
    the only place that expansion is allowed to happen.

    Per sample rather than per batch, because a corpus can hold several grippers and every cloud
    records which jaw its labels came from. A batch that mixes them is the case the conditioning has
    to learn from: the same geometry, a different hand, a different correct answer. Requiring one
    vector per batch would have forced the loop to group by gripper and would have removed exactly
    the contrast the seam exists to pick up.

    The seeds are drawn from a detached field. The graspability head is trained by its own loss; if
    the draw carried a gradient the head would learn to propose seeds that are easy to answer rather
    than seeds worth grasping, which is a reward for looking away.
    """
    config = config or SetStepConfig()
    if not samples:
        raise ValueError("a step needs at least one sample")
    if gripper.dim() != 2 or gripper.shape[0] != len(samples):
        raise ValueError(f"gripper must be ({len(samples)}, 14), one row per sample, "
                         f"got {tuple(gripper.shape)}")

    device = next(net.parameters()).device
    points, features, supervised = _padded_batch(samples, device)

    encoded = net.encode(points, features)
    field = net.graspability_logit(encoded)

    batch_index: list[torch.Tensor] = []
    point_index: list[torch.Tensor] = []
    targets: list[Any] = []
    truncated = 0
    drawn = {"labelled": 0, "predicted": 0, "random": 0}
    for row, sample in enumerate(samples):
        supervise = supervised[row]
        pair_point = torch.as_tensor(np.asarray(sample["set_pair_point"]), dtype=torch.int64,
                                     device=device)
        pair_grasp = torch.as_tensor(np.asarray(sample["set_pair_grasp"]), dtype=torch.int64,
                                     device=device)
        labelled = torch.zeros_like(supervise)
        if pair_point.numel():
            labelled[pair_point] = True

        draw = sample_seeds(count=config.seeds, labelled=labelled, score=field[row].detach(),
                            candidate=supervise, shares=config.shares, generator=generator)
        if not draw.point_index.numel():
            continue
        target, cut = gather_set_targets(
            draw.point_index, pair_point, pair_grasp, _table_of(sample, device), points[row],
            max_labels=config.max_labels, generator=generator)
        truncated += cut
        drawn["labelled"] += draw.from_labelled
        drawn["predicted"] += draw.from_predicted
        drawn["random"] += draw.from_random
        batch_index.append(torch.full_like(draw.point_index, row))
        point_index.append(draw.point_index)
        targets.append(target)

    if not targets:
        raise ValueError("no sample yielded a seed; every scene had an empty candidate mask")

    flat_batch = torch.cat(batch_index)
    flat_point = torch.cat(point_index)
    # Every field is listed by hand, so a target field added upstream and not added here is dropped
    # silently: the loss sees `part_index=None`, adds no term, nothing raises, and the epoch row
    # simply has no `part` key.
    #
    # All-or-nothing across the batch. A batch mixing samples with and without roles would give
    # `torch.cat` a ragged list; there is no such mix in practice, because the key is emitted for
    # every sample once `grasp_set` is on, but relying on that silently is how the next gap starts.
    part = ([t.part_index for t in targets]
            if all(t.part_index is not None for t in targets) else None)
    target = type(targets[0])(
        approach=torch.cat([t.approach for t in targets]),
        axis=torch.cat([t.axis for t in targets]),
        offset_m=torch.cat([t.offset_m for t in targets]),
        width_m=torch.cat([t.width_m for t in targets]),
        valid=torch.cat([t.valid for t in targets]),
        part_index=(torch.cat(part) if part is not None else None))  # type: ignore[arg-type]

    # One gripper row per seed, gathered from the sample each seed came from. `flat_batch` already
    # says which sample that is, so the expansion is a gather rather than a broadcast and a mixed
    # batch stays correct without anybody having to sort it.
    # The cloud goes through too, because stage 3 crops it. `propose` refuses rather than silently
    # skipping the crop when a crop-enabled net is called without it, so this is the one place that
    # decides whether the flag does anything at all: an inert `--crop-mm` would train an ordinary head
    # and stamp a crop into its card.
    prediction = net.propose(encoded, flat_batch, flat_point, gripper.to(device)[flat_batch],
                             points)
    # One resolution, used for both. Deriving the cost weights and the loss weights separately is
    # how a run ends up matching on one quantity and fitting another.
    weights = config.weights.for_target(config.target)
    if getattr(net, "generative", None) is not None:
        # Stage 4b. The pose loss is a denoising objective over every valid label, and the metrics
        # are computed on samples scored by the same `_metrics` the baseline uses, at the same
        # thresholds. Two things are deliberately identical between the arms: the seed features and
        # the metric code. Everything else is the architecture under test.
        #
        # `top1_hit` means something different here and the report has to say so: a generative head
        # emits no confidence, so the "first" sample is the first drawn rather than the most trusted.
        # Coverage is directly comparable; top-1 is comparable only as "what a cell reaches for".
        from src.robot.grasping.deep.net.generative_loss import (  # noqa: PLC0415
            generative_loss,
            samples_as_prediction,
        )

        head = net.generative
        assert head is not None                      # narrowed for the type checker, not a guard
        seed_features = net.seed_features(encoded, flat_batch, flat_point, points)
        seed_gripper = gripper.to(device)[flat_batch]
        terms = generative_loss(head, seed_features, seed_gripper, target, generator=generator)
        with torch.no_grad():
            # Not `drawn` and not `samples`. `drawn` already holds the seed-source counters the
            # report reads, and `samples` is the function's own argument, the list of scene dicts;
            # rebinding either silently replaces what a later line depends on, and nothing at
            # runtime says so until a report shows no seed counts at all.
            poses = head.sample(seed_features, seed_gripper,
                                count=net.config.head.slots, generator=generator)
            prediction = samples_as_prediction(poses)
            matching = optimal_one_to_one(
                grasp_set_cost(prediction, target, weights), target.valid)
    else:
        matching = optimal_one_to_one(grasp_set_cost(prediction, target, weights), target.valid)
        terms = grasp_set_loss(prediction, target, weights, matching=matching)

    # The where stage, trained by its own loss and reported on its own. Points that no grasp reaches
    # are the true negatives it exists for; masking them out would leave a head that says yes
    # everywhere.
    field_target = torch.zeros_like(field)
    for row, sample in enumerate(samples):
        pair_point = torch.as_tensor(np.asarray(sample["set_pair_point"]), dtype=torch.int64,
                                     device=device)
        if pair_point.numel():
            field_target[row, pair_point] = 1.0
        # Suction counts as graspable here and nowhere else. The pose heads above are jaw-only
        # by construction; this stage answers "is this bit of surface worth attempting", and a cell
        # that carries a cup can attempt a suction-only object. Teaching the field to call such an
        # object empty is a false negative on every object a cup could lift and a jaw could not.
        #
        # Present only in a corpus built with `--kinds both`. Absent, this is a no-op and the run is
        # byte-identical.
        suction = sample.get("graspability_suction")
        if suction is not None:
            positive = torch.as_tensor(np.asarray(suction) > 0.5, dtype=torch.bool, device=device)
            field_target[row, : positive.shape[0]] = torch.maximum(
                field_target[row, : positive.shape[0]], positive.to(field_target.dtype))
    graspability = F.binary_cross_entropy_with_logits(
        field[supervised], field_target[supervised]) if bool(supervised.any()) else field.sum() * 0.0

    out = dict(terms)
    out["graspability"] = graspability
    out["total"] = terms["total"] + config.graspability_weight * graspability
    out.update(_metrics(prediction, target, matching, config))
    out["truncated"] = torch.tensor(float(truncated), device=device)
    for source, count in drawn.items():
        out[f"seeds_{source}"] = torch.tensor(float(count), device=device)
    return out


def _metrics(prediction: Any, target: Any, matching: Any,
             config: SetStepConfig) -> dict[str, torch.Tensor]:
    """The reported numbers that are not losses, in units a cell can act on.

    `top1_hit` asks about the slot the head is most confident in, not about its best one. That is
    the question a cell asks: it receives ranked candidates and takes the first. Scoring the best of
    K would report an oracle over the head's own output.
    """
    with torch.no_grad():
        confidence = prediction.confidence
        best = confidence.argmax(dim=1)
        rows = torch.arange(confidence.shape[0], device=confidence.device)
        chosen_approach = prediction.approach[rows, best]
        chosen_offset = prediction.offset_m[rows, best]
        decoded_axis = axis_decode(prediction.axis_params, mode=prediction.axis_mode)
        chosen_axis = decoded_axis[rows, best]

        # The gate follows the target. See `SetStepConfig.target`: gating on the approach while
        # training the axis reports a flat zero for a head doing exactly what was asked.
        on_axis = config.target == "axis"
        angle = (angular_error_deg(chosen_axis.unsqueeze(1), target.axis, antipodal=True)
                 if on_axis else
                 angular_error_deg(chosen_approach.unsqueeze(1), target.approach, antipodal=False))
        distance = (chosen_offset.unsqueeze(1) - target.offset_m).norm(dim=-1) * _M_TO_MM
        close = (angle <= config.hit_angle_deg) & (distance <= config.hit_offset_mm) & target.valid
        has_label = target.valid.any(dim=1)

        matched = matching.slot_matched
        index = matching.slot_label.clamp_min(0)
        pairs = matched.sum().clamp_min(1)
        matched_approach = angular_error_deg(
            prediction.approach,
            target.approach.gather(1, index.unsqueeze(-1).expand(*index.shape, 3)),
            antipodal=False)
        matched_offset = (prediction.offset_m - target.offset_m.gather(
            1, index.unsqueeze(-1).expand(*index.shape, 3))).norm(dim=-1) * _M_TO_MM
        # Reported under both targets, always. A number that exists in one arm and not the other
        # cannot compare them, and comparing them is the entire point of having a target switch.
        matched_axis = angular_error_deg(
            decoded_axis,
            target.axis.gather(1, index.unsqueeze(-1).expand(*index.shape, 3)),
            antipodal=True)
        # Sums and counts, never ratios. A batch in which no seed carries a label has no
        # `top1_hit` to report: the metric is undefined there, not zero. A ratio makes the caller
        # average an undefined batch in as a nought, which drags a perfect head's ceiling far below
        # 1.0. A filtered metric has to be reported over its own denominator, paired, so a mean over
        # batches can weight it.
        #
        # Coverage counts only the labels whose matched slot actually landed on them. Counting every
        # matched pair instead says nothing: the matching is one-to-one and unconstrained, so it
        # binds `min(K, L)` pairs however badly they fit, and unrelated random models then all score
        # the same number, a pure function of the label counts and K. With the hit criterion
        # applied, a random model scores near zero and a perfect one reaches the ceiling.
        pair_angle = matched_axis if on_axis else angular_error_deg(
            prediction.approach,
            target.approach.gather(1, index.unsqueeze(-1).expand(*index.shape, 3)),
            antipodal=False)
        pair_distance = (prediction.offset_m - target.offset_m.gather(
            1, index.unsqueeze(-1).expand(*index.shape, 3))).norm(dim=-1) * _M_TO_MM
        landed = (matched & (pair_angle <= config.hit_angle_deg)
                  & (pair_distance <= config.hit_offset_mm))
        # The same question at looser distances, so a head on its way to 20 mm is visible rather
        # than reported as a flat zero. The angle criterion is unchanged: only the distance moves.
        extra: dict[str, torch.Tensor] = {}
        for limit in config.extra_hit_offsets_mm:
            wider = ((angle <= config.hit_angle_deg) & (distance <= limit) & target.valid)
            extra[f"top1_hit_{limit:.0f}mm_sum"] = (
                wider.any(dim=1) & has_label).sum().to(torch.float32)
            extra[f"top1_hit_{limit:.0f}mm_count"] = has_label.sum().to(torch.float32)
        return {
            **extra,
            "coverage_sum": landed.sum().to(torch.float32),
            "coverage_count": target.valid.sum().to(torch.float32),
            "top1_hit_sum": (close.any(dim=1) & has_label).sum().to(torch.float32),
            "top1_hit_count": has_label.sum().to(torch.float32),
            "approach_error_deg_sum": (matched_approach * matched).sum(),
            "approach_error_deg_count": pairs.to(matched_approach.dtype),
            "axis_error_deg_sum": (matched_axis * matched).sum(),
            "axis_error_deg_count": pairs.to(matched_axis.dtype),
            "offset_error_mm_sum": (matched_offset * matched).sum(),
            "offset_error_mm_count": pairs.to(matched_offset.dtype),
            # A diagnostic, not a quality metric. It is `min(K, L)` averaged over seeds, so it is
            # identical for every model and says only how many slots the data gives a head to fill.
            # Kept because that is worth watching, labelled because it would otherwise be read as
            # a score.
            "slots_filled_sum": matched.sum().to(torch.float32),
            "slots_filled_count": torch.tensor(float(matched.shape[0]),
                                               device=matched.device),
            **_spread(prediction, chosen_approach),
        }


def _spread(prediction: Any, chosen_approach: torch.Tensor) -> dict[str, torch.Tensor]:
    """How much the head's own output varies. Three numbers, and they are diagnostics, not scores.

    They cost one extra reduction over tensors already in hand, and they are the only instrument
    that names a collapsed head directly. A collapsed head proposes one direction for every seed,
    with widths and confidences that barely differ across its own output. Every quality metric
    reports that honestly as a low number without being able to say why.

        slot_spread_deg   how far a seed's K slots differ from each other. Near zero means the slots
                          are one grasp copied K times, which is what `affine` mixing produces, and
                          it caps coverage at the value of a single slot however good that slot is.
        seed_spread_deg   how far different seeds differ from each other. Near zero means one global
                          direction for the whole scene, which is the collapse above.
        width_spread_mm   the same question for the width head, in millimetres.

    Not a quality metric in either direction. A large spread is not skill: a random head has the
    largest spread of all. It is only ever read as "the head can still tell its inputs apart", and a
    near-zero value is the finding.
    """
    seeds, slots = prediction.approach.shape[:2]
    # Against each seed's own mean direction rather than against slot 0. A frozen head is identical
    # either way, but a head with one outlying slot reads as noisy against slot 0 and correctly as
    # varied against the mean.
    slot_mean = prediction.approach.mean(dim=1, keepdim=True).expand_as(prediction.approach)
    slot_spread = (angular_error_deg(prediction.approach, slot_mean, antipodal=False)
                   if slots > 1 else torch.zeros_like(prediction.approach[..., 0]))
    seed_mean = chosen_approach.mean(dim=0, keepdim=True).expand_as(chosen_approach)
    seed_spread = angular_error_deg(chosen_approach, seed_mean, antipodal=False)
    width_mm = prediction.width_m * _M_TO_MM
    width_spread = (width_mm - width_mm.mean()).abs()
    # A mean direction can be the zero vector when directions cancel, and `_unit` then divides by a
    # clamped epsilon and returns an arbitrary angle. Only the near-zero end of this metric is ever
    # read, and that end is unaffected, but the NaN guard is here so a cancelled batch cannot poison
    # an epoch's sum.
    slot_spread = torch.nan_to_num(slot_spread, nan=0.0)
    seed_spread = torch.nan_to_num(seed_spread, nan=0.0)
    return {
        "slot_spread_deg_sum": slot_spread.sum(),
        "slot_spread_deg_count": torch.tensor(float(seeds * slots),
                                              device=slot_spread.device),
        "seed_spread_deg_sum": seed_spread.sum(),
        "seed_spread_deg_count": torch.tensor(float(seeds), device=seed_spread.device),
        "width_spread_mm_sum": width_spread.sum(),
        "width_spread_mm_count": torch.tensor(float(width_spread.numel()),
                                              device=width_spread.device),
    }
