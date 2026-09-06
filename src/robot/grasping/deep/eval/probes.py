"""Did the model learn, or did it memorise its asset groups?

The acceptance criterion for scaling the generator up. It is built before the first result is read:
a threshold chosen after seeing the number it judges is not a threshold.

This generator has many parameters and the corpus has few asset groups. Outside work reports that a
high-capacity model on a small dataset approximately memorises rather than generalises, ending up
as a lookup table whose inference overlays its training data, so the question needs an instrument
rather than an opinion.

The fold key is `asset_group` and not the asset id. It folds the procedurally generated variants of
one object into a single group, because "can it grasp a jug it has never seen" is the honest
question and "can it grasp jug seven having seen jugs one to six" is not.

What is measured: the trained model, in eval mode, on two matched samples.

    seen      units drawn from the training fold
    unseen    units drawn from the held-out fold

Same count, same sampler, same seed. The gap between them is the probe.

The gap alone means nothing without a floor, so an untrained copy of the same architecture is
measured the same way in the same call. An untrained net has no reason to treat the two sets
differently, so its gap is what "no memorisation" looks like on this corpus with this metric and
this number of units. A trained gap is read against that, not against zero: the two folds are
different assets and a small difference is sampling, not recall.

No pass/fail threshold is hard-coded. Nothing has been measured yet, so any number written here
would be invented. The probe reports; the first run establishes what normal looks like; the
threshold is then written down with the measurement that produced it.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.robot.grasping.deep.train.trainer import (
    CorpusIndex,
    SetTrainingPlan,
    _batch_samples,
)
from src.robot.grasping.deep.net.set_generator import SetGenerator
from src.robot.grasping.deep.train.step import set_training_step

__all__ = ["ProbeResult", "baseline_floor", "memorisation_probe", "oracle_ceiling"]

#: The numbers the gap is reported over. Deliberately not just the loss: a model can memorise its way
#: to a lower total while getting worse at the thing a cell reads.
_TRACKED = ("total", "coverage_sum", "coverage_count", "top1_hit_sum", "top1_hit_count",
            "approach_error_deg_sum", "approach_error_deg_count",
            "offset_error_mm_sum", "offset_error_mm_count",
            "slots_filled_sum", "slots_filled_count")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Seen against unseen, for the trained model and for an untrained control."""

    seen: dict[str, float]
    unseen: dict[str, float]
    control_seen: dict[str, float]
    control_unseen: dict[str, float]
    units: int
    seconds: float

    @property
    def gap(self) -> dict[str, float]:
        """`seen - unseen`, per tracked metric. Positive means the model does better on what it saw."""
        return {key: self.seen[key] - self.unseen[key] for key in self.seen}

    @property
    def control_gap(self) -> dict[str, float]:
        """The same difference for an untrained net: what sampling alone produces on these folds."""
        return {key: self.control_seen[key] - self.control_unseen[key] for key in self.control_seen}

    @property
    def excess_gap(self) -> dict[str, float]:
        """The number to read. The trained gap with the untrained one subtracted off.

        The two folds hold different assets, so some difference exists before any training does.
        What is attributable to learning is what exceeds that; the raw gap would credit the model
        with the corpus's own asymmetry.
        """
        control = self.control_gap
        return {key: value - control.get(key, 0.0) for key, value in self.gap.items()}

    def as_row(self) -> dict[str, Any]:
        return {"units": self.units, "seconds": round(self.seconds, 1),
                **{f"seen_{k}": round(v, 5) for k, v in self.seen.items()},
                **{f"unseen_{k}": round(v, 5) for k, v in self.unseen.items()},
                **{f"gap_{k}": round(v, 5) for k, v in self.gap.items()},
                **{f"control_gap_{k}": round(v, 5) for k, v in self.control_gap.items()},
                **{f"excess_{k}": round(v, 5) for k, v in self.excess_gap.items()}}


def _evaluate(net: SetGenerator, index: CorpusIndex, units: np.ndarray, plan: SetTrainingPlan,
              rng: np.random.Generator, generator: torch.Generator) -> dict[str, float]:
    """One eval-mode pass. No optimiser, no gradient, no dropout."""
    net.eval()
    totals: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for start in range(0, len(units), plan.batch):
            chunk = units[start:start + plan.batch]
            if not len(chunk):
                continue
            samples, grippers = _batch_samples(index, chunk.tolist(), plan.sample, rng)
            terms = set_training_step(net, samples, grippers, config=plan.step,
                                      generator=generator)
            for key in _TRACKED:
                totals[key] = totals.get(key, 0.0) + float(terms[key])
            batches += 1
    if not batches:
        raise ValueError("the probe drew no batch at all")
    from src.robot.grasping.deep.train.trainer import _resolve  # noqa: PLC0415

    return _resolve(totals, batches)


def memorisation_probe(net: SetGenerator, index: CorpusIndex, train_index: np.ndarray,
                       test_index: np.ndarray, plan: SetTrainingPlan, *, units: int = 256,
                       seed: int = 0) -> ProbeResult:
    """Measure the trained model on seen and unseen assets, with an untrained control.

    The two samples are drawn the same way and are the same size. Comparing the whole training
    fold against a small held-out one reports the difference between two sample sizes as a
    difference between two populations.

    The control is a fresh net of the same architecture, not the trained one with its weights
    reset: same shapes, same initialisation distribution, no history. It answers "what gap do these
    two folds show before anyone trains on either of them".
    """
    if units < plan.batch:
        raise ValueError(f"units {units} is smaller than one batch of {plan.batch}")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    count = min(units, len(train_index), len(test_index))
    seen_units = rng.permutation(train_index)[:count]
    unseen_units = rng.permutation(test_index)[:count]

    device = next(net.parameters()).device
    control = SetGenerator(plan.model).to(device)

    def run(model: SetGenerator, chosen: np.ndarray) -> dict[str, float]:
        # A fresh generator per pass, so the seed draw and the label subsample are identical between
        # the two folds and between the two models. Without it the probe would compare four different
        # random draws and call the difference memorisation.
        return _evaluate(model, index, chosen, plan, np.random.default_rng(seed),
                         torch.Generator().manual_seed(seed))

    return ProbeResult(
        seen=run(net, seen_units), unseen=run(net, unseen_units),
        control_seen=run(control, seen_units), control_unseen=run(control, unseen_units),
        units=int(count), seconds=time.perf_counter() - started)


def oracle_ceiling(index: CorpusIndex, plan: SetTrainingPlan, *, units: int = 256,
                   seed: int = 0, device: str = "cpu") -> dict[str, float]:
    """What a perfect head would score on these metrics, with these thresholds and this K.

    The number every other number is read against. A metric whose maximum nobody has measured
    cannot say whether a model is bad or the task is: under the single-label head the position
    target lets a perfect network round-trip only part of its labels, and multimodality puts a
    ceiling well below one on top-1 at a 20 mm level.

    Those limits belong to the single-label head, which is why the ceiling is computed here rather
    than written down as a constant. This one measures the ceiling of the head that is running,
    where K slots can answer several of a seed's labels while one slot had to pick. What K caps is
    coverage, and coverage is a property of the fold it is measured on, so it is computed per run
    rather than quoted from another. A retired architecture's limit is not this architecture's
    ceiling.

    The oracle is built from the labels themselves: each seed's first `K` labels become its `K`
    slots, in order, with confidence descending so the most confident slot is a real grasp rather
    than an arbitrary one. Then the ordinary metrics run over it.

    This is a ceiling on the metric, not on the task. It answers "if the head emitted exactly the
    labelled grasps, what would the reported numbers be", which is bounded by three things a model
    cannot fix: `K` against how many modes a seed holds, the hit thresholds, and the fact that
    `top1_hit` reads one slot while a seed may admit many.
    """
    from src.robot.grasping.deep.net.assignment import optimal_one_to_one
    from src.robot.grasping.deep.net.rotation import director_matrix
    from src.robot.grasping.deep.net.set_loss import (
        GraspSetPrediction,
        grasp_set_cost,
    )
    from src.robot.grasping.deep.net.set_targets import (
        gather_set_targets,
        sample_seeds,
    )
    from src.robot.grasping.deep.train.step import _metrics, _table_of
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    chosen = rng.permutation(len(index.units))[:units]
    slots = plan.model.head.slots
    torch_device = torch.device(device)

    totals: dict[str, float] = {}
    counted = 0
    for position in chosen:
        unit = index.units[int(position)]
        from src.robot.grasping.deep.corpus.sample import build_sample, load_scene  # noqa: PLC0415

        sample = build_sample(load_scene(index.files[unit.scene]), rng, plan.sample,
                              target_instance=unit.instance)
        supervise = torch.as_tensor(np.asarray(sample["supervise"]), dtype=torch.bool)
        pair_point = torch.as_tensor(np.asarray(sample["set_pair_point"]), dtype=torch.int64)
        pair_grasp = torch.as_tensor(np.asarray(sample["set_pair_grasp"]), dtype=torch.int64)

        # The seeds are drawn the way training draws them. Taking the first supervised points
        # instead reaches almost no label, so `target.valid` is empty, every metric divides by
        # nothing and the ceiling reads 0.000 for every K.
        #
        # A random score stands in for the graspability field, which does not exist without a
        # trained model. The mixture is what matters: a ceiling measured only on labelled seeds
        # would be a ceiling on a question training never asks.
        labelled = torch.zeros_like(supervise)
        if pair_point.numel():
            labelled[pair_point] = True
        draw = sample_seeds(count=plan.step.seeds, labelled=labelled,
                            score=torch.rand(supervise.shape[0], generator=generator),
                            candidate=supervise, shares=plan.step.shares, generator=generator)
        seeds = draw.point_index
        if not seeds.numel():
            continue
        points = torch.as_tensor(np.asarray(sample["points_m"]), dtype=torch.float32)
        target, _ = gather_set_targets(
            seeds, pair_point, pair_grasp, _table_of(sample, torch_device), points,
            max_labels=plan.step.max_labels, generator=generator)

        # The first K labels become the K slots. Confidence descends so `top1_hit` reads a real
        # grasp: a perfect head that ranked its slots at random would score below its own ceiling for
        # a reason that has nothing to do with the task.
        take = target.approach[:, :slots]
        pad = slots - take.shape[1]
        def stretch(block: torch.Tensor) -> torch.Tensor:
            piece = block[:, :slots]
            if pad <= 0:
                return piece
            return torch.cat([piece, piece[:, -1:].expand(piece.shape[0], pad, *piece.shape[2:])],
                             dim=1)

        axis = stretch(target.axis)
        matrix = director_matrix(axis)
        params = torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                              matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)
        prediction = GraspSetPrediction(
            approach=stretch(take), axis_params=params, offset_m=stretch(target.offset_m),
            width_m=stretch(target.width_m.unsqueeze(-1)).squeeze(-1),
            confidence=torch.linspace(2.0, -2.0, slots).expand(take.shape[0], slots).contiguous())
        matching = optimal_one_to_one(grasp_set_cost(prediction, target, plan.step.weights),
                                      target.valid)
        row = _metrics(prediction, target, matching, plan.step)
        row["coverage_sum"] = matching.label_matched.sum().float()
        row["coverage_count"] = target.valid.sum().float()
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        counted += 1
    if not counted:
        raise ValueError("no unit yielded a supervised seed")
    from src.robot.grasping.deep.train.trainer import _resolve  # noqa: PLC0415

    return _resolve(totals, counted)


def baseline_floor(index: CorpusIndex, plan: SetTrainingPlan, *, units: int = 256,
                   seed: int = 0) -> dict[str, dict[str, float]]:
    """What a head that has learned nothing scores. The other end of the scale from the ceiling.

    A hit rate cannot be read without knowing what zero effort achieves. A per-object constant can
    beat a trained per-point head, and the surface normal can score worse than chance.

    Four baselines, cheapest first:

        random          a slot per seed drawn from nothing. The absolute floor.
        top_down        every slot straight down, at the seed, at the median width. The single most
                        common grasp in the corpus, and the one a top-down cell would try anyway.
        normal          along the point's own surface normal, inverted to face the object.
        normal_inset    the same direction, but the grasp centre pushed a fixed distance along it
                        rather than left at the seed.

    `normal_inset` is the floor for the offset head specifically. Every other arm here leaves the
    centre at the seed, which is an error the size of the corpus median offset by construction; a
    head that cannot beat "step `inset_m` along the inward normal" has learned nothing about
    position.

    `normal` can score worse than the random arm. It stays in the floor so that stays visible.
    """
    from src.robot.grasping.deep.corpus.sample import build_sample, load_scene  # noqa: PLC0415
    from src.robot.grasping.deep.net.assignment import optimal_one_to_one
    from src.robot.grasping.deep.net.rotation import director_matrix
    from src.robot.grasping.deep.net.set_loss import (
        GraspSetPrediction,
        grasp_set_cost,
    )
    from src.robot.grasping.deep.net.set_targets import (
        gather_set_targets,
        sample_seeds,
    )
    from src.robot.grasping.deep.train.trainer import _resolve  # noqa: PLC0415
    from src.robot.grasping.deep.train.step import _metrics, _table_of

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    chosen = rng.permutation(len(index.units))[:units]
    slots = plan.model.head.slots
    device = torch.device("cpu")
    names = ("random", "top_down", "normal", "normal_inset")
    totals: dict[str, dict[str, float]] = {name: {} for name in names}
    counted = 0
    #: How far `normal_inset` steps along the inward normal. The corpus median seed-to-centre
    #: offset, so the arm is "know the direction, know the typical depth".
    inset_m = 0.0363

    for position in chosen:
        unit = index.units[int(position)]
        sample = build_sample(load_scene(index.files[unit.scene]), rng, plan.sample,
                              target_instance=unit.instance)
        supervise = torch.as_tensor(np.asarray(sample["supervise"]), dtype=torch.bool)
        pair_point = torch.as_tensor(np.asarray(sample["set_pair_point"]), dtype=torch.int64)
        pair_grasp = torch.as_tensor(np.asarray(sample["set_pair_grasp"]), dtype=torch.int64)
        labelled = torch.zeros_like(supervise)
        if pair_point.numel():
            labelled[pair_point] = True
        draw = sample_seeds(count=plan.step.seeds, labelled=labelled,
                            score=torch.rand(supervise.shape[0], generator=generator),
                            candidate=supervise, shares=plan.step.shares, generator=generator)
        seeds = draw.point_index
        if not seeds.numel():
            continue
        points = torch.as_tensor(np.asarray(sample["points_m"]), dtype=torch.float32)
        target, _ = gather_set_targets(seeds, pair_point, pair_grasp, _table_of(sample, device),
                                       points, max_labels=plan.step.max_labels,
                                       generator=generator)
        count = seeds.numel()
        # `features` is normal (3) then the two flags, per `corpus.sample.FEATURE_NAMES`.
        normals = torch.as_tensor(np.asarray(sample["features"])[:, :3],
                                  dtype=torch.float32)[seeds]
        down = torch.tensor([0.0, 0.0, -1.0]).expand(count, slots, 3)
        width = torch.full((count, slots), float(plan.model.head.width_centre_m))
        axis = torch.tensor([1.0, 0.0, 0.0]).expand(count, slots, 3)
        matrix = director_matrix(axis)
        params = torch.stack([matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2],
                              matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]], dim=-1)
        confidence = torch.linspace(2.0, -2.0, slots).expand(count, slots).contiguous()

        inward = (-normals).unsqueeze(1).expand(count, slots, 3)
        arms = {
            "random": (torch.randn(count, slots, 3, generator=generator),
                       torch.zeros(count, slots, 3)),
            "top_down": (down, torch.zeros(count, slots, 3)),
            # Inverted: a normal points out of the surface, an approach points into it.
            "normal": (inward, torch.zeros(count, slots, 3)),
            # The only arm that moves the centre off the seed. See the docstring.
            "normal_inset": (inward, inward * inset_m),
        }
        for name, (approach, offset) in arms.items():
            prediction = GraspSetPrediction(
                approach=approach.contiguous(), axis_params=params,
                offset_m=offset.contiguous(), width_m=width, confidence=confidence)
            matching = optimal_one_to_one(grasp_set_cost(prediction, target, plan.step.weights),
                                          target.valid)
            for key, value in _metrics(prediction, target, matching, plan.step).items():
                totals[name][key] = totals[name].get(key, 0.0) + float(value)
        counted += 1

    if not counted:
        raise ValueError("no unit yielded a supervised seed")
    return {name: _resolve(values, counted) for name, values in totals.items()}


def approach_headroom(index: CorpusIndex, plan: SetTrainingPlan, *, units: int = 256,
                      seed: int = 0) -> dict[str, dict[str, float]]:
    """How many degrees of approach signal exist above a constant, and at what granularity.

    Every arm so far has an approach head that ties a fixed direction on unseen assets, which
    leaves two possibilities: the signal is there and the model cannot reach it, or the signal is
    not there at this granularity and no amount of corpus or capacity would find it. This probe
    separates them.

    The probe asks the corpus directly, with no network involved. For a set of labelled approaches it
    computes the single direction that minimises the mean angular error to them, which for the
    objective `sum (1 - cos)` is exactly the normalised mean vector, and reports that error at four
    granularities:

        straight_down   the fixed (0, 0, -1) the baseline floor uses
        global          one direction for the whole corpus
        per_object      one direction per object instance
        per_seed        one direction per seed

    Reading it: `per_object` far below `global` means object identity carries the signal, and a model
    that can identify the object has something to win. `per_object` near `global` means there is
    nothing to learn at this granularity, and the problem is the target rather than the model. The
    gap from `per_object` to `per_seed` is what per-point conditioning could add on top.

    Both figures are computed, with and without the z-rotation augmentation, and one may not be
    quoted for the other. The sample spec rotates each scene about z by a random angle. With it on,
    a single global constant serves every rotation of every object at once, so its error is inflated
    by the augmentation rather than by the corpus. With it off, the number is a property of the
    data. A model trained under the augmentation faces the first; a claim about what the corpus
    contains needs the second.
    """
    from src.robot.grasping.deep.corpus.sample import build_sample, load_scene  # noqa: PLC0415

    out: dict[str, dict[str, float]] = {}
    for augmented in (False, True):
        spec = dataclasses.replace(plan.sample, rotate_z=augmented)
        rng = np.random.default_rng(seed)
        chosen = rng.permutation(len(index.units))[:units]
        per_seed_error: list[float] = []
        per_object_error: list[float] = []
        pooled: list[np.ndarray] = []
        for position in chosen:
            unit = index.units[int(position)]
            sample = build_sample(load_scene(index.files[unit.scene]), rng, spec,
                                  target_instance=unit.instance)
            pair_point = np.asarray(sample["set_pair_point"])
            if not pair_point.size:
                continue
            approach = np.asarray(sample["set_grasp_approach"], dtype=np.float64)
            approach = approach / np.linalg.norm(approach, axis=-1, keepdims=True).clip(1e-9)
            labels = approach[np.asarray(sample["set_pair_grasp"])]
            pooled.append(labels)
            per_object_error.append(_mean_angle_to_best(labels))
            for point in np.unique(pair_point):
                per_seed_error.append(_mean_angle_to_best(labels[pair_point == point]))
        if not pooled:
            continue
        every = np.concatenate(pooled, axis=0)
        down = np.array([0.0, 0.0, -1.0])
        out["augmented" if augmented else "unaugmented"] = {
            "straight_down": round(float(np.degrees(
                np.arccos(np.clip(every @ down, -1.0, 1.0))).mean()), 2),
            "global": round(_mean_angle_to_best(every), 2),
            "per_object": round(float(np.mean(per_object_error)), 2),
            "per_seed": round(float(np.mean(per_seed_error)), 2),
            "labels": float(len(every)),
            "objects": float(len(per_object_error)),
            "seeds": float(len(per_seed_error)),
        }
    return out


def _mean_angle_to_best(directions: np.ndarray) -> float:
    """Mean angular error, in degrees, to the single direction that minimises it.

    The minimiser of `sum (1 - cos)` is the normalised mean vector, stated rather than searched.
    """
    if not len(directions):
        return float("nan")
    total = directions.sum(axis=0)
    norm = float(np.linalg.norm(total))
    if norm < 1e-9:
        # Perfectly opposed labels have no best direction; every one is 90 degrees off on average.
        return 90.0
    best = total / norm
    return float(np.degrees(np.arccos(np.clip(directions @ best, -1.0, 1.0))).mean())
