"""Training the set generator over a corpus: folds, epochs, and what each epoch reports.

The layer above `set_training_step`. It owns three things the step deliberately does not: which
(scene, object) pairs exist, how they are split so a held-out number means something, and how the
scenes reach the GPU.

Scenes are streamed, not held. A cloud is about a megabyte in memory, so a corpus of tens of
thousands of them does not fit in one; a small corpus can be read eagerly and a large one cannot.
Streaming costs a measured 4.93 ms per scene, so a batch of four is 20 ms against a step that takes
hundreds, and the memory constraint disappears.

But the unit index needs every scene. `training_units` decides which objects can be conditioned on
and which asset each belongs to, and it must see the whole corpus or the folds are cut over a
subset. So there are two passes with very different costs:

    metadata only, three small arrays per file    0.80 ms each
    the full cloud                                4.93 ms each

The index pass reads only `object_instance`, `object_asset_id` and `grasp_instance`. `np.load` is
lazy, so the point cloud beside them is never decompressed.

The gripper comes from the cloud, not from the caller. Every extracted scene records which jaw its
labels were written for, and a corpus may mix them: the same geometry under a different hand has a
different correct answer, which is exactly the contrast the conditioning has to learn from. A batch
therefore carries one gripper row per sample, gathered per scene.

An unknown stamp refuses. Defaulting to the 2F-85 would train the head on a hand the labels did not
come from, and a wrong conditioning input is worse than a constant one.

Folds are `grouped_folds` over asset groups, the same function the earlier arms used. Deliberately
not a new split: an architecture comparison against those arms is only a comparison if the held-out
assets are the same ones.

The folds are load-bearing. A model of this capacity can memorise a corpus rather than generalise
from it, so the seen-versus-unseen gap is an acceptance criterion and not a footnote.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample, load_scene
from src.robot.grasping.deep.corpus.index import (
    TrainingUnit,
    grouped_folds,
    rebalance_units,
    training_units,
)
from src.robot.grasping.deep.train.progress import epoch_bar
from src.robot.grasping.deep.train.progress import postfix as _postfix
from src.robot.grasping.deep.net.gripper import gripper_vector
from src.robot.grasping.deep.net.set_generator import SetGenerator, SetGeneratorConfig
from src.robot.grasping.deep.train.step import SetStepConfig, set_training_step
from src.utility.log_cfg import create_logger

__all__ = ["CorpusIndex", "SetTrainingPlan", "corpus_index", "train_set_generator"]

logger = create_logger("SetLoop", "deep_set_loop.log", log_dir="logs")

#: The three arrays the index pass reads. Everything else in a cloud stays compressed on disk.
_INDEX_KEYS = ("object_instance", "object_asset_id", "grasp_instance")

#: What a cloud written without a gripper stamp is treated as. Safe only because every corpus that
#: predates the stamp was labelled with the 2F-85; that is a fact about those corpora, not a
#: property of the format.
_DEFAULT_GRIPPER = "2f85"


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    """Everything the loop needs about a corpus without holding any of it."""

    files: list[Path]
    units: list[TrainingUnit]
    groups: list[str]
    #: Per file, the jaw its labels were written for.
    grippers: list[str]
    seconds: float

    @property
    def asset_groups(self) -> int:
        return len(set(self.groups))


@dataclass(frozen=True, slots=True)
class SetTrainingPlan:
    """Every knob of a run, in one place a model card can quote."""

    epochs: int = 12
    folds: int = 5
    #: How many of the folds are actually run. One is enough to compare two architectures on the same
    #: held-out assets; five is for a number worth quoting on its own.
    run_folds: int = 1
    batch: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    #: Units per epoch, or `None` for all of them. A cap keeps an epoch comparable across corpora of
    #: different sizes, which is what made the earlier arms readable.
    train_units: int | None = 4000
    eval_units: int = 512

    #: After the folds earn the numbers, refit on everything and ship that.
    #:
    #: The fold pass exists to produce an honest held-out estimate, so it must hold assets back. The
    #: shipped model has no such reason: an artifact that has never seen part of the corpus it was
    #: trained for is worse at exactly those parts, for nothing.
    #:
    #: With `--folds 5 --run-folds 1` the artifact never sees the units of the folds that were not
    #: run. For a customer that is a share of their own catalogue, chosen by a hash rather than by
    #: anything they decided.
    #:
    #: Unmeasured on this architecture. `ranker/training.py` states the same discipline for the
    #: grasp ranker: the out-of-fold pass earns the numbers and the final model is refit on all
    #: rows because that is the one that ships. No number for it exists here. The case for it is a
    #: product case, not a metric one: shipping a model blind to part of the customer's catalogue
    #: is a defect whatever the aggregate says. The case against is that it costs a second training
    #: run.
    #:
    #: Off by default so every arm already measured reproduces byte-identically, and because a
    #: silent doubling of training time is not something to spring on anyone. The customer recipe
    #: turns it on.
    #:
    #: The refit reports no held-out number, and cannot: it trained on everything, so any
    #: held-out set it could be scored against is inside its training data. Its metrics live under
    #: `report["refit"]` and are training metrics, labelled as such. The numbers to quote for the run
    #: remain the fold pass's.
    refit: bool = False

    #: Which named recipe produced this run, or None for a hand-assembled one.
    #:
    #: Record only. The recipe's settings are applied by `train.plan.build_plan` before the plan is
    #: built, so this field changes nothing about the run. It exists so the artifact can say which
    #: bundle a customer asked for, which is the difference between two `.pt` files a person can
    #: tell apart and two they cannot.
    recipe: str | None = None

    #: Which compute tier the recipe was narrowed to, or None.
    #:
    #: This is the field that can mislead. `--tier smoke` is two epochs on four hundred units with
    #: the refit off, and `recipes.py` says in its own text that it "says nothing about grasp
    #: quality, and is not a model to deploy". Recording it here is what makes a smoke artifact
    #: distinguishable at load time from one that took hours: a tier that reaches only the command
    #: line reaches neither the plan, nor the run report, nor the model card.
    tier: str | None = None

    #: Stop the fold if the head has collapsed to a constant. Degrees of held-out
    #: `seed_spread_deg`, below which the run is abandoned. `0.0` is off, which is the default,
    #: because a stop condition is a behaviour change and every seam here ships default-off.
    #:
    #: Why it exists: a head can learn one direction and propose it for every seed, with widths
    #: that barely differ. That stays invisible until the poses themselves are read, three stages
    #: and several hours downstream, because every quality metric reports it honestly as a low
    #: number without being able to say why.
    #:
    #: Not checked before epoch two. An untrained head starts near a constant and has to be given
    #: the chance to leave it; killing at epoch zero would refuse every run.
    collapse_floor_deg: float = 0.0

    #: What share of an epoch's units carry a grasp label at all.
    #:
    #: Most training units carry no labelled point at all, because a unit is an object and most
    #: objects earn no label. Such a unit trains the graspability head with true negatives and gives
    #: the pose heads nothing at all.
    #:
    #: The seed mixture then compounds it: 50 % of 64 seeds asks for 32 labelled points, and few
    #: units can supply that many, so most drawn seeds carry no label.
    #:
    #: Not 1.0. `training_units` keeps unlabelled objects deliberately: they are the true negatives
    #: the where stage is trained on, and an epoch without them produces a field that says yes
    #: everywhere. Half is the balance, and it is an assumption, not a measurement.
    #:
    #: `None` leaves the natural mix.
    labelled_unit_share: float | None = 0.5
    sample: SampleSpec = field(default_factory=lambda: SampleSpec(grasp_set=True))

    #: Replace the corpus labels with a target the stack must be able to learn. `None` is every
    #: real run; `"normal"` or `"local"` turns the run into a control.
    #:
    #: It exists because two explanations fit a flat curve equally well: the architecture cannot
    #: learn a grasp head, or the corpus cannot teach one, and nothing else separates them. See
    #: `deep/train/synthetic_control.py`. Any number from a control run belongs beside the word
    #: "control".
    control: str | None = None
    step: SetStepConfig = field(default_factory=SetStepConfig)
    model: SetGeneratorConfig = field(default_factory=SetGeneratorConfig)

    def __post_init__(self) -> None:
        if not self.sample.grasp_set:
            raise ValueError("the set loop needs SampleSpec(grasp_set=True); without it the sample "
                             "carries one label per point and the K-slot head has nothing to match")
        if self.run_folds > self.folds:
            raise ValueError(f"run_folds {self.run_folds} exceeds folds {self.folds}")


#: The angle a run is not training on, printed beside the one it is. Both are always computed, so a
#: reader can see whether the untrained one drifted, and the two arms stay comparable.
_OTHER_ANGLE: Final[dict[str, str]] = {"approach": "axis", "axis": "approach"}


def corpus_index(files: Sequence[str | Path]) -> CorpusIndex:
    """Walk the corpus reading only what the split needs. See the module docstring for the costs."""
    started = time.perf_counter()
    paths = [Path(f) for f in files]
    if not paths:
        raise ValueError("no clouds to index")
    stubs: list[dict[str, np.ndarray]] = []
    grippers: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            stubs.append({key: handle[key] for key in _INDEX_KEYS if key in handle.files})
            stamp = (str(np.asarray(handle["gripper"]).reshape(-1)[0])
                     if "gripper" in handle.files else _DEFAULT_GRIPPER)
        grippers.append(stamp)
    units = training_units(stubs)
    if not units:
        raise ValueError("no groupable objects: this corpus carries no `object_asset_id`, so no "
                         "asset-disjoint split is possible and a held-out number would be fiction")
    index = CorpusIndex(files=paths, units=units, groups=[u.group for u in units],
                        grippers=grippers, seconds=time.perf_counter() - started)
    logger.info("indexed %d cloud(s): %d unit(s), %d asset group(s), %d gripper(s) in %.1f s",
                len(paths), len(units), index.asset_groups, len(set(grippers)), index.seconds)
    return index


def _batch_samples(index: CorpusIndex, units: Sequence[int], spec: SampleSpec,
                   rng: np.random.Generator,
                   control: str | None = None) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """Read the scenes these units live in and build one sample each, plus their gripper rows.

    A unit is a (scene, object) pair, so two units of one batch may share a scene. It is read twice
    rather than cached: a cache keyed by scene would make a batch's cost depend on which units landed
    in it, and a step whose duration depends on the draw is a step nobody can time.
    """
    samples: list[dict[str, Any]] = []
    rows: list[torch.Tensor] = []
    for position in units:
        unit = index.units[position]
        scene = load_scene(index.files[unit.scene])
        sample = build_sample(scene, rng, spec, target_instance=unit.instance)
        if control is not None:
            from src.robot.grasping.deep.train.synthetic_control import (  # noqa: PLC0415
                synthetic_labels,
            )

            sample = synthetic_labels(sample, control)
        samples.append(sample)
        rows.append(gripper_vector(index.grippers[unit.scene]))
    return samples, torch.stack(rows)


def _run_epoch(net: SetGenerator, index: CorpusIndex, order: np.ndarray, plan: SetTrainingPlan,
               optimiser: torch.optim.Optimizer | None, rng: np.random.Generator,
               generator: torch.Generator, bar: Any = None) -> dict[str, float]:
    """One pass over `order`. Trains when `optimiser` is given, evaluates when it is not.

    `bar` is optional and defaults to nothing. The evaluation pass calls this too and a second bar
    per epoch would flicker against the first; only the training pass passes one. Anything with
    `update` and `set_postfix_str` works, which is what `progress.epoch_bar` yields whether or not a
    terminal is watching.
    """
    training = optimiser is not None
    net.train(training)
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, len(order), plan.batch):
        chunk = order[start:start + plan.batch]
        if not len(chunk):
            continue
        samples, grippers = _batch_samples(index, chunk.tolist(), plan.sample, rng, plan.control)
        with torch.set_grad_enabled(training):
            terms = set_training_step(net, samples, grippers, config=plan.step,
                                      generator=generator)
        if training:
            assert optimiser is not None
            optimiser.zero_grad(set_to_none=True)
            terms["total"].backward()
            # A clip rather than a hope. The confidence term is a BCE over slots and the pose terms
            # are metric; early on they are not the same size, and one spike through a 21M-parameter
            # backbone costs an epoch.
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimiser.step()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        batches += 1
        if bar is not None:
            bar.update(1)
            # Every 20 batches, not every one. The postfix is a string format plus a terminal
            # write; at batch sizes of four that is thousands of writes an epoch, and the display
            # would start costing measurable time in the thing it is describing.
            if batches % 20 == 0:
                bar.set_postfix_str(_postfix(totals, batches))
    if not batches:
        raise ValueError("an epoch drew no batch at all")
    return _resolve(totals, batches)


def _resolve(totals: dict[str, float], batches: int) -> dict[str, float]:
    """Turn the accumulated sums and counts into the numbers a row reports.

    A metric with a denominator is divided once, here, over the whole epoch. Dividing per batch
    and averaging the ratios weights every batch equally, so a batch whose seeds carried no labels
    contributes a nought to a metric that is undefined there rather than zero, which deflates a
    perfect head's `top1_hit` ceiling far below 1.0.

    Losses keep the plain per-batch mean: they are defined for every batch, an empty one included.
    """
    out: dict[str, float] = {}
    for key, value in totals.items():
        if key.endswith("_count"):
            continue
        if key.endswith("_sum"):
            base = key[: -len("_sum")]
            denominator = totals.get(f"{base}_count", 0.0)
            out[base] = value / denominator if denominator else float("nan")
        else:
            out[key] = value / batches
    return out


def _artifact_report(plan: SetTrainingPlan, index: "CorpusIndex | None",
                     report: "dict[str, Any] | None", *, refit: bool) -> dict[str, Any]:
    """What the card says about how these weights came to exist.

    Without this card a customer holding two `.pt` files cannot tell which corpus, which target or
    which loss trained either, and cannot tell a diagnostic from a shippable model: `plan.control`
    trains on synthetic labels derived from an input channel, and a control artifact would be
    indistinguishable from a real one at load time. `write_set_generator` takes it as its `report`
    argument.

    Every field here is a fact about the run, never a quality claim. `held` carries the fold
    pass's last epoch because that is the honest number; a refit has none and says so.
    """
    card: dict[str, Any] = {
        "recipe": plan.recipe,
        # Beside the recipe, because a tier narrows a recipe and `smoke` is not a deployable run.
        "tier": plan.tier,
        "weights_are": "refit_on_every_unit" if refit else "one_fold",
        "epochs": plan.epochs, "folds": plan.folds, "run_folds": plan.run_folds,
        "target": plan.step.target,
        "slot_mixing": plan.model.head.slot_mixing, "axis_mode": plan.model.head.axis_mode,
        # The one that matters most. A control run fits synthetic labels derived from an input
        # channel: useful as a diagnostic, catastrophic if served.
        "control": plan.control,
    }
    if index is not None:
        card["corpus"] = {"units": len(index.units), "groups": index.asset_groups,
                          "grippers": sorted(set(index.grippers))}
    if refit:
        card["held"] = None
        card["why_no_held"] = ("these weights saw every unit, so no held-out number exists for "
                               "them. The fold pass's numbers are the ones to quote.")
    elif report is not None and report.get("epochs"):
        last = report["epochs"][-1]
        card["held"] = {key: value for key, value in last.items() if key.startswith("held_")}
    return card


def _write_artifact(directory: Path, net: Any, plan: SetTrainingPlan, grippers: Sequence[str],
                    chosen: str | None, *, card: "dict[str, Any] | None" = None) -> dict[str, Any]:
    """Write a loadable artifact beside the fold checkpoint, or say why it could not be written.

    The gripper is not guessed. The artifact stamps which hand the model plans for, and the loader
    resolves it to the conditioning vector. On a single-gripper corpus the answer is obvious and is
    taken; on a varied one it is a decision, and defaulting it would put the 2F-85's geometry on a
    model trained for a 140 mm hand without a word.

    Returns a block for the report either way, because "no artifact was written" is a fact a card
    should carry rather than an absence a reader has to notice.
    """
    from src.robot.grasping.deep.set_artifact import write_set_generator  # noqa: PLC0415

    present = sorted(set(grippers))
    gripper = chosen or (present[0] if len(present) == 1 else None)
    if gripper is None:
        reason = (f"the corpus carries {len(present)} grippers ({', '.join(present)}) and none was "
                  f"named; pass artifact_gripper to say which hand the artifact plans for")
        logger.warning("no artifact written: %s", reason)
        return {"written": False, "reason": reason}
    try:
        # Both: the hand it plans for and the hands it saw. Writing only the first presents a model
        # fitted across three hands as belonging to one and puts its generalisation out of reach at
        # runtime. A model trained across several jaws proposes a width that follows the aperture it
        # is conditioned on, and only the second field says which jaws it saw.
        paths = write_set_generator(directory, net, sample=plan.sample, step=plan.step,
                                    gripper=gripper, trained_grippers=present, report=card)
    except (ValueError, OSError) as exc:
        logger.warning("no artifact written: %s", exc)
        return {"written": False, "reason": f"{type(exc).__name__}: {exc}"}
    logger.info("artifact written for %s (trained across %s): %s",
                gripper, ", ".join(present), paths["weights"].name)
    return {"written": True, "gripper": gripper, "trained_grippers": present,
            **{key: str(value) for key, value in paths.items()}}


def _plan_fingerprint(plan: SetTrainingPlan) -> dict[str, Any]:
    """The plan as comparable data, so a resume into a changed one can be refused rather than run.

    Everything that changes what the run does, and nothing that does not. `epochs` is included
    because `CosineAnnealingLR` takes its `T_max` from it: continuing a twelve-epoch schedule as a
    twenty-epoch one is a different learning-rate curve wearing the old run's name.

    Deliberately absent, so nobody adds them meaning well: `refit` and `collapse_floor_deg`.
    Neither changes the fold trajectory. `refit` is a separate pass that runs after every fold is
    finished, and `collapse_floor_deg` only decides whether to stop early. Putting either here would
    refuse a legitimate resume, which is the opposite of what this function is for.
    """
    return {"epochs": plan.epochs, "folds": plan.folds, "run_folds": plan.run_folds,
            "batch": plan.batch, "learning_rate": plan.learning_rate,
            "weight_decay": plan.weight_decay, "seed": plan.seed,
            "train_units": plan.train_units, "eval_units": plan.eval_units,
            "labelled_unit_share": plan.labelled_unit_share, "control": plan.control,
            "sample": dataclasses.asdict(plan.sample), "step": dataclasses.asdict(plan.step),
            "model": dataclasses.asdict(plan.model)}


def _resume_fold(path: Path, net: Any, optimiser: Any, schedule: Any, plan: SetTrainingPlan,
                 rng: Any, generator: Any, device: str) -> int:
    """Restore one fold and return the epoch to start from. Refuses anything that would lie.

    The refusal is the feature, and it is the same reasoning the older `train --resume` uses: a
    resume into a changed world is a different experiment wearing the old run's card. The difference
    from fine-tuning is exactly this check. `--init-from` deliberately allows a changed corpus, which
    is what a customer training on their own objects needs; a resume must not, because its whole
    claim is that the curve it continues is the curve it started.
    """
    if not path.is_file():
        raise FileNotFoundError(f"nothing to resume: no checkpoint at {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    stored = payload.get("plan")
    if not isinstance(stored, dict):
        raise ValueError(f"{path} predates resumable checkpoints: its plan is a display string, so "
                         f"a changed plan could not be detected. Start the run again, or use "
                         f"--init-from to warm-start from its weights")
    wanted = _plan_fingerprint(plan)
    changed = sorted(key for key in wanted if stored.get(key) != wanted[key])
    if changed:
        raise ValueError(f"{path} was written under a different plan: {', '.join(changed)} differ. "
                         f"A resume continues one experiment; use --init-from to start a new one "
                         f"from these weights")
    for key in ("optimiser", "schedule", "numpy_state", "torch_state"):
        if key not in payload:
            raise ValueError(f"{path} carries no {key}; it cannot be continued, only warm-started")
    net.load_state_dict(payload["model"])
    optimiser.load_state_dict(payload["optimiser"])
    schedule.load_state_dict(payload["schedule"])
    rng.bit_generator.state = payload["numpy_state"]
    generator.set_state(payload["torch_state"])
    resumed = int(payload["epoch"]) + 1
    logger.info("resuming fold %d at epoch %d from %s", int(payload["fold"]), resumed + 1, path)
    return resumed


def _load_weights(net: Any, source: str | Path, device: str) -> dict[str, Any]:
    """Copy weights into `net`, refusing anything that does not fit rather than fitting partly.

    `load_state_dict(..., strict=False)` is the trap this function exists to avoid. It is the
    obvious way to load a checkpoint from a slightly different model, and it silently leaves every
    mismatched tensor at its random initialisation. The result trains, reports plausible numbers, and
    is a model half inherited and half fresh that nobody chose. A shape that does not match is a
    question for a person.

    Returns what was inherited, for the report.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"no weights at {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"{path} carries neither `model` nor `state_dict`; it is not a checkpoint "
                         f"this loop wrote")
    target = net.state_dict()
    missing = sorted(set(target) - set(state))
    unexpected = sorted(set(state) - set(target))
    mismatched = sorted(name for name in set(target) & set(state)
                        if tuple(target[name].shape) != tuple(state[name].shape))
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{path} does not fit this architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(mismatched)} of a "
            f"different shape. First of each: {missing[:1]} {unexpected[:1]} {mismatched[:1]}. "
            f"Fine-tuning needs the same model, so build the plan the checkpoint was trained with")
    net.load_state_dict(state)
    return {"path": str(path), "tensors": len(state),
            "trained_epochs": int(payload.get("epoch", -1)) + 1,
            "plan": str(payload.get("plan", ""))[:400]}


def train_set_generator(files: Sequence[str | Path], plan: SetTrainingPlan | None = None, *,
                        out_dir: str | Path | None = None,
                        device: str | None = None,
                        init_from: str | Path | None = None,
                        freeze_backbone: bool = False,
                        resume: bool = False,
                        artifact_gripper: str | None = None,
                        probe_units: int = 256,
                        on_epoch: "Callable[[Mapping[str, Any]], None] | None" = None,
                        ) -> dict[str, Any]:
    """Train the K-slot generator and return one row per epoch per fold.

    Every row carries the held-out numbers beside the training ones, because the gap between them
    is the memorisation probe and a run that reported only one of the two could not answer the
    question this architecture was scaled up to ask.

    The probe runs at the end of every fold, in the run rather than afterwards. An acceptance
    criterion somebody has to remember to invoke is one that gets skipped on the run where it
    matters. `probe_units = 0` turns it off, which is for a smoke run and nothing else.

    `init_from` is fine-tuning, and it is a different thing from resuming. The older
    `train --resume` continues one interrupted experiment and refuses on any difference in the plan,
    the architecture, the corpus by content, the device or the torch version, because a resume into a
    changed world is a different experiment wearing the old run's card. That strictness is right for
    what it does and it makes `--resume` useless for the thing a customer actually needs: take a
    model trained on somebody's corpus and continue it on their own objects, which is a changed
    corpus by definition.

    So this starts a new run, with its own fresh optimiser, schedule and split, whose weights happen
    to begin somewhere other than random. Everything that makes a run a run is rebuilt; only the
    initialisation moves. The report records where the weights came from, because a card that did not
    say so would claim a model learned from scratch what it in fact inherited.

    And it writes a loadable artifact through `set_artifact`: a self-describing format with four
    named refusals, so a binned model cannot be loaded into a set decoder and a state dict cannot be
    loaded into a config it did not travel with. A bare
    `torch.save({"model": ..., "plan": repr(plan)})` is refused by kind by `load_set_generator`, so
    nothing written that way can be deployed and the plan's primary metric, the referee success
    rate, has no model to propose grasps with.

    `artifact_gripper` is required when the corpus carries more than one, and refused rather than
    guessed. A default would be the 2F-85, which is right today and silently wrong the first time
    anybody trains on a varied corpus, which is exactly what importing a foreign one produces.

    `freeze_backbone` holds the encoder still and trains only the heads. It is the cheap form of
    fine-tuning and the right default to reach for when the new corpus is small, since 99.0 % of this
    model's parameters are in the backbone and a few hundred scenes cannot re-estimate them. It is
    not the default here, because which one is better is unmeasured on this project's data.
    """
    plan = plan or SetTrainingPlan()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = corpus_index(files)
    splits = grouped_folds(index.groups, folds=plan.folds, seed=plan.seed)

    report: dict[str, Any] = {
        "clouds": len(index.files), "units": len(index.units),
        # Stamped whether or not it was used. A card that mentions the source only when there is
        # one leaves a reader to infer "from scratch" from an absence, and an absence is also what a
        # writer that forgot to stamp it produces.
        "initialised_from": str(init_from) if init_from is not None else None,
        "frozen_backbone": bool(freeze_backbone),
        "asset_groups": index.asset_groups, "grippers": sorted(set(index.grippers)),
        "index_seconds": round(index.seconds, 1), "device": device,
        "epochs": [],
    }
    if probe_units:
        # The ceiling, measured once and carried in the report so no number below it can be read
        # without it. A perfect head reaches a coverage well below 1.0, because a seed holds more
        # labels than the head has slots and one-to-one matching can bind at most K of them.
        # Reading a coverage against 1.0 rather than against that ceiling is the misreading this
        # line exists to prevent.
        from src.robot.grasping.deep.eval.probes import (  # noqa: PLC0415
            baseline_floor,
            oracle_ceiling,
        )

        report["ceiling"] = {k: round(v, 5) for k, v in
                             oracle_ceiling(index, plan, units=min(probe_units, 120),
                                            seed=plan.seed).items()}
        # And the floor, because a model number is unreadable without both ends. Three trivial
        # heads are scored: a random one, the surface normal, and "always straight down". That last
        # one is the bar: a trained head that does not clearly beat it has not beaten "point the
        # gripper down".
        #
        # `plan_epochs` records how long the run was asked to be. Without it a reader of
        # `epochs.json` cannot tell four epochs of four from four of thirty-six, and `deep report`
        # prints "4 epoch(s) of 4" for a run that has barely started.
        report["plan_epochs"] = int(plan.epochs)
        report["floor"] = {name: {k: round(v, 5) for k, v in values.items()}
                           for name, values in
                           baseline_floor(index, plan, units=min(probe_units, 120),
                                          seed=plan.seed).items()}
        logger.info("floor at K=%d: random %.4f, normal %.4f, top_down %.4f (top1_hit)",
                    plan.model.head.slots, report["floor"]["random"]["top1_hit"],
                    report["floor"]["normal"]["top1_hit"],
                    report["floor"]["top_down"]["top1_hit"])
        logger.info("ceiling at K=%d over the corpus: top1 %.3f, coverage %.3f, slots filled %.2f",
                    plan.model.head.slots, report["ceiling"]["top1_hit"],
                    report["ceiling"]["coverage"], report["ceiling"]["slots_filled"])
    for fold in range(plan.run_folds):
        train_index, test_index = splits[fold]
        net = SetGenerator(plan.model).to(device)
        if init_from is not None:
            report.setdefault("inherited", _load_weights(net, init_from, device))
        if freeze_backbone:
            for parameter in net.backbone.parameters():
                parameter.requires_grad_(False)
        report.setdefault("parameters", net.parameter_count)
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        report.setdefault("trainable_parameters", trainable)
        # Only the parameters that still want a gradient. This does not prevent a defect: the AdamW
        # step skips any parameter whose gradient is None, so a frozen tensor handed to it does not
        # drift under decoupled weight decay. The filter stays because it makes
        # `trainable_parameters` mean what it says and keeps the optimiser state proportional to
        # what is being trained.
        optimiser = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                      lr=plan.learning_rate, weight_decay=plan.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=plan.epochs)
        rng = np.random.default_rng(plan.seed + fold)
        generator = torch.Generator().manual_seed(plan.seed + fold)
        first_epoch = 0
        if resume:
            # Refused by name rather than crashing on a None path. A resume without an output
            # directory has nothing to resume from, and the failure it would otherwise produce names
            # a path type rather than the mistake.
            if out_dir is None:
                raise ValueError("resume needs --out: there is no checkpoint without one")
            first_epoch = _resume_fold(Path(out_dir) / f"fold{fold}.pt", net, optimiser, schedule,
                                       plan, rng, generator, device)
            report.setdefault("resumed_from_epoch", {})[str(fold)] = first_epoch
        # A random sample, never a prefix. `grouped_folds` returns its indices in group order, so
        # `test_index[:512]` is a handful of asset groups rather than a cross-section of the fold.
        # Its ceiling and its labels per seed are then those of a different population, so every
        # held-out number would be about those few groups and not about unseen assets at all.
        #
        # A prefix is not a sample. The same trap sits on scene ids, which sort by family.
        #
        # Drawn once per fold from its own generator, so the held-out set is identical across every
        # epoch of a run and across both arms of a comparison.
        held = np.random.default_rng(plan.seed + 1000 + fold).permutation(test_index)[
            :plan.eval_units]
        logger.info("fold %d/%d: %d train unit(s), %d held-out, %d parameter(s)",
                    fold + 1, plan.run_folds, len(train_index), len(held), net.parameter_count)
        if probe_units:
            # The ceiling is a property of a population, not of the corpus, so the held-out slice
            # gets its own. Comparing a held-out coverage against the training population's ceiling
            # is how a number came to read as above its own maximum.
            from src.robot.grasping.deep.eval.probes import (  # noqa: PLC0415
                baseline_floor,
                oracle_ceiling,
            )

            held_view = replace(index, units=[index.units[i] for i in held],
                                groups=[index.groups[i] for i in held])
            held_ceiling = oracle_ceiling(held_view, plan, units=min(probe_units, 120),
                                          seed=plan.seed)
            report.setdefault("held_ceiling", []).append(
                {"fold": fold, **{k: round(v, 5) for k, v in held_ceiling.items()}})
            logger.info("fold %d held-out ceiling: coverage %.3f, labels per seed %.2f",
                        fold, held_ceiling["coverage"], held_ceiling["slots_filled"])
            # The floor gets the same treatment as the ceiling. A floor is a property of a
            # population exactly as a ceiling is, and `report["floor"]` above is drawn from the
            # whole index while every `held_*` column is the held-out slice, so a headline
            # comparison against it ("did the head beat pointing the gripper down") is made against
            # a bar measured on different units.
            #
            # The corpus-wide floor stays in `report["floor"]`. It is not wrong, it answers a
            # different question, and deleting it would break every comparison against a run already
            # on disk. The per-fold one is what a `held_*` number is read against.
            held_floor = baseline_floor(held_view, plan, units=min(probe_units, 120),
                                        seed=plan.seed)
            report.setdefault("held_floor", []).append(
                {"fold": fold,
                 **{name: {k: round(v, 5) for k, v in values.items()}
                    for name, values in held_floor.items()}})
            logger.info("fold %d held-out floor: random %.4f, normal %.4f, top_down %.4f "
                        "(top1_hit; the bar a held-out number is read against)",
                        fold, held_floor["random"]["top1_hit"], held_floor["normal"]["top1_hit"],
                        held_floor["top_down"]["top1_hit"])

        for epoch in range(first_epoch, plan.epochs):
            order = rng.permutation(train_index)
            if plan.train_units is not None:
                order = order[:plan.train_units]
            if plan.labelled_unit_share is not None:
                # The epoch keeps its length and changes its mix, so a run with this on is comparable
                # to one without on everything except the mix.
                order = rebalance_units(order, index.units, plan.labelled_unit_share, rng)
            started = time.perf_counter()
            # The only place a person sees anything for ten minutes. Batches, not units: a bar
            # counting units would jump by the batch size and read as stuck between jumps.
            with epoch_bar(total=max(1, (len(order) + plan.batch - 1) // plan.batch),
                           epoch=epoch + 1, epochs=plan.epochs, fold=fold) as bar:
                train = _run_epoch(net, index, order, plan, optimiser, rng, generator, bar=bar)
            evaluated = _run_epoch(net, index, held, plan, None, rng, generator)
            schedule.step()
            row = {"fold": fold, "epoch": epoch, "seconds": round(time.perf_counter() - started, 1),
                   **{f"train_{k}": round(v, 5) for k, v in train.items()},
                   **{f"held_{k}": round(v, 5) for k, v in evaluated.items()}}
            report["epochs"].append(row)
            spread = float(evaluated.get("seed_spread_deg", float("nan")))
            if (plan.collapse_floor_deg > 0.0 and epoch >= 2
                    and spread == spread and spread < plan.collapse_floor_deg):
                # Recorded in the report, not only raised: a run that stopped for this reason is a
                # result about the configuration, and a reader of `epochs.json` alone must see it.
                report.setdefault("collapsed", []).append(
                    {"fold": fold, "epoch": epoch, "seed_spread_deg": round(spread, 5),
                     "floor_deg": plan.collapse_floor_deg})
                logger.error(
                    "fold %d STOPPED at epoch %d: the head proposes one direction for every seed "
                    "(seed spread %.4f deg, below the %.2f deg floor). Slot spread %.4f deg, width "
                    "spread %.3f mm. This is a collapsed head, not a slow one; the run is abandoned "
                    "rather than trained for hours into a constant.",
                    fold, epoch + 1, spread, plan.collapse_floor_deg,
                    float(evaluated.get("slot_spread_deg", float("nan"))),
                    float(evaluated.get("width_spread_mm", float("nan"))))
                break
            logger.info(
                # The target's own angle comes first, and the spread belongs here too. A number
                # that reaches only `epochs.json` is invisible to anyone watching a six-hour run,
                # and printing the approach error under a run that trains the axis hides the one
                # number the run exists to move.
                "fold %d epoch %2d/%d  total %.4f -> held %.4f  cover %.3f/%.3f  top1 %.3f/%.3f  "
                "%s %.1f/%.1f deg  (other %.1f/%.1f)  offset %.1f/%.1f mm  "
                "hit30 %.3f  hit40 %.3f  spread seed %.1f slot %.1f deg  %.0f s",
                fold, epoch + 1, plan.epochs, train["total"], evaluated["total"],
                train["coverage"], evaluated["coverage"], train["top1_hit"], evaluated["top1_hit"],
                plan.step.target,
                train[f"{plan.step.target}_error_deg"], evaluated[f"{plan.step.target}_error_deg"],
                train[f"{_OTHER_ANGLE[plan.step.target]}_error_deg"],
                evaluated[f"{_OTHER_ANGLE[plan.step.target]}_error_deg"],
                train["offset_error_mm"], evaluated["offset_error_mm"],
                evaluated.get("top1_hit_30mm", float("nan")),
                evaluated.get("top1_hit_40mm", float("nan")),
                evaluated.get("seed_spread_deg", float("nan")),
                evaluated.get("slot_spread_deg", float("nan")), row["seconds"])
            # The listener fires before the files are written, so a watcher sees the epoch at the
            # moment it finished rather than after two disk writes and a plot. Off by default and
            # therefore byte-identical: `GeneratorTraining.attach_progress_listener` is the only
            # thing that sets it, and an exception from it is deliberately not caught, because a
            # broken callback should surface instead of quietly costing a six-hour run its watcher.
            if on_epoch is not None:
                on_epoch(row)
            if out_dir is not None:
                directory = Path(out_dir)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "epochs.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                _refresh_curve(directory)
                # Everything a continuation needs. Weights alone cannot resume a run: AdamW's
                # moments are half its state, the cosine schedule reads the epoch it is on, and the
                # seed streams decide which units and seeds the next epoch draws. Restarting from
                # weights alone is a different experiment with a warm start, which is a thing worth
                # having and is not this.
                #
                # The plan is stored structurally rather than as `repr(plan)`. A repr is a display
                # string: it cannot be compared field by field, so a resume into a changed plan
                # could not be refused, only eyeballed.
                torch.save({"model": net.state_dict(),
                            "optimiser": optimiser.state_dict(),
                            "schedule": schedule.state_dict(),
                            "plan": _plan_fingerprint(plan),
                            "plan_text": repr(plan),
                            "numpy_state": rng.bit_generator.state,
                            "torch_state": generator.get_state(),
                            "fold": fold, "epoch": epoch},
                           directory / f"fold{fold}.pt")

        if out_dir is not None and not plan.refit:
            # The fold's net is the artifact only when there is no refit. With `refit` on, the
            # shipped weights come from the pass below, which has seen every unit.
            report.setdefault("artifact", _write_artifact(
                Path(out_dir), net, plan, index.grippers, artifact_gripper,
                card=_artifact_report(plan, index, report, refit=False)))
        if probe_units:
            # Imported here rather than at module scope: the probe imports this module for the batch
            # builder, and a top-level import both ways is a cycle.
            from src.robot.grasping.deep.eval.probes import (  # noqa: PLC0415
                memorisation_probe,
            )

            result = memorisation_probe(net, index, train_index, test_index, plan,
                                        units=probe_units, seed=plan.seed)
            report.setdefault("probe", []).append({"fold": fold, **result.as_row()})
            logger.info(
                "fold %d PROBE over %d unit(s): coverage seen %.3f unseen %.3f "
                "(gap %+.3f, control %+.3f, EXCESS %+.3f); approach seen %.1f unseen %.1f deg "
                "(EXCESS %+.1f)",
                fold, result.units, result.seen["coverage"], result.unseen["coverage"],
                result.gap["coverage"], result.control_gap["coverage"],
                result.excess_gap["coverage"], result.seen["approach_error_deg"],
                result.unseen["approach_error_deg"], result.excess_gap["approach_error_deg"])
            if out_dir is not None:
                (Path(out_dir) / "epochs.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if plan.refit:
        # The folds earned the numbers; this earns the model. Deliberately a fresh net rather
        # than a continuation of fold 0: continuing would train the last fold's weights on data they
        # were held out from, which is a warm start on a different split and not a clean fit on the
        # whole corpus. The cost is a second training run, which is why it is opt-in.
        report["refit"] = _refit_on_everything(
            index, plan, device=device, init_from=init_from, freeze_backbone=freeze_backbone,
            out_dir=out_dir, artifact_gripper=artifact_gripper)
        if out_dir is not None:
            (Path(out_dir) / "epochs.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _refresh_curve(directory: Path) -> None:
    """Redraw `curve.png` from the epochs written so far, so a run can be watched rather than waited on.

    `deep report` draws the same chart once a run is finished; this redraws it after every epoch,
    from data that is already on disk, so the hours in between are not just a JSON file and a log
    line every ten minutes.

    Never raises, and that is the whole contract. This runs inside the training loop. A run that
    died because matplotlib could not write a PNG would be hours lost for a picture, so every failure
    here is swallowed: a missing backend, a locked file (a customer with the image open in a viewer
    on Windows), a report too short to plot.

    It costs one epoch's worth of nothing: the write is milliseconds against an epoch of roughly
    ten minutes.
    """
    try:
        from src.robot.grasping.deep.eval.run_report import (  # noqa: PLC0415
            build_report,
            write_curve,
        )

        report = build_report(directory.name, directory)
        if len(report.epochs) < 2:
            return                                   # one point is not a curve
        write_curve(report, directory / "curve.png")
    except (Exception, SystemExit):  # noqa: BLE001 (a picture must never end a training run)
        # `SystemExit` is listed by name. `build_report` raises it for a missing `epochs.json`,
        # because it grew up as a CLI helper where exiting is the right answer. `SystemExit`
        # derives from `BaseException`, so a plain `except Exception` does not catch it and a
        # six-hour run would die inside the call that draws a picture.
        #
        # `BaseException` is not used. That would also swallow `KeyboardInterrupt`, and an
        # operator pressing Ctrl+C on a training run must be obeyed rather than logged.
        logger.debug("could not refresh curve.png", exc_info=True)


def _refit_on_everything(index: CorpusIndex, plan: SetTrainingPlan, *, device: Any,
                         init_from: "str | Path | None", freeze_backbone: bool,
                         out_dir: "str | Path | None",
                         artifact_gripper: str | None) -> dict[str, Any]:
    """Train one net on every unit and write it as the artifact. Returns training metrics only.

    The fold pass has to hold assets back to produce an honest held-out estimate. The shipped model
    has no such reason. Without this pass the artifact is fold 0's net, and the units of the folds
    that were not run never reach the weights a cell would serve. On a customer's corpus that is a
    share of their own catalogue.

    No held-out number comes out of here, and that is not an omission. This net trained on
    everything, so every unit is a training unit and any "held-out" score would be a memorisation
    score wearing the wrong name. The numbers to quote for a run are the fold pass's; these are
    training metrics and the keys say so.

    No collapse floor either. That check reads `held_seed_spread_deg`, which does not exist
    here. A collapse would already have stopped the fold pass, which runs first.
    """
    from src.robot.grasping.deep.net.set_generator import SetGenerator  # noqa: PLC0415

    net = SetGenerator(plan.model).to(device)
    inherited = _load_weights(net, init_from, device) if init_from is not None else None
    if freeze_backbone:
        for parameter in net.backbone.parameters():
            parameter.requires_grad_(False)
    optimiser = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                  lr=plan.learning_rate, weight_decay=plan.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=plan.epochs)
    # A different seed stream from any fold, so the refit is not a replay of fold 0's unit order.
    rng = np.random.default_rng(plan.seed + 500)
    generator = torch.Generator().manual_seed(plan.seed + 500)

    everything = np.arange(len(index.units))
    logger.info("refit on EVERYTHING: %d unit(s), %d parameter(s); the folds earned the numbers, "
                "this earns the model", len(everything), net.parameter_count)
    rows: list[dict[str, float]] = []
    for epoch in range(plan.epochs):
        order = rng.permutation(everything)
        if plan.train_units is not None:
            order = order[:plan.train_units]
        if plan.labelled_unit_share is not None:
            order = rebalance_units(order, index.units, plan.labelled_unit_share, rng)
        started = time.perf_counter()
        with epoch_bar(total=max(1, (len(order) + plan.batch - 1) // plan.batch),
                       epoch=epoch + 1, epochs=plan.epochs, fold=-1) as bar:
            train = _run_epoch(net, index, order, plan, optimiser, rng, generator, bar=bar)
        schedule.step()
        rows.append({"epoch": epoch,
                     "seconds": round(time.perf_counter() - started, 1),
                     **{f"train_{k}": round(v, 5) for k, v in train.items()}})
        logger.info("refit epoch %2d/%d  total %.4f  cover %.3f  top1 %.3f  %s %.1f deg  %.0f s",
                    epoch + 1, plan.epochs, train["total"], train["coverage"], train["top1_hit"],
                    plan.step.target, train[f"{plan.step.target}_error_deg"], rows[-1]["seconds"])

    out: dict[str, Any] = {"units": int(len(everything)), "epochs": rows,
                           "held_out": None,
                           "why_no_held_out": "trained on every unit; any held-out score would be a "
                                              "memorisation score. Quote the fold pass instead."}
    if inherited is not None:
        out["inherited"] = inherited
    if out_dir is not None:
        out["artifact"] = _write_artifact(Path(out_dir), net, plan, index.grippers,
                                          artifact_gripper,
                                          card=_artifact_report(plan, index, None, refit=True))
    return out


def plan_with_slots(plan: SetTrainingPlan, slots: int) -> SetTrainingPlan:
    """The same plan with a different number of head slots, and nothing else touched.

    A `slots=4` against `slots=1` comparison is only worth anything if exactly one thing differs.
    `slots=1` is one grasp per seed, supervised against whichever label happens to be nearest, and
    unable to score above one on coverage by construction.
    """
    return replace(plan, model=replace(plan.model, head=replace(plan.model.head, slots=slots)))
