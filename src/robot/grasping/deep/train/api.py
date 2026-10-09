"""Training the K-slot set generator from code, in the idiom of `AutonomousGraspService`.

Parameters go in code, and there is no YAML here. `SetTrainingPlan` is a frozen dataclass with
three nested config objects, and it reaches strictly more of the stack than the command line does:
`learning_rate`, `weight_decay`, `eval_units`, most of `SlotHeadConfig` and every `SampleSpec`
field but `points` have no flag at all. `build_plan(recipe=..., overrides=...)` is the convenience
layer for the named bundles; constructing a `SetTrainingPlan(...)` by hand stays first-class.

The shape mirrors `AutonomousGraspService` (`execution/autonomous_grasp/service.py`): a dataclass
whose fields are the wiring surface, keyword-only classmethod factories that mirror the two ways a
caller can arrive, one verb, a frozen typed report, and opt-in side effects as separate methods
rather than constructor booleans.

    from src.robot.grasping.deep.corpus.discovery import scene_files
    from src.robot.grasping.deep.train.api import GeneratorTraining
    from src.robot.grasping.deep.train.plan import PlanOverrides

    run = GeneratorTraining.from_recipe(
        corpus="logs/dl/clouds/v5_jaw",
        recipe="v1", tier="full",
        overrides=PlanOverrides(target="approach", slots=4),
        out_dir="logs/dl/models/arm_jaw", artifact_gripper="2f85")
    report = run.train()
    print(report.render())
    if report.succeeded:
        run.write_report(report)

This is not a second trainer. `train_set_generator` keeps its name, its signature and its module,
and this wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.robot.grasping.deep.corpus.discovery import scene_files
from src.robot.grasping.deep.train.plan import PlanOverrides, build_plan
from src.robot.grasping.deep.train.report import TrainingRunReport

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.deep.train.trainer import SetTrainingPlan

__all__ = ["CorpusProbe", "GeneratorTraining", "TrainingContext"]

#: The line separator for the rendered text, built without a backslash escape.
NEWLINE = chr(10)


@dataclass(frozen=True, slots=True)
class CorpusProbe:
    """What a number measured on this corpus can possibly mean, before anything is trained.

    Three instruments, none of which touches a trained weight, so all three answer in minutes on a
    CPU and answer about the DATA rather than about a model:

        ceiling   what a perfect head scores here. Below 1.0, because a seed that admits several
                  grasps caps what any single answer can score, and K caps how many it can answer.
        floor     what four heads that learned nothing score. A hit rate cannot be read without
                  knowing what zero effort achieves, and `top_down` in particular is the grasp a
                  cell with no model at all would try.
        headroom  how many degrees of approach signal exist above a single constant direction, at
                  four granularities. It separates "the model cannot reach the signal" from "the
                  signal is not there at this granularity".

    A trained number between the floor and the ceiling is the only kind that means anything, and
    the gap between them is how much there was to learn. Where floor and ceiling nearly meet, the
    corpus has nothing to teach and a better model will not change that.
    """

    units: int
    ceiling: Mapping[str, float]
    floor: Mapping[str, Mapping[str, float]]
    headroom: Mapping[str, Mapping[str, float]]

    def render(self) -> str:
        keys = ("top1_hit", "coverage", "approach_error_deg", "offset_error_mm")
        lines = [f"  probed over {self.units} training unit(s), no weights involved",
                 "  " + f"{'arm':14}" + "".join(f"{key:>20}" for key in keys)]
        rows: list[tuple[str, Mapping[str, float]]] = [*sorted(self.floor.items()),
                                                       ("ORACLE ceiling", self.ceiling)]
        for name, values in rows:
            lines.append("  " + f"{name:14}"
                         + "".join(f"{float(values.get(key, 0.0)):20.3f}" for key in keys))
        for label, block in sorted(self.headroom.items()):
            lines.append(f"  approach signal, {label}: "
                         + ", ".join(f"{key} {float(block[key]):.2f} deg"
                                     for key in ("straight_down", "global", "per_object",
                                                 "per_seed") if key in block))
        return NEWLINE.join(lines)

    def __str__(self) -> str:
        return self.render()

    def as_dict(self) -> dict[str, Any]:
        return {"units": self.units, "ceiling": dict(self.ceiling),
                "floor": {k: dict(v) for k, v in self.floor.items()},
                "headroom": {k: dict(v) for k, v in self.headroom.items()}}


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Run context, as opposed to the run description that is `SetTrainingPlan`.

    Where the data is, where output goes, what to inherit from, which hand the artifact claims.
    Every field here is a `train_set_generator` keyword and none of it is plan state, which is why
    it is a separate object: a model card quotes the plan, and none of this belongs on a card.
    """

    corpus: tuple[Path, ...]
    out_dir: Path | None = None
    device: str | None = None
    init_from: Path | None = None
    freeze_backbone: bool = False
    resume: bool = False
    artifact_gripper: str | None = None
    probe_units: int = 256
    #: Train across these hands on the same clouds, each read with its grasp table. None is the hand
    #: each cloud was extracted for.
    hands: tuple[str, ...] | None = None


def _corpus_paths(corpus: Sequence[str | Path] | str | Path) -> tuple[Path, ...]:
    """A directory is walked, a sequence is taken as given.

    The walk is not optional for a directory. `scene_files` is recursive because a scene id counts
    within one dataset, so two shards hold different scenes with identical names; a flat listing
    overwrites the collisions silently.
    """
    if isinstance(corpus, (str, Path)):
        return tuple(scene_files(corpus))
    paths = tuple(Path(item) for item in corpus)
    if not paths:
        raise ValueError("the corpus is empty; pass a directory to walk or a non-empty sequence")
    return paths


@dataclass
class GeneratorTraining:
    """Train a learned grasp generator (the set generator) on a point-cloud corpus.

        run = GeneratorTraining.from_recipe(corpus="corpora/my_parts", recipe="v1", tier="smoke",
                                            out_dir="models/my_parts")
        print(run.describe())       # what it will do, before it costs anything
        print(run.probe())          # the floor and the ceiling of this corpus; trains nothing
        report = run.train()
        run.write_report(report)

    A trained artifact drives a cell only once its proof has passed (``deep/promotion.py``); until then it can be
    inspected and evaluated (``deep propose``, the ladder's deep rung).

    Attributes:
        plan (SetTrainingPlan): Every training setting, resolved.
        context (TrainingContext): The corpus, the out dir, the device and the run's handles.
        recipe_notes (Mapping[str, Any]): What the recipe and the tier set, for a caller that logs it (default: {}).
    """

    plan: "SetTrainingPlan"
    context: TrainingContext
    #: What a recipe or tier supplied, and what the caller had already chosen. Empty without one.
    recipe_notes: Mapping[str, Any] = field(default_factory=dict)
    _on_epoch: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_plan(cls, *, corpus: Sequence[str | Path] | str | Path,
                  plan: "SetTrainingPlan | None" = None,
                  out_dir: str | Path | None = None,
                  device: str | None = None,
                  init_from: str | Path | None = None,
                  freeze_backbone: bool = False,
                  resume: bool = False,
                  artifact_gripper: str | None = None,
                  probe_units: int = 256,
                  hands: Sequence[str] | None = None) -> "GeneratorTraining":
        """A run from a plan you built yourself, with no recipe resolution.

        Args:
            corpus (Sequence[str | Path] | str | Path): The corpus: a directory to walk for scene files, or an explicit
                sequence of them (such as ``stratified_scenes(...)``).
            plan (SetTrainingPlan | None): Every setting; ``None`` is the defaults (default: None).
            out_dir (str | Path | None): Where the run writes ``epochs.json``, the fold checkpoints and the artifact;
                required before ``train()`` (default: None).
            device (str | None): ``"cuda"``, ``"cpu"`` or ``"mps"``; ``None`` is ``WILLY_DEVICE`` or the first of CUDA,
                MPS and the CPU (default: None).
            init_from (str | Path | None): Start from the weights in this checkpoint instead of from random:
                fine-tuning, a new run with its own optimiser and split, not a resume. A checkpoint that does not fit is
                refused (default: None).
            freeze_backbone (bool): Train only the heads and hold the encoder still: the cheap fine-tuning for a small
                new corpus, since 99 % of the parameters are in the backbone (default: False).
            resume (bool): Continue an interrupted run in ``out_dir``; any change to the corpus or the plan is refused
                (default: False).
            artifact_gripper (str | None): Which hand the written artifact plans for, by model name; taken from the
                corpus where it carries exactly one, refused rather than guessed where it carries several (default:
                None).
            probe_units (int): How many units each probe draws (default: 256).
            hands (Sequence[str] | None): Train across these hands on the same clouds, each read from its grasp table
                (``datagen build-grasp-tables``); ``None`` is the hand each cloud was extracted for (default: None).

        Returns:
            GeneratorTraining: The run; nothing is read or trained yet.
        """
        from src.robot.grasping.deep.train.trainer import (  # noqa: PLC0415
            SetTrainingPlan,
        )

        return cls(
            plan=plan or SetTrainingPlan(),
            context=TrainingContext(
                corpus=_corpus_paths(corpus),
                out_dir=Path(out_dir) if out_dir is not None else None,
                device=device,
                init_from=Path(init_from) if init_from is not None else None,
                freeze_backbone=freeze_backbone, resume=resume,
                artifact_gripper=artifact_gripper, probe_units=probe_units,
                hands=tuple(hands) if hands is not None else None))

    @classmethod
    def from_recipe(cls, *, corpus: Sequence[str | Path] | str | Path,
                    recipe: str | None = None,
                    tier: str | None = None,
                    overrides: PlanOverrides | None = None,
                    base: "SetTrainingPlan | None" = None,
                    out_dir: str | Path | None = None,
                    device: str | None = None,
                    init_from: str | Path | None = None,
                    freeze_backbone: bool = False,
                    resume: bool = False,
                    artifact_gripper: str | None = None,
                    probe_units: int = 256,
                    hands: Sequence[str] | None = None) -> "GeneratorTraining":
        """A run from a named recipe and tier, plus what you chose explicitly.

        Args:
            corpus (Sequence[str | Path] | str | Path): The corpus: a directory, or a sequence of scene files.
            recipe (str | None): The frozen recipe, such as ``"v1"``; ``None`` is the plan's defaults (default: None).
            tier (str | None): ``"smoke"`` (two epochs on few units: proves the chain, says nothing about grasp
                quality), ``"full"``, ...; ``None`` the recipe's own (default: None).
            overrides (PlanOverrides | None): The settings you choose explicitly; they outrank the recipe and the tier
                (default: None).
            base (SetTrainingPlan | None): The plan everything is laid onto; ``None`` the defaults (default: None).
            out_dir (str | Path | None): Where the run writes ``epochs.json``, the fold checkpoints and the artifact;
                required before ``train()`` (default: None).
            device (str | None): ``"cuda"``, ``"cpu"`` or ``"mps"``; ``None`` is ``WILLY_DEVICE`` or the first of CUDA,
                MPS and the CPU (default: None).
            init_from (str | Path | None): Start from the weights in this checkpoint instead of from random:
                fine-tuning, a new run with its own optimiser and split, not a resume. A checkpoint that does not fit is
                refused (default: None).
            freeze_backbone (bool): Train only the heads and hold the encoder still: the cheap fine-tuning for a small
                new corpus, since 99 % of the parameters are in the backbone (default: False).
            resume (bool): Continue an interrupted run in ``out_dir``; any change to the corpus or the plan is refused
                (default: False).
            artifact_gripper (str | None): Which hand the written artifact plans for, by model name; taken from the
                corpus where it carries exactly one, refused rather than guessed where it carries several (default:
                None).
            probe_units (int): How many units each probe draws (default: 256).
            hands (Sequence[str] | None): Train across these hands on the same clouds, each read from its grasp table
                (``datagen build-grasp-tables``); ``None`` is the hand each cloud was extracted for (default: None).

        Returns:
            GeneratorTraining: The run; nothing is read or trained yet.

        Raises:
            ValueError: An unknown recipe or tier: refused rather than falling back to the defaults, so a ``v2`` typed
                before it exists never runs ``v1`` under its name.
        """
        plan, notes = build_plan(recipe=recipe, tier=tier, overrides=overrides, base=base)
        built = cls.from_plan(
            corpus=corpus, plan=plan, out_dir=out_dir, device=device, init_from=init_from,
            freeze_backbone=freeze_backbone, resume=resume, artifact_gripper=artifact_gripper,
            probe_units=probe_units, hands=hands)
        built.recipe_notes = notes
        return built

    # ------------------------------------------------------------------ opt-in side channels
    def attach_progress_listener(
            self, listener: Callable[[Mapping[str, Any]], None] | None) -> None:
        """Be told as each epoch finishes, on the training thread. A slow listener slows the run; its exceptions are not
        caught.

        Args:
            listener (Callable[[Mapping[str, Any]], None] | None): Called with the epoch's raw row; ``None`` detaches.
        """
        self._on_epoch = listener

    # ------------------------------------------------------------------ what is about to run
    def describe(self) -> str:
        """What this run will do, before it costs anything, so a mistyped run can be stopped in the first second rather
        than the sixth hour.

        Returns:
            str: The plan, the corpus and the hands, as operator text, ASCII.
        """
        head, backbone = self.plan.model.head, self.plan.model.backbone
        lines = [
            f"  corpus     {len(self.context.corpus)} scene file(s)",
            f"  labels     {self.plan.control or 'the corpus labels'}",
            f"  target     {self.plan.step.target} ({head.axis_mode})",
            f"  slots      {head.slots} ({head.slot_mixing})",
            f"  backbone   width {backbone.width} depth {backbone.depth} heads {backbone.heads}",
            f"  schedule   {self.plan.epochs} epoch(s), {self.plan.run_folds} of "
            f"{self.plan.folds} fold(s), refit {'on' if self.plan.refit else 'off'}",
        ]
        if self.context.hands:
            lines.append(f"  hands      {', '.join(self.context.hands)}")
        if self.plan.recipe or self.plan.tier:
            lines.append(f"  recipe     {self.plan.recipe or 'none'}"
                         + (f"  tier {self.plan.tier}" if self.plan.tier else ""))
        if self.plan.tier == "smoke":
            lines.append("  ** SMOKE TIER: proves the chain closes on your corpus and your box. "
                         "It says NOTHING about grasp quality and is not a model to deploy.")
        return NEWLINE.join(lines)

    # ------------------------------------------------------------------ before it costs anything
    def probe(self, *, units: int = 256, seed: int = 0) -> CorpusProbe:
        """The floor, the ceiling and the approach headroom of this corpus, with no weights, on a CPU, in minutes: a hit
        rate is unreadable without both ends of its scale.

        Args:
            units (int): How many training units each instrument draws; more costs time linearly (default: 256).
            seed (int): The draw's seed; the same seed draws the same units (default: 0).

        Returns:
            CorpusProbe: The floor (what a straight-down grasp scores), the ceiling and the headroom; prints as itself.

        Raises:
            ValueError: The corpus yields no supervised seed at all: nothing can be learned there.
        """
        from src.robot.grasping.deep.eval.probes import (  # noqa: PLC0415 (torch, only for this)
            approach_headroom, baseline_floor, oracle_ceiling,
        )
        from src.robot.grasping.deep.train.trainer import corpus_index  # noqa: PLC0415

        index = corpus_index(self.context.corpus, self.context.hands)
        return CorpusProbe(
            units=units,
            ceiling=oracle_ceiling(index, self.plan, units=units, seed=seed,
                                   device=self.context.device or "cpu"),
            floor=baseline_floor(index, self.plan, units=units, seed=seed),
            headroom=approach_headroom(index, self.plan, units=units, seed=seed),
        )

    # ------------------------------------------------------------------ the one verb
    def train(self) -> TrainingRunReport:
        """Run the folds, the probes and the optional refit; writes ``epochs.json`` after every epoch and the artifact
        at the end, as a shell run does.

        Returns:
            TrainingRunReport: What happened: per fold the held-out numbers against their floors, the probes, where the
                artifact went; prints as itself. ``report.json`` is written by :meth:`write_report`.

        Raises:
            ValueError: A corpus that cannot be cut into ``plan.folds`` asset-disjoint groups, a checkpoint whose plan
                does not match a ``resume``, an ambiguous ``artifact_gripper``. Never ``SystemExit``.
        """
        from src.robot.grasping.deep.train.trainer import (  # noqa: PLC0415
            train_set_generator,
        )

        context = self.context
        if context.out_dir is not None:
            context.out_dir.mkdir(parents=True, exist_ok=True)
        raw = train_set_generator(
            context.corpus, self.plan, out_dir=context.out_dir, device=context.device,
            init_from=context.init_from, freeze_backbone=context.freeze_backbone,
            resume=context.resume, artifact_gripper=context.artifact_gripper,
            probe_units=context.probe_units, on_epoch=self._on_epoch, hands=context.hands)
        return TrainingRunReport.from_trainer(raw, self.plan)

    def write_report(self, report: TrainingRunReport, path: str | Path | None = None) -> Path:
        """Write ``report.json`` beside the run; ``deep report --run DIR`` reads it.

        Args:
            report (TrainingRunReport): What :meth:`train` returned.
            path (str | Path | None): Where to write it; ``None`` is ``out_dir/report.json`` (default: None).

        Returns:
            Path: Where it went.
        """
        import json  # noqa: PLC0415

        if path is not None:
            target = Path(path)
        elif self.context.out_dir is not None:
            target = self.context.out_dir / "report.json"
        else:
            raise ValueError("no out_dir was given, so there is nowhere to write report.json; "
                             "pass an explicit path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target
