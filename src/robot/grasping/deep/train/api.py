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

__all__ = ["GeneratorTraining", "TrainingContext"]

#: The line separator for the rendered text, built without a backslash escape.
NEWLINE = chr(10)


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
    """Fit a set generator. Construct with a factory, then call `train()` once."""

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
                  probe_units: int = 256) -> "GeneratorTraining":
        """Raw handles, no recipe resolution. The `from_components` analogue.

        `corpus` takes a directory to walk or an explicit sequence, so a notebook can hand in
        `stratified_scenes(...)` directly.
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
                artifact_gripper=artifact_gripper, probe_units=probe_units))

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
                    probe_units: int = 256) -> "GeneratorTraining":
        """Resolve a named bundle plus explicit overrides into a plan. The `from_robot_config` analogue.

        Fails closed with `ValueError`: an unknown recipe or tier refuses rather than falling back to
        the defaults, because a customer who typed `v2` before it exists would otherwise get `v1`'s
        behaviour under `v2`'s name, which is the failure a version string exists to prevent.
        """
        plan, notes = build_plan(recipe=recipe, tier=tier, overrides=overrides, base=base)
        built = cls.from_plan(
            corpus=corpus, plan=plan, out_dir=out_dir, device=device, init_from=init_from,
            freeze_backbone=freeze_backbone, resume=resume, artifact_gripper=artifact_gripper,
            probe_units=probe_units)
        built.recipe_notes = notes
        return built

    # ------------------------------------------------------------------ opt-in side channels
    def attach_progress_listener(
            self, listener: Callable[[Mapping[str, Any]], None] | None) -> None:
        """Called once per finished epoch with that epoch's raw row. `None` detaches.

        Opt-in and off by default, so a run started without one is byte-identical. The terminal
        bar is unaffected: `train/progress.py` writes to stderr only when stderr is a tty, and the
        training core holds no `print` at all.

        Called inline, on the thread running the training loop. A slow listener slows the run, so
        a service hands the row to a queue and returns. Exceptions from a listener are not caught
        here: a broken callback surfaces rather than corrupting a six-hour run silently.
        """
        self._on_epoch = listener

    # ------------------------------------------------------------------ what is about to run
    def describe(self) -> str:
        """What this run will do, as operator text, before it costs anything.

        This exists so a mistyped run can be stopped in the first second rather than the sixth
        hour. It is the counterpart to `TrainingRunReport.render()`, which says what happened, and
        the two deliberately do not repeat each other.

        ASCII only: this reaches a Windows console under cp1252.
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
        if self.plan.recipe or self.plan.tier:
            lines.append(f"  recipe     {self.plan.recipe or 'none'}"
                         + (f"  tier {self.plan.tier}" if self.plan.tier else ""))
        if self.plan.tier == "smoke":
            lines.append("  ** SMOKE TIER: proves the chain closes on your corpus and your box. "
                         "It says NOTHING about grasp quality and is not a model to deploy.")
        return NEWLINE.join(lines)

    # ------------------------------------------------------------------ the one verb
    def train(self) -> TrainingRunReport:
        """Run the folds, the probes and the optional refit, and return what happened.

        Raises `ValueError` for anything the run cannot proceed with: a corpus that cannot be cut
        into `plan.folds` asset-disjoint groups, a checkpoint whose plan does not match a `resume`,
        an ambiguous `artifact_gripper`. Never raises `SystemExit`.

        Writes `epochs.json` after every epoch and the artifact at the end, exactly as a shell run
        does. `report.json` is not written here: see `write_report`, which is separate so a caller
        who wants the numbers without the file can have that.
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
            probe_units=context.probe_units, on_epoch=self._on_epoch)
        return TrainingRunReport.from_trainer(raw, self.plan)

    def write_report(self, report: TrainingRunReport, path: str | Path | None = None) -> Path:
        """Write `report.json` beside the run, and return where it went.

        `deep report --run DIR` reads this file: it merges `report.json` when `epochs.json` carries
        no `held_floor`. A run that skips it leaves an output directory the analysis tool degrades
        on, and the five stamps this file carries exist nowhere else.
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
