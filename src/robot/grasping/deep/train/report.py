"""What a training run did, as a typed object rather than a dictionary the CLI decorates.

Five stamps ride here that the trainer's own dict does not carry: `control_level`, `slot_mixing`,
`target`, `axis_mode` and `backbone`. `control_level` is the one that matters: a control run fits a
synthetic target derived from an input channel, its numbers are not grasping numbers, and losing
that stamp is how a diagnostic becomes a quoted result.

This wraps the trainer's dict, it does not replace it. `train_set_generator` still returns the same
mapping, and `epochs.json` and `report.json` keep their exact shape so `deep report --run DIR`
reads a run written from code the same way it reads one written from the shell. The typed object is
what a program reads; the JSON is what the tools read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.grasping.deep.train.trainer import SetTrainingPlan

__all__ = ["EpochRow", "TrainingOutcome", "TrainingRunReport"]


class TrainingOutcome(StrEnum):
    """Why a run ended, readable without reading the epochs."""

    #: Every requested fold ran and an artifact was written.
    COMPLETED = "completed"
    #: The folds ran, and the artifact writer refused. The reason is in `artifact["reason"]`.
    ARTIFACT_REFUSED = "artifact_refused"
    #: `plan.collapse_floor_deg` tripped: the head had collapsed towards a constant.
    COLLAPSED = "collapsed"
    #: A resume into a checkpoint that had already finished. Legal, and it produces no epochs.
    NOTHING_TO_RESUME = "nothing_to_resume"


@dataclass(frozen=True, slots=True)
class EpochRow:
    """One logged epoch, held-out numbers beside the training ones.

    A view over the raw row, not a replacement for it. `raw` carries every column the trainer
    wrote, because the columns change as instruments are added and a typed subset that silently
    dropped a new one would be worse than no typing at all.
    """

    fold: int
    epoch: int
    seconds: float
    total: float
    held_total: float
    train_top1_hit: float
    held_top1_hit: float
    held_coverage: float
    held_offset_error_mm: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EpochRow":
        def number(key: str) -> float:
            value = row.get(key)
            return float(value) if isinstance(value, (int, float)) else float("nan")

        return cls(fold=int(row.get("fold", 0)), epoch=int(row.get("epoch", 0)),
                   seconds=number("seconds"), total=number("total"),
                   held_total=number("held_total"),
                   train_top1_hit=number("train_top1_hit"),
                   held_top1_hit=number("held_top1_hit"),
                   held_coverage=number("held_coverage"),
                   held_offset_error_mm=number("held_offset_error_mm"), raw=dict(row))


@dataclass(frozen=True, slots=True)
class TrainingRunReport:
    """The result of one `GeneratorTraining.train()`, in the shape a program can branch on."""

    outcome: TrainingOutcome
    plan: "SetTrainingPlan"
    epochs: tuple[EpochRow, ...]
    artifact: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def succeeded(self) -> bool:
        """Ran to the end and left something a cell can load. Both halves, deliberately.

        A run whose folds all completed and whose artifact was refused is not a success from the
        point of view of the only caller who matters, which is the person who wanted weights.
        """
        return self.outcome is TrainingOutcome.COMPLETED and bool(self.artifact.get("written"))

    @property
    def final(self) -> EpochRow | None:
        """The last epoch, or None.

        Guarded once, here, because an empty list is reachable: with `--resume` on a checkpoint
        already at `plan.epochs` the resume reader returns `first_epoch == plan.epochs`, the epoch
        range is empty and no row is ever appended, after a successful trainer return.
        """
        return self.epochs[-1] if self.epochs else None

    @property
    def is_control(self) -> bool:
        """True when this run fitted a synthetic target. Its numbers are not about grasping."""
        return self.plan.control is not None

    def failure_summary(self) -> str:
        """One line naming what stopped the run and what to change. Empty when it succeeded."""
        if self.succeeded:
            return ""
        if self.outcome is TrainingOutcome.NOTHING_TO_RESUME:
            return (f"the checkpoint had already finished all {self.plan.epochs} epoch(s); raise "
                    f"epochs or point --out somewhere new")
        if self.outcome is TrainingOutcome.COLLAPSED:
            return (f"the head collapsed below {self.plan.collapse_floor_deg} deg of held-out seed "
                    f"spread and the fold was abandoned")
        reason = self.artifact.get("reason")
        if reason:
            return f"the folds ran and no artifact was written: {reason}"
        return f"the run ended as {self.outcome}"

    def render(self) -> str:
        """What happened, as operator text. What was going to happen is `GeneratorTraining.describe`.

        The two do not overlap: this says nothing the configuration block printed before the run
        already said.

        Mirrors `PreflightReport.render()` in the real-cell package: the command becomes a formatter
        that prints this, so a service or a notebook sees the same summary an operator does.

        ASCII only. This reaches a Windows console under cp1252, where a non-ASCII character in a
        printed report line raises.
        """
        lines: list[str] = []
        if self.is_control:
            lines.append(f"  ** CONTROL RUN ({self.plan.control}): fitted to a synthetic target. "
                         f"These numbers are NOT about grasping.")
        if self.artifact.get("written"):
            lines.append(f"  artifact   {self.artifact['weights']}")
        elif self.artifact:
            lines.append(f"  artifact   NOT written: {self.artifact.get('reason')}")
        last = self.final
        if last is not None:
            lines.append(f"  held total {last.held_total:.4f}  coverage {last.held_coverage:.4f}  "
                         f"top1 {last.held_top1_hit:.4f}  "
                         f"offset {last.held_offset_error_mm:.2f} mm")
        else:
            lines.append("  no epoch ran")
        summary = self.failure_summary()
        if summary:
            lines.append(f"  outcome    {self.outcome}: {summary}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """The exact JSON written as `report.json`.

        The keys are fixed. `eval/run_report.build_report` merges `report.json` when `epochs.json`
        carries no `held_floor`, so changing the shape here quietly degrades `deep report --run DIR`
        on every run written from code.
        """
        out = dict(self.raw)
        out["control_level"] = self.plan.control
        out["slot_mixing"] = self.plan.model.head.slot_mixing
        out["target"] = self.plan.step.target
        out["axis_mode"] = self.plan.model.head.axis_mode
        out["recipe"] = self.plan.recipe
        out["tier"] = self.plan.tier
        out["backbone"] = {"width": self.plan.model.backbone.width,
                           "depth": self.plan.model.backbone.depth,
                           "heads": self.plan.model.backbone.heads}
        out["outcome"] = str(self.outcome)
        return out

    @classmethod
    def from_trainer(cls, raw: Mapping[str, Any], plan: "SetTrainingPlan") -> "TrainingRunReport":
        """Wrap what `train_set_generator` returned."""
        rows = tuple(EpochRow.from_row(row) for row in raw.get("epochs", ()))
        artifact = dict(raw.get("artifact", {}) or {})
        if not rows and plan.epochs:
            outcome = TrainingOutcome.NOTHING_TO_RESUME
        elif raw.get("collapsed"):
            outcome = TrainingOutcome.COLLAPSED
        elif artifact and not artifact.get("written"):
            outcome = TrainingOutcome.ARTIFACT_REFUSED
        else:
            outcome = TrainingOutcome.COMPLETED
        return cls(outcome=outcome, plan=plan, epochs=rows, artifact=artifact, raw=dict(raw))
