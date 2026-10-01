"""What a detector training run did, as a typed object a program can branch on; the JSON beside it is what tools read.

It wraps the trainer's dictionary rather than replacing it: ``as_dict()`` is that dictionary plus the plan's recipe
and tier and the outcome, and it is exactly what ``DetectorTraining.write_report`` writes as ``report.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.models.detection.closed_set.training.plan import DetectorPlan

__all__ = ["DetectionEpoch", "DetectorTrainingOutcome", "DetectorTrainingReport"]


class DetectorTrainingOutcome(StrEnum):
    """Why a run ended, readable without reading the epochs."""

    #: Every planned epoch ran; the best one is the model.
    COMPLETED = "completed"
    #: ``patience`` epochs passed without a better validation mAP; the best epoch is the model. A success.
    STOPPED_EARLY = "stopped_early"
    #: The loss went non-finite and stayed there. A best epoch before it, if any, is still on disk.
    DIVERGED = "diverged"
    #: A resume into a run that had already finished every epoch. Legal, and it trains nothing.
    NOTHING_TO_RESUME = "nothing_to_resume"


@dataclass(frozen=True, slots=True)
class DetectionEpoch:
    """One epoch's row, the numbers typed and the raw row kept."""

    epoch: int
    seconds: float
    train_loss: float
    val_loss: float
    map: float
    map50: float
    map75: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DetectionEpoch":
        def number(key: str) -> float:
            value = row.get(key)
            return float(value) if isinstance(value, (int, float)) else float("nan")

        return cls(epoch=int(row.get("epoch", 0)), seconds=number("seconds"), train_loss=number("train_loss"),
                   val_loss=number("val_loss"), map=number("map"), map50=number("map50"), map75=number("map75"),
                   raw=dict(row))


@dataclass(frozen=True, slots=True)
class DetectorTrainingReport:
    """The result of one ``DetectorTraining.train()``."""

    outcome: DetectorTrainingOutcome
    plan: "DetectorPlan"
    epochs: tuple[DetectionEpoch, ...]
    best_epoch: int | None
    best: Mapping[str, Any] = field(default_factory=dict)
    artifact: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def succeeded(self) -> bool:
        """Ran to its end and left a model a cell can load. Both halves, deliberately."""
        return (self.outcome in (DetectorTrainingOutcome.COMPLETED, DetectorTrainingOutcome.STOPPED_EARLY)
                and bool(self.artifact.get("written")))

    @property
    def model_dir(self) -> str | None:
        """The folder RtDetrObjectDetector and ``models.rtdetr.model_path`` take, when a model was written."""
        return str(self.artifact["model_dir"]) if self.artifact.get("written") else None

    @property
    def validated(self) -> bool:
        """True when the best epoch was chosen by a validation mAP rather than by the training loss."""
        value = self.best.get("map") if self.best else None
        return isinstance(value, (int, float)) and not math.isnan(value)

    def failure_summary(self) -> str:
        """One line naming what stopped the run and what to change. Empty when it succeeded."""
        if self.succeeded:
            return ""
        if self.outcome is DetectorTrainingOutcome.NOTHING_TO_RESUME:
            return (f"the run had already finished all {self.plan.epochs} epoch(s); raise epochs to train on, or "
                    f"point out_dir somewhere new")
        if self.outcome is DetectorTrainingOutcome.DIVERGED:
            return (f"{self.artifact.get('reason', 'the loss went non-finite')}; lower learning_rate, or set "
                    f"amp='fp16' or 'off'")
        return f"the run ended as {self.outcome} and wrote no model"

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """What happened, as operator text. What was going to happen is ``DetectorTraining.describe``.

        ASCII only: this reaches a Windows console under cp1252.
        """
        lines: list[str] = []
        if self.plan.tier == "smoke":
            lines.append("  ** SMOKE TIER: proves the chain closes. These numbers are NOT about detection quality.")
        if self.artifact.get("written"):
            lines.append(f"  model      {self.artifact['model_dir']} (epoch {self.best_epoch}, "
                         f"{'averaged weights' if self.plan.ema else 'weights'})")
        elif self.artifact:
            lines.append(f"  model      NOT written: {self.artifact.get('reason', 'no epoch finished')}")
        if self.validated:
            dataset = self.raw.get("dataset", {}) if self.raw else {}
            lines.append(f"  mAP        {self.best['map']:.3f} (mAP50 {self.best['map50']:.3f}, mAP75 "
                         f"{self.best['map75']:.3f}) on {dataset.get('val_images', '?')} validation image(s)")
            per_class = self.best.get("per_class_ap") or {}
            if per_class:
                lines.append("  per class  " + ", ".join(f"{name} {value:.3f}" for name, value in per_class.items()))
        elif self.epochs:
            lines.append("  mAP        none: no validation box, so the best epoch is the lowest training loss")
        if self.epochs:
            seconds = sum(row.seconds for row in self.epochs if not math.isnan(row.seconds))
            speed = self.raw.get("train_images_per_second") if self.raw else None
            lines.append(f"  epochs     {len(self.epochs)} of {self.plan.epochs} in {seconds:.0f} s"
                         + (f", {speed} training image(s)/s" if speed else "")
                         + (f" on {self.raw.get('device')}" if self.raw and self.raw.get("device") else ""))
        else:
            lines.append("  no epoch ran")
        if self.outcome is DetectorTrainingOutcome.STOPPED_EARLY:
            lines.append(f"  outcome    stopped early: no better validation mAP for {self.plan.patience} epoch(s) "
                         f"after epoch {self.best_epoch}")
        summary = self.failure_summary()
        if summary:
            lines.append(f"  outcome    {self.outcome}: {summary}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """The exact JSON written as ``report.json``."""
        out = dict(self.raw)
        out["outcome"] = str(self.outcome)
        out["recipe"] = self.plan.recipe
        out["tier"] = self.plan.tier
        out["plan"] = self.plan.as_dict()
        return out

    @classmethod
    def from_trainer(cls, raw: Mapping[str, Any], plan: "DetectorPlan") -> "DetectorTrainingReport":
        """Wrap what ``train_detector`` returned."""
        rows = tuple(DetectionEpoch.from_row(row) for row in raw.get("epochs", ()))
        if raw.get("nothing_to_resume"):
            outcome = DetectorTrainingOutcome.NOTHING_TO_RESUME
        elif raw.get("diverged"):
            outcome = DetectorTrainingOutcome.DIVERGED
        elif raw.get("stopped_early"):
            outcome = DetectorTrainingOutcome.STOPPED_EARLY
        else:
            outcome = DetectorTrainingOutcome.COMPLETED
        return cls(outcome=outcome, plan=plan, epochs=rows, best_epoch=raw.get("best_epoch"),
                   best=dict(raw.get("best") or {}), artifact=dict(raw.get("artifact") or {}), raw=dict(raw))
