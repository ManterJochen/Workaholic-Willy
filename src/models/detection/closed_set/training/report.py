"""What a detector training run did, as a typed object a program can branch on, and the pictures a person reads.

It wraps the trainer's dictionary rather than replacing it: ``as_dict()`` is that dictionary plus the plan's recipe
and tier and the outcome, and it is exactly what ``DetectorTraining.write_report`` writes as ``report.json``.

:func:`write_curves` draws ``curves.png``: the training and validation loss, mAP@0.5:0.95 with mAP50 and mAP75 and the
best epoch, the learning rate, and AP per class at the best epoch. The trainer draws it again after every epoch, as
``GeneratorTraining`` draws its curve, so a run can be watched. :func:`write_html` writes ``report.html``: the curves,
AP per class and every epoch as tables, the dataset and the plan, one file that opens in any browser.
"""

from __future__ import annotations

import base64
import html
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.models.detection.closed_set.training.plan import DetectorPlan

__all__ = ["DetectionEpoch", "DetectorTrainingOutcome", "DetectorTrainingReport", "write_curves", "write_html"]

#: The charts' surface and ink, and the series colours: the first three slots of the validated reference palette
#: (blue, orange, aqua; every check passes, aqua under 3:1 against the surface, so every line is labelled at its end).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e1"
_NEUTRAL = "#8a8985"
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")


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
    #: The head's learning rate at the epoch's end.
    lr: float = float("nan")
    #: Mean recall at 100 detections per image.
    mar100: float = float("nan")
    #: AP@0.5:0.95 per class on the validation split; empty without one.
    per_class_ap: Mapping[str, float] = field(default_factory=dict)
    #: Whether this epoch was the best so far, and so was exported.
    best: bool = False

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "DetectionEpoch":
        def number(key: str) -> float:
            value = row.get(key)
            return float(value) if isinstance(value, (int, float)) else float("nan")

        per_class = row.get("per_class_ap") or {}
        return cls(epoch=int(row.get("epoch", 0)), seconds=number("seconds"), train_loss=number("train_loss"),
                   val_loss=number("val_loss"), map=number("map"), map50=number("map50"), map75=number("map75"),
                   raw=dict(row), lr=number("lr"), mar100=number("mar100"),
                   per_class_ap={str(k): float(v) for k, v in dict(per_class).items()
                                 if isinstance(v, (int, float))},
                   best=bool(row.get("best", False)))


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
                lines.append("  AP per class at the best epoch, best first:")
                width = max(len(str(name)) for name in per_class)
                for name, value in sorted(per_class.items(), key=lambda item: -float(item[1])):
                    lines.append(f"    {str(name):<{width}}  {float(value):.3f}")
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
        folder = self.artifact.get("model_dir")
        for what, target in (("curves", self.artifact.get("curves")),
                             ("report", Path(str(folder)) / "report.html" if folder else None)):
            if target and Path(str(target)).is_file():
                lines.append(f"  {what:<10} {target}")
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


# --------------------------------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------------------------------
def _value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _style(axis: Any, title: str) -> None:
    axis.set_facecolor(_SURFACE)
    axis.set_title(title, loc="left", color=_INK, fontsize=11)
    axis.grid(True, color=_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_NEUTRAL)
    axis.tick_params(colors=_INK_SECONDARY, labelsize=9)


def _lines(axis: Any, epochs: Sequence[int], series: Sequence[tuple[str, Sequence[float]]], *,
           empty: str = "") -> None:
    """Each series a 2 px line in its slot's colour, labelled at its last value; a single epoch is a dot. A legend
    where more than one line is drawn, and ``empty`` in the middle where none is."""
    drawn = 0
    for slot, (name, values) in enumerate(series):
        colour = _SERIES[slot % len(_SERIES)]
        points = [(e, v) for e, v in zip(epochs, values) if not math.isnan(v)]
        if not points:
            continue
        axis.plot(epochs, values, color=colour, linewidth=2.0, label=name,
                  marker="o" if len(points) == 1 else None, markersize=8)
        last_epoch, last_value = points[-1]
        axis.annotate(name, (last_epoch, last_value), xytext=(6, 0), textcoords="offset points", va="center",
                      color=_INK_SECONDARY, fontsize=9)
        drawn += 1
    if drawn > 1:
        axis.legend(loc="best", fontsize=8, frameon=False, labelcolor=_INK_SECONDARY)
    elif not drawn and empty:
        axis.text(0.5, 0.5, empty, ha="center", va="center", transform=axis.transAxes, color=_INK_SECONDARY)


def write_curves(rows: Sequence[Mapping[str, Any]], path: str | Path, *, classes: Sequence[str] = (),
                 best_epoch: int | None = None, title: str = "") -> Path:
    """Draw ``curves.png`` from a run's epoch rows: the losses, the mAPs with the best epoch, the learning rate, and AP
    per class at the best epoch (at the last one where none was best).

    Args:
        rows (Sequence[Mapping[str, Any]]): The trainer's epoch rows, in order.
        path (str | Path): Where the PNG goes.
        classes (Sequence[str]): The dataset's classes, for the AP per class in their order of AP (default: ()).
        best_epoch (int | None): The epoch to mark; ``None`` takes the row marked best last (default: None).
        title (str): What the figure is called, usually the run's folder (default: "").

    Returns:
        Path: Where the PNG went.

    Raises:
        ValueError: No row to draw.
    """
    import matplotlib  # noqa: PLC0415 (heavy, and only a report needs it)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    if not rows:
        raise ValueError("no epoch to draw")
    epochs = [int(row.get("epoch", i + 1)) for i, row in enumerate(rows)]
    if best_epoch is None:
        marked = [int(row["epoch"]) for row in rows if row.get("best")]
        best_epoch = marked[-1] if marked else None
    best_row = next((row for row in rows if int(row.get("epoch", 0)) == best_epoch), rows[-1])

    figure, ((loss_axis, map_axis), (lr_axis, class_axis)) = plt.subplots(2, 2, figsize=(13, 8.5),
                                                                          facecolor=_SURFACE)
    _style(loss_axis, "Loss")
    _lines(loss_axis, epochs, [("training", [_value(r, "train_loss") for r in rows]),
                               ("validation", [_value(r, "val_loss") for r in rows])])
    _style(map_axis, "Validation mAP")
    _lines(map_axis, epochs, [("mAP 0.5:0.95", [_value(r, "map") for r in rows]),
                              ("mAP 0.5", [_value(r, "map50") for r in rows]),
                              ("mAP 0.75", [_value(r, "map75") for r in rows])],
           empty="no validation box: no mAP is measured")
    best_map = _value(best_row, "map")
    if best_epoch is not None and not math.isnan(best_map):
        for axis in (loss_axis, map_axis, lr_axis):
            axis.axvline(best_epoch, color=_NEUTRAL, linestyle="--", linewidth=1.0)
        map_axis.annotate(f"best epoch {best_epoch}: {best_map:.3f}", (best_epoch, best_map), xytext=(6, -14),
                          textcoords="offset points", color=_INK, fontsize=9)
    _style(lr_axis, "Learning rate (head)")
    rates = [_value(r, "lr") for r in rows]
    lr_axis.plot(epochs, rates, color=_SERIES[0], linewidth=2.0, marker="o" if len(rows) == 1 else None,
                 markersize=8)
    if all(rate > 0.0 for rate in rates if not math.isnan(rate)) and any(not math.isnan(rate) for rate in rates):
        lr_axis.set_yscale("log")
    for axis in (loss_axis, map_axis, lr_axis):
        axis.set_xlabel("epoch", color=_INK_SECONDARY, fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    per_class = {str(k): float(v) for k, v in dict(best_row.get("per_class_ap") or {}).items()
                 if isinstance(v, (int, float))}
    _style(class_axis, f"AP per class, epoch {int(best_row.get('epoch', 0))}" if per_class else "AP per class")
    if per_class:
        names = sorted(per_class, key=lambda name: per_class[name])
        values = [per_class[name] for name in names]
        bars = class_axis.barh(names, values, color=_SERIES[0], height=0.7)
        for bar, value in zip(bars, values):
            class_axis.annotate(f"{value:.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2.0),
                                xytext=(4, 0), textcoords="offset points", va="center", color=_INK_SECONDARY,
                                fontsize=8)
        class_axis.set_xlim(0.0, max(1.0, max(values) * 1.12))
        class_axis.tick_params(axis="y", labelsize=8)
        class_axis.grid(False, axis="y")
    else:
        class_axis.text(0.5, 0.5, "no validation box: no AP is measured", ha="center", va="center",
                        transform=class_axis.transAxes, color=_INK_SECONDARY)
        class_axis.set_xticks([])
        class_axis.set_yticks([])
    head = f"{title}: " if title else ""
    summary = (f"best mAP {best_map:.3f} at epoch {best_epoch}" if best_epoch is not None and not math.isnan(best_map)
               else "no validation mAP")
    figure.suptitle(f"{head}{summary}, {len(rows)} epoch(s)", color=_INK, fontsize=13, x=0.01, ha="left")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp.png")
    figure.savefig(temporary, dpi=140, facecolor=_SURFACE)
    plt.close(figure)
    temporary.replace(out)
    return out


def _cell(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "&ndash;" if math.isnan(value) else f"{value:.{digits}f}"
    return html.escape(str(value))


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title: detector training</title>
<style>
:root { color-scheme: light; --surface: #fcfcfb; --surface-2: #f3f2ef; --ink: #0b0b0b; --ink-2: #52514e;
        --rule: #e6e5e1; --accent: #2a78d6; --best: #e8f1fc; }
@media (prefers-color-scheme: dark) {
  :root { color-scheme: dark; --surface: #1a1a19; --surface-2: #232321; --ink: #ffffff; --ink-2: #c3c2b7;
          --rule: #383835; --accent: #3987e5; --best: #1d2b3d; }
}
body { margin: 0; background: var(--surface); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 10px; }
.muted { color: var(--ink-2); }
.facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin: 16px 0; }
.fact { background: var(--surface-2); border-radius: 8px; padding: 10px 12px; }
.fact b { display: block; font-size: 12px; font-weight: 600; color: var(--ink-2); }
img { width: 100%; height: auto; border-radius: 8px; background: #fcfcfb; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--rule); }
th { font-size: 12px; color: var(--ink-2); font-weight: 600; }
tr.best td { background: var(--best); font-weight: 600; }
.bar { display: inline-block; height: 10px; border-radius: 2px; background: var(--accent); }
.scroll { overflow-x: auto; }
</style>
</head>
<body>
<main>
<h1>$title</h1>
<p class="muted">$intro</p>
<div class="facts">$facts</div>
<h2>Curves</h2>
$picture
<h2>AP per class at the best epoch</h2>
$classes
<h2>Every epoch</h2>
<div class="scroll"><table><tr><th>epoch</th><th>s</th><th>lr</th><th>train loss</th><th>val loss</th><th>mAP</th>
<th>mAP 0.5</th><th>mAP 0.75</th><th>mAR 100</th></tr>$epochs</table></div>
<h2>Dataset</h2>
<p>$dataset</p>
$notes
<h2>Plan</h2>
<div class="scroll"><table><tr><th>setting</th><th>value</th></tr>$plan</table></div>
</main>
</body>
</html>
"""


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _fact(label: str, value: str) -> str:
    return '<div class="fact"><b>' + html.escape(label) + "</b>" + value + "</div>"


def _epoch_row(row: DetectionEpoch, best_epoch: int | None) -> str:
    cells = [str(row.epoch), _cell(row.seconds, 0), "&ndash;" if math.isnan(row.lr) else f"{row.lr:.2e}",
             _cell(row.train_loss), _cell(row.val_loss), _cell(row.map), _cell(row.map50), _cell(row.map75),
             _cell(row.mar100)]
    opening = '<tr class="best">' if row.epoch == best_epoch else "<tr>"
    return opening + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr>"


def _class_row(name: str, value: float) -> str:
    width = f"{max(0.0, min(1.0, value)) * 100.0:.1f}%"
    return ("<tr><td>" + html.escape(name) + "</td><td>" + f"{value:.3f}" + '</td><td><span class="bar" style="width:'
            + width + '"></span></td></tr>')


def _curves_png(report: "DetectorTrainingReport", out: Path, curves: str | Path | None, name: str) -> bytes:
    if curves is not None and Path(curves).is_file():
        return Path(curves).read_bytes()
    if not report.epochs:
        return b""
    drawn = out.with_name(out.stem + ".curves.tmp.png")
    try:
        write_curves([row.raw for row in report.epochs], drawn, classes=tuple(report.raw.get("classes", ())),
                     best_epoch=report.best_epoch, title=name)
        return drawn.read_bytes()
    except Exception:  # noqa: BLE001 (a page without its picture is still the page)
        return b""
    finally:
        drawn.unlink(missing_ok=True)


def write_html(report: "DetectorTrainingReport", path: str | Path, *, curves: str | Path | None = None,
               title: str = "") -> Path:
    """Write ``report.html``: the run in one file that opens in any browser, light or dark. The curves (embedded,
    from ``curves`` or drawn here), what the run left and why it ended, AP per class at the best epoch and every epoch
    as tables, the dataset and the plan.

    Args:
        report (DetectorTrainingReport): What ``DetectorTraining.train()`` returned.
        path (str | Path): Where the page goes.
        curves (str | Path | None): A ``curves.png`` to embed; ``None`` draws one from the report (default: None).
        title (str): What the page is called; empty is the model folder's name (default: "").

    Returns:
        Path: Where the page went.
    """
    from string import Template  # noqa: PLC0415

    out = Path(path)
    name = title or Path(str(report.artifact.get("model_dir") or out.parent)).name
    image = _curves_png(report, out, curves, name)
    dataset = dict(report.raw.get("dataset") or {})
    best = dict(report.best or {})
    per_class = {str(k): float(v) for k, v in dict(best.get("per_class_ap") or {}).items()
                 if isinstance(v, (int, float))}
    seconds = sum(row.seconds for row in report.epochs if not math.isnan(row.seconds))
    speed = report.raw.get("train_images_per_second")
    if report.artifact.get("written"):
        model = html.escape(str(report.artifact.get("model_dir")))
    else:
        model = "not written: " + html.escape(str(report.artifact.get("reason", "no epoch finished")))
    facts = [
        _fact("Outcome", html.escape(str(report.outcome).replace("_", " "))),
        _fact("Model", model),
        _fact("Best epoch", (f"{report.best_epoch} of {len(report.epochs)} run, {report.plan.epochs} planned"
                             if report.best_epoch else "&ndash;")),
        _fact("mAP 0.5:0.95", _cell(_number(best.get("map")))),
        _fact("mAP 0.5 / 0.75", _cell(_number(best.get("map50"))) + " / " + _cell(_number(best.get("map75")))),
        _fact("Validation images", html.escape(str(dataset.get("val_images", "-")))),
        _fact("Training images", html.escape(str(dataset.get("train_images_used", dataset.get("train_images", "-"))))),
        _fact("Time", f"{seconds / 60.0:.1f} min" + (f", {speed} image(s)/s" if speed else "")),
        _fact("Device", html.escape(str(report.raw.get("device", "-")))),
    ]
    if image:
        picture = ('<img alt="Loss, validation mAP, learning rate and AP per class over the epochs" '
                   'src="data:image/png;base64,' + base64.b64encode(image).decode("ascii") + '">')
    else:
        picture = '<p class="muted">No curves: no epoch was drawn.</p>'
    if per_class:
        rows = "".join(_class_row(n, v) for n, v in sorted(per_class.items(), key=lambda item: -item[1]))
        classes_html = ('<div class="scroll"><table><tr><th>class</th><th>AP 0.5:0.95</th><th></th></tr>' + rows
                        + "</table></div>")
    else:
        classes_html = '<p class="muted">No validation box, so no AP per class is measured.</p>'
    classes = [str(c) for c in report.raw.get("classes", ())]
    notes = "".join("<li>" + html.escape(str(line)) + "</li>" for line in dataset.get("skipped", ()))
    plan = report.plan.as_dict() if hasattr(report.plan, "as_dict") else {}
    page = Template(_PAGE).substitute(
        title=html.escape(name),
        intro=html.escape(f"RT-DETR fine-tuned from {report.plan.base_model} on {dataset.get('root', '?')}, recipe "
                          f"{report.plan.recipe or 'none'}, tier {report.plan.tier or 'none'}."),
        facts="".join(facts),
        picture=picture,
        classes=classes_html,
        epochs="".join(_epoch_row(row, report.best_epoch) for row in report.epochs),
        dataset=html.escape(f"{str(dataset.get('format', '?')).upper()}, {len(classes)} class(es): "
                            + ", ".join(classes) + "."),
        notes=("<ul>" + notes + "</ul>") if notes else "",
        plan="".join("<tr><td>" + html.escape(str(k)) + "</td><td>" + html.escape(str(v)) + "</td></tr>"
                     for k, v in plan.items()),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
