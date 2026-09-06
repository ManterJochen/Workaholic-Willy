"""What a training run actually did: the curve, the lift over the floor, and whether it had stopped.

    python -m src.robot.grasping.deep report --run logs/dl/models/arm_film

The question this answers is "was it still learning when it stopped?", and it is the only question
that decides what to run next. A final number cannot answer it: one value is a good result or a
wasted day depending entirely on whether the last epochs were still climbing. Early arms were read
as "architecture X does not help" while every one of them was in total mode collapse, sitting
exactly on its own floor, and more epochs were what broke the collapse. A run without a curve is a
run that can be misread the same way.

Every rate is read against its measured floor, never against zero. The trainer logs the floor for
each split beside each metric, and this module refuses to report a hit rate without one: a floor is
what separates "the net learned something" from "the majority class is most of the data". The
`--- 0.0000 ---` that a missing floor produces is a number that looks like an answer.

The segment seam. A resumed run is one experiment split across wall-clock gaps, so wall-clock time
is not a valid x-axis and epochs are; the report states the segments explicitly rather than drawing
a curve with a flat spot in it that means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np

__all__ = ["Epoch", "RunReport", "build_report", "format_report", "write_curve"]

#: Epochs per comparison window. The last window is compared against the one before it.
#:
#: Windowed means, not a slope through the tail. A ramp estimator straddling a step reports the flat
#: parts on either side of it: a least-squares slope over the final epochs can come out negative
#: while the run is gaining, because the gain arrives as a step, and the run reads as finished with
#: a real improvement still landing.
#:
#: 15 rather than 5: epoch-to-epoch noise leaves a five-epoch mean with a standard error of the
#: same order as the effect being looked for, and a window of 15 brings it well below that.
_WINDOW_EPOCHS: Final[int] = 15

#: How many standard errors the window-over-window difference must clear to count as real. Two, the
#: ordinary bar; the difference must also clear `_REAL_DIFFERENCE`, because a tiny gain measured
#: precisely is still a tiny gain.
_SIGMA: Final[float] = 2.0

#: Measured convention of this arc: two arms differing by less than 0.02 in the final metric are not
#: distinguishable. An improvement that would not accumulate to this over another window is not a
#: reason to keep paying for epochs.
_REAL_DIFFERENCE: Final[float] = 0.02


@dataclass(frozen=True, slots=True)
class Epoch:
    stamp: datetime
    fold: int
    epoch: int
    total: int
    loss: float
    test_hit: float
    test_floor: float
    train_hit: float
    train_floor: float
    held_offset_error_mm: float

    @property
    def test_lift(self) -> float:
        """How far above its own floor the test metric sits. The number, not `test_hit`."""
        return self.test_hit - self.test_floor

    @property
    def train_lift(self) -> float:
        return self.train_hit - self.train_floor


@dataclass(frozen=True, slots=True)
class RunReport:
    name: str
    epochs: tuple[Epoch, ...]
    segments: tuple[tuple[int, int], ...]
    compute_hours: float
    best: Epoch
    final: Epoch
    #: `(previous_window_mean, final_window_mean, standard_error_of_the_difference)`.
    window_comparison: tuple[float, float, float]
    still_improving: bool
    #: Whether the window verdicts mean anything at all. Below two full windows there is nothing to
    #: compare, so no verdict is given. A placeholder `(mean, mean, 0.0)` with `still_improving =
    #: False` makes the formatter print "it had flattened... more epochs are not the lever" on a run
    #: four epochs into a thirty-six-epoch schedule, and a placeholder that reads as a finding is
    #: worse than a gap.
    comparable: bool
    #: The window actually used, which is `min(_WINDOW_EPOCHS, len // 2)` and not `_WINDOW_EPOCHS`.
    #: A verdict line naming 15 while a 7-epoch run compares 3 against 3 describes an experiment
    #: that was not run. A number in a report has to be the one that was computed.
    window: int
    #: Whether the curve is bigger than the band it is judged against. `_REAL_DIFFERENCE` is an
    #: absolute 0.02, so a run whose lifts are all smaller than that falls inside the band by
    #: arithmetic and "plateaued at epoch 1, book 1 epoch next time" is guaranteed rather than
    #: observed. When the whole curve spans less than the band, the honest answer is that the signal
    #: is below the resolution of the question.
    resolvable: bool
    #: The first epoch from which the curve never again left `_REAL_DIFFERENCE` of its final level.
    #: `None` when it never settled, which is the interesting answer, not a missing one.
    plateau_epoch: int | None
    collapsed: bool
    generalisation_gap: float


def parse_run_directory(directory: Path | str) -> tuple[list[Epoch], list[tuple[int, int]]]:
    """`Epoch` rows, read from a set run's `epochs.json` rather than scraped from a log.

    A regex is not the right reader. `train.trainer` writes `epochs.json` after every epoch, with
    the columns already parsed. Scraping a human-readable line for numbers that exist as JSON one
    directory over is how a report comes to disagree with the run it describes.

    The floor comes from `report.json`, per fold. `test_lift` is the number this module exists to
    compute, and it is meaningless against the wrong population: the corpus-wide floor is measured on
    units the held-out columns never saw. `held_floor` is used when present, and the run is refused
    rather than silently scored against the corpus floor when it is not.
    """
    import json  # noqa: PLC0415

    root = Path(directory)
    rows_path, report_path = root / "epochs.json", root / "report.json"
    if not rows_path.is_file():
        raise ValueError(
            f"{rows_path} does not exist; --run takes the --out directory of a train-set run")

    # `epochs.json` is the whole report, and it is rewritten every epoch, so a running arm is
    # already fully readable: the floors, the ceilings and every row so far are in it. `report.json`
    # lands only when the fold finishes; requiring it first would refuse a live run over a file that
    # is not needed.
    card = json.loads(rows_path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        # A bare list is an older layout. It carries the rows and not the floors, so the run is
        # readable only once `report.json` exists.
        card = {"epochs": card}
    if "held_floor" not in card and report_path.is_file():
        card = {**json.loads(report_path.read_text(encoding="utf-8")), "epochs": card["epochs"]}
    rows = card.get("epochs") or []
    if not rows:
        raise ValueError(f"{rows_path} holds no epoch")

    # The floor comes per fold, from the population the held columns were measured on.
    floors: dict[int, float] = {}
    for entry in card.get("held_floor", []):
        # `top_down` is the bar: a fixed constant can beat a trained head, and a head that does not
        # clearly beat "point the gripper down" has not beaten anything.
        floors[int(entry["fold"])] = float(entry["top_down"]["top1_hit"])
    if not floors:
        raise ValueError(
            f"{rows_path} carries no `held_floor`, so a lift cannot be computed against the "
            f"population the held columns were measured on. That block landed on 2026-09-04; a run "
            f"started before it has to be re-run, or read the raw columns in epochs.json.")

    # A synthetic clock, accumulated from the recorded `seconds`. `compute_hours` must be the time
    # the run spent, and wall-clock stamps would fold in the gaps between sittings.
    base = datetime(2000, 1, 1)
    elapsed = 0.0
    epochs: list[Epoch] = []
    breaks: list[int] = []
    previous = None
    for row in rows:
        fold = int(row.get("fold", 0))
        if previous is not None and fold != previous:
            breaks.append(len(epochs))
        previous = fold
        elapsed += float(row.get("seconds", 0.0))
        floor = floors.get(fold, 0.0)
        epochs.append(Epoch(
            stamp=base + timedelta(seconds=elapsed),
            fold=fold,
            epoch=int(row["epoch"]) + 1,
            # 0 means unknown, and it is printed as "so far" rather than as a completed run. A
            # schedule length inferred from the rows would make every live run read as finished.
            total=int(card.get("plan_epochs", row.get("total", 0))),
            loss=float(row.get("held_total", float("nan"))),
            test_hit=float(row.get("held_top1_hit", 0.0)),
            test_floor=floor,
            train_hit=float(row.get("train_top1_hit", 0.0)),
            train_floor=floor,
            held_offset_error_mm=float(row.get("held_offset_error_mm", float("nan"))),
        ))
    segments = _segments(len(epochs), breaks)
    return epochs, segments


def _segments(count: int, breaks: list[int]) -> list[tuple[int, int]]:
    """`(first, last)` per sitting, from the indices where the fold changed."""
    bounds = [0, *breaks, count]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def _compute_hours(epochs: list[Epoch], segments: list[tuple[int, int]]) -> float:
    """Wall clock spent computing, with the gaps between sittings removed.

    Summing `last - first` over the whole run counts the hours the machine was asleep, which is how
    a run gets reported as longer than it was and every per-epoch figure derived from it is wrong.
    """
    by_epoch = {e.epoch: e for e in epochs}
    total = 0.0
    for first, last in segments:
        if first in by_epoch and last in by_epoch and last > first:
            total += (by_epoch[last].stamp - by_epoch[first].stamp).total_seconds()
    return total / 3600.0


def _plateau_epoch(epochs: list[Epoch], lifts: "np.ndarray") -> int | None:
    """The first epoch from which the smoothed curve stayed within `_REAL_DIFFERENCE` of the end.

    The only number here that changes what the next run costs. A final metric does not say whether
    to book the same schedule again; "everything after this epoch was noise" does, and the epochs
    after that one are hours that could have run another architecture.

    Smoothed with the same window the improvement check uses, because a single epoch dipping below
    the band is noise and would otherwise reset the answer to the very end of the run every time.
    """
    window = min(_WINDOW_EPOCHS, max(3, len(lifts) // 4))
    if len(lifts) < 2 * window:
        return None
    kernel = np.ones(window) / float(window)
    smooth = np.convolve(lifts, kernel, mode="valid")
    final_level = float(smooth[-1])
    # Walk backwards to the last point outside the band; the plateau starts after it. Walking forward
    # would stop at the first point that happens to fall inside the band on its way up.
    outside = np.flatnonzero(np.abs(smooth - final_level) > _REAL_DIFFERENCE)
    if outside.size == 0:
        return int(epochs[0].epoch)
    start = int(outside[-1]) + 1
    # A plateau that begins inside the last window is not a plateau, it is the end of the data. A
    # steadily climbing run always has its final few epochs within the band of its own last value,
    # and without this a run that never settled reports "plateaued at epoch 76 of 80", which reads
    # as "it was done" about the one case where it certainly was not.
    if start + window >= len(smooth):
        return None
    return int(epochs[start + window - 1].epoch)


def build_report(name: str, path: Path | str) -> RunReport:
    """Read a set run's output directory and turn it into the report.

    One reader, one analysis: `parse_run_directory` reads the run's `epochs.json` and
    `_build_from_epochs` turns those rows into the report.
    """
    epochs, segments = parse_run_directory(path)
    if not epochs:
        raise ValueError(f"no epoch rows in {path}: `epochs.json` is empty or carries none")
    return _build_from_epochs(name, epochs, segments)


def _build_from_epochs(name: str, epochs: list[Epoch],
                       segments: list[tuple[int, int]]) -> RunReport:
    """The analysis, over rows that are already parsed. `parse_run_directory` produces the rows;
    this turns them into the report."""

    lifts = np.array([e.test_lift for e in epochs], dtype=np.float64)
    # The single-epoch best is noise-prone and is reported as such: a best epoch sits above the
    # window average around it. A maximum over many noisy draws is a maximum over many noisy draws,
    # not a level the model ever held.
    best = epochs[int(np.argmax(lifts))]
    final = epochs[-1]

    window = min(_WINDOW_EPOCHS, len(lifts) // 2)
    if window >= 3:
        tail, prior = lifts[-window:], lifts[-2 * window:-window]
        difference = float(tail.mean() - prior.mean())
        error = float(np.sqrt(tail.var(ddof=1) / window + prior.var(ddof=1) / window))
        comparison = (float(prior.mean()), float(tail.mean()), error)
        improving = difference > _SIGMA * error and difference > _REAL_DIFFERENCE / 2.0
        comparable = True
    else:
        comparison = (float(lifts.mean()), float(lifts.mean()), 0.0)
        improving = False
        comparable = False
    resolvable = float(lifts.max() - lifts.min()) >= _REAL_DIFFERENCE

    return RunReport(
        plateau_epoch=_plateau_epoch(epochs, lifts),
        name=name, epochs=tuple(epochs), segments=tuple(segments),
        compute_hours=_compute_hours(epochs, segments), best=best, final=final,
        window_comparison=comparison, still_improving=improving, comparable=comparable,
        window=window, resolvable=resolvable,
        # The collapse check. A run whose final metric is within noise of its floor learned nothing,
        # whatever its loss curve did, and no conclusion may be drawn from it: earlier arms sat
        # exactly on their own floor to ten decimals.
        collapsed=abs(final.test_lift) < 1e-6,
        generalisation_gap=final.train_lift - final.test_lift)


def write_curve(report: RunReport, path: Path | str) -> Path:
    """The learning curve as a PNG. Epochs on x, never wall clock; see the module docstring."""
    import matplotlib                                          # noqa: PLC0415 (heavy, and optional)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                            # noqa: PLC0415

    epochs = [e.epoch for e in report.epochs]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 1]})

    top.plot(epochs, [e.test_hit for e in report.epochs], color="#1f6feb", lw=1.6, label="test hit")
    top.plot(epochs, [e.train_hit for e in report.epochs], color="#8250df", lw=1.2, alpha=0.75,
             label="train hit")
    # The floors, drawn rather than described. A reader who sees the curve without them reads the
    # test line as rising from zero, which is the misreading this whole module exists to prevent.
    top.axhline(report.final.test_floor, color="#1f6feb", ls="--", lw=1.0, alpha=0.6,
                label=f"test floor {report.final.test_floor:.4f}")
    top.axhline(report.final.train_floor, color="#8250df", ls="--", lw=1.0, alpha=0.6,
                label=f"train floor {report.final.train_floor:.4f}")
    top.axvline(report.best.epoch, color="#1a7f37", ls=":", lw=1.2,
                label=f"best epoch {report.best.epoch}")
    for first, _last in report.segments[1:]:
        top.axvline(first, color="#57606a", ls="-", lw=0.8, alpha=0.4)
    top.set_ylabel("hit rate")
    top.set_title(f"{report.name}: test lift {report.final.test_lift:+.4f} over floor "
                  f"({report.final.epoch}/{report.final.total} epochs, "
                  f"{report.compute_hours:.1f} h compute)")
    top.legend(loc="lower right", fontsize=8)
    top.grid(alpha=0.25)

    bottom.plot(epochs, [e.loss for e in report.epochs], color="#cf222e", lw=1.3)
    bottom.set_ylabel("loss")
    bottom.set_xlabel("epoch")
    bottom.grid(alpha=0.25)

    figure.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    plt.close(figure)
    return out


def format_report(report: RunReport) -> str:
    lines = [
        f"{report.name}: {len(report.epochs)} epoch(s)"
        + (f" of {report.final.total}" if report.final.total else " so far")
        + (" [RUNNING]" if report.final.total and len(report.epochs) < report.final.total else "")
        + f", {report.compute_hours:.1f} h of compute across {len(report.segments)} sitting(s)",
        "",
        f"  {'':16} {'value':>9} {'floor':>9} {'LIFT':>9}",
        f"  {'final test':16} {report.final.test_hit:9.4f} {report.final.test_floor:9.4f} "
        f"{report.final.test_lift:+9.4f}",
        f"  {'final train':16} {report.final.train_hit:9.4f} {report.final.train_floor:9.4f} "
        f"{report.final.train_lift:+9.4f}",
        f"  {'best test':16} {report.best.test_hit:9.4f} {report.best.test_floor:9.4f} "
        f"{report.best.test_lift:+9.4f}   at epoch {report.best.epoch}",
        "",
        f"  generalisation gap  {report.generalisation_gap:+.4f}  (train lift minus test lift)",
        f"  final offset error  {report.final.held_offset_error_mm:.2f} mm",
        f"  best single epoch   {report.best.test_lift:+.4f} at epoch {report.best.epoch} "
        f"-- a maximum over {len(report.epochs)} noisy draws, NOT a level the model held",
    ]
    # The window line prints only when there are two windows. Below that, `window_comparison` is the
    # mean compared against itself, and printing it gives a line that looks like a converged run and
    # is a placeholder.
    if report.comparable:
        lines.append(
            f"  last {report.window} epochs vs the {report.window} before: "
            f"{report.window_comparison[0]:+.4f} -> {report.window_comparison[1]:+.4f}  "
            f"(difference {report.window_comparison[1] - report.window_comparison[0]:+.4f} "
            f"+- {report.window_comparison[2]:.4f})")

    lines.append("")
    # ASCII in every printed line. This report is read on a Windows console at cp1252, where a
    # `print` of a non-ASCII marker raises UnicodeEncodeError and takes the whole command down. The
    # rule covers the report body, not only the CLI help text.
    if not report.comparable:
        # No verdict at all below two full windows. The alternative is a placeholder comparison of
        # the mean against itself, which prints "it had flattened ... more epochs are not the lever"
        # on a run four epochs into a thirty-six-epoch schedule. A report that answers a question it
        # cannot answer is worse than one that declines.
        lines.append(f"  [TOO EARLY] {len(report.epochs)} epoch(s) is not enough for a "
                     f"window-over-window verdict, so none is given. The lift above is real; whether "
                     f"the curve is still climbing needs at least six epochs and reads best at "
                     f"{2 * _WINDOW_EPOCHS}.")
        if report.collapsed:
            lines.append("  [ON THE FLOOR SO FAR] the final test metric sits on its own floor. Early "
                         "in a run that is ordinary rather than fatal; late it is a collapse that "
                         "voids every reading of the run.")
    elif report.collapsed:
        lines.append("  [COLLAPSED] the final test metric sits on its own floor. This run learned "
                     "nothing, and no conclusion about the architecture, target or corpus may be "
                     "drawn from it.")
    elif report.still_improving:
        gain = report.window_comparison[1] - report.window_comparison[0]
        lines.append(f"  [STILL IMPROVING] when it stopped: the final {report.window} epochs beat "
                     f"the {report.window} before them by {gain:+.4f}, which is "
                     f"{gain / max(report.window_comparison[2], 1e-9):.1f} standard errors. The "
                     f"schedule is the binding constraint here, not the architecture, and any arm "
                     f"compared against this one at fewer epochs is a different experiment rather "
                     f"than a worse architecture.")
    else:
        gain = report.window_comparison[1] - report.window_comparison[0]
        error = report.window_comparison[2]
        if report.resolvable:
            lines.append(
                f"  [FLATTENED] before it stopped: the final {report.window} epochs beat the "
                f"{report.window} before them by only {gain:+.4f} (+-{error:.4f}), which does not "
                f"reach the {_REAL_DIFFERENCE} this arc treats as a real difference. More epochs are "
                f"not the lever; resolution, the target and the corpus are the open ones.")
        else:
            # The same band problem as the plateau line. A curve spanning less than
            # `_REAL_DIFFERENCE` cannot reach it by construction, so "it does not reach 0.02" is
            # arithmetic and "more epochs are not the lever" is advice drawn from arithmetic. Against
            # its own standard error the comparison still says something, and that is all it says.
            verdict = ("not distinguishable from zero" if abs(gain) < _SIGMA * max(error, 1e-9)
                       else "larger than its own noise")
            lines.append(
                f"  [NO WINDOW VERDICT] the final {report.window} epochs differ from the "
                f"{report.window} before them by {gain:+.4f} (+-{error:.4f}), {verdict}. The "
                f"{_REAL_DIFFERENCE} threshold is not applied: the whole curve is smaller than it, so "
                f"nothing here could reach it and no claim about the schedule follows.")

    if report.generalisation_gap > 0.05:
        lines.append(f"  [WARNING] the train lift exceeds the test lift by {report.generalisation_gap:.4f}. "
                     f"The net is fitting the training assets rather than the task; more capacity "
                     f"or more epochs would widen this, not close it.")

    if not report.resolvable:
        # The band is bigger than the whole curve. `_REAL_DIFFERENCE` is an absolute 0.02 and this
        # arc's lifts sit around 0.006, so "every epoch stayed within the band" is arithmetic rather
        # than an observation, and "plateaued at epoch 1, book 1 epoch next time" would be advice
        # drawn from it. The run is below the resolution of the question this verdict asks.
        span = max(e.test_lift for e in report.epochs) - min(e.test_lift for e in report.epochs)
        lines.append(
            f"  [BELOW RESOLUTION] the whole curve spans {span:.4f}, less than the "
            f"{_REAL_DIFFERENCE} this arc treats as a real difference, so NO plateau verdict is "
            f"given: every epoch sits inside the band by arithmetic. Read the lift and its floor "
            f"above instead.")
    elif report.plateau_epoch is not None and not report.collapsed:
        wasted = report.final.epoch - report.plateau_epoch
        share = wasted / max(report.final.epoch, 1)
        lines.append(
            f"  [PLATEAU] at epoch {report.plateau_epoch}: every epoch after that stayed "
            f"within {_REAL_DIFFERENCE} of the final level. That is {wasted} epoch(s), {share:.0%} "
            f"of the run and about {report.compute_hours * share:.1f} h of this machine, and what "
            f"they bought was smaller than the {_REAL_DIFFERENCE} this arc treats as a real "
            f"difference, which is not the same as nothing and is exactly why it is phrased as a "
            f"threshold. Book {report.plateau_epoch} + a margin next time and spend the rest on "
            f"another arm.")
    elif report.plateau_epoch is None and not report.collapsed:
        lines.append("  it never settled: the curve was still outside the band at the end, so the "
                     "schedule, not the architecture, is what this run measured.")

    if len(report.segments) > 1:
        spans = ", ".join(f"{first}-{last}" for first, last in report.segments)
        lines.append(f"  resumed {len(report.segments) - 1}x (epochs {spans}): the compute hours "
                     f"above exclude the gaps between sittings.")
    return "\n".join(lines)
