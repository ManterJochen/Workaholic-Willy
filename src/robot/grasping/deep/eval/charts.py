"""Four panels that answer "is my model any good", from the two files the pipeline already writes.

A hold rate on its own is a number with no scale: it means nothing until the referee's own ceiling
is known, and it means something different again once some of those proposals turn out to sit on
geometry far wider than the width they ask for. `score-proposals` prints five lines and none of
them carries that scale.

The four, and the question each answers:

1. Hold rate against the label control. The only panel with a scale on it. The control is the same
   referee, same shard, judging the corpus labels instead of the model, so it is the ceiling a
   perfect model would reach, and that ceiling is not guaranteed to be 100 %. Without it a customer
   has no way to tell whether a hold rate is good.
2. Where the failures are. Aimed at nothing / gripped and slipped / refused before physics: three
   different defects with three different repairs, and a bare hold rate merges them. Proposals not
   on an object at all make the hold rate a statement about aim rather than about grip.
3. What the model asks for against what its own pose implies, both against the configured aperture.
   A refusal note read alone suggests a width head that asks for more than the hand has. The number
   in that note is not the request: it is the contact separation the referee derives by laying the
   proposed closing axis through the object and measuring the object there. Reading it as the
   request confuses a width head that overshoots with a width head that disagrees with its own
   pose. A grasp of the second kind never closes on the object at the width it predicted, and a
   panel showing only the request hides it.
4. Distinct proposals. How many survive a 20 mm / 20 degree radius. A collapsed head proposes one
   grasp many times, and a referee then judges one pose several times and reports it as that many
   trials.

ASCII in every label. The figure is written by matplotlib, but the text that goes with it is
printed to a cp1252 terminal, where a non-ASCII character raises `UnicodeEncodeError`.

A missing optional input never raises into a caller's face. The label control is optional: a
customer who has not run one still gets the other three panels, with the missing scale said in
words rather than left blank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["ChartInputs", "write_eval_charts"]

#: Past this a proposal is not on an object at all. Half the 2F-85's aperture: a grasp centre further
#: than that from every object point cannot have the object between its fingers.
_OFF_OBJECT_MM: float = 42.5
#: The radius the distinctness panel counts at, matching `propose`'s own diagnostic.
_DISTINCT_MM: float = 20.0
_DISTINCT_DEG: float = 20.0


class ChartInputs:
    """The three files a full picture needs, two of which are required.

    `control` is optional and its absence is shown, not hidden. A hold rate with no control is a
    number without a scale, and a panel that omits the reference line invites reading the rate as a
    fraction of 100.
    """

    def __init__(self, proposals: Path | str, verdicts: Path | str,
                 control: Path | str | None = None, aperture_mm: float = 85.0,
                 gripper: str = "2f85") -> None:
        self.proposals = Path(proposals)
        self.verdicts = Path(verdicts)
        self.control = Path(control) if control else None
        self.aperture_mm = float(aperture_mm)
        self.gripper = str(gripper)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("{")]


def _held_rate(rows: Sequence[dict[str, Any]]) -> tuple[int, int, int]:
    """`(held, judged, refused)` over generator rows.

    The denominator is the whole point. A refusal is a proposal physics never judged, so counting it
    as a failure and excluding it are two different numbers. Both are returned; the caller shows
    both.
    """
    generator = [r for r in rows if r.get("source") == "generator"]
    refused = [r for r in generator if str(r.get("note", "")).startswith("refused")]
    held = sum(1 for r in generator if r.get("held") is True)
    return held, len(generator) - len(refused), len(refused)


def _distinct(proposals: Sequence[dict[str, Any]]) -> float:
    """The share of proposals no higher-ranked one duplicates, at the diagnostic radius and angle.

    The angle is not optional. Two poses 5 mm apart approaching from opposite sides are two
    grasps, and a radius alone would merge them, flattering the number.
    """
    if not proposals:
        return 0.0
    limit = float(np.cos(np.radians(_DISTINCT_DEG)))
    kept: list[dict[str, Any]] = []
    for row in proposals:
        position = np.asarray(row["position_mm"], dtype=np.float64)
        approach = np.asarray(row["approach"], dtype=np.float64)
        duplicate = any(
            other["scene_id"] == row["scene_id"]
            and float(np.linalg.norm(position - np.asarray(other["position_mm"]))) <= _DISTINCT_MM
            and float(approach @ np.asarray(other["approach"])) >= limit
            for other in kept)
        if not duplicate:
            kept.append(row)
    return len(kept) / len(proposals)


def write_eval_charts(inputs: ChartInputs, out: Path | str) -> dict[str, Any]:
    """Draw the four panels and return the numbers behind them.

    Returns the numbers as well as writing the picture, so a caller can print them, assert on them,
    or put them in a report without re-deriving anything from the plot.
    """
    import matplotlib                                            # noqa: PLC0415 (heavy, optional)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                              # noqa: PLC0415

    proposals = _rows(inputs.proposals)
    verdicts = _rows(inputs.verdicts)
    held, judged, refused = _held_rate(verdicts)
    rate = held / judged if judged else 0.0
    strict = held / (judged + refused) if (judged + refused) else 0.0

    control_rate: float | None = None
    if inputs.control and inputs.control.is_file():
        rows = _rows(inputs.control)
        labels = [r for r in rows if r.get("source") != "generator" and "held" in r]
        if labels:
            control_rate = sum(1 for r in labels if r.get("held") is True) / len(labels)

    distances = np.asarray([float(r.get("object_distance_mm", np.nan)) for r in proposals])
    widths = np.asarray([float(r.get("width_mm", np.nan)) for r in proposals])
    # What the pose implies, recovered from the referee's own refusal text. The harness lays the
    # proposed closing axis through the object and measures the object there; when that exceeds the
    # modelled hand it refuses and says the number. There is no other record of it, so this parses
    # the sentence rather than inventing a second geometry pass.
    implied = np.asarray([
        float(str(row.get("note", "")).split("the jaw reads", 1)[1].split("mm", 1)[0])
        for row in verdicts
        if "the jaw reads" in str(row.get("note", ""))
    ]) if verdicts else np.zeros(0)
    off_object = int(np.sum(~np.isfinite(distances) | (distances > _OFF_OBJECT_MM)))
    slipped = max(0, judged - held - off_object)
    distinct = _distinct(proposals)
    over_aperture = int(np.sum(widths > inputs.aperture_mm))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(f"{inputs.proposals.stem}: {held} of {judged} judged grasps held "
                    f"({rate * 100:.1f} %), gripper {inputs.gripper}", fontsize=13)

    # --- 1. the only panel with a scale on it -------------------------------------------------
    top_left = axes[0][0]
    names, values, colours = ["model"], [rate * 100], ["#1f6feb"]
    if control_rate is not None:
        names.append("labels (ceiling)")
        values.append(control_rate * 100)
        colours.append("#8250df")
    top_left.bar(names, values, color=colours)
    for index, value in enumerate(values):
        top_left.text(index, value, f" {value:.1f} %", va="bottom", ha="center")
    top_left.set_ylabel("hold rate, percent")
    top_left.set_title("1. against the ceiling a perfect model would reach")
    if control_rate is None:
        # Said, not left blank. A single bar with no reference invites reading it out of 100.
        top_left.text(0.5, 0.5, "no label control given:\nthis bar has NO scale",
                      transform=top_left.transAxes, ha="center", va="center", color="#cf222e")
    top_left.set_ylim(0, max(values + [1.0]) * 1.35)

    # --- 2. three different defects, three different repairs ----------------------------------
    top_right = axes[0][1]
    parts = [("held", held, "#1a7f37"), ("aimed at nothing", off_object, "#cf222e"),
             ("gripped, slipped", slipped, "#bf8700"),
             (f"refused (> {inputs.aperture_mm:.0f} mm)", refused, "#6e7781")]
    top_right.barh([p[0] for p in parts], [p[1] for p in parts], color=[p[2] for p in parts])
    for index, (_name, count, _colour) in enumerate(parts):
        top_right.text(count, index, f" {count}", va="center")
    top_right.set_xlabel("proposals")
    top_right.set_title("2. where the failures are")
    top_right.invert_yaxis()

    # --- 3. the request against what the pose implies, and they disagree ----------------------
    bottom_left = axes[1][0]
    finite = widths[np.isfinite(widths)]
    if finite.size:
        bottom_left.hist(finite, bins=30, color="#1f6feb", alpha=0.85,
                         label="what the model ASKS FOR")
    if implied.size:
        # The second histogram is the point of the panel: the gap between the two distributions is
        # the defect. Drawing only the requests reports zero over the aperture while proposals are
        # being refused for exactly that.
        bottom_left.hist(implied, bins=20, color="#cf222e", alpha=0.55,
                         label="what its POSE implies (refused only)")
    bottom_left.axvline(inputs.aperture_mm, color="#24292f", lw=2, ls="--",
                        label=f"{inputs.gripper} aperture {inputs.aperture_mm:.0f} mm")
    bottom_left.set_xlabel("jaw opening, mm")
    bottom_left.set_ylabel("proposals")
    if implied.size:
        bottom_left.set_title(
            f"3. {over_aperture} request over the aperture, but {implied.size} POSES imply "
            f"{implied.min():.0f} to {implied.max():.0f} mm")
    else:
        bottom_left.set_title(f"3. {over_aperture} proposal(s) ask for more than this hand has")
    bottom_left.legend(loc="upper right", fontsize=7)

    # --- 4. a collapsed head judged many times looks like many trials --------------------------
    bottom_right = axes[1][1]
    bottom_right.bar(["distinct", "duplicates"],
                     [distinct * 100, (1.0 - distinct) * 100],
                     color=["#1a7f37", "#6e7781"])
    bottom_right.text(0, distinct * 100, f" {distinct * 100:.1f} %", va="bottom", ha="center")
    bottom_right.set_ylabel("percent of proposals")
    bottom_right.set_ylim(0, 108)
    bottom_right.set_title(f"4. distinct at {_DISTINCT_MM:.0f} mm / {_DISTINCT_DEG:.0f} deg")

    figure.tight_layout()
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=140)
    plt.close(figure)

    return {
        "held": held, "judged": judged, "refused": refused,
        "hold_rate": round(rate, 5), "hold_rate_counting_refusals": round(strict, 5),
        "control_hold_rate": None if control_rate is None else round(control_rate, 5),
        "off_object": off_object, "slipped": slipped,
        "over_aperture": over_aperture, "aperture_mm": inputs.aperture_mm,
        "implied_separation_over_aperture": int(implied.size),
        "implied_separation_max_mm": None if not implied.size else round(float(implied.max()), 1),
        "distinct_share": round(distinct, 5),
        "chart": str(target),
    }


def summarise_charts(numbers: dict[str, Any]) -> list[str]:
    """The same facts as ASCII lines, for the terminal beside the picture."""
    lines = [
        f"  held {numbers['held']} of {numbers['judged']} judged"
        f"   = {numbers['hold_rate'] * 100:.1f} %"
        f"   ({numbers['hold_rate_counting_refusals'] * 100:.1f} % counting the "
        f"{numbers['refused']} refusals as failures)",
    ]
    control = numbers.get("control_hold_rate")
    if control is None:
        lines.append("  NO LABEL CONTROL: the rate above has no scale. Run "
                     "`datagen physics-sample --per-class 300` on the same shard to get one")
    else:
        share = numbers["hold_rate"] / control if control else 0.0
        lines.append(f"  the label control on the same shard held {control * 100:.1f} %, "
                     f"so the model reaches {share * 100:.0f} % of what the harness manages "
                     f"with perfect labels")
    lines.append(f"  aimed at nothing {numbers['off_object']}   gripped and slipped "
                 f"{numbers['slipped']}   requested width over the "
                 f"{numbers['aperture_mm']:.0f} mm aperture {numbers['over_aperture']}")
    implied_count = numbers.get("implied_separation_over_aperture") or 0
    if implied_count:
        lines.append(
            f"  but {implied_count} proposal(s) sit on geometry up to "
            f"{numbers['implied_separation_max_mm']:.0f} mm wide while asking for far less: the "
            f"width head and the pose head disagree, which is a different defect from asking for "
            f"too much")
    lines.append(f"  distinct at 20 mm / 20 deg: {numbers['distinct_share'] * 100:.1f} % "
                 f"(a low number means one grasp was judged several times)")
    lines.append(f"  -> {numbers['chart']}")
    return lines
