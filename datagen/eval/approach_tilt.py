"""Coverage by approach tilt: the metric that makes the side-grasp population measurable.

The ladder's single `top1` cannot tell "found a side grasp for a short object" from "found a top
grasp for a tall one", so a generator could solve the whole off-vertical population and the headline
number would barely move. This report splits coverage by how far off vertical an object has to be
approached, and reports the object-views the reference cannot grasp at all beside it.

The bands are cut at the labelled distribution's own landmarks rather than at a round number that
sounds right. A single threshold placed by intuition puts most of the corpus on one side of it and
measures almost nothing; a curve over bands shows where each generator actually lives.

The band belongs to the object, not to the candidate. An object's band is the minimum tilt among its
admissible labels, the most top-down it can be grasped. Bucketing candidates instead would measure
what the generator likes to propose, which is the thing under test, and every generator would score
well on the bands it happens to prefer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["BANDS", "BandStats", "NO_LABEL", "band_of", "object_reach_tilt",
           "side_approach_report", "format_report"]

#: The band for an object-view the reference could not grasp at all.
#:
#: Reported, never scored. There is nothing for a generator to be right about here, because no label
#: means no target, but leaving these object-views out lets a report over the labelled ones alone
#: read as if it covered everything. The count is what keeps every rate below it honest about its
#: denominator.
NO_LABEL: Final[str] = "no_jaw_label"

#: Tilt bands, in degrees from straight down, half-open except the last.
#:
#: Cut at the labelled distribution's own landmarks rather than at round numbers somebody liked: 30
#: is well inside the top-down cone, 60 is the median admissible tilt, 90 is its p95 and the point
#: past which the gripper is coming from below the object's equator, and 135 is
#: `APPROACH_MAX_TILT_DEG`, beyond which the reference refuses rather than bins.
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("top_down", 0.0, 30.0),
    ("tilted", 30.0, 60.0),
    ("side", 60.0, 90.0),
    ("under", 90.0, 135.0),
)


def band_of(tilt_deg: float) -> str:
    """Which band a tilt falls in. Out-of-range tilts get their own name rather than a nearest band."""
    if not math.isfinite(tilt_deg) or tilt_deg < 0.0:
        return "unknown"
    for name, low, high in BANDS:
        if low <= tilt_deg < high:
            return name
    return "beyond_max" if tilt_deg >= BANDS[-1][2] else "unknown"


def object_reach_tilt(label_rows: Sequence[dict]) -> dict[tuple[str, int], float]:
    """`(scene_id, instance_id) -> the minimum jaw-label tilt`, i.e. its most top-down grasp.

    Minimum, not mean. The question a band answers is "can this object be reached from above at
    all", and one admissible top-down label is enough to say yes however many side labels exist
    beside it. A mean would put a cylinder with one top grasp and twenty side grasps in the side
    band and then credit a generator for finding the easy one.

    Objects with no jaw label at all are absent from the result rather than present with infinity:
    they are not in any coverage denominator, because there is nothing to cover.
    """
    best: dict[tuple[str, int], float] = {}
    for row in label_rows:
        if row.get("kind") != "jaw":
            continue
        tilt = row.get("approach_tilt_deg")
        if tilt is None:
            continue
        key = (str(row.get("scene_id", "")), int(row.get("instance_id", -1)))
        value = float(tilt)
        if math.isfinite(value) and (key not in best or value < best[key]):
            best[key] = value
    return best


@dataclass(frozen=True, slots=True)
class BandStats:
    """One configuration's performance on one band of objects."""

    band: str
    #: Objects whose most top-down admissible grasp falls in this band, per (scene, view, instance).
    objects: int
    #: Objects the generator produced at least one candidate for.
    with_candidate: int
    #: Objects it produced at least one valid candidate for.
    with_valid: int
    #: Objects whose best-ranked candidate is valid.
    top1: int

    @property
    def coverage(self) -> float:
        return self.with_valid / self.objects if self.objects else 0.0

    @property
    def top1_rate(self) -> float:
        return self.top1 / self.objects if self.objects else 0.0

    @property
    def proposal_rate(self) -> float:
        """How often it proposed anything: the denominator that separates "cannot see it" from
        "sees it and gets it wrong". Two very different failures with one `top1`."""
        return self.with_candidate / self.objects if self.objects else 0.0


def side_approach_report(eval_rows: Sequence[dict], reach: dict[tuple[str, int], float],
                         ) -> dict[str, list[BandStats]]:
    """`config -> [BandStats]`, one entry per band, in `BANDS` order.

    `eval_rows` are the ladder's own jaw rows; `reach` comes from `object_reach_tilt`. The unit is a
    (scene, view, instance) triple, the same unit the ladder's coverage uses, so the two numbers can
    be read against each other rather than being two different populations wearing one word.
    """
    # `config -> band -> unit -> [best_rank_valid, any_candidate, any_valid]`
    seen: dict[str, dict[str, dict[tuple[str, str, int], list[Any]]]] = {}
    for row in eval_rows:
        if row.get("kind") != "jaw":
            continue
        key = (str(row.get("scene_id", "")), int(row.get("instance_id", -1)))
        tilt = reach.get(key)
        # Counted, not skipped. An object the reference cannot grasp is not scored, because there
        # is no target to hit, but dropping it silently is how a report over a third of the
        # population comes to read as a report over all of it.
        band = NO_LABEL if tilt is None else band_of(tilt)
        config = str(row.get("config", "?"))
        unit = (key[0], str(row.get("view", "")), key[1])
        bucket = seen.setdefault(config, {}).setdefault(band, {})
        state = bucket.setdefault(unit, [None, False, False])
        if row.get("outcome") != "candidate":
            continue
        state[1] = True
        valid = bool(row.get("valid"))
        state[2] = state[2] or valid
        rank = row.get("rank")
        if rank is not None and (state[0] is None or int(rank) < int(state[0][0])):
            state[0] = (int(rank), valid)

    report: dict[str, list[BandStats]] = {}
    for config, bands in sorted(seen.items()):
        rows: list[BandStats] = []
        for name in [band[0] for band in BANDS] + [NO_LABEL]:
            units = bands.get(name, {})
            rows.append(BandStats(
                band=name,
                objects=len(units),
                with_candidate=sum(1 for state in units.values() if state[1]),
                with_valid=sum(1 for state in units.values() if state[2]),
                top1=sum(1 for state in units.values() if state[0] is not None and state[0][1]),
            ))
        report[config] = rows
    return report


def format_report(report: dict[str, list[BandStats]]) -> str:
    lines = [
        "coverage by the object's most top-down admissible grasp",
        "  bands: " + ", ".join(f"{n} {int(a)}-{int(b)}deg" for n, a, b in BANDS),
        "",
    ]
    for config, bands in report.items():
        total = sum(band.objects for band in bands)
        unreachable = next((b.objects for b in bands if b.band == NO_LABEL), 0)
        scored = total - unreachable
        lines.append(
            f"  === {config} ===   {total} object-view(s); {scored} scoreable "
            f"({scored / total:.1%}), {unreachable} with NO admissible jaw grasp at all")
        lines.append(f"    {'band':10} {'objects':>8} {'proposed':>9} {'covered':>9} {'top-1':>8}")
        for band in bands:
            if band.band == NO_LABEL:
                # Its proposal rate is still worth seeing: a generator that proposes confidently
                # for objects the reference calls ungraspable is saying something about itself.
                lines.append(f"    {band.band:10} {band.objects:8d} {band.proposal_rate:8.1%} "
                             f"{'n/a':>9} {'n/a':>8}")
                continue
            if not band.objects:
                # Printed anyway, at zero. An absent row reads as "not measured"; a zero row reads
                # as "measured, nothing there", and for the side bands that difference is the
                # finding.
                lines.append(f"    {band.band:10} {0:8d} {'-':>9} {'-':>9} {'-':>8}")
                continue
            lines.append(f"    {band.band:10} {band.objects:8d} {band.proposal_rate:8.1%} "
                         f"{band.coverage:8.1%} {band.top1_rate:7.1%}")
    lines.append(
        "\n  THE BAND IS THE OBJECT'S, not the candidate's: the MINIMUM tilt among its admissible "
        "labels, i.e. the most top-down it can be grasped. Bucketing candidates would measure what a "
        "generator likes to propose, which is the thing under test.")
    return "\n".join(lines)
