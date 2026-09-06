"""What does the approach target look like to a rule that does no learning at all?

The floor an approach head is read against is "one fixed bin everywhere", and it is a weak one: it
uses no per-point information, so beating it should be the easiest thing the head ever does. This
module asks the two questions that floor cannot answer, and it asks them without training anything.

Is there a one-line geometric rule that already works? `normal` and `normal_inward` predict, per
point, the bin nearest to that point's own surface normal. The normal is one of the five channels the
network already receives, so if a rule reading that single channel beats a trained network reading all
of them, the network is failing to extract something that needs no learning at all. That is a problem
in the training rather than in the corpus.

How much of the target is a per-object constant? `object_oracle` gives each object the single best
bin for that object, chosen with full knowledge of its own labels. It is not a baseline and must never
be quoted as one. It is a ceiling, and it answers a structural question: within an object the
admissible directions are nearly unanimous, and between objects they differ. If a per-object
constant scores far above what a per-point network achieves, then the head is predicting at the
wrong granularity, and no amount of resolution, capacity, data or loss tuning addresses that.

Every rule is marked honest or oracle and the table prints the mark. An oracle reads the answers of
the very points it is scored on. Mixing the two columns without the label is how a ceiling becomes a
reported result.

The normal's sign convention is measured, not assumed. A surface normal may point out of the
object or into it, and nothing in this repository's corpus format states which. Guessing costs a
factor of two on the answer and looks exactly like "the normal carries no signal", so both signs are
computed and both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

__all__ = [
    "GeometricRule",
    "ObjectDirections",
    "format_directions",
    "format_geometric",
    "object_direction_report",
    "run_geometric_rules",
]


@dataclass(frozen=True, slots=True)
class GeometricRule:
    """One training-free rule, scored on the held-out units with `approach_hit`.

    `approach_hit` is the bin-admissibility metric the retired binned arms were read against, so a
    saved result from one of them can be printed beside this table with `--reference`. The trainer
    in this tree scores `top1_hit` instead, on a different population, and is not comparable.
    """

    name: str
    #: Whether the rule reads the labels of the points it is scored on. See the module docstring.
    oracle: bool
    seed: int
    hit: float
    #: Supervised points it was scored over, so a rule is never read off a handful of rows.
    supervised_points: int
    #: Mean admissible bins per supervised point, out of the head's bin count.
    admissible_bins: float
    #: One line on what the rule actually is, carried into the report so the table explains itself.
    what: str


@dataclass(frozen=True, slots=True)
class ObjectDirections:
    """Where each object's single best approach direction actually points.

    The follow-up to `object_oracle`, and it decides whether a per-object head is worth building.
    One direction per object covering most of that object's supervised points says the signal is
    per-object; it does not say whether inferring that direction is easy or hard. If almost every
    object wants something within a few degrees of straight down, a head has only small corrections
    to make and its failure to reach even `down` is the whole story. If the directions spread
    widely, a per-object head has a real inference problem.
    """

    seed: int
    objects: int
    #: Tilt of each object's best direction away from straight down, in degrees.
    median_deg: float
    p90_deg: float
    max_deg: float
    within_10_deg: float
    within_20_deg: float
    within_45_deg: float
    #: Distinct bins used by the objects' best directions, out of the bin count.
    distinct_bins: int
    bins: int
    #: Share of objects whose best direction is the global one.
    share_at_down: float
    #: What the per-object oracle scores, split by whether the object wants `down` or something else.
    hit_all: float
    hit_objects_at_down: float
    hit_objects_off_down: float
    #: The same objects scored by `down` instead, so the per-object gain is attributable.
    down_hit_objects_off_down: float

    @property
    def gain_where_off_down(self) -> float:
        """How much the per-object direction buys on the objects that diverge from straight down.

        The per-object oracle read against a single global bin mixes two populations. This is the
        half that is about inferring a direction; the rest is objects that want what everything
        else wants.
        """
        return self.hit_objects_off_down - self.down_hit_objects_off_down


def object_direction_report(test_batches: "Sequence[dict[str, torch.Tensor]]", *,
                            approach_bins: int = 100,
                            max_tilt_deg: float = 135.0,
                            seed: int = 0) -> "ObjectDirections | None":
    """Per object: its best single bin, how far that sits from straight down, and what it buys."""
    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.grasp_encoding import approach_directions  # noqa: PLC0415

    directions = np.asarray(approach_directions(approach_bins, max_tilt_deg), dtype=np.float64)
    down_bin = int(np.argmax(directions @ np.array([0.0, 0.0, -1.0])))
    # Clipped before `arccos`. The dot product of two unit vectors can land a hair outside
    # [-1, 1] in floating point, and `arccos` returns NaN there rather than 0, which would silently
    # poison a median with the very objects that agree with `down` most exactly.
    tilt_deg = np.degrees(np.arccos(np.clip(directions @ directions[down_bin], -1.0, 1.0)))

    best_bins: list[int] = []
    hits: list[float] = []
    down_hits: list[float] = []
    weights: list[int] = []
    for chunk in test_batches:
        rows = (chunk["approach_bin"] >= 0) & chunk["supervise"].bool()
        for item in range(chunk["approach_bin"].shape[0]):
            mask = rows[item]
            if not bool(mask.any()):
                continue
            admissible = chunk["approach_set"][item][mask]
            best = int(admissible.sum(dim=0).argmax())
            best_bins.append(best)
            weights.append(int(mask.sum()))
            hits.append(float(admissible[:, best].float().mean()))
            down_hits.append(float(admissible[:, down_bin].float().mean()))
    if not best_bins:
        return None

    chosen = np.asarray(best_bins)
    tilts = tilt_deg[chosen]
    at_down = chosen == down_bin
    hit_array, down_array = np.asarray(hits), np.asarray(down_hits)
    weight = np.asarray(weights, dtype=np.float64)

    def _mean(values: np.ndarray, mask: np.ndarray) -> float:
        """Weighted by supervised points, because objects carry very different point counts."""
        if not mask.any() or weight[mask].sum() <= 0:
            return float("nan")
        return float(np.average(values[mask], weights=weight[mask]))

    every = np.ones(len(chosen), dtype=bool)
    return ObjectDirections(
        seed=seed, objects=len(chosen),
        median_deg=float(np.median(tilts)), p90_deg=float(np.percentile(tilts, 90)),
        max_deg=float(tilts.max()),
        within_10_deg=float((tilts <= 10.0).mean()),
        within_20_deg=float((tilts <= 20.0).mean()),
        within_45_deg=float((tilts <= 45.0).mean()),
        distinct_bins=int(len(np.unique(chosen))), bins=approach_bins,
        share_at_down=float(at_down.mean()),
        hit_all=_mean(hit_array, every),
        hit_objects_at_down=_mean(hit_array, at_down),
        hit_objects_off_down=_mean(hit_array, ~at_down),
        down_hit_objects_off_down=_mean(down_array, ~at_down))


def format_directions(reports: "Sequence[ObjectDirections]") -> str:
    """ASCII only, for the reason recorded throughout this package."""
    import statistics  # noqa: PLC0415

    if not reports:
        return "no object measured"
    lines: list[str] = []
    lines.append(f"{reports[0].objects} held-out object(s) per seed, {len(reports)} seed(s), "
                 f"{reports[0].bins} bins")
    lines.append("")
    lines.append(f"  {'seed':>5}{'objects':>9}{'median':>9}{'p90':>8}{'max':>8}"
                 f"{'<=10deg':>9}{'<=20deg':>9}{'<=45deg':>9}{'at down':>9}{'distinct':>10}")
    for report in reports:
        lines.append(
            f"  {report.seed:>5}{report.objects:>9}{report.median_deg:>9.1f}"
            f"{report.p90_deg:>8.1f}{report.max_deg:>8.1f}"
            f"{report.within_10_deg:>9.3f}{report.within_20_deg:>9.3f}{report.within_45_deg:>9.3f}"
            f"{report.share_at_down:>9.3f}{report.distinct_bins:>10}")
    lines.append("")
    lines.append("  what the per-object direction is worth, split by population:")
    lines.append(f"    {'seed':>5}{'all':>10}{'wants down':>13}{'wants other':>13}"
                 f"{'down scores':>13}{'gain there':>12}")
    for report in reports:
        lines.append(
            f"    {report.seed:>5}{report.hit_all:>10.4f}{report.hit_objects_at_down:>13.4f}"
            f"{report.hit_objects_off_down:>13.4f}{report.down_hit_objects_off_down:>13.4f}"
            f"{report.gain_where_off_down:>+12.4f}")
    lines.append("")
    # The decomposition is the point. `object_oracle` beating `down` mixes two populations:
    # objects that want straight down anyway, where a per-object head buys nothing, and objects that
    # want something else, where all of the value lives. Only the second number says whether a
    # per-object head is worth building.
    gains = [r.gain_where_off_down for r in reports]
    shares = [1.0 - r.share_at_down for r in reports]
    lines.append(f"  {statistics.fmean(shares) * 100:.1f} % of objects want something other than "
                 f"straight down;")
    lines.append(f"  on those, knowing the object's own direction buys "
                 f"{statistics.fmean(gains):+.4f} over answering down.")
    medians = [r.median_deg for r in reports]
    lines.append(f"  median tilt away from down: {statistics.fmean(medians):.1f} deg")
    return chr(10).join(lines)


def _score(approach_set: torch.Tensor, chosen: torch.Tensor) -> float:
    """`approach_hit`: is the chosen bin one this point actually admits?

    This is `approach_hit` as the retired binned arms defined it, so a saved result from that family
    reads on the same scale and the two tables can be put beside each other. The set generator's
    `top1_hit` is a different metric on a different population and must not go in the same column: a
    rule and a network scored differently cannot be compared.
    """
    return float(approach_set.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).float().mean())


def run_geometric_rules(train_batches: "Sequence[dict[str, torch.Tensor]]",
                        test_batches: "Sequence[dict[str, torch.Tensor]]", *,
                        approach_bins: int = 100,
                        max_tilt_deg: float = 135.0,
                        seed: int = 0) -> list[GeometricRule]:
    """Score every training-free rule on `test_batches`, using `train_batches` only where honest.

    The split is the one the trained arms saw. `constant` is chosen on the training side exactly as
    the trained runs choose their floor, so it reproduces their floor column and anchors this table
    to theirs.
    """
    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.grasp_encoding import approach_directions  # noqa: PLC0415

    directions = torch.as_tensor(
        np.asarray(approach_directions(approach_bins, max_tilt_deg), dtype=np.float32))

    # The honest constant: the most common labelled bin on the training units, applied everywhere.
    prior = torch.zeros(approach_bins, dtype=torch.float64)
    for chunk in train_batches:
        rows = (chunk["approach_bin"] >= 0) & chunk["supervise"].bool()
        if bool(rows.any()):
            prior += torch.bincount(chunk["approach_bin"][rows].flatten().cpu(),
                                    minlength=approach_bins).double()
    constant_bin = int(prior.argmax()) if float(prior.sum()) > 0 else 0

    # Straight down is found, not assumed to be bin 0. The spiral's construction happens to put its
    # first sample nearest the pole, but that is an implementation detail of `approach_directions`,
    # and a rule that silently depends on one would break the day the spiral is reordered.
    down_bin = int(torch.argmax(directions @ torch.tensor([0.0, 0.0, -1.0])))

    collected: list[tuple[str, bool, str, list[torch.Tensor]]] = [
        ("constant", False, f"the most common bin on the TRAINING units, bin {constant_bin}", []),
        ("down", False, f"straight down, bin {down_bin}; no data of any kind", []),
        ("normal", False, "per point, the bin nearest that point's surface normal", []),
        ("normal_inward", False, "per point, the bin nearest the NEGATED normal", []),
        ("object_oracle", True, "the single best bin for THIS object, from its own labels", []),
        ("global_oracle", True, "the single best bin for the HELD-OUT set, from its own labels", []),
    ]
    by_name = {name: bucket for name, _, _, bucket in collected}

    sets: list[torch.Tensor] = []
    global_prior = torch.zeros(approach_bins, dtype=torch.float64)
    for chunk in test_batches:
        rows = (chunk["approach_bin"] >= 0) & chunk["supervise"].bool()
        if not bool(rows.any()):
            continue
        approach_set = chunk["approach_set"][rows]
        sets.append(approach_set)
        count = int(rows.sum())
        device = approach_set.device
        local_directions = directions.to(device)

        by_name["constant"].append(torch.full((count,), constant_bin, dtype=torch.long,
                                              device=device))
        by_name["down"].append(torch.full((count,), down_bin, dtype=torch.long, device=device))

        # The normals are the first three feature channels, exactly as `build_sample` stacks them.
        normals = chunk["features"][..., :3][rows].to(local_directions.dtype)
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        by_name["normal"].append((normals @ local_directions.T).argmax(dim=-1))
        by_name["normal_inward"].append(((-normals) @ local_directions.T).argmax(dim=-1))

        # Per sample, not per batch. One batch item is one target object, so the per-object oracle
        # has to be taken inside the item; taking it over the batch would silently make it a
        # per-batch constant and understate the ceiling it exists to measure.
        per_object: list[torch.Tensor] = []
        item_rows = rows
        for item in range(chunk["approach_bin"].shape[0]):
            mask = item_rows[item]
            if not bool(mask.any()):
                continue
            counts = chunk["approach_set"][item][mask].sum(dim=0)
            per_object.append(torch.full((int(mask.sum()),), int(counts.argmax()),
                                         dtype=torch.long, device=device))
        by_name["object_oracle"].append(torch.cat(per_object) if per_object
                                        else torch.zeros(0, dtype=torch.long, device=device))
        global_prior += approach_set.sum(dim=0).double().cpu()

    if not sets:
        return []
    every_set = torch.cat(sets)
    best_global = int(global_prior.argmax())
    by_name["global_oracle"].append(
        torch.full((every_set.shape[0],), best_global, dtype=torch.long, device=every_set.device))

    supervised = int(every_set.shape[0])
    admissible = float(every_set.sum(-1).float().mean())
    rules: list[GeometricRule] = []
    for name, oracle, what, chunks in collected:
        chosen = torch.cat(chunks)
        if chosen.shape[0] != supervised:  # pragma: no cover (guards a future edit, not a path)
            raise AssertionError(f"rule {name!r} produced {chosen.shape[0]} choices for "
                                 f"{supervised} supervised points")
        rules.append(GeometricRule(
            name=name, oracle=oracle, seed=seed, hit=_score(every_set, chosen),
            supervised_points=supervised, admissible_bins=admissible,
            what=what if name != "global_oracle" else f"{what}, bin {best_global}"))
    return rules


def format_geometric(rules: "Sequence[GeometricRule]",
                     reference: "dict[str, float] | None" = None) -> str:
    """The table.

    ASCII only in every string this returns. A non-ASCII character raises UnicodeEncodeError on a
    cp1252 console and loses a finished run before it writes anything.
    """
    import statistics  # noqa: PLC0415

    if not rules:
        return "no rule measured"
    lines: list[str] = []
    seeds = sorted({rule.seed for rule in rules})
    lines.append(f"{rules[0].supervised_points} supervised point(s) per seed, "
                 f"{rules[0].admissible_bins:.2f} admissible bins each, {len(seeds)} seed(s)")
    lines.append("")

    grouped: dict[str, list[GeometricRule]] = {}
    for rule in rules:
        grouped.setdefault(rule.name, []).append(rule)

    lines.append(f"  {'rule':<16}{'kind':>8}{'mean hit':>10}{'spread':>9}   what it is")
    for name, found in grouped.items():
        hits = [r.hit for r in found]
        spread = (max(hits) - min(hits)) if len(hits) > 1 else float("nan")
        kind = "ORACLE" if found[0].oracle else "honest"
        lines.append(f"  {name:<16}{kind:>8}{statistics.fmean(hits):>10.4f}{spread:>9.4f}   "
                     f"{found[0].what}")

    if len(seeds) > 1:
        lines.append("")
        lines.append(f"  {'rule':<16}" + "".join(f"{'seed ' + str(s):>10}" for s in seeds))
        for name, found in grouped.items():
            by_seed = {r.seed: r.hit for r in found}
            lines.append(f"  {name:<16}"
                         + "".join(f"{by_seed.get(s, float('nan')):>10.4f}" for s in seeds))

    if reference:
        lines.append("")
        lines.append("  for comparison, the TRAINED arms on the same metric and the same split:")
        for name, value in reference.items():
            lines.append(f"    {name:<20}{value:>10.4f}")

    honest = {n: statistics.fmean([r.hit for r in f])
              for n, f in grouped.items() if not f[0].oracle}
    oracles = {n: statistics.fmean([r.hit for r in f])
               for n, f in grouped.items() if f[0].oracle}
    lines.append("")
    if honest:
        best = max(honest, key=honest.__getitem__)
        lines.append(f"  best HONEST rule: {best} at {honest[best]:.4f}")
        per_point = {n: v for n, v in honest.items() if n.startswith("normal")}
        flat = max((v for n, v in honest.items() if not n.startswith("normal")), default=0.0)
        if per_point:
            best_normal = max(per_point, key=per_point.__getitem__)
            margin = per_point[best_normal] - flat
            # The question the module exists for, stated rather than left to the reader. The
            # per-point rules read the surface normal; the flat rules read nothing about the point.
            # If the normal buys nothing, the target carries no per-point geometric signal that a
            # simple rule can see, and a per-point head is solving a problem that may not be there.
            lines.append(f"  the surface normal buys {margin:+.4f} over the best rule that ignores "
                         f"the point ({best_normal} against a flat rule)")
    if oracles and honest:
        ceiling = max(oracles, key=oracles.__getitem__)
        lines.append(f"  ceiling: {ceiling} reaches {oracles[ceiling]:.4f} by reading the answers, "
                     f"which no rule and no network may do")
    lines.append("")
    lines.append("  ORACLE rows read the labels of the points they are scored on. They bound what is")
    lines.append("  achievable; they are NOT results and must not be compared against a trained arm")
    lines.append("  as though they were.")
    return "\n".join(lines)
