"""Judge a corpus before anyone trains on it. Four ways a corpus is unusable without looking wrong.

1. Zero variance. A column that holds one value on every row looks exactly like a feature. Training
   on it is harmless to the fit and fatal to the diagnosis, because a correlation study over it
   reports "no signal" for a reason that has nothing to do with grasping. The mirror image is a
   trainer reading a key nobody wrote, which lands on a constant the same way.

2. Train/serve skew. The more dangerous half, and the one only two corpora can reveal. A feature
   that is constant in training and varies at serving is not merely wasted: a tree learns one branch
   and sends every serving row down it, a network applies an untrained weight to a live input. The
   same field can be one distinct value in the analytic reference and a wide range over the
   calculator's own candidates: same name, two populations.

3. Group leakage. A `split` column with one value on every row is no held-out set at all, and a
   row-level split puts the same object on both sides, which inflates every number reported from
   the corpus. The RL layer checks the same property over its own datasets with `audit-leakage`.

4. Class balance. A trainer has to be told the positive rate before it starts, and a corpus with no
   positive row, or with a positive in fewer than two groups, is unusable rather than difficult.

A fifth check the numbers alone cannot give: the frame. Object-frame and BASE-frame vectors sit under
identical column names, so comparing them compares different quantities. `FeatureSpec.frame` carries
the frame and a mismatch is refused outright.

Pure and read-only: it opens no cell, loads no model and trains nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.robot.grasping.deep.ranker.features import FeatureSpec
from src.utility.log_cfg import create_logger

from datagen.constants import CORPUS_GATE_LOG_FILE, DATAGEN_LOG_DIR
from datagen.corpus.columns import ColumnStats, Table, column_stats

__all__ = ["CorpusVerdict", "assess_corpus", "format_verdict"]

logger = create_logger("CorpusGate", CORPUS_GATE_LOG_FILE, log_dir=DATAGEN_LOG_DIR)

#: How many extra split values one group may span before it counts as leakage. Zero: a single shared
#: object is the leak, and a tolerance here would only decide how much of one to accept.
_LEAK_TOLERANCE = 0


@dataclass(frozen=True, slots=True)
class CorpusVerdict:
    """The answer, and every number behind it: a bare bool would hide what to fix."""

    spec: str
    rows: int
    groups: int
    positives: int
    groups_with_positive: int
    features: tuple[ColumnStats, ...] = ()
    #: Columns present in the file but not in the spec. Reported, never blocking: this is where a dead
    #: column is named so it can be dropped deliberately instead of quietly entering a study.
    unused: tuple[ColumnStats, ...] = ()
    blocking: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        """No blocking finding. A warning is something to know, not something that stops a run."""
        return not self.blocking

    @property
    def positive_rate(self) -> float:
        return (self.positives / self.rows) if self.rows else 0.0


def _label_values(table: Table, spec: FeatureSpec) -> np.ndarray | None:
    values = table.get(spec.label)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def assess_corpus(
    table: Table, spec: FeatureSpec, *, serving: Table | None = None,
    serving_spec: FeatureSpec | None = None,
) -> CorpusVerdict:
    """Judge one corpus against a spec, optionally against a second corpus it will be served on.

    ``serving`` is another corpus rather than a live runtime: the check is whether a feature's
    distribution survives the move, and that question is the same whichever two populations are
    named, one corpus against another or training against live.
    """
    blocking: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    rows = max((len(np.asarray(v)) for v in table.values()), default=0)
    if rows == 0:
        return CorpusVerdict(spec.name, 0, 0, 0, 0, blocking=("the corpus is empty",))

    # ---- 1. the declared features exist and vary ---------------------------------------------
    features: list[ColumnStats] = []
    for name in spec.features:
        values = table.get(name)
        stats = column_stats(name, values if values is not None else np.empty(0))
        features.append(stats)
        if stats.verdict == "absent":
            blocking.append(
                f"feature {name!r} is not in this corpus; a trainer that maps a missing key to 0.0 "
                f"would fit a column of zeros and report a converged model"
            )
        elif stats.verdict == "nonfinite":
            blocking.append(
                f"feature {name!r} has {stats.nonfinite} NaN/infinite row(s) of {stats.rows}")
        elif stats.verdict == "constant":
            blocking.append(
                f"feature {name!r} is CONSTANT across all {stats.rows} rows "
                f"({'at ' + format(stats.minimum, '.6g') if stats.numeric else 'one value'}); "
                f"it carries no information, and any study that reports 'no signal' from it is "
                f"reporting the column, not the world"
            )

    # ---- 2. what is in the file and not in the spec -------------------------------------------
    unused = tuple(
        column_stats(name, values) for name, values in sorted(table.items())
        if name not in spec.features and name not in (spec.label, spec.group)
    )
    dead = [stats.key for stats in unused if stats.verdict == "constant"]
    if dead:
        # Named, not blocked: a constant column that is not a feature can be correct.
        # `contact_angle_deg` is exactly zero because the labeller enumerates exact antipodal axes.
        # What is not fine is nobody knowing.
        notes.append(
            f"constant column(s) present but not declared as features: {', '.join(dead)}; "
            f"correct only if that is deliberate"
        )

    # ---- 3. the label --------------------------------------------------------------------------
    label = _label_values(table, spec)
    positives = 0
    if label is None:
        blocking.append(f"label {spec.label!r} is not in this corpus; there is nothing to predict")
    else:
        positives = int(np.count_nonzero(label > 0.5))
        if positives == 0:
            blocking.append(f"label {spec.label!r} has no positive row; nothing to learn from")
        elif positives == len(label):
            blocking.append(f"label {spec.label!r} is positive on every row; nothing to separate")
        else:
            rate = positives / len(label)
            if rate < 0.01 or rate > 0.99:
                warnings.append(
                    f"label {spec.label!r} is {100 * rate:.2f} % positive ({positives}/{len(label)}); "
                    f"heavily imbalanced, so accuracy will be meaningless and the split has to be "
                    f"stratified or the validation set may hold almost no positives"
                )

    # ---- 4. groups, and whether a split can even be made ---------------------------------------
    groups_array = table.get(spec.group)
    groups = 0
    groups_with_positive = 0
    if groups_array is None:
        blocking.append(
            f"group column {spec.group!r} is not in this corpus; without it a split cannot be made "
            f"that keeps one object on one side, and every reported number is inflated by leakage"
        )
    else:
        keys = np.asarray(groups_array)
        unique, counts = np.unique(keys, return_counts=True)
        groups = int(unique.size)
        if groups < 2:
            blocking.append(f"every row belongs to one {spec.group}; no grouped split is possible")
        if label is not None:
            for key in unique:
                if bool((label[keys == key] > 0.5).any()):
                    groups_with_positive += 1
            if groups_with_positive < 2:
                blocking.append(
                    f"only {groups_with_positive} {spec.group}(s) hold a positive; a grouped split "
                    f"cannot put a positive on both sides"
                )
        notes.append(
            f"{groups} {spec.group}(s), {int(counts.min())} .. {int(counts.max())} rows each "
            f"(median {int(np.median(counts))}); a ROW-level split would put the same "
            f"{spec.group} on both sides"
        )

    # ---- 5. an existing split column, if the corpus ships one -----------------------------------
    split = table.get("split")
    if split is not None and groups_array is not None:
        split_stats = column_stats("split", split)
        if split_stats.distinct <= 1:
            warnings.append(
                f"the corpus carries a 'split' column with one value; there is NO held-out set in "
                f"it, and one grouped by {spec.group!r} has to be made before training"
            )
        else:
            leaked = _leaking_groups(np.asarray(groups_array), np.asarray(split))
            if leaked:
                blocking.append(
                    f"{len(leaked)} {spec.group}(s) appear in more than one split "
                    f"(e.g. {', '.join(str(g) for g in leaked[:3])}); the same object is on both "
                    f"sides, so every validation number from this corpus is inflated"
                )

    # ---- 6. train vs serve ---------------------------------------------------------------------
    if serving is not None:
        blocking.extend(_serving_findings(
            table, spec, serving, serving_spec or spec, features, warnings))

    return CorpusVerdict(
        spec=spec.name, rows=rows, groups=groups, positives=positives,
        groups_with_positive=groups_with_positive,
        features=tuple(features), unused=unused,
        blocking=tuple(blocking), warnings=tuple(warnings), notes=tuple(notes),
    )


def _leaking_groups(groups: np.ndarray, split: np.ndarray) -> list[object]:
    """Group keys that appear under more than one split value."""
    leaked: list[object] = []
    for key in np.unique(groups):
        if np.unique(split[groups == key]).size > _LEAK_TOLERANCE + 1:
            leaked.append(key)
    return leaked


def _serving_findings(
    table: Table, spec: FeatureSpec, serving: Table, serving_spec: FeatureSpec,
    train_stats: list[ColumnStats], warnings: list[str],
) -> list[str]:
    """Does every feature survive the move to the population the model will actually see?"""
    if serving_spec.frame != spec.frame:
        # Refused outright. The columns line up by name and hold different quantities, and nothing in
        # the numbers can show that; see the module docstring.
        return [
            f"the training corpus is in the {spec.frame!r} frame and the serving corpus in "
            f"{serving_spec.frame!r}; the same column names hold different quantities, so comparing "
            f"them is meaningless. Convert one, or compare against a corpus in the same frame."
        ]

    findings: list[str] = []
    for stats in train_stats:
        values = serving.get(stats.key)
        if values is None:
            findings.append(
                f"feature {stats.key!r} is in the training corpus and ABSENT from the serving one; "
                f"at inference it would be filled with a default the model never saw"
            )
            continue
        served = column_stats(stats.key, values)
        if stats.verdict == "constant" and served.verdict == "live":
            findings.append(
                f"feature {stats.key!r} is CONSTANT in training ({stats.minimum:.6g}) and LIVE when "
                f"served ({served.distinct} distinct, {served.minimum:.4g} .. {served.maximum:.4g}); "
                f"train/serve skew: the model learnt one branch and every served row goes down it"
            )
        elif stats.verdict == "live" and served.verdict == "constant":
            findings.append(
                f"feature {stats.key!r} varies in training and is CONSTANT when served "
                f"({served.minimum:.6g}); whatever the model learnt from it cannot apply"
            )
        elif stats.verdict == "live" and served.verdict == "live":
            _range_warning(stats, served, warnings)
    return findings


def _range_warning(train: ColumnStats, served: ColumnStats, warnings: list[str]) -> None:
    """A feature that stays inside its trained range is interpolation; outside it is a guess."""
    span = train.maximum - train.minimum
    if span <= 0.0:
        return
    below = (train.minimum - served.minimum) / span
    above = (served.maximum - train.maximum) / span
    worst = max(below, above)
    if worst > 0.25:
        warnings.append(
            f"feature {train.key!r} is served outside its trained range by {100 * worst:.0f} % of that "
            f"range (trained {train.minimum:.4g} .. {train.maximum:.4g}, served {served.minimum:.4g} "
            f".. {served.maximum:.4g}); extrapolation, not interpolation"
        )


def format_verdict(verdict: CorpusVerdict, *, verbose: bool = True) -> str:
    """The report a person reads. Blocking findings first, because that is the answer."""
    lines: list[str] = [
        f"corpus against spec {verdict.spec!r}: {verdict.rows} row(s), {verdict.groups} group(s), "
        f"{verdict.positives} positive ({100 * verdict.positive_rate:.2f} %), "
        f"{verdict.groups_with_positive} group(s) with a positive",
        f"verdict: {'USABLE' if verdict.usable else 'REFUSED'}",
    ]
    for title, entries in (("BLOCKING", verdict.blocking), ("warning", verdict.warnings),
                           ("note", verdict.notes)):
        for entry in entries:
            lines.append(f"  [{title}] {entry}")
    if verbose:
        lines.append("  features:")
        lines.extend(f"    {stats.describe()}" for stats in verdict.features)
        if verdict.unused:
            lines.append("  present but not declared as features:")
            lines.extend(f"    {stats.describe()}" for stats in verdict.unused)
    return "\n".join(lines)
