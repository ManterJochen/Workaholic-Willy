"""Is this record log trainable? Ask before spending a training run, or a month of collecting.

A cell logs `GraspAttemptRecord` lines for weeks and someone eventually trains on them. Two failures
make that corpus worthless, both silent, and both cheap to detect up front:

1. Dead features. ``_project_feature`` maps a missing key to ``0.0`` without a word. A record log
   whose features are absent trains a model over a column of zeros and reports a converged fit.
2. No pairs. The pairwise ranker forms success/fail pairs within a group. A log where every group
   is all-success, all-failure, or a single row yields zero pairs, and the trainer reports a
   converged model over nothing at all.

There is a third, which is the one a customer is most likely to hit: the per-candidate feature rows
only exist when the ranking shadow runs. A cell with `rl.mode` other than `rl_shadow` logs records
that look perfectly healthy and contain no per-candidate features whatsoever. This module names that
specifically, because "your features are dead" sends someone to the wrong place.

The distinction that matters most, and the one no dashboard shows: a pairwise ranker differences
features within a group. A feature identical for every member of a group contributes exactly zero to
the gradient, however well populated it is. Scene-level, pick-level and object-level quantities are
all in that category. So this reports occupancy at both levels and judges them differently: a
record-level feature that varies across scenes is useful for a pointwise model and useless for a
pairwise one.

Pure and read-only: it opens no cell, loads no policy, and trains nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "ColumnStats",
    "DatasetReadiness",
    "assess_records",
    "format_readiness",
]


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """What one feature column did across the rows it was measured on."""

    key: str
    rows: int
    nonzero: int
    distinct: int
    stdev: float
    minimum: float
    maximum: float

    @property
    def verdict(self) -> str:
        if self.rows == 0:
            return "absent"
        if self.nonzero == 0:
            return "absent"
        if self.distinct <= 1:
            return "constant"
        return "live"


@dataclass(frozen=True, slots=True)
class DatasetReadiness:
    """The verdict, and every number behind it."""

    records: int
    groups: int
    groups_with_pairs: int
    pairs: int
    successes: int
    failures: int
    candidate_rows: int
    record_level: tuple[ColumnStats, ...] = ()
    per_candidate: tuple[ColumnStats, ...] = ()
    blocking: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def trainable(self) -> bool:
        """No blocking finding. Warnings are things to know, not things that stop a run."""
        return not self.blocking

    @property
    def live_per_candidate(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.per_candidate if c.verdict == "live")


def _column(key: str, values: Sequence[float]) -> ColumnStats:
    rows = len(values)
    if rows == 0:
        return ColumnStats(key, 0, 0, 0, 0.0, 0.0, 0.0)
    mean = sum(values) / rows
    variance = sum((v - mean) ** 2 for v in values) / rows
    return ColumnStats(
        key=key,
        rows=rows,
        nonzero=sum(1 for v in values if v != 0.0),
        distinct=len({round(v, 9) for v in values}),
        stdev=math.sqrt(variance),
        minimum=min(values),
        maximum=max(values),
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def assess_records(records: Sequence[Mapping[str, Any]]) -> DatasetReadiness:
    """Judge a record log without training on it."""
    from .dataset import OUTCOME_CLASS_SUCCESS, derive_outcome_class
    from .ranking_policy import RANKING_FEATURE_KEYS
    from .train_ranking import _group_key, _record_features

    blocking: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if not records:
        return DatasetReadiness(
            0, 0, 0, 0, 0, 0, 0,
            blocking=("the record log is empty",),
        )

    # ---- grouping + pairs -------------------------------------------------------------------
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    pairs = 0
    groups_with_pairs = 0
    successes = 0
    failures = 0
    for members in groups.values():
        wins = [m for m in members if derive_outcome_class(m) == OUTCOME_CLASS_SUCCESS]
        losses = [m for m in members if derive_outcome_class(m) != OUTCOME_CLASS_SUCCESS]
        successes += len(wins)
        failures += len(losses)
        if wins and losses:
            groups_with_pairs += 1
            pairs += len(wins) * len(losses)

    if pairs == 0:
        blocking.append(
            f"ZERO success/fail pairs across {len(groups)} group(s): the pairwise ranker has nothing "
            f"to learn from and will report a converged fit over no data. Either every group is "
            f"all-success / all-failure, or the grouping key is wrong; `_group_key` looks for "
            f"`scene_family_id`, then `scene_id`, then `episode_id` / `session_id` / `bin_id`, and "
            f"falls back to the attempt-id prefix. A log whose scene identity is under a different "
            f"name puts every record in its own group."
        )
    if len(groups) == len(records):
        warnings.append(
            f"every record is its own group ({len(groups)} groups / {len(records)} records); almost "
            f"always a missing `scene_id` (or `scene_family_id`) in `extra`, not a property of the data"
        )
    if successes == 0:
        blocking.append("no record has final_outcome 'succeeded': there is no positive class")
    if failures == 0:
        blocking.append("every record succeeded: there is no negative class")

    # ---- per-candidate rows -----------------------------------------------------------------
    candidate_columns: dict[str, list[float]] = {}
    candidate_rows = 0
    records_with_candidates = 0
    for record in records:
        extra = record.get("extra") or {}
        candidates = extra.get("rl_candidate_features") or ()
        if candidates:
            records_with_candidates += 1
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or "features" not in candidate:
                continue  # the tail-aggregate row
            candidate_rows += 1
            for block, prefix in (("features", ""), ("geometry", "geom.")):
                for key, value in (candidate.get(block) or {}).items():
                    number = _numeric(value)
                    if number is not None:
                        candidate_columns.setdefault(prefix + key, []).append(number)

    if candidate_rows == 0:
        # The customer's most likely failure, and it is not "your features are dead".
        warnings.append(
            "no per-candidate feature rows at all (`extra.rl_candidate_features` is absent). Those "
            "are written only when the RANKING SHADOW ran: the cell needs `robot.rl.mode: rl_shadow` "
            "with `policy_id` + `artifact_path` set. Without them the log can still support a "
            "pointwise model and KPIs, but a pairwise RANKER has no per-candidate features to rank."
        )
    else:
        notes.append(
            f"{candidate_rows} per-candidate rows from {records_with_candidates}/{len(records)} records"
        )

    per_candidate = tuple(
        _column(key, values) for key, values in sorted(candidate_columns.items())
    )
    live_candidate = [c for c in per_candidate if c.verdict == "live"]
    if per_candidate and not live_candidate:
        blocking.append(
            "every per-candidate feature is constant or absent: a pairwise model differences features "
            "within a group, so there is literally nothing for it to separate candidates by"
        )
    elif per_candidate and len(live_candidate) < 2:
        warnings.append(
            f"only {len(live_candidate)} per-candidate feature varies "
            f"({', '.join(c.key for c in live_candidate)}); a ranker over one feature is a threshold"
        )

    # ---- record-level features ---------------------------------------------------------------
    record_columns: dict[str, list[float]] = {key: [] for key in RANKING_FEATURE_KEYS}
    for record in records:
        row = _record_features(record)
        for key, value in zip(RANKING_FEATURE_KEYS, row):
            record_columns[key].append(float(value))
    record_level = tuple(_column(key, values) for key, values in record_columns.items())

    dead = [c.key for c in record_level if c.verdict != "live"]
    if len(dead) == len(record_level):
        blocking.append(
            "every one of the declared ranking features reads 0.0 or a constant. `_project_feature` "
            "maps a missing key to 0.0 SILENTLY, so this looks identical to a healthy log; check "
            "that the producing blocks are enabled for the mode the cell runs in"
        )
    elif dead:
        warnings.append(
            f"{len(dead)}/{len(record_level)} declared ranking features are constant or absent: "
            f"{', '.join(sorted(dead))}"
        )

    notes.append(
        "a feature constant WITHIN a group contributes exactly zero to a pairwise ranker, whatever "
        "its record-level spread; scene, pick and object-level quantities are all in that class"
    )

    return DatasetReadiness(
        records=len(records),
        groups=len(groups),
        groups_with_pairs=groups_with_pairs,
        pairs=pairs,
        successes=successes,
        failures=failures,
        candidate_rows=candidate_rows,
        record_level=record_level,
        per_candidate=per_candidate,
        blocking=tuple(blocking),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def format_readiness(readiness: DatasetReadiness, *, verbose: bool = True) -> str:
    """A report a person can act on. The verdict first, then the numbers behind it."""
    lines: list[str] = []
    verdict = "TRAINABLE" if readiness.trainable else "NOT TRAINABLE"
    lines.append(
        f"  {verdict}: {readiness.records} records, {readiness.groups} groups, "
        f"{readiness.pairs} success/fail pairs across {readiness.groups_with_pairs} group(s)"
    )
    lines.append(
        f"  outcomes: {readiness.successes} succeeded / {readiness.failures} not, "
        f"{readiness.candidate_rows} per-candidate rows"
    )
    for item in readiness.blocking:
        lines.append(f"  BLOCKING  {item}")
    for item in readiness.warnings:
        lines.append(f"  warning   {item}")
    if verbose and readiness.per_candidate:
        lines.append("  --- per-candidate (what a PAIRWISE ranker can use) ---")
        for column in sorted(readiness.per_candidate, key=lambda c: (c.verdict != "live", c.key)):
            lines.append(
                f"    {column.key:<38} {column.verdict:<9} distinct={column.distinct:<4} "
                f"stdev={column.stdev:.4f}  [{column.minimum:.4g}, {column.maximum:.4g}]"
            )
        lines.append(f"    LIVE per-candidate: {len(readiness.live_per_candidate)}/{len(readiness.per_candidate)}")
    if verbose and readiness.record_level:
        live = sum(1 for c in readiness.record_level if c.verdict == "live")
        lines.append(f"  record-level declared features live: {live}/{len(readiness.record_level)}")
    for note in readiness.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
