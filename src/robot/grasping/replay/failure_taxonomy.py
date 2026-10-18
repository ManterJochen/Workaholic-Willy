"""Replay-only failure taxonomy and recommendation engine.

This module is read-only with respect to the runtime. It operates
offline on :class:`GraspAttemptRecord` instances (loaded from the JSONL
logs a cell or a simulation run writes) and produces a deterministic taxonomy verdict
per record plus an aggregate JSON report carrying static
per-root-cause recommendations.

Design contract:

* Root causes are ordered by safety/actionability precedence in
  :class:`FailureRootCause`. When multiple rules fire on a single
  record, the earliest enum member becomes the primary cause and
  the rest are recorded (in enum order) in ``also_matched``.

* The classifier reads only the following honest signals:

  * ``final_outcome``: the outcome string already stable in the
    schema.
  * A small set of optional ``extra.*`` symptom flags
    (``collision_evidence``, ``calibration_drift_evidence``,
    ``slip_evidence``, ``empty_air_evidence``,
    ``deformable_misclass_evidence``, ``occlusion_misread_evidence``).
    Real runtime records without these flags fall to
    :attr:`FailureRootCause.UNCLASSIFIED` rather than being
    force-classified.

* Successful records (``final_outcome == "succeeded"``) are excluded
  from the coverage denominator. :func:`classify_record` returns
  :attr:`FailureRootCause.UNCLASSIFIED` for them so the per-record
  contract is uniform; :func:`build_taxonomy_report` counts them
  under ``success_count`` instead of entering them into
  ``counts_by_cause``.

* Recommendations live in the in-module :data:`RECOMMENDATIONS`
  table. Config-driven recommendation application is deferred.

* Versions. :data:`TAXONOMY_VERSION` 2 (2026-09-29) rewrote the
  recommendations of version 1, which advised features removed on
  2026-09-28 and 2026-09-29 (the multi-view fusion volume, the
  active-perception attempts, the commit gate's corridor radius, the
  ``dense_autonomous`` mode). Version 3 (2026-09-29, later) rewrote the
  occlusion recommendation once a wrist pick fuses its looks: it no
  longer says a pick stops at the first look that finds something. The
  root causes, their precedence and the rules are unchanged, so the
  counts of a version-1 or version-2 report compare directly with
  version 3. Nothing in this module reads a report back:
  :class:`TaxonomyReport` is built by :func:`build_taxonomy_report` and
  refuses any version but the current one at construction, so an
  older report on disk stays a file, and its recommendations are the
  old wording. Re-run ``--failure-taxonomy`` over the same logs for the
  current ones. The adaptation planner, the one reader of a report's
  ``counts_by_cause``, left on 2026-10-16 with the keys it adapted, and
  the synthetic labeled pack the classifier was graded against left with
  the synthetic soak.

* The JSON report is byte-stable: keys are sorted and lists are
  ordered by impact (count desc, then enum order tie-break).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.robot.grasping.constants import (
    REPLAY_FAILURE_TAXONOMY_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
)

#: The CLI records the coverage number; what only exists here is which root cause
#: dominated and how many failures the classifier could not name at all: the two
#: facts that decide where the next fix goes.
logger = create_grasping_logger("FailureTaxonomy", REPLAY_FAILURE_TAXONOMY_LOG_FILE)


#: Bump on any contract-level change to taxonomy semantics,
#: recommendation wording, or report schema. Do not bump for
#: pure additions that preserve old behaviour byte-identically.
#: 2 (2026-09-29): the recommendations stop advising removed features
#: (see the module docstring); the classification is unchanged.
#: 3 (2026-09-29, later): the occlusion recommendation says what a wrist
#: pick does with its looks now, fused; the classification is unchanged.
TAXONOMY_VERSION: int = 3


class FailureRootCause(StrEnum):
    """Root-cause taxonomy in safety/actionability precedence order (declaration order = precedence)."""

    COLLISION_REJECTION = "collision_rejection"
    CALIBRATION_DRIFT_SUSPECTED = "calibration_drift_suspected"
    SLIP_AFTER_GRASP = "slip_after_grasp"
    EMPTY_AIR_GRASP = "empty_air_grasp"
    DEFORMABLE_MISCLASSIFICATION = "deformable_misclassification"
    OCCLUSION_MISREAD = "occlusion_misread"
    UNCLASSIFIED = "unclassified"


#: Stable list in declaration order, used to drive deterministic
#: rule evaluation and tie-breaking.
ROOT_CAUSE_ORDER: tuple[FailureRootCause, ...] = tuple(FailureRootCause)


#: Static recommendation table. Every non-UNCLASSIFIED member must
#: have an entry; UNCLASSIFIED intentionally has none (no actionable
#: guidance for unmatched records).
RECOMMENDATIONS: Mapping[FailureRootCause, str] = {
    FailureRootCause.COLLISION_REJECTION: (
        "Investigate planner safety margin and gripper clearance; "
        "verify the scene occupancy the planner checked, the live "
        "camera world (robot.safety.planning_world) that every "
        "calibrated camera feeds, before re-enabling AUTO mode."
    ),
    FailureRootCause.CALIBRATION_DRIFT_SUSPECTED: (
        "Trigger the ETH/EIH recalibration workflow and freeze AUTO "
        "execution until the drift watchdog reports a clean residual."
    ),
    FailureRootCause.SLIP_AFTER_GRASP: (
        "Review gripper force and pad friction for this part, and read "
        "the hold the gripper reported right after the close "
        "(is_object_detected and hold_evidence; gOBJ on a Robotiq): a "
        "hold reported and then lost points at grip force, pad wear or "
        "a grasp far from the part's centre of mass."
    ),
    FailureRootCause.EMPTY_AIR_GRASP: (
        "Inspect segmentation mask confidence and depth_confidence "
        "floor; review the approach-point lift-off offset to avoid "
        "closing on empty space."
    ),
    FailureRootCause.DEFORMABLE_MISCLASSIFICATION: (
        "Audit the deformable-flag detector and widen profile "
        "coverage; consider the dense_clutter mode for deformable "
        "scenes."
    ),
    FailureRootCause.OCCLUSION_MISREAD: (
        "Give the grasp a view the clutter does not hide: add a second "
        "calibrated camera, which the cell then fuses by itself, or "
        "declare a wrist camera's looks "
        "(in the program, or robot.motion.look_deg) from the "
        "sides the clutter leaves open. A wrist pick visits its declared "
        "looks in order, fuses each with the ones before, and stops at "
        "the first whose grasp is valid and certain; one generated view "
        "is taken only as a last resort, once every declared look is "
        "used up. Order the looks so the part's open side comes first."
    ),
}


_SUCCESS_OUTCOME: str = "succeeded"

#: Outcome strings that must be present for the corresponding
#: symptom flag to be considered. This keeps classification honest:
#: a stray ``slip_evidence`` flag on a ``succeeded`` record will not
#: relabel the record as a slip.
_FAILURE_GATING_OUTCOMES: frozenset[str] = frozenset(
    {
        "execution_failed",
        "verification_failed",
        "no_valid_grasp",
        "decision_fail_closed",
        "recovery_exhausted",
        "unsafe_recovery_refused",
        "no_target",
        # Retired with the two-scan refinement on 2026-09-29: no pick reports it since. Kept so a
        # record logged before then is still read as the failure it was.
        "refinement_diverged",
        # Retired with the multi-view commit gate on 2026-09-28: no pick reports it since. Kept so
        # a record logged before then is still read as the failure it was.
        "no_commit_insufficient_fusion",
    }
)


@dataclass(frozen=True, slots=True)
class TaxonomyVerdict:
    """Per-record classification result."""

    primary: FailureRootCause
    also_matched: tuple[FailureRootCause, ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.primary, FailureRootCause):
            raise TypeError(
                f"primary must be FailureRootCause; got "
                f"{type(self.primary).__name__}"
            )
        if not isinstance(self.also_matched, tuple):
            raise TypeError("also_matched must be a tuple")
        for member in self.also_matched:
            if not isinstance(member, FailureRootCause):
                raise TypeError(
                    "also_matched entries must be FailureRootCause"
                )
            if member is self.primary:
                raise ValueError(
                    "also_matched must not contain the primary cause"
                )
        # also_matched must be in enum declaration order, no dupes.
        seen: set[FailureRootCause] = set()
        last_index = -1
        for member in self.also_matched:
            if member in seen:
                raise ValueError(
                    "also_matched must not contain duplicates"
                )
            seen.add(member)
            idx = ROOT_CAUSE_ORDER.index(member)
            if idx <= last_index:
                raise ValueError(
                    "also_matched must be sorted in enum order"
                )
            last_index = idx


def _extra_flag(extra: Mapping[str, Any], key: str) -> bool:
    return extra.get(key) is True


def _is_failure(final_outcome: str) -> bool:
    return (
        final_outcome != _SUCCESS_OUTCOME
        and final_outcome in _FAILURE_GATING_OUTCOMES
    )


def _check_collision(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if not _is_failure(final_outcome):
        return False
    if _extra_flag(extra, "collision_evidence"):
        return True
    if final_outcome == "unsafe_recovery_refused":
        return True
    return False


def _check_calibration_drift(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if not _is_failure(final_outcome):
        return False
    return _extra_flag(extra, "calibration_drift_evidence")


def _check_slip(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if final_outcome != "verification_failed":
        return False
    return _extra_flag(extra, "slip_evidence")


def _check_empty_air(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if final_outcome != "verification_failed":
        return False
    return _extra_flag(extra, "empty_air_evidence")


def _check_deformable(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if not _is_failure(final_outcome):
        return False
    return _extra_flag(extra, "deformable_misclass_evidence")


def _check_occlusion(final_outcome: str, extra: Mapping[str, Any]) -> bool:
    if not _is_failure(final_outcome):
        return False
    return _extra_flag(extra, "occlusion_misread_evidence")


#: Rule table: each entry pairs a FailureRootCause with its predicate,
#: in :data:`ROOT_CAUSE_ORDER` order. UNCLASSIFIED has no predicate.
_RULES: tuple[tuple[FailureRootCause, Any], ...] = (
    (FailureRootCause.COLLISION_REJECTION, _check_collision),
    (FailureRootCause.CALIBRATION_DRIFT_SUSPECTED, _check_calibration_drift),
    (FailureRootCause.SLIP_AFTER_GRASP, _check_slip),
    (FailureRootCause.EMPTY_AIR_GRASP, _check_empty_air),
    (FailureRootCause.DEFORMABLE_MISCLASSIFICATION, _check_deformable),
    (FailureRootCause.OCCLUSION_MISREAD, _check_occlusion),
)


def classify_symptoms(
    final_outcome: str,
    extra: Mapping[str, Any],
    *,
    attempt_id: str = "<unnamed>",
) -> TaxonomyVerdict:
    """The rule table applied to the two things it reads, without a record around them.

    Why this exists as a seam: ``rl.dataset.derive_outcome_class`` stratifies on the same failure
    axis and, until 2026-09-10, decided it from a disjoint set of fields
    (``extra.failure_taxonomy_class`` and ``extra.expected_root_cause``). Neither reader could see
    what the other keyed on, so one record was ``collision_rejection`` here and ``unclassified``
    there. The alternative to this seam was a second copy of the rule table, which is the shape that
    produced the divergence in the first place.

    The flow is one-way on purpose. This classifier still does not read
    ``extra.failure_taxonomy_class``: that field is the producer's own label (the sim runners stamp
    it from ground truth), and a classifier whose coverage number counts rows where the producer
    handed it the answer is measuring the producer, not itself.
    """

    matches: list[FailureRootCause] = []
    fired_evidence: dict[str, bool] = {}
    for cause, predicate in _RULES:
        try:
            fired = bool(predicate(final_outcome, extra))
        except Exception as exc:  # pragma: no cover (defensive)
            raise RuntimeError(
                f"taxonomy rule {cause.value!r} raised on attempt "
                f"{attempt_id!r}: {exc!r}"
            ) from exc
        fired_evidence[cause.value] = fired
        if fired:
            matches.append(cause)

    if not matches:
        return TaxonomyVerdict(
            primary=FailureRootCause.UNCLASSIFIED,
            also_matched=(),
            evidence={
                "final_outcome": final_outcome,
                "rules_fired": fired_evidence,
            },
        )

    primary = matches[0]
    also_matched = tuple(matches[1:])
    return TaxonomyVerdict(
        primary=primary,
        also_matched=also_matched,
        evidence={
            "final_outcome": final_outcome,
            "rules_fired": fired_evidence,
        },
    )


def classify_record(record: GraspAttemptRecord) -> TaxonomyVerdict:
    """Classify a single record deterministically (successes classify as ``UNCLASSIFIED``, excluded from the coverage denominator)."""

    if not isinstance(record, GraspAttemptRecord):
        raise TypeError(
            f"record must be GraspAttemptRecord; got "
            f"{type(record).__name__}"
        )

    return classify_symptoms(
        record.final_outcome,
        record.extra or {},
        attempt_id=record.attempt_id,
    )


@dataclass(frozen=True, slots=True)
class TaxonomyReport:
    """Aggregate report over a sequence of classified records."""

    taxonomy_version: int
    total_records: int
    success_count: int
    failure_count: int
    classified_failure_count: int
    unclassified_failure_count: int
    coverage_fraction: float
    counts_by_cause: Mapping[str, int]
    recommendations: tuple[Mapping[str, Any], ...]
    pack_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.taxonomy_version != TAXONOMY_VERSION:
            raise ValueError(
                "taxonomy_version mismatch; got "
                f"{self.taxonomy_version!r}, expected {TAXONOMY_VERSION!r}"
            )
        if self.total_records < 0:
            raise ValueError("total_records must be non-negative")
        if self.success_count + self.failure_count != self.total_records:
            raise ValueError(
                "success_count + failure_count must equal total_records"
            )
        if (
            self.classified_failure_count + self.unclassified_failure_count
            != self.failure_count
        ):
            raise ValueError(
                "classified + unclassified must equal failure_count"
            )
        if not (0.0 <= self.coverage_fraction <= 1.0):
            raise ValueError(
                f"coverage_fraction out of range: {self.coverage_fraction!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, byte-stable dict representation.

        ``counts_by_cause`` is read back by the adaptation planner's taxonomy rules
        (:mod:`src.robot.grasping.replay.adaptation`): keep the key and its flat
        ``cause -> int`` shape.
        """

        return {
            "taxonomy_version": int(self.taxonomy_version),
            "total_records": int(self.total_records),
            "success_count": int(self.success_count),
            "failure_count": int(self.failure_count),
            "classified_failure_count": int(self.classified_failure_count),
            "unclassified_failure_count": int(
                self.unclassified_failure_count
            ),
            "coverage_fraction": float(self.coverage_fraction),
            "counts_by_cause": {
                k: int(v) for k, v in sorted(self.counts_by_cause.items())
            },
            "recommendations": [
                dict(entry) for entry in self.recommendations
            ],
            "pack_paths": list(self.pack_paths),
        }


def build_taxonomy_report(
    records: Iterable[GraspAttemptRecord],
    *,
    pack_paths: Sequence[str] = (),
) -> TaxonomyReport:
    """Classify ``records`` and assemble the aggregate report (coverage_fraction = classified/failure, or 1.0 when there are no failures)."""

    counts: dict[FailureRootCause, int] = {c: 0 for c in ROOT_CAUSE_ORDER}
    success_count = 0
    failure_count = 0
    total = 0
    for record in records:
        total += 1
        if record.final_outcome == _SUCCESS_OUTCOME:
            success_count += 1
            continue
        failure_count += 1
        verdict = classify_record(record)
        counts[verdict.primary] += 1

    classified_failure = sum(
        v for k, v in counts.items() if k is not FailureRootCause.UNCLASSIFIED
    )
    unclassified_failure = counts[FailureRootCause.UNCLASSIFIED]

    if failure_count > 0:
        coverage = float(classified_failure) / float(failure_count)
    else:
        coverage = 1.0

    # Build recommendations list ordered by (count desc, enum order).
    rec_entries: list[Mapping[str, Any]] = []
    for cause in ROOT_CAUSE_ORDER:
        if cause is FailureRootCause.UNCLASSIFIED:
            continue
        cnt = counts[cause]
        if cnt <= 0:
            continue
        rec_entries.append(
            {
                "root_cause": cause.value,
                "count": int(cnt),
                "recommendation": RECOMMENDATIONS[cause],
            }
        )
    rec_entries.sort(
        key=lambda e: (
            -int(e["count"]),
            ROOT_CAUSE_ORDER.index(FailureRootCause(e["root_cause"])),
        )
    )

    counts_by_cause = {cause.value: counts[cause] for cause in ROOT_CAUSE_ORDER}

    # Aggregated over the whole record stream, never per record.
    ranked = ", ".join(
        f"{cause}={n}" for cause, n in sorted(
            ((c, n) for c, n in counts_by_cause.items() if n), key=lambda kv: -kv[1]
        )
    )
    log = logger.warning if unclassified_failure else logger.info
    log(
        "Classified %d failure(s) of %d record(s): %.1f%% covered, %d unclassified "
        "[%s]",
        failure_count,
        total,
        coverage * 100.0,
        unclassified_failure,
        ranked or "none",
    )
    return TaxonomyReport(
        taxonomy_version=TAXONOMY_VERSION,
        total_records=total,
        success_count=success_count,
        failure_count=failure_count,
        classified_failure_count=classified_failure,
        unclassified_failure_count=unclassified_failure,
        coverage_fraction=coverage,
        counts_by_cause=counts_by_cause,
        recommendations=tuple(rec_entries),
        pack_paths=tuple(pack_paths),
    )


def render_report_json(report: TaxonomyReport) -> str:
    """Return the byte-stable JSON payload for ``report`` (sorted keys + fixed indent)."""

    return json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n"


def write_report(report: TaxonomyReport, out_path: Path) -> Path:
    """Write the report to ``out_path`` (creating parents)."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = render_report_json(report)
    out_path.write_text(body, encoding="utf-8")
    logger.info(
        "Wrote taxonomy report to %s (%d bytes)", out_path, len(body.encode("utf-8"))
    )
    return out_path


__all__ = [
    "TAXONOMY_VERSION",
    "FailureRootCause",
    "ROOT_CAUSE_ORDER",
    "RECOMMENDATIONS",
    "TaxonomyVerdict",
    "TaxonomyReport",
    "classify_record",
    "classify_symptoms",
    "build_taxonomy_report",
    "render_report_json",
    "write_report",
]
