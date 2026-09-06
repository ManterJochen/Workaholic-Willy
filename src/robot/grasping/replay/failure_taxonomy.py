"""Replay-only failure taxonomy and recommendation engine.

This module is read-only with respect to the runtime. It operates
offline on :class:`GraspAttemptRecord` instances (typically loaded from
canonical JSONL packs) and produces a deterministic taxonomy verdict
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

* The JSON report is byte-stable: keys are sorted and lists are
  ordered by impact (count desc, then enum order tie-break).
"""

from __future__ import annotations

import hashlib
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
    iter_jsonl,
)

#: The CLI records the coverage number; what only exists here is which root cause
#: dominated and how many failures the classifier could not name at all: the two
#: facts that decide where the next fix goes.
logger = create_grasping_logger("FailureTaxonomy", REPLAY_FAILURE_TAXONOMY_LOG_FILE)


#: Bump on any contract-level change to taxonomy semantics,
#: recommendation wording, or report schema. Do not bump for
#: pure additions that preserve old behaviour byte-identically.
TAXONOMY_VERSION: int = 1


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
        "verify scene occupancy from the multi-view fusion volume "
        "before re-enabling AUTO mode."
    ),
    FailureRootCause.CALIBRATION_DRIFT_SUSPECTED: (
        "Trigger the ETH/EIH recalibration workflow and freeze AUTO "
        "execution until the drift watchdog reports a clean residual."
    ),
    FailureRootCause.SLIP_AFTER_GRASP: (
        "Review gripper torque/friction profile and tighten the "
        "post-close verification residual; consider enabling the "
        "regrasp recovery policy."
    ),
    FailureRootCause.EMPTY_AIR_GRASP: (
        "Inspect segmentation mask confidence and depth_confidence "
        "floor; review the approach-point lift-off offset to avoid "
        "closing on empty space."
    ),
    FailureRootCause.DEFORMABLE_MISCLASSIFICATION: (
        "Audit the deformable-flag detector and widen profile "
        "coverage; consider DENSE_AUTONOMOUS escalation for "
        "deformable scenes."
    ),
    FailureRootCause.OCCLUSION_MISREAD: (
        "Increase active_perception max attempts, lower the commit "
        "gate corridor radius, and require multi-view fusion before "
        "committing in cluttered scenes."
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
        "refinement_diverged",
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


def _extra_flag(record: GraspAttemptRecord, key: str) -> bool:
    extra = record.extra or {}
    value = extra.get(key)
    return value is True


def _is_failure(record: GraspAttemptRecord) -> bool:
    return (
        record.final_outcome != _SUCCESS_OUTCOME
        and record.final_outcome in _FAILURE_GATING_OUTCOMES
    )


def _check_collision(record: GraspAttemptRecord) -> bool:
    if not _is_failure(record):
        return False
    if _extra_flag(record, "collision_evidence"):
        return True
    if record.final_outcome == "unsafe_recovery_refused":
        return True
    return False


def _check_calibration_drift(record: GraspAttemptRecord) -> bool:
    if not _is_failure(record):
        return False
    return _extra_flag(record, "calibration_drift_evidence")


def _check_slip(record: GraspAttemptRecord) -> bool:
    if record.final_outcome != "verification_failed":
        return False
    return _extra_flag(record, "slip_evidence")


def _check_empty_air(record: GraspAttemptRecord) -> bool:
    if record.final_outcome != "verification_failed":
        return False
    return _extra_flag(record, "empty_air_evidence")


def _check_deformable(record: GraspAttemptRecord) -> bool:
    if not _is_failure(record):
        return False
    return _extra_flag(record, "deformable_misclass_evidence")


def _check_occlusion(record: GraspAttemptRecord) -> bool:
    if not _is_failure(record):
        return False
    return _extra_flag(record, "occlusion_misread_evidence")


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


def classify_record(record: GraspAttemptRecord) -> TaxonomyVerdict:
    """Classify a single record deterministically (successes classify as ``UNCLASSIFIED``, excluded from the coverage denominator)."""

    if not isinstance(record, GraspAttemptRecord):
        raise TypeError(
            f"record must be GraspAttemptRecord; got "
            f"{type(record).__name__}"
        )

    matches: list[FailureRootCause] = []
    fired_evidence: dict[str, bool] = {}
    for cause, predicate in _RULES:
        try:
            fired = bool(predicate(record))
        except Exception as exc:  # pragma: no cover (defensive)
            raise RuntimeError(
                f"taxonomy rule {cause.value!r} raised on attempt "
                f"{record.attempt_id!r}: {exc!r}"
            ) from exc
        fired_evidence[cause.value] = fired
        if fired:
            matches.append(cause)

    if not matches:
        return TaxonomyVerdict(
            primary=FailureRootCause.UNCLASSIFIED,
            also_matched=(),
            evidence={
                "final_outcome": record.final_outcome,
                "rules_fired": fired_evidence,
            },
        )

    primary = matches[0]
    also_matched = tuple(matches[1:])
    return TaxonomyVerdict(
        primary=primary,
        also_matched=also_matched,
        evidence={
            "final_outcome": record.final_outcome,
            "rules_fired": fired_evidence,
        },
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
        """Return a JSON-safe, byte-stable dict representation."""

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
    "build_taxonomy_report",
    "render_report_json",
    "write_report",
    "LABELED_PACK_RELATIVE_PATH",
    "LABELED_PACK_MANIFEST_RELATIVE_PATH",
    "generate_labeled_records",
    "render_labeled_pack_jsonl",
    "write_labeled_pack",
    "build_labeled_pack_manifest",
    "write_labeled_pack_manifest",
    "load_labeled_pack",
]


# Labeled pack: synthetic, deterministic on-disk ground truth. Mirrors the
# canonical-pack pattern (byte-stable JSONL + sha256 manifest) but uses a
# separate sidecar manifest so the canonical MANIFEST.json bytes stay frozen.

#: Manifest version for the taxonomy-labeled pack. Independent of the
#: canonical-pack manifest version.
LABELED_PACK_MANIFEST_VERSION: int = 1

LABELED_PACK_RELATIVE_PATH: str = (
    "tests/data/replay/replay_failure_taxonomy_v1.jsonl"
)
LABELED_PACK_MANIFEST_RELATIVE_PATH: str = (
    "tests/data/replay/MANIFEST_failure_taxonomy.json"
)
LABELED_PACK_NAME: str = "replay_failure_taxonomy_v1"


def _labeled_record(
    *,
    index: int,
    final_outcome: str,
    expected: FailureRootCause,
    extras_extra: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
    multi_evidence: Sequence[str] = (),
) -> GraspAttemptRecord:
    """Construct one deterministic labeled record for the on-disk pack (``expected`` is ground truth; the classifier never reads it)."""

    extra: dict[str, Any] = {
        "cycle_time_s": 2.0 + (index % 5) * 0.1,
        "expected_root_cause": expected.value,
    }
    for key in multi_evidence:
        extra[key] = True
    extra.update(dict(extras_extra))
    return GraspAttemptRecord.new(
        timestamp=float(index),
        attempt_id=f"{LABELED_PACK_NAME}-{index:06d}",
        mode="auto",
        final_outcome=final_outcome,
        extra=extra,
    )


#: Static recipe driving :func:`generate_labeled_records`. Each entry
#: is ``(count, final_outcome, expected_cause, symptom_flags)``.
#:
#: Coverage target: classified / failures >= 0.95.
_LABELED_RECIPE: tuple[tuple[int, str, FailureRootCause, tuple[str, ...]], ...] = (
    # Successes (excluded from coverage denominator).
    (5, "succeeded", FailureRootCause.UNCLASSIFIED, ()),
    # Collision rejections.
    (
        8,
        "execution_failed",
        FailureRootCause.COLLISION_REJECTION,
        ("collision_evidence",),
    ),
    (
        4,
        "unsafe_recovery_refused",
        FailureRootCause.COLLISION_REJECTION,
        (),
    ),
    # Calibration drift.
    (
        10,
        "decision_fail_closed",
        FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
        ("calibration_drift_evidence",),
    ),
    # Slip after grasp.
    (
        10,
        "verification_failed",
        FailureRootCause.SLIP_AFTER_GRASP,
        ("slip_evidence",),
    ),
    # Empty air.
    (
        10,
        "verification_failed",
        FailureRootCause.EMPTY_AIR_GRASP,
        ("empty_air_evidence",),
    ),
    # Deformable misclassification.
    (
        8,
        "execution_failed",
        FailureRootCause.DEFORMABLE_MISCLASSIFICATION,
        ("deformable_misclass_evidence",),
    ),
    # Occlusion misread.
    (
        8,
        "no_valid_grasp",
        FailureRootCause.OCCLUSION_MISREAD,
        ("occlusion_misread_evidence",),
    ),
    # Honest UNCLASSIFIED failures (no symptom flag set).
    (2, "execution_failed", FailureRootCause.UNCLASSIFIED, ()),
)

#: Multi-match precedence cases: these set multiple symptom flags so
#: the classifier must resolve via enum order. The ``expected_cause``
#: is the highest-precedence match.
_LABELED_MULTI_MATCH: tuple[
    tuple[str, FailureRootCause, tuple[str, ...]], ...
] = (
    # Collision wins over calibration_drift.
    (
        "execution_failed",
        FailureRootCause.COLLISION_REJECTION,
        (
            "collision_evidence",
            "calibration_drift_evidence",
        ),
    ),
    # Calibration drift wins over deformable.
    (
        "execution_failed",
        FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
        (
            "calibration_drift_evidence",
            "deformable_misclass_evidence",
        ),
    ),
    # Slip wins over empty_air when both fire on verification_failed.
    (
        "verification_failed",
        FailureRootCause.SLIP_AFTER_GRASP,
        ("slip_evidence", "empty_air_evidence"),
    ),
)


def generate_labeled_records() -> tuple[GraspAttemptRecord, ...]:
    """Deterministically materialise the labeled taxonomy pack."""

    records: list[GraspAttemptRecord] = []
    index = 0
    for count, outcome, expected, flags in _LABELED_RECIPE:
        for _ in range(int(count)):
            records.append(
                _labeled_record(
                    index=index,
                    final_outcome=outcome,
                    expected=expected,
                    multi_evidence=flags,
                )
            )
            index += 1
    for outcome, expected, flags in _LABELED_MULTI_MATCH:
        records.append(
            _labeled_record(
                index=index,
                final_outcome=outcome,
                expected=expected,
                multi_evidence=flags,
            )
        )
        index += 1
    return tuple(records)


def render_labeled_pack_jsonl() -> str:
    """Return the byte-stable JSONL payload for the labeled pack."""

    records = generate_labeled_records()
    lines = [record.to_json() for record in records]
    return "\n".join(lines) + ("\n" if lines else "")


def _sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_labeled_pack(repo_root: Path) -> Path:
    """Render the labeled pack and write it to disk."""

    payload = render_labeled_pack_jsonl()
    out_path = (repo_root / LABELED_PACK_RELATIVE_PATH).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    # The sha is the contract the sidecar manifest is checked against later.
    logger.info(
        "Wrote labeled taxonomy pack to %s (%d bytes, sha256 %s)",
        out_path,
        len(payload.encode("utf-8")),
        _sha256_hex(payload)[:16],
    )
    return out_path


def build_labeled_pack_manifest(repo_root: Path) -> dict[str, Any]:
    """Build the sidecar manifest for the labeled pack."""

    abs_path = (repo_root / LABELED_PACK_RELATIVE_PATH).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(
            f"labeled taxonomy pack missing on disk: {abs_path}"
        )
    raw = abs_path.read_bytes()
    records = tuple(iter_jsonl(abs_path))
    return {
        "manifest_version": int(LABELED_PACK_MANIFEST_VERSION),
        "taxonomy_version": int(TAXONOMY_VERSION),
        "label_policy": (
            "extra.expected_root_cause carries ground truth; "
            "classifier reads only extra.*_evidence flags and "
            "final_outcome. Successes excluded from coverage "
            "denominator."
        ),
        "pack": {
            "name": LABELED_PACK_NAME,
            "path": LABELED_PACK_RELATIVE_PATH,
            "capability_group": "failure_taxonomy",
            "record_count": len(records),
            "sha256": _sha256_hex(raw),
            "generator": {
                "module": "src.robot.grasping.replay.failure_taxonomy",
                "callable": "generate_labeled_records",
            },
        },
    }


def write_labeled_pack_manifest(repo_root: Path) -> Path:
    """Write the sidecar manifest to disk."""

    manifest = build_labeled_pack_manifest(repo_root)
    out_path = (repo_root / LABELED_PACK_MANIFEST_RELATIVE_PATH).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out_path.write_text(body, encoding="utf-8")
    logger.info(
        "Wrote labeled-pack manifest to %s (%d bytes)",
        out_path,
        len(body.encode("utf-8")),
    )
    return out_path


def load_labeled_pack(repo_root: Path) -> tuple[GraspAttemptRecord, ...]:
    """Read the on-disk labeled pack back as a tuple of records."""

    abs_path = (repo_root / LABELED_PACK_RELATIVE_PATH).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(
            f"labeled taxonomy pack missing on disk: {abs_path}"
        )
    return tuple(iter_jsonl(abs_path))
