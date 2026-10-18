"""Frozen telemetry-completeness catalog.

Every supported :class:`AutonomousGraspOutcome` value maps to the set
of :class:`GraspAttemptRecord` fields that must be populated for the
record to count as "incident-grade". A retired outcome keeps its entry,
by its string, so a record logged while it existed still audits.

A required field name may reference a top-level record attribute
(``"execution"``, ``"recovery_actions"``, etc.)
or a key inside ``record.extra`` using the ``"extra.<key>"`` dotted
form. The audit treats both forms uniformly.

Universally required fields (``timestamp``, ``attempt_id``, ``mode``,
``final_outcome``) are validated by
:class:`GraspAttemptRecord.__post_init__` itself, so the catalog only
enumerates the additional fields required per outcome.

The extra-telemetry extension layer adds a frozen catalog of optional
telemetry fields that runtime capabilities (multi-view fusion, the failure
taxonomy, the attempt's wall time, the camera world) populate in ``record.extra``. They are
intentionally not required by the strict :func:`audit_record` gate; a
field is promoted to "required" incrementally, one capability at a time.
The :func:`audit_extra_record` helper performs a type-only validation
when the fields are present, so canonical datasets stay valid as new
content lands.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


_O = AutonomousGraspOutcome


TELEMETRY_CATALOG: dict[str, frozenset[str]] = {
    # ``verification`` left both requirements below on 2026-09-29 with the post-grasp verification
    # stage: no live pick writes the block (a simulator's ground-truth lift is its only writer), and
    # the hold is the execution policy's own post-close check, whose verdict is the outcome itself
    # (an empty close ends as ``verification_failed``). A record logged before then that carries the
    # verifier's block audits as it did: a present field is never a missing one.
    _O.SUCCEEDED.value: frozenset({"execution"}),
    _O.NO_TARGET.value: frozenset(),
    # A cancelled attempt requires nothing: it stopped before it could produce an execution block, and
    # demanding one would make every operator stop look like a malformed record.
    _O.CANCELLED.value: frozenset(),
    # Also what a pick the retired relocate path ended (``relocated_exhausted``, removed
    # 2026-09-29) was reported as, so such a record still audits here.
    _O.NO_VALID_GRASP.value: frozenset(),
    _O.EXECUTION_FAILED.value: frozenset({"execution"}),
    _O.VERIFICATION_FAILED.value: frozenset({"execution"}),
    _O.RECOVERY_EXHAUSTED.value: frozenset({"recovery_actions"}),
    _O.UNSAFE_RECOVERY_REFUSED.value: frozenset({"recovery_actions"}),
    _O.MISSING_CAMERA_FRAME.value: frozenset(),
    # Retired: the two-scan pre-grasp refinement that ended a pick these three ways was removed on
    # 2026-09-29 with the closed_loop and dense_autonomous modes, and no pick reports them since. Kept
    # by their strings, each requiring the refinement block exactly as it did, so a record logged
    # before then still audits; without them that record would read as an unknown outcome.
    "refinement_failed": frozenset({"refinement"}),
    "target_lost_during_refine": frozenset({"refinement"}),
    "refinement_diverged": frozenset({"refinement"}),
    _O.MODE_NOT_AVAILABLE.value: frozenset(),
    # Retired: the decision and uncertainty layers that ended a pick these three ways left on 2026-10-16 (the
    # owner's cleanup), and no pick reports them since. Kept by their strings, each requiring what it did, so a
    # record logged before then still audits; without them that record would read as an unknown outcome. The
    # decision entries require their keys, never a value: a record whose action or reason code was retired on
    # 2026-09-29 (the camera re-observation's ``move_camera`` with its ``low_confidence``,
    # ``reobserve_budget_exhausted`` and ``reobserve_planner_unavailable``) audits as it did.
    "decision_fail_closed": frozenset(
        {
            "extra.decision_reason_code",
            "extra.uncertainty_score",
            "extra.threshold_used",
        }
    ),
    "decision_recover_pending": frozenset(
        {"extra.decision_reason_code"}
    ),
    "uncertainty_fail_closed": frozenset(
        {
            "extra.decision_reason_code",
            "extra.uncertainty_score",
            "extra.threshold_used",
        }
    ),
    # Retired: the multi-view commit gate that ended a pick this way was removed on 2026-09-28, and
    # no pick reports it since. Kept by its string so a record logged before then still audits, and
    # requires nothing, exactly as it did; without it that record would read as an unknown outcome.
    "no_commit_insufficient_fusion": frozenset(),
    # Retired with the drift / OOD watchdog on 2026-10-16. Kept by their strings, each requiring the watchdog
    # telemetry it did, so a record logged before then still audits.
    "drift_blocked_auto": frozenset(
        {
            "extra.decision_reason_code",
            "extra.drift_severity",
            "extra.degraded_mode_active",
        }
    ),
    "ood_blocked_auto": frozenset(
        {
            "extra.decision_reason_code",
            "extra.ood_flagged",
            "extra.degraded_mode_active",
        }
    ),
}


def _field_present(record: GraspAttemptRecord, name: str) -> bool:
    if name.startswith("extra."):
        key = name[len("extra.") :]
        extra = record.extra or {}
        return key in extra and extra[key] is not None
    value = getattr(record, name, None)
    if value is None:
        return False
    # Empty sequences (e.g. recovery_actions=()) count as missing.
    if isinstance(value, (tuple, list)) and len(value) == 0:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False
    return True


def audit_record(record: GraspAttemptRecord) -> tuple[str, ...]:
    """Return the names of required fields that are missing (empty tuple == record passes the audit)."""

    required = TELEMETRY_CATALOG.get(record.final_outcome)
    if required is None:
        # Unknown outcome is itself an audit failure surfaced as the
        # special ``final_outcome`` token so callers can distinguish
        # it from a recognised-but-incomplete record.
        return ("final_outcome:unknown",)
    return tuple(
        sorted(name for name in required if not _field_present(record, name))
    )


def audit_records(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(attempt_id, missing_fields)`` for every offender."""

    offenders: list[tuple[str, tuple[str, ...]]] = []
    for r in records:
        missing = audit_record(r)
        if missing:
            offenders.append((r.attempt_id, missing))
    return offenders


#: Frozen contract version. Bumped only on additive field additions
#: that downstream tools must learn to read. Removing or retyping any
#: field already published here is a breaking change and requires a
#: deliberate, documented schema-version bump on
#: :class:`GraspAttemptRecord` itself.
#:
#: 3 (2026-10-16): the success-probability, uncertainty, drift/OOD, stage-latency, adaptation and RL fields left
#: with the capabilities that wrote them (the owner's cleanup). A record that still carries one is no longer
#: type-checked on it; nothing reads them.
EXTRA_TELEMETRY_VERSION: int = 3


def _is_str_or_none(v: Any) -> bool:
    return v is None or isinstance(v, str)


def _is_bool_or_none(v: Any) -> bool:
    return v is None or isinstance(v, bool)


#: The camera-world uses a record may carry: every spoken use. UNSTATED is never written, because
#: a record that says nothing about the camera world leaves the key out.
_RECORDED_CAMERA_WORLD_USES: frozenset[str] = frozenset({"planned", "declined", "unplanned", "missing"})


def _is_camera_world_use_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in _RECORDED_CAMERA_WORLD_USES)


def _is_finite_number_or_none(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):  # bool is a subclass of int; reject here
        return False
    if not isinstance(v, (int, float)):
        return False
    f = float(v)
    return f == f and f not in (float("inf"), float("-inf"))


def _is_unit_interval_or_none(v: Any) -> bool:
    if not _is_finite_number_or_none(v):
        return False
    return v is None or 0.0 <= float(v) <= 1.0


def _is_non_negative_int_or_none(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    return isinstance(v, int) and v >= 0


def _is_non_negative_number_or_none(v: Any) -> bool:
    if not _is_finite_number_or_none(v):
        return False
    return v is None or float(v) >= 0.0


#: Per-field validator + capability-group tag. The tag is informational
#: today; a future strict gate may reference it when promoting a field
#: to required. Keep insertion order stable: the catalog is treated as
#: an immutable, ordered contract by :func:`audit_extra_record`.
EXTRA_TELEMETRY_FIELDS: tuple[tuple[str, str, Any], ...] = (
    # multi-view fusion
    # `_fusion_geometry_extras` joins the geometry fusion telemetry stamped on
    # `orchestrator._fusion_geometry_telemetry` to these catalog names, which
    # declare types and not meanings: view_count = cameras that contributed a
    # usable view; evidence_quality = mean association score over the pairs that
    # won their assignment; occlusion_reduced = whether any object gained
    # surface from another view.
    ("fused_view_count", "multi_view_fusion", _is_non_negative_int_or_none),
    ("fusion_evidence_quality", "multi_view_fusion", _is_unit_interval_or_none),
    ("multi_view_occlusion_reduced", "multi_view_fusion", _is_bool_or_none),
    # failure taxonomy + recommendations
    ("failure_taxonomy_class", "failure_taxonomy", _is_str_or_none),
    ("failure_recommendation_id", "failure_taxonomy", _is_str_or_none),
    # the attempt's wall time
    ("attempt_wall_time_s", "latency", _is_non_negative_number_or_none),
    # camera world. The weakest camera world behind the attempt's typed
    # motions, read by the record serializer from the pick report's stamps.
    # Absent when no typed motion was commanded or every stamp is UNSTATED.
    # The reason is "" on ``planned``, which declines nothing; the cameras
    # and the capture time stay on the report.
    ("camera_world", "camera_world", _is_camera_world_use_or_none),
    ("camera_world_reason", "camera_world", _is_str_or_none),
)


def extra_field_names() -> tuple[str, ...]:
    """Return the ordered tuple of extra-bag telemetry field names."""

    return tuple(name for name, _group, _validator in EXTRA_TELEMETRY_FIELDS)


def extra_field_group_map() -> dict[str, str]:
    """Return ``{field_name: capability_group}`` for every extra telemetry field."""

    return {name: group for name, group, _validator in EXTRA_TELEMETRY_FIELDS}


def audit_extra_record(record: GraspAttemptRecord) -> tuple[str, ...]:
    """Type-only audit of the extra telemetry fields: return the names whose present value violates the locked type contract (missing/absent fields are not flagged)."""

    extra = record.extra or {}
    offenders: list[str] = []
    for name, _group, validator in EXTRA_TELEMETRY_FIELDS:
        if name not in extra:
            continue
        if not validator(extra[name]):
            offenders.append(name)
    return tuple(offenders)


def audit_extra_records(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(attempt_id, bad_fields)`` for every extra-field type violation."""

    offenders: list[tuple[str, tuple[str, ...]]] = []
    for r in records:
        bad = audit_extra_record(r)
        if bad:
            offenders.append((r.attempt_id, bad))
    return offenders


def extra_field_coverage(
    records: Iterable[GraspAttemptRecord] | Sequence[GraspAttemptRecord],
) -> dict[str, float]:
    """Return ``{field_name: fraction_of_records_with_non_null_value}`` (``0.0`` for every field on empty input)."""

    records = tuple(records)
    total = len(records)
    coverage: dict[str, float] = {}
    for name, _group, _validator in EXTRA_TELEMETRY_FIELDS:
        if total == 0:
            coverage[name] = 0.0
            continue
        present = 0
        for r in records:
            extra = r.extra or {}
            if name in extra and extra[name] is not None:
                present += 1
        coverage[name] = present / total
    return coverage


