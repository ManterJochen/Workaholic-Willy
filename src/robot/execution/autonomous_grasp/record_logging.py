"""Serializer from ``AutonomousGraspReport`` to ``GraspAttemptRecord``, plus an optional JSONL log.

Maps a typed report into a JSONL-safe record so a deployment can opt into logging real pick attempts,
which the soak, ``--records-gate`` and RL layers then consume. Without this bridge the
``safety_rejection_rate`` and ``false_positive_grasp_rate`` KPIs have no production source and read
``0.0``.

Default-off: the serializer is pure; nothing logs unless the operator wires a path (see :func:`log_record`),
so a cell that wires none is byte-identical.

KPI sourcing:
* ``extra["safety_rejected"]`` is set from the fail-closed outcome family (``decision_fail_closed`` /
  ``uncertainty_fail_closed`` / ``drift_blocked_auto`` / ``ood_blocked_auto`` / ``missing_camera_frame`` /
  ``unsafe_recovery_refused`` / ``no_commit_insufficient_fusion``), the live "auto/safety refused to
  dispatch" signal that fires on a real run. The per-step ``TrajectoryStepReport.safety_rejected``
  predicate is dead in production: its sole caller omits ``safety_check`` entirely, so both validators
  fall back to the no-op ``AcceptAllTrajectorySafetyCheck`` default, and the validator is dense-only
  and inert, so it is deliberately not the source here.
* ``extra["verification_failed_after_success"]`` (the ``false_positive_grasp_rate`` source) is left unset
  on purpose. The KPI counts it only on a row whose ``final_outcome == "succeeded"``, meaning an attempt
  that reported success and was later found false by an independent re-verification. The pick path has no
  such secondary detector, and ``VERIFICATION_FAILED`` is a distinct terminal outcome that never co-occurs
  with ``SUCCEEDED``, so there is no honest signal to write. The key stays wired: the KPI reads it the
  moment a false-positive detector (a post-grasp re-check) is added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from src.robot.constants import GRASP_RECORD_LOG_FILE, create_robot_logger
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    execution_metadata_from,
    profile_metadata_from,
    verification_metadata_from,
)

from .report import AutonomousGraspOutcome


# One line per pick attempt, never per candidate. The JSONL file is the machine-readable record; this
# is the human-readable trail that says the record actually reached disk, which is what stops the KPI
# layer from reading 0.0 because nothing was written.
logger = create_robot_logger("GraspRecordLogging", GRASP_RECORD_LOG_FILE)

#: Outcomes that mean a fail-closed safety/decision/integrity gate refused to dispatch the grasp. These are
#: the live ``safety_rejection_rate`` signal (an operator reads them as "auto refused to move").
SAFETY_REJECTED_OUTCOMES: frozenset[AutonomousGraspOutcome] = frozenset(
    {
        AutonomousGraspOutcome.DECISION_FAIL_CLOSED,
        AutonomousGraspOutcome.UNCERTAINTY_FAIL_CLOSED,
        AutonomousGraspOutcome.DRIFT_BLOCKED_AUTO,
        AutonomousGraspOutcome.OOD_BLOCKED_AUTO,
        AutonomousGraspOutcome.MISSING_CAMERA_FRAME,
        AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
        AutonomousGraspOutcome.NO_COMMIT_INSUFFICIENT_FUSION,
    }
)


def _outcome_value(outcome: Any) -> str:
    return outcome.value if hasattr(outcome, "value") else str(outcome)


#: Map the calculator's typed ``GraspFailureReason`` value to the offline failure-taxonomy
#: ``extra.*_evidence`` flag it evidences (pure typed labeling, no numeric thresholds).
_REASON_TO_EVIDENCE: Mapping[str, str] = {
    "all_collided": "collision_evidence",
    "heavy_occlusion": "occlusion_misread_evidence",
    "deformable_routing_required": "deformable_misclass_evidence",
}
#: Map the ``GraspVerifier`` verdict reason (verification-failed family) to the evidence flag.
_VERIFICATION_REASON_TO_EVIDENCE: Mapping[str, str] = {
    "jaws_collapsed_to_minimum": "empty_air_evidence",
    "gripper_object_not_detected": "empty_air_evidence",
    "target_still_visible": "slip_evidence",
}


def _stamp_taxonomy_evidence(record_extra: dict[str, Any], report: Any) -> None:
    """Derive the offline failure-taxonomy ``extra.*_evidence`` flags from the runtime's own typed
    verdicts (the calculator ``GraspFailureReason``s across the attempt history plus the post-grasp
    ``verification_reason`` already in telemetry). A flag is set only when its typed cause is present on a
    failed record; absent causes leave the record byte-identical, so the default open-loop pick (no typed
    failure cause) logs no flag. Collision, occlusion and deformable flags light up under the dense path,
    slip and empty-air under verification, when their detector fires.

    ``calibration_drift_evidence`` is intentionally not derived: the drift/OOD watchdog is inert by
    default and its ``drift_blocked_auto``/``ood_blocked_auto`` outcomes sit outside the taxonomy gating
    set, so a stamped flag could never classify. Deferred until the watchdog is wired."""

    if _outcome_value(report.outcome) == "succeeded":
        return  # a recovered-success record carries no failure-evidence flag
    # (1) calculator typed reasons, collected across the full attempt history.
    pick_report = getattr(report, "pick_report", None)
    for attempt in getattr(pick_report, "attempts", ()) or ():
        for reason in getattr(attempt, "reasons", ()) or ():
            key = _REASON_TO_EVIDENCE.get(getattr(reason, "value", None) or str(reason))
            if key is not None:
                record_extra[key] = True
    # (2) post-grasp verification verdict reason (already stamped into telemetry, so into record_extra).
    vkey = _VERIFICATION_REASON_TO_EVIDENCE.get(str(record_extra.get("verification_reason", "")))
    if vkey is not None:
        record_extra[vkey] = True


def to_attempt_record(
    report: Any,
    *,
    attempt_id: str,
    timestamp: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> GraspAttemptRecord:
    """Serialize an ``AutonomousGraspReport`` into a JSONL-safe :class:`GraspAttemptRecord`.

    Duck-typed on the report: it reads ``outcome`` / ``mode`` / ``profile`` / ``telemetry``. The report's
    free-form ``telemetry`` bag is carried verbatim into ``extra``, then the ``safety_rejected`` flag is
    stamped on top (see the module docstring for the KPI contract).

    ``extra`` is an optional caller-supplied bag merged on top of the telemetry-derived fields. It is how
    a sim runner stamps ground-truth labels the pipeline cannot self-report (``sim_lift_mm`` /
    ``sim_lifted`` measured from the object's world pose), so the records carry a reward signal for the RL
    layer instead of only the pipeline's own ``final_outcome`` self-assessment. Default ``None`` merges
    nothing and leaves behaviour byte-identical.
    """

    # The writer for `initial_telemetry`, a field the record contract, the serialiser and the reader
    # all carry. It holds the calculator's own telemetry, including the `deep_ranker_*` shadow keys
    # the learned ranker stamps, so the learned ranker's shadow can be measured from a record.
    # Empty when nothing executed, so a run that reaches no grasp is byte-identical.
    initial_telemetry = dict(
        getattr(getattr(report, "pick_report", None), "calculator_telemetry", None) or {})
    record_extra: dict[str, Any] = dict(getattr(report, "telemetry", {}) or {})
    record_extra["safety_rejected"] = report.outcome in SAFETY_REJECTED_OUTCOMES
    if extra:
        record_extra.update(extra)
    # Derive the offline failure-taxonomy ``extra.*_evidence`` flags from the report's typed verdicts.
    _stamp_taxonomy_evidence(record_extra, report)
    # Retain the executed grasp's success-model feature vector (23-d) when the shadow success predictor
    # scored it, so a logged record carries the (features, outcome) pair the offline success-model trainer
    # learns from (calibration.success_model_calibration.build_dataset_from_records). Absent means not
    # stamped and byte-identical; a caller-supplied override wins.
    shadow_features = getattr(getattr(report, "shadow_success_telemetry", None), "features", None)
    if shadow_features is not None and "success_model_features" not in record_extra:
        record_extra["success_model_features"] = [float(v) for v in shadow_features]
    mode = getattr(report, "mode", "")
    mode_str = mode.value if hasattr(mode, "value") else str(mode)
    # Carry the per-step recovery trail into the record's top-level ``recovery_actions`` field (the offline
    # train_recovery source). Empty () on the non-recovery path, so that path is byte-identical. Not in
    # ``extra``, since it is an existing top-level field, so the frozen extra/catalog policy is untouched.
    recovery_actions = tuple(getattr(report, "recovery_actions", ()) or ())
    # Populate the execution and verification telemetry blocks the GraspAttemptRecord contract requires
    # for succeeded/execution_failed/verification_failed outcomes (the soak telemetry audit checks them),
    # from the report itself. execution is the executed-grasp outcome. ``AutonomousGraspReport`` carries
    # no ``verification`` field and its slots forbid one, so the duck-typed lookup is a seam for a future
    # report that does carry one and never fires today; verification is the sim ground-truth lift,
    # explicitly labelled, when the runner stamped it in extra (sim_lifted/sim_lift_mm measured from the
    # object's world pose, the physical truth, not a hardware verifier). None when neither exists, so the
    # audit flags an incomplete record.
    execution = execution_metadata_from(report)
    verification = verification_metadata_from(getattr(report, "verification", None))
    if verification is None and "sim_lifted" in record_extra:
        verification = {
            "source": "sim_ground_truth_lift",
            "lifted": bool(record_extra.get("sim_lifted")),
            "lift_mm": record_extra.get("sim_lift_mm"),
            "max_distractor_lift_mm": record_extra.get("sim_max_distractor_lift_mm"),
        }
    return GraspAttemptRecord.new(
        attempt_id=attempt_id,
        mode=mode_str,
        final_outcome=_outcome_value(report.outcome),
        timestamp=timestamp,
        profile=profile_metadata_from(getattr(report, "profile", None)),
        recovery_actions=recovery_actions,
        execution=execution,
        verification=verification,
        # Kept out of `extra`: it is an existing top-level field, and moving calculator telemetry
        # into the extra bag would change what the frozen extra/catalog policy covers.
        initial_telemetry=initial_telemetry,
        extra=record_extra,
    )


def log_record(
    report: Any,
    *,
    attempt_id: str,
    log_path: str | Path,
    timestamp: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> GraspAttemptRecord:
    """Append the serialized record as one JSON line to ``log_path`` (creating parent dirs). Returns the
    record. This is the opt-in production logging seam: a deployment calls it after a pick, and with no
    ``log_path`` wired nothing is written and behaviour is byte-identical. ``extra`` is forwarded to
    :func:`to_attempt_record` (ground-truth-label merge; default ``None`` is byte-identical)."""

    record = to_attempt_record(report, attempt_id=attempt_id, timestamp=timestamp, extra=extra)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    logger.info(
        "attempt %s (%s) appended to %s: %d bytes written, file now %d bytes",
        attempt_id, record.final_outcome, path, len(line.encode("utf-8")), path.stat().st_size,
    )
    return record


__all__ = [
    "SAFETY_REJECTED_OUTCOMES",
    "to_attempt_record",
    "log_record",
]
