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
  ``unsafe_recovery_refused``), the live "auto/safety refused to
  dispatch" signal that fires on a real run. The per-step ``TrajectoryStepReport.safety_rejected``
  predicate is dead in production: its sole caller omits ``safety_check`` entirely, so both validators
  fall back to the no-op ``AcceptAllTrajectorySafetyCheck`` default, and the validator is dense-only
  and inert, so it is deliberately not the source here.
* ``extra["verification_failed_after_success"]`` (the ``false_positive_grasp_rate`` source) is left unset
  on purpose. The KPI counts it only on a row whose ``final_outcome == "succeeded"``, meaning an attempt
  that reported success and was later found false by an independent re-verification. The pick path has no
  such secondary detector, and ``VERIFICATION_FAILED`` (the execution policy's own post-close check found
  the jaws empty) is a distinct terminal outcome that never co-occurs with ``SUCCEEDED``, so there is no
  honest signal to write. The key stays wired: the KPI reads it the moment a false-positive detector (a
  post-grasp re-check) is added.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from src.robot.constants import GRASP_RECORD_LOG_FILE, create_robot_logger
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    execution_metadata_from,
    grasp_metadata_from,
    profile_metadata_from,
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
#: The verifier-reason map that stamped ``empty_air_evidence`` and ``slip_evidence`` left on 2026-09-29
#: with the post-grasp verifiers whose verdicts it read; no pick writes those reasons any more. A
#: record logged before then keeps the flags it was stamped with, and the failure taxonomy still reads
#: them.


def _stamp_taxonomy_evidence(record_extra: dict[str, Any], report: Any) -> None:
    """Derive the offline failure-taxonomy ``extra.*_evidence`` flags from the runtime's own typed
    verdicts (the calculator ``GraspFailureReason``s across the attempt history). A flag is set only when
    its typed cause is present on a failed record; absent causes leave the record byte-identical, so the
    default open-loop pick (no typed failure cause) logs no flag. Collision, occlusion and deformable flags
    light up under the dense path. Slip and empty-air had their detector in the post-grasp verifiers,
    which left on 2026-09-29, so no live record is stamped with them.

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


def _camera_world_extras(report: Any) -> dict[str, str]:
    """The weakest camera world behind the attempt's typed motions, as two record keys.

    Read from ``pick_report.camera_worlds`` through the same :func:`weakest_camera_world` the
    report's ``render`` and ``to_dict`` read, so the record and the report cannot name two different
    stamps. Nothing is written for an attempt that commanded no typed motion or whose weakest stamp
    is UNSTATED, which keeps every unstamped record byte-identical. Only the use and the reason are
    written; the cameras and the capture time stay on the report.

    Never raises. A record is written after the pick, and a stamp this cannot read writes nothing
    rather than losing the record.
    """
    from src.robot.core.camera_world import CameraWorldUse
    from src.robot.grasping.motion.execution_policy import weakest_camera_world

    try:
        stamps = tuple(getattr(getattr(report, "pick_report", None), "camera_worlds", ()) or ())
        weakest = weakest_camera_world(stamps)
        if weakest is None or weakest.use is CameraWorldUse.UNSTATED:
            return {}
        return {"camera_world": str(weakest.use.value), "camera_world_reason": str(weakest.reason)}
    except Exception:  # noqa: BLE001 (an unreadable stamp must not cost the record)
        return {}


def _look_extras(report: Any) -> dict[str, Any]:
    """What a pick's looks came to, as record keys, each only where the report says it.

    ``looks_visited`` (the looks the pick perceived from, the report's ``looks`` less a fixed camera's look the arm did
    not reach), ``looks_fused`` (the look the grasp was ranked on first), ``jaw_faces_seen`` (jaw 1, jaw 2: whether
    each contact face of the chosen grasp was seen), ``hand_eye_gap_mm`` (the median distance at which the looks
    measure the part's shared surface), ``generated_view_deg`` (how far the one generated view of a wrist pick turned
    about the part) and ``both_faces`` (the pick asked for both faces; ``true`` only, on a wrist or a fixed camera):
    ``looks_visited`` and ``both_faces`` aside, a wrist camera's alone. A fixed camera handed looks (by its program, or
    by the cell profile's configured looks) moves to them too, and records the ones it reached. Additive and outside the
    telemetry catalog, so the record of a pick handed no look on a fixed camera is the record it always was. Read in the
    types the report promises, so a report double that answers every attribute adds nothing. The look a wrist pick did
    not reach (``refused_look``) and a move back that was refused (``move_back_refused``) come with the telemetry.
    """
    extras: dict[str, Any] = {}
    looks = getattr(report, "looks", ())
    telemetry = getattr(report, "telemetry", None)
    said = telemetry if isinstance(telemetry, Mapping) else {}
    if isinstance(looks, tuple) and said.get("look_refused") and not said.get("refused_look"):
        # A fixed camera's service loop names the looks it moved to, the one it did not reach last.
        looks = looks[:-1]
    if isinstance(looks, tuple) and looks and all(isinstance(look, str) for look in looks):
        extras["looks_visited"] = list(looks)
    fused = getattr(report, "looks_fused", ())
    if isinstance(fused, tuple) and fused and all(isinstance(look, str) for look in fused):
        extras["looks_fused"] = list(fused)
    faces = getattr(report, "jaw_faces_seen", None)
    if isinstance(faces, tuple) and len(faces) == 2 and all(isinstance(seen, bool) for seen in faces):
        extras["jaw_faces_seen"] = list(faces)
    gap = getattr(report, "hand_eye_gap_mm", None)
    if isinstance(gap, (int, float)) and not isinstance(gap, bool):
        extras["hand_eye_gap_mm"] = float(gap)
    turn = getattr(report, "generated_view_deg", None)
    if isinstance(turn, (int, float)) and not isinstance(turn, bool):
        extras["generated_view_deg"] = float(turn)
    if getattr(report, "both_faces", False) is True:
        extras["both_faces"] = True
    return extras


def _attempt_extras(report: Any) -> dict[str, Any]:
    """Each pick-loop attempt's motion, as record keys (fix plan Track E, RC6): its ``action``, its typed ``reasons``,
    the ``motion_status`` and ``motion_message`` of the motion it ended on, and every grasp it tried
    (``PickAttempt.tries``, the fix plan's contract 3: ``rank``, ``outcome``, ``motion_status``, ``motion_message``,
    ``sent``, ``reached_part``).

    Written only where an attempt says something of a motion, so the record of a pick that commanded none is the
    record it always was. Read in the types the attempt promises; a double that answers every attribute adds nothing.
    """
    attempts = getattr(getattr(report, "pick_report", None), "attempts", ())
    if not isinstance(attempts, tuple):
        return {}
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        status, message = getattr(attempt, "motion_status", None), getattr(attempt, "motion_message", None)
        tries = getattr(attempt, "tries", ())
        tried = [dict(entry) for entry in tries if isinstance(entry, Mapping)] if isinstance(tries, tuple) else []
        if not isinstance(status, str) and not isinstance(message, str) and not tried:
            continue
        reasons = getattr(attempt, "reasons", ())
        rows.append({
            "attempt_index": getattr(attempt, "attempt_index", None) if isinstance(
                getattr(attempt, "attempt_index", None), int) else None,
            "action": str(getattr(attempt, "action", "")),
            "reasons": [getattr(reason, "value", str(reason)) for reason in reasons] if isinstance(reasons, tuple)
            else [],
            "motion_status": status if isinstance(status, str) else None,
            "motion_message": message if isinstance(message, str) else None,
            "tries": tried,
        })
    return {"attempts": rows} if rows else {}


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

    What a pick's looks came to is stamped into ``extra`` beside the telemetry, each key only where the report says
    it (:func:`_look_extras`: ``looks_visited``, ``looks_fused``, ``jaw_faces_seen``, ``hand_eye_gap_mm``,
    ``generated_view_deg``, ``both_faces``).

    ``extra`` is an optional caller-supplied bag merged on top of the telemetry-derived fields. It is how
    a sim runner stamps ground-truth labels the pipeline cannot self-report (``sim_lift_mm`` /
    ``sim_lifted`` measured from the object's world pose), so the records carry a reward signal for the RL
    layer instead of only the pipeline's own ``final_outcome`` self-assessment. Default ``None`` merges
    nothing and leaves behaviour byte-identical. The ``camera_world`` keys are stamped after it, so a
    caller cannot overwrite them.
    """

    # The writer for `initial_telemetry`, a field the record contract, the serialiser and the reader
    # all carry. It holds the calculator's own telemetry, including the `deep_ranker_*` shadow keys
    # the learned ranker stamps, so the learned ranker's shadow can be measured from a record.
    # Empty when nothing executed, so a run that reaches no grasp is byte-identical.
    initial_telemetry = dict(
        getattr(getattr(report, "pick_report", None), "calculator_telemetry", None) or {})
    record_extra: dict[str, Any] = dict(getattr(report, "telemetry", {}) or {})
    record_extra["safety_rejected"] = report.outcome in SAFETY_REJECTED_OUTCOMES
    record_extra.update(_look_extras(report))
    # Every attempt's motion and the grasps it tried, where it commanded any (RC6: P5's failed moveJ was in no file).
    record_extra.update(_attempt_extras(report))
    if extra:
        record_extra.update(extra)
    # The camera world the motions stood on, applied after the caller's bag, so a runner cannot
    # overwrite what the arm stamped.
    record_extra.update(_camera_world_extras(report))
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
    # Populate the execution block the GraspAttemptRecord contract requires for
    # succeeded/execution_failed/verification_failed outcomes (the soak telemetry audit checks it), from
    # the report itself: the executed-grasp outcome. The verification block is optional since the
    # post-grasp verification stage, its one live writer, left on 2026-09-29: it is written only from the
    # sim ground-truth lift, explicitly labelled, when the runner stamped it in extra (sim_lifted /
    # sim_lift_mm measured from the object's world pose, the physical truth, not a hardware verifier),
    # and is None otherwise.
    execution = execution_metadata_from(report)
    # The field that cost the RL layer its action column. `BaselineSARExtractor` projects an action
    # from `selected_grasp`, then `refined_grasp`, then `initial_grasp`, and no writer in this
    # repository set any of the three, so every record a real pick produced fell through all three to
    # the literal token "noop". Measured 2026-09-10: two attempts on two different grasps extracted
    # to the same action, which is an action space of size one, a dataset that cannot express a
    # preference between two grasps, which is the only thing it is for.
    #
    # The executed grasp, not the ranked-best. The question a record must answer is which grasp this
    # outcome graded, and only the one the arm actually went to has that standing. It is the same
    # object `execution_metadata_from` summarises one line up, so the two blocks cannot disagree.
    # `None` when nothing executed, which is byte-identical to what this wrote before, and that is
    # every attempt that never reached a grasp.
    selected_grasp = grasp_metadata_from(
        getattr(getattr(report, "pick_report", None), "executed_grasp", None))
    verification: Optional[dict[str, Any]] = None
    if "sim_lifted" in record_extra:
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
        selected_grasp=selected_grasp,
        verification=verification,
        # `refinement` has no writer since the two-scan refinement left on 2026-09-29, so it stays
        # `None`; the block stays in the record so a record logged before then still reads, and the
        # three refine-stage outcomes that required it are known to the catalog by their strings.
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
