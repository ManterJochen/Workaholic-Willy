"""Opt-in pick instrumentation for the Isaac sim runners.

Wires two seams that give the offline layers a data source:

* the record-logging seam (``record_logging.log_record``) appends one
  :class:`~src.robot.grasping.telemetry.outcome_logging.GraspAttemptRecord` JSONL line per pick, so
  the soak, KPI and RL layers read sim data; and
* the grasp-point viewer (the calculator's rendered debug PNG, surfaced via
  ``AutonomousGraspService.last_debug_image_png``) writes the annotated frame to disk: segmentation
  mask, projected gripper wireframes, contact arrows, and per-candidate rank and score.

Both are off by default. With neither a ``record_log`` path nor a ``debug_dir`` given,
:func:`write_pick_artifacts` does nothing and a pick is unchanged. This module imports neither
``isaacsim`` nor ``cv2``, so it is importable anywhere.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utility.log_cfg import create_logger
from src.willy_sim.constants import INSTRUMENTATION_LOG_FILE, WILLY_SIM_LOG_DIR

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.execution.autonomous_grasp import AutonomousGraspReport

#: The index of what a collection run left on disk. Every runner writes its records, overlays and
#: result JSON through this module, and a JSONL append is otherwise silent, so a collection that
#: produced half the records it should have would look like one that worked.
_LOG = create_logger(
    "PickInstrumentation", log_file=INSTRUMENTATION_LOG_FILE, log_dir=WILLY_SIM_LOG_DIR,
)


def cell_identity(cell: Any) -> dict[str, Any]:
    """The identity facts every logged record needs so two robots never blur into one dataset.

    ``GraspAttemptRecord`` is a frozen contract with no robot field, so nothing downstream can tell
    a UR3e attempt from a UR5e one. The two cells have different reach envelopes, scene anchors and
    carry heights, and ``build-dataset`` would merge them with no stratification key and no
    leakage-audit dimension, which poisons the RL ranking signal. The stamp rides the
    caller-supplied ``extra=`` bag, so the frozen contract and the committed canonical replay packs
    are untouched.

    ``degraded_engines`` is carried for the same reason in the other direction: a run made without
    cuRobo or the exact-mesh guard describes a different motion stack, and its records say so rather
    than sit unmarked beside fully anchored ones.
    """
    sim = getattr(cell, "sim", None)
    if sim is None:
        return {}
    return {
        "robot_vendor": "sim",
        "robot_model": getattr(sim, "robot_model", None),
        "gripper_mount": getattr(sim, "gripper_mount", None) or "baked",
        "degraded_engines": list(getattr(cell, "degraded_engines", ()) or ()),
    }


def write_pick_artifacts(
    report: "AutonomousGraspReport",
    *,
    attempt_id: str,
    record_log: "str | Path | None" = None,
    debug_dir: "str | Path | None" = None,
    debug_png: bytes | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Persist the opt-in artifacts for one pick attempt.

    Parameters
    ----------
    report
        The :class:`AutonomousGraspReport` returned by ``service.pick()``.
    attempt_id
        Stable id for this attempt. The runner uses ``f"{run_id}-{i:04d}"``, so records and frames
        share a key and are unique within a collection.
    record_log
        JSONL path to append the serialized record to. ``None`` writes no record.
    debug_dir
        Directory to write ``<attempt_id>.png`` into, the grasp-point overlay. ``None`` writes no
        frame.
    debug_png
        The PNG bytes to write, typically ``service.last_debug_image_png``. A falsy value writes no
        frame even when ``debug_dir`` is set, which is what a pick whose perception frame carried no
        rgb produces.
    extra
        Ground-truth labels the pipeline cannot report about itself, such as ``sim_lift_mm`` and
        ``sim_lifted`` measured from the object's world pose, merged into the record's ``extra``
        bag. ``None`` merges nothing.
    """

    if record_log is not None:
        # Local import keeps this module free of any heavy/execution-layer import at module load.
        from src.robot.execution.autonomous_grasp.record_logging import log_record

        log_record(report, attempt_id=attempt_id, log_path=record_log, extra=extra)
        # The outcome rides along because it is the one field that decides whether this record is
        # usable training data, and reading it back out of the JSONL is the slow way to find out.
        _outcome = getattr(report, "outcome", None)
        _LOG.info(
            "attempt %s: record appended to %s (outcome=%s)",
            attempt_id, record_log, getattr(_outcome, "value", _outcome),
        )

    if debug_dir is not None and debug_png:
        out = Path(debug_dir) / f"{attempt_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(debug_png)
        _LOG.info("attempt %s: wrote grasp overlay %s (%d bytes)", attempt_id, out, len(debug_png))
    elif debug_dir is not None:
        # Overlays were asked for and none were produced: the perception frame carried no rgb, which
        # otherwise reads as ``--debug-frames`` being broken. The Isaac 5.1 ground-truth path used to
        # be exactly this.
        _LOG.warning(
            "attempt %s: overlay requested (%s) but the pick rendered no PNG; the perception frame "
            "carried no rgb", attempt_id, debug_dir,
        )


def write_run_result(
    path: "str | Path | None",
    result: Mapping[str, Any],
    **meta: Any,
) -> None:
    """Write a runner's per-run result dict to ``path`` as JSON.

    Cell metadata such as ``scene`` and ``mode`` is merged in from ``meta``, so the mode-matrix
    harness aggregates cells without parsing stdout. A ``path`` of ``None`` writes nothing.
    """

    if path is None:
        return
    payload = {**dict(meta), **dict(result)}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _text = json.dumps(payload, indent=2, sort_keys=True)
    out.write_text(_text, encoding="utf-8")
    # The mode-matrix harness reads these files rather than stdout, so "which cell wrote which file"
    # is the join key when a matrix cell later disagrees with its run.
    _LOG.info(
        "run result written to %s (%d bytes, %s)", out, len(_text.encode("utf-8")),
        ", ".join(f"{k}={v!r}" for k, v in sorted(meta.items())) or "no cell metadata",
    )


__all__ = ["write_pick_artifacts", "write_run_result"]
