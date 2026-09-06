"""Replay-side drift + OOD watchdog KPI evaluator.

Re-runs the runtime watchdog evaluator
(:func:`src.robot.execution.calibration_watchdog.evaluate_watchdog`)
over a labelled canonical replay pack and computes precision/recall
against the ground-truth labels stamped into each record's ``extra``
bag.

Pure-function evaluator: replay tooling builds an in-memory rolling
window of :class:`WatchdogSample` instances from the record stream and
calls the same code the runtime uses (no watchdog state lives here).
Locked thresholds: drift precision >= 0.90, recall >= 0.85; OOD
precision >= 0.90, recall >= 0.80. Gate failure surfaces a non-zero exit
on the CLI (``--watchdog-eval`` returns 2). No caller passes a policy,
so ``_default_policy`` builds the ``WatchdogMode.SHADOW`` policy here
and every record is evaluated (the ``enforced`` flag is irrelevant for
the KPI gate).

Per-record prediction rules (mirroring runtime semantics):

* Drift pack: the binary prediction is
  ``drift_severity >= MODERATE`` from the per-tick
  :class:`WatchdogReport`. Ground truth is ``extra.drift_label``.
* OOD pack: the binary prediction is ``ood_flagged`` from the per-
  tick report (already ``ood_severity >= MODERATE`` per the
  watchdog's own threshold). Ground truth is ``extra.ood_label``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from src.robot.execution.calibration_watchdog import (
    DriftSeverity,
    WatchdogHistory,
    WatchdogMode,
    WatchdogPolicy,
    WatchdogReport,
    WatchdogSample,
    evaluate_watchdog,
)
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    iter_jsonl,
)

__all__ = [
    "DRIFT_PRECISION_GATE",
    "DRIFT_RECALL_GATE",
    "OOD_PRECISION_GATE",
    "OOD_RECALL_GATE",
    "BinaryKPI",
    "WatchdogPackKPIReport",
    "build_sample_from_record",
    "evaluate_drift_pack",
    "evaluate_ood_pack",
    "evaluate_drift_pack_path",
    "evaluate_ood_pack_path",
]


#: Locked KPI thresholds.
DRIFT_PRECISION_GATE: float = 0.90
DRIFT_RECALL_GATE: float = 0.85
OOD_PRECISION_GATE: float = 0.90
OOD_RECALL_GATE: float = 0.80


def _opt_float(extra: dict[str, Any], key: str) -> Optional[float]:
    v = extra.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        # Defensive: booleans are ints in Python; reject so ``True`` is
        # never silently coerced to ``1.0``.
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _opt_bool(extra: dict[str, Any], key: str) -> Optional[bool]:
    v = extra.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return None


def build_sample_from_record(record: GraspAttemptRecord) -> WatchdogSample:
    """Build a :class:`WatchdogSample` from a labelled replay record.

    The field mapping follows the ``canonical_datasets`` enrichment contract.
    Missing fields are ``None``, and the monitor skips them.
    """

    extra = dict(record.extra or {})
    return WatchdogSample(
        calibration_residual_mm=_opt_float(
            extra, "drift_hand_eye_residual_mm"
        ),
        verification_residual_mm=_opt_float(
            extra, "drift_verification_residual_mm"
        ),
        predicted_observed_calibration_delta_mm=_opt_float(
            extra, "drift_predicted_observed_delta_mm"
        ),
        depth_confidence_mean=_opt_float(
            extra, "drift_depth_confidence_mean"
        ),
        fail_closed=_opt_bool(extra, "drift_fail_closed"),
        ood_score=_opt_float(extra, "ood_score"),
    )


@dataclass(frozen=True, slots=True)
class BinaryKPI:
    """Precision / recall / counts for a binary detector."""

    total: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def negatives(self) -> int:
        return self.true_negatives + self.false_positives

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": int(self.total),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "true_positives": int(self.true_positives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "true_negatives": int(self.true_negatives),
            "precision": float(self.precision),
            "recall": float(self.recall),
        }


@dataclass(frozen=True, slots=True)
class WatchdogPackKPIReport:
    """KPI rollup for a single labelled pack."""

    pack_name: str
    detector: str  # "drift" or "ood"
    kpi: BinaryKPI
    precision_gate: float
    recall_gate: float

    @property
    def passes_precision_gate(self) -> bool:
        return self.kpi.precision >= self.precision_gate

    @property
    def passes_recall_gate(self) -> bool:
        return self.kpi.recall >= self.recall_gate

    @property
    def passes_gate(self) -> bool:
        return self.passes_precision_gate and self.passes_recall_gate

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "detector": self.detector,
            "kpi": self.kpi.to_dict(),
            "precision_gate": float(self.precision_gate),
            "recall_gate": float(self.recall_gate),
            "passes_precision_gate": bool(self.passes_precision_gate),
            "passes_recall_gate": bool(self.passes_recall_gate),
            "passes_gate": bool(self.passes_gate),
        }


def _default_policy() -> WatchdogPolicy:
    """Shadow-mode policy for replay KPI scoring (full report, ``enforced`` ignored)."""

    return WatchdogPolicy(mode=WatchdogMode.SHADOW)


def _iter_reports(
    records: Sequence[GraspAttemptRecord],
    policy: WatchdogPolicy,
) -> Iterable[tuple[GraspAttemptRecord, WatchdogReport]]:
    """Yield ``(record, report)`` pairs over a rolling window."""

    window: deque[WatchdogSample] = deque(maxlen=policy.window_size)
    for rec in records:
        window.append(build_sample_from_record(rec))
        history = WatchdogHistory(tuple(window))
        report = evaluate_watchdog(history, policy)
        yield rec, report


def _score_binary(
    records: Sequence[GraspAttemptRecord],
    policy: WatchdogPolicy,
    *,
    label_key: str,
    predict: Callable[[WatchdogReport], bool],
) -> BinaryKPI:
    tp = fp = fn = tn = total = 0
    for rec, report in _iter_reports(records, policy):
        extra = rec.extra or {}
        label_raw = extra.get(label_key)
        if not isinstance(label_raw, bool):
            # Record without a ground-truth label: skip it. The canonical
            # packs label every record, but a mis-shaped input must not
            # poison the metrics.
            continue
        label = bool(label_raw)
        pred = bool(predict(report))
        total += 1
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif (not pred) and label:
            fn += 1
        else:
            tn += 1
    return BinaryKPI(
        total=total,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
    )


def _drift_prediction(report: WatchdogReport) -> bool:
    """True iff the watchdog raised ``drift_severity >= MODERATE``."""

    return report.drift_severity in (
        DriftSeverity.MODERATE,
        DriftSeverity.HIGH,
        DriftSeverity.SEVERE,
    )


def _ood_prediction(report: WatchdogReport) -> bool:
    """True iff the watchdog flagged the record as OOD."""

    return bool(report.ood_flagged)


def evaluate_drift_pack(
    records: Sequence[GraspAttemptRecord],
    policy: Optional[WatchdogPolicy] = None,
    *,
    pack_name: str = "replay_drift_synthetic_v1",
) -> WatchdogPackKPIReport:
    """Score the drift detector over a labelled record stream."""

    pol = policy if policy is not None else _default_policy()
    kpi = _score_binary(
        records,
        pol,
        label_key="drift_label",
        predict=_drift_prediction,
    )
    return WatchdogPackKPIReport(
        pack_name=pack_name,
        detector="drift",
        kpi=kpi,
        precision_gate=DRIFT_PRECISION_GATE,
        recall_gate=DRIFT_RECALL_GATE,
    )


def evaluate_ood_pack(
    records: Sequence[GraspAttemptRecord],
    policy: Optional[WatchdogPolicy] = None,
    *,
    pack_name: str = "replay_ood_synthetic_v1",
) -> WatchdogPackKPIReport:
    """Score the OOD detector over a labelled record stream."""

    pol = policy if policy is not None else _default_policy()
    kpi = _score_binary(
        records,
        pol,
        label_key="ood_label",
        predict=_ood_prediction,
    )
    return WatchdogPackKPIReport(
        pack_name=pack_name,
        detector="ood",
        kpi=kpi,
        precision_gate=OOD_PRECISION_GATE,
        recall_gate=OOD_RECALL_GATE,
    )


def evaluate_drift_pack_path(
    jsonl_path: Path,
    policy: Optional[WatchdogPolicy] = None,
) -> WatchdogPackKPIReport:
    """Convenience: load a JSONL pack from disk and score the drift detector."""

    records = tuple(iter_jsonl(jsonl_path))
    return evaluate_drift_pack(records, policy)


def evaluate_ood_pack_path(
    jsonl_path: Path,
    policy: Optional[WatchdogPolicy] = None,
) -> WatchdogPackKPIReport:
    """Convenience: load a JSONL pack from disk and score the OOD detector."""

    records = tuple(iter_jsonl(jsonl_path))
    return evaluate_ood_pack(records, policy)
