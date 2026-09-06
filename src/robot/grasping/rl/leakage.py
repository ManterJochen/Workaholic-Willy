"""Leakage audits.

The four audits and their severities:

1. Hard reject on any overlap of ``attempt_id`` between splits.
2. Hard reject on any overlap of ``(scene_id, camera_pose_hash)``
   when both fields are present.
3. Soft warn at >5%, hard reject at >10% for ``object_set`` overlap;
   warns become rejects when ``strict=True``.
4. Hard reject when ``extra.timestamp`` ranges between train and
   test overlap by more than the median per-attempt duration.

A field missing from the records is skipped with an audit note, which
keeps the audit compatible with the existing outcome schema. A note
marks an audit that did not run, not one that passed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.robot.grasping.constants import (
    RL_LEAKAGE_LOG_FILE,
    create_grasping_logger,
)

#: A NOTE here means an audit was skipped because the field it needs is absent:
#: a green leakage report that proved nothing. That distinction survives in the
#: JSON, but only for whoever reads the JSON, and a training run that consumed a
#: vacuous audit needs it in the record of the run itself.
logger = create_grasping_logger("RLLeakage", RL_LEAKAGE_LOG_FILE)


SEVERITY_NOTE: str = "note"
SEVERITY_WARN: str = "warn"
SEVERITY_REJECT: str = "reject"

OBJECT_SET_SOFT_THRESHOLD: float = 0.05
OBJECT_SET_HARD_THRESHOLD: float = 0.10


@dataclass
class LeakageFinding:
    audit: str
    severity: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "audit": self.audit,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)
    strict: bool = False

    @property
    def has_rejects(self) -> bool:
        return any(f.severity == SEVERITY_REJECT for f in self.findings)

    @property
    def has_warns(self) -> bool:
        return any(f.severity == SEVERITY_WARN for f in self.findings)

    @property
    def has_notes(self) -> bool:
        # Every NOTE finding in this module is a skipped audit (a required field was absent), so
        # ``has_notes`` is exactly "the audit was vacuous / could not fully run".
        return any(f.severity == SEVERITY_NOTE for f in self.findings)

    @property
    def passed(self) -> bool:
        if self.has_rejects:
            return False
        # Under ``strict`` a skipped audit is a failure, not a free pass: an operator who asks for
        # a real leakage audit must not get a green light when 3 of the 4 checks silently could not run.
        if self.strict and (self.has_warns or self.has_notes):
            return False
        return True

    def to_json(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "passed": self.passed,
            "interpretation": dict(LEAKAGE_INTERPRETATION),
            "findings": [f.to_json() for f in self.findings],
            "counts_by_severity": {
                SEVERITY_NOTE: sum(1 for f in self.findings if f.severity == SEVERITY_NOTE),
                SEVERITY_WARN: sum(1 for f in self.findings if f.severity == SEVERITY_WARN),
                SEVERITY_REJECT: sum(1 for f in self.findings if f.severity == SEVERITY_REJECT),
            },
        }


#: Honesty banner: what a leakage ``passed`` does and does not prove. A NOTE finding marks a skipped
#: audit (a required field was absent); under ``strict=False`` warns + notes are free passes.
LEAKAGE_INTERPRETATION: dict[str, Any] = {
    "what_passed_means": (
        "passed=true means no HARD-REJECT finding. Under strict=False, WARN + NOTE findings are free passes; "
        "a NOTE marks an audit that was SKIPPED because a required field was absent. This proves split "
        "separation on the AUDITED dimensions only."
    ),
    "audit_scopes": {
        "attempt_id_overlap": "HARD-REJECT if the same attempt_id appears in >1 split",
        "scene_camera_overlap": "HARD-REJECT if a (scene_id, camera_pose_hash) pair spans train+test (SKIP if absent)",
        "object_set_overlap": "WARN >5% / REJECT >10% of test object_sets seen in train (SKIP if absent)",
        "time_bleed": "REJECT if train/test time ranges overlap beyond the median attempt duration (SKIP if synthetic/index timestamps)",
    },
    "what_this_does_not_prove": [
        "real-world generalization",
        "sim-vs-real fidelity",
        "leakage on UNMEASURED fields (operator, time-of-day, lighting, ...)",
    ],
}


def _attempt_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(r.get("attempt_id")) for r in records if r.get("attempt_id") is not None]


def audit_attempt_id_overlap(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    by_split: dict[str, set[str]] = {
        name: set(_attempt_ids(records)) for name, records in splits.items()
    }
    names = sorted(by_split.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = by_split[a] & by_split[b]
            if inter:
                findings.append(
                    LeakageFinding(
                        audit="attempt_id_overlap",
                        severity=SEVERITY_REJECT,
                        message=f"{len(inter)} attempt_id(s) appear in both '{a}' and '{b}'",
                        detail={
                            "split_a": a,
                            "split_b": b,
                            "overlap_count": len(inter),
                            "sample": sorted(inter)[:5],
                        },
                    )
                )
    return findings


def _scene_camera_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for r in records:
        extra = r.get("extra") or {}
        if not isinstance(extra, Mapping):
            continue
        scene = extra.get("scene_id")
        cam = extra.get("camera_pose_hash")
        if isinstance(scene, str) and isinstance(cam, str):
            keys.add((scene, cam))
    return keys


def audit_scene_camera_overlap(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    by_split = {name: _scene_camera_keys(records) for name, records in splits.items()}
    if not any(by_split.values()):
        findings.append(
            LeakageFinding(
                audit="scene_camera_overlap",
                severity=SEVERITY_NOTE,
                message="scene_id / camera_pose_hash absent from records; audit skipped",
            )
        )
        return findings
    names = sorted(by_split.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = by_split[a] & by_split[b]
            if inter:
                findings.append(
                    LeakageFinding(
                        audit="scene_camera_overlap",
                        severity=SEVERITY_REJECT,
                        message=f"{len(inter)} (scene_id, camera_pose_hash) overlap between '{a}' and '{b}'",
                        detail={
                            "split_a": a,
                            "split_b": b,
                            "overlap_count": len(inter),
                            "sample": sorted(map(list, inter))[:5],
                        },
                    )
                )
    return findings


def _object_sets(records: Sequence[Mapping[str, Any]]) -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    for r in records:
        extra = r.get("extra") or {}
        if not isinstance(extra, Mapping):
            continue
        obj = extra.get("object_set")
        if isinstance(obj, (list, tuple)):
            tokens = tuple(sorted({str(o) for o in obj}))
            if tokens:
                out.append(frozenset(tokens))
    return out


def audit_object_set_overlap(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    by_split = {name: _object_sets(records) for name, records in splits.items()}
    if not any(by_split.values()):
        findings.append(
            LeakageFinding(
                audit="object_set_overlap",
                severity=SEVERITY_NOTE,
                message="object_set absent from records; audit skipped",
            )
        )
        return findings
    train_unique: set[frozenset[str]] = set(by_split.get("train", []))
    test_list = by_split.get("test", [])
    if not test_list:
        return findings
    overlap = sum(1 for s in test_list if s in train_unique)
    fraction = overlap / len(test_list)
    detail = {
        "overlap_records_in_test": overlap,
        "test_records": len(test_list),
        "fraction": round(fraction, 6),
        "soft_threshold": OBJECT_SET_SOFT_THRESHOLD,
        "hard_threshold": OBJECT_SET_HARD_THRESHOLD,
    }
    if fraction > OBJECT_SET_HARD_THRESHOLD:
        findings.append(
            LeakageFinding(
                audit="object_set_overlap",
                severity=SEVERITY_REJECT,
                message=(
                    f"object_set overlap between train and test is "
                    f"{fraction:.2%} (> {OBJECT_SET_HARD_THRESHOLD:.0%})"
                ),
                detail=detail,
            )
        )
    elif fraction > OBJECT_SET_SOFT_THRESHOLD:
        findings.append(
            LeakageFinding(
                audit="object_set_overlap",
                severity=SEVERITY_WARN,
                message=(
                    f"object_set overlap between train and test is "
                    f"{fraction:.2%} (> {OBJECT_SET_SOFT_THRESHOLD:.0%})"
                ),
                detail=detail,
            )
        )
    return findings


def _timestamps(records: Sequence[Mapping[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in records:
        extra = r.get("extra") or {}
        if isinstance(extra, Mapping):
            ts = extra.get("timestamp")
            if isinstance(ts, (int, float)):
                out.append(float(ts))
                continue
        ts_top = r.get("timestamp")
        if isinstance(ts_top, (int, float)):
            out.append(float(ts_top))
    return out


def _attempt_durations(records: Sequence[Mapping[str, Any]]) -> list[float]:
    out: list[float] = []
    for r in records:
        extra = r.get("extra") or {}
        if not isinstance(extra, Mapping):
            continue
        d = extra.get("attempt_wall_time_s")
        if isinstance(d, (int, float)) and d > 0:
            out.append(float(d))
    return out


def _looks_synthetic_timestamps(values: Sequence[float]) -> bool:
    """Integer ``extra.timestamp`` in a small range is a record-index counter, not wall-clock time.
    Such values cannot encode collection-order leakage after stratified shuffling, so the time-bleed
    audit downgrades to a NOTE."""

    if not values:
        return True
    for v in values:
        if abs(v - round(v)) > 1e-6:
            return False
    # Index-counter style: max value is bounded by #records.
    return max(values) <= max(10_000.0, 4.0 * len(values))


def audit_time_bleed(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[LeakageFinding]:
    train = splits.get("train", [])
    test = splits.get("test", [])
    train_ts = _timestamps(train)
    test_ts = _timestamps(test)
    if len(set(train_ts)) < 2 or len(set(test_ts)) < 2:
        return [
            LeakageFinding(
                audit="time_bleed",
                severity=SEVERITY_NOTE,
                message="timestamps absent or constant; time-bleed audit skipped",
            )
        ]
    if _looks_synthetic_timestamps(list(train_ts) + list(test_ts)):
        return [
            LeakageFinding(
                audit="time_bleed",
                severity=SEVERITY_NOTE,
                message="extra.timestamp values appear synthetic (record-index counters); audit skipped",
            )
        ]
    durations = _attempt_durations(list(train) + list(test))
    if not durations:
        return [
            LeakageFinding(
                audit="time_bleed",
                severity=SEVERITY_NOTE,
                message="attempt_wall_time_s absent; time-bleed audit skipped",
            )
        ]
    median_dur = statistics.median(durations)
    train_min, train_max = min(train_ts), max(train_ts)
    test_min, test_max = min(test_ts), max(test_ts)
    overlap = max(0.0, min(train_max, test_max) - max(train_min, test_min))
    if overlap > median_dur:
        return [
            LeakageFinding(
                audit="time_bleed",
                severity=SEVERITY_REJECT,
                message=(
                    f"train/test timestamp ranges overlap by {overlap:.4f}s, "
                    f"exceeding median attempt duration {median_dur:.4f}s"
                ),
                detail={
                    "train_range": [train_min, train_max],
                    "test_range": [test_min, test_max],
                    "overlap_seconds": overlap,
                    "median_attempt_seconds": median_dur,
                },
            )
        ]
    return []


def run_leakage_audits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    strict: bool = False,
) -> LeakageReport:
    """Run all leakage audits in deterministic order."""

    findings: list[LeakageFinding] = []
    findings.extend(audit_attempt_id_overlap(splits))
    findings.extend(audit_scene_camera_overlap(splits))
    findings.extend(audit_object_set_overlap(splits))
    findings.extend(audit_time_bleed(splits))
    report = LeakageReport(findings=findings, strict=strict)
    skipped = [f.audit for f in findings if f.severity == SEVERITY_NOTE]
    if report.has_rejects:
        logger.error(
            "Leakage audit REJECTED the split over %s: %s",
            "/".join(splits),
            "; ".join(
                f"{f.audit}: {f.message}"
                for f in findings
                if f.severity == SEVERITY_REJECT
            ),
        )
    elif skipped or report.has_warns:
        logger.warning(
            "Leakage audit passed%s with %d warning(s) and %d SKIPPED audit(s) (%s); "
            "a skipped audit proves nothing",
            " (strict)" if strict else "",
            sum(1 for f in findings if f.severity == SEVERITY_WARN),
            len(skipped),
            ", ".join(skipped) or "none",
        )
    else:
        logger.info(
            "Leakage audit clean over %d split(s): all 4 audits ran", len(splits)
        )
    return report


__all__ = (
    "LeakageFinding",
    "LeakageReport",
    "OBJECT_SET_HARD_THRESHOLD",
    "OBJECT_SET_SOFT_THRESHOLD",
    "SEVERITY_NOTE",
    "SEVERITY_REJECT",
    "SEVERITY_WARN",
    "audit_attempt_id_overlap",
    "audit_object_set_overlap",
    "audit_scene_camera_overlap",
    "audit_time_bleed",
    "run_leakage_audits",
)
