"""Baseline KPI/SLO/telemetry report generator.

Builds the committed baseline that later work is measured against.
Runtime SLO numbers come from the packs' own latency telemetry:
``_slo_aggregate`` keeps the worst (largest) p95 per stage across
the packs, and reports ``null`` only for a stage no pack
contributed a sample for. This generator locks the shape of the
report; it invents no latency values.

Determinism contract: given the same canonical packs on disk, this
generator produces byte-identical JSON output, which is the property
the strict gate keys off when re-running the report in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from src.robot.grasping.constants import (
    REPLAY_BASELINE_REPORT_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    iter_jsonl,
)
from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    CanonicalPackSpec,
    build_manifest,
)
from src.robot.grasping.replay.kpi import compute_kpis
from src.robot.grasping.replay.slo_eval import (
    DECISION_P95_MAX_MS,
    RANKING_P95_MAX_MS,
    FUSION_P95_MAX_MS,
    _percentile_nearest_rank,
    evaluate_slo_pack,
)
from src.robot.grasping.replay.telemetry_catalog import (
    EXTRA_TELEMETRY_VERSION,
    audit_records,
    audit_extra_records,
    extra_field_coverage,
    extra_field_group_map,
)


#: Bumped only when the report contract changes (new top-level
#: keys, removed fields, semantic redefinitions).
BASELINE_REPORT_VERSION: int = 1

#: Locked thresholds tied to KPI/SLO IDs. Every entry here is a float;
#: the ``| None`` in the annotation is headroom for a threshold a later
#: phase adds without changing the schema.
U_PLUS_TARGET_THRESHOLDS: dict[str, dict[str, float | None]] = {
    "easy": {
        "false_positive_grasp_rate_max": 0.005,
        "dead_loop_rate_max": 0.0,
        "median_cycle_time_increase_pct_max": 5.0,
    },
    "auto": {
        "dead_loop_rate_max": 0.002,
    },
    "dense": {
        "dead_loop_rate_max": 0.01,
        "false_positive_grasp_rate_max": 0.02,
    },
    "calibration": {
        "calibration_brier_score_max": 0.08,
        "calibration_log_loss_max": 0.28,
    },
    "ranking": {
        "probability_rank_lift_top1_min_pts": 1.5,
    },
    "multi_view": {
        "multi_view_occlusion_reduction_min_pct": 20.0,
    },
    "drift_ood": {
        "drift_detection_precision_min": 0.90,
        "drift_detection_recall_min": 0.85,
        "ood_rejection_precision_min": 0.90,
        "ood_rejection_recall_min": 0.80,
    },
    "runtime_slo": {
        "p95_decision_latency_ms_max": 60.0,
        "p95_ranking_latency_ms_max": 80.0,
        "p95_fusion_latency_ms_max": 220.0,
    },
    "soak": {
        "min_attempts": 2000.0,
        "dead_loop_rate_max": 0.005,
    },
}


@dataclass(frozen=True, slots=True)
class _PackResult:
    name: str
    relative_path: str
    record_count: int
    kpi: dict[str, object]
    telemetry_offender_count: int
    extra_type_offender_count: int
    extra_field_coverage: dict[str, float]
    slo: dict[str, object]
    #: Per-mode ``attempt_wall_time_s`` samples collected from each
    #: record's ``extra`` block, keyed by record ``mode``. Used by
    #: ``_slo_aggregate`` for the ``p95_attempt_wall_time_s_by_mode``
    #: runtime-SLO field.
    attempt_wall_time_samples_by_mode: dict[str, tuple[float, ...]]


#: This report is the number later work is measured against, and it is committed.
#: What is worth keeping is the roll-up that produced it (records in, offenders
#: found) and the regression verdict: both are decided, not raised.
logger = create_grasping_logger("BaselineReport", REPLAY_BASELINE_REPORT_LOG_FILE)


def _load_pack_records(
    repo_root: Path, pack: CanonicalPackSpec
) -> tuple[GraspAttemptRecord, ...]:
    abs_path = (repo_root / pack.relative_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(
            f"canonical pack missing on disk: {abs_path}. "
            "Regenerate via "
            "`python -m src.robot.grasping.replay --regenerate-canonical`."
        )
    return tuple(iter_jsonl(abs_path))


def _summarise_pack(
    repo_root: Path, pack: CanonicalPackSpec
) -> _PackResult:
    records = _load_pack_records(repo_root, pack)
    summary = compute_kpis(records)
    catalog_offenders = audit_records(records)
    extra_offenders = audit_extra_records(records)
    coverage = extra_field_coverage(records)
    slo_report = evaluate_slo_pack(records, pack_name=pack.name)
    # Collect per-mode attempt_wall_time_s samples.
    wall_by_mode: dict[str, list[float]] = {}
    for record in records:
        extra = getattr(record, "extra", None) or {}
        raw = extra.get("attempt_wall_time_s")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0.0 or value != value:  # NaN guard
            continue
        wall_by_mode.setdefault(str(record.mode), []).append(value)
    samples = {
        mode: tuple(sorted(values))
        for mode, values in wall_by_mode.items()
    }
    return _PackResult(
        name=pack.name,
        relative_path=pack.relative_path,
        record_count=len(records),
        kpi=summary.to_dict(),
        telemetry_offender_count=len(catalog_offenders),
        extra_type_offender_count=len(extra_offenders),
        extra_field_coverage=coverage,
        slo=slo_report.to_dict(),
        attempt_wall_time_samples_by_mode=samples,
    )


def _slo_aggregate(pack_results: Sequence["_PackResult"]) -> dict[str, object]:
    """Return the runtime-SLO aggregate: the worst (largest) p95 per stage across packs (``None`` when no pack contributed, e.g. ``easy`` omits fusion)."""

    worst: dict[str, float | None] = {
        "decision_latency_ms": None,
        "ranking_latency_ms": None,
        "fusion_latency_ms": None,
    }
    pack_pass: list[bool] = []
    for r in pack_results:
        slo = r.slo
        pack_pass.append(bool(slo.get("passes_gate", False)))
        for stage in cast("list[dict[str, Any]]", slo.get("stages", [])):
            name = stage["stage"]
            if name not in worst:
                continue
            kpi = stage["kpi"]
            p95 = kpi.get("p95_ms")
            if p95 is None:
                continue
            prev = worst[name]
            worst[name] = float(p95) if prev is None else max(prev, float(p95))
    return {
        "capability_group": "latency",
        "p95_decision_latency_ms": worst["decision_latency_ms"],
        "p95_ranking_latency_ms": worst["ranking_latency_ms"],
        "p95_fusion_latency_ms": worst["fusion_latency_ms"],
        "p95_decision_latency_ms_gate": float(DECISION_P95_MAX_MS),
        "p95_ranking_latency_ms_gate": float(RANKING_P95_MAX_MS),
        "p95_fusion_latency_ms_gate": float(FUSION_P95_MAX_MS),
        "passes_gate": bool(all(pack_pass)) if pack_pass else False,
        "p95_attempt_wall_time_s_by_mode": _wall_time_p95_by_mode(
            pack_results
        ),
    }


def _wall_time_p95_by_mode(
    pack_results: Sequence["_PackResult"],
) -> dict[str, float | None]:
    """Aggregate per-mode ``attempt_wall_time_s`` p95 (nearest-rank); ``None`` for modes that contributed no samples."""

    modes = ("easy", "auto", "dense_clutter", "dense_autonomous", "closed_loop")
    pooled: dict[str, list[float]] = {m: [] for m in modes}
    for r in pack_results:
        for mode, mode_samples in r.attempt_wall_time_samples_by_mode.items():
            if mode not in pooled:
                continue
            pooled[mode].extend(float(v) for v in mode_samples)
    out: dict[str, float | None] = {}
    for mode in modes:
        samples = pooled[mode]
        if not samples:
            out[mode] = None
        else:
            samples.sort()
            out[mode] = float(_percentile_nearest_rank(samples, 95.0))
    return out


def _adaptation_aggregate(repo_root: Path) -> dict[str, object]:
    """Summarise the adaptation audit log (plan/apply/rollback counts + latest markers); all-zero when no audit log is present."""

    audit_path = (
        repo_root / "logs" / "adaptation" / "adaptation_audit.jsonl"
    )
    active_overlay = (
        repo_root / "configs" / "overlays" / "adaptation_active.yaml"
    )
    total_plans = 0
    total_applies = 0
    total_rollbacks = 0
    last_apply_plan_id: str | None = None
    last_apply_at_ns: int | None = None
    last_rollback_plan_id: str | None = None
    last_rollback_at_ns: int | None = None
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = entry.get("action")
                if action == "plan":
                    total_plans += 1
                elif action == "apply":
                    total_applies += 1
                    last_apply_plan_id = entry.get("plan_id")
                    ts = entry.get("timestamp_ns")
                    last_apply_at_ns = int(ts) if ts is not None else None
                elif action == "rollback":
                    total_rollbacks += 1
                    last_rollback_plan_id = entry.get("plan_id")
                    ts = entry.get("timestamp_ns")
                    last_rollback_at_ns = int(ts) if ts is not None else None
    return {
        "capability_group": "guarded_adaptation",
        "active_overlay_present": bool(active_overlay.exists()),
        "audit_log_present": bool(audit_path.exists()),
        "total_plans": int(total_plans),
        "total_applies": int(total_applies),
        "total_rollbacks": int(total_rollbacks),
        "last_apply_plan_id": last_apply_plan_id,
        "last_apply_at_ns": last_apply_at_ns,
        "last_rollback_plan_id": last_rollback_plan_id,
        "last_rollback_at_ns": last_rollback_at_ns,
    }


def build_baseline_report(
    repo_root: Path,
    packs: Sequence[CanonicalPackSpec] = CANONICAL_PACKS,
) -> dict[str, object]:
    """Build the JSON-safe baseline report dict."""

    pack_results = [_summarise_pack(repo_root, p) for p in packs]
    manifest = build_manifest(repo_root, packs=packs)
    # One line for the whole build, not one per pack: the offender counts are the
    # only part a human acts on, and they are meaningless pack by pack.
    offenders = sum(int(r.telemetry_offender_count) for r in pack_results)
    extra_offenders = sum(int(r.extra_type_offender_count) for r in pack_results)
    log = logger.warning if (offenders or extra_offenders) else logger.info
    log(
        "Built baseline report over %d pack(s), %d record(s): %d telemetry "
        "offender(s), %d extra-type offender(s)",
        len(pack_results),
        sum(int(r.record_count) for r in pack_results),
        offenders,
        extra_offenders,
    )
    return {
        "baseline_report_version": int(BASELINE_REPORT_VERSION),
        "extra_telemetry_version": int(EXTRA_TELEMETRY_VERSION),
        "extra_field_group_map": extra_field_group_map(),
        "target_thresholds": U_PLUS_TARGET_THRESHOLDS,
        "runtime_slo": _slo_aggregate(pack_results),
        "adaptation": _adaptation_aggregate(repo_root),
        "packs": [
            {
                "name": r.name,
                "path": r.relative_path,
                "record_count": int(r.record_count),
                "kpi": r.kpi,
                "telemetry_offender_count": int(r.telemetry_offender_count),
                "extra_type_offender_count": int(
                    r.extra_type_offender_count
                ),
                "extra_field_coverage": r.extra_field_coverage,
                "slo": r.slo,
            }
            for r in pack_results
        ],
        "manifest": manifest,
    }


DEFAULT_REPORT_RELATIVE_PATH: str = "docs/baselines/u_plus_baseline_v1.json"


def write_baseline_report(
    repo_root: Path,
    out_relative_path: str = DEFAULT_REPORT_RELATIVE_PATH,
    packs: Sequence[CanonicalPackSpec] = CANONICAL_PACKS,
) -> Path:
    """Write the baseline report to ``out_relative_path`` (deterministic)."""

    payload = build_baseline_report(repo_root, packs=packs)
    out_path = (repo_root / out_relative_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    # newline="" so the rendered line feed reaches the file as a line feed. MEASURED
    # 2026-09-10: without it write_text translated on Windows and the committed golden stood
    # at 25074 CRLF bytes against the 24375 LF bytes rendered here (699 carriage returns and
    # no content between them), so this regenerator reproduced its own output on one platform
    # only. The packs, the manifest and the promotion/OPE reports were repaired the same way.
    out_path.write_text(body, encoding="utf-8", newline="")
    logger.info(
        "Wrote baseline report to %s (%d bytes)", out_path, len(body.encode("utf-8"))
    )
    return out_path


@dataclass(frozen=True, slots=True)
class RegressionVerdict:
    """Outcome of comparing two baseline reports (empty ``regressions`` means the new report is >= the baseline on every tracked metric)."""

    regressions: tuple[str, ...]
    decision_p95_delta_ms: float | None
    ranking_p95_delta_ms: float | None
    fusion_p95_delta_ms: float | None

    @property
    def ok(self) -> bool:
        return not self.regressions

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "regressions": list(self.regressions),
            "decision_p95_delta_ms": self.decision_p95_delta_ms,
            "ranking_p95_delta_ms": self.ranking_p95_delta_ms,
            "fusion_p95_delta_ms": self.fusion_p95_delta_ms,
        }


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _delta_or_none(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None:
        return None
    return float(curr) - float(prev)


def compare_kpi_deltas(
    prev_report: dict,
    curr_report: dict,
    *,
    tolerance_ms: float = 0.0,
) -> RegressionVerdict:
    """Compare runtime-SLO p95 deltas between two baseline reports; flag each stage where ``curr - prev > tolerance_ms`` (``None`` p95 on either side is skipped, not a regression)."""

    prev_slo = prev_report.get("runtime_slo", {}) or {}
    curr_slo = curr_report.get("runtime_slo", {}) or {}
    pairs = (
        ("decision_p95_delta_ms", "p95_decision_latency_ms"),
        ("ranking_p95_delta_ms", "p95_ranking_latency_ms"),
        ("fusion_p95_delta_ms", "p95_fusion_latency_ms"),
    )
    deltas: dict[str, float | None] = {}
    regressions: list[str] = []
    for delta_key, slo_key in pairs:
        prev_val = _opt_float(prev_slo.get(slo_key))
        curr_val = _opt_float(curr_slo.get(slo_key))
        delta = _delta_or_none(curr_val, prev_val)
        deltas[delta_key] = delta
        if delta is not None and delta > tolerance_ms:
            regressions.append(
                f"{slo_key} regressed by {delta:.3f} ms "
                f"(prev={prev_val}, curr={curr_val}, "
                f"tolerance={tolerance_ms})"
            )
    if regressions:
        # Returned, never raised: the auto-rollback guardrail acts on this and the
        # reason it acted has to outlive the process that decided.
        logger.warning(
            "SLO regression vs the previous report (tolerance %.3f ms): %s",
            float(tolerance_ms),
            "; ".join(regressions),
        )
    return RegressionVerdict(
        regressions=tuple(regressions),
        decision_p95_delta_ms=deltas["decision_p95_delta_ms"],
        ranking_p95_delta_ms=deltas["ranking_p95_delta_ms"],
        fusion_p95_delta_ms=deltas["fusion_p95_delta_ms"],
    )
