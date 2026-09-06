"""Replay-side per-stage latency SLO evaluator.

Computes per-stage latency KPIs (p50 / p95 / p99 / count) over a
canonical replay pack and gates the result against the locked SLO
budgets:

* ``decision_latency_ms`` p95 <= 60 ms
* ``ranking_latency_ms``  p95 <= 80 ms
* ``fusion_latency_ms``   p95 <= 220 ms (dense packs only)

Null contract: missing stage spans are absent from a record's
``extra`` bag. The evaluator skips ``None`` values and computes KPIs
only over the populated stream. Easy packs therefore produce a
``None`` fusion KPI (count == 0), which is reported but never
gate-failed.

Stdlib only (no numpy) so the module loads in the minimal
replay-tooling environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    iter_jsonl,
)

__all__ = [
    "DECISION_P95_MAX_MS",
    "RANKING_P95_MAX_MS",
    "FUSION_P95_MAX_MS",
    "STAGE_GATES_MS",
    "LatencyKPI",
    "SLOGateReport",
    "SLOPackReport",
    "evaluate_slo_pack",
    "evaluate_slo_pack_path",
]


#: Locked p95 SLO budgets in milliseconds.
DECISION_P95_MAX_MS: float = 60.0
RANKING_P95_MAX_MS: float = 80.0
FUSION_P95_MAX_MS: float = 220.0


#: One entry per stage: (telemetry field, p95 gate, gate-required flag).
#: ``required`` controls whether an empty stream counts as a gate
#: failure. Fusion is optional (easy packs omit it); the
#: decision/ranking stages are always required.
STAGE_GATES_MS: tuple[tuple[str, float, bool], ...] = (
    ("decision_latency_ms", DECISION_P95_MAX_MS, True),
    ("ranking_latency_ms", RANKING_P95_MAX_MS, True),
    ("fusion_latency_ms", FUSION_P95_MAX_MS, False),
)


def _percentile_nearest_rank(samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of a non-empty sequence (matches ``LatencyTelemetryCoordinator._p95`` so replay + runtime SLO agree to the millisecond)."""

    if not samples:
        raise ValueError("samples must be non-empty")
    if not 0.0 < pct <= 100.0:
        raise ValueError("pct must be in (0, 100]")
    ordered = sorted(samples)
    rank = max(int(round((pct / 100.0) * len(ordered))) - 1, 0)
    return float(ordered[rank])


def _opt_finite_float(extra: dict[str, Any], key: str) -> Optional[float]:
    """Extract a finite, non-negative float from a record's ``extra`` bag.

    Missing, non-numeric, NaN, inf and negative values return ``None``.
    """

    v = extra.get(key)
    if v is None:
        return None
    if isinstance(v, bool):
        # Booleans are ints in Python; reject so ``True`` never
        # silently becomes ``1.0`` latency.
        return None
    if not isinstance(v, (int, float)):
        return None
    fv = float(v)
    if fv != fv:  # NaN
        return None
    if fv == float("inf") or fv == float("-inf"):
        return None
    if fv < 0.0:
        return None
    return fv


def _collect_stage_samples(
    records: Iterable[GraspAttemptRecord],
    field_name: str,
) -> list[float]:
    out: list[float] = []
    for rec in records:
        extra = rec.extra or {}
        v = _opt_finite_float(dict(extra), field_name)
        if v is not None:
            out.append(v)
    return out


@dataclass(frozen=True, slots=True)
class LatencyKPI:
    """Per-stage latency rollup (counts + percentiles, ms)."""

    count: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    p99_ms: Optional[float]

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> "LatencyKPI":
        if not samples:
            return cls(count=0, p50_ms=None, p95_ms=None, p99_ms=None)
        return cls(
            count=len(samples),
            p50_ms=_percentile_nearest_rank(samples, 50.0),
            p95_ms=_percentile_nearest_rank(samples, 95.0),
            p99_ms=_percentile_nearest_rank(samples, 99.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "p50_ms": (
                float(self.p50_ms) if self.p50_ms is not None else None
            ),
            "p95_ms": (
                float(self.p95_ms) if self.p95_ms is not None else None
            ),
            "p99_ms": (
                float(self.p99_ms) if self.p99_ms is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SLOGateReport:
    """Per-stage gate decision (p95 vs locked budget)."""

    stage: str  # telemetry field name, e.g. ``decision_latency_ms``
    kpi: LatencyKPI
    p95_gate_ms: float
    required: bool

    @property
    def passes_gate(self) -> bool:
        if self.kpi.count == 0:
            # Empty stream: pass iff the stage is optional.
            return not self.required
        p95 = self.kpi.p95_ms
        if p95 is None:  # defensive: should not happen with count > 0
            return not self.required
        return float(p95) <= float(self.p95_gate_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kpi": self.kpi.to_dict(),
            "p95_gate_ms": float(self.p95_gate_ms),
            "required": bool(self.required),
            "passes_gate": bool(self.passes_gate),
        }


@dataclass(frozen=True, slots=True)
class SLOPackReport:
    """Whole-pack SLO rollup (one entry per stage)."""

    pack_name: str
    stages: tuple[SLOGateReport, ...]

    @property
    def passes_gate(self) -> bool:
        return all(s.passes_gate for s in self.stages)

    def stage(self, name: str) -> SLOGateReport:
        for s in self.stages:
            if s.stage == name:
                return s
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "stages": [s.to_dict() for s in self.stages],
            "passes_gate": bool(self.passes_gate),
        }


def evaluate_slo_pack(
    records: Sequence[GraspAttemptRecord],
    *,
    pack_name: str,
) -> SLOPackReport:
    """Compute the per-stage SLO report for an in-memory record stream."""

    stage_reports: list[SLOGateReport] = []
    for field_name, gate_ms, required in STAGE_GATES_MS:
        samples = _collect_stage_samples(records, field_name)
        kpi = LatencyKPI.from_samples(samples)
        stage_reports.append(
            SLOGateReport(
                stage=field_name,
                kpi=kpi,
                p95_gate_ms=float(gate_ms),
                required=bool(required),
            )
        )
    return SLOPackReport(pack_name=pack_name, stages=tuple(stage_reports))


def evaluate_slo_pack_path(
    jsonl_path: Path,
    *,
    pack_name: Optional[str] = None,
) -> SLOPackReport:
    """Convenience: load a JSONL pack from disk and score every stage."""

    records = tuple(iter_jsonl(jsonl_path))
    resolved = pack_name if pack_name is not None else jsonl_path.stem
    return evaluate_slo_pack(records, pack_name=resolved)
