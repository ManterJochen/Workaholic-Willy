"""KPI replay harness.

Public surface for the production-readiness toolbox: the KPI
summary, the soak gate, the baseline, the preset overlays
and the telemetry catalogue. See ``README.md`` for the full
design contract. The CLI entry point is
``python -m src.robot.grasping.replay``.
"""

from .kpi import KpiSummary, compute_kpis
from .presets import apply_preset, list_presets, load_preset
from .runs import (
    Baseline,
    BaselineMeasurement,
    GateKeyStatus,
    KpiRollup,
    RecordLog,
    SoakGate,
    SoakSource,
    SoakVerdict,
    TelemetryOffender,
    TelemetryVerdict,
)
from .soak import SoakScenarioSpec, generate_soak_records
from .telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_record,
    audit_records,
)

__all__ = [
    "Baseline",
    "BaselineMeasurement",
    "GateKeyStatus",
    "KpiRollup",
    "KpiSummary",
    "RecordLog",
    "SoakGate",
    "SoakSource",
    "SoakVerdict",
    "TelemetryOffender",
    "TelemetryVerdict",
    "SoakScenarioSpec",
    "TELEMETRY_CATALOG",
    "apply_preset",
    "audit_record",
    "audit_records",
    "compute_kpis",
    "generate_soak_records",
    "list_presets",
    "load_preset",
]
