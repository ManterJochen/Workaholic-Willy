"""KPI replay harness.

Public surface for the production-readiness toolbox: the KPI
summary, the soak gate over a record log and the telemetry
catalogue. See ``README.md`` for the full
design contract. The CLI entry point is
``python -m src.robot.grasping.replay``.
"""

from .kpi import (
    UNMEASURABLE_KPIS,
    KpiSummary,
    compute_kpis,
    unmeasurable_kpis,
)
from .runs import (
    GateKeyStatus,
    KpiRollup,
    RecordLog,
    SoakGate,
    SoakSource,
    SoakVerdict,
    TelemetryOffender,
    TelemetryVerdict,
)
from .telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_record,
    audit_records,
)

__all__ = [
    "GateKeyStatus",
    "KpiRollup",
    "KpiSummary",
    "UNMEASURABLE_KPIS",
    "RecordLog",
    "SoakGate",
    "SoakSource",
    "SoakVerdict",
    "TelemetryOffender",
    "TelemetryVerdict",
    "TELEMETRY_CATALOG",
    "audit_record",
    "audit_records",
    "compute_kpis",
    "unmeasurable_kpis",
]
