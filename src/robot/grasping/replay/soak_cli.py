"""Replay soak / records CLI handlers.

The records, records-gate, sim-soak, soak and soak-report mode handlers live here
with their soak-only constants. ``main`` imports them for its dispatch and
``_SIM_SOAK_REPORT_RELATIVE_PATH`` for its help text; argparse and dispatch stay
in ``__main__``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.robot.grasping.constants import (
    REPLAY_SOAK_CLI_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.replay.canonical_datasets import (
    repo_root_from_module,
)
from src.robot.grasping.replay.runs import RecordLog, SoakGate
from src.robot.grasping.replay.soak import (
    DEFAULT_SOAK_REPORT_RELATIVE_PATH,
    SOAK_DEFAULT_ATTEMPTS,
)

#: Every mode below is a gate: CI reads the verdict from the exit code, a person
#: reads it from stdout. The file log keeps the verdict, the input and the
#: report path together.
logger = create_grasping_logger("SoakCLI", REPLAY_SOAK_CLI_LOG_FILE)

_SIM_SOAK_REPORT_RELATIVE_PATH = "logs/u12/sim_soak_report.json"


def _refuse_missing_input(mode: str, message: str, *, log: Any = None) -> int:
    """Refuse a mode whose input file is not there: one sentence on stdout, exit 2, no traceback.

    Measured: ``--records-gate``, ``--sim-soak-report`` and ``--failure-taxonomy`` each answered a
    mistyped path with a raw ``FileNotFoundError`` stack trace, while ``--records`` answered exit 2
    with a reason and every adaptation mode printed ``{"mode": ..., "error": ...}``. Three legs of one
    CLI behaving as if a typo were an internal error is what this exists to stop; the payload shape
    and the exit code are copied from the adaptation modes rather than invented here.
    """
    (log or logger).error("%s refused: %s", mode, message)
    print(json.dumps({"mode": mode, "error": message}, sort_keys=True))
    return 2


def _records_mode(records_path: Path) -> int:
    """A shim over :class:`RecordLog`. The roll-up, the two audits and the exit code live there."""
    rollup = RecordLog.from_jsonl(records_path).kpis()
    payload = {
        "mode": "records",
        "records_path": str(records_path),
        "kpi": dict(rollup.kpi),
        # The rates this log cannot measure, named. They used to print as numbers here: an empty
        # denominator returns 0.0, so `false_positive_grasp_rate: 0.0` read as "no false positives"
        # rather than "nothing on this stack writes the field that rate divides by". The operator
        # console said so for months while this CLI, over the same records, did not.
        "unmeasurable": dict(rollup.unmeasurable),
        "telemetry_offenders": [
            {"attempt_id": o.attempt_id, "missing": list(o.missing)}
            for o in rollup.missing_telemetry
        ],
        "extra_type_offenders": [
            {"attempt_id": o.attempt_id, "bad_fields": list(o.bad_types)}
            for o in rollup.wrong_types
        ],
    }
    log = logger.info if rollup.sound else logger.warning
    log(
        "KPI roll-up over %d record(s) from %s: %d telemetry offender(s), "
        "%d extra-type offender(s)",
        rollup.records,
        records_path,
        len(rollup.missing_telemetry),
        len(rollup.wrong_types),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return rollup.exit_code


def _records_gate_mode(records_path: Path) -> int:
    """Apply the record-intrinsic soak thresholds over a real GraspAttemptRecord log.

    Unlike ``--soak-report`` (a synthetic self-check that passes by construction), this runs the same
    7 record-judgeable gate keys over a captured log and can fail. The pack-dependent keys
    (slo/drift/ood/easy-wall-time) need the on-disk canonical packs and are reported ``not_applicable``.
    Exit 0 iff no violations, else 1.
    """

    verdict = SoakGate.over_records(records_path).evaluate()
    if verdict.unreadable:
        return _refuse_missing_input(
            "records-gate", f"{verdict.unreadable}: {records_path}"
        )
    violations = list(verdict.violations)
    baseline_pick = verdict.baseline_pick_rate
    gate = verdict.gate_wire()
    payload = {
        "mode": "records-gate",
        "records_path": str(records_path),
        "input_provenance": "real_grasp_attempt_record_log",
        "note": (
            "record-intrinsic soak thresholds over a REAL log: unlike --soak-report's synthetic "
            "self-check, this CAN fail. Pack-dependent keys (slo/drift/ood/wall-time) are "
            "not_applicable without the on-disk canonical packs."
        ),
        "baseline_pick_rate": baseline_pick,
        "gate": gate,
        "violations": list(violations),
    }
    log = logger.info if not violations else logger.warning
    log(
        "Records gate over %s: passes=%s%s",
        records_path,
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return verdict.exit_code


def _sim_soak_report_mode(
    records_path: Path, out_override: Path | None, *, min_attempts: int
) -> int:
    """Real-sim-records soak quality gate + persistent report over a real Isaac-sim GraspAttemptRecord log.

    ``pick_success_rate`` is reported but not gated: ``baseline_pick_rate=None``, so there is no
    cross-population comparison against the synthetic baseline, which is a different distribution. The
    record-intrinsic integrity keys (min_attempts/untyped/unbounded/telemetry/extra/dead_loop) gate.
    ``false_positive_grasp_rate`` is shown but structurally 0 (no secondary verifier in sim). Pack-dependent
    keys are not_applicable. Exit 0 iff no violations.
    """
    verdict = SoakGate.over_sim_records(
        records_path, min_attempts=min_attempts
    ).evaluate()
    if verdict.unreadable:
        # Before the write, deliberately. A report file for a log nobody could open is a document
        # asserting a gate ran, and it would outlive the terminal that said otherwise.
        return _refuse_missing_input(
            "sim-soak-report", f"{verdict.unreadable}: {records_path}"
        )
    violations = list(verdict.violations)
    gate = verdict.gate_wire()
    payload = {
        "mode": "sim-soak-report",
        "report_version": 1,
        "records_path": str(records_path),
        "provenance": {
            "input": "real_sim_grasp_record_log",
            "measures": [
                "record_intrinsic_integrity_over_real_sim_picks",
                "sim_pick_success_rate_reported_not_gated",
            ],
            "does_not_measure": [
                "real_hardware_grasp_quality",
                "false_positive_grasp_rate (structurally 0: no secondary verifier in sim)",
            ],
            "note": (
                "Real Isaac-physics sim picks (not hardware-representative). Gates the record-intrinsic "
                "integrity keys; pick_success_rate is reported but not gated (no comparable sim baseline "
                "yet: baseline_pick_rate=None; the synthetic baseline is a different distribution). "
                "Pack-dependent keys (slo/drift/ood/wall-time) need the on-disk canonical packs: "
                f"not_applicable. sim min_attempts floor={min_attempts} (not the synthetic 2000). Run "
                "--soak-report for the contract self-check; this --sim-soak-report is the quality signal."
            ),
        },
        "total_attempts": verdict.attempts,
        "min_attempts_floor": int(min_attempts),
        "baseline_pick_rate": verdict.baseline_pick_rate,
        "kpi": dict(verdict.kpi),
        "gate": gate,
        "violations": list(violations),
    }
    repo_root = repo_root_from_module()
    out_path = (
        out_override if out_override is not None
        else repo_root / _SIM_SOAK_REPORT_RELATIVE_PATH
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.write_text(report_bytes)
    log = logger.info if not violations else logger.warning
    log(
        "Sim-soak report over %d record(s) from %s -> %s (%d bytes): passes=%s%s",
        verdict.attempts,
        records_path,
        out_path,
        len(report_bytes),
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(
        json.dumps(
            {
                "mode": "sim-soak-report",
                "report_path": str(out_path),
                "gate_passes": gate["passes"],
                "violations": list(violations),
                "total_attempts": verdict.attempts,
                "min_attempts_floor": int(min_attempts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return verdict.exit_code


def _soak_mode(thresholds_path: Path) -> int:
    """A shim over :meth:`SoakGate.over_synthetic`. The three scenarios and the thresholds live there."""
    verdict = SoakGate.over_synthetic(thresholds_path).evaluate()
    violations = list(verdict.violations)
    payload = {
        "mode": "soak",
        "attempts": verdict.attempts,
        "thresholds_path": str(thresholds_path),
        "kpi": dict(verdict.kpi),
        "violations": violations,
    }
    log = logger.info if not violations else logger.warning
    log(
        "Synthetic soak over %d generated attempt(s) (thresholds %s): passes=%s%s",
        verdict.attempts,
        thresholds_path,
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return verdict.exit_code


def _soak_report_mode(
    out_override: Path | None,
    *,
    total_attempts: int = SOAK_DEFAULT_ATTEMPTS,
) -> int:
    """Build the soak report, write it to disk, and exit non-zero on any locked violation."""

    verdict = SoakGate.over_canonical_packs(total_attempts=total_attempts).evaluate()
    payload = dict(verdict.report)
    violations = list(verdict.violations)
    repo_root = repo_root_from_module()
    if out_override is not None:
        out_path = out_override
    else:
        out_path = repo_root / DEFAULT_SOAK_REPORT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.write_text(report_bytes)
    log = logger.info if not violations else logger.warning
    log(
        "Soak report (%d synthetic attempt(s)) -> %s (%d bytes): passes=%s%s",
        int(cast("int", payload["total_attempts"])),
        out_path,
        len(report_bytes),
        not violations,
        "" if not violations else f" violations: {'; '.join(violations)}",
    )
    print(
        json.dumps(
            {
                "mode": "soak-report",
                "report_path": str(out_path),
                "gate_passes": cast("dict[str, Any]", payload["gate"])["passes"],
                "violations": list(violations),
                "total_attempts": payload["total_attempts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return verdict.exit_code
