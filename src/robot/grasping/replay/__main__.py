"""CLI entrypoint for the replay harness.

Usage
-----

Compute KPIs over a captured JSONL log::

    python -m src.robot.grasping.replay --records run.jsonl

Judge a real log against the soak thresholds, which can fail::

    python -m src.robot.grasping.replay --records-gate run.jsonl

Classify the failures of one or more logs::

    python -m src.robot.grasping.replay --failure-taxonomy run.jsonl --out taxonomy.json

All modes print a JSON summary to stdout. The synthetic soak, the canonical packs, the baseline report and the
watchdog, SLO and adaptation modes left on 2026-10-16 (the owner's cleanup).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


from src.robot.grasping.constants import (
    REPLAY_CLI_LOG_FILE,
    create_grasping_logger,
)
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.replay.failure_taxonomy import (
    build_taxonomy_report,
    write_report,
)

from src.robot.grasping.replay.soak_cli import (
    _SIM_SOAK_REPORT_RELATIVE_PATH,
    _records_gate_mode,
    _records_mode,
    _refuse_missing_input,
    _sim_soak_report_mode,
)


# The sim-soak quality gate. A real on-box sim run of the 2000-attempt floor would take hours; the sim-soak states
# a smaller floor in its report instead.
_SIM_SOAK_DEFAULT_MIN_ATTEMPTS = 300


#: Each mode prints its JSON summary to stdout. The gate verdict and the report
#: path also go to a file log.
logger = create_grasping_logger("ReplayCLI", REPLAY_CLI_LOG_FILE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grasping-replay",
        description="replay harness: KPI rollup and the soak gate over real JSONL logs, and the failure taxonomy.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--records",
        type=Path,
        help="path to a JSONL file of GraspAttemptRecord entries",
    )
    group.add_argument(
        "--records-gate",
        type=Path,
        metavar="RECORDS_JSONL",
        help=(
            "apply the soak thresholds over a REAL GraspAttemptRecord log; this CAN fail. "
            "Exit 0 iff no violations, else 1."
        ),
    )
    group.add_argument(
        "--sim-soak-report",
        type=Path,
        metavar="RECORDS_JSONL",
        help=(
            "Real-SIM-records soak QUALITY gate + persistent report. Runs the record-intrinsic gate over a REAL "
            f"Isaac-sim GraspAttemptRecord log, writes the report (default: {_SIM_SOAK_REPORT_RELATIVE_PATH}, or "
            "--out), exits 0 iff no violations. pick_success_rate is REPORTED but not gated; sim min_attempts "
            "floor via --sim-min-attempts."
        ),
    )
    group.add_argument(
        "--failure-taxonomy",
        type=Path,
        nargs="+",
        metavar="PACK",
        help=(
            "failure-taxonomy report: classify records in "
            "one or more JSONL logs and write a deterministic JSON "
            "report (path supplied via --out)."
        ),
    )
    parser.add_argument(
        "--sim-min-attempts",
        type=int,
        default=_SIM_SOAK_DEFAULT_MIN_ATTEMPTS,
        help=(
            f"Honest min_attempts floor for --sim-soak-report (default "
            f"{_SIM_SOAK_DEFAULT_MIN_ATTEMPTS}; a sim run of the 2000+ floor would take hours)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output path for --failure-taxonomy (where it is required) or --sim-soak-report; once it is given, "
            "neither falls back to its default."
        ),
    )
    args = parser.parse_args(argv)

    if args.records is not None:
        return _records_mode(args.records)
    if args.records_gate is not None:
        return _records_gate_mode(args.records_gate)
    if args.sim_soak_report is not None:
        return _sim_soak_report_mode(
            args.sim_soak_report, args.out, min_attempts=args.sim_min_attempts
        )
    if args.failure_taxonomy is not None:
        if args.out is None:
            parser.error("--failure-taxonomy requires --out")
        return _failure_taxonomy_mode(args.failure_taxonomy, args.out)
    parser.error("no mode selected")
    return 2  # pragma: no cover


def _failure_taxonomy_mode(
    pack_paths: Sequence[Path], out_path: Path
) -> int:
    """Classify records in JSONL logs and write the taxonomy report.

    Always returns 0: coverage is a reported KPI, not an exit code.
    """

    records: list = []
    pack_strs: list[str] = []
    for path in pack_paths:
        if not path.is_file():
            # Was `raise FileNotFoundError`, meaning a stack trace for a mistyped path; an operator cannot tell a
            # traceback from a broken installation.
            return _refuse_missing_input(
                "failure-taxonomy", f"replay pack not found: {path}", log=logger
            )
        records.extend(iter_jsonl(path))
        pack_strs.append(str(path))
    report = build_taxonomy_report(records, pack_paths=tuple(pack_strs))
    written = write_report(report, out_path)
    payload = {
        "mode": "failure-taxonomy",
        "report_path": str(written),
        "total_records": report.total_records,
        "failure_count": report.failure_count,
        "classified_failure_count": report.classified_failure_count,
        "unclassified_failure_count": report.unclassified_failure_count,
        "coverage_fraction": report.coverage_fraction,
    }
    logger.info(
        "Failure taxonomy over %d pack(s) -> %s: %d record(s), %d failure(s), "
        "%.1f%% classified",
        len(pack_strs),
        written,
        report.total_records,
        report.failure_count,
        report.coverage_fraction * 100.0,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
