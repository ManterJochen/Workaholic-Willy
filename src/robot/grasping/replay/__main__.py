"""CLI entrypoint for the replay harness.

Usage
-----

Compute KPIs over a captured JSONL log::

    python -m src.robot.grasping.replay --records run.jsonl

Run the locked synthetic soak gate (deterministic, in-process)::

    python -m src.robot.grasping.replay --soak

Regenerate the canonical replay packs and manifest on
disk (deterministic; commits the bytes that downstream tooling
hashes)::

    python -m src.robot.grasping.replay --regenerate-canonical

Build the baseline KPI/SLO/telemetry report
(``docs/baselines/u_plus_baseline_v1.json`` by default)::

    python -m src.robot.grasping.replay --baseline-report

All modes print a JSON summary to stdout. ``--soak`` exits non-zero
if any locked KPI threshold is violated; ``--baseline-report``
exits non-zero if any canonical pack fails its telemetry audit.
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
from src.robot.grasping.replay.runs import Baseline
from src.robot.grasping.replay.baseline_report import (
    DEFAULT_REPORT_RELATIVE_PATH,
)
from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    regenerate_all,
    repo_root_from_module,
)
from src.robot.grasping.replay.failure_taxonomy import (
    build_taxonomy_report,
    write_report,
)
from src.robot.grasping.replay.soak import (
    DEFAULT_SOAK_REPORT_RELATIVE_PATH,
    SOAK_DEFAULT_ATTEMPTS,
)
from src.robot.grasping.replay.watchdog_eval import (
    evaluate_drift_pack_path,
    evaluate_ood_pack_path,
)
from src.robot.grasping.replay.slo_eval import (
    evaluate_slo_pack_path,
)

from src.robot.grasping.replay.soak_cli import (
    _SIM_SOAK_REPORT_RELATIVE_PATH,
    _records_gate_mode,
    _records_mode,
    _refuse_missing_input,
    _sim_soak_report_mode,
    _soak_mode,
    _soak_report_mode,
)
from src.robot.grasping.replay.adaptation_cli import (
    _adaptation_apply_mode,
    _adaptation_plan_mode,
    _adaptation_rollback_mode,
    _adaptation_verify_mode,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_THRESHOLDS = (
    _REPO_ROOT / "config" / "robot" / "kpi_thresholds.yaml"
)

# The sim-soak quality gate. A real on-box sim run of the synthetic 2000-attempt floor would take
# hours; the sim-soak states a smaller floor in its report instead. Default output sits beside the
# synthetic soak_report.json, which stays byte-identical: the sim-soak augments it.
_SIM_SOAK_DEFAULT_MIN_ATTEMPTS = 300


#: Each mode prints its JSON summary to stdout. The gate verdict and the report
#: path also go to a file log.
logger = create_grasping_logger("ReplayCLI", REPLAY_CLI_LOG_FILE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grasping-replay",
        description=(
            "replay harness: KPI rollup over real JSONL "
            "logs, the locked synthetic soak gate, canonical-pack "
            "regeneration, and the baseline report."
        ),
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
            "apply the record-intrinsic soak thresholds over a REAL GraspAttemptRecord log "
            "(unlike --soak-report's synthetic self-check, this CAN fail). Pack-dependent keys "
            "(slo/drift/ood/wall-time) are reported not_applicable. Exit 0 iff no violations, else 1."
        ),
    )
    group.add_argument(
        "--soak",
        action="store_true",
        help="run the synthetic soak gate against the locked thresholds",
    )
    group.add_argument(
        "--regenerate-canonical",
        action="store_true",
        help=(
            "regenerate the canonical replay packs and "
            "manifest on disk (deterministic)"
        ),
    )
    group.add_argument(
        "--baseline-report",
        action="store_true",
        help=(
            "build the baseline KPI/SLO/telemetry report "
            f"(default output: {DEFAULT_REPORT_RELATIVE_PATH})"
        ),
    )
    group.add_argument(
        "--soak-report",
        action="store_true",
        help=(
            "SYNTHETIC self-check (NOT a hardware quality gate): run the locked "
            f">= {SOAK_DEFAULT_ATTEMPTS}-attempt synthetic soak and verify the "
            "telemetry->KPI->taxonomy->SLO->watchdog pipeline is internally CONSISTENT (the outcome "
            "distribution is authored, so 'gate.passes' proves contract-consistency, not real grasp "
            "quality). Compares the on-disk packs against the committed U+ baseline and writes the report "
            f"(default: {DEFAULT_SOAK_REPORT_RELATIVE_PATH}). For a real quality signal use --records-gate."
        ),
    )
    group.add_argument(
        "--sim-soak-report",
        type=Path,
        metavar="RECORDS_JSONL",
        help=(
            "Real-SIM-records soak QUALITY gate + persistent report (AUGMENTS the synthetic "
            "--soak-report, which stays byte-identical). Runs the record-intrinsic gate over a REAL "
            "Isaac-sim GraspAttemptRecord log, writes the report (default: "
            f"{_SIM_SOAK_REPORT_RELATIVE_PATH}, or --baseline-out), exits 0 iff no violations. "
            "pick_success_rate is REPORTED but not gated; sim min_attempts floor via --sim-min-attempts."
        ),
    )
    group.add_argument(
        "--failure-taxonomy",
        type=Path,
        nargs="+",
        metavar="PACK",
        help=(
            "failure-taxonomy report: classify records in "
            "one or more JSONL packs and write a deterministic JSON "
            "report (path supplied via --out)."
        ),
    )
    group.add_argument(
        "--watchdog-eval",
        action="store_true",
        help=(
            "watchdog KPI evaluator: score the canonical "
            "drift + OOD packs against the locked precision/recall "
            "gates (drift >=0.90/0.85, OOD >=0.90/0.80). Exits "
            "non-zero on gate failure."
        ),
    )
    group.add_argument(
        "--slo-gate",
        action="store_true",
        help=(
            "latency SLO evaluator: score every canonical "
            "pack's per-stage decision/ranking/fusion p95 against "
            "the locked budgets (60/80/220 ms). Exits non-zero on "
            "gate failure."
        ),
    )
    group.add_argument(
        "--adaptation-plan",
        action="store_true",
        help=(
            "guarded adaptation: build an AdaptationPlan from "
            "a baseline-report (--baseline-in) and an optional "
            "failure-taxonomy report (--taxonomy-in). Emits the plan "
            "as JSON on stdout. Default mode is 'recommend_only'."
        ),
    )
    group.add_argument(
        "--adaptation-verify",
        type=Path,
        metavar="PLAN_JSON",
        help=(
            "validate an AdaptationPlan JSON against the "
            "current schema allow-list and per-key bounds. Exits "
            "non-zero on any validation issue."
        ),
    )
    group.add_argument(
        "--adaptation-apply",
        type=Path,
        metavar="PLAN_JSON",
        help=(
            "apply an AdaptationPlan by writing a YAML "
            "overlay sidecar and refreshing the active overlay "
            "pointer. Requires mode='apply_with_guardrails'."
        ),
    )
    group.add_argument(
        "--adaptation-rollback",
        type=str,
        metavar="PLAN_ID",
        help=(
            "roll back the currently active overlay by "
            "clearing the active overlay pointer and appending a "
            "'rollback' audit entry referencing PLAN_ID."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=_DEFAULT_THRESHOLDS,
        help="path to kpi_thresholds.yaml (default: shipped artefact)",
    )
    parser.add_argument(
        "--sim-min-attempts",
        type=int,
        default=_SIM_SOAK_DEFAULT_MIN_ATTEMPTS,
        help=(
            f"Honest min_attempts floor for --sim-soak-report (default "
            f"{_SIM_SOAK_DEFAULT_MIN_ATTEMPTS}; a sim run of the synthetic 2000+ floor would take hours)."
        ),
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=None,
        help=(
            "the older spelling of --out, and exactly the same thing: the output path for "
            f"--baseline-report (default: {DEFAULT_REPORT_RELATIVE_PATH}), --soak-report "
            f"(default: {DEFAULT_SOAK_REPORT_RELATIVE_PATH}) or --sim-soak-report. Passing both "
            "names with different paths is an error, not a preference."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output path for whichever report mode is selected: --failure-taxonomy (where it is "
            "required), --baseline-report, --soak-report or --sim-soak-report. Every writing mode "
            "honours it; none of them falls back to the committed default once it is given."
        ),
    )
    parser.add_argument(
        "--baseline-in",
        type=Path,
        default=None,
        help=(
            "input baseline JSON for --adaptation-plan "
            "(defaults to docs/baselines/u_plus_baseline_v1.json)"
        ),
    )
    parser.add_argument(
        "--taxonomy-in",
        type=Path,
        default=None,
        help=(
            "optional taxonomy report JSON for "
            "--adaptation-plan"
        ),
    )
    parser.add_argument(
        "--adaptation-mode",
        type=str,
        default="recommend_only",
        choices=("off", "recommend_only", "apply_with_guardrails"),
        help="plan mode (default: recommend_only)",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help=(
            "overlay sidecar directory "
            "(default: configs/overlays)"
        ),
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help=(
            "audit JSONL path "
            "(default: logs/adaptation/adaptation_audit.jsonl)"
        ),
    )
    parser.add_argument(
        "--no-auto-rollback",
        action="store_true",
        default=False,
        help=(
            "disable the auto-rollback guardrail on "
            "--adaptation-apply. By default an apply that produces "
            "a runtime-SLO regression > --regression-tolerance-ms is "
            "rolled back automatically and the audit log records the "
            "auto-rollback."
        ),
    )
    parser.add_argument(
        "--regression-tolerance-ms",
        type=float,
        default=0.0,
        help=(
            "tolerance (ms) for the post-apply "
            "regression check. Stage p95 deltas > this value trigger "
            "the auto-rollback guardrail (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--post-apply-report",
        type=Path,
        default=None,
        help=(
            "path to a baseline-report JSON RE-MEASURED under the applied overlay "
            "(the 'after' signal for the auto-rollback guardrail). Without it the guardrail is a "
            "structural no-op (after==before, signal_available=false); with it, a real regression "
            "in the supplied report triggers an actual in-place revert."
        ),
    )
    args = parser.parse_args(argv)

    # One output path, two spellings, and one of them used to be silently discarded.
    # `--baseline-report --out mine.json` read `args.baseline_out`, found `None`, and rewrote the
    # git-tracked `docs/baselines/u_plus_baseline_v1.json` instead: an operator who named their own
    # file overwrote a committed one and was told the report had been written. Resolving both names
    # here, once, is what makes an ignored argument impossible rather than merely unlikely.
    if (
        args.out is not None
        and args.baseline_out is not None
        and args.out != args.baseline_out
    ):
        parser.error(
            "--out and --baseline-out name the same thing and disagree "
            f"({args.out} vs {args.baseline_out}); pass one"
        )
    report_out = args.out if args.out is not None else args.baseline_out

    if args.records is not None:
        return _records_mode(args.records)
    if args.records_gate is not None:
        return _records_gate_mode(args.records_gate)
    if args.soak:
        return _soak_mode(args.thresholds)
    if args.regenerate_canonical:
        return _regenerate_canonical_mode()
    if args.baseline_report:
        return _baseline_report_mode(report_out)
    if args.soak_report:
        return _soak_report_mode(report_out)
    if args.sim_soak_report is not None:
        return _sim_soak_report_mode(
            args.sim_soak_report, report_out, min_attempts=args.sim_min_attempts
        )
    if args.failure_taxonomy is not None:
        if report_out is None:
            parser.error("--failure-taxonomy requires --out")
        return _failure_taxonomy_mode(args.failure_taxonomy, report_out)
    if args.watchdog_eval:
        return _watchdog_eval_mode()
    if args.slo_gate:
        return _slo_gate_mode()
    if args.adaptation_plan:
        return _adaptation_plan_mode(
            baseline_in=args.baseline_in,
            taxonomy_in=args.taxonomy_in,
            mode=args.adaptation_mode,
        )
    if args.adaptation_verify is not None:
        return _adaptation_verify_mode(args.adaptation_verify)
    if args.adaptation_apply is not None:
        return _adaptation_apply_mode(
            plan_path=args.adaptation_apply,
            overlay_dir=args.overlay_dir,
            audit_path=args.audit_path,
            auto_rollback_on_regression=not args.no_auto_rollback,
            regression_tolerance_ms=float(args.regression_tolerance_ms),
            post_apply_report_path=args.post_apply_report,
        )
    if args.adaptation_rollback is not None:
        return _adaptation_rollback_mode(
            plan_id=args.adaptation_rollback,
            overlay_dir=args.overlay_dir,
            audit_path=args.audit_path,
        )
    parser.error("no mode selected")
    return 2  # pragma: no cover


def _regenerate_canonical_mode() -> int:
    repo_root = repo_root_from_module()
    written = regenerate_all(repo_root)
    payload = {
        "mode": "regenerate-canonical",
        "repo_root": str(repo_root),
        "written": {name: str(p) for name, p in written.items()},
    }
    logger.info(
        "Regenerated %d canonical pack(s) under %s: %s",
        len(written),
        repo_root,
        ", ".join(sorted(written)),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _baseline_report_mode(out_override: Path | None) -> int:
    """A shim over :class:`Baseline`. The measurement and the offender total live there.

    ``out_override`` need not sit inside the repository tree: the resolved absolute path is
    written as given, with no repo-relative rewriting.
    """
    measured = Baseline.canonical().measure()
    out_path = (
        out_override.resolve()
        if out_override is not None
        else (repo_root_from_module() / DEFAULT_REPORT_RELATIVE_PATH).resolve()
    )
    measured.write(out_path)
    payload = {
        "mode": "baseline-report",
        "report_path": str(out_path),
        "total_audit_offenders": measured.audit_offenders,
        "report": dict(measured.report),
    }
    offenders_total = measured.audit_offenders
    log = logger.info if offenders_total == 0 else logger.warning
    log(
        "Baseline report written to %s: %d telemetry audit offender(s)",
        out_path,
        int(offenders_total),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return measured.exit_code


def _failure_taxonomy_mode(
    pack_paths: Sequence[Path], out_path: Path
) -> int:
    """Classify records in JSONL packs and write the taxonomy report.

    Always returns 0: coverage is a reported KPI, not an exit code.
    """

    records: list = []
    pack_strs: list[str] = []
    for path in pack_paths:
        if not path.is_file():
            # Was `raise FileNotFoundError`, meaning a stack trace for a mistyped path. Every
            # adaptation mode in this same CLI answers the identical mistake with one sentence and
            # exit 2, and an operator cannot tell a traceback from a broken installation.
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


def _watchdog_eval_mode() -> int:
    """Score the canonical drift + OOD packs against the watchdog KPI gates.

    Exit 0 iff both pass, else 2.
    """

    repo_root = repo_root_from_module()
    drift_path = (
        repo_root
        / "tests"
        / "data"
        / "replay"
        / "replay_drift_synthetic_v1.jsonl"
    )
    ood_path = (
        repo_root
        / "tests"
        / "data"
        / "replay"
        / "replay_ood_synthetic_v1.jsonl"
    )
    drift_report = evaluate_drift_pack_path(drift_path)
    ood_report = evaluate_ood_pack_path(ood_path)
    payload = {
        "mode": "watchdog-eval",
        "drift": drift_report.to_dict(),
        "ood": ood_report.to_dict(),
        "passes_gate": bool(
            drift_report.passes_gate and ood_report.passes_gate
        ),
    }
    log = logger.info if payload["passes_gate"] else logger.warning
    log(
        "Watchdog eval: drift pass=%s, ood pass=%s (packs %s, %s)",
        drift_report.passes_gate,
        ood_report.passes_gate,
        drift_path.name,
        ood_path.name,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passes_gate"] else 2


def _slo_gate_mode() -> int:
    """Score every canonical pack's per-stage p95 against the SLO budgets.

    Exit 0 iff all required stages pass; optional empty stages pass automatically.
    """

    repo_root = repo_root_from_module()
    pack_reports: list[dict] = []
    all_pass = True
    for pack in CANONICAL_PACKS:
        path = (repo_root / pack.relative_path).resolve()
        report = evaluate_slo_pack_path(path, pack_name=pack.name)
        pack_reports.append(report.to_dict())
        if not report.passes_gate:
            all_pass = False
    payload = {
        "mode": "slo-gate",
        "packs": pack_reports,
        "passes_gate": bool(all_pass),
    }
    log = logger.info if all_pass else logger.warning
    log(
        "SLO gate over %d canonical pack(s): passes=%s",
        len(pack_reports),
        all_pass,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())