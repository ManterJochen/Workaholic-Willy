"""Eval CLI command group (ope / promote-policy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.robot.grasping.constants import (
    RL_COMMANDS_EVAL_LOG_FILE,
    create_grasping_logger,
)

from .ope import (
    MalformedOPEInputError,
    build_ope_report,
    load_records_for_ope,
    sequencing_target_action_from_policy_path,
)
from .promotion import (
    POLICY_FAMILIES,
    POLICY_FAMILY_PERCEPTION,
    PromotionInputError,
    PromotionThresholds,
    build_promotion_report_artifact,
    evaluate_policy_for_promotion,
)


from ._cli_common import (
    COMMITTED_MANIFEST_DIR_REL,
    COMMITTED_RL_POLICY_DIR_REL,
    DEFAULT_OPE_REPLAY_PACKS,
    OPE_EXIT_MALFORMED_INPUT,
    SEQUENCING_POLICY_FILENAME,
    OPE_REPORT_FILENAME,
    PERCEPTION_PROMOTION_REPORT_FILENAME,
    RECOVERY_PROMOTION_REPORT_FILENAME,
    _relpath,
    _resolve_repo_root,
    _resolve_under_repo,
)


#: The promotion verdict and the OPE estimate are logged by their own modules. What
#: only exists here is the invocation (which artifact, which packs, which
#: thresholds) and the input refusals that abort before either of them runs.
logger = create_grasping_logger("RLEvalCLI", RL_COMMANDS_EVAL_LOG_FILE)


def _cmd_promote_policy(args: argparse.Namespace) -> int:
    """Promotion gate (offline, report-only)."""

    repo_root = _resolve_repo_root(args.repo_root)
    family = args.policy_family
    if family not in POLICY_FAMILIES:
        logger.error("promote-policy refused: unknown --policy-family %r", family)
        print(
            f"promote-policy: unknown --policy-family {family!r}; "
            f"supported: {POLICY_FAMILIES}",
            file=sys.stderr,
        )
        return 2

    artifact_arg = _resolve_under_repo(args.policy_artifact, repo_root)

    if args.replay_pack:
        pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
    else:
        pack_paths = [repo_root / rel for rel in DEFAULT_OPE_REPLAY_PACKS]

    thresholds = PromotionThresholds(
        min_lift_over_baseline=float(args.min_lift_over_baseline),
        min_lower_bound_lift=float(args.min_lower_bound_lift),
        min_n_effective=float(args.min_n_effective),
        min_records_with_weight=int(args.min_records_with_weight),
    )

    logger.info(
        "promote-policy %s: artifact %s over %d replay pack(s), dataset %r, scope "
        "%s, seed %d, thresholds min_lift %.4f / min_ci_lower %.4f / min_n_eff "
        "%.1f / min_records_with_weight %d",
        family,
        artifact_arg,
        len(pack_paths),
        args.dataset_id,
        args.training_scope,
        int(args.seed),
        thresholds.min_lift_over_baseline,
        thresholds.min_lower_bound_lift,
        thresholds.min_n_effective,
        thresholds.min_records_with_weight,
    )
    try:
        report = evaluate_policy_for_promotion(
            policy_family=family,
            policy_artifact_path=artifact_arg,
            pack_paths=pack_paths,
            dataset_id=args.dataset_id,
            training_scope=args.training_scope,
            seed=int(args.seed),
            thresholds=thresholds,
            repo_root=repo_root,
        )
    except PromotionInputError as exc:
        logger.error("promote-policy refused: %s", exc)
        print(f"promote-policy: {exc}", file=sys.stderr)
        return 2
    # `PromotionGate` is the Python caller's door to the same evaluation, and it requires
    # `repo_root` as a keyword. Omitting it moves `dataset_hash` and turns `dataset_paths` into
    # backslash strings, with the same verdict and no warning.

    if args.output:
        out_path = _resolve_under_repo(args.output, repo_root)
    else:
        default_name = (
            PERCEPTION_PROMOTION_REPORT_FILENAME
            if family == POLICY_FAMILY_PERCEPTION
            else RECOVERY_PROMOTION_REPORT_FILENAME
        )
        out_path = repo_root / COMMITTED_RL_POLICY_DIR_REL / default_name

    # A promotion report's sha is its identity, so the digest has to describe the bytes on disk.
    # `write_promotion_report` builds the blob with a bare newline, writes it with `write_text`
    # (which translates that newline to a carriage-return pair on Windows) and hashes the
    # pre-translation bytes, so its digest does not match the file it wrote.
    # `PromotionGateReport.write` writes bytes, which makes the digest true and makes the file
    # match what every other machine produces.
    from .evaluation import PromotionGateReport, PromotionVerdict

    gate_report = PromotionGateReport(
        verdict=PromotionVerdict(str(report.verdict)),
        artifact=build_promotion_report_artifact(report),
        reasons=tuple(report.reasons),
    )
    sha = gate_report.write(out_path)

    artifact = build_promotion_report_artifact(report)
    summary = {
        "verdict": report.verdict,
        "reasons": list(report.reasons),
        "policy_id": report.policy_id,
        "policy_family": report.policy_family,
        "dataset_id": report.dataset_id,
        "num_triples": artifact["extraction"]["num_triples"],
        "wis_target_value": artifact["estimators"]["wis"]["target_value"],
        "wis_baseline_value": artifact["estimators"]["wis"]["baseline_value"],
        "wis_lift": artifact["estimators"]["wis"]["lift"],
        "wis_lift_ci_lower": (
            artifact["estimators"]["wis"]["lift_ci_bootstrap"]["lower"]
        ),
        "dm_value": artifact["estimators"]["direct_method"]["value"],
        "output_path": str(out_path),
        "output_sha256": sha,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _cmd_ope(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    # `--replay-pack` overrides the default canonical pair; when neither the override nor the
    # defaults are on disk, the fallback below reads the sequencing training dataset manifest
    # splits.
    if args.replay_pack:
        pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
    else:
        pack_paths = [repo_root / rel for rel in DEFAULT_OPE_REPLAY_PACKS]
        if not all(p.exists() for p in pack_paths):
            # Fallback: read the split paths out of the dataset manifest.
            manifest_path = (
                repo_root
                / COMMITTED_MANIFEST_DIR_REL
                / f"{args.fallback_dataset_id}.json"
            )
            if not manifest_path.exists():
                logger.error(
                    "OPE refused: canonical replay packs missing and no fallback "
                    "manifest at %s",
                    manifest_path,
                )
                print(
                    f"OPE: canonical replay packs missing and fallback "
                    f"manifest not found at {manifest_path}",
                    file=sys.stderr,
                )
                return OPE_EXIT_MALFORMED_INPUT
            logger.warning(
                "OPE falling back to the %r dataset manifest splits: the canonical "
                "replay packs are not all on disk",
                args.fallback_dataset_id,
            )
            # This fallback cannot work, and it is left that way on purpose: it reads a `"splits"`
            # key that no manifest carries. A manifest carries `split_files`, `split_counts`,
            # `split_hashes` and `class_counts_by_split`, so the list is always empty and the run
            # dies a step later with "empty after loading all paths", at a place with nothing to do
            # with the cause. Repairing it changes which records an OPE report is computed over,
            # which is a deliberate, announced change and not a side effect of a key rename.
            manifest = json.loads(manifest_path.read_text())
            split_paths = manifest.get("splits") or {}
            pack_paths = [
                repo_root / rel for rel in split_paths.values() if rel
            ]

    try:
        records = load_records_for_ope(pack_paths)
    except MalformedOPEInputError as exc:
        logger.error("OPE refused: malformed input: %s", exc)
        print(f"OPE: malformed input: {exc}", file=sys.stderr)
        return OPE_EXIT_MALFORMED_INPUT

    sequencing_path = _resolve_under_repo(args.sequencing_artifact, repo_root)
    if not sequencing_path.exists():
        logger.error(
            "OPE refused: sequencing artifact not found at %s", sequencing_path
        )
        print(
            f"OPE: sequencing artifact not found at {sequencing_path}",
            file=sys.stderr,
        )
        return OPE_EXIT_MALFORMED_INPUT
    sequencing_target = sequencing_target_action_from_policy_path(sequencing_path)

    try:
        report = build_ope_report(
            records=records,
            sequencing_target_action_for_state=sequencing_target,
            dataset_id=args.dataset_id,
            dataset_paths=[_relpath(p, repo_root) for p in pack_paths],
            rng_seed=args.seed,
        )
    except MalformedOPEInputError as exc:
        logger.error("OPE refused: malformed input during report build: %s", exc)
        print(f"OPE: malformed input during report build: {exc}", file=sys.stderr)
        return OPE_EXIT_MALFORMED_INPUT

    out_path = (
        Path(args.output)
        if args.output
        else repo_root / COMMITTED_RL_POLICY_DIR_REL / OPE_REPORT_FILENAME
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, sort_keys=True, indent=2) + "\n"
    # Bytes, not write_text. MEASURED 2026-09-10: write_text translated the rendered line
    # feed to a carriage-return pair on Windows, so this handler could not reproduce the
    # committed LF golden it tells the operator to re-run, the same defect
    # PromotionGateReport.write already fixed for promotion reports. It also had no
    # encoding= and inherited the locale.
    out_path.write_bytes(body.encode("utf-8"))
    logger.info(
        "Wrote OPE report to %s (%d bytes)", out_path, len(body.encode("utf-8"))
    )

    summary = {
        "schema_version": report["schema_version"],
        "report_kind": report["report_kind"],
        "dataset_id": report["dataset_id"],
        "num_records": report["num_records"],
        "output_path": str(out_path),
        "wis_candidate": report["sections"]["candidate"]["estimators"]["wis"]["value"],
        "wis_ranking": report["sections"]["ranking"]["estimators"]["wis"]["value"],
        "wis_sequencing": report["sections"]["sequencing"]["estimators"]["wis"]["value"],
        "dm_sequencing": report["sections"]["sequencing"]["estimators"]["direct_method"]["value"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def register_eval_commands(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    op = sub.add_parser(
        "ope",
        help=(
            "off-policy evaluation harness over the candidate/ranking/"
            "sequencing shadow telemetry"
        ),
    )
    op.add_argument(
        "--dataset-id",
        default="v5_ope_canonical",
        help="stable id surfaced in the report metadata",
    )
    op.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed")
    op.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help=(
            "override replay-pack path(s) (repeatable). Default: "
            "tests/data/replay/replay_easy_canonical_v1.jsonl + "
            "tests/data/replay/replay_dense_canonical_v1.jsonl"
        ),
    )
    op.add_argument(
        "--sequencing-artifact",
        default=f"{COMMITTED_RL_POLICY_DIR_REL}/{SEQUENCING_POLICY_FILENAME}",
        help="path to the sequencing-policy artifact",
    )
    op.add_argument(
        "--fallback-dataset-id",
        default="v1_bootstrap",
        help=(
            "dataset id used to locate splits if canonical replay "
            "packs are missing (fallback)"
        ),
    )
    op.add_argument(
        "--output",
        default=None,
        help=(
            "output path for the OPE report JSON (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/{OPE_REPORT_FILENAME})"
        ),
    )
    op.set_defaults(handler=_cmd_ope)

    pp = sub.add_parser(
        "promote-policy",
        help=(
            "offline promotion gate (WIS + DM) for a perception-budget "
            "or recovery policy artifact"
        ),
    )
    pp.add_argument(
        "--policy-family",
        required=True,
        choices=POLICY_FAMILIES,
        help="policy family (v5_perception_budget | v6_recovery)",
    )
    pp.add_argument(
        "--policy-artifact",
        required=True,
        help="path to the candidate policy artifact JSON",
    )
    pp.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help=(
            "replay-pack path (repeatable). Default: canonical easy + "
            "dense packs"
        ),
    )
    pp.add_argument(
        "--dataset-id",
        default="v7_promotion_canonical",
        help="stable dataset id surfaced in the report metadata",
    )
    pp.add_argument(
        "--training-scope",
        default="dense",
        help="training scope filter (default: dense)",
    )
    pp.add_argument("--seed", type=int, default=1, help="bootstrap RNG seed")
    pp.add_argument(
        "--min-lift-over-baseline",
        type=float,
        default=0.01,
        help="minimum point lift over baseline for verdict=pass (positive floor; default 0.01)",
    )
    pp.add_argument(
        "--min-lower-bound-lift",
        type=float,
        default=0.0,
        help=(
            "minimum bootstrap-CI lower bound on the lift for "
            "verdict=pass"
        ),
    )
    pp.add_argument(
        "--min-n-effective",
        type=float,
        default=10.0,
        help="minimum WIS effective sample size required to grade",
    )
    pp.add_argument(
        "--min-records-with-weight",
        type=int,
        default=5,
        help=(
            "minimum number of records where target policy agrees "
            "with behavior (required to grade)"
        ),
    )
    pp.add_argument(
        "--output",
        default=None,
        help=(
            "output path for the promotion-report JSON (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/<family>_promotion_v1.json)"
        ),
    )
    pp.set_defaults(handler=_cmd_promote_policy)
