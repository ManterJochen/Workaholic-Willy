"""Dataset CLI command group (check-dataset / build-dataset / audit-leakage / replay-env-check)."""

from __future__ import annotations

import argparse
import json
import sys

from .dataset import (
    CANONICAL_BOOTSTRAP_SOURCES,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    build_dataset,
    load_jsonl,
    write_manifest,
)
from .honesty import INPUT_CANONICAL_BOOTSTRAP
from .replay_env import build_default_envs


from src.robot.grasping.constants import (
    RL_COMMANDS_DATASET_LOG_FILE,
    create_grasping_logger,
)

from ._cli_common import (
    COMMITTED_MANIFEST_DIR_REL,
    _resolve_repo_root,
)

#: The dataset build itself logs its counts; what is only visible at this seam is
#: what the operator asked for, the refusals that never reach the builder, and the
#: readiness verdict, which is reported and deliberately not gating.
logger = create_grasping_logger("RLDatasetCLI", RL_COMMANDS_DATASET_LOG_FILE)


def _cmd_check_dataset(args: argparse.Namespace) -> int:
    """Judge a record log before anyone trains on it. Exit 0 = trainable, 1 = not, 2 = unreadable.

    Runs on any GraspAttemptRecord log, wherever it came from: a customer's own cell, a sim
    runner, a replay pack. It opens no cell, loads no policy and trains nothing.
    """
    from .runs import ReadinessVerdict, RecordLogCheck

    report = RecordLogCheck.from_records_log(args.records).assess()

    # The wording of these two messages is fixed: it is what an operator sees after mistyping a
    # path. `runs.py` carries the reason in `detail` rather than the sentence, so the sentence is
    # composed here, where the wording belongs to this CLI.
    if report.verdict is ReadinessVerdict.UNREADABLE:
        if report.detail == "no such record log":
            logger.error("check-dataset refused: no such record log: %s", report.source)
            print(f"no such record log: {report.source}", file=sys.stderr)
        else:
            logger.error("check-dataset could not read %s: %s", report.source, report.detail)
            print(f"could not read {report.source}: {report.detail}", file=sys.stderr)
        return report.exit_code

    readiness = report.readiness
    assert readiness is not None  # every verdict except ReadinessVerdict.UNREADABLE carries one
    log = logger.info if report.trainable else logger.warning
    log(
        "check-dataset over %d record(s) from %s: trainable=%s, %d usable pair(s)",
        readiness.records,
        report.source,
        report.trainable,
        readiness.pairs,
    )
    print(report.summary() if args.quiet else report.render())
    return report.exit_code


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    extras: tuple[str, ...] = tuple(args.extra_source or ())
    canonical: tuple[str, ...] = (
        tuple(args.canonical_source) if args.canonical_source else CANONICAL_BOOTSTRAP_SOURCES
    )
    ratios = tuple(args.ratios) if args.ratios else SPLIT_RATIOS  # type: ignore[assignment]
    if len(ratios) != 3:
        print("--ratios requires exactly three floats", file=sys.stderr)
        return 2

    manifest = build_dataset(
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        seed=args.seed,
        ratios=ratios,
        canonical_paths=canonical,
        extra_paths=extras,
        emit_parquet=not args.no_parquet,
        dataset_origin=args.dataset_origin,
        fitness_note=args.fitness_note,
        strict_leakage=args.strict_leakage,
        group_split_field=args.group_split_field,
    )

    # build_dataset embeds the leakage audit in the manifest; the summary and the exit code
    # read it back from there.
    leakage_payload = manifest.leakage or {}
    leakage_passed = bool(leakage_payload.get("passed", False))

    manifest_dir = repo_root / COMMITTED_MANIFEST_DIR_REL
    manifest_path = manifest_dir / f"{args.dataset_id}.json"
    # write_manifest emits byte-identical deterministic sort-keyed JSON (one serialiser, not a copy).
    write_manifest(manifest, manifest_path)

    # Readiness, on the way past. `build_dataset` indexes and splits; it does not look at whether
    # there is anything to learn. A corpus with no success/fail pair, or with every feature reading
    # 0.0, builds a perfectly valid manifest and then trains a converged model over nothing. One
    # read of the same records, reported here, catches that before a training run.
    readiness = None
    try:
        from .readiness import assess_records, format_readiness

        rows: list = []
        for source in (*canonical, *extras):
            path = repo_root / source
            if path.is_file():
                rows.extend(load_jsonl(path))
        if rows:
            readiness = assess_records(rows)
            print(format_readiness(readiness, verbose=False), file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 (a diagnosis must never fail the build it is diagnosing)
        logger.warning("Readiness check skipped for this build: %s", exc, exc_info=True)
        print(f"  (readiness check skipped: {type(exc).__name__}: {exc})", file=sys.stderr)

    summary = {
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "record_count": manifest.record_count,
        "split_counts": manifest.split_counts,
        "parquet_emitted": manifest.parquet_emitted,
        "parquet_skip_reason": manifest.parquet_skip_reason,
        "leakage_passed": leakage_passed,
        "leakage_findings": len(leakage_payload.get("findings", [])),
        # Reported, never gating: a dataset that is not trainable is still a dataset, and the build
        # is not the place to decide what someone may do with it.
        "trainable": (readiness.trainable if readiness is not None else None),
        "pairs": (readiness.pairs if readiness is not None else None),
    }
    if readiness is not None and not readiness.trainable:
        # Reported, never gating, so it is logged loudly here.
        logger.warning(
            "Dataset %s built but is NOT trainable (%d usable pair(s)); the build "
            "does not gate on this",
            args.dataset_id,
            readiness.pairs,
        )
    print(json.dumps(summary, sort_keys=True, indent=2))
    if not leakage_passed:
        logger.error(
            "build-dataset exits 3: the leakage audit did not pass for %s",
            manifest_path,
        )
        return 3
    return 0


def _cmd_audit_leakage(args: argparse.Namespace) -> int:
    # The audit verdict itself is logged by ``leakage.run_leakage_audits``.
    repo_root = _resolve_repo_root(args.repo_root)
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    # A shim over `LeakageAudit`: the audit itself is `run_leakage_audits`, and the noun adds the
    # two things around it. It resolves a manifest's relative split paths against a repo root, and
    # it treats missing split files as the ordinary state of a fresh clone (they are gitignored),
    # reporting a typed verdict and a documented exit code instead of an uncaught
    # FileNotFoundError.
    from .datasets import LeakageAudit

    audit = LeakageAudit.from_manifest(
        args.manifest, repo_root=repo_root, strict=args.strict_leakage
    ).audit()
    if audit.findings is None:
        print(audit.render(), file=sys.stderr)
        return audit.exit_code
    print(json.dumps(audit.findings.to_json(), sort_keys=True, indent=2))
    return audit.exit_code


def _cmd_replay_env_check(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec_env, geo_env = build_default_envs()
    out: dict[str, dict[str, str | int]] = {}
    for name in SPLIT_NAMES:
        rel = payload["split_files"][name]["jsonl"]
        records = load_jsonl(repo_root / rel)
        out[name] = {
            "records": len(records),
            "recorded_observation_fingerprint": rec_env.fingerprint(records),
            "geometric_rerun_fingerprint": geo_env.fingerprint(records),
        }
    print(json.dumps(out, sort_keys=True, indent=2))
    return 0


def register_dataset_commands(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    cd = sub.add_parser(
        "check-dataset",
        help="is a record log trainable? (occupancy, variance, success/fail pairs); trains nothing",
    )
    cd.add_argument("--records", required=True, help="path to a GraspAttemptRecord JSONL log")
    cd.add_argument("--quiet", action="store_true", help="verdict and findings only, no per-column table")
    cd.set_defaults(handler=_cmd_check_dataset)

    bd = sub.add_parser("build-dataset", help="build an offline RL dataset")
    bd.add_argument("--dataset-id", required=True, help="stable dataset identifier")
    bd.add_argument("--seed", type=int, default=1, help="deterministic split seed")
    bd.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test ratios (must sum to 1.0; default 0.6 0.2 0.2)",
    )
    bd.add_argument(
        "--canonical-source",
        action="append",
        default=None,
        help="override canonical source paths (repeatable)",
    )
    bd.add_argument(
        "--extra-source",
        action="append",
        default=None,
        help="extra JSONL source paths (repeatable)",
    )
    bd.add_argument(
        "--no-parquet",
        action="store_true",
        help="skip parquet emission even if pyarrow is available",
    )
    bd.add_argument(
        "--strict-leakage",
        action="store_true",
        help="treat leakage warnings as failures",
    )
    bd.add_argument(
        "--dataset-origin",
        default=INPUT_CANONICAL_BOOTSTRAP,
        help="provenance of the records (honesty stamp): e.g. canonical_bootstrap_replay_packs | sim | real_hardware",
    )
    bd.add_argument(
        "--fitness-note",
        default=None,
        help="honesty stamp: one line on what the dataset IS / ISN'T fit for",
    )
    bd.add_argument(
        "--group-split-field",
        default=None,
        help=(
            "split leakage-safely BY this record/extra field (e.g. scene_family_id): every record "
            "sharing the value stays in ONE split (no object identity leaks; ranking pairs stay within a split). "
            "Default None -> the per-record stratified split (byte-identical; committed datasets unaffected)."
        ),
    )
    bd.set_defaults(handler=_cmd_build_dataset)

    al = sub.add_parser("audit-leakage", help="re-run leakage audits on an existing manifest")
    al.add_argument("--manifest", required=True, help="manifest path (repo-relative)")
    al.add_argument("--strict-leakage", action="store_true")
    al.set_defaults(handler=_cmd_audit_leakage)

    rc = sub.add_parser("replay-env-check", help="emit replay-env determinism fingerprints")
    rc.add_argument("--manifest", required=True, help="manifest path (repo-relative)")
    rc.set_defaults(handler=_cmd_replay_env_check)
