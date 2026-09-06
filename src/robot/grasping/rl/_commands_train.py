"""Train CLI command group (train-candidate / -ranking / -sequencing / -perception-budget / -recovery)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

from src.robot.grasping.constants import (
    RL_COMMANDS_TRAIN_LOG_FILE,
    create_grasping_logger,
)

from .train_sequencing import (
    DEFAULT_MIN_SUPPORT_THRESHOLD,
)
from .train_perception_budget import (
    DEFAULT_LINUCB_ALPHA,
    DEFAULT_LINUCB_LAMBDA,
    train_perception_budget_from_manifest,
    train_perception_budget_from_packs,
    write_perception_budget_artifact,
)
from .perception_budget_policy import (
    DEFAULT_MIN_SUPPORT_THRESHOLD as PERCEPTION_DEFAULT_MIN_SUPPORT_THRESHOLD,
    MODE_BUCKET_DENSE,
)
from .train_recovery import (
    TRAINING_SCOPES as RECOVERY_TRAINING_SCOPES,
    TRAINING_SCOPE_DENSE as RECOVERY_TRAINING_SCOPE_DENSE,
    train_recovery_from_packs,
    write_recovery_artifact,
)
from .recovery_policy import (
    DEFAULT_LINUCB_ALPHA as RECOVERY_DEFAULT_LINUCB_ALPHA,
    DEFAULT_LINUCB_LAMBDA as RECOVERY_DEFAULT_LINUCB_LAMBDA,
    DEFAULT_MIN_SUPPORT_THRESHOLD as RECOVERY_DEFAULT_MIN_SUPPORT_THRESHOLD,
    RECOVERY_ACTIONS,
)


if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .policy_training import TrainingFamily

from ._cli_common import (
    COMMITTED_MANIFEST_DIR_REL,
    COMMITTED_RL_POLICY_DIR_REL,
    PERCEPTION_POLICY_FILENAME,
    RECOVERY_POLICY_FILENAME,
    _resolve_repo_root,
    _resolve_under_repo,
)


#: Each trainer logs the fit; this seam is the only place that knows where the
#: artifact landed and under which sha, the pair a promotion report is later
#: matched against. The refusals below abort before any trainer runs at all.
logger = create_grasping_logger("RLTrainCLI", RL_COMMANDS_TRAIN_LOG_FILE)

#: The command name each family answers to, for the two operator-facing lines. Kept here rather than
#: on the family enum because it is CLI wording: a library caller never sees "train-ranking-policy".
_COMMANDS: dict[str, str] = {
    "candidate": "train-candidate-policy",
    "ranking": "train-ranking-policy",
    "sequencing": "train-sequencing-policy",
}


def _train_from_manifest_command(
    family: "TrainingFamily", args: argparse.Namespace, **trainer_options: object
) -> int:
    """The shared body of the three manifest-driven train commands.

    The order, the refusal, the output-path policy and the summary live in `policy_training`.

    The operator wording stays here, and so does the repo-relative manifest path in it. The
    report carries the absolute manifest path in `detail`, which is what a library caller needs.
    """
    from .policy_training import PolicyTraining, TrainingVerdict  # noqa: PLC0415

    repo_root = _resolve_repo_root(args.repo_root)
    run = PolicyTraining.from_dataset_id(
        family,
        args.dataset_id,
        repo_root=repo_root,
        output_path=None if args.output is None else (repo_root / args.output).resolve(),
        seed=int(args.seed),
        **trainer_options,
    )
    report = run.train()
    if report.verdict is TrainingVerdict.NO_MANIFEST:
        manifest_rel = f"{COMMITTED_MANIFEST_DIR_REL}/{args.dataset_id}.json"
        logger.error("%s refused: no manifest at %s", _COMMANDS[family.value], run.manifest_path)
        print(f"{_COMMANDS[family.value]}: manifest not found at {manifest_rel}", file=sys.stderr)
        return report.exit_code
    logger.info(
        "Wrote %s policy artifact %s to %s (sha256 %s)",
        family.value,
        report.summary["policy_id"],
        report.summary["artifact_path"],
        str(report.summary["artifact_sha256"])[:16],
    )
    print(report.as_cli_json())
    return report.exit_code


def _cmd_train_candidate_policy(args: argparse.Namespace) -> int:
    """Train + write the deterministic logistic candidate policy."""
    from .policy_training import TrainingFamily  # noqa: PLC0415

    return _train_from_manifest_command(
        TrainingFamily.CANDIDATE, args, prune_threshold=float(args.prune_threshold),
    )


def _cmd_train_ranking_policy(args: argparse.Namespace) -> int:
    """Train + write the deterministic pairwise-logistic ranking policy."""
    from .policy_training import TrainingFamily  # noqa: PLC0415

    return _train_from_manifest_command(TrainingFamily.RANKING, args)


def _cmd_train_sequencing_policy(args: argparse.Namespace) -> int:
    """Train + write the deterministic lookup-table sequencing policy."""
    from .policy_training import TrainingFamily  # noqa: PLC0415

    return _train_from_manifest_command(
        TrainingFamily.SEQUENCING, args, min_support_threshold=int(args.min_support),
    )


def _cmd_train_perception_budget_policy(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    # Support both dataset-manifest and replay-pack sources.
    if args.replay_pack:
        pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
        for p in pack_paths:
            if not p.exists():
                logger.error(
                    "train-perception-budget-policy refused: replay pack not found at %s", p
                )
                print(
                    f"train-perception-budget-policy: replay pack not "
                    f"found at {p}",
                    file=sys.stderr,
                )
                return 3
        policy, train_result, artifact = train_perception_budget_from_packs(
            pack_paths=pack_paths,
            seed=args.seed,
            training_scope=args.training_scope,
            alpha=args.alpha,
            ridge_lambda=args.ridge_lambda,
            min_support_threshold=args.min_support,
        )
    else:
        manifest_path = (
            repo_root
            / COMMITTED_MANIFEST_DIR_REL
            / f"{args.dataset_id}.json"
        )
        if not manifest_path.exists():
            logger.error(
                "train-perception-budget-policy refused: no manifest at %s",
                manifest_path,
            )
            print(
                f"train-perception-budget-policy: manifest not found at "
                f"{manifest_path}",
                file=sys.stderr,
            )
            return 3
        policy, train_result, artifact = train_perception_budget_from_manifest(
            manifest_path=manifest_path,
            seed=args.seed,
            training_scope=args.training_scope,
            alpha=args.alpha,
            ridge_lambda=args.ridge_lambda,
            min_support_threshold=args.min_support,
        )

    if args.output is None:
        output_path = (
            repo_root
            / COMMITTED_RL_POLICY_DIR_REL
            / PERCEPTION_POLICY_FILENAME
        )
    else:
        output_path = (repo_root / args.output).resolve()
    artifact_hash = write_perception_budget_artifact(artifact, output_path)
    logger.info(
        "Wrote perception-budget policy artifact %s to %s (sha256 %s)",
        policy.policy_id,
        output_path,
        artifact_hash[:16],
    )
    try:
        artifact_path_str = output_path.relative_to(repo_root).as_posix()
    except ValueError:
        artifact_path_str = output_path.as_posix()
    summary = {
        "artifact_path": artifact_path_str,
        "artifact_sha256": artifact_hash,
        "policy_id": policy.policy_id,
        "training_scope": policy.training_scope,
        "num_records": train_result.num_records,
        "num_records_kept": train_result.num_records_kept,
        "num_records_dropped_dense_scope": (
            train_result.num_records_dropped_dense_scope
        ),
        "num_action_stop": train_result.num_action_stop,
        "num_action_continue": train_result.num_action_continue,
        "num_cells_observed": train_result.num_cells_observed,
        "mean_reward_stop": train_result.mean_reward_stop,
        "mean_reward_continue": train_result.mean_reward_continue,
        "min_support_threshold": int(args.min_support),
        "alpha": float(args.alpha),
        "ridge_lambda": float(args.ridge_lambda),
        "offline_kpi_estimates": artifact["offline_kpi_estimates"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def _cmd_train_recovery_policy(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    if not args.replay_pack:
        logger.error(
            "train-recovery-policy refused: at least one --replay-pack is required"
        )
        print(
            "train-recovery-policy: at least one --replay-pack is required",
            file=sys.stderr,
        )
        return 3
    pack_paths = [_resolve_under_repo(p, repo_root) for p in args.replay_pack]
    for p in pack_paths:
        if not p.exists():
            logger.error(
                "train-recovery-policy refused: replay pack not found at %s", p
            )
            print(
                f"train-recovery-policy: replay pack not found at {p}",
                file=sys.stderr,
            )
            return 3
    policy, train_result, artifact = train_recovery_from_packs(
        pack_paths=pack_paths,
        seed=args.seed,
        training_scope=args.training_scope,
        alpha=args.alpha,
        ridge_lambda=args.ridge_lambda,
        min_support_threshold=args.min_support,
        dataset_id=args.dataset_id,
    )
    if args.output is None:
        output_path = (
            repo_root
            / COMMITTED_RL_POLICY_DIR_REL
            / RECOVERY_POLICY_FILENAME
        )
    else:
        output_path = (repo_root / args.output).resolve()
    artifact_hash = write_recovery_artifact(artifact, output_path)
    logger.info(
        "Wrote recovery policy artifact %s to %s (sha256 %s)",
        policy.policy_id,
        output_path,
        artifact_hash[:16],
    )
    try:
        artifact_path_str = output_path.relative_to(repo_root).as_posix()
    except ValueError:
        artifact_path_str = output_path.as_posix()
    summary = {
        "artifact_path": artifact_path_str,
        "artifact_sha256": artifact_hash,
        "policy_id": policy.policy_id,
        "training_scope": policy.training_scope,
        "num_records_total": train_result.num_records_total,
        "num_records_in_scope": train_result.num_records_in_scope,
        "num_records_dropped_no_actions": train_result.num_records_dropped_no_actions,
        "num_tuples_extracted": train_result.num_tuples_extracted,
        "num_tuples_dropped_unknown_token": train_result.num_tuples_dropped_unknown_token,
        "num_per_action": {
            a: train_result.num_per_action[a] for a in RECOVERY_ACTIONS
        },
        "mean_reward_per_action": {
            a: train_result.mean_reward_per_action[a]
            for a in RECOVERY_ACTIONS
        },
        "num_distinct_state_keys": train_result.num_distinct_state_keys,
        "min_support_threshold": int(args.min_support),
        "alpha": float(args.alpha),
        "ridge_lambda": float(args.ridge_lambda),
        "offline_kpi_estimates": artifact["offline_kpi_estimates"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


def register_train_commands(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    tc = sub.add_parser(
        "train-candidate-policy",
        help="train the deterministic logistic candidate policy",
    )
    tc.add_argument(
        "--dataset-id",
        default="v1_bootstrap",
        help="dataset id whose manifest lives under docs/baselines/rl_datasets/",
    )
    tc.add_argument("--seed", type=int, default=1)
    tc.add_argument(
        "--output",
        default=None,
        help=(
            "output artifact path (default: "
            "docs/baselines/rl_policies/v2_candidate_baseline_v1.json)"
        ),
    )
    tc.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="score threshold below which a candidate is flagged pruned",
    )
    tc.set_defaults(handler=_cmd_train_candidate_policy)

    tr = sub.add_parser(
        "train-ranking-policy",
        help="train the deterministic pairwise-logistic ranking policy",
    )
    tr.add_argument(
        "--dataset-id",
        default="v1_bootstrap",
        help="dataset id whose manifest lives under docs/baselines/rl_datasets/",
    )
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument(
        "--output",
        default=None,
        help=(
            "output artifact path (default: "
            "docs/baselines/rl_policies/v3_ranking_baseline_v1.json)"
        ),
    )
    tr.set_defaults(handler=_cmd_train_ranking_policy)

    ts = sub.add_parser(
        "train-sequencing-policy",
        help=(
            "train the deterministic lookup-table "
            "sequencing policy"
        ),
    )
    ts.add_argument(
        "--dataset-id",
        default="v1_bootstrap",
        help="dataset id whose manifest lives under docs/baselines/rl_datasets/",
    )
    ts.add_argument("--seed", type=int, default=1)
    ts.add_argument(
        "--output",
        default=None,
        help=(
            "output artifact path (default: "
            "docs/baselines/rl_policies/v4_sequencing_baseline_v1.json)"
        ),
    )
    ts.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT_THRESHOLD,
        help=(
            "minimum per-cell support to commit a learned action "
            "(below this the artifact falls back to the hand-"
            "authored default table)"
        ),
    )
    ts.set_defaults(handler=_cmd_train_sequencing_policy)

    tpb = sub.add_parser(
        "train-perception-budget-policy",
        help=(
            "train the LinUCB contextual-bandit perception-"
            "budget policy (shadow-only)"
        ),
    )
    tpb.add_argument(
        "--dataset-id",
        default="v1_bootstrap",
        help=(
            "dataset id whose manifest lives under "
            "docs/baselines/rl_datasets/ (used when --replay-pack is "
            "not supplied)"
        ),
    )
    tpb.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help=(
            "replay-pack path(s) to train on (repeatable); when "
            "supplied, overrides --dataset-id"
        ),
    )
    tpb.add_argument("--seed", type=int, default=1)
    tpb.add_argument(
        "--output",
        default=None,
        help=(
            "output artifact path (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/{PERCEPTION_POLICY_FILENAME})"
        ),
    )
    tpb.add_argument(
        "--training-scope",
        default=MODE_BUCKET_DENSE,
        choices=("easy", "auto", "dense", "unknown"),
        help="mode scope of the trained policy (default: dense)",
    )
    tpb.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_LINUCB_ALPHA,
        help="LinUCB exploration coefficient alpha",
    )
    tpb.add_argument(
        "--ridge-lambda",
        type=float,
        default=DEFAULT_LINUCB_LAMBDA,
        help="ridge regularisation lambda",
    )
    tpb.add_argument(
        "--min-support",
        type=int,
        default=PERCEPTION_DEFAULT_MIN_SUPPORT_THRESHOLD,
        help=(
            "minimum per-cell support to commit a learned action "
            "(below this the policy falls back to the hand-authored "
            "default table)"
        ),
    )
    tpb.set_defaults(handler=_cmd_train_perception_budget_policy)

    trp = sub.add_parser(
        "train-recovery-policy",
        help=(
            "train the LinUCB recovery optimization policy "
            "(shadow-only)"
        ),
    )
    trp.add_argument(
        "--dataset-id",
        default="v6_recovery_canonical",
        help="stable dataset id surfaced in the artifact metadata",
    )
    trp.add_argument(
        "--replay-pack",
        action="append",
        default=None,
        help="replay-pack path (repeatable, required)",
    )
    trp.add_argument("--seed", type=int, default=1)
    trp.add_argument(
        "--output",
        default=None,
        help=(
            "output artifact path (default: "
            f"{COMMITTED_RL_POLICY_DIR_REL}/{RECOVERY_POLICY_FILENAME})"
        ),
    )
    trp.add_argument(
        "--training-scope",
        default=RECOVERY_TRAINING_SCOPE_DENSE,
        choices=RECOVERY_TRAINING_SCOPES,
        help="training scope (default: dense)",
    )
    trp.add_argument(
        "--alpha",
        type=float,
        default=RECOVERY_DEFAULT_LINUCB_ALPHA,
        help="LinUCB exploration coefficient alpha",
    )
    trp.add_argument(
        "--ridge-lambda",
        type=float,
        default=RECOVERY_DEFAULT_LINUCB_LAMBDA,
        help="ridge regularisation lambda",
    )
    trp.add_argument(
        "--min-support",
        type=int,
        default=RECOVERY_DEFAULT_MIN_SUPPORT_THRESHOLD,
        help=(
            "minimum per-feature support across active onehots required "
            "to commit a learned action (below this the policy falls "
            "back to the hand-authored table)"
        ),
    )
    trp.set_defaults(handler=_cmd_train_recovery_policy)
