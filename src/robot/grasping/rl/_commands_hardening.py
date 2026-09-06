"""Hardening-harness CLI command group (paired-soak / rollback-drill)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



from ._cli_common import (
    _resolve_repo_root,
    _resolve_under_repo,
)


def _cmd_paired_soak(args: argparse.Namespace) -> int:
    """Paired RL-on/RL-off soak (offline).

    Lift is measured against a hardcoded synthetic outcome model (mechanics-only), so the
    emitted JSON carries the ``reward_model`` honesty stamp.
    """

    from src.robot.grasping.rl.paired_soak import (
        PARITY_FAIL,
        SoakScenarioSpec,
        run_paired_soak,
    )

    repo_root = _resolve_repo_root(args.repo_root)

    def _abs(p: str) -> Path:
        return _resolve_under_repo(p, repo_root)

    report = run_paired_soak(
        policy_artifact_path=_abs(args.policy_artifact),
        promotion_report_path=_abs(args.promotion_report),
        spec=SoakScenarioSpec(
            seed=int(args.seed), attempts_per_arm=int(args.attempts_per_arm)
        ),
        output_dir=_abs(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    # Honest exit code: nonzero only on a parity fail so CI can gate on it.
    return 1 if report.verdict == PARITY_FAIL else 0


def _cmd_rollback_drill(args: argparse.Namespace) -> int:
    """Canary + specialist + cross-router rollback drills (offline)."""

    from src.robot.grasping.rl.rollback_drill import (
        DRILL_PASS,
        run_rollback_drills,
    )

    repo_root = _resolve_repo_root(args.repo_root)

    def _abs(p: str) -> Path:
        return _resolve_under_repo(p, repo_root)

    report = run_rollback_drills(
        policy_artifact_path=_abs(args.policy_artifact),
        promotion_report_path=_abs(args.promotion_report),
        output_path=_abs(args.output) if args.output else None,
    )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    return 0 if report.overall_verdict == DRILL_PASS else 1


def register_hardening_commands(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    vs = sub.add_parser(
        "paired-soak",
        help=(
            "Paired RL-on/RL-off soak (offline). Lift is vs a SYNTHETIC reward model "
            "(mechanics-only, NOT real lift); artifacts carry the reward_model honesty stamp"
        ),
    )
    vs.add_argument(
        "--policy-artifact", required=True, help="path to the recovery policy artifact JSON"
    )
    vs.add_argument(
        "--promotion-report", required=True, help="path to the policy's promotion report JSON"
    )
    vs.add_argument("--seed", type=int, default=20251015, help="deterministic soak seed")
    vs.add_argument(
        "--attempts-per-arm", type=int, default=200, help="attempts per arm (floor 200)"
    )
    vs.add_argument(
        "--output-dir", default=None, help="optional dir to write the three JSON artifacts"
    )
    vs.set_defaults(handler=_cmd_paired_soak)

    vr = sub.add_parser(
        "rollback-drill",
        help="Canary + specialist + cross-router rollback drills (offline)",
    )
    vr.add_argument(
        "--policy-artifact", required=True, help="path to the recovery policy artifact JSON"
    )
    vr.add_argument(
        "--promotion-report", required=True, help="path to the policy's promotion report JSON"
    )
    vr.add_argument(
        "--output", default=None, help="optional path to write the rollback-drill report JSON"
    )
    vr.set_defaults(handler=_cmd_rollback_drill)
