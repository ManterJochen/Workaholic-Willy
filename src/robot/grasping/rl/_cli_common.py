"""Shared CLI constants and path helpers for the rl/ command-group modules.

A leaf: it imports nothing from rl/.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COMMITTED_MANIFEST_DIR_REL = "docs/baselines/rl_datasets"
COMMITTED_RL_POLICY_DIR_REL = "docs/baselines/rl_policies"
CANDIDATE_POLICY_FILENAME = "v2_candidate_baseline_v1.json"
RANKING_POLICY_FILENAME = "v3_ranking_baseline_v1.json"
SEQUENCING_POLICY_FILENAME = "v4_sequencing_baseline_v1.json"
OPE_REPORT_FILENAME = "v5_ope_report_v1.json"
PERCEPTION_POLICY_FILENAME = "v5_perception_budget_baseline_v1.json"
RECOVERY_POLICY_FILENAME = "v6_recovery_baseline_v1.json"
RECOVERY_PROMOTION_REPORT_FILENAME = "v6_recovery_baseline_v1_promotion_v1.json"
PERCEPTION_PROMOTION_REPORT_FILENAME = "v5_perception_budget_baseline_v1_promotion_v1.json"
DEFAULT_OPE_REPLAY_PACKS = (
    "tests/data/replay/replay_easy_canonical_v1.jsonl",
    "tests/data/replay/replay_dense_canonical_v1.jsonl",
)
OPE_EXIT_MALFORMED_INPUT = 3


def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return REPO_ROOT


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_under_repo(arg: str | Path, repo_root: Path) -> Path:
    """Resolve a possibly-relative CLI path argument under ``repo_root`` (absolute paths pass through)."""

    p = Path(arg)
    return p if p.is_absolute() else repo_root / p
