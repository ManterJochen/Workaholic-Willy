"""Shared constants for the grasping tier.

One ``constants.py`` for the whole tier, the same shape
:mod:`src.robot.constants` already uses for the driver, gripper and safety
subpackages: log paths live in one file so a renamed log never has to be chased
through a dozen modules.

The grasping tier is the noisiest part of the robot package, between generation,
replay with its KPIs, and RL, so its per-module sub-logs get their own directory
rather than sharing ``logs/robot/modules`` with the drivers, and an operator
tracing a UR fault does not have to scroll past forty RL files. The aggregate
stays the package-wide :data:`~src.robot.constants.ROBOT_LOG_FILE`, so every
grasping event is still co-located chronologically with the arm, gripper and
safety events that surround it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from src.robot.constants import ROBOT_LOG_DIR, ROBOT_LOG_FILE

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Directory holding the per-module sub-logs of the grasping tier. The second
#: sink is the package aggregate ``robot.log``; see the module docstring.
GRASPING_MODULES_LOG_DIR: Final[str] = "logs/robot/grasping"

# --- calibration/ (offline model calibration, promotion, signing) -----------
MODEL_PROMOTION_LOG_FILE: Final[str] = "model_promotion.log"
CALIBRATION_SIGNING_LOG_FILE: Final[str] = "calibration_signing.log"
SUCCESS_MODEL_CALIBRATION_LOG_FILE: Final[str] = "success_model_calibration.log"
UNCERTAINTY_CALIBRATION_LOG_FILE: Final[str] = "uncertainty_calibration.log"

# --- generation (candidate synthesis; per-attempt aggregates, never per pose)
#: Why the generator produced the candidate count it did: which sampler ran, how
#: many antipodal pairs survived, and every empty result with its reason.
CANDIDATE_GENERATOR_LOG_FILE: Final[str] = "candidate_generator.log"
#: The near-isotropic closing-axis snap. Its own file because it changes the
#: executed close direction of a candidate the operator already saw ranked, which
#: belongs next to the pick rather than buried in the counts of the generator.
ISOTROPIC_CLOSING_LOG_FILE: Final[str] = "isotropic_closing.log"
#: The debug-overlay renderer: the PNG it wrote, and the candidates it could not
#: draw, since a base-frame pose is silently unprojectable and leaves an overlay
#: that looks suspiciously empty.
DEBUG_DRAW_LOG_FILE: Final[str] = "debug_draw.log"
#: Everything the learned generator writes: the construction decisions and refusals of
#: the factory, the model loads of the calculator, and the per-epoch line of the
#: training loop.
#:
#: It lives here so the stable tree does not import the moving one. `calculator_factory`
#: is on the boot path of every cell, and reading this name out of `grasping/deep` would
#: point the dependency the wrong way: the analytic side of the package would break
#: whenever the learned side moved a file, and the deep tree moves often.
#:
#: This is the only definition of the name. `deep/calculator.py` reads it from here, and
#: the deep package holds no copy and no re-export, because a re-export would leave two
#: addresses for one symbol.
DEEP_GENERATOR_LOG_FILE: Final[str] = "deep_generator.log"

# --- perception -> pose pipeline (per-attempt runtime events) ---------------
SCENE_GEOMETRY_LOG_FILE: Final[str] = "multiview_scene_geometry.log"
TARGET_TRACKING_LOG_FILE: Final[str] = "target_tracking.log"
REACHABILITY_LOG_FILE: Final[str] = "reachability.log"

# --- decision / loop / recovery (per-attempt runtime events) ----------------
DECISION_LOG_FILE: Final[str] = "decision.log"
SHADOW_AGGREGATOR_LOG_FILE: Final[str] = "shadow_aggregator.log"
RECOVERY_ORCHESTRATOR_LOG_FILE: Final[str] = "recovery_orchestrator.log"

# --- replay/ (offline KPI, soak, taxonomy, adaptation) ----------------------
REPLAY_CLI_LOG_FILE: Final[str] = "replay_cli.log"
REPLAY_ADAPTATION_LOG_FILE: Final[str] = "replay_adaptation.log"
REPLAY_ADAPTATION_CLI_LOG_FILE: Final[str] = "replay_adaptation_cli.log"
REPLAY_ADAPTATION_IO_LOG_FILE: Final[str] = "replay_adaptation_io.log"
REPLAY_BASELINE_REPORT_LOG_FILE: Final[str] = "replay_baseline_report.log"
REPLAY_CANONICAL_DATASETS_LOG_FILE: Final[str] = "replay_canonical_datasets.log"
REPLAY_FAILURE_TAXONOMY_LOG_FILE: Final[str] = "replay_failure_taxonomy.log"
REPLAY_SOAK_LOG_FILE: Final[str] = "replay_soak.log"
REPLAY_SOAK_CLI_LOG_FILE: Final[str] = "replay_soak_cli.log"

# --- rl/ offline CLI command handlers ---------------------------------------
#: The invocation and every refusal that aborts before the pipeline module runs.
#: The hardening group has no file of its own: it is a thin argparse shim whose two
#: callees, paired_soak and rollback_drill, already log their inputs and verdicts.
RL_COMMANDS_DATASET_LOG_FILE: Final[str] = "rl_commands_dataset.log"
#: Fitting the learned grasp ranker offline: what it was fitted on and what it scored.
#: Its own file, because a model shipped without its measurement is a number nobody can
#: defend later.
DEEP_RANKER_LOG_FILE: Final[str] = "deep_ranker.log"
RL_COMMANDS_EVAL_LOG_FILE: Final[str] = "rl_commands_eval.log"
RL_COMMANDS_TRAIN_LOG_FILE: Final[str] = "rl_commands_train.log"

# --- rl/ offline pipeline (dataset -> train -> OPE -> promote -> drill) -----
RL_DATASET_LOG_FILE: Final[str] = "rl_dataset.log"
RL_LEAKAGE_LOG_FILE: Final[str] = "rl_leakage.log"
RL_OPE_LOG_FILE: Final[str] = "rl_ope.log"
RL_PAIRED_SOAK_LOG_FILE: Final[str] = "rl_paired_soak.log"
RL_PROMOTION_LOG_FILE: Final[str] = "rl_promotion.log"
RL_ROLLBACK_DRILL_LOG_FILE: Final[str] = "rl_rollback_drill.log"
RL_TRAIN_CANDIDATE_LOG_FILE: Final[str] = "rl_train_candidate.log"
RL_TRAIN_PERCEPTION_BUDGET_LOG_FILE: Final[str] = "rl_train_perception_budget.log"
RL_TRAIN_RANKING_LOG_FILE: Final[str] = "rl_train_ranking.log"
RL_TRAIN_RECOVERY_LOG_FILE: Final[str] = "rl_train_recovery.log"
RL_TRAIN_SEQUENCING_LOG_FILE: Final[str] = "rl_train_sequencing.log"

# --- rl/ runtime seam (routers + policy artifact loading) -------------------
#: online_state has no file: it is a pure-stdlib leaf whose only non-raising outcome
#: is immediately raised by apply_online_state, and the selected tier is already
#: logged by the online router that chose it.
RL_ROUTER_LOG_FILE: Final[str] = "rl_router.log"
RL_CANARY_ROUTER_LOG_FILE: Final[str] = "rl_canary_router.log"
RL_ONLINE_ROUTER_LOG_FILE: Final[str] = "rl_online_router.log"
RL_SPECIALIST_ROUTER_LOG_FILE: Final[str] = "rl_specialist_router.log"
RL_CANDIDATE_POLICY_LOG_FILE: Final[str] = "rl_candidate_policy.log"
RL_RANKING_POLICY_LOG_FILE: Final[str] = "rl_ranking_policy.log"
RL_RECOVERY_POLICY_LOG_FILE: Final[str] = "rl_recovery_policy.log"
RL_SEQUENCING_POLICY_LOG_FILE: Final[str] = "rl_sequencing_policy.log"
RL_PERCEPTION_BUDGET_POLICY_LOG_FILE: Final[str] = "rl_perception_budget_policy.log"

# --- decision-adjacent gates (per-attempt, fail-closed) ---------------------
#: The fused-uncertainty gate. A pick refused here is refused for a reason the
#: channel values explain, and that reason is not reconstructable afterwards.
UNCERTAINTY_LOG_FILE: Final[str] = "uncertainty_fusion.log"

# --- scoring / training data ------------------------------------------------
#: Loading the success-probability artifact (family, version, provenance).
SUCCESS_MODEL_LOG_FILE: Final[str] = "success_model.log"
#: The probability-blend reranker. Separate from the file of the model itself,
#: because these lines answer a question the model log cannot: the blend is enabled,
#: so why did the order not change?
RANKING_BLEND_LOG_FILE: Final[str] = "ranking_blend.log"


def create_grasping_logger(
    name: str, sub_file: str, level: int = logging.INFO
) -> Logger:
    """Grasping logger with two file sinks: a per-module ``sub_file`` under
    :data:`GRASPING_MODULES_LOG_DIR` and the package aggregate
    :data:`~src.robot.constants.ROBOT_LOG_FILE`. It is the same contract as
    :func:`~src.robot.constants.create_robot_logger`, with the grasping tier's own
    sub-log directory in place of the package one. Console output and the
    path-keyed handler sharing that makes rotation safe come from ``create_logger``.
    """
    from src.utility.log_cfg import create_logger

    return create_logger(
        name,
        sub_file,
        level=level,
        log_dir=GRASPING_MODULES_LOG_DIR,
        aggregate_file=ROBOT_LOG_FILE,
        aggregate_dir=ROBOT_LOG_DIR,
    )
