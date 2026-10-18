"""Shared constants for the grasping tier.

One ``constants.py`` for the whole tier, the same shape
:mod:`src.robot.constants` already uses for the driver, gripper and safety
subpackages: log paths live in one file so a renamed log never has to be chased
through a dozen modules.

The grasping tier is the noisiest part of the robot package, between generation
and replay with its KPIs, so its per-module sub-logs get their own directory
rather than sharing ``logs/robot/modules`` with the drivers, and an operator
tracing a UR fault does not have to scroll past them. The aggregate
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

# --- generation (candidate synthesis; per-attempt aggregates, never per pose)
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
REACHABILITY_LOG_FILE: Final[str] = "reachability.log"

# --- loop / recovery (per-attempt runtime events) ---------------------------
RECOVERY_ORCHESTRATOR_LOG_FILE: Final[str] = "recovery_orchestrator.log"

# --- replay/ (offline KPI, soak, taxonomy) ----------------------------------
REPLAY_CLI_LOG_FILE: Final[str] = "replay_cli.log"
REPLAY_FAILURE_TAXONOMY_LOG_FILE: Final[str] = "replay_failure_taxonomy.log"
REPLAY_SOAK_CLI_LOG_FILE: Final[str] = "replay_soak_cli.log"

# --- deep/ranker (offline fitting) -----------------------------------------
#: Fitting the learned grasp ranker offline: what it was fitted on and what it scored.
#: Its own file, because a model shipped without its measurement is a number nobody can
#: defend later.
DEEP_RANKER_LOG_FILE: Final[str] = "deep_ranker.log"


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
