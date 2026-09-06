"""The real-cell entry point: one config-driven pick loop on a physical arm.

This package is the live caller of ``from_robot_config``, the composition root of the stack, plus
the preflight that makes a bring-up survivable and the per-camera hand-eye calibration that
multi-view needs.

It does not know how to build a cell. That construction lives in ``execution/autonomous_grasp`` as
``build_real_cell`` and ``build_rehearsal_cell``, so the operator console reaches it without
importing a command-line runner. This package is a command-line surface over the service, not a
second way to compose it.

    python -m src.robot.execution.real_cell --help
"""

from .preflight import CheckStatus, PreflightCheck, PreflightReport, run_config_preflight

__all__ = ["CheckStatus", "PreflightCheck", "PreflightReport", "run_config_preflight"]
