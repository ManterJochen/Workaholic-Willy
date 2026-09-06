"""Shared runner harness: the seams every sim runner is built from.

One place for the pieces the runners share: the Isaac cell bootstrap, the pick and lift gate
scoring, the environment-knob parsing, the grasp-mode wiring, the opt-in record and debug
instrumentation, domain randomization, and the arm-backed IK service. Isaac imports stay lazy,
after the ``mock_mode`` check, so the harness is importable where ``isaacsim`` is absent.
"""

from __future__ import annotations

from .bootstrap import SimCell, announce_planning_environment, bootstrap_sim_cell
from .env import RunnerEnv
from .gate import GateResult, gate_passed, lift_mm_since, reset_object_to_home_z0
from .ik_service import ArmBackedIKService
from .instrumentation import write_pick_artifacts, write_run_result
from .modes import DEMO_MODES, DENSE_MODES, mode_service_kwargs, resolve_demo_mode
from .randomizer import DomainRandomizer, RandomizationConfig

__all__ = [
    "SimCell",
    "announce_planning_environment",
    "bootstrap_sim_cell",
    "RunnerEnv",
    "GateResult",
    "gate_passed",
    "lift_mm_since",
    "reset_object_to_home_z0",
    "ArmBackedIKService",
    "write_pick_artifacts",
    "write_run_result",
    "DEMO_MODES",
    "DENSE_MODES",
    "mode_service_kwargs",
    "resolve_demo_mode",
    "DomainRandomizer",
    "RandomizationConfig",
]
