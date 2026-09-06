"""Autonomous grasp service package.

One concern per module: service, config, report, builders, cells,
rehearsal, record_logging, action_mask_eval, watchdog, latency, shadow.
The public API is re-exported here so
``src.robot.execution.autonomous_grasp`` import paths are unchanged.
"""

from .config import (
    EffectiveCorridorConfig,
    EffectiveDecisionConfig,
    EffectiveFeasibilityConfig,
    EffectiveGraspingConfig,
    EffectiveOrderingConfig,
    EffectivePerformanceConfig,
    EffectiveRecoveryOrchestratorConfig,
    EffectiveUncertaintyConfig,
    EffectiveWatchdogConfig,
    GraspBehaviorProfile,
    GraspMode,
    GraspModeInput,
    _profile_for,
    resolve_grasp_mode,
)
from .report import AutonomousGraspOutcome, AutonomousGraspReport
from .cells import (
    build_real_cell,
    build_real_components,
    build_rehearsal_cell,
    build_rehearsal_components,
)
from .service import (
    AutonomousGraspService,
    DecisionAction,
    DecisionEngine,
    DecisionPolicy,
    DecisionReport,
    _RecoveryReportAdapter,
)

__all__ = [
    "AutonomousGraspOutcome",
    "build_real_cell",
    "build_rehearsal_cell",
    "build_real_components",
    "build_rehearsal_components",
    "AutonomousGraspReport",
    "AutonomousGraspService",
    "DecisionAction",
    "DecisionEngine",
    "DecisionPolicy",
    "DecisionReport",
    "EffectiveCorridorConfig",
    "EffectiveDecisionConfig",
    "EffectiveFeasibilityConfig",
    "EffectiveGraspingConfig",
    "EffectiveOrderingConfig",
    "EffectivePerformanceConfig",
    "EffectiveRecoveryOrchestratorConfig",
    "EffectiveUncertaintyConfig",
    "EffectiveWatchdogConfig",
    "GraspBehaviorProfile",
    "GraspMode",
    "GraspModeInput",
    "resolve_grasp_mode",
]
