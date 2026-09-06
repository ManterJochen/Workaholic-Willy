"""
Execution-layer robot orchestration.

This package is the public architecture slot for motion execution,
pose provisioning, and calibration orchestration. Imports are resolved
lazily so importing :mod:`src.robot.execution` does not eagerly
pull the UR driver chain.
"""

from __future__ import annotations

from importlib import import_module

_LAZY: dict[str, tuple[str, str]] = {
    # Lazy like everything else here. `pick_run` pulls nothing heavier than
    # `src.contracts`, but importing it eagerly would put a name on this package
    # that a schema-only import does not need.
    "PassRule": ("src.robot.execution.pick_run", "PassRule"),
    "PickAttempt": ("src.robot.execution.pick_run", "PickAttempt"),
    "PickOutcome": ("src.robot.execution.pick_run", "PickOutcome"),
    "PickRun": ("src.robot.execution.pick_run", "PickRun"),
    "PickRunReport": ("src.robot.execution.pick_run", "PickRunReport"),
    "Recording": ("src.robot.execution.pick_run", "Recording"),
    "PoseProvider": ("src.robot.execution.pose_provider", "PoseProvider"),
    "CalibrationRoutine": (
        "src.robot.execution.calibration",
        "CalibrationRoutine",
    ),
    "CalibrationResult": (
        "src.robot.execution.calibration",
        "CalibrationResult",
    ),
    "MarkerPoseProvider": (
        "src.robot.execution.calibration",
        "MarkerPoseProvider",
    ),
    "RobotArmIKService": (
        "src.robot.execution.ik_service",
        "RobotArmIKService",
    ),
    "CachedIKService": (
        "src.robot.execution.ik_service",
        "CachedIKService",
    ),
    "URAnalyticIKService": (
        "src.robot.execution.ik_service",
        "URAnalyticIKService",
    ),
    "PickSessionReport": (
        "src.robot.execution.runtime_pick",
        "PickSessionReport",
    ),
    "PickTimings": (
        "src.robot.execution.runtime_pick",
        "PickTimings",
    ),
    "RuntimePickService": (
        "src.robot.execution.runtime_pick",
        "RuntimePickService",
    ),
    "AutonomousGraspService": (
        "src.robot.execution.autonomous_grasp",
        "AutonomousGraspService",
    ),
    "AutonomousGraspReport": (
        "src.robot.execution.autonomous_grasp",
        "AutonomousGraspReport",
    ),
    "AutonomousGraspOutcome": (
        "src.robot.execution.autonomous_grasp",
        "AutonomousGraspOutcome",
    ),
    "GraspMode": (
        "src.robot.execution.autonomous_grasp",
        "GraspMode",
    ),
    "GraspBehaviorProfile": (
        "src.robot.execution.autonomous_grasp",
        "GraspBehaviorProfile",
    ),
    "resolve_grasp_mode": (
        "src.robot.execution.autonomous_grasp",
        "resolve_grasp_mode",
    ),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'src.robot.execution' has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CachedIKService",
    "PassRule",
    "PickAttempt",
    "PickOutcome",
    "PickRun",
    "PickRunReport",
    "Recording",
    "CalibrationResult",
    "CalibrationRoutine",
    "MarkerPoseProvider",
    "PickSessionReport",
    "PickTimings",
    "PoseProvider",
    "RobotArmIKService",
    "RuntimePickService",
    "URAnalyticIKService",
    "AutonomousGraspService",
    "AutonomousGraspReport",
    "AutonomousGraspOutcome",
    "GraspMode",
    "GraspBehaviorProfile",
    "resolve_grasp_mode",
]
