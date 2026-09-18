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
    # The arm and the gripper as one noun, with no pick service.
    "Robot": ("src.robot.execution.robot", "Robot"),
    # The whole cell, and its planner start at a desk.
    "Cell": ("src.robot.execution.cell", "Cell"),
    "PlannerStart": ("src.robot.execution.planner_start", "PlannerStart"),
    "PlannerStartReport": ("src.robot.execution.planner_start", "PlannerStartReport"),
    # How a pick moves, which the service builds into its one guarded policy.
    "GraspMotion": ("src.robot.grasping.motion.grasp_motion", "GraspMotion"),
    # The hand-eye sweep as a noun, which the calibrate command calls.
    "HandEyeCalibration": ("src.robot.execution.hand_eye", "HandEyeCalibration"),
    "SweepOptions": ("src.robot.execution.hand_eye", "SweepOptions"),
    "CalibrationRunReport": ("src.robot.execution.hand_eye", "CalibrationRunReport"),
    # The refusals a build or a connect raises, each from the module that raises it.
    "CameraWorldRequired": ("src.robot.execution.camera_world_wiring", "CameraWorldRequired"),
    "CellBusy": ("src.robot.execution.cell_lock", "CellBusy"),
    "CellNotBuilt": ("src.robot.execution.cell", "CellNotBuilt"),
    "LockKeyRequired": ("src.robot.execution.robot", "LockKeyRequired"),
    "NoRealGripper": ("src.robot.execution.lifecycle", "NoRealGripper"),
    "WristBodyRequired": ("src.robot.execution.wrist_bodies", "WristBodyRequired"),
    # What a robot's motion verbs report.
    "MotionReport": ("src.robot.execution.motion", "MotionReport"),
    "MotionOutcome": ("src.robot.execution.motion", "MotionOutcome"),
    "MotionRoute": ("src.robot.execution.motion", "MotionRoute"),
    "MotionVerb": ("src.robot.execution.motion", "MotionVerb"),
    "RouteReading": ("src.robot.execution.motion", "RouteReading"),
    # What a robot's hand verbs report.
    "HandReport": ("src.robot.execution.handling", "HandReport"),
    "HandOutcome": ("src.robot.execution.handling", "HandOutcome"),
    "PayloadState": ("src.robot.execution.handling", "PayloadState"),
    "HandlingReport": ("src.robot.execution.handling", "HandlingReport"),
    "HandlingOutcome": ("src.robot.execution.handling", "HandlingOutcome"),
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
    "Robot",
    "Cell",
    "PlannerStart",
    "PlannerStartReport",
    "GraspMotion",
    "HandEyeCalibration",
    "SweepOptions",
    "CalibrationRunReport",
    "CameraWorldRequired",
    "CellBusy",
    "CellNotBuilt",
    "LockKeyRequired",
    "NoRealGripper",
    "WristBodyRequired",
    "MotionReport",
    "MotionOutcome",
    "MotionRoute",
    "MotionVerb",
    "RouteReading",
    "HandReport",
    "HandOutcome",
    "PayloadState",
    "HandlingReport",
    "HandlingOutcome",
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
