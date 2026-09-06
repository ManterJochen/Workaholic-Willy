"""Shared constants for the robot package.

The high-level facade and the motion controller need the same values. Keeping
them here is what stops the two from drifting apart as separate magic numbers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from logging import Logger

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Aggregate rotating log for every class in the robot package. Each line carries the name
#: of the logger that wrote it, so robot events stay co-located and in chronological order.
#: Every class writes to a per-module sub-file as well.
ROBOT_LOG_FILE: Final[str] = "robot.log"
ROBOT_LOG_DIR: Final[str] = "logs/robot"
#: Per-module sub-logs, the second sink alongside the aggregate ``robot.log``.
ROBOT_MODULES_LOG_DIR: Final[str] = "logs/robot/modules"

# Per-module log filenames. Each subsystem writes here and to the aggregate robot.log.
UR_CONNECTION_LOG_FILE: Final[str] = "ur_connection.log"
UR_ARM_LOG_FILE: Final[str] = "ur_arm.log"
UR_MOTION_LOG_FILE: Final[str] = "ur_motion.log"
UR_CUROBO_LOG_FILE: Final[str] = "ur_curobo.log"
KUKA_ARM_LOG_FILE: Final[str] = "kuka_arm.log"
KUKA_EKI_LOG_FILE: Final[str] = "kuka_eki.log"
SIM_ARM_LOG_FILE: Final[str] = "sim_arm.log"
GRIPPER_LOG_FILE: Final[str] = "gripper.log"
SAFETY_PREFLIGHT_LOG_FILE: Final[str] = "safety_preflight.log"
SAFETY_WORKSPACE_LOG_FILE: Final[str] = "safety_workspace.log"
POSE_PROVIDER_LOG_FILE: Final[str] = "pose_provider.log"
CALIBRATION_LOG_FILE: Final[str] = "calibration.log"
GRASP_CALCULATOR_LOG_FILE: Final[str] = "grasp_calculator.log"
MASK_ANALYZER_LOG_FILE: Final[str] = "mask_analyzer.log"
RUNTIME_PICK_LOG_FILE: Final[str] = "runtime_pick.log"

# driver selection + host readiness -------------------------------------------------------------
#: Which vendor factory was registered and instantiated. Answers which driver a cell is
#: running without reading the config tree back.
DRIVER_REGISTRY_LOG_FILE: Final[str] = "driver_registry.log"
#: The fail-early vendor-SDK gate (``drivers/doctor.py``).
DRIVER_DOCTOR_LOG_FILE: Final[str] = "driver_doctor.log"

# digital-I/O bench (the numbers that become gripper config) ------------------------------------
#: Every pin driven and every transition timed on a real cell. Separate from ``ur_arm.log``
#: because a bench session is a different activity from a pick and its timings are read back.
UR_IO_BENCH_LOG_FILE: Final[str] = "ur_io_bench.log"
#: The operator CLI in front of the bench: who asked for what, and every refusal.
UR_IO_CLI_LOG_FILE: Final[str] = "ur_io_cli.log"

# grippers --------------------------------------------------------------------------------------
GRIPPER_REGISTRY_LOG_FILE: Final[str] = "gripper_registry.log"
#: The two I/O end-effectors. Their filenames sit here with the rest so the set of log files
#: a cell produces is readable in one place.
JAW_IO_GRIPPER_LOG_FILE: Final[str] = "jaw_io_gripper.log"
VACUUM_GRIPPER_LOG_FILE: Final[str] = "vacuum_gripper.log"
ONROBOT_GRIPPER_LOG_FILE: Final[str] = "onrobot_gripper.log"

# execution / composition root ------------------------------------------------------------------
#: Who owned the cell and when. The ownership trail outlives the process that held the lock.
CELL_LOCK_LOG_FILE: Final[str] = "cell_lock.log"
IK_SERVICE_LOG_FILE: Final[str] = "ik_service.log"
#: What ``from_robot_config`` built and which advanced overlays ended up live. The default
#: pick path is open-loop and most ``grasping.*`` blocks are off, which makes what is wired
#: the most-asked question about a running cell.
GRASP_BUILDERS_LOG_FILE: Final[str] = "grasp_builders.log"
GRASP_RECORD_LOG_FILE: Final[str] = "grasp_record_logging.log"
GRASP_LATENCY_LOG_FILE: Final[str] = "grasp_latency.log"
RL_SHADOW_LOG_FILE: Final[str] = "rl_shadow.log"

# motion-stack externals ------------------------------------------------------------------------
#: The process-isolated cuRobo sidecar: spawn, warm-up cost, and every plan it refused.
CUROBO_CLIENT_LOG_FILE: Final[str] = "curobo_client.log"
#: Which exact-mesh collision engine resolved (Coal, python-fcl or none). An engine that is
#: absent resolves silently and leaves the guard above it inert.
PLANNING_ENVIRONMENT_LOG_FILE: Final[str] = "planning_environment.log"
#: The deep health check that loads every engine. Separate from ``planning_environment.log``
#: because the blocked verdict that ``looks_policy_blocked`` in ``safety/planning/doctor.py``
#: reports comes from a live reputation service and changes over time, so the history is the
#: value and would be lost among the per-build resolution lines.
PLANNING_DOCTOR_LOG_FILE: Final[str] = "planning_doctor.log"
# There is deliberately no constant for ``safety/planning/robot/build_gripper_spheres.py``.
# That generator is documented to run by path, which puts only the script's own directory on
# ``sys.path``, so importing ``src.robot.constants`` there would break the documented
# invocation. It already prints its output path and sphere count.


def create_robot_logger(name: str, sub_file: str, level: int = logging.INFO) -> Logger:
    """Robot logger with two file sinks.

    One is the per-module ``sub_file`` under :data:`ROBOT_MODULES_LOG_DIR`, the other the
    shared :data:`ROBOT_LOG_FILE` under :data:`ROBOT_LOG_DIR`, so every robot event is both
    separated per subsystem and co-located chronologically. Console output and path-keyed
    rotation come from ``create_logger``.
    """
    from src.utility.log_cfg import create_logger

    return create_logger(
        name,
        sub_file,
        level=level,
        log_dir=ROBOT_MODULES_LOG_DIR,
        aggregate_file=ROBOT_LOG_FILE,
        aggregate_dir=ROBOT_LOG_DIR,
    )

# ----------------------------------------------------------------------
# Kinematics defaults
# ----------------------------------------------------------------------
#: Default home joint configuration for a 6-DOF UR arm, in radians. This is the looking-down
#: pose of the URSim default scene and of most operator manuals: shoulder lift and wrist 2 at
#: -pi/2, every other joint at zero.
HOME_JOINTS_DEFAULT: Final[tuple[float, ...]] = (
    0.0,
    -1.5707963267948966,  # -pi/2
    0.0,
    -1.5707963267948966,  # -pi/2
    0.0,
    0.0,
)


def home_joints_default() -> list[float]:
    """Return a fresh, mutable copy of :data:`HOME_JOINTS_DEFAULT`."""
    return list(HOME_JOINTS_DEFAULT)
