"""`safety.trajectory_check.enabled`: the waypoints between the two that are gated.

A planner hands over a path, and the one-shot guards judge only where a move ends. With the check
off, `gate_trajectory` logs one warning and returns None, so a plan that dips through the bench and
lands clear passes. Turning it on refuses nothing declared under `self_collision.fixtures`.
"""

import sys
from pathlib import Path

import numpy as np

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 04_safety, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.config.schema.robot import FixtureBoxConfig, TrajectoryCheckConfig  # noqa: E402
from src.robot.constants import home_joints_default  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.safety import SafetyPreflight  # noqa: E402

# 1. No profile in this tree declares a fixture, so the bench is declared here. Measure your own.
try:
    config = load_robot_config()
except ConfigError as no_cell:
    # `robot` is optional on AppConfig, so a tree without one is a configuration answer rather
    # than a crash, and the four files under scripts/checks/ answer it the same way.
    print(f"no cell in this config tree ({no_cell}). Write a `robot` block, or select a "
          f"profile that carries one: WILLY_PROFILE=ur5e")
    raise SystemExit
declared = config.safety.self_collision.model_copy(update={"fixtures": [FixtureBoxConfig(
    name="bench", center_mm=(500.0, 0.0, -75.0), half_extents_mm=(300.0, 300.0, 50.0))]})

# 2. The arm the guard places its link meshes on, and a path out and back: both endpoints are the
#    same clear home configuration, and everything that reaches into the bench lies between them.
arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
home = np.asarray(config.home_joint_positions or home_joints_default(), dtype=np.float64)
dip = np.asarray([3.14, -0.8, 2.0, -2.15, -1.57, 0.0])
waypoints = [tuple(home + (dip - home) * min(i, 40 - i) / 20) for i in range(41)]

# 3. The pipeline as the tree ships it; None means nothing refused. One WARNING per preflight.
safety = config.safety.model_copy(update={"self_collision": declared})
shipped = SafetyPreflight.from_safety_config(safety, config.workspace_limits)
print("endpoint:", shipped.gate_joint_target(JointPositions(list(waypoints[-1])), arm=arm))
print("path, check off:", shipped.gate_trajectory(waypoints, arm=arm))

# 4. The same path with the check on: refused at a waypoint in the middle, ~10 ms per configuration.
checked = SafetyPreflight.from_safety_config(
    safety.model_copy(update={"trajectory_check": TrajectoryCheckConfig(enabled=True)}),
    config.workspace_limits)
refusal = checked.gate_trajectory(waypoints, arm=arm)
print("path, check on:",
      "clear" if refusal is None else f"{refusal.status.value}: {refusal.message}")

# 5. The switch is not the declaration: with the check on and the fixtures this tree really ships,
#    all 41 configurations are judged against an empty room and the path comes back clear.
blind = SafetyPreflight.from_safety_config(
    config.safety.model_copy(update={"trajectory_check": TrajectoryCheckConfig(enabled=True)}),
    config.workspace_limits)
print("path, nothing declared:", blind.gate_trajectory(waypoints, arm=arm))
