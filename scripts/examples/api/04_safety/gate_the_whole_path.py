"""The waypoints between the two that are gated, and what a cell has to declare for it to matter.

A planner hands over a path, and the one-shot guards judge only where a move ends. `gate_planned_path`
judges the whole of it: it samples every leg so that no step moves any point of the arm further than
the collision margin, then runs the same guards a commanded joint move gets over every sample. There
is no switch. What there still is, and what this script is really about, is the difference between
running the check and having declared anything for it to find: with no fixtures declared the guards
judge all of it against an empty room and the path comes back clear.
"""

import sys
from pathlib import Path

import numpy as np

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 04_safety, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.config.schema.robot import FixtureBoxConfig  # noqa: E402
from src.robot.constants import home_joints_default  # noqa: E402
from src.robot.core import JointPositions, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.safety import SafetyPreflight  # noqa: E402
from src.robot.safety.planning.hand import planner_hand  # noqa: E402

# 1. No profile in this tree declares a fixture, and the base tree names no hand, so both are
#    declared here: the bench, and the 2F-85 the arm bundles carry, which a UR arm needs to build.
try:
    config = load_robot_config()
except ConfigError as no_cell:
    # `robot` is optional on AppConfig: a tree without one is a configuration answer, not a crash.
    print(f"no cell in this config tree ({no_cell}). Write a `robot` block, or select a "
          f"profile that carries one: WILLY_PROFILE=ur5e")
    raise SystemExit
config = config.model_copy(update={"gripper": config.gripper.model_copy(
    update={"model": config.gripper.model or "robotiq_2f85"})})
hand = planner_hand(config)
declared = config.safety.self_collision.model_copy(update={"fixtures": [FixtureBoxConfig(
    name="bench", center_mm=(500.0, 0.0, -75.0), half_extents_mm=(300.0, 300.0, 50.0))]})

# 2. The arm the guard places its link meshes on, and a path that dips into the bench and comes back
#    out of it. Both ends are clear: the last configuration stops a tenth short of the dip, which is
#    the last point on the way in that the guard still accepts. Everything that reaches the bench lies
#    between them, so an endpoint check has nothing to find.
#    Measured on a ur5e: the way in sweeps 6962 mm of arm travel and the way back out 696 mm, so at a
#    10 mm collision margin this is about 766 samples. A full round trip would be 13924 mm and about
#    1393 samples, which is judged too: the cuRobo client splits a path longer than one request across
#    requests, and the exact mesh gate here is a loop. What that costs is time, about 2.2 ms a sample.
arm = create_arm(RobotVendor.from_string(config.vendor), config=config)
home = np.asarray(config.home_joint_positions or home_joints_default(), dtype=np.float64)
dip = np.asarray([3.14, -0.8, 2.0, -2.15, -1.57, 0.0])
waypoints = [tuple(home + (dip - home) * i / 20) for i in range(21)]
waypoints.append(tuple(home + (dip - home) * 0.9))

# 3. The endpoint alone, which is what every one-shot guard sees. It is the home configuration, so
#    nothing refuses it however far the middle of the path reaches into the bench.
safety = config.safety.model_copy(update={"self_collision": declared})
declaring = SafetyPreflight.from_safety_config(safety, config.workspace_limits, hand=hand)
print("endpoint:", declaring.gate_joint_target(JointPositions(list(waypoints[-1])), arm=arm))

# 4. The same path judged whole, sampled at the collision margin; the refusal names its sample.
refusal = declaring.gate_planned_path(waypoints, arm=arm)
print("path, bench declared:",
      "clear" if refusal is None else f"{refusal.status.value}: {refusal.message}")

# 5. The check is not the declaration: with the fixtures this tree ships, the path comes back clear.
blind = SafetyPreflight.from_safety_config(config.safety, config.workspace_limits, hand=hand)
print("path, nothing declared:", blind.gate_planned_path(waypoints, arm=arm))
