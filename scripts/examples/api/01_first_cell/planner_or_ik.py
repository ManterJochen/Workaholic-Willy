"""`robot.ur.motion_planner`: a collision-free plan, or the controller's straight line.

`ik` needs nothing installed and knows nothing about your bench, bin or fixtures, so it will drive
through all of them. `curobo` plans in an isolated sidecar interpreter and is fail-closed: where that
environment is missing the UR driver refuses every commanded move with CONTROLLER_REJECTED rather
than degrading to blind IK, so a cell configured for a planner it does not have does not run badly,
it does not run. Nothing here moves an arm, and nothing here spawns the planner.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 01_first_cell, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.robot.safety.planning.curobo_client import (  # noqa: E402
    CuroboPlanClient, CuroboUnavailableError, curobo_env_available)
from src.robot.safety.planning.stack import MotionStack  # noqa: E402

# 1. The key, and which robot the reading is about. Only the ARM bundle carries the model in its
#    name, {model}_collision_meshes.npz; a gripper variant is {gripper}_{arm}_..., and the sphere
#    map is per hand with no model in it at all. So a present ur5e bundle says nothing about a UR3e,
#    and an arm bundle says nothing about the hand on it.
try:
    robot = load_robot_config()
except ConfigError as no_cell:
    # `robot` is optional on AppConfig, so a tree without one is a configuration answer rather
    # than a crash, and the four files under scripts/checks/ answer it the same way.
    print(f"no cell in this config tree ({no_cell}). Write a `robot` block, or select a "
          f"profile that carries one: WILLY_PROFILE=ur5e")
    raise SystemExit
stack = MotionStack.from_robot_config(robot)
print(f"planner {robot.ur.motion_planner}, robot {stack.model} (from {stack.model_source})")

# 2. Are both engines anchored on this box? This reads paths and spawns nothing. Only one of the two
#    rows is this decision: the exact-mesh collision engine backs the self-collision guard whichever
#    planner runs, and the cuRobo row matters only where the planner is `curobo`.
print(stack.probe().render())

# 3. What the planner would know about this cell. Only a planner consults a declared world, so `ik`
#    plus a fully declared bench is a bench that nothing in the motion path knows about.
print(f"planning_world enabled {robot.safety.planning_world.enabled}, "
      f"{len(robot.safety.self_collision.fixtures or ())} declared fixture(s)")

# 4. `ik` needs nothing beyond this repository: the controller resolves one IK solution and the
#    static guards judge that one configuration, so there is nothing left to ask this box.
if robot.ur.motion_planner == "ik":
    print("`ik`: nothing to install, and nothing that routes around the cell")

# 5. `curobo` with no environment: the refusal itself, through the same client the driver reaches.
#    The `CuroboUnavailableError` below is what the UR driver turns into CONTROLLER_REJECTED.
elif not curobo_env_available():
    try:
        CuroboPlanClient().start()
    except CuroboUnavailableError as error:
        print(f"every move would be CONTROLLER_REJECTED: {error}")

# 6. A present interpreter is still not a working planner: the robot descriptor lives inside that
#    environment, and only a process running there can confirm it.
else:
    print("planner environment present; confirm the descriptor inside it with "
          "`python -m src.robot.safety.planning --doctor`")
