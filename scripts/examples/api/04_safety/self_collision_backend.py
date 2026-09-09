"""`robot.safety.self_collision.backend`: the exact meshes, the capsule proxy, and the silent swap.

`fcl` measures the real per-link collision meshes and refuses a fold the `capsule` proxy accepts.
With no collision engine, or no `{model}_collision_meshes.npz` bundle for the robot this cell is,
`fcl` does not fail closed: it logs one warning, runs the proxy for the life of the cell, and still
answers `ok`. The reading to trust at boot is therefore the probe below, not the config key.
"""

import logging
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 04_safety, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_robot_config  # noqa: E402
from src.robot.core import JointPositions, MotionCommand, RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.safety import SafetyContext, SelfCollisionGuard  # noqa: E402
from src.robot.safety.planning import probe_collision_engine  # noqa: E402
from src.robot.safety.planning.stack import MotionStack  # noqa: E402

# 1. The fallback announces itself on the logging channel and nowhere else, so listen to it.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# 2. The cell, and the one block this file is about.
config = load_robot_config()
block = config.safety.self_collision
print(f"backend {block.backend}, enforce {block.enforce}, "
      f"min_distance {block.min_distance_mm:.1f} mm, {len(block.fixtures)} fixture(s)")

# 3. Which robot the mesh bundle is keyed on, and what resolves on this box. A bundle is per robot,
#    so a present ur5e bundle says nothing about a UR3e cell.
print(MotionStack.from_robot_config(config).probe().render())

# 4. An arm to hang the link meshes on. Nothing connects; the guard reads capabilities only.
arm = create_arm(RobotVendor.from_string(config.vendor), config=config)

# 5. One folded configuration, a gripper finger driven into the forearm. This is the context
#    `gate_joint_target` builds for every commanded joint move and for a plan's final pose.
folded = SafetyContext(command=MotionCommand.MOVE_JOINTS, arm=arm,
                       target_joints=JointPositions([1.95, 0.38, -1.33, -0.55, 2.00, 0.79]))

# 6. The same fold, both backends. The proxy roots its tool capsule at `target_pose`, and a
#    joint-only context carries none, so it cannot see the gripper at all.
for backend in ("capsule", "fcl"):
    decision = SelfCollisionGuard(block.model_copy(update={"backend": backend})).evaluate(folded)
    print(f"{backend:<8} {'ACCEPTED' if decision.accepted else decision.message}")

# 7. Now ask for `fcl` on a model nobody baked meshes for. Read the warning, then the verdict:
#    nothing in the return value tells this cell apart from one with exact-mesh authority.
print(probe_collision_engine("ur10").summary)
degraded = block.model_copy(update={"backend": "fcl", "kinematics_model": "ur10"})
print("ur10 with backend fcl:", SelfCollisionGuard(degraded).evaluate(folded).reason.value)
