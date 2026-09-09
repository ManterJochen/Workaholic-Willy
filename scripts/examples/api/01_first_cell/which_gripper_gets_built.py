"""Which end-effector `robot.gripper.vendor` actually builds, and what stands in for it.

Four config combinations cannot produce the gripper they name, and none of them raises: the builder
hands back a `NullGripper` that accepts every command, holds nothing and reports the configured
opening, so every pick reports SUCCEEDED while the jaws close on air. The reason rides on the object
as `gripper.substitution`, read before the first success rather than grepped out of a log after ten.
Nothing is connected here: a Robotiq `connect()` is a calibration sweep of the full finger travel.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 01_first_cell, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_robot_config  # noqa: E402
from src.robot.core import RobotVendor  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.execution.autonomous_grasp import AutonomousGraspService  # noqa: E402
from src.robot.execution.autonomous_grasp import build_rehearsal_components  # noqa: E402
from src.robot.grippers import available_gripper_vendors  # noqa: E402

# 1. The two keys that decide this together: the end-effector, and the arm it hangs on. Every
#    branch now asks the ARM IN HAND rather than only the config, because a supplied handle and a
#    config vendor used to be able to disagree in silence: a dummy arm under `vendor: ur` built a
#    real Robotiq driver aimed at `robot.ur.ip`. The Robotiq branch still reads the config too,
#    because `robot.ur.ip` is the only address a Robotiq has.
robot = load_robot_config()
print(f"gripper.vendor {robot.gripper.vendor} on a {robot.vendor} arm")

# 2. Which gripper drivers are registered in this checkout. None of them needs a pip package, so
#    `pip install` fixes nothing here: what is missing on a bring-up is a URCap, a Compute Box or a
#    wire. A vendor the schema accepts but this list omits is a reserved name, and it substitutes.
print("registered:", ", ".join(available_gripper_vendors()))

# 3. Desk components: a synthetic scene and a nadir resolver, so no camera opens and no model loads.
#    The gripper branch consults neither.
calculator, perception, resolver, _cameras, _lenses = build_rehearsal_components(robot)

# 4. This cell, and the same cell with the arm vendor moved to `dummy`, which is the one field a
#    rehearsal changes. Constructing a driver opens no socket; connect() does.
for config in (robot, robot.model_copy(update={"vendor": "dummy"})):
    arm = create_arm(RobotVendor.from_string(config.vendor), config=config)

    # 5. The config-driven build path a real cell takes, one layer below `Cell.build()`: one layer
    #    down because the question is which gripper comes out for a GIVEN arm.
    service = AutonomousGraspService.from_robot_config(
        config, calculator=calculator, perception=perception, frame_resolver=resolver, arm=arm)
    gripper = service.runtime.orchestrator.gripper
    print(f"{config.vendor} arm -> {type(gripper).__name__}")

    # 6. A substitution carries what was asked for, why it was refused, and what to do about it.
    substitution = getattr(gripper, "substitution", None)
    if substitution is not None:
        print(f"  SUBSTITUTED, reason {substitution.reason.value}: {substitution.detail}")
        print(f"  fix: {substitution.fix}")
