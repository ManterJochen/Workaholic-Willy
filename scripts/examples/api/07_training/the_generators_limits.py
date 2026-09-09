"""What the learned generator does not cover, triggered against the code rather than promised.

Four limits are structural rather than a matter of training longer: every slot describes a
two-finger grasp, no weights ship in this repository, asking for one that is not there refuses
instead of quietly serving the analytic stack, and the confidence is a logit over a seed's own
slots rather than a calibrated probability that a grasp holds.
"""

import dataclasses
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 07_training, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_robot_config  # noqa: E402
from src.robot.grasping.calculator_factory import preflight_calculator  # noqa: E402
from src.robot.grasping.deep.net.gripper import (  # noqa: E402
    GRIPPER_VECTOR_DIM,
    JAW_GEOMETRY,
)
from src.robot.grasping.deep.net.set_loss import GraspSetPrediction  # noqa: E402
from src.robot.grasping.deep.train.trainer import SetTrainingPlan  # noqa: E402

# 1. What this config selects, read through the same selector `build_calculator` uses.
robot = load_robot_config()
print(f"grasping.calculator {robot.grasping.calculator!r} resolves to "
      f"{preflight_calculator(robot)!r}, artifact_path "
      f"{robot.grasping.deep_generator.artifact_path!r}")

# 2. Ask for the learned generator with no weights. It refuses. It does NOT fall back to geometric,
#    which would file the analytic stack's numbers under the learned one's name.
deep = robot.model_copy(update={
    "grasping": robot.grasping.model_copy(update={"calculator": "deep"})})
try:
    preflight_calculator(deep)
except FileNotFoundError as refusal:
    print(f"refused, as designed: {refusal}")

# 3. What one slot of the head predicts. Read the field list as a description of a hand: a closing
#    axis and an opening, which a suction cup has neither of. Suction reaches only the where stage.
plan = SetTrainingPlan()
print(f"{plan.model.head.slots} slot(s) per seed, each carrying "
      f"{', '.join(field.name for field in dataclasses.fields(GraspSetPrediction))}")

# 4. The hand reaches the net as a conditioning vector. That is a seam that exists, not a
#    demonstrated capability: nothing in this repository has measured cross-gripper transfer.
print(f"{GRIPPER_VECTOR_DIM}-number gripper vector, hands named: {', '.join(sorted(JAW_GEOMETRY))}")

# 5. `confidence` above is a logit over the slots of one seed. It ranks a seed's own candidates and
#    it is not P(hold): no physics referee has ever been fitted to calibrate it.
print(f"confidence is a per-seed logit over {plan.model.head.slots} slots, not P(hold)")
