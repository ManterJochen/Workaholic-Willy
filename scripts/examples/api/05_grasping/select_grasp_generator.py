"""`robot.grasping.calculator` chooses the generator, and `deep` fails closed without weights.

`geometric` is the analytic stack that ships and runs; `deep` is a learned 6-DoF generator that
replaces the proposal stage. No trained weights ship in this repository, deliberately, because a
customer trains on their own cell's data, so the second branch refuses here by name.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 05_grasping, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_robot_config  # noqa: E402
from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource  # noqa: E402
from src.robot.grasping.calculator_factory import (  # noqa: E402
    build_calculator,
    preflight_calculator,
)

# 1. The cell, and one synthetic box for whichever generator gets built. What is exercised is the
#    selector and its wiring, never grasp quality, which is measured in simulation and on a bench.
robot = load_robot_config()
frame = RehearsalPerceptionSource().acquire()

# 2. Check the selector, then build what it chose. `preflight_calculator` answers without
#    constructing anything, which is what a sweep calls once before its first scene: building
#    inside a per-scene `try` turns one configuration error into N warnings and tidy zeros.
#    The build goes through the factory; naming `GraspCalculator` here would let a cell configured
#    for `deep` silently run the analytic stack under the learned generator's name.
try:
    print("selector:", preflight_calculator(robot))
    calculator = build_calculator(
        robot, camera_matrix=frame.intrinsics,
        min_grip_width_mm=robot.gripper.min_width_mm,
        max_grip_width_mm=robot.gripper.max_width_mm, max_candidates=8)
    candidates = calculator.compute(frame.segmentations[0], frame.depth_map, unit="mm")
    print(f"{type(calculator).__name__}: {len(candidates)} candidate(s)")
except (FileNotFoundError, ValueError) as refusal:
    print(f"{type(refusal).__name__}: {refusal}")

# 3. The same cell asking for `deep`. The refusal is typed and names the command that would produce
#    the missing artifact; it does not fall back to `geometric`, because a cell that asked for the
#    learned generator and quietly got the analytic one is unanswerable from outside.
deep = robot.model_copy(update={
    "grasping": robot.grasping.model_copy(update={"calculator": "deep"})})
try:
    build_calculator(deep, camera_matrix=frame.intrinsics)
    print("deep built: this cell has weights for it")
except (FileNotFoundError, ValueError) as refusal:
    print(f"{type(refusal).__name__}: {refusal}")
