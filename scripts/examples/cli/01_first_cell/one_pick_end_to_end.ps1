# One grasp, end to end, from the command line.
#
# The Python twin of this file is scripts/examples/api/01_first_cell/one_pick_end_to_end.py and it
# builds the same cell through the same factory. If the two ever disagree, the disagreement is a
# defect in the library and not in one of the examples: `from_robot_config` resolves config into
# arguments and then calls the Python door, so there is one builder and not two.

# 1. Rehearse: the whole call path on a dummy arm, commanding nothing.
python -m src.robot.execution.real_cell --rehearse --runs 1

# 2. Everything decidable at a desk, without building anything.
python -m src.robot.execution.real_cell --check

# 3. Live, on a cell whose controller answers. This one moves a real arm.
#    python -m src.robot.execution.real_cell --runs 1 --prompt "a red cube"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
