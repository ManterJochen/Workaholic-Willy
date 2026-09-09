#!/usr/bin/env bash
# Which end-effector this cell builds, from the command line.
#
# The Python twin of this file is scripts/examples/api/01_first_cell/which_gripper_gets_built.py and
# it reaches the same builder. A `NullGripper` in the output is the answer to read twice: it accepts
# every command and holds nothing, so the picks after it report SUCCEEDED with the jaws on air.
set -euo pipefail

# 1. Preflight, then build on a dummy arm and stop. It prints the gripper class that came out, and
#    the builder's substitution line sits above it. Nothing is connected and nothing moves.
python -m src.robot.execution.real_cell --rehearse --dry-run

# 2. The registry half: which arm and gripper drivers this checkout has at all. A `NO` row is
#    either a reserved slot with no driver written or an SDK this host lacks, and the note says which.
python -m src.robot.drivers.doctor

# 3. The same build against the arm the config really names. Still no motion, but it is not free:
#    a real cell opens its RGB-D camera and loads two models onto the GPU.
#    python -m src.robot.execution.real_cell --dry-run
