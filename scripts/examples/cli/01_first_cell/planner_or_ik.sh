#!/usr/bin/env bash
# Which planner this cell would use, and whether this box can provide it.
#
# The Python twin of this file is scripts/examples/api/01_first_cell/planner_or_ik.py and it takes
# the same reading through `MotionStack`. `curobo` is fail-closed: where the sidecar environment is
# missing, the UR driver refuses every commanded move rather than degrading to blind IK.
set -euo pipefail

# 1. Read the paths for the robot this config names. Spawns nothing, exits 0 iff fully anchored.
python -m src.robot.safety.planning --check

# 2. Load every engine instead of reading paths: it imports the collision engine, runs one distance
#    query, and spawns the planner's own interpreter to ask what resolves inside it. Seconds, not
#    milliseconds, and it is the only check that can confirm the robot descriptor.
python -m src.robot.safety.planning --doctor

# 3. A different robot than the config names. The bundle and the descriptor are both per-model, so
#    a ur5e cell that is fully anchored says nothing about the UR3e on the next bench.
python -m src.robot.safety.planning --check --model ur3e
