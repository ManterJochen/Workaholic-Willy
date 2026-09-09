#!/usr/bin/env bash
# Which self-collision authority this cell actually has, from the command line.
#
# The Python twin of this file is scripts/examples/api/04_safety/self_collision_backend.py and it
# reads the same probe. `robot.safety.self_collision.backend` states an intention; these commands
# state what will run, and the two can differ in silence whenever an engine or a bundle is missing.
set -uo pipefail

# 1. Probe both engines. Exit 0 iff fully anchored: an engine imports AND this model has a bundle.
python -m src.robot.safety.planning --check

# 2. The deep read: load every engine and name whatever refused. Exit 2 if an OS policy blocks.
python -m src.robot.safety.planning --doctor

# 3. A robot nobody baked meshes for. `backend: fcl` on this one silently runs the capsule proxy,
#    and the exit 1 here is that reading, not a broken command.
python -m src.robot.safety.planning --model ur10 --check
