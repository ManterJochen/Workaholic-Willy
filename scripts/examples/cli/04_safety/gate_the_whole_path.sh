#!/usr/bin/env bash
# What this cell has declared about the space it moves through, from the command line.
#
# The Python twin of this file is scripts/examples/api/04_safety/gate_the_whole_path.py, which gates
# a path against a declared bench. There is no CLI that judges a trajectory: the preflight below
# reports the DECLARATIONS a path is judged against, and says nothing about trajectory_check itself.
set -uo pipefail

# 1. The desk audit. `fixtures` and `planning world` come back [warn] on a tree that declares
#    neither, and it exits non-zero while anything is still blocking.
python -m src.robot.execution.real_cell --check

# 2. After declaring the bench and turning safety.trajectory_check on, check the YAML still parses.
python -m src.config

# 3. Who would produce the path in the first place; nothing plans without it.
python -m src.robot.safety.planning --check
