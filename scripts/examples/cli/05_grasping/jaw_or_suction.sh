#!/usr/bin/env bash
# A jaw or a cup, from the command line: the three keys that have to agree.
#
# The Python twin is scripts/examples/api/05_grasping/jaw_or_suction.py, and it is the half that
# proposes candidates: nothing on the command line synthesizes a suction grasp, because the
# config-driven pick path has no suction branch to reach. What the CLI answers is which end
# effector this cell declares, and which envelope its candidates are filtered against.
set -euo pipefail

# 1. Which driver actuates it. `vacuum` is the suction one, and it is vendor neutral: an ejector on
#    an output pin, with an optional vacuum switch on an input, is the whole interface.
python -m src.config explain robot.gripper.vendor

# 2. Which envelope the generator filters against. `suction` changes the shape a candidate is
#    checked against; it does not move the pick path to suction.
python -m src.config explain robot.grasping.gripper_geometry.kind

# 3. The half only hardware can answer: pins, port block, and how long the ejector takes to build
#    vacuum. Config, so the suction path can be built and tested before the cup exists.
python -m src.config explain robot.gripper.vacuum
