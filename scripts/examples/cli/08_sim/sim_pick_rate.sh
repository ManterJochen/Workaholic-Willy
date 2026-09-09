#!/usr/bin/env bash
# A sim pick rate, from the command line.
#
# The Python twin of this file is scripts/examples/api/08_sim/sim_pick_rate.py and it calls the same
# run_gate. The runner's main() reads sys.argv and returns nothing, so the command line's only way to
# keep the number is --result-json: the dict run_gate returns, with the scene and mode stamped on it.
#
# Every line here needs Isaac's bundled interpreter: python.bat, not this repository's venv.
# `python -m src.willy_sim.run_m1_pick --help` is the one form the venv can answer.
set -euo pipefail

# 1. Ten known-pose picks, headless, with the verdict kept as JSON.
python -m src.willy_sim.run_m1_pick --runs 10 --result-json logs/m1.json

# 2. Real vision in the loop: detector and segmenter decide what to pick, so a failure can be
#    perception rather than motion. This is the honest end to end.
python -m src.willy_sim.run_m2_pick --runs 5 --prompt "a red cube"

# 3. The wrist camera rides the arm and re-perceives from where it moved.
python -m src.willy_sim.run_eih_pick --runs 10
