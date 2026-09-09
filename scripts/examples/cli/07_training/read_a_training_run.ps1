# Read what a training run produced, from the command line.
#
# The Python twin of this file is scripts/examples/api/07_training/read_a_training_run.py, and both
# read the same `epochs.json` through the same `build_report`: the command is a formatter around the
# library call, so a service, a notebook and an operator see the same numbers. `--run` takes the
# trainer's own output DIRECTORY, never a log file, and never the weights.

# 1. The four numbers, the lift among them, and the verdict on top. Writes the curve beside the run.
python -m src.robot.grasping.deep report --run logs/examples/training/my_parts_model

# 2. The same report with the curve somewhere of your choosing. Epochs on the x axis, never time:
#    a resumed run is one experiment split across wall-clock gaps.
python -m src.robot.grasping.deep report --run logs/examples/training/my_parts_model `
    --curve logs/examples/training/my_parts_model/curve.png

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
