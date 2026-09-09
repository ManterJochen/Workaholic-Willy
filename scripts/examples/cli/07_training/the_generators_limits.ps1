# What the learned generator does not cover, from the command line.
#
# The Python twin of this file is scripts/examples/api/07_training/the_generators_limits.py, and it
# can do one thing this file cannot: TRIGGER the refusal. There is no command that asks for the deep
# calculator without also running a cell, so a command line can print the two config keys and the
# tool that would read the weights, and the Python twin is where the refusal actually fires.

# 1. The two keys that select the learned generator. Both ship unset, and `calculator` is
#    `geometric`: nothing in this repository serves a learned model until you train one.
python -m src.config --print | Select-String -Pattern '"calculator"|"artifact_path"'

# 2. What a weights file says about itself. It needs weights, and none ship here, so this prints
#    its own refusal rather than a model card.
python -m src.robot.grasping.deep inspect --help

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
