# Which generator proposes the candidates, from the command line.
#
# The Python twin is scripts/examples/api/05_grasping/select_grasp_generator.py, and it builds
# through the same factory. `explain` answers the selector half; the artifact half is a file, so
# `inspect` is what asks a weights file what it says about itself.

# 1. The selector: its type, its default, and which YAML line set the value in force. A key no YAML
#    writes is running on its schema default, which is worth seeing before building on top of it.
python -m src.config explain robot.grasping.calculator

# 2. Where the deep branch would read its weights.
python -m src.config explain robot.grasping.deep_generator.artifact_path

# 3. What a weights file says about itself: kind, artifact version, gripper. No trained weights ship
#    in this repository, so there is nothing here to point it at until a run writes one.
#    python -m src.robot.grasping.deep inspect --artifact PATH
#    python -m src.robot.grasping.deep train-set --clouds DIR --out DIR

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
