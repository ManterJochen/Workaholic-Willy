# What `recipe` and `tier` resolve to, from the command line.
#
# The Python twin of this file is scripts/examples/api/07_training/recipe_and_tier.py, and it
# resolves the same two names through the same `build_plan`. There is no subcommand that ONLY
# resolves them: the flags live on the trainer, so what a command line can show is the vocabulary
# and the bill, and the Python twin is where a merged plan can be read field by field.

# 1. The two flags, on the command that takes them.
python -m src.robot.grasping.deep train-set --help

# 2. What the `full` tier costs over 2,000 scenes. `--no-refit` prices one pass, which is what a
#    tier alone buys: the refit is a recipe setting and this is the flag that shows the difference.
python -m datagen cost --scenes 2000 --engine none --epochs 36 --train-folds 1 --no-refit

# 3. Train under the named bundle. This one wants a corpus and a GPU, and hours at `--tier full`.
#    python -m src.robot.grasping.deep train-set --clouds logs/examples/my_parts_clouds `
#        --out logs/examples/my_parts_model --recipe v1 --tier smoke --run-folds 1

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
