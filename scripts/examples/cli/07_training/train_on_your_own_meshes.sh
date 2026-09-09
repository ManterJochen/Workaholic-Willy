#!/usr/bin/env bash
# Train a grasp generator on your own parts, from the command line.
#
# The Python twin of this file is scripts/examples/api/07_training/train_on_your_own_meshes.py, and
# each command below is one verb of the same two objects: `datagen.api.DatasetBuild` for the three
# data stages, `deep.train.api.GeneratorTraining` for the model. The CLI resolves flags into
# arguments and then calls those, so there is one builder and not two.
set -euo pipefail

# 1. Price the whole chain first. `none` refuses the pile family by name, so it asks for more
#    scenes than you want: a request is not a yield.
python -m datagen cost --scenes 8 --engine none --epochs 2 --train-folds 1 --no-refit

# 2. What the mesh library already holds. Your own parts go in with
#    `--fetch --source custom --from DIR --license own`; the licence has no default.
python -m datagen.assets --check

# 3. Render, label, and extract the corpus the trainer reads.
python -m datagen build --name my_parts --scenes 11 --engine none --out logs/examples/training
python -m datagen label-grasps --name my_parts --out logs/examples/training
python -m datagen build-cloud-corpus --name my_parts --out logs/examples/training \
    --corpus-out logs/examples/training/my_parts_clouds --kinds jaw

# 4. Fit the generator, then read the lift. The smoke tier is minutes; `--tier full` is hours.
python -m src.robot.grasping.deep train-set --tier smoke --recipe v1 --run-folds 1 \
    --clouds logs/examples/training/my_parts_clouds --out logs/examples/training/my_parts_model
python -m src.robot.grasping.deep report --run logs/examples/training/my_parts_model
