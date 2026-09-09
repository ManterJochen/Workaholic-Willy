#!/usr/bin/env bash
# What a dataset will contain, from the command line, before anything renders.
#
# The Python twin of this file is scripts/examples/api/06_datagen/plan_scenes.py and it calls the
# same `plan_families` and `layout_scene`. Both are engine-free and finish in milliseconds, which is
# why layout is separate from rendering: a layout mistake is cheap to find here and dear to find later.
set -euo pipefail

# 1. The family mix and the object counts for a whole dataset.
python -m datagen plan --scenes 40 --seed 0

# 2. One scene in full, as JSON: assets, spawn poses, cameras, lighting.
python -m datagen describe --scenes 40 --index 0

# 3. `flat_orientation` is the largest single lever in the corpus and the default is the other
#    value. This writes a config that sets it, and says what it measured, without rendering.
python -m datagen init-config --recipe v1 --out logs/examples/tipped.json
